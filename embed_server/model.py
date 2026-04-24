"""Embedding model loader for embed_server.

Selected via EMBED_BACKEND env var:
  onnx-cpu        — ONNX Runtime INT8 CPU (default, no GPU libs needed)
  onnx-cuda       — ONNX Runtime CUDA (NVIDIA, requires onnxruntime-gpu + CUDA)
  onnx-rocm       — ONNX Runtime ROCm  (AMD,    requires onnxruntime-rocm + ROCm)
  onnx-auto       — ONNX Runtime: GPU if available, else CPU (auto-detect)
  bge-m3-pytorch  — sentence-transformers CPU  (requires torch + sentence-transformers)
  bge-m3-gpu      — sentence-transformers GPU  (requires torch + CUDA + sentence-transformers)

Two independent ORT sessions are created for ONNX backends:
  - ingest session: ONNX_NUM_THREADS (default 4) — for ingest batches
  - search session: SEARCH_THREADS   (default 1) — for query vectors

Both sessions run truly in parallel at the OS level (GIL is released
during native ORT C++ inference). The tokenizer is shared (read-only,
thread-safe via Rust tokenizers library).

GPU device selection:
  CUDA_DEVICE_ID  — GPU index for NVIDIA (default 0)
  ROCM_DEVICE_ID  — GPU index for AMD    (default 0)
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import List

import numpy as np

logger = logging.getLogger("embed_server.model")

# ─── configuration (loaded from .env files) ──────────────────────────────────

_PKG_ROOT  = Path(__file__).resolve().parents[1]
_SELF_DIR  = Path(__file__).resolve().parent

# .env load priority: mcp-rag root (defaults) → embed_server/ (overrides)
try:
    from dotenv import load_dotenv  # type: ignore
    if (_PKG_ROOT / ".env").exists():
        load_dotenv(str(_PKG_ROOT / ".env"), override=False)
    if (_SELF_DIR / ".env").exists():
        load_dotenv(str(_SELF_DIR / ".env"), override=False)
except Exception:
    pass

_EMBED_BACKEND  = os.environ.get("EMBED_BACKEND", "onnx-cpu").strip().lower()

_HF_HOME  = str(_SELF_DIR / "embed_data" / "hf_home")
_HF_CACHE = _HF_HOME
_ONNX_DIR = str(_SELF_DIR / "embed_data" / "onnx_exports" / "bge-m3")
_INGEST_THREADS = int(os.environ.get("ONNX_NUM_THREADS", "4"))
_SEARCH_THREADS = int(os.environ.get("SEARCH_THREADS", "1"))
_INTER_THREADS  = int(os.environ.get("ONNX_INTER_THREADS", "1"))
_EXEC_MODE      = os.environ.get("ONNX_EXECUTION_MODE", "sequential").strip().lower()
_CUDA_DEVICE_ID = int(os.environ.get("CUDA_DEVICE_ID", "0"))
_ROCM_DEVICE_ID = int(os.environ.get("ROCM_DEVICE_ID", "0"))
_BATCH_SIZE     = int(os.environ.get("EMBED_BATCH_SIZE", "32"))
_MAX_SEQ_LEN    = int(os.environ.get("EMBED_MAX_SEQ_LEN", "512"))

_ingest_embedder = None
_search_embedder = None
_active_providers: List[str] = []

# Tokenizer is loaded once and shared between sessions (thread-safe)
_shared_tokenizer = None
_tokenizer_lock = threading.Lock()


# ─── public API ──────────────────────────────────────────────────────────────

def get_ingest_embedder():
    global _ingest_embedder
    if _ingest_embedder is None:
        _ingest_embedder = _build(num_threads=_INGEST_THREADS, role="ingest")
    return _ingest_embedder


def get_search_embedder():
    global _search_embedder
    if _search_embedder is None:
        _search_embedder = _build(num_threads=_SEARCH_THREADS, role="search")
    return _search_embedder


def get_embedder():
    """Backward-compatible alias — returns the ingest embedder."""
    return get_ingest_embedder()


def embedder_info() -> dict:
    primary = _active_providers[0] if _active_providers else "CPUExecutionProvider"
    # providers list may contain tuples (name, options) — extract name if so
    if isinstance(primary, tuple):
        primary = primary[0]
    gpu = primary in ("CUDAExecutionProvider", "ROCMExecutionProvider")
    return {
        "backend": _EMBED_BACKEND,
        "type": "DualSession" if _is_onnx_backend() else _EMBED_BACKEND,
        "gpu": gpu,
        "providers": [p[0] if isinstance(p, tuple) else p for p in _active_providers],
        "ingest_threads": _INGEST_THREADS,
        "search_threads": _SEARCH_THREADS,
        "inter_threads": _INTER_THREADS,
        "execution_mode": _EXEC_MODE,
        "cuda_device_id": _CUDA_DEVICE_ID,
        "rocm_device_id": _ROCM_DEVICE_ID,
        "batch_size": _BATCH_SIZE,
        "max_seq_len": _MAX_SEQ_LEN,
        "onnx_dir": _ONNX_DIR,
    }


# ─── dependency guards ───────────────────────────────────────────────────────

def _require_ort():
    """Import onnxruntime with a clear install message on failure."""
    try:
        import onnxruntime as ort
        return ort
    except ImportError:
        raise RuntimeError(
            "onnxruntime not installed.\n"
            "  CPU:    pip install onnxruntime>=1.19.0\n"
            "  NVIDIA: pip install onnxruntime-gpu>=1.19.0\n"
            "  AMD:    pip install onnxruntime-rocm\n"
            "  Or run: bash embed_server/install.sh\n"
            "  See:    embed_server/requirements.txt"
        ) from None


def _require_transformers():
    """Import transformers (tokenizer only, no torch) with a clear install message."""
    try:
        import transformers  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "transformers not installed (needed for tokenizer).\n"
            "  Install: pip install transformers>=4.40.0\n"
            "  Or run:  bash embed_server/install.sh\n"
            "  See:     embed_server/requirements.txt"
        ) from None


def _require_sentence_transformers():
    """Import sentence-transformers with a clear install message."""
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "sentence-transformers not installed.\n"
            "  Install: pip install sentence-transformers>=2.7.0\n"
            "  Note: pulls in ~2GB of PyTorch weights.\n"
            "  Lightweight alternative: use EMBED_BACKEND=onnx-cpu instead.\n"
            "  See: embed_server/requirements.txt"
        ) from None


def _require_torch():
    """Import torch with a clear install message."""
    try:
        import torch  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "PyTorch not installed (required for EMBED_BACKEND=bge-m3-gpu).\n"
            "  CPU:    pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "  NVIDIA: pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
            "  AMD:    pip install torch --index-url https://download.pytorch.org/whl/rocm6.0\n"
            "  Lightweight alternative: use EMBED_BACKEND=onnx-cuda or onnx-rocm instead.\n"
            "  See: embed_server/requirements.txt"
        ) from None


def _require_optimum():
    """Import optimum.onnxruntime with a clear install message."""
    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "optimum[onnxruntime] not installed (required for ONNX model export).\n"
            "  Install: pip install optimum[onnxruntime]>=1.21.0\n"
            "  Or run:  bash embed_server/install.sh\n"
            "  Note: only needed for first-time model export; pre-exported models skip this.\n"
            "  See: embed_server/requirements.txt"
        ) from None


# ─── helpers ─────────────────────────────────────────────────────────────────

def _is_onnx_backend() -> bool:
    return _EMBED_BACKEND in (
        "onnx-cpu", "onnx-cuda", "onnx-rocm", "onnx-auto", "bge-m3-onnx", "bge-m3",
    )


def _ort_providers() -> list:
    ort = _require_ort()
    available = ort.get_available_providers()
    logger.info("[model] ORT available providers: %s", available)
    b = _EMBED_BACKEND

    if b in ("onnx-cuda", "bge-m3-gpu"):
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                f"CUDAExecutionProvider not available for EMBED_BACKEND={b}.\n"
                "  Required: onnxruntime-gpu + CUDA Toolkit 11.8+ + cuDNN.\n"
                "  Install:  pip uninstall onnxruntime -y && pip install onnxruntime-gpu>=1.19.0\n"
                "  Or run:   bash embed_server/install.sh  (select NVIDIA)\n"
                "  Check:    nvidia-smi  (driver must be ≥525)\n"
                "  Fallback: set EMBED_BACKEND=onnx-cpu in embed_server/.env"
            )
        return [
            ("CUDAExecutionProvider", {
                "device_id": _CUDA_DEVICE_ID,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
            }),
            "CPUExecutionProvider",
        ]

    if b == "onnx-rocm":
        if "ROCMExecutionProvider" not in available:
            raise RuntimeError(
                f"ROCMExecutionProvider not available for EMBED_BACKEND={b}.\n"
                "  Required: onnxruntime-rocm + ROCm 5.6 or 6.x.\n"
                "  Install:  pip uninstall onnxruntime -y && pip install onnxruntime-rocm\n"
                "  Or run:   bash embed_server/install.sh  (select AMD)\n"
                "  Check:    rocm-smi --showproductname\n"
                "  Consumer GPU (RX 6xxx/7xxx): may need HSA_OVERRIDE_GFX_VERSION\n"
                "  Fallback: set EMBED_BACKEND=onnx-cpu in embed_server/.env"
            )
        return [
            ("ROCMExecutionProvider", {"device_id": _ROCM_DEVICE_ID}),
            "CPUExecutionProvider",
        ]

    if b == "onnx-auto":
        if "CUDAExecutionProvider" in available:
            logger.info("[model] onnx-auto → CUDAExecutionProvider device_id=%d", _CUDA_DEVICE_ID)
            return [
                ("CUDAExecutionProvider", {"device_id": _CUDA_DEVICE_ID}),
                "CPUExecutionProvider",
            ]
        if "ROCMExecutionProvider" in available:
            logger.info("[model] onnx-auto → ROCMExecutionProvider device_id=%d", _ROCM_DEVICE_ID)
            return [
                ("ROCMExecutionProvider", {"device_id": _ROCM_DEVICE_ID}),
                "CPUExecutionProvider",
            ]
        logger.info("[model] onnx-auto → CPUExecutionProvider (no GPU detected)")
        return ["CPUExecutionProvider"]

    # onnx-cpu and any unknown → CPU only (no GPU libs touched)
    return ["CPUExecutionProvider"]


def _get_onnx_path() -> str:
    onnx_dir = Path(_ONNX_DIR)
    quantized = onnx_dir / "model_quantized.onnx"
    fp32 = onnx_dir / "model.onnx"

    if quantized.exists():
        logger.info("[model] using quantized ONNX INT8: %s", quantized)
        return str(quantized)
    if fp32.exists():
        logger.info("[model] using FP32 ONNX: %s", fp32)
        return str(fp32)

    logger.info("[model] exporting ONNX from HuggingFace: BAAI/bge-m3")
    _require_optimum()
    from optimum.onnxruntime import ORTModelForFeatureExtraction  # type: ignore
    onnx_dir.mkdir(parents=True, exist_ok=True)
    tmp = ORTModelForFeatureExtraction.from_pretrained(
        "BAAI/bge-m3", export=True, cache_dir=_HF_CACHE
    )
    tmp.save_pretrained(str(onnx_dir))
    return str(onnx_dir / "model.onnx")


def _load_tokenizer_no_torch(model_id: str):
    import glob as _glob
    _require_transformers()
    from transformers import PreTrainedTokenizerFast

    if _HF_CACHE:
        safe_name = model_id.replace("/", "--")
        pattern = str(Path(_HF_CACHE) / f"models--{safe_name}" / "snapshots" / "*" / "tokenizer.json")
        matches = sorted(_glob.glob(pattern))
        if matches:
            tok_path = Path(matches[-1]).parent
            logger.info("[model] loading tokenizer from cache: %s", tok_path)
            return PreTrainedTokenizerFast.from_pretrained(str(tok_path))

    logger.info("[model] loading tokenizer from HuggingFace Hub: %s", model_id)
    return PreTrainedTokenizerFast.from_pretrained(model_id, cache_dir=_HF_CACHE)


def _get_shared_tokenizer():
    global _shared_tokenizer
    if _shared_tokenizer is None:
        with _tokenizer_lock:
            if _shared_tokenizer is None:
                _shared_tokenizer = _load_tokenizer_no_torch("BAAI/bge-m3")
    return _shared_tokenizer


# ─── builders ────────────────────────────────────────────────────────────────

def _build(num_threads: int, role: str = "ingest"):
    backend = _EMBED_BACKEND
    logger.info("[model] building %s embedder backend=%s threads=%d", role, backend, num_threads)

    if _is_onnx_backend():
        return _build_onnx(num_threads=num_threads, role=role)
    if backend == "bge-m3-pytorch":
        return _build_pytorch_cpu()
    if backend == "bge-m3-gpu":
        return _build_gpu()

    logger.warning("[model] unknown backend=%s, falling back to onnx-cpu", backend)
    return _build_onnx(num_threads=num_threads, role=role)


def _build_onnx(num_threads: int, role: str = "ingest"):
    global _active_providers
    ort = _require_ort()

    onnx_path = _get_onnx_path()

    so = ort.SessionOptions()
    so.intra_op_num_threads = num_threads
    so.inter_op_num_threads = _INTER_THREADS
    if _EXEC_MODE == "parallel":
        so.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    else:
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = _ort_providers()
    session = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)

    if role == "ingest":
        _active_providers = list(session.get_providers())

    tokenizer = _get_shared_tokenizer()

    active = list(session.get_providers())
    logger.info(
        "[model] ORT %s session ready  providers=%s  intra=%d  inter=%d  mode=%s",
        role, active, num_threads, _INTER_THREADS, _EXEC_MODE,
    )

    class _OnnxEmbedder:
        def encode(self, texts, batch_size=_BATCH_SIZE):
            all_vecs = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=_MAX_SEQ_LEN, return_tensors="np",
                )
                inputs = {k: v for k, v in enc.items()
                          if k in {inp.name for inp in session.get_inputs()}}
                out = session.run(None, inputs)
                hidden = out[0][:, 0, :].astype(np.float32)
                norms = np.linalg.norm(hidden, axis=1, keepdims=True)
                all_vecs.append(hidden / np.maximum(norms, 1e-9))
            return np.vstack(all_vecs)

    return _OnnxEmbedder()


def _build_pytorch_cpu():
    # sentence-transformers + CPU only — no CUDA libs required
    _require_sentence_transformers()
    from sentence_transformers import SentenceTransformer  # type: ignore
    m = SentenceTransformer("BAAI/bge-m3", device="cpu", cache_folder=_HF_CACHE)
    logger.info("[model] PyTorch CPU embedder ready")
    return m


def _build_gpu():
    # sentence-transformers + PyTorch CUDA — requires torch with GPU support
    _require_torch()
    _require_sentence_transformers()
    import torch
    from sentence_transformers import SentenceTransformer  # type: ignore
    if not torch.cuda.is_available():
        raise RuntimeError(
            "EMBED_BACKEND=bge-m3-gpu but torch.cuda.is_available() is False.\n"
            "  Install CUDA-enabled PyTorch:\n"
            "    pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
            "  Or switch to ONNX-based GPU: EMBED_BACKEND=onnx-cuda (lighter, no full PyTorch)\n"
            "  See: embed_server/requirements.txt"
        )
    device = f"cuda:{_CUDA_DEVICE_ID}"
    m = SentenceTransformer("BAAI/bge-m3", device=device, cache_folder=_HF_CACHE)
    logger.info("[model] GPU embedder ready device=%s", device)

    class _GpuEmbedder:
        def encode(self, texts, batch_size=_BATCH_SIZE):
            return m.encode(texts, batch_size=batch_size,
                            convert_to_numpy=True, normalize_embeddings=True)

    return _GpuEmbedder()
