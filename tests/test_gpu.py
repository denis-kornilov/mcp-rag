"""
GPU test — step-by-step, with visible progress and a watchdog.

Normal run (model loading on GPU is skipped for Vega 8):
    cd mcp_rag_package && HSA_OVERRIDE_GFX_VERSION=9.0.0 python3 test_gpu.py

Run with forced model loading (RISKY — may hang the system):
    EMBED_GPU_FORCE=1 HSA_OVERRIDE_GFX_VERSION=9.0.0 python3 test_gpu.py

⚠️  GPU Hang — hardware-level failure at the OS kernel level.
    Python try/except will NOT catch it. If the system hangs — this is expected
    for integrated Vega 8 when loading a model >1GB into VRAM.

Watchdog: with EMBED_GPU_FORCE=1, before loading the model the script asks
    "Do you see the image on screen? (y/n):" — if no answer in 30 sec → stop.
"""
import os, sys, time, threading

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "9.0.0")

# Load .env
from pathlib import Path
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

FORCE_GPU = os.environ.get("EMBED_GPU_FORCE", "").strip() == "1"

print("=" * 60)
print("GPU TEST — AMD ROCm + PyTorch")
if FORCE_GPU:
    print("  ⚠️  EMBED_GPU_FORCE=1 — Vega 8 safety block DISABLED")
print("=" * 60)


def _ask_screen_watchdog(timeout_sec: int = 30) -> bool:
    """Asks the user if they can see the screen. Returns False if no answer."""
    print(f"\n{'='*60}")
    print("  WATCHDOG: Model is about to be loaded onto GPU.")
    print("  If the screen freezes/goes black — reboot required.")
    print(f"  Do you see the image on screen? Enter 'y' and press Enter ({timeout_sec}s):")
    print("  (no answer = loading cancelled automatically)")
    print(f"{'='*60}", flush=True)

    result = [None]

    def _read():
        try:
            result[0] = input("  > ").strip().lower()
        except Exception:
            result[0] = ""

    t = threading.Thread(target=_read, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)

    if result[0] is None:
        print(f"\n  No answer in {timeout_sec}s — cancelling GPU load.", flush=True)
        return False
    if result[0] in ("y", "yes", "да", "д"):
        return True
    print(f"  Answer '{result[0]}' — cancelling GPU load.", flush=True)
    return False


# ── 1. PyTorch ───────────────────────────────────────────────
print("\n[1/5] PyTorch ROCm ...", flush=True)
import torch
print(f"  Version  : {torch.__version__}")
print(f"  CUDA/ROCm: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("\n❌ GPU unavailable. ROCm PyTorch is not installed.")
    sys.exit(1)

props = torch.cuda.get_device_properties(0)
name = props.name
total_gb = props.total_memory / 1024**3
print(f"  Device     : {name}")
print(f"  Memory     : {total_gb:.1f} GB (shared with system RAM on APU)")
print(f"  CU count   : {props.multi_processor_count}")

_INTEGRATED = ("vega 8", "vega 11", "vega 3", "vega 6")
is_integrated = any(p in name.lower() for p in _INTEGRATED)
if is_integrated:
    print(f"\n  ℹ️  Integrated GPU — matrix ops work,")
    print("     loading a large model causes GPU Hang (hardware failure).")

# ── 2. Basic matrix ──────────────────────────────────────────
print("\n[2/5] Matrix multiply 100×100 ...", end=" ", flush=True)
x = torch.randn(100, 100, device="cuda")
y = torch.randn(100, 100, device="cuda")
z = x @ y
print(f"OK  norm={z.norm().item():.2f}")

# ── 3. Benchmark ─────────────────────────────────────────────
print("\n[3/5] Benchmark 2048×2048 (10 iterations) ...", flush=True)
a = torch.randn(2048, 2048, device="cuda")
b = torch.randn(2048, 2048, device="cuda")
torch.cuda.synchronize()
t0 = time.perf_counter()
for i in range(10):
    c = a @ b
    print(f"  iter {i+1}/10", end="\r", flush=True)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
gflops = 10 * 2 * 2048**3 / elapsed / 1e9
print(f"  10 iterations in {elapsed:.2f}s → {gflops:.1f} GFLOPS          ")

# ── 4. BGE-M3 model loading ──────────────────────────────────
if is_integrated and not FORCE_GPU:
    print("\n[4/5] Loading BGE-M3 onto GPU — SKIPPED")
    print("      (Vega 8 APU: GPU Hang protection active)")
    print("      To test: EMBED_GPU_FORCE=1 HSA_OVERRIDE_GFX_VERSION=9.0.0 python3 test_gpu.py")
else:
    print(f"\n[4/5] Loading BGE-M3 onto GPU (fp16) ...", flush=True)
    # Watchdog — ask user before the risky operation
    if is_integrated:
        proceed = _ask_screen_watchdog(timeout_sec=30)
        if not proceed:
            print("  Loading cancelled by user/timeout.")
        else:
            print("  Starting load ...", flush=True)
    else:
        proceed = True

    if proceed:
        try:
            from sentence_transformers import SentenceTransformer
            print("  SentenceTransformer OK", flush=True)
            t0 = time.perf_counter()
            model = SentenceTransformer(
                "BAAI/bge-m3", device="cuda",
                model_kwargs={"use_safetensors": True, "torch_dtype": torch.float16},
            )
            load_time = time.perf_counter() - t0
            print(f"  Model loaded in {load_time:.1f}s", flush=True)

            texts = ["Hello world", "GPU embedding test", "AMD Vega ROCm"]
            t0 = time.perf_counter()
            vecs = model.encode(texts, normalize_embeddings=True)
            enc_time = time.perf_counter() - t0
            print(f"  Vectors: {vecs.shape}  in {enc_time:.2f}s")
            print(f"  Norms: {(vecs**2).sum(1)**0.5}")
            print("  ✅ GPU model works!")
        except Exception as e:
            print(f"\n  ❌ Python caught exception: {e}")
            print("  (if screen went black — that was a GPU Hang, not this exception)")

# ── 5. ONNX Runtime ─────────────────────────────────────────
print("\n[5/5] ONNX Runtime (optimum) ...", end=" ", flush=True)
try:
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    print("✅ installed")
except ImportError:
    print("❌ not installed")

# ── Summary ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY:")
print(f"  GPU (ROCm)  : ✅")
print(f"  Device      : {name}")
print(f"  Type        : {'Integrated APU (shared RAM)' if is_integrated else 'Discrete'}")
print(f"  Matrix      : ✅ {gflops:.0f} GFLOPS")
if is_integrated and not FORCE_GPU:
    print(f"  Model GPU   : ⛔ blocked (safety)")
    print(f"  Recommended : python3 test_onnx.py (CPU ONNX)")
print("=" * 60)
