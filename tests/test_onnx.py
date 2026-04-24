"""
ONNX CPU embedding test — with visible progress.
Run: cd mcp_rag_package && python3 test_onnx.py

Stages:
  1. Export model BGE-M3 → ONNX  (first run ~3–8 min, then from cache)
  2. Tokenize + encode 3 texts
  3. Speed benchmark (10 texts × 5 iterations)
"""
import os, sys, time

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

print("=" * 60)
print("ONNX CPU EMBEDDING TEST")
print("=" * 60)

# ── 1. Check cache ───────────────────────────────────────────
hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
import glob
cached = glob.glob(f"{hf_home}/**/*.onnx", recursive=True)
if cached:
    print(f"\n[1] ONNX model already cached:")
    for p in cached[:3]:
        size = Path(p).stat().st_size / 1024**3
        print(f"     {p}  ({size:.2f} GB)")
else:
    print("\n[1] ONNX cache not found — export will run (first launch).")
    print("    This will take ~3–8 minutes. Progress will appear below.")
    print("    (After — cached permanently, not repeated)")

# ── 2. Load / export ─────────────────────────────────────────
print("\n[2] Loading model...")
sys.stdout.flush()

from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction

MODEL_ID = "BAAI/bge-m3"
MAX_SEQ_LEN = 512

t_load = time.perf_counter()
print("    AutoTokenizer.from_pretrained ...", end=" ", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=hf_home)
print("OK")

print("    ORTModelForFeatureExtraction.from_pretrained (export=True) ...")
print("    (if first run — conversion in progress, please wait...)")
sys.stdout.flush()
model = ORTModelForFeatureExtraction.from_pretrained(MODEL_ID, export=True, cache_dir=hf_home)
load_elapsed = time.perf_counter() - t_load
print(f"    Model ready in {load_elapsed:.1f}s")

# ── 3. Encode test ────────────────────────────────────────────
print("\n[3] Encoding test for 3 texts ...")
import numpy as np

def encode(texts):
    inputs = tokenizer(
        texts, padding=True, truncation=True,
        max_length=MAX_SEQ_LEN, return_tensors="pt"
    )
    outputs = model(**inputs)
    token_emb = outputs.last_hidden_state.numpy()
    mask = inputs["attention_mask"].numpy()[:, :, np.newaxis].astype(float)
    vecs = (token_emb * mask).sum(1) / mask.sum(1).clip(min=1e-9)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-9)
    return vecs / norms

t0 = time.perf_counter()
vecs = encode(["Hello world", "ONNX test on CPU", "AMD Raven Ridge APU"])
elapsed = time.perf_counter() - t0
print(f"    Vectors: {vecs.shape}  in {elapsed:.2f}s")
print(f"    Norms: {(vecs**2).sum(1)**0.5}")

# ── 4. Speed benchmark ────────────────────────────────────────
print("\n[4] Speed benchmark (batch=10 texts, 5 iterations) ...")
texts_bench = [f"Sample document number {i} for benchmarking ONNX" for i in range(10)]
times = []
for i in range(5):
    t0 = time.perf_counter()
    encode(texts_bench)
    elapsed = time.perf_counter() - t0
    times.append(elapsed)
    print(f"    iter {i+1}/5: {elapsed:.2f}s  ({10/elapsed:.1f} texts/s)", flush=True)

avg = sum(times) / len(times)
print(f"\n    Average: {avg:.2f}s per batch of 10 → {10/avg:.1f} texts/s")

# ── 5. Summary ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESULT:")
print(f"  Backend      : ONNX Runtime (CPU)")
print(f"  Model        : {MODEL_ID}")
print(f"  Vector dim   : {vecs.shape[1]}")
print(f"  Speed        : {10/avg:.1f} texts/s")
print(f"  Load time    : {load_elapsed:.1f}s (cached: {'yes' if load_elapsed < 10 else 'no'})")
print(f"  GPU          : DISABLED (Vega 8 integrated = GPU Hang)")
print("=" * 60)
print("✅ ONNX CPU test passed")
