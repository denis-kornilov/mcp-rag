"""
ONNX CPU embedding test — с видимым прогрессом.
Запуск: cd mcp_rag_package && python3 test_onnx.py

Этапы:
  1. Экспорт модели BGE-M3 → ONNX  (первый раз ~3–8 мин, потом из кэша)
  2. Токенизация + encode 3 текстов
  3. Замер скорости (10 текстов × 5 итераций)
"""
import os, sys, time

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "9.0.0")

# Подгружаем .env
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

# ── 1. Проверяем кэш ─────────────────────────────────────────
hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
import glob
cached = glob.glob(f"{hf_home}/**/*.onnx", recursive=True)
if cached:
    print(f"\n[1] ONNX модель уже в кэше:")
    for p in cached[:3]:
        size = Path(p).stat().st_size / 1024**3
        print(f"     {p}  ({size:.2f} GB)")
else:
    print("\n[1] ONNX кэш не найден — будет экспорт (первый запуск).")
    print("    Это займёт ~3–8 минут. Прогресс появится ниже.")
    print("    (После — кэшируется навсегда, повторно не делается)")

# ── 2. Загрузка / экспорт ────────────────────────────────────
print("\n[2] Загружаем модель...")
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
print("    (если первый раз — идёт конвертация, patience...)")
sys.stdout.flush()
model = ORTModelForFeatureExtraction.from_pretrained(MODEL_ID, export=True, cache_dir=hf_home)
load_elapsed = time.perf_counter() - t_load
print(f"    Модель готова за {load_elapsed:.1f}s")

# ── 3. Тест encode ────────────────────────────────────────────
print("\n[3] Тест кодирования 3 текстов ...")
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
print(f"    Векторов: {vecs.shape}  за {elapsed:.2f}s")
print(f"    Нормы: {(vecs**2).sum(1)**0.5}")

# ── 4. Замер скорости ─────────────────────────────────────────
print("\n[4] Замер скорости (batch=10 текстов, 5 итераций) ...")
texts_bench = [f"Sample document number {i} for benchmarking ONNX" for i in range(10)]
times = []
for i in range(5):
    t0 = time.perf_counter()
    encode(texts_bench)
    elapsed = time.perf_counter() - t0
    times.append(elapsed)
    print(f"    iter {i+1}/5: {elapsed:.2f}s  ({10/elapsed:.1f} texts/s)", flush=True)

avg = sum(times) / len(times)
print(f"\n    Среднее: {avg:.2f}s на батч 10 → {10/avg:.1f} texts/s")

# ── 5. Итог ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("РЕЗУЛЬТАТ:")
print(f"  Backend      : ONNX Runtime (CPU)")
print(f"  Модель       : {MODEL_ID}")
print(f"  Dim вектора  : {vecs.shape[1]}")
print(f"  Скорость     : {10/avg:.1f} texts/s")
print(f"  Загрузка     : {load_elapsed:.1f}s (кэш: {'да' if load_elapsed < 10 else 'нет'})")
print("  GPU          : ОТКЛЮЧЁН (Vega 8 integrated = GPU Hang)")
print("=" * 60)
print("✅ ONNX CPU тест пройден")
