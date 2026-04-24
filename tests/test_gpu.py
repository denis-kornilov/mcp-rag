"""
GPU тест — пошаговый, с видимым прогрессом и watchdog.

Запуск (нормальный — модель на GPU пропускается для Vega 8):
    cd mcp_rag_package && HSA_OVERRIDE_GFX_VERSION=9.0.0 python3 test_gpu.py

Запуск с принудительной загрузкой модели (РИСКОВАННО — может вешать систему):
    EMBED_GPU_FORCE=1 HSA_OVERRIDE_GFX_VERSION=9.0.0 python3 test_gpu.py

⚠️  GPU Hang — аппаратный сбой уровня ядра ОС.
    Python try/except его НЕ поймает. Если система зависнет — ожидаемо
    для интегрированной Vega 8 при загрузке модели >1GB в VRAM.

Watchdog: при EMBED_GPU_FORCE=1 перед загрузкой модели скрипт спрашивает
    "Видите изображение на экране? (y/n):" — если нет ответа 30 сек → стоп.
"""
import os, sys, time, threading

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

FORCE_GPU = os.environ.get("EMBED_GPU_FORCE", "").strip() == "1"

print("=" * 60)
print("GPU TEST — AMD ROCm + PyTorch")
if FORCE_GPU:
    print("  ⚠️  EMBED_GPU_FORCE=1 — защитный блок Vega 8 СНЯТ")
print("=" * 60)


def _ask_screen_watchdog(timeout_sec: int = 30) -> bool:
    """Спрашивает пользователя видит ли он экран. Возвращает False если нет ответа."""
    print(f"\n{'='*60}")
    print("  WATCHDOG: Сейчас будет загрузка модели на GPU.")
    print("  Если экран зависнет/почернеет — нужна перезагрузка.")
    print(f"  Видите изображение на экране? Введите 'y' и Enter ({timeout_sec}с):")
    print("  (нет ответа = автоматически отменяем загрузку)")
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
        print(f"\n  Нет ответа за {timeout_sec}с — отменяем загрузку на GPU.", flush=True)
        return False
    if result[0] in ("y", "yes", "да", "д"):
        return True
    print(f"  Ответ '{result[0]}' — отменяем загрузку на GPU.", flush=True)
    return False


# ── 1. PyTorch ───────────────────────────────────────────────
print("\n[1/5] PyTorch ROCm ...", flush=True)
import torch
print(f"  Версия   : {torch.__version__}")
print(f"  CUDA/ROCm: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("\n❌ GPU недоступен. ROCm PyTorch не установлен.")
    sys.exit(1)

props = torch.cuda.get_device_properties(0)
name = props.name
total_gb = props.total_memory / 1024**3
print(f"  Устройство : {name}")
print(f"  Память     : {total_gb:.1f} GB (shared с системной RAM на APU)")
print(f"  CU count   : {props.multi_processor_count}")

_INTEGRATED = ("vega 8", "vega 11", "vega 3", "vega 6")
is_integrated = any(p in name.lower() for p in _INTEGRATED)
if is_integrated:
    print(f"\n  ℹ️  Интегрированный GPU — матрица работает,")
    print("     загрузка большой модели вызывает GPU Hang (аппаратный сбой).")

# ── 2. Базовая матрица ───────────────────────────────────────
print("\n[2/5] Матричное умножение 100×100 ...", end=" ", flush=True)
x = torch.randn(100, 100, device="cuda")
y = torch.randn(100, 100, device="cuda")
z = x @ y
print(f"OK  norm={z.norm().item():.2f}")

# ── 3. Бенчмарк ─────────────────────────────────────────────
print("\n[3/5] Бенчмарк 2048×2048 (10 итераций) ...", flush=True)
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
print(f"  10 итераций за {elapsed:.2f}s → {gflops:.1f} GFLOPS          ")

# ── 4. Загрузка модели BGE-M3 ────────────────────────────────
if is_integrated and not FORCE_GPU:
    print("\n[4/5] Загрузка BGE-M3 на GPU — ПРОПУЩЕНО")
    print("      (Vega 8 APU: GPU Hang защита активна)")
    print("      Для теста: EMBED_GPU_FORCE=1 HSA_OVERRIDE_GFX_VERSION=9.0.0 python3 test_gpu.py")
else:
    print(f"\n[4/5] Загрузка BGE-M3 на GPU (fp16) ...", flush=True)
    # Watchdog — спрашиваем пользователя перед рискованной операцией
    if is_integrated:
        proceed = _ask_screen_watchdog(timeout_sec=30)
        if not proceed:
            print("  Загрузка отменена пользователем/таймаутом.")
        else:
            print("  Начинаем загрузку ...", flush=True)
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
            print(f"  Модель загружена за {load_time:.1f}s", flush=True)

            texts = ["Hello world", "GPU embedding test", "AMD Vega ROCm"]
            t0 = time.perf_counter()
            vecs = model.encode(texts, normalize_embeddings=True)
            enc_time = time.perf_counter() - t0
            print(f"  Векторов: {vecs.shape}  за {enc_time:.2f}s")
            print(f"  Нормы: {(vecs**2).sum(1)**0.5}")
            print("  ✅ GPU модель работает!")
        except Exception as e:
            print(f"\n  ❌ Python поймал исключение: {e}")
            print("  (если был чёрный экран — это был GPU Hang, не это исключение)")

# ── 5. ONNX Runtime ─────────────────────────────────────────
print("\n[5/5] ONNX Runtime (optimum) ...", end=" ", flush=True)
try:
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    print("✅ установлен")
except ImportError:
    print("❌ не установлен")

# ── Итог ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ИТОГ:")
print(f"  GPU (ROCm)  : ✅")
print(f"  Устройство  : {name}")
print(f"  Тип         : {'Интегрированный APU (shared RAM)' if is_integrated else 'Дискретный'}")
print(f"  Матрица     : ✅ {gflops:.0f} GFLOPS")
if is_integrated and not FORCE_GPU:
    print(f"  Модель GPU  : ⛔ заблокировано (безопасность)")
    print(f"  Рекомендация: python3 test_onnx.py (CPU ONNX)")
print("=" * 60)
