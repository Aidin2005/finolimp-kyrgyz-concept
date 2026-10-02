"""
check_env.py — Проверка готовности среды перед запуском FinOlimp пайплайна.
Запустите: python check_env.py
"""
import sys
import os

# ── подавляем шум matplotlib заранее ──────────────────────────────────────────
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".matplotlib_cache"))

REQUIRED = {
    "pandas":        ">=1.5",
    "numpy":         ">=1.23",
    "openpyxl":      ">=3.0",
    "sklearn":       "scikit-learn>=1.1",
    "lightgbm":      ">=4.0",
    "flask":         ">=2.2",
}

REQUIRED_DATA = [
    "data_2_final/acts.csv",
    "data_2_final/etm.csv",
    "data_2_final/registry.csv",
]

ok = True

print("=" * 60)
print("  FinOlimp 2026 · Kyrgyz Concept — Проверка окружения")
print("=" * 60)

# 1. Python version
print(f"\n[1/3] Python версия: {sys.version}")
major, minor = sys.version_info[:2]
if major < 3 or (major == 3 and minor < 8):
    print("  ❌ ОШИБКА: Требуется Python 3.8+. У вас: {}.{}".format(major, minor))
    print("     Скачайте: https://www.python.org/downloads/")
    ok = False
else:
    print(f"  ✅ Python {major}.{minor} — OK")

# 2. Libraries
print("\n[2/3] Библиотеки:")
missing = []
for module, install_name in REQUIRED.items():
    try:
        mod = __import__(module)
        version = getattr(mod, "__version__", "?")
        print(f"  ✅ {module} {version}")
    except ImportError:
        print(f"  ❌ {module} — НЕ УСТАНОВЛЕН")
        missing.append(install_name)
        ok = False

if missing:
    print("\n  Установите недостающие пакеты командой:")
    print("    pip install " + " ".join(m.split(">=")[0].replace("scikit-learn", "scikit-learn") for m in missing))
    print("  Или сразу всё:")
    print("    pip install -r requirements.txt")

# 3. Data files
print("\n[3/3] Файлы данных:")
base = os.path.dirname(__file__)
for rel_path in REQUIRED_DATA:
    full = os.path.join(base, rel_path)
    if os.path.exists(full):
        size_kb = os.path.getsize(full) // 1024
        print(f"  ✅ {rel_path} ({size_kb} KB)")
    else:
        print(f"  ❌ {rel_path} — ФАЙЛ НЕ НАЙДЕН")
        print(f"     Ожидается по пути: {full}")
        ok = False

# Summary
print("\n" + "=" * 60)
if ok:
    print("✅ Всё готово! Запускайте:")
    print("   python main.py        — консольный режим")
    print("   python app.py         — веб-интерфейс (открыть http://localhost:5050)")
else:
    print("❌ Есть проблемы — исправьте их выше, затем повторите проверку.")
    print("   python check_env.py")
print("=" * 60)

sys.exit(0 if ok else 1)
