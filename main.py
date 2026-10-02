"""FinOlimp 2026 · Kyrgyz Concept Subagent Reconciliation Pipeline.

Usage:
    python main.py [--data_dir data_2_final/] [--output_dir output/]
"""
from __future__ import annotations
import os, sys
# Suppress matplotlib's 'not a writable directory' noise before any heavy imports
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".matplotlib_cache"))
import argparse
from pathlib import Path
from src import (
    step1_load_clean,
    step2_matching,
    step3_reconcile,
    step4_anomalies,
    step5_ml_model,
    step6_report
)
from src.config import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR

def main(data_dir: str | Path = DEFAULT_DATA_DIR, output_dir: str | Path = DEFAULT_OUTPUT_DIR):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("  FinOlimp 2026 · Kyrgyz Concept: Автоматический пайплайн сверки расчетов")
    print("================================================================================")

    print("\n[1/6] Загрузка и нормализация данных (1С, ETM, Реестр)...")
    acts, etm, registry = step1_load_clean.run(data_dir)
    print(f"      ✓ 1С акты:   {len(acts):,} строк (100 субагентов)")
    print(f"      ✓ ETM логи:  {len(etm):,} транзакций")
    print(f"      ✓ Реестр:    {len(registry):,} записей")

    print("\n[2/6] Побилетное сопоставление операций и клиринг войдов...")
    matched, discrepancies = step2_matching.run(acts, etm, registry)
    print(f"      ✓ Успешно сматчено:      {len(matched):,} операций")
    print(f"      ✓ Выявлено расхождений:  {len(discrepancies):,} записей")

    print("\n[3/6] Сведение баланса и расчет дельты по субагентам...")
    reconciliation, discrepancies = step3_reconcile.run(acts, etm, matched, discrepancies)
    print(f"      ✓ Сводный реестр сформирован для {len(reconciliation)} субагентов")

    print("\n[4/6] Поиск финансовых аномалий и антифрод...")
    anomalies = step4_anomalies.run(etm, registry, matched)
    print(f"      ✓ Найдено аномальных операций: {len(anomalies):,}")

    print("\n[5/6] Генерация финансового отчета в Excel...")
    report_path = step6_report.run(reconciliation, discrepancies, anomalies, output_dir)
    print(f"      ✓ Книга Excel сохранена: {report_path}")

    print("\n[6/6] Обучение ML-модели (LightGBM) и скоринг рисков...")
    step5_ml_model.run(etm, registry, matched, discrepancies, output_dir)
    print(f"      ✓ Метрики модели сохранены: {output_dir / 'model_metrics.txt'}")
    print(f"      ✓ Рейтинг риска сохранен:   {output_dir / 'risk_ranking.csv'}")

    print("\n" + "=" * 80)
    print(f"✅ ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЕН. Главный отчет: {report_path}")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subagent Reconciliation Pipeline")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR, help="Path to input datasets folder")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Path to output results folder")
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
