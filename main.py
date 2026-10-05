"""FinOlimp 2026 · Kyrgyz Concept Subagent Reconciliation Pipeline (CLI).

Usage:
    python main.py [--data data_2_final/] [--output output/] [--models models/]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src import pipeline


def main():
    parser = argparse.ArgumentParser(description="FinOlimp 2026 · Сверка ETM ↔ 1С ↔ реестр субагентов")
    parser.add_argument("--data", default="data_2_final", help="Папка с acts.csv, etm.csv, registry.csv")
    parser.add_argument("--output", default="output", help="Папка для сохранения артефактов и отчёта")
    parser.add_argument("--models", default="models", help="Папка для сохранения моделей")
    args = parser.parse_args()

    pipeline.main(["--data", args.data, "--output", args.output, "--models", args.models])


if __name__ == "__main__":
    main()
