"""CLI:  python -m src.pipeline --data data --output output --models models"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .model import employee_risk, run_model
from .reconcile import run_reconciliation
from .report import build_workbook


def build_summary(R, M, emp):
    g, p, b = R.groups, R.payments, R.bridge
    st = g["status"].value_counts()
    both = g[(g.act_rows > 0) & (g.etm_rows > 0)]
    matched_share = (both["residual"].abs() <= 5).mean()
    resid = g[g.status == "AMOUNT_RESIDUAL"]
    n_review = int((resid.error_source == "REVIEW").sum())
    ok_lag = g.loc[g.status == "OK", "lag_days"]
    a = R.actions
    pay_missing = p[p.status == "PAYMENT_NOT_IN_ETM"]
    headline = [
        ("Сумма модулей: сдвиг баланса ETM по субагентам, KGS", float(a["etm_adjust_kgs"].abs().sum())),
        ("Сумма модулей: правки в 1С по субагентам, KGS", float(a["onec_adjust_kgs"].abs().sum())),
        ("Сумма модулей: «сначала выяснить» по субагентам, KGS", float(a["to_clarify_kgs"].abs().sum())),
        ("Операций ETM / строк 1С (последние версии актов) / строк реестра", f"{len(R.etm):,} / {len(R.acts):,} / {len(R.reg):,}"),
        ("Групп билетов (заказов) всего", int(len(g))),
        ("Групп, присутствующих и в ETM, и в 1С", int(len(both))),
        ("…из них сошлись до 5 KGS", f"{matched_share:.1%}"),
        ("Группы с неверной суммой", int(st.get("AMOUNT_RESIDUAL", 0))),
        ("…подтверждено реестром как ошибка 1С (A)", int(((resid.error_source == "1C") & (resid.confidence == "A")).sum())),
        ("…подтверждено реестром как ошибка бота (A)", int(((resid.error_source == "BOT") & (resid.confidence == "A")).sum())),
        ("…похоже на опечатку цифр в 1С (гипотеза, B)", int(((resid.error_source == "1C") & (resid.confidence == "B")).sum())),
        ("…источник не установить без бухгалтера (C)", n_review),
        ("Двойных списаний бота", int(st.get("DOUBLE_DEBIT", 0))),
        ("Продажи реестра и 1С, которые бот не перенёс в ETM", int(st.get("ONEC_ONLY_REGISTRY_SALE", 0))),
        ("Операции ETM без реализации в 1С (акт за месяц есть)", int(st.get("NOT_IN_1C", 0))),
        ("Операции ETM, где акт 1С за месяц не выгружен", int(st.get("ETM_IN_MISSING_ACT_MONTH", 0))),
        ("Поступления 1С без оплаты в ETM", int((p.status == "PAYMENT_NOT_IN_ETM").sum())),
        ("…их сумма, KGS (главный по величине источник разрыва)", float(-pay_missing["gap_effect"].sum())),
        ("Повторные зачисления оплат в ETM", int((p.status == "REPEATED_CREDIT").sum())),
        ("Продажи реестра корпоративным клиентам (не ошибка)", int(st.get("REGISTRY_DIRECT_CLIENT", 0))),
        ("Задвоенные строки реестра (одинаковые все поля; сумма без копии), групп", int(st.get("REGISTRY_DUPLICATE_ROW", 0))),
        ("Продажи только в реестре (нет ни ETM, ни 1С)", int(st.get("REGISTRY_ONLY_SALE", 0))),
        ("Разрывов сальдо между актами 1С (месяцы подряд)", int(R.act_table["continuity_break"].sum())),
        ("Субагент-месяцев без акта 1С", int(len(R.missing))),
        ("Нарушений арифметики акта 1С (по последним версиям)", int((R.act_table["internal_delta"].abs() > 0.02).sum())),
        ("Разрывов цепочки баланса ETM (по субагенту; одинаковая секунда)", int(R.checks["etm_chain_breaks_agent_level"])),
    ]
    caveats = [
        "Это кандидаты с источником и уверенностью, а не «N ошибок»: A — подтверждено вторым источником, B — гипотеза, C — данных недостаточно.",
        f"Источник нельзя установить для {n_review} расхождений в сумме: у билетов, выписанных самим субагентом, нет записи в реестре — третьего голоса нет.",
        "Мост построен без заплаток: каждая строка 1С и ETM попала ровно в одну колонку, «не объяснено» — формула и равна нулю; "
        "но колонки «выяснить» (начальный разрыв, недостающие акты) — честно нераспределённые суммы.",
        "Итоги «нетто» скрывают взаимное погашение разных знаков — рядом показаны суммы модулей по субагентам.",
        "ETM balance_after — сквозной баланс субагента, а не договора: цепочка проверяется по субагенту.",
        "ML размечен правилами (слабая разметка) и работает как приоритизатор; его качество показано рядом с baseline и скромное.",
    ]
    questions = [
        (1, "Сервисный сбор: проводится ли он в 1С отдельной строкой, а в ETM входит в сумму выкупа? Подтвердить, что суммы групп должны совпадать до копейки.",
         f"{int(st.get('AMOUNT_RESIDUAL', 0))} групп с остатком", "Группы с остатком, кратным сбору, перейдут из «проверка» в «норма»"),
        (2, "Лаг между проводкой ETM и реализацией 1С: какой допустим (дни)? Операция на границе месяцев — ошибка или перенос?",
         f"медиана лага OK-групп {ok_lag.median():.0f} дн.; 95-й процентиль {ok_lag.quantile(.95):.0f} дн.",
         "Операции на стыке месяцев исключаются из «нет в 1С» / «акт не выгружен»"),
        (3, "Повторяющиеся суммы разрыва сальдо 1С у разных субагентов (107 557,10; 283 308,99; 4 800; 27 500) — пакетная корректировка/сторно?",
         f"{int(R.act_table['continuity_break'].sum())} разрывов, {R.act_table.loc[R.act_table['continuity_break'], 'jump'].abs().sum():,.2f} KGS",
         "Разрыв считается задокументированной корректировкой, а не ошибкой"),
        (4, "Выгрузить недостающие акты 1С по перечню на листе «1С нет актов».",
         f"{len(R.missing)} субагент-месяцев", "Снимает группы «акт не выгружен» и «скачок через пропуск»"),
        (5, "Кто вводит сумму акта вручную и бывают ли опечатки? Как отличить их от ошибки субагента при самостоятельной выписке (нет записи в реестре)?",
         f"{n_review} групп без третьего источника", "Часть «проверка» получит владельца ошибки"),
        (6, "Начальный разрыв на 1 января: есть ли акт сверки/сальдо открытия, которое его объясняет?",
         f"{int((b['opening_gap'].abs() > 5).sum())} субагентов, нетто {b['opening_gap'].sum():,.2f} KGS", "Начальный разрыв уйдёт из «выяснить»"),
        (7, "Поступления 1С, по которым в ETM нет оплаты: зачисляются ли такие оплаты на другой договор/в другой период или действительно не доходят на депозит? "
            "Номер ЦБ-С… не уникален между субагентами, поэтому как ключ не использован.",
         f"{len(pay_missing)} поступлений, {-pay_missing['gap_effect'].sum():,.2f} KGS",
         "Крупнейшая компонента разрыва; ответ определяет, кому её исправлять — боту или казначею"),
    ]
    assumptions = [
        ("Версия акта", "Берётся последняя: переиздан > без статуса > черновик. Различия — на листе «1С черновик и переиздан»."),
        ("Единица сравнения", "Связная группа билетов внутри субагента (union-find по билетам ETM, 1С и реестра), а не строка к строке."),
        ("Допуск", "5 KGS для ETM↔1С (оба в сомах); 0,5% для реестра в валюте (курс реестра может отличаться)."),
        ("Баланс ETM", "balance_after — сквозной баланс субагента; цепочка проверяется по субагенту; одинаковая секунда — не потеря данных."),
        ("Реестр", "Сумма разбирается с учётом разделителей (7.889,70 → 7889.70) и курсов строки; номер билета в колонке суммы помечается отдельно."),
        ("Корпоративные клиенты", "Стороны реестра без единого общего билета с субагентами считаются прямыми клиентами; баланс субагента они не затрагивают."),
        ("Платежи", "Оплата ETM ↔ поступление 1С: тот же субагент, сумма ±1 KGS, дата ±7 дней, один к одному. Номера ЦБ-С… не уникальны между субагентами и ключом не служат."),
        ("Корректировки", "Сторона правки определяется источником: 1С → правка 1С; бот/ETM → правка баланса ETM; реестр → баланс не меняется; остальное → «выяснить»."),
        ("Ограничение", "Билеты, выписанные субагентом самостоятельно, не попадают в реестр, поэтому для них источник ошибки в сумме определяется только по формальным признакам."),
    ]
    return {"headline": headline, "caveats": caveats, "questions": questions, "assumptions": assumptions}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Сверка ETM ↔ 1С ↔ реестр субагентов")
    ap.add_argument("--data", default="data", help="папка с acts.csv, etm.csv, registry.csv")
    ap.add_argument("--output", default="output")
    ap.add_argument("--models", default="models")
    args = ap.parse_args(argv)
    out, mod = Path(args.output), Path(args.models)
    out.mkdir(parents=True, exist_ok=True)
    mod.mkdir(parents=True, exist_ok=True)

    print("1/4 Сверка ETM ↔ 1С ↔ реестр ...")
    R = run_reconciliation(args.data)
    print("2/4 Модель и риск-рейтинги ...")
    M = run_model(R)
    emp = employee_risk(R)
    print("3/4 Сохранение таблиц ...")
    keep = ["group_id", "agent", "tickets", "status", "error_source", "confidence", "act_sum", "etm_sum", "reg_sum", "residual",
            "gap_effect", "etm_date_min", "act_date_min", "lag_days", "amount_pattern", "employees", "explanation", "etm_txn_ids"]
    R.groups[keep].to_csv(out / "ticket_groups.csv", index=False)
    R.payments.to_csv(out / "payment_exceptions.csv", index=False)
    R.actions.to_csv(out / "agent_actions.csv", index=False)
    R.bridge.to_csv(out / "bridge_agent.csv", index=False)
    R.monthly.to_csv(out / "bridge_monthly.csv", index=False)
    R.act_table.to_csv(out / "onec_acts_integrity.csv", index=False)
    R.versions.to_csv(out / "onec_draft_vs_reissued.csv", index=False)
    R.missing.to_csv(out / "onec_missing_months.csv", index=False)
    R.mapping.to_csv(out / "party_mapping.csv", index=False)
    emp.to_csv(out / "employee_risk.csv", index=False)
    M["metrics"].to_csv(out / "model_metrics.csv", index=False)
    M["per_class"].to_csv(out / "model_per_class.csv", index=False)
    M["importance"].to_csv(out / "model_importance.csv", index=False)
    sc = M["scored_last_month"]
    sc.groupby("agent").agg(operations=("txn_id", "size"), expected_error_ops=("p_error", "sum"),
                            actual_flagged_ops=("is_err", "sum")).sort_values("expected_error_ops", ascending=False) \
        .reset_index().to_csv(out / "agent_risk_last_month.csv", index=False)
    joblib.dump({"metrics": M["metrics"].to_dict(orient="records")}, mod / "model_card.joblib")
    summary = build_summary(R, M, emp)
    print("4/4 Excel-отчёт ...")
    xlsx = out / "KC_reconciliation_v3_report.xlsx"
    build_workbook(R, M, emp, summary, xlsx)
    js = {"checks": R.checks, "status_counts": R.groups["status"].value_counts().to_dict(),
          "payment_counts": R.payments["status"].value_counts().to_dict(),
          "model": M["metrics"].to_dict(orient="records"),
          "headline": [(k, v if not isinstance(v, np.integer) else int(v)) for k, v in summary["headline"]]}
    (out / "summary.json").write_text(json.dumps(js, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("Готово:", xlsx)


if __name__ == "__main__":
    main()
