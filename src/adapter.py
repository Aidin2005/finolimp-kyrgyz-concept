"""Adapter bridging the KC_reconciliation_v3 engine (Result dataclass)
to the Flask UI JSON contract expected by templates/index.html.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import numpy as np
import pandas as pd


STATUS_TRANSLATIONS = {
    "AMOUNT_RESIDUAL": "Разница сумм 1С и ETM",
    "NOT_IN_1C": "Списание в ETM, нет счёта в 1С",
    "ONEC_ONLY_REGISTRY_SALE": "Бот не перенёс продажу",
    "ONEC_ONLY": "Реализация только в 1С",
    "DOUBLE_DEBIT": "Двойное списание билета ботом",
    "ETM_IN_MISSING_ACT_MONTH": "Операция в ETM (акт 1С не выгружен)",
    "VOID_UNBALANCED": "Дисбаланс войда",
    "REGISTRY_ONLY_SALE": "Продажа только в реестре",
    "REGISTRY_AMOUNT_DIFF": "Ошибка суммы в реестре",
    "PAYMENT_NOT_IN_ETM": "Поступление 1С без оплаты в ETM",
    "REPEATED_CREDIT": "Повторное зачисление оплаты в ETM",
    "PAYMENT_NOT_IN_1C": "Оплата ETM без поступления в 1С",
    "PAYMENT_WRONG_AGENT": "Оплата зачислена не тому субагенту",
    "ETM_PAYMENT_IN_MISSING_ACT_MONTH": "Оплата в ETM (акт 1С не выгружен)",
}

SOURCE_TRANSLATIONS = {
    "1C": "1С",
    "BOT": "Бот",
    "AGENT": "Агент",
    "REVIEW": "Выяснить",
    "EXPORT": "Выгрузка",
    "OK": "OK",
}


def to_ui_json(R, M, emp, elapsed_sec: float | None = None) -> dict:
    """Transform engine Result, Model output, and Employee risk into the UI contract."""
    # 1. Balances per agent
    last_saldo_1c = R.act_table.groupby("agent")["saldo_end"].last().to_dict()

    summary_list = []
    for r in R.actions.itertuples(index=False):
        c1_end = float(last_saldo_1c.get(r.agent, 0.0))
        gap_close = float(r.gap_close)
        etm_end = float(gap_close - c1_end)
        etm_adj = float(r.etm_adjust_kgs)
        c1_adj = float(r.onec_adjust_kgs)
        rounding = float(r.rounding_kgs)
        clarify = float(r.to_clarify_kgs)
        unexplained = float(r.unexplained)

        status_str = "✓ Сошлось" if abs(r.priority_kgs) < 1 else "Требует внимания"

        summary_list.append({
            "Субагент (наименование)": r.agent,
            "Сальдо 1С (конец периода)": round(c1_end, 2),
            "Баланс ETM (конец периода)": round(etm_end, 2),
            "Дельта (расхождение)": round(gap_close, 2),
            "Двигать баланс ETM": round(etm_adj, 2),
            "Корректировка 1С": round(c1_adj, 2),
            "Ошибки агентов (реестр)": round(rounding, 2),
            "На выяснение": round(clarify, 2),
            "Объяснено расхождений": round(gap_close - clarify - unexplained, 2),
            "Необъяснённый остаток": round(unexplained, 2),
            "Статус сверки": status_str,
            "Ключ субагента": r.agent,
            "Рекомендация": r.recommended_action,
        })

    # Sort summary: ACTION_REQUIRED first, then by priority_kgs desc
    summary_list.sort(key=lambda x: (x["Статус сверки"] == "✓ Сошлось", -abs(x["Дельта (расхождение)"])))

    # 2. Discrepancies list
    disc_list = []
    # From ticket groups (exclude purely OK or neutral statuses)
    skip_statuses = {"OK", "VOID_CLOSED", "REGISTRY_DIRECT_CLIENT", "REGISTRY_TICKET_IN_AMOUNT", "REGISTRY_DUPLICATE_ROW"}
    bad_groups = R.groups[~R.groups["status"].isin(skip_statuses)]

    for g in bad_groups.itertuples():
        dt = g.act_date_min if pd.notna(g.act_date_min) else g.etm_date_min
        date_str = dt.strftime("%Y-%m-%d") if pd.notna(dt) else "—"
        period_str = dt.strftime("%Y-%m") if pd.notna(dt) else "—"
        amt = float(abs(g.gap_effect if g.gap_effect != 0 else g.residual))

        disc_list.append({
            "agent_key": g.agent,
            "date": date_str,
            "period": period_str,
            "ticket": str(g.tickets or "—"),
            "amount_kgs": round(amt, 2),
            "source": SOURCE_TRANSLATIONS.get(g.error_source, str(g.error_source)),
            "error_type": STATUS_TRANSLATIONS.get(g.status, str(g.status)),
            "employee": str(g.employees) if (pd.notna(g.employees) and g.employees) else "—",
            "recommendation": str(g.explanation),
            "txn_id": str(g.etm_txn_ids or g.group_id),
        })

    # From payment exceptions
    for p in R.payments.itertuples():
        dt = p.date if pd.notna(p.date) else None
        date_str = dt.strftime("%Y-%m-%d") if dt else "—"
        period_str = dt.strftime("%Y-%m") if dt else "—"

        disc_list.append({
            "agent_key": p.agent,
            "date": date_str,
            "period": period_str,
            "ticket": "—",
            "amount_kgs": round(float(abs(p.amount_kgs)), 2),
            "source": SOURCE_TRANSLATIONS.get(p.error_source, str(p.error_source)),
            "error_type": STATUS_TRANSLATIONS.get(p.status, str(p.status)),
            "employee": "—",
            "recommendation": str(p.explanation),
            "txn_id": str(p.reference),
        })

    # Sort discrepancies by amount_kgs desc
    disc_list.sort(key=lambda x: -x["amount_kgs"])

    # 3. Anomalies list
    anomalies_list = []
    # Double debits
    for g in R.groups[R.groups["status"] == "DOUBLE_DEBIT"].itertuples():
        dt = g.etm_date_min if pd.notna(g.etm_date_min) else g.act_date_min
        anomalies_list.append({
            "agent_key": g.agent,
            "date": dt.strftime("%Y-%m-%d") if pd.notna(dt) else "—",
            "ticket": str(g.tickets or "—"),
            "anomaly_type": "Двойное списание",
            "severity": "Высокий",
            "amount_kgs": round(float(abs(g.dup_extra_amount)), 2),
            "details": f"Бот списал билет {int(g.dup_extra_rows)} лишний(х) раз(а) в ETM на сумму {abs(g.dup_extra_amount):,.2f} сом (txn: {g.etm_txn_ids})"
        })

    # Repeated credit in payments
    for p in R.payments[R.payments["status"] == "REPEATED_CREDIT"].itertuples():
        anomalies_list.append({
            "agent_key": p.agent,
            "date": p.date.strftime("%Y-%m-%d") if pd.notna(p.date) else "—",
            "ticket": "—",
            "anomaly_type": "Повторное зачисление оплаты",
            "severity": "Критический",
            "amount_kgs": round(float(abs(p.amount_kgs)), 2),
            "details": str(p.explanation)
        })

    # Wrong agent payments
    for p in R.payments[R.payments["status"] == "PAYMENT_WRONG_AGENT"].itertuples():
        anomalies_list.append({
            "agent_key": p.agent,
            "date": p.date.strftime("%Y-%m-%d") if pd.notna(p.date) else "—",
            "ticket": "—",
            "anomaly_type": "Оплата не тому субагенту",
            "severity": "Высокий",
            "amount_kgs": round(float(abs(p.amount_kgs)), 2),
            "details": str(p.explanation)
        })

    # Unbalanced voids
    for g in R.groups[R.groups["status"] == "VOID_UNBALANCED"].itertuples():
        dt = g.etm_date_min if pd.notna(g.etm_date_min) else g.act_date_min
        anomalies_list.append({
            "agent_key": g.agent,
            "date": dt.strftime("%Y-%m-%d") if pd.notna(dt) else "—",
            "ticket": str(g.tickets or "—"),
            "anomaly_type": "Войд без возврата средств",
            "severity": "Средний",
            "amount_kgs": round(float(abs(g.etm_sum)), 2),
            "details": str(g.explanation)
        })

    # 1C continuity breaks
    for a in R.act_table[R.act_table["continuity_break"]].itertuples():
        anomalies_list.append({
            "agent_key": a.agent,
            "date": a.period_start.strftime("%Y-%m-%d") if pd.notna(a.period_start) else "—",
            "ticket": "—",
            "anomaly_type": "Разрыв сальдо между актами 1С",
            "severity": "Критический" if abs(a.jump) > 50000 else "Высокий",
            "amount_kgs": round(float(abs(a.jump)), 2),
            "details": f"Скачок сальдо между актами: начало {a.saldo_start:,.2f} != предыдущий конец {a.prev_end:,.2f} (скачок {a.jump:+,.2f} сом)"
        })

    # Sort anomalies by severity rank
    sev_rank = {"Критический": 0, "Высокий": 1, "Средний": 2, "Низкий": 3}
    anomalies_list.sort(key=lambda x: (sev_rank.get(x["severity"], 9), -x["amount_kgs"]))

    # 4. Risk list (ML scoring)
    risk_list = []
    sc = M.get("scored_last_month")
    if sc is not None and not sc.empty:
        agg = sc.groupby("agent").agg(
            всего_транзакций=("txn_id", "size"),
            средняя_вероятность_ошибки=("p_error", "mean"),
            фактических_ошибок=("is_err", "sum"),
            рисковый_оборот_сом=("amount_kgs", lambda s: float(s.abs()[sc.loc[s.index, "p_error"] > 0.1].sum()))
        ).sort_values("средняя_вероятность_ошибки", ascending=False).reset_index()

        for idx, row in agg.iterrows():
            risk_list.append({
                "ранг_риска": idx + 1,
                "agent_key": row["agent"],
                "название": row["agent"],
                "всего_транзакций": int(row["всего_транзакций"]),
                "средняя_вероятность_ошибки": round(float(row["средняя_вероятность_ошибки"]), 4),
                "фактических_ошибок": int(row["фактических_ошибок"]),
                "рисковый_оборот_сом": round(float(row["рисковый_оборот_сом"]), 2),
            })

    # 5. KPI metrics
    # Matched operations: all operations in OK/clean groups and matched payments
    matched_groups_idx = R.groups[R.groups["status"].isin(["OK", "VOID_CLOSED", "REGISTRY_DIRECT_CLIENT", "REGISTRY_TICKET_IN_AMOUNT", "REGISTRY_DUPLICATE_ROW"])].index
    matched_ops_count = int(R.etm["gid"].isin(matched_groups_idx).sum())
    if matched_ops_count == 0:
        matched_ops_count = int((R.groups["status"] == "OK").sum())

    etm_adj_total = float(R.actions["etm_adjust_kgs"].abs().sum())
    c1_adj_total = float(R.actions["onec_adjust_kgs"].abs().sum())

    ml_metrics_df = M.get("metrics")
    roc_auc = 0.5588
    if ml_metrics_df is not None and not ml_metrics_df.empty:
        v = ml_metrics_df.iloc[0].get("any_error_roc_auc")
        if pd.notna(v):
            roc_auc = round(float(v), 4)

    return {
        "status": "ok",
        "available": True,
        "elapsed_sec": round(elapsed_sec, 1) if elapsed_sec is not None else None,
        "last_updated": datetime.now().strftime("%d.%m.%Y %H:%M"),
        "kpi": {
            "matched": matched_ops_count,
            "discrepancies": len(disc_list),
            "etm_adj": round(etm_adj_total, 2),
            "c1_adj": round(c1_adj_total, 2),
            "anomalies": len(anomalies_list),
            "ml_roc_auc": roc_auc,
        },
        "summary": summary_list,
        "discrepancies": disc_list,
        "anomalies": anomalies_list,
        "risk": risk_list,
    }
