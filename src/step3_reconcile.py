"""Build accountant-facing balance reconciliation by subagent and monthly progression."""
from __future__ import annotations
import pandas as pd

def run(acts: pd.DataFrame, etm: pd.DataFrame, matched: pd.DataFrame, discrepancies: pd.DataFrame):
    # 1. Monthly ending balances
    act_monthly = (acts.sort_values("date").groupby(["agent_key", "period"], as_index=False)
                   .agg(saldo_1c=("saldo_end", "last")))
    etm_monthly = (etm.sort_values("date").groupby(["agent_key", "period"], as_index=False)
                   .agg(balance_etm=("balance_after", "last")))

    monthly = act_monthly.merge(etm_monthly, on=["agent_key", "period"], how="outer").fillna(0)
    monthly["Дельта (расхождение)"] = monthly.saldo_1c + monthly.balance_etm

    # Aggregate discrepancies by agent, period and source
    if not discrepancies.empty:
        disc_monthly = discrepancies.groupby(["agent_key", "period", "source"])["amount_kgs"].apply(lambda s: s.abs().sum()).unstack(fill_value=0).reset_index()
    else:
        disc_monthly = pd.DataFrame(columns=["agent_key", "period", "1С", "Бот", "Агент", "Выяснить"])

    for col in ["1С", "Бот", "Агент", "Выяснить"]:
        if col not in disc_monthly.columns:
            disc_monthly[col] = 0.0

    monthly = monthly.merge(disc_monthly, on=["agent_key", "period"], how="left").fillna(0)
    monthly = monthly.rename(columns={
        "agent_key": "Субагент",
        "period": "Период",
        "saldo_1c": "Сальдо 1С (конец)",
        "balance_etm": "Баланс ETM (конец)",
        "1С": "Ошибки 1С",
        "Бот": "Ошибки бота",
        "Агент": "Ошибки агентов",
        "Выяснить": "На выяснение"
    })

    # 2. Executive Summary by Subagent (1 row per subagent at July 31, 2026)
    act_last = (acts.sort_values("date").groupby("agent_key", as_index=False)
                .agg(saldo_1c=("saldo_end", "last"), agent_display=("folder", "last")))
    etm_last = (etm.sort_values("date").groupby("agent_key", as_index=False)
                .agg(balance_etm=("balance_after", "last")))

    summary = act_last.merge(etm_last, on="agent_key", how="outer").fillna(0)
    summary["Дельта (расхождение)"] = summary.saldo_1c + summary.balance_etm

    if not discrepancies.empty:
        disc_totals = discrepancies.groupby(["agent_key", "source"])["amount_kgs"].apply(lambda s: s.abs().sum()).unstack(fill_value=0).reset_index()
    else:
        disc_totals = pd.DataFrame(columns=["agent_key", "1С", "Бот", "Агент", "Выяснить"])

    for col in ["1С", "Бот", "Агент", "Выяснить"]:
        if col not in disc_totals.columns:
            disc_totals[col] = 0.0

    summary = summary.merge(disc_totals, on="agent_key", how="left").fillna(0)
    summary = summary.rename(columns={
        "agent_display": "Субагент (наименование)",
        "agent_key": "Ключ субагента",
        "saldo_1c": "Сальдо 1С (конец периода)",
        "balance_etm": "Баланс ETM (конец периода)",
        "1С": "Корректировка 1С",
        "Бот": "Двигать баланс ETM",
        "Агент": "Ошибки агентов (реестр)",
        "Выяснить": "На выяснение"
    })

    summary["Объяснено расхождений"] = (
        summary["Корректировка 1С"] + summary["Двигать баланс ETM"] + summary["Ошибки агентов (реестр)"] + summary["На выяснение"]
    )
    summary["Необъяснённый остаток"] = summary["Дельта (расхождение)"].abs() - summary["Объяснено расхождений"]
    summary["Статус сверки"] = summary["Необъяснённый остаток"].map(
        lambda x: "✓ Сошлось" if abs(x) <= 1000.0 else "Требует внимания"
    )

    ordered_cols = [
        "Субагент (наименование)",
        "Сальдо 1С (конец периода)",
        "Баланс ETM (конец периода)",
        "Дельта (расхождение)",
        "Двигать баланс ETM",
        "Корректировка 1С",
        "Ошибки агентов (реестр)",
        "На выяснение",
        "Объяснено расхождений",
        "Необъяснённый остаток",
        "Статус сверки",
        "Ключ субагента"
    ]
    summary = summary[[c for c in ordered_cols if c in summary.columns]]
    summary = summary.sort_values("Дельта (расхождение)", ascending=False)
    return summary, discrepancies

