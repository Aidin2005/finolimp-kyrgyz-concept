"""Operational and anti-fraud anomaly detection for travel agency accounting."""
from __future__ import annotations
import pandas as pd

def run(etm: pd.DataFrame, registry: pd.DataFrame, matched: pd.DataFrame):
    records = []

    # 1. Double debits in ETM (ticket debited more than once for same agent)
    ex = etm.explode("ticket_ids").rename(columns={"ticket_ids": "ticket"})
    purchases = ex[ex.kind.eq("выкуп")]
    for (agent, ticket), group in purchases.groupby(["agent_key", "ticket"]):
        if ticket and len(group) > 1:
            records.append({
                "agent_key": agent,
                "date": group.date.min(),
                "ticket": ticket,
                "anomaly_type": "Двойное списание",
                "severity": "Высокий",
                "amount_kgs": round(abs(group.amount_kgs).sum(), 2),
                "details": f"Один билет списан {len(group)} раз в ETM (суммарно {group.amount_kgs.abs().sum():,.2f} KGS)"
            })

    # 2. Voids without return / Voids without prior purchase
    voids = ex[ex.kind.eq("войд")]
    for _, row in voids.iterrows():
        has_purchase = ((purchases.agent_key == row.agent_key) & (purchases.ticket == row.ticket)).any()
        if not has_purchase:
            records.append({
                "agent_key": row.agent_key,
                "date": row.date,
                "ticket": row.ticket,
                "anomaly_type": "Войд без выкупа",
                "severity": "Высокий",
                "amount_kgs": round(abs(row.amount_kgs), 2),
                "details": f"Операция войд на сумму {abs(row.amount_kgs):,.2f} KGS без предшествующего выкупа в ETM"
            })
        if row.amount_kgs <= 0:
            records.append({
                "agent_key": row.agent_key,
                "date": row.date,
                "ticket": row.ticket,
                "anomaly_type": "Войд без возврата средств",
                "severity": "Критический",
                "amount_kgs": round(abs(row.amount_kgs), 2),
                "details": "Аннуляция билета не увеличила депозитный баланс субагента"
            })

    # 3. Duplicate payment top-ups within 3 days
    payments = etm[etm.kind.eq("оплата")].sort_values("date").copy()
    for (agent, amount), g in payments.groupby(["agent_key", "amount_kgs"]):
        if len(g) > 1:
            time_diffs = g.date.diff().dt.total_seconds() / 86400.0
            if (time_diffs <= 3).any():
                records.append({
                    "agent_key": agent,
                    "date": g.date.min(),
                    "ticket": "—",
                    "anomaly_type": "Повторное зачисление оплаты",
                    "severity": "Высокий",
                    "amount_kgs": round(float(amount), 2),
                    "details": f"Повторный платёж {amount:,.2f} KGS зачислен {len(g)} раза в течение 3 дней"
                })

    # 4. Outlier transaction amounts (>3 sigma per operation kind)
    stats = etm.groupby(["agent_key", "kind"]).amount_kgs.agg(["mean", "std"])
    for _, row in etm.iterrows():
        key = (row.agent_key, row.kind)
        if key in stats.index:
            s = stats.loc[key]
            if pd.notna(s["std"]) and s["std"] > 0:
                deviation = abs(abs(row.amount_kgs) - abs(s["mean"]))
                if deviation > 3.5 * s["std"]:
                    records.append({
                        "agent_key": row.agent_key,
                        "date": row.date,
                        "ticket": "|".join(row.ticket_ids),
                        "anomaly_type": "Нетипично крупная сумма",
                        "severity": "Средний",
                        "amount_kgs": round(abs(row.amount_kgs), 2),
                        "details": f"Отклонение >3.5σ от среднего чека субагента ({abs(s['mean']):,.2f} KGS)"
                    })

    # 5. Registry employee error anomalies
    emp_stats = registry.groupby("employee").agg(
        total_ops=("kind", "count"),
        manual_fares=("amount_kgs_parsed", lambda x: (x > 100000).sum())
    ).reset_index()
    if len(emp_stats) > 0:
        top_vol_emp = emp_stats.sort_values("manual_fares", ascending=False).iloc[0]
        records.append({
            "agent_key": "Все субагенты",
            "date": registry.date.max(),
            "ticket": "—",
            "anomaly_type": "Концентрация крупных сумм у оператора",
            "severity": "Низкий",
            "amount_kgs": 0.0,
            "details": f"Оператор {top_vol_emp['employee']} заполнил наибольшее число билетов >100к KGS ({top_vol_emp['manual_fares']} операций)"
        })

    df = pd.DataFrame(records, columns=["agent_key", "date", "ticket", "anomaly_type", "severity", "amount_kgs", "details"])
    return df.drop_duplicates(subset=["agent_key", "ticket", "anomaly_type", "amount_kgs"]).sort_values("severity")
