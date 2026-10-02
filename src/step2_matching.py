"""Ticket-level and payment matching between ETM, 1C acts and the sales registry."""
from __future__ import annotations
import pandas as pd
from .config import AMOUNT_TOLERANCE

OUT_COLUMNS = [
    "agent_key", "period", "date", "ticket", "amount_kgs",
    "source", "error_type", "employee", "recommendation", "txn_id"
]

def _record(agent_key, period, date, ticket, amount, source, error, recommendation, employee="—", txn_id=pd.NA):
    return {
        "agent_key": str(agent_key),
        "period": str(period),
        "date": date,
        "ticket": str(ticket),
        "amount_kgs": round(float(amount), 2),
        "source": source,
        "error_type": error,
        "employee": str(employee or "—"),
        "recommendation": recommendation,
        "txn_id": txn_id
    }

def run(acts: pd.DataFrame, etm: pd.DataFrame, registry: pd.DataFrame):
    detail = []
    matches = []

    # 1. Identify voided tickets in ETM (netted day-of-issuance cancellations)
    void_txns = etm[etm.kind.eq("войд")]
    void_tickets = set(t for cell in void_txns.ticket_ids for t in cell)

    # 2. Explode ticket IDs
    ee = etm.explode("ticket_ids").rename(columns={"ticket_ids": "ticket"})
    ee = ee[ee.ticket.notna() & ee.ticket.ne("")]

    ae = acts.explode("ticket_ids").rename(columns={"ticket_ids": "ticket"})
    ae = ae[ae.ticket.notna() & ae.ticket.ne("")]

    # Filter registry to confirmed subagents only (exclude direct corporate accounts)
    reg_sub = registry[registry.is_subagent].copy()
    re = reg_sub.explode("ticket_ids").rename(columns={"ticket_ids": "ticket"})
    re = re[re.ticket.notna() & re.ticket.ne("")]

    # Group 1C acts by (agent_key, ticket)
    # Note: 1C emits both fare and service fee rows for the same ticket, so we sum debet/credit.
    act_sales = ae[ae.doc.str.startswith("Реализация", na=False)].groupby(["agent_key", "ticket"])["unit_debet"].sum()
    act_refunds = ae[ae.doc.str.startswith("Возврат", na=False)].groupby(["agent_key", "ticket"])["unit_credit"].sum()
    act_sales_dates = ae[ae.doc.str.startswith("Реализация", na=False)].groupby(["agent_key", "ticket"])["date"].min()
    act_sales_periods = ae[ae.doc.str.startswith("Реализация", na=False)].groupby(["agent_key", "ticket"])["period"].first()

    # Group Registry by (agent_key, ticket)
    reg_sales = re[re.kind.eq("продажа")].groupby(["agent_key", "ticket"])["unit_amount_kgs"].sum()
    reg_employees = re.groupby(["agent_key", "ticket"])["employee"].first()
    reg_dates = re.groupby(["agent_key", "ticket"])["date"].min()
    reg_periods = re.groupby(["agent_key", "ticket"])["period"].first()

    etm_tickets_set = set(ee.ticket)
    etm_active_tickets_set = set(ee[~ee.ticket.isin(void_tickets)].ticket)

    # --- Match 1: ETM non-payment transactions vs 1C ---
    for _, row in ee[ee.kind.ne("оплата")].iterrows():
        t = row.ticket
        key = (row.agent_key, t)
        unit_amt = abs(float(row.unit_amount_kgs))

        if t in void_tickets:
            # Voided ticket: balanced internally within ETM, not billed by airline in 1C
            matches.append({"txn_id": row.txn_id, "ticket": t, "match_type": "VOID_CLEARED", "status": "MATCHED", "source": "OK"})
            continue

        if row.kind == "выкуп":
            if key in act_sales:
                act_val = act_sales[key]
                diff = unit_amt - act_val
                if abs(diff) <= AMOUNT_TOLERANCE:
                    matches.append({"txn_id": row.txn_id, "ticket": t, "match_type": "ETM_1C_SALE", "status": "MATCHED", "source": "OK"})
                else:
                    detail.append(_record(
                        row.agent_key, row.period, row.date, t, diff,
                        "1С", "Разница сумм 1С и ETM",
                        "Проверить тариф/сбор и скорректировать счёт в 1С",
                        txn_id=row.txn_id
                    ))
            else:
                detail.append(_record(
                    row.agent_key, row.period, row.date, t, unit_amt,
                    "1С", "Счёт не выставлен в 1С",
                    "Выставить счёт-фактуру в 1С на сумму выкупленного билета",
                    txn_id=row.txn_id
                ))

        elif row.kind == "возврат":
            if key in act_refunds:
                act_val = act_refunds[key]
                diff = unit_amt - act_val
                if abs(diff) <= AMOUNT_TOLERANCE:
                    matches.append({"txn_id": row.txn_id, "ticket": t, "match_type": "ETM_1C_REFUND", "status": "MATCHED", "source": "OK"})
                else:
                    detail.append(_record(
                        row.agent_key, row.period, row.date, t, diff,
                        "1С", "Разница сумм возврата в 1С",
                        "Скорректировать сумму возврата в 1С",
                        txn_id=row.txn_id
                    ))
            else:
                detail.append(_record(
                    row.agent_key, row.period, row.date, t, unit_amt,
                    "1С", "Возврат не проведён в 1С",
                    "Провести операцию возврата билета в 1С",
                    txn_id=row.txn_id
                ))

    # --- Match 2: 1C sales that do not exist in ETM ---
    for (agent_key, ticket), amount in act_sales.items():
        if ticket not in etm_tickets_set:
            date = act_sales_dates.get((agent_key, ticket), pd.NaT)
            period = act_sales_periods.get((agent_key, ticket), "2026-01")
            detail.append(_record(
                agent_key, period, date, ticket, amount,
                "1С", "Только в 1С (нет в ETM)",
                "Проверить обоснованность выставления счёта в 1С или сторнировать",
            ))

    # --- Match 3: Bot transactions vs Sales Registry ---
    # Bots execute purchases entered by managers into the registry.
    bots = ee[(ee.creator == "etm-bot") & (ee.kind == "выкуп") & (~ee.ticket.isin(void_tickets))]
    for _, row in bots.iterrows():
        t = row.ticket
        key = (row.agent_key, t)
        unit_amt = abs(float(row.unit_amount_kgs))

        if key in reg_sales:
            reg_val = reg_sales[key]
            diff = unit_amt - reg_val
            emp = reg_employees.get(key, "—")
            if abs(diff) <= AMOUNT_TOLERANCE:
                matches.append({"txn_id": row.txn_id, "ticket": t, "match_type": "BOT_REGISTRY", "status": "MATCHED", "source": "OK"})
            else:
                detail.append(_record(
                    row.agent_key, row.period, row.date, t, diff,
                    "Агент", "Агент ошибся в сумме",
                    f"Исправить сумму/валюту сбора в реестре (оператор: {emp})",
                    employee=emp, txn_id=row.txn_id
                ))
        else:
            detail.append(_record(
                row.agent_key, row.period, row.date, t, unit_amt,
                "Бот", "Бот взял операцию без реестра",
                "Проверить причину списания ботом в ETM при отсутствии строки реестра",
                txn_id=row.txn_id
            ))

    # Registry sales that were missed by the ETM bot
    for (agent_key, ticket), amount in reg_sales.items():
        if ticket not in etm_active_tickets_set:
            emp = reg_employees.get((agent_key, ticket), "—")
            date = reg_dates.get((agent_key, ticket), pd.NaT)
            period = reg_periods.get((agent_key, ticket), "2026-01")
            detail.append(_record(
                agent_key, period, date, ticket, amount,
                "Бот", "Бот пропустил операцию",
                f"Списать сумму билета с баланса ETM (билет введён: {emp})",
                employee=emp
            ))

    # --- Match 4: Payment reconciliation between ETM and 1C ---
    etm_payments = etm[etm.kind.eq("оплата")].copy()
    act_payments = acts[acts.doc.str.contains("Поступление|Приход", na=False) & acts.credit.notna()].copy()

    matched_etm_txns = set()
    matched_act_indices = set()

    # Pre-index 1C payments by agent_key for fast lookups
    act_pmt_by_agent = {}
    for idx, row in act_payments.iterrows():
        act_pmt_by_agent.setdefault(row.agent_key, []).append((idx, row))

    for _, row in etm_payments.iterrows():
        candidates = act_pmt_by_agent.get(row.agent_key, [])
        found_match = False
        amt = abs(float(row.amount_kgs))
        p_date = row.date.normalize() if pd.notna(row.date) else None

        for idx, act_row in candidates:
            if idx in matched_act_indices:
                continue
            act_amt = float(act_row.credit)
            if abs(act_amt - amt) <= AMOUNT_TOLERANCE:
                act_date = act_row.date.normalize() if pd.notna(act_row.date) else None
                days_diff = abs((act_date - p_date).days) if (act_date and p_date) else 0
                if days_diff <= 3:
                    matched_etm_txns.add(row.txn_id)
                    matched_act_indices.add(idx)
                    matches.append({"txn_id": row.txn_id, "ticket": "—", "match_type": "PAYMENT", "status": "MATCHED", "source": "OK"})
                    found_match = True
                    break

        if not found_match:
            detail.append(_record(
                row.agent_key, row.period, row.date, "—", amt,
                "1С", "Оплата не проведена в 1С",
                "Провести платёж в 1С (депозит в ETM пополнен, но нет документа в 1С)",
                txn_id=row.txn_id
            ))

    # Payments in 1C that were never credited to ETM balance
    for idx, act_row in act_payments.iterrows():
        if idx not in matched_act_indices:
            detail.append(_record(
                act_row.agent_key, act_row.period, act_row.date, "—", float(act_row.credit),
                "Бот", "Оплата не зачислена в ETM",
                "Зачислить поступившую оплату на депозит субагента в ETM"
            ))

    discrepancies = pd.DataFrame(detail, columns=OUT_COLUMNS)
    matches_df = pd.DataFrame(matches)
    return matches_df, discrepancies
