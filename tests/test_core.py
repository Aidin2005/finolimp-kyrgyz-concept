"""Unit + integration tests.  Run:  python -m pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src.parsing import digit_pattern, extract_tickets, norm_text, parse_number, parse_registry_amount, registry_amount_table
from src.reconcile import COMPONENTS, prepare_acts, run_reconciliation

RATES = {"USD": 88.0, "EUR": 94.0, "RUB": 1.0, "KZT": 0.17}


def test_ticket_normalisation():
    assert extract_tickets("376-2459531798|3762459531800") == ["2459531798", "2459531800"]
    assert extract_tickets("2459531798 2459531798") == ["2459531798"]
    assert extract_tickets(None) == [] and extract_tickets("СЧ-000066") == []


def test_text_normalisation():
    assert norm_text("  SkyWay-Travel ") == "SKYWAY TRAVEL"
    assert norm_text('ОсОО "Жибек Эйр"') == norm_text("ООСО Жибек Эйр".replace("ООСО", "ОсОО"))


@pytest.mark.parametrize("raw,expected", [
    ("7.889,70", 7889.70), ("7,889.70", 7889.70), ("115 877,00", 115877.0), ("15 666.00", 15666.0),
    ("24656", 24656.0), ("331.304,00", 331304.0), ("-24.782,00", -24782.0), ("289.000", 289000.0), ("1.234.567", 1234567.0),
])
def test_parse_number(raw, expected):
    assert parse_number(raw) == pytest.approx(expected)


def test_registry_amount_with_fx_and_service_fee():
    assert parse_registry_amount("7.889,70 USD + 35usd sf", RATES) == pytest.approx((7889.70 + 35) * 88)
    assert parse_registry_amount("kgs 15 666.00 + 300 KGS sf", RATES) == pytest.approx(15966.0)
    assert parse_registry_amount("-10 768,65 kgs (штраф 15%)", RATES) == pytest.approx(-10768.65)
    assert parse_registry_amount("kzt -279.760,00", RATES) == pytest.approx(-279760 * 0.17)
    assert np.isnan(parse_registry_amount(None, RATES))


def test_ticket_typed_into_amount_column_is_flagged():
    reg = pd.DataFrame({"pay_cell": ["3762459527703", "10 000 kgs"], "rate_usd": 88.0, "rate_eur": 94.0, "rate_rub": 1.0, "rate_kzt": 0.17})
    t = registry_amount_table(reg)
    assert t.loc[0, "registry_amount_issue"] == "TICKET_IN_AMOUNT" and np.isnan(t.loc[0, "registry_amount_kgs"])
    assert t.loc[1, "registry_amount_kgs"] == pytest.approx(10000.0)


def test_digit_pattern():
    assert digit_pattern(12345.00, 12354.00) == "TRANSPOSITION"
    assert digit_pattern(12345.00, 12845.00) == "ONE_WRONG_DIGIT"
    assert digit_pattern(100.0, 100.0) == ""


# ---------------------------------------------------------------- integration on a tiny synthetic case
def _write_case(tmp_path):
    A = "Agent A"
    acts = pd.DataFrame([
        # order 1: two tickets in ONE 1C line (one-to-many against ETM) — must match, no flag
        dict(date="2026-01-03", doc="Реализация NB AAAAAA", invoice="СЧ-1", ticket_cell="1000000001 1000000002", debet=3000.0, credit=np.nan),
        # order 2: ticket booked once in 1C, debited twice in ETM -> DOUBLE_DEBIT
        dict(date="2026-01-04", doc="Реализация NB BBBBBB", invoice="СЧ-2", ticket_cell="1000000003", debet=1000.0, credit=np.nan),
        # order 3: wrong amount in 1C (500) while registry and ETM say 800 -> error of 1C, confirmed
        dict(date="2026-01-05", doc="Реализация NB CCCCCC", invoice="СЧ-3", ticket_cell="1000000004", debet=500.0, credit=np.nan),
        # payment matched with ETM payment
        dict(date="2026-01-10", doc="Поступление на р/с, п/п ЦБ-С000001", invoice=np.nan, ticket_cell=np.nan, debet=np.nan, credit=5000.0),
    ])
    acts["folder"] = A
    acts["period_start"], acts["period_end"], acts["act_status"] = "2026-01-01", "2026-01-31", np.nan
    acts["saldo_start"], acts["saldo_end"], acts["pax"] = 0.0, -500.0, np.nan
    etm_rows = [("2026-01-03 09:00:00", "выкуп", "1000000001", -1500.0, "etm-bot"), ("2026-01-03 09:00:01", "выкуп", "1000000002", -1500.0, "etm-bot"),
                ("2026-01-04 10:00:00", "выкуп", "1000000003", -1000.0, "etm-bot"), ("2026-01-04 10:05:00", "выкуп", "1000000003", -1000.0, "etm-bot"),
                ("2026-01-05 11:00:00", "выкуп", "1000000004", -800.0, "etm-bot"), ("2026-01-10 12:00:00", "оплата", None, 5000.0, "etm-bot")]
    bal, rows = 0.0, []
    for i, (d, k, t, a, c) in enumerate(etm_rows):
        bal += a
        rows.append(dict(agent=A, agreement_id="AGR-1", txn_id=100 + i, date=d, kind=k, tickets=t, amount=a, currency="KGS",
                         amount_kgs=a, balance_after=bal, creator=c, comment=""))
    etm = pd.DataFrame(rows)
    reg = pd.DataFrame([
        dict(date="03.01.2026", employee="e1", kind="продажа", party=A, tickets="1000000001 1000000002", pay_cell="3000 kgs"),
        dict(date="04.01.2026", employee="e1", kind="продажа", party=A, tickets="1000000003", pay_cell="1000 kgs"),
        dict(date="05.01.2026", employee="e2", kind="продажа", party=A, tickets="1000000004", pay_cell="800 kgs"),
        dict(date="06.01.2026", employee="e2", kind="продажа", party="ОсОО Клиент", tickets="9999999999", pay_cell="700 kgs"),
    ])
    for c in ("pax", "pnr", "airline", "route"):
        reg[c] = ""
    reg["rate_usd"], reg["rate_eur"], reg["rate_rub"], reg["rate_kzt"] = 88.0, 94.0, 1.0, 0.17
    d = tmp_path / "data"
    d.mkdir()
    acts.to_csv(d / "acts.csv", index=False)
    etm.to_csv(d / "etm.csv", index=False)
    reg.to_csv(d / "registry.csv", index=False)
    return d


def test_end_to_end_synthetic(tmp_path):
    R = run_reconciliation(_write_case(tmp_path))
    st = R.groups.set_index("tickets")["status"].to_dict()
    assert st["1000000001|1000000002"] == "OK"                       # one-to-many handled as a group
    assert st["1000000003"] == "DOUBLE_DEBIT"
    assert st["1000000004"] == "AMOUNT_RESIDUAL"
    row = R.groups.set_index("tickets").loc["1000000004"]
    assert row["error_source"] == "1C" and row["confidence"] == "A"  # registry agrees with ETM
    assert (R.groups["status"] == "REGISTRY_DIRECT_CLIENT").sum() == 1
    assert R.payments.empty                                          # payment matched 1:1
    b = R.bridge.iloc[0]
    assert b["gap_close"] == pytest.approx(-1300.0)                  # -500 (1C) + -800 (ETM)
    assert b["double_debit"] == pytest.approx(-1000.0) and b["amount_residual_1c"] == pytest.approx(-300.0)
    assert abs(b["unexplained"]) < 1e-6                              # the bridge is exact
    a = R.actions.iloc[0]
    assert a["etm_adjust_kgs"] == pytest.approx(1000.0) and a["onec_adjust_kgs"] == pytest.approx(300.0)
    assert abs(a["check_after_fixes"]) < 1e-6


def test_bridge_has_all_components():
    assert len(COMPONENTS) == len(set(COMPONENTS)) == 19


def test_latest_version_of_act_is_used():
    base = dict(folder="F", period_start="2026-01-01", period_end="2026-01-31", saldo_start=0.0, invoice=np.nan, pax=np.nan,
                ticket_cell="1000000001", credit=np.nan, date="2026-01-02")
    acts = pd.DataFrame([
        {**base, "act_status": "черновик", "saldo_end": 100.0, "doc": "Реализация NB A", "debet": 100.0},
        {**base, "act_status": "переиздан", "saldo_end": 150.0, "doc": "Реализация NB A", "debet": 100.0},
        {**base, "act_status": "переиздан", "saldo_end": 150.0, "doc": "Реализация NB B", "debet": 50.0},
    ])
    latest, versions = prepare_acts(acts)
    assert len(latest) == 2 and set(latest["act_status"]) == {"переиздан"}
    assert versions.loc[0, "delta_saldo_end"] == pytest.approx(50.0)
    assert abs(versions.loc[0, "draft_saldo_check"]) < 1e-9 and abs(versions.loc[0, "reissued_saldo_check"]) < 1e-9


def test_duplicated_registry_row_is_not_an_amount_error(tmp_path):
    """Полностью одинаковая строка реестра: сумма не удваивается, тип — REGISTRY_DUPLICATE_ROW, на баланс не влияет."""
    d = _write_case(tmp_path)
    reg = pd.read_csv(d / "registry.csv")
    pd.concat([reg, reg.iloc[[0]]], ignore_index=True).to_csv(d / "registry.csv", index=False)
    R = run_reconciliation(d)
    g = R.groups.set_index("tickets")
    assert g.loc["1000000001|1000000002", "status"] == "REGISTRY_DUPLICATE_ROW"
    assert g.loc["1000000001|1000000002", "reg_sum"] == pytest.approx(3000.0)   # без копии
    assert g.loc["1000000001|1000000002", "reg_dup_rows"] == 1
    assert abs(R.bridge.iloc[0]["unexplained"]) < 1e-6
