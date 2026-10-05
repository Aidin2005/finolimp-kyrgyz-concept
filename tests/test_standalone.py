"""Standalone unit and integration tests using standard unittest (no external test runner needed)."""
import math
import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

from src.parsing import (
    digit_pattern,
    extract_tickets,
    norm_text,
    parse_number,
    parse_registry_amount,
    registry_amount_table,
)
from src.reconcile import COMPONENTS, prepare_acts, run_reconciliation


RATES = {"USD": 88.0, "EUR": 94.0, "RUB": 1.0, "KZT": 0.17}


class TestEngine(unittest.TestCase):

    def test_ticket_normalisation(self):
        self.assertEqual(extract_tickets("376-2459531798|3762459531800"), ["2459531798", "2459531800"])
        self.assertEqual(extract_tickets("2459531798 2459531798"), ["2459531798"])
        self.assertEqual(extract_tickets(None), [])
        self.assertEqual(extract_tickets("СЧ-000066"), [])

    def test_text_normalisation(self):
        self.assertEqual(norm_text("  SkyWay-Travel "), "SKYWAY TRAVEL")
        self.assertEqual(norm_text('ОсОО "Жибек Эйр"'), norm_text("ООСО Жибек Эйр".replace("ООСО", "ОсОО")))

    def test_parse_number(self):
        cases = [
            ("7.889,70", 7889.70), ("7,889.70", 7889.70), ("115 877,00", 115877.0),
            ("15 666.00", 15666.0), ("24656", 24656.0), ("331.304,00", 331304.0),
            ("-24.782,00", -24782.0), ("289.000", 289000.0), ("1.234.567", 1234567.0),
        ]
        for raw, expected in cases:
            self.assertTrue(math.isclose(parse_number(raw), expected, abs_tol=1e-2), f"Failed on {raw}")

    def test_registry_amount_with_fx_and_service_fee(self):
        self.assertTrue(math.isclose(parse_registry_amount("7.889,70 USD + 35usd sf", RATES), (7889.70 + 35) * 88, abs_tol=1e-2))
        self.assertTrue(math.isclose(parse_registry_amount("kgs 15 666.00 + 300 KGS sf", RATES), 15966.0, abs_tol=1e-2))
        self.assertTrue(math.isclose(parse_registry_amount("-10 768,65 kgs (штраф 15%)", RATES), -10768.65, abs_tol=1e-2))
        self.assertTrue(math.isclose(parse_registry_amount("kzt -279.760,00", RATES), -279760 * 0.17, abs_tol=1e-2))
        self.assertTrue(np.isnan(parse_registry_amount(None, RATES)))

    def test_ticket_typed_into_amount_column_is_flagged(self):
        reg = pd.DataFrame({"pay_cell": ["3762459527703", "10 000 kgs"], "rate_usd": 88.0, "rate_eur": 94.0, "rate_rub": 1.0, "rate_kzt": 0.17})
        t = registry_amount_table(reg)
        self.assertEqual(t.loc[0, "registry_amount_issue"], "TICKET_IN_AMOUNT")
        self.assertTrue(np.isnan(t.loc[0, "registry_amount_kgs"]))
        self.assertTrue(math.isclose(t.loc[1, "registry_amount_kgs"], 10000.0, abs_tol=1e-2))

    def test_digit_pattern(self):
        self.assertEqual(digit_pattern(12345.00, 12354.00), "TRANSPOSITION")
        self.assertEqual(digit_pattern(12345.00, 12845.00), "ONE_WRONG_DIGIT")
        self.assertEqual(digit_pattern(100.0, 100.0), "")

    def test_synthetic_reconciliation_bridge(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            A = "Agent A"
            acts = pd.DataFrame([
                dict(date="2026-01-03", doc="Реализация NB AAAAAA", invoice="СЧ-1", ticket_cell="1000000001 1000000002", debet=3000.0, credit=np.nan),
                dict(date="2026-01-04", doc="Реализация NB BBBBBB", invoice="СЧ-2", ticket_cell="1000000003", debet=1000.0, credit=np.nan),
                dict(date="2026-01-05", doc="Поступление на р/с ЦБ-1", invoice=np.nan, ticket_cell=np.nan, debet=np.nan, credit=5000.0),
            ])
            acts["folder"] = A
            acts["period_start"] = "2026-01-01"
            acts["period_end"] = "2026-01-31"
            acts["saldo_start"] = 0.0
            acts["saldo_end"] = -1000.0
            acts["act_status"] = np.nan

            etm = pd.DataFrame([
                dict(agent=A, agreement_id="D1", txn_id=1, date="2026-01-03 10:00", kind="выкуп", tickets="376-1000000001", amount=1200.0, currency="KGS", amount_kgs=-1200.0, balance_after=-1200.0, creator="etm-bot", comment=""),
                dict(agent=A, agreement_id="D1", txn_id=2, date="2026-01-03 10:05", kind="выкуп", tickets="376-1000000002", amount=1800.0, currency="KGS", amount_kgs=-1800.0, balance_after=-3000.0, creator="etm-bot", comment=""),
                dict(agent=A, agreement_id="D1", txn_id=3, date="2026-01-04 11:00", kind="выкуп", tickets="376-1000000003", amount=1000.0, currency="KGS", amount_kgs=-1000.0, balance_after=-4000.0, creator="etm-bot", comment=""),
                dict(agent=A, agreement_id="D1", txn_id=4, date="2026-01-04 11:01", kind="выкуп", tickets="376-1000000003", amount=1000.0, currency="KGS", amount_kgs=-1000.0, balance_after=-5000.0, creator="etm-bot", comment=""),
                dict(agent=A, agreement_id="D1", txn_id=5, date="2026-01-05 12:00", kind="оплата", tickets=np.nan, amount=5000.0, currency="KGS", amount_kgs=5000.0, balance_after=0.0, creator="kc-admin", comment=""),
            ])

            reg = pd.DataFrame([
                dict(date="03.01.2026", employee="Иванов И.", party=A, kind="продажа", tickets="1000000001", pax="PAX", pnr="P1", route="FRU-IST", airline="TK", pay_cell="1 200 kgs", rate_usd=88.0, rate_eur=94.0, rate_rub=1.0, rate_kzt=0.17),
                dict(date="03.01.2026", employee="Иванов И.", party=A, kind="продажа", tickets="1000000002", pax="PAX", pnr="P1", route="FRU-IST", airline="TK", pay_cell="1 800 kgs", rate_usd=88.0, rate_eur=94.0, rate_rub=1.0, rate_kzt=0.17),
                dict(date="04.01.2026", employee="Петров П.", party=A, kind="продажа", tickets="1000000003", pax="PAX", pnr="P2", route="FRU-DXB", airline="FZ", pay_cell="1 000 kgs", rate_usd=88.0, rate_eur=94.0, rate_rub=1.0, rate_kzt=0.17),
            ])

            acts.to_csv(tmp / "acts.csv", index=False)
            etm.to_csv(tmp / "etm.csv", index=False)
            reg.to_csv(tmp / "registry.csv", index=False)

            R = run_reconciliation(tmp)
            # Order 1 (2 tickets in 1 1C row: 1200 + 1800 = 3000) must match as OK without splitting
            g1 = R.groups[R.groups["tickets"].str.contains("1000000001")].iloc[0]
            self.assertEqual(g1["status"], "OK")
            self.assertTrue(math.isclose(g1["residual"], 0.0, abs_tol=1e-2))

            # Order 2 must detect DOUBLE_DEBIT
            g2 = R.groups[R.groups["tickets"].str.contains("1000000003")].iloc[0]
            self.assertEqual(g2["status"], "DOUBLE_DEBIT")

            # Balance bridge unexplained residual must be zero
            self.assertTrue(math.isclose(R.bridge["unexplained"].iloc[0], 0.0, abs_tol=1e-6))


if __name__ == "__main__":
    unittest.main()
