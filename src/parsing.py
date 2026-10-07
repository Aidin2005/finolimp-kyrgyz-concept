"""Parsing / normalisation helpers shared by the whole tool."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

TICKET_RE = re.compile(r"\d{3}-\d{10}|\d{10,13}")
FX_RE = re.compile(r"USD|EUR|RUB|KZT|\$|€", re.I)


def norm_text(value: object) -> str:
    """Upper-case, strip punctuation, unify Ё/Е and quotes -> comparable name."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    s = str(value).upper().replace("Ё", "Е")
    s = re.sub(r"[^A-ZА-Я0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def extract_tickets(value: object) -> list[str]:
    """'376-2459531798' / '3762459531798' / '2459531798' -> '2459531798' (10-digit core), order kept, no repeats."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    out: list[str] = []
    for n in TICKET_RE.findall(str(value)):
        n = n[-10:]
        if n not in out:
            out.append(n)
    return out


def parse_number(token: str) -> float | None:
    """Parse '7.889,70' / '7,889.70' / '115 877,00' / '15 666.00' / '24656' / '331.304,00' / '289.000'."""
    s = token.replace(" ", "").replace(" ", "").strip(".,")
    if not s:
        return None
    neg = s.startswith("-")
    s = s.lstrip("+-")
    if "," in s and "." in s:
        dec = "," if s.rfind(",") > s.rfind(".") else "."
        thou = "." if dec == "," else ","
        s = s.replace(thou, "").replace(dec, ".")
    elif "," in s or "." in s:
        sep = "," if "," in s else "."
        parts = s.split(sep)
        if len(parts) > 2:                                  # 1.234.567
            s = "".join(parts)
        elif len(parts[1]) == 3 and len(parts[0]) <= 3 and parts[0] != "0":
            s = "".join(parts)                              # 289.000 -> thousands separator
        else:
            s = parts[0] + "." + parts[1]
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


_CUR = {
    "KGS": ("KGS", "СОМ"),
    "USD": ("USD", "$"),
    "EUR": ("EUR", "€"),
    "RUB": ("RUB", "РУБ"),
    "KZT": ("KZT", "ТГ", "ТЕНГЕ"),
}
_NUM_RE = re.compile(r"[-+]?\d[\d\s .,]*\d|[-+]?\d")


def parse_registry_amount(cell: object, rates: dict[str, float]) -> float:
    """'7.889,70 USD + 35usd sf' -> KGS amount using the row's own FX rates (fare + service fee).

    A leading '-' means a refund; text in brackets ('(штраф 15%)') is ignored. NaN if nothing parsed.
    """
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return float("nan")
    text = re.sub(r"\(.*?\)", " ", str(cell).upper())
    parts = re.split(r"\s\+\s|\+(?=\s*[A-ZА-Я$€\d])", text)
    total, found = 0.0, False
    for part in parts:
        m = _NUM_RE.search(part)
        if not m:
            continue
        val = parse_number(m.group(0))
        if val is None:
            continue
        cur = "KGS"
        for c, names in _CUR.items():
            if any(n in part for n in names):
                cur = c
                break
        rate = 1.0 if cur == "KGS" else rates.get(cur, np.nan)
        if rate is None or (isinstance(rate, float) and np.isnan(rate)):
            continue
        total += val * float(rate)
        found = True
    return total if found else float("nan")


def registry_amount_table(reg: pd.DataFrame) -> pd.DataFrame:
    """registry_amount_kgs (NaN when unusable), registry_fx, registry_amount_issue.

    A pay_cell that is really a 10-13 digit ticket number (typed into the amount column)
    is reported as TICKET_IN_AMOUNT and not parsed.
    """
    rate_cols = {"USD": "rate_usd", "EUR": "rate_eur", "RUB": "rate_rub", "KZT": "rate_kzt"}
    amts = []
    for r in reg.itertuples(index=False):
        rates = {c: getattr(r, col, np.nan) for c, col in rate_cols.items()}
        amts.append(parse_registry_amount(r.pay_cell, rates))
    amt = pd.Series(amts, index=reg.index, dtype=float)
    cell = reg["pay_cell"].fillna("").astype(str)
    issue = pd.Series("", index=reg.index)
    digits_only = cell.str.fullmatch(r"\s*\d{10,13}\s*")
    issue[digits_only] = "TICKET_IN_AMOUNT"
    issue[amt.isna() & ~digits_only] = "UNPARSEABLE"
    amt[digits_only] = np.nan
    return pd.DataFrame({
        "registry_amount_kgs": amt,
        "registry_fx": cell.map(lambda s: bool(FX_RE.search(s))),
        "registry_amount_issue": issue,
    })


def digit_pattern(a: float, b: float) -> str:
    """Typing-error signature between two amounts: TRANSPOSITION / ONE_WRONG_DIGIT / ''."""
    sa, sb = f"{abs(a):.2f}".replace(".", ""), f"{abs(b):.2f}".replace(".", "")
    if sa == sb:
        return ""
    if len(sa) == len(sb):
        if sorted(sa) == sorted(sb):
            return "TRANSPOSITION"
        if sum(x != y for x, y in zip(sa, sb)) == 1:
            return "ONE_WRONG_DIGIT"
    return ""
