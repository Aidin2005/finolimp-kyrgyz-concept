"""Load source CSVs and make comparison keys explicit and reproducible."""
from __future__ import annotations
import re
import unicodedata
from pathlib import Path
import pandas as pd
from .config import AGENT_MAP, LEGAL_FORMS

def normalize_agent(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = text.replace("ё", "е")
    for form in LEGAL_FORMS:
        text = re.sub(rf"\b{form}\b", " ", text)
    text = re.sub(r"[\"'«»“”.,;:()\-_/\\]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return AGENT_MAP.get(text, text)

def extract_ticket_ids(value: object) -> list[str]:
    """Return unique universal 10-digit ticket identifiers, preserving order."""
    digits = re.findall(r"\d{10,13}", str(value or ""))
    result = []
    for raw in digits:
        key = raw[-10:]
        if len(key) == 10 and key not in result:
            result.append(key)
    return result

def _number(text: str) -> float:
    text = text.strip().replace("\u00a0", " ")
    sign = -1 if "-" in text else 1
    text = text.replace("-", "")
    text = re.sub(r"[^0-9,\. ]", "", text).strip().replace(" ", "")
    if not text:
        return 0.0
    if "," in text and "." in text:
        # The last separator is decimal; all preceding separators are thousands.
        dec = "," if text.rfind(",") > text.rfind(".") else "."
        text = text.replace("." if dec == "," else ",", "").replace(dec, ".")
    elif "," in text:
        text = text.replace(",", ".")
    return sign * float(text)

def parse_pay_cell(text: object, rate_usd=1.0, rate_eur=1.0, rate_rub=1.0, rate_kzt=1.0) -> float:
    """Parse free-form registry amount into KGS, including a service fee component."""
    # In several malformed registry rows the ticket column is duplicated into pay_cell.
    # It is an identifier, not a monetary value, and must never become a trillion-KGS amount.
    if re.fullmatch(r"\s*\d{10,13}\s*", str(text or "")):
        return 0.0
    rates = {"usd": rate_usd, "eur": rate_eur, "rub": rate_rub, "kzt": rate_kzt, "kgs": 1.0, "сом": 1.0}
    total = 0.0
    for part in re.split(r"\+", str(text or "").lower()):
        currency_match = re.search(r"\b(kgs|usd|eur|rub|kzt|сом)\b", part)
        currency = currency_match.group(1) if currency_match else "kgs"
        # supports both 'eur 909,00' and '115 877,00 kzt'
        numeric = re.search(r"-?\s*[\d][\d .]*(?:,\d+|\.\d+)?", part)
        if numeric:
            total += _number(numeric.group()) * float(rates.get(currency, 1.0) or 1.0)
    return round(total, 2)

def _filter_final_acts(acts: pd.DataFrame) -> pd.DataFrame:
    status = acts["act_status"].fillna("").str.lower().str.strip()
    key = ["folder", "period_start"]
    has_reissued = status.eq("переиздан").groupby([acts[x] for x in key]).transform("any")
    return acts.loc[~(has_reissued & status.eq("черновик"))].copy()

def run(data_dir: str | Path):
    data_dir = Path(data_dir)
    acts = pd.read_csv(data_dir / "acts.csv")
    etm = pd.read_csv(data_dir / "etm.csv")
    registry = pd.read_csv(data_dir / "registry.csv")
    acts = _filter_final_acts(acts)
    for df, name_col, tickets_col in ((acts, "folder", "ticket_cell"), (etm, "agent", "tickets"), (registry, "party", "tickets")):
        df["agent_key"] = df[name_col].map(normalize_agent)
        df["ticket_ids"] = df[tickets_col].map(extract_ticket_ids)
        df["ticket_key"] = df["ticket_ids"].map(lambda x: "|".join(x))
        df["ticket_count"] = df["ticket_ids"].map(len).clip(lower=1)
    acts["date"] = pd.to_datetime(acts["date"], errors="coerce")
    acts["period"] = pd.to_datetime(acts["period_start"], errors="coerce").dt.to_period("M").astype(str)
    acts["unit_debet"] = acts["debet"] / acts["ticket_count"]
    acts["unit_credit"] = acts["credit"] / acts["ticket_count"]
    etm["date"] = pd.to_datetime(etm["date"], errors="coerce")
    etm["period"] = etm["date"].dt.to_period("M").astype(str)
    etm["unit_amount_kgs"] = etm["amount_kgs"] / etm["ticket_count"]
    registry["date"] = pd.to_datetime(registry["date"], format="%d.%m.%Y", errors="coerce")
    registry["period"] = registry["date"].dt.to_period("M").astype(str)
    registry["amount_kgs_parsed"] = registry.apply(lambda r: parse_pay_cell(r.pay_cell, r.rate_usd, r.rate_eur, r.rate_rub, r.rate_kzt), axis=1)
    registry["amount_kgs_signed"] = registry.apply(lambda r: -abs(r.amount_kgs_parsed) if str(r.kind).lower() in ("возврат", "войд") else r.amount_kgs_parsed, axis=1)
    registry["unit_amount_kgs"] = registry["amount_kgs_parsed"] / registry["ticket_count"]
    registry["unit_amount_kgs_signed"] = registry["amount_kgs_signed"] / registry["ticket_count"]
    etm_canonical = set(etm["agent_key"].unique())
    registry["is_subagent"] = registry["agent_key"].isin(etm_canonical)
    return acts, etm, registry

