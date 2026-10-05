"""Core reconciliation engine: ETM <-> 1C <-> registry.

Design decisions (all defended in README):
  * ETM `balance_after` is the running balance of the SUBAGENT (shared by all his contracts),
    so the ETM ledger is chained per subagent, not per contract.
  * Only the latest version of every 1C act is used (переиздан > no status > черновик).
  * Operations are compared as CONNECTED TICKET GROUPS (one order = several tickets),
    never one ETM row against one 1C line.
  * The bridge is an exact partition of every 1C and ETM row: nothing is plugged,
    `unexplained` is computed and must be ~0; what cannot be attributed stays visible ("clarify").
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .parsing import digit_pattern, extract_tickets, norm_text, registry_amount_table

TOL = 5.0            # KGS, ETM-vs-1C comparison (both are in KGS)
FX_REL_TOL = 0.005   # registry vs ETM/1C when a foreign currency is involved (registry rate drift)
PAY_TOL = 1.0
PAY_DAYS = 7
REPEAT_HOURS = 72
BOT = "etm-bot"


# --------------------------------------------------------------------------- IO
def load_sources(data_dir: str | Path):
    d = Path(data_dir)
    return (pd.read_csv(d / "acts.csv"), pd.read_csv(d / "etm.csv"), pd.read_csv(d / "registry.csv"))


# ------------------------------------------------------------------ preparation
def _act_kind(doc: str) -> str:
    doc = str(doc)
    if doc.startswith("Реализация сервисный сбор"):
        return "service_fee"
    if doc.startswith("Реализация"):
        return "sale"
    if doc.startswith("Возврат"):
        return "refund"
    if doc.startswith(("Поступление", "Приход")):
        return "receipt"
    return "other"


def prepare_acts(acts_raw: pd.DataFrame):
    a = acts_raw.copy()
    for c in ("period_start", "period_end", "date"):
        a[c] = pd.to_datetime(a[c], errors="coerce")
    a["ticket_list"] = a["ticket_cell"].map(extract_tickets)
    a["kind1c"] = a["doc"].map(_act_kind)
    a["mv"] = a["debet"].fillna(0.0) - a["credit"].fillna(0.0)
    rank = {"переиздан": 2, "черновик": 0}
    a["version_rank"] = a["act_status"].map(lambda s: rank.get(s, 1))
    a["act_row"] = np.arange(len(a))

    key = ["folder", "period_start"]
    best = a.groupby(key)["version_rank"].transform("max")
    latest = a[a["version_rank"] == best].copy()

    rows = []
    multi = a.groupby(key)["act_status"].nunique(dropna=False)
    for (folder, ps) in multi[multi > 1].index:
        g = a[(a["folder"] == folder) & (a["period_start"] == ps)]
        v = {s: x for s, x in g.groupby("act_status", dropna=False)}
        old, new = v.get("черновик"), v.get("переиздан")
        if old is None or new is None:
            continue
        ko = set(zip(old["doc"], old["invoice"].fillna("")))
        kn = set(zip(new["doc"], new["invoice"].fillna("")))
        rows.append({
            "folder": folder, "period": ps.strftime("%Y-%m"),
            "saldo_end_draft": old["saldo_end"].iloc[0], "saldo_end_reissued": new["saldo_end"].iloc[0],
            "delta_saldo_end": new["saldo_end"].iloc[0] - old["saldo_end"].iloc[0],
            "rows_draft": len(old), "rows_reissued": len(new),
            "rows_only_in_reissued": len(kn - ko), "rows_only_in_draft": len(ko - kn),
            "draft_saldo_check": old["saldo_end"].iloc[0] - (old["saldo_start"].iloc[0] + old["mv"].sum()),
            "reissued_saldo_check": new["saldo_end"].iloc[0] - (new["saldo_start"].iloc[0] + new["mv"].sum()),
        })
    return latest, pd.DataFrame(rows)


def prepare_etm(etm_raw: pd.DataFrame) -> pd.DataFrame:
    e = etm_raw.copy()
    e["date"] = pd.to_datetime(e["date"], errors="coerce")
    e["ticket_list"] = e["tickets"].map(extract_tickets)
    e["month"] = e["date"].dt.to_period("M").astype(str)
    for c in ("amount_kgs", "balance_after", "amount"):
        e[c] = pd.to_numeric(e[c], errors="coerce")
    e = e.sort_values(["agent", "date", "txn_id"]).reset_index(drop=True)
    e["etm_row"] = np.arange(len(e))
    return e


def prepare_registry(reg_raw: pd.DataFrame) -> pd.DataFrame:
    r = reg_raw.copy()
    # полностью одинаковая строка реестра (все поля) = задвоенная запись; первая остаётся, остальные помечаются
    r["registry_dup"] = r.duplicated(keep="first")
    r["date"] = pd.to_datetime(r["date"], dayfirst=True, errors="coerce")
    r["ticket_list"] = r["tickets"].map(extract_tickets)
    r = pd.concat([r, registry_amount_table(r)], axis=1)
    r["registry_row"] = np.arange(len(r))
    return r


def map_agents(latest_acts, etm, reg):
    """ticket->agent, 1C folder->ETM agent, registry party->ETM agent, with evidence table."""
    t2a: dict[str, str] = {}
    conflicts = 0
    for row in etm.itertuples(index=False):
        for t in row.ticket_list:
            if t2a.setdefault(t, row.agent) != row.agent:
                conflicts += 1

    folder_map, folder_evidence = {}, []
    etm_norm = {norm_text(a): a for a in etm["agent"].unique()}
    for folder, g in latest_acts.groupby("folder"):
        c = Counter(t2a[t] for l in g["ticket_list"] for t in l if t in t2a)
        if c:
            agent, n = c.most_common(1)[0]
            folder_map[folder] = agent
            folder_evidence.append((folder, agent, n, n / sum(c.values())))
        elif norm_text(folder) in etm_norm:
            folder_map[folder] = etm_norm[norm_text(folder)]
            folder_evidence.append((folder, folder_map[folder], 0, 1.0))
        else:
            folder_map[folder] = f"UNMAPPED:{folder}"
            folder_evidence.append((folder, folder_map[folder], 0, 0.0))
    for row in latest_acts.itertuples(index=False):
        for t in row.ticket_list:
            t2a.setdefault(t, folder_map[row.folder])

    party_cnt = defaultdict(Counter)
    for row in reg.itertuples(index=False):
        for t in row.ticket_list:
            if t in t2a:
                party_cnt[row.party][t2a[t]] += 1
    party_map, party_evidence = {}, []
    for party, c in party_cnt.items():
        agent, n = c.most_common(1)[0]
        conf = n / sum(c.values())
        party_evidence.append((party, agent, n, conf))
        if conf >= 0.9:
            party_map[party] = agent
    mapping = pd.DataFrame(
        [("1C", f, a, n, c) for f, a, n, c in folder_evidence]
        + [("registry", f, a, n, c) for f, a, n, c in party_evidence],
        columns=["source", "name_in_source", "etm_agent", "shared_tickets", "confidence"],
    ).sort_values(["source", "confidence"]).reset_index(drop=True)
    return t2a, folder_map, party_map, mapping, conflicts


# ----------------------------------------------------------------------- groups
class _UF:
    def __init__(self):
        self.p: dict = {}

    def find(self, x):
        p = self.p
        p.setdefault(x, x)
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def build_groups(acts, etm, reg):
    uf = _UF()
    for df in (acts, etm, reg):
        for ag, tl in zip(df["agent"], df["ticket_list"]):
            if tl:
                first = (ag, tl[0])
                uf.find(first)
                for t in tl[1:]:
                    uf.union(first, (ag, t))
    for df in (acts, etm, reg):
        df["gid"] = ["|".join(uf.find((a, l[0]))) if l else None for a, l in zip(df["agent"], df["ticket_list"])]

    x = acts[acts["gid"].notna() & acts["kind1c"].ne("receipt")]
    g = x.groupby("gid")
    a_agg = pd.DataFrame({
        "act_sum": g["mv"].sum(), "act_rows": g.size(),
        "act_date_min": g["date"].min(), "act_date_max": g["date"].max(),
        "act_docs": g["doc"].agg(lambda s: " | ".join(list(dict.fromkeys(s))[:3])),
    })

    x = etm[etm["gid"].notna()]
    g = x.groupby("gid")
    dup = x[x["kind"].eq("выкуп")].groupby(["gid", "tickets", "amount_kgs"]).size().rename("n").reset_index()
    dup = dup[dup["n"] > 1]
    dup["extra"] = dup["amount_kgs"] * (dup["n"] - 1)
    dupsum = dup.groupby("gid")["extra"].sum().rename("dup_extra_amount")
    dupn = dup.groupby("gid")["n"].apply(lambda s: int((s - 1).sum())).rename("dup_extra_rows")
    e_agg = pd.DataFrame({
        "etm_sum": g["amount_kgs"].sum(), "etm_rows": g.size(),
        "etm_date_min": g["date"].min(), "etm_date_max": g["date"].max(),
        "etm_kinds": g["kind"].agg(lambda s: "|".join(sorted(set(s)))),
        "etm_creators": g["creator"].agg(lambda s: "|".join(sorted(set(s)))),
        "etm_bot": g["creator"].agg(lambda s: bool((s == BOT).any())),
        "has_void": g["kind"].agg(lambda s: bool((s == "войд").any())),
        "has_refund": g["kind"].agg(lambda s: bool((s == "возврат").any())),
        "etm_txn_ids": g["txn_id"].agg(lambda s: ",".join(map(str, list(s)[:6]))),
    }).join(dupsum).join(dupn)

    x_all = reg[reg["gid"].notna()]
    reg_dups = x_all.groupby("gid")["registry_dup"].sum().rename("reg_dup_rows")
    x = x_all[~x_all["registry_dup"]]
    g = x.groupby("gid")
    r_agg = pd.DataFrame({
        "reg_sum": g["registry_amount_kgs"].sum(min_count=1),
        "reg_rows": g.size(),
        "reg_unparsed": g["registry_amount_kgs"].apply(lambda s: int(s.isna().sum())),
        "reg_fx": g["registry_fx"].any(),
        "reg_kinds": g["kind"].agg(lambda s: "|".join(sorted(set(s)))),
        "employees": g["employee"].agg(lambda s: "|".join(sorted(set(s)))),
        "reg_issue": g["registry_amount_issue"].agg(lambda s: "|".join(sorted(set(i for i in s if i)))),
    })

    tickets = pd.concat([
        acts.loc[acts["gid"].notna(), ["gid", "agent", "ticket_list"]],
        etm.loc[etm["gid"].notna(), ["gid", "agent", "ticket_list"]],
        reg.loc[reg["gid"].notna(), ["gid", "agent", "ticket_list"]],
    ])
    tk = tickets.explode("ticket_list").dropna().groupby("gid").agg(
        agent=("agent", "first"), tickets=("ticket_list", lambda s: "|".join(sorted(set(s)))),
        ticket_count=("ticket_list", "nunique"))

    groups = tk.join(a_agg).join(e_agg).join(r_agg).join(reg_dups)
    for c in ("act_sum", "etm_sum", "act_rows", "etm_rows", "reg_rows", "reg_unparsed"):
        groups[c] = groups[c].fillna(0)
    for c in ("etm_bot", "has_void", "has_refund", "reg_fx"):
        groups[c] = groups[c].fillna(False).astype(bool)
    for c in ("dup_extra_amount", "dup_extra_rows", "reg_dup_rows"):
        groups[c] = groups[c].fillna(0)
    for c in ("etm_kinds", "etm_creators", "reg_kinds", "employees", "reg_issue", "act_docs", "etm_txn_ids"):
        groups[c] = groups[c].fillna("")
    groups["residual"] = groups["act_sum"] + groups["etm_sum"]
    groups = groups.rename_axis("gid").reset_index()
    groups.insert(0, "group_id", [f"G{i:05d}" for i in range(len(groups))])
    return groups


def _month_missing(agent, ts, missing_by_agent) -> bool:
    """True if the act of ts's month, or of the following month (lag), is absent for the agent."""
    if pd.isna(ts):
        return False
    m0 = ts.to_period("M")
    ms = missing_by_agent.get(agent, set())
    return m0 in ms or (m0 + 1) in ms


def _reg_agrees(reg_sum, reg_fx, ref):
    tol = max(TOL, FX_REL_TOL * abs(ref)) if reg_fx else TOL
    return abs(reg_sum - ref) <= tol


def classify_groups(groups: pd.DataFrame, missing_by_agent) -> pd.DataFrame:
    """status / error_source / confidence / explanation / fix_side for each ticket group."""
    out = []
    for g in groups.itertuples(index=False):
        has1c, hasetm, hasreg = g.act_rows > 0, g.etm_rows > 0, g.reg_rows > 0
        reg_ok = hasreg and g.reg_unparsed == 0 and not np.isnan(g.reg_sum)
        r = g.residual
        status, source, grade, why, side = "OK", "OK", "A", "Три источника не противоречат друг другу", "none"

        if hasetm and not has1c:
            if g.has_void and abs(g.etm_sum) <= TOL:
                status, why = "VOID_CLOSED", "Выкуп и войд взаимно закрыты в ETM; в 1С операции нет — это норма"
            elif g.has_void:
                status, source, grade, side = "VOID_UNBALANCED", "BOT", "B", "etm"
                why = f"Войд не возвращает на баланс всю сумму выкупа (остаток {g.etm_sum:,.2f})"
            elif _month_missing(g.agent, g.etm_date_min, missing_by_agent):
                status, source, grade, side = "ETM_IN_MISSING_ACT_MONTH", "EXPORT", "B", "clarify"
                why = "Операция есть в ETM, а акт 1С за этот (или следующий) месяц не выгружен — сначала получить акт"
            else:
                status, source, side = "NOT_IN_1C", "1C", "onec"
                grade = "A" if (hasreg or not g.etm_bot) else "B"
                why = ("Списание есть в ETM" + (" и продажа в реестре" if hasreg else "")
                       + ", но в 1С нет реализации по этим билетам — счёт не выставлен")
        elif has1c and not hasetm:
            if hasreg:
                status, source, grade, side = "ONEC_ONLY_REGISTRY_SALE", "BOT", "A", "etm"
                why = "Продажа есть в реестре и в 1С, но бот не перенёс её на баланс ETM — списания нет"
            else:
                status, source, grade, side = "ONEC_ONLY", "REVIEW", "C", "clarify"
                why = "Реализация в 1С есть, в ETM и реестре — нет; выяснить, кто оформил"
        elif has1c and hasetm:
            if abs(r) <= TOL:
                if g.reg_dup_rows > 0:
                    # лишняя копия строки реестра; сумма без копии сверяется как обычно
                    if reg_ok and not _reg_agrees(g.reg_sum, g.reg_fx, g.act_sum):
                        status, source, grade, side = "REGISTRY_AMOUNT_DIFF", "AGENT", "B", "none"
                        why = (f"1С и ETM согласованы ({g.act_sum:,.2f}), а в реестре сумма {g.reg_sum:,.2f} "
                               f"(без {int(g.reg_dup_rows)} лишн. копии строки): агент ошибся в реестре; на баланс это не влияет")
                    else:
                        status, source, grade, side = "REGISTRY_DUPLICATE_ROW", "AGENT", "B", "none"
                        why = (f"Строка реестра записана {int(g.reg_dup_rows) + 1} раза (все поля совпадают); "
                               f"1С и ETM согласованы ({g.act_sum:,.2f}), без копии реестр тоже совпадает. "
                               "Удалить лишнюю строку в реестре; на баланс это не влияет")
                elif reg_ok and g.etm_bot and not _reg_agrees(g.reg_sum, g.reg_fx, g.act_sum):
                    status, source, grade, side = "REGISTRY_AMOUNT_DIFF", "AGENT", "B", "none"
                    why = (f"1С и ETM согласованы ({g.act_sum:,.2f}), а в реестре сумма {g.reg_sum:,.2f}: "
                           "агент ошибся в реестре; на баланс это не влияет")
            else:
                status = "AMOUNT_RESIDUAL"
                if g.dup_extra_rows > 0 and abs(r - g.dup_extra_amount) <= TOL:
                    status, source, grade, side = "DOUBLE_DEBIT", "BOT", "A", "etm"
                    why = (f"В ETM {int(g.dup_extra_rows)} лишн. списание(й) одного билета на "
                           f"{-g.dup_extra_amount:,.2f}; без дубля сумма совпадает с 1С")
                elif reg_ok:
                    a_etm = _reg_agrees(g.reg_sum, g.reg_fx, -g.etm_sum)
                    a_1c = _reg_agrees(g.reg_sum, g.reg_fx, g.act_sum)
                    if a_etm and not a_1c:
                        source, grade, side = "1C", "A", "onec"
                        why = (f"Реестр ({g.reg_sum:,.2f}) совпадает с ETM ({-g.etm_sum:,.2f}), "
                               f"а в 1С {g.act_sum:,.2f}: неверная сумма в 1С")
                    elif a_1c and not a_etm:
                        source, grade, side = ("BOT", "A", "etm") if g.etm_bot else ("REVIEW", "A", "clarify")
                        why = (f"Реестр ({g.reg_sum:,.2f}) совпадает с 1С ({g.act_sum:,.2f}), "
                               f"а в ETM {-g.etm_sum:,.2f}: бот перенёс неверную сумму")
                    else:
                        source, grade, side = "REVIEW", "C", "clarify"
                        why = "Реестр не подтверждает ни 1С, ни ETM (либо состав группы неоднозначен) — нужна проверка"
                else:
                    pat = digit_pattern(g.act_sum, g.etm_sum)
                    if pat and not g.etm_bot:
                        source, grade, side = "1C", "B", "onec"
                        why = (f"Суммы 1С ({g.act_sum:,.2f}) и ETM ({-g.etm_sum:,.2f}) отличаются "
                               f"{'перестановкой' if pat == 'TRANSPOSITION' else 'одной цифрой'} — похоже на ручную опечатку "
                               "при вводе акта в 1С (гипотеза, подтвердить у бухгалтера)")
                    else:
                        source, grade, side = "REVIEW", "C", "clarify"
                        why = ("Нет записи реестра (билет выписан субагентом сам) — определить виновного по данным нельзя"
                               if not g.etm_bot else "Сумма реестра не разобрана/отсутствует — нужна проверка")
        elif g.agent == "DIRECT_CLIENT":
            status, why = "REGISTRY_DIRECT_CLIENT", "Продажа корпоративному клиенту напрямую (не субагент): баланс субагента не затрагивается"
        else:   # registry only
            if hasreg and g.reg_kinds == "продажа":
                status, source, grade, side = "REGISTRY_ONLY_SALE", "REVIEW", "C", "none"
                why = ("Продажа есть только в реестре: нет ни ETM, ни 1С. Возможно, бот пропустил и счёт не "
                       "выставлен, либо агент внёс лишнюю запись")
            else:
                status, source, grade, side = "REGISTRY_ONLY_OTHER", "REVIEW", "C", "none"
                why = "Возврат/войд только в реестре"

        if status == "OK" and g.reg_issue == "TICKET_IN_AMOUNT":
            status, source, grade, side = "REGISTRY_TICKET_IN_AMOUNT", "AGENT", "A", "none"
            why = "В реестре в колонке суммы записан номер билета; на балансе расхождения нет"
        out.append((status, source, grade, why, side))
    res = pd.DataFrame(out, columns=["status", "error_source", "confidence", "explanation", "fix_side"], index=groups.index)
    groups = pd.concat([groups, res], axis=1)
    no_balance = ["REGISTRY_ONLY_SALE", "REGISTRY_ONLY_OTHER", "REGISTRY_DIRECT_CLIENT"]
    groups["gap_effect"] = np.where(groups["status"].isin(no_balance), 0.0, groups["residual"])
    groups["amount_pattern"] = [digit_pattern(a, b) if s == "AMOUNT_RESIDUAL" else ""
                                for a, b, s in zip(groups["act_sum"], groups["etm_sum"], groups["status"])]
    groups["lag_days"] = (groups["act_date_min"] - groups["etm_date_min"].dt.normalize()).dt.days
    return groups


# --------------------------------------------------------------------- payments
def match_payments(acts, etm):
    """ETM 'оплата' <-> 1C receipt: same agent, |amount| <= 1 KGS, date <= 7 days, one-to-one."""
    pay = etm[etm["kind"].eq("оплата")].copy()
    rc = acts[acts["kind1c"].eq("receipt")].copy()
    rc["amt"] = -rc["mv"]
    pay["matched_act_row"] = np.nan
    used: set[int] = set()
    rc_by_agent = {a: g.sort_values("date") for a, g in rc.groupby("agent")}
    for idx, p in pay.sort_values("date").iterrows():
        cand = rc_by_agent.get(p["agent"])
        if cand is None:
            continue
        c = cand[(cand["amt"] - p["amount_kgs"]).abs() <= PAY_TOL]
        c = c[~c["act_row"].isin(used)]
        if c.empty:
            continue
        dd = (c["date"] - p["date"].normalize()).abs().dt.days
        dd = dd[dd <= PAY_DAYS]
        if dd.empty:
            continue
        best = dd.idxmin()
        used.add(int(c.loc[best, "act_row"]))
        pay.loc[idx, "matched_act_row"] = int(c.loc[best, "act_row"])
    return pay, rc, used


def classify_payments(pay, rc, used, missing_by_agent):
    rows = []
    pay = pay.sort_values(["agent", "amount_kgs", "date"]).copy()
    prev_dt = pay.groupby(["agent", "amount_kgs"])["date"].shift()
    pay["hours_since_same"] = (pay["date"] - prev_dt).dt.total_seconds() / 3600
    for p in pay.itertuples(index=False):
        if not np.isnan(p.matched_act_row):
            continue
        rep = (not np.isnan(p.hours_since_same)) and p.hours_since_same <= REPEAT_HOURS
        if rep:
            st, src, side, gr = "REPEATED_CREDIT", "BOT", "etm", "A"
            why = f"Оплата на {p.amount_kgs:,.2f} повторно зачислена в ETM через {p.hours_since_same:.1f} ч, в 1С поступление одно"
        elif _month_missing(p.agent, p.date, missing_by_agent):
            st, src, side, gr = "ETM_PAYMENT_IN_MISSING_ACT_MONTH", "EXPORT", "clarify", "B"
            why = "Оплата есть в ETM, акт 1С за месяц не выгружен"
        else:
            st, src, side, gr = "PAYMENT_NOT_IN_1C", "1C", "onec", "B"
            why = f"Оплата {p.amount_kgs:,.2f} есть в ETM, поступления в 1С нет"
        rows.append((p.agent, p.date, "ETM", p.txn_id, p.amount_kgs, st, src, gr, why, side, p.amount_kgs))

    unmatched_1c = rc[~rc["act_row"].isin(used)]
    # pair 1C receipts with unmatched ETM payments of ANOTHER agent (same amount, <= 10 days)
    cand = [i for i, r in enumerate(rows) if r[5] in ("PAYMENT_NOT_IN_1C", "ETM_PAYMENT_IN_MISSING_ACT_MONTH")]
    taken: set[int] = set()
    paired_1c: dict[int, int] = {}
    for r in unmatched_1c.itertuples(index=False):
        for i in cand:
            if i in taken:
                continue
            c = rows[i]
            if c[0] != r.agent and abs(c[4] - r.amt) <= PAY_TOL and abs((c[1] - r.date).days) <= 10:
                taken.add(i)
                paired_1c[r.act_row] = i
                break
    for i in taken:
        c = rows[i]
        rows[i] = (c[0], c[1], "ETM", c[3], c[4], "PAYMENT_WRONG_AGENT", "BOT", "B",
                   f"Оплата {c[4]:,.2f} зачислена в ETM этому субагенту, а в 1С поступление проведено другому — вероятно, бот зачислил не тому",
                   "etm", c[4])
    for r in unmatched_1c.itertuples(index=False):
        if r.act_row in paired_1c:
            other = rows[paired_1c[r.act_row]][0]
            rows.append((r.agent, r.date, "1C", r.doc, r.amt, "PAYMENT_WRONG_AGENT", "BOT", "B",
                         f"Поступление {r.amt:,.2f} по 1С у этого субагента, в ETM такая оплата зачислена субагенту «{other}»",
                         "etm", -r.amt))
        else:
            rows.append((r.agent, r.date, "1C", r.doc, r.amt, "PAYMENT_NOT_IN_ETM", "BOT", "B",
                         f"Поступление {r.amt:,.2f} есть в 1С, зачисления на баланс ETM нет", "etm", -r.amt))
    return pd.DataFrame(rows, columns=["agent", "date", "where", "reference", "amount_kgs", "status",
                                       "error_source", "confidence", "explanation", "fix_side", "gap_effect"])


# ----------------------------------------------------------------------- bridge
def etm_balance_series(etm):
    """Per-agent opening balance and the chain test, chained on the subagent."""
    e = etm.copy()
    first = e.groupby("agent").first()
    open_bal = (first["balance_after"] - first["amount_kgs"]).rename("etm_open")
    last = e.groupby("agent").last()
    check = (open_bal + e.groupby("agent")["amount_kgs"].sum() - last["balance_after"]).rename("chain_check")
    e["prev"] = e.groupby("agent")["balance_after"].shift()
    e["chk"] = e["balance_after"] - e["prev"] - e["amount_kgs"]
    bad = e[(e["chk"].abs() > 1) & e["prev"].notna()]
    return open_bal, check, e, bad


def month_windows(latest_acts):
    a = latest_acts
    starts = a.groupby("agent")["period_start"].min()
    ends = a.groupby("agent")["period_end"].max()
    gm = pd.period_range(a["period_start"].min().to_period("M"), a["period_end"].max().to_period("M"), freq="M")
    present = a.groupby("agent")["period_end"].agg(lambda s: set(s.dt.to_period("M")))
    missing = {ag: {m for m in gm if m not in present[ag]} for ag in present.index}
    return starts, ends, gm, missing


def build_act_table(latest_acts):
    g = latest_acts.groupby(["agent", "period_start", "period_end"], sort=True)
    acts = g.agg(saldo_start=("saldo_start", "first"), saldo_end=("saldo_end", "first"),
                 mv=("mv", "sum"), rows=("doc", "size"), status=("act_status", "first"),
                 folder=("folder", "first")).reset_index()
    acts["internal_delta"] = acts["saldo_end"] - (acts["saldo_start"] + acts["mv"])
    acts["prev_end"] = acts.groupby("agent")["saldo_end"].shift()
    acts["prev_period_end"] = acts.groupby("agent")["period_end"].shift()
    acts["jump"] = acts["saldo_start"] - acts["prev_end"]
    acts["consecutive"] = (acts["period_start"] - acts["prev_period_end"]).dt.days.eq(1)
    acts["continuity_break"] = acts["jump"].abs().gt(0.01) & acts["consecutive"]
    acts["jump_over_missing"] = acts["jump"].abs().gt(0.01) & acts["prev_end"].notna() & ~acts["consecutive"]
    return acts


def monthly_bridge(acts_tbl, etm, open_bal):
    """Per agent-month identity: d(gap) = act arithmetic + 1C movement + ETM movement (no plugs)."""
    rows = []
    for ag, g in acts_tbl.groupby("agent"):
        ea = etm[etm["agent"] == ag]
        for r in g.itertuples(index=False):
            end_ts = r.period_end + pd.Timedelta(days=1)
            bal_end = open_bal[ag] + ea.loc[ea["date"] < end_ts, "amount_kgs"].sum()
            mv_etm = ea.loc[(ea["date"] >= r.period_start) & (ea["date"] < end_ts), "amount_kgs"].sum()
            bal_start = bal_end - mv_etm
            gap_start, gap_end = r.saldo_start + bal_start, r.saldo_end + bal_end
            expected = r.internal_delta + r.mv + mv_etm
            rows.append({
                "agent": ag, "month": r.period_end.strftime("%Y-%m"), "saldo_start_1c": r.saldo_start,
                "saldo_end_1c": r.saldo_end, "movement_1c": r.mv, "act_internal_delta": r.internal_delta,
                "etm_balance_start": bal_start, "etm_balance_end": bal_end, "movement_etm": mv_etm,
                "gap_start": gap_start, "gap_end": gap_end, "gap_change": gap_end - gap_start,
                "identity_expected": expected, "identity_residual": gap_end - gap_start - expected,
                "jump_in_1c_saldo": r.jump if (pd.notna(r.jump) and abs(r.jump) > 0.01) else 0.0,
                "continuity_break": bool(r.continuity_break),
            })
    return pd.DataFrame(rows)


COMPONENTS = [
    "opening_gap", "onec_continuity_breaks", "onec_jumps_over_missing_acts", "onec_act_arithmetic",
    "rounding_matched_groups",
    "amount_residual_1c", "amount_residual_bot", "amount_residual_review",
    "double_debit", "onec_only_registry_sale", "onec_only_no_registry", "etm_not_in_1c",
    "void_unbalanced", "etm_in_missing_act_month", "payment_not_in_1c", "payment_not_in_etm",
    "repeated_credit", "payment_wrong_agent", "payment_in_missing_act_month",
]
ETM_SIDE = ["amount_residual_bot", "double_debit", "onec_only_registry_sale", "void_unbalanced",
            "payment_not_in_etm", "repeated_credit", "payment_wrong_agent"]
ONEC_SIDE = ["amount_residual_1c", "etm_not_in_1c", "payment_not_in_1c", "onec_continuity_breaks", "onec_act_arithmetic"]
CLARIFY = ["opening_gap", "onec_jumps_over_missing_acts", "amount_residual_review", "onec_only_no_registry",
           "etm_in_missing_act_month", "payment_in_missing_act_month"]


def agent_bridge(acts_tbl, groups, payments, etm, open_bal, windows):
    starts, ends, _, _ = windows
    rows = []
    e_by = {a: g for a, g in etm.groupby("agent")}
    g_by = {a: g for a, g in groups.groupby("agent")}
    p_by = {a: g for a, g in payments.groupby("agent")} if len(payments) else {}
    for ag, at in acts_tbl.groupby("agent"):
        ea = e_by.get(ag)
        t0, t1 = starts[ag], ends[ag] + pd.Timedelta(days=1)
        bal0 = open_bal[ag] + ea.loc[ea["date"] < t0, "amount_kgs"].sum()
        bal1 = open_bal[ag] + ea.loc[ea["date"] < t1, "amount_kgs"].sum()
        gap0 = at["saldo_start"].iloc[0] + bal0
        gap1 = at["saldo_end"].iloc[-1] + bal1
        c = dict.fromkeys(COMPONENTS, 0.0)
        c["opening_gap"] = gap0
        c["onec_continuity_breaks"] = at.loc[at["continuity_break"], "jump"].sum()
        c["onec_jumps_over_missing_acts"] = at.loc[at["jump_over_missing"], "jump"].sum()
        c["onec_act_arithmetic"] = at["internal_delta"].sum()
        gg = g_by.get(ag)
        if gg is not None:
            m = gg["status"]
            c["rounding_matched_groups"] = gg.loc[m.isin(["OK", "VOID_CLOSED", "REGISTRY_AMOUNT_DIFF", "REGISTRY_TICKET_IN_AMOUNT", "REGISTRY_DUPLICATE_ROW"]), "gap_effect"].sum()
            ar = gg[m == "AMOUNT_RESIDUAL"]
            c["amount_residual_1c"] = ar.loc[ar["error_source"] == "1C", "gap_effect"].sum()
            c["amount_residual_bot"] = ar.loc[ar["error_source"] == "BOT", "gap_effect"].sum()
            c["amount_residual_review"] = ar.loc[ar["error_source"] == "REVIEW", "gap_effect"].sum()
            c["double_debit"] = gg.loc[m == "DOUBLE_DEBIT", "gap_effect"].sum()
            c["onec_only_registry_sale"] = gg.loc[m == "ONEC_ONLY_REGISTRY_SALE", "gap_effect"].sum()
            c["onec_only_no_registry"] = gg.loc[m == "ONEC_ONLY", "gap_effect"].sum()
            c["etm_not_in_1c"] = gg.loc[m == "NOT_IN_1C", "gap_effect"].sum()
            c["void_unbalanced"] = gg.loc[m == "VOID_UNBALANCED", "gap_effect"].sum()
            c["etm_in_missing_act_month"] = gg.loc[m == "ETM_IN_MISSING_ACT_MONTH", "gap_effect"].sum()
        pp = p_by.get(ag)
        if pp is not None:
            s = pp["status"]
            c["payment_not_in_1c"] = pp.loc[s == "PAYMENT_NOT_IN_1C", "gap_effect"].sum()
            c["payment_not_in_etm"] = pp.loc[s == "PAYMENT_NOT_IN_ETM", "gap_effect"].sum()
            c["repeated_credit"] = pp.loc[s == "REPEATED_CREDIT", "gap_effect"].sum()
            c["payment_wrong_agent"] = pp.loc[s == "PAYMENT_WRONG_AGENT", "gap_effect"].sum()
            c["payment_in_missing_act_month"] = pp.loc[s == "ETM_PAYMENT_IN_MISSING_ACT_MONTH", "gap_effect"].sum()
        total = sum(c.values())
        rows.append({"agent": ag, "window_start": t0, "window_end": ends[ag], "gap_close": gap1, **c,
                     "components_sum": total, "unexplained": gap1 - total})
    return pd.DataFrame(rows)


def agent_actions(bridge: pd.DataFrame, groups: pd.DataFrame, payments: pd.DataFrame, missing_months):
    b = bridge.copy()
    b["etm_adjust_kgs"] = -b[ETM_SIDE].sum(axis=1)
    b["onec_adjust_kgs"] = -b[ONEC_SIDE].sum(axis=1)
    b["to_clarify_kgs"] = b[CLARIFY].sum(axis=1)
    b["rounding_kgs"] = b["rounding_matched_groups"]
    b["gap_after_fixes"] = b["gap_close"] + b["etm_adjust_kgs"] + b["onec_adjust_kgs"]
    b["check_after_fixes"] = b["gap_after_fixes"] - (b["to_clarify_kgs"] + b["rounding_kgs"] + b["unexplained"])
    cnt = groups.groupby(["agent", "status"]).size().unstack(fill_value=0)
    b = b.merge(cnt.add_prefix("n_"), left_on="agent", right_index=True, how="left")
    if len(payments):
        pc = payments.groupby(["agent", "status"]).size().unstack(fill_value=0)
        b = b.merge(pc.add_prefix("n_"), left_on="agent", right_index=True, how="left")
    b = b.fillna({c: 0 for c in b.columns if c.startswith("n_")})
    b["missing_1c_months"] = b["agent"].map(lambda a: len(missing_months.get(a, [])))

    def text(r):
        parts = []
        if abs(r["etm_adjust_kgs"]) >= 1:
            what = [f"{lab}: {int(r[col])}" for col, lab in [
                ("n_DOUBLE_DEBIT", "дубль списания"), ("n_ONEC_ONLY_REGISTRY_SALE", "бот не перенёс продажу"),
                ("n_REPEATED_CREDIT", "повторная оплата"), ("n_PAYMENT_NOT_IN_ETM", "оплата не зачислена"),
                ("n_PAYMENT_WRONG_AGENT", "оплата не тому субагенту")] if r.get(col, 0)]
            parts.append(f"Сдвинуть баланс ETM на {r['etm_adjust_kgs']:+,.2f}" + (f" ({'; '.join(what)})" if what else ""))
        if abs(r["onec_adjust_kgs"]) >= 1:
            what = [f"{lab}: {int(r[col])}" for col, lab in [
                ("n_NOT_IN_1C", "не выставлено в 1С"), ("n_PAYMENT_NOT_IN_1C", "оплата не проведена")] if r.get(col, 0)]
            parts.append(f"Исправить 1С на {r['onec_adjust_kgs']:+,.2f}" + (f" ({'; '.join(what)})" if what else ""))
        if abs(r["to_clarify_kgs"]) >= 1:
            parts.append(f"Сначала выяснить {r['to_clarify_kgs']:+,.2f} (начальный разрыв, недостающие акты, спорные суммы)")
        return "; ".join(parts) if parts else "Расхождений, требующих действий, нет"

    b["recommended_action"] = b.apply(text, axis=1)
    b["priority_kgs"] = b["etm_adjust_kgs"].abs() + b["onec_adjust_kgs"].abs() + b["to_clarify_kgs"].abs()
    b["status"] = np.where(b["priority_kgs"] < 1, "OK", "ACTION_REQUIRED")
    return b.sort_values("priority_kgs", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------- orchestration
@dataclass
class Result:
    acts: pd.DataFrame
    etm: pd.DataFrame
    reg: pd.DataFrame
    versions: pd.DataFrame
    mapping: pd.DataFrame
    groups: pd.DataFrame
    payments: pd.DataFrame
    act_table: pd.DataFrame
    monthly: pd.DataFrame
    bridge: pd.DataFrame
    actions: pd.DataFrame
    missing: pd.DataFrame
    checks: dict = field(default_factory=dict)


def run_reconciliation(data_dir) -> Result:
    acts_raw, etm_raw, reg_raw = load_sources(data_dir)
    latest, versions = prepare_acts(acts_raw)
    etm = prepare_etm(etm_raw)
    reg = prepare_registry(reg_raw)
    t2a, folder_map, party_map, mapping, conflicts = map_agents(latest, etm, reg)

    latest["agent"] = latest["folder"].map(folder_map)
    ambiguous = set(mapping.loc[(mapping["source"] == "registry") & (mapping["confidence"] < 0.9), "name_in_source"])

    def reg_agent(row):
        c = Counter(t2a[t] for t in row.ticket_list if t in t2a)
        if c:
            return c.most_common(1)[0][0]
        if row.party in party_map:
            return party_map[row.party]
        # a party that never shares a ticket with any subagent is a direct (corporate) client
        return "UNASSIGNED" if row.party in ambiguous else "DIRECT_CLIENT"
    reg["agent"] = [reg_agent(r) for r in reg.itertuples(index=False)]

    windows = month_windows(latest)
    starts, ends, _, missing = windows
    missing_df = pd.DataFrame([(a, str(m)) for a, ms in missing.items() for m in sorted(ms)], columns=["agent", "month"])

    # ETM rows outside every agent's act window cannot be compared -> counted, excluded from groups
    in_win = (etm["date"] >= etm["agent"].map(starts)) & (etm["date"] < etm["agent"].map(ends) + pd.Timedelta(days=1))
    etm_out, etm_in = etm[~in_win].copy(), etm[in_win].copy()
    open_bal, chain_check, _, bad_rows = etm_balance_series(etm)

    groups = classify_groups(build_groups(latest.copy(), etm_in, reg), missing)
    pay, rc, used = match_payments(latest, etm_in)
    payments = classify_payments(pay, rc, used, missing)

    act_table = build_act_table(latest)
    monthly = monthly_bridge(act_table, etm, open_bal)
    bridge = agent_bridge(act_table, groups, payments, etm, open_bal, windows)
    actions = agent_actions(bridge, groups, payments, missing)

    other_etm = etm_in[etm_in["ticket_list"].map(len).eq(0) & etm_in["kind"].ne("оплата")]
    other_1c = latest[latest["ticket_list"].map(len).eq(0) & latest["kind1c"].ne("receipt")]
    checks = {
        "ticket_agent_conflicts": conflicts,
        "etm_rows_outside_window": int(len(etm_out)),
        "etm_chain_breaks_agent_level": int(len(bad_rows)),
        "etm_chain_breaks_agents": int(bad_rows["agent"].nunique()) if len(bad_rows) else 0,
        "etm_chain_total_mismatch_agents": int((chain_check.abs() > 1).sum()),
        "etm_rows_without_ticket_not_payment": int(len(other_etm)),
        "1c_rows_without_ticket_not_receipt": int(len(other_1c)),
        "registry_rows_unassigned_agent": int((reg["agent"] == "UNASSIGNED").sum()),
        "bridge_max_abs_unexplained": float(bridge["unexplained"].abs().max()),
        "monthly_identity_max_abs": float(monthly["identity_residual"].abs().max()),
        "actions_max_abs_check": float(actions["check_after_fixes"].abs().max()),
    }
    return Result(latest, etm_in, reg, versions, mapping, groups, payments, act_table, monthly, bridge, actions, missing_df, checks)
