"""ML layer: early-warning model that predicts the SOURCE of a problem for each ETM operation.

Honest framing (also printed into the report):
  * labels are produced by the rule engine (weak labels) -> the model does not prove errors,
    it learns to prioritise operations the way the audited rules would;
  * features are known at posting time (no 1C amounts, no group outcome of the same month);
  * evaluation is strictly temporal: train on earlier months, test on a later one, twice;
  * every metric is shown next to a trivial baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.utils.class_weight import compute_sample_weight

CLASSES = ["OK", "BOT", "1C", "AGENT", "REVIEW"]
CAT = ["kind", "creator", "currency", "employee"]
NUM = ["log_amount", "hour", "weekday", "dom", "month_end", "n_tickets", "agent_n_agreements",
       "prior_same_ticket_rows", "hours_since_same_payment", "agent_prior_err_rate", "agent_prior_n"]


def build_dataset(R) -> pd.DataFrame:
    e = R.etm.copy()
    g = R.groups.set_index("gid")
    e["src"] = e["gid"].map(g["error_source"])
    pay = R.payments[R.payments["where"] == "ETM"].set_index("reference")
    is_pay = e["kind"].eq("оплата")
    e.loc[is_pay, "src"] = e.loc[is_pay, "txn_id"].map(pay["error_source"]).fillna("OK")
    e["label"] = e["src"].replace({"EXPORT": np.nan})
    e = e[e["label"].notna()].copy()
    e["label"] = e["label"].where(e["label"].isin(CLASSES), "REVIEW")

    e["log_amount"] = np.log1p(e["amount_kgs"].abs())
    e["hour"] = e["date"].dt.hour
    e["weekday"] = e["date"].dt.weekday
    e["dom"] = e["date"].dt.day
    e["month_end"] = (e["dom"] >= 28).astype(int)
    e["n_tickets"] = e["ticket_list"].map(len)
    e["agent_n_agreements"] = e["agent"].map(R.etm.groupby("agent")["agreement_id"].nunique())
    key = e["agent"] + "|" + e["tickets"].fillna("")
    e["prior_same_ticket_rows"] = e.assign(_k=key).groupby("_k").cumcount().where(e["n_tickets"] > 0, 0)
    srt = e.sort_values(["agent", "amount_kgs", "date"])
    h = srt[srt["kind"].eq("оплата")].groupby(["agent", "amount_kgs"])["date"].diff().dt.total_seconds() / 3600
    e["hours_since_same_payment"] = h.reindex(e.index).fillna(9999.0)
    reg_emp = {}
    for r in R.reg.itertuples(index=False):
        for t in r.ticket_list:
            reg_emp.setdefault((r.agent, t), r.employee)
    e["employee"] = [next((reg_emp[(a, t)] for t in tl if (a, t) in reg_emp), "none") for a, tl in zip(e["agent"], e["ticket_list"])]
    e["is_err"] = (e["label"] != "OK").astype(int)
    # leak-free agent history: error rate over strictly EARLIER months only
    m = e.groupby(["agent", "month"])["is_err"].agg(["sum", "count"]).reset_index().sort_values(["agent", "month"])
    m["cs"] = m.groupby("agent")["sum"].cumsum() - m["sum"]
    m["cc"] = m.groupby("agent")["count"].cumsum() - m["count"]
    m["agent_prior_err_rate"] = (m["cs"] + 0.05 * 20) / (m["cc"] + 20)      # smoothed toward 5%
    m["agent_prior_n"] = m["cc"]
    return e.merge(m[["agent", "month", "agent_prior_err_rate", "agent_prior_n"]], on=["agent", "month"], how="left")


def _prep(df, cats):
    X = df[CAT + NUM].copy()
    for c in CAT:
        X[c] = pd.Categorical(X[c], categories=cats[c]).codes
    return X


def _fit_eval(df, train_months, test_month):
    df = df.assign(**{c: df[c].astype(str) for c in CAT})
    cats = {c: sorted(df[c].unique()) for c in CAT}
    tr, te = df[df["month"].isin(train_months)], df[df["month"] == test_month]
    Xtr, Xte = _prep(tr, cats), _prep(te, cats)
    y_tr, y_te = tr["label"].values, te["label"].values
    clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=200, l2_regularization=1.0,
                                         categorical_features=[Xtr.columns.get_loc(c) for c in CAT], random_state=7)
    clf.fit(Xtr, y_tr, sample_weight=np.sqrt(compute_sample_weight("balanced", y_tr)))
    proba = clf.predict_proba(Xte)
    classes = list(clf.classes_)
    pred = np.array(classes)[proba.argmax(1)]
    p_err = 1 - proba[:, classes.index("OK")]
    yb = (y_te != "OK").astype(int)
    k = max(1, int(0.05 * len(te)))
    top = np.argsort(-p_err)[:k]
    base_rate = yb.mean()
    pr, rc, f1, sup = precision_recall_fscore_support(y_te, pred, labels=CLASSES, zero_division=0)
    per_class = pd.DataFrame({"class": CLASSES, "precision": pr, "recall": rc, "f1": f1, "support": sup})
    out = {
        "test_month": test_month, "train_months": f"{train_months[0]}..{train_months[-1]}", "n_train": len(tr), "n_test": len(te),
        "test_error_rate": float(base_rate),
        "accuracy": float((pred == y_te).mean()), "accuracy_majority_baseline": float((y_te == "OK").mean()),
        "macro_f1": float(f1_score(y_te, pred, labels=CLASSES, average="macro", zero_division=0)),
        "macro_f1_majority_baseline": float(f1_score(y_te, np.array(["OK"] * len(y_te)), labels=CLASSES, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_te, pred)),
        "any_error_pr_auc": float(average_precision_score(yb, p_err)) if yb.sum() else float("nan"),
        "any_error_pr_auc_baseline_prior_rate": float(average_precision_score(yb, te["agent_prior_err_rate"].values)) if yb.sum() else float("nan"),
        "any_error_pr_auc_random": float(base_rate),
        "any_error_roc_auc": float(roc_auc_score(yb, p_err)) if 0 < yb.sum() < len(yb) else float("nan"),
        "precision_at_top5pct": float(yb[top].mean()),
        "lift_at_top5pct": float(yb[top].mean() / base_rate) if base_rate else float("nan"),
        "recall_at_top5pct": float(yb[top].sum() / max(1, yb.sum())),
    }
    imp = pd.DataFrame({"feature": CAT + NUM})
    rng = np.random.default_rng(7)
    drops = []
    for c in imp["feature"]:
        Xp = Xte.copy()
        Xp[c] = rng.permutation(Xp[c].values)
        pp = 1 - clf.predict_proba(Xp)[:, classes.index("OK")]
        drops.append(out["any_error_pr_auc"] - average_precision_score(yb, pp) if yb.sum() else 0.0)
    imp["pr_auc_drop_when_shuffled"] = drops
    te = te.assign(p_error=p_err, pred=pred)
    return out, per_class, imp.sort_values("pr_auc_drop_when_shuffled", ascending=False), te


def run_model(R):
    df = build_dataset(R)
    months = sorted(df["month"].unique())
    results, per_class, importances, scored = [], [], None, None
    for i, test in enumerate([months[-1], months[-2]]):
        train = [m for m in months if m < test]
        out, pc, imp, te = _fit_eval(df, train, test)
        out["fold"] = "main (last month)" if i == 0 else "check (previous month)"
        results.append(out)
        per_class.append(pc.assign(test_month=test))
        if i == 0:
            importances, scored = imp, te
    return {"dataset": df, "metrics": pd.DataFrame(results), "per_class": pd.concat(per_class),
            "importance": importances, "scored_last_month": scored,
            "label_counts": df["label"].value_counts().rename_axis("class").reset_index(name="rows")}


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def employee_risk(R):
    """Who enters errors into the registry (registry-own errors only), with CI and a chi-square test."""
    reg = R.reg[R.reg["kind"] == "продажа"].copy()
    g = R.groups.set_index("gid")
    st = reg["gid"].map(g["status"])
    reg["is_error"] = st.isin(["REGISTRY_AMOUNT_DIFF", "REGISTRY_TICKET_IN_AMOUNT"]) | reg["registry_amount_issue"].ne("") | reg["registry_dup"]
    reg = reg[~reg["agent"].isin(["DIRECT_CLIENT", "UNASSIGNED"])]
    t = reg.groupby("employee")["is_error"].agg(["sum", "count"]).rename(columns={"sum": "errors", "count": "rows"})
    t["error_rate"] = t["errors"] / t["rows"]
    ci = [wilson(k, n) for k, n in zip(t["errors"], t["rows"])]
    t["ci95_low"], t["ci95_high"] = [c[0] for c in ci], [c[1] for c in ci]
    chi = stats.chi2_contingency(np.vstack([t["errors"], t["rows"] - t["errors"]]).T)
    t["chi2_p_value_all_employees"] = chi[1]
    t["significant_difference"] = chi[1] < 0.05
    return t.sort_values("error_rate", ascending=False).reset_index()
