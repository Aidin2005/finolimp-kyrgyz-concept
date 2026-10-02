"""Machine Learning pipeline: predicts transaction error risk and error source with temporal validation."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

def run(etm: pd.DataFrame, registry: pd.DataFrame, matched: pd.DataFrame, discrepancies: pd.DataFrame, output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Graceful fallback if lightgbm is not installed ──────────────────────
    if not _LGB_AVAILABLE:
        msg = (
            "================================================================================\n"
            "ML модель не обучена: библиотека lightgbm не установлена.\n"
            "Установите: pip install lightgbm>=4.0\n"
            "После установки запустите main.py заново — все остальные шаги работают.\n"
            "================================================================================\n"
        )
        (output_dir / "model_metrics.txt").write_text(msg, encoding="utf-8")
        empty = pd.DataFrame(columns=["agent_key", "название", "ранг_риска"])
        empty.to_csv(output_dir / "risk_ranking.csv", index=False)
        return None, empty

    # 1. Map transaction targets from detected discrepancies
    disc_sub = discrepancies[discrepancies.txn_id.notna()].copy()
    disc_txn_map = {}
    for _, row in disc_sub.iterrows():
        disc_txn_map[row.txn_id] = row.source

    df = etm.copy()
    df["is_error"] = df.txn_id.isin(disc_txn_map).astype(int)

    source_code_map = {"OK": 0, "1С": 1, "Бот": 2, "Агент": 3}
    df["source_name"] = df.txn_id.map(lambda tid: disc_txn_map.get(tid, "OK"))
    df["source_class"] = df.source_name.map(source_code_map).fillna(0).astype(int)

    # 2. Extract rich features
    # Map operator employee and airline from registry if available
    reg_exp = registry.explode("ticket_ids")
    t_to_emp = dict(zip(reg_exp.ticket_ids, reg_exp.employee))
    t_to_air = dict(zip(reg_exp.ticket_ids, reg_exp.airline))

    df["employee"] = df.ticket_ids.map(lambda lst: t_to_emp.get(lst[0]) if lst else "none").fillna("none")
    df["airline"] = df.ticket_ids.map(lambda lst: t_to_air.get(lst[0]) if lst else "none").fillna("none")

    def _get_prefix(val):
        digits = "".join(c for c in str(val) if c.isdigit())
        return digits[:3] if len(digits) >= 13 else "other"

    df["airline_prefix"] = df.tickets.map(_get_prefix)
    df["abs_amount"] = df.amount_kgs.abs()
    df["log_amount"] = np.log1p(df.abs_amount)
    df["hour"] = df.date.dt.hour.fillna(12).astype(int)
    df["day"] = df.date.dt.day.fillna(15).astype(int)
    df["dayofweek"] = df.date.dt.dayofweek.fillna(2).astype(int)
    df["is_weekend"] = df.dayofweek.isin([5, 6]).astype(int)
    df["is_bot"] = (df.creator == "etm-bot").astype(int)
    df["is_multi_ticket"] = (df.ticket_count > 1).astype(int)
    df["is_round_100"] = (df.abs_amount % 100 == 0).astype(int)

    # 3. Honest Temporal Split: Months 1-6 for Training, Month 7 (July 2026) for Hold-out Testing
    train = df[df.period < "2026-07"].copy()
    test = df[df.period == "2026-07"].copy()

    # Target encodings computed strictly on train set to prevent data leakage
    agent_err = train.groupby("agent_key")["is_error"].mean()
    emp_err = train.groupby("employee")["is_error"].mean()
    air_err = train.groupby("airline_prefix")["is_error"].mean()
    global_err = train["is_error"].mean()

    for d in (train, test):
        d["agent_err_rate"] = d.agent_key.map(agent_err).fillna(global_err)
        d["emp_err_rate"] = d.employee.map(emp_err).fillna(global_err)
        d["air_err_rate"] = d.airline_prefix.map(air_err).fillna(global_err)
        for col in ["kind", "currency", "creator", "employee", "airline", "airline_prefix"]:
            d[col] = d[col].astype("category")

    features = [
        "abs_amount", "log_amount", "hour", "day", "dayofweek", "is_weekend",
        "is_bot", "is_multi_ticket", "is_round_100", "ticket_count",
        "kind", "currency", "creator", "employee", "airline", "airline_prefix",
        "agent_err_rate", "emp_err_rate", "air_err_rate"
    ]

    # 4. Train Binary Model: P(error)
    binary_clf = lgb.LGBMClassifier(
        n_estimators=180,
        learning_rate=0.04,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbose=-1
    )
    binary_clf.fit(train[features], train["is_error"])

    test_probs = binary_clf.predict_proba(test[features])[:, 1]
    test_preds = (test_probs >= 0.35).astype(int)

    try:
        roc_auc = roc_auc_score(test["is_error"], test_probs)
        pr_auc = average_precision_score(test["is_error"], test_probs)
    except ValueError:
        # Edge case: test set has only one class (e.g. no errors in July)
        roc_auc = float("nan")
        pr_auc = float("nan")

    # 5. Train Multiclass Model: P(source) in {OK, 1С, Бот, Агент}
    multi_clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=4,
        n_estimators=150,
        learning_rate=0.04,
        max_depth=6,
        random_state=42,
        verbose=-1
    )
    multi_clf.fit(train[features], train["source_class"])
    multi_preds = multi_clf.predict(test[features])
    multi_acc = accuracy_score(test["source_class"], multi_preds)
    multi_f1 = f1_score(test["source_class"], multi_preds, average="macro")

    # 6. Feature Importance
    importances = pd.Series(binary_clf.feature_importances_, index=features).sort_values(ascending=False)

    # 7. Risk Ranking of Subagents & Employees
    df_all = df.copy()
    for d in [df_all]:
        d["agent_err_rate"] = d.agent_key.map(agent_err).fillna(global_err)
        d["emp_err_rate"] = d.employee.map(emp_err).fillna(global_err)
        d["air_err_rate"] = d.airline_prefix.map(air_err).fillna(global_err)
        for col in ["kind", "currency", "creator", "employee", "airline", "airline_prefix"]:
            d[col] = d[col].astype("category")

    df_all["pred_error_prob"] = binary_clf.predict_proba(df_all[features])[:, 1]

    subagent_risk = df_all.groupby("agent_key").agg(
        название=("agent", "last"),
        всего_транзакций=("txn_id", "count"),
        средняя_вероятность_ошибки=("pred_error_prob", "mean"),
        фактических_ошибок=("is_error", "sum"),
        рисковый_оборот_сом=("abs_amount", lambda s: s[df_all.loc[s.index, "pred_error_prob"] > 0.4].sum())
    ).reset_index()
    subagent_risk = subagent_risk.sort_values("средняя_вероятность_ошибки", ascending=False)
    subagent_risk["ранг_риска"] = range(1, len(subagent_risk) + 1)

    employee_risk = registry.groupby("employee").agg(
        всего_записей=("kind", "count"),
        продаж=("kind", lambda x: (x == "продажа").sum())
    ).reset_index()
    emp_disc_counts = discrepancies[discrepancies.employee.ne("—")].groupby("employee").size()
    employee_risk["выявлено_ошибок"] = employee_risk.employee.map(emp_disc_counts).fillna(0).astype(int)
    employee_risk["доля_ошибок_процент"] = (employee_risk["выявлено_ошибок"] / employee_risk["всего_записей"] * 100).round(2)
    employee_risk = employee_risk.sort_values("доля_ошибок_процент", ascending=False)

    # Save risk ranking
    subagent_risk.head(25).to_csv(output_dir / "risk_ranking.csv", index=False)

    # 8. Produce Detailed Metrics Report for Jury
    try:
        cls_report = classification_report(
            test["is_error"], test_preds,
            target_names=["Без ошибок (OK)", "Ошибка"],
            labels=[0, 1], digits=3
        )
        cm = confusion_matrix(test["is_error"], test_preds, labels=[0, 1])
    except ValueError:
        cls_report = "Недостаточно классов в тестовой выборке для classification_report."
        cm = np.zeros((2, 2), dtype=int)

    report_text = f"""================================================================================
FinOlimp 2026 · Kyrgyz Concept: Machine Learning Quality Report
================================================================================
1. ПОСТАНОВКА ЗАДАЧИ И МЕТОДОЛОГИЯ ВАЛИДАЦИИ:
   • Модели: Градиентный бустинг LightGBM (бинарная + мультиклассовая классификация).
   • Целевые метки: сформированы на основе аудита расхождений (Шаг 2-3).
   • Валидация: честный Hold-out по времени (Temporal Split).
     - Обучающая выборка (Train): 2026-01 - 2026-06 ({len(train):,} транзакций)
     - Отложенная тестовая выборка (Test): 2026-07 ({len(test):,} транзакций)
   • Отсутствие утечек (No data leakage): все target encoding вычислены строго на Train.

2. МЕТРИКИ КАЧЕСТВА НА ОТЛОЖЕННОЙ ВЫБОРКЕ (ИЮЛЬ 2026):
   • ROC-AUC (дискриминативная способность):  {roc_auc:.4f}
   • PR-AUC (Average Precision):               {pr_auc:.4f}
   • Мультиклассовая точность (Accuracy):      {multi_acc:.4f}
   • Мультиклассовый Macro F1-score:           {multi_f1:.4f}

3. CLASSIFICATION REPORT (Бинарная классификация):
{cls_report}

4. CONFUSION MATRIX (Тестовая выборка):
   [TN={cm[0,0]:>4}  FP={cm[0,1]:>4}]  (True OK / False Alarm)
   [FN={cm[1,0]:>4}  TP={cm[1,1]:>4}]  (Missed Error / Caught Error)

5. ВАЖНОСТЬ ПРИЗНАКОВ (TOP-10 FEATURE IMPORTANCE):
"""
    for idx, (feat, score) in enumerate(importances.head(10).items(), 1):
        report_text += f"   {idx:2d}. {feat:<22} : {score:>6d}\n"

    report_text += f"""
6. РАНЖИРОВАНИЕ РИСКОВ ДЛЯ БИЗНЕСА:
   • Топ-5 наиболее рискованных субагентов для приоритетной проверки:
"""
    for _, r in subagent_risk.head(5).iterrows():
        report_text += f"     - {r['название']} (P(error)={r['средняя_вероятность_ошибки']:.1%}, ошибок={r['фактических_ошибок']})\n"

    report_text += f"""
   • Рейтинг операторов реестра по доле брака:
"""
    for _, r in employee_risk.iterrows():
        report_text += f"     - {r['employee']:<15}: {r['доля_ошибок_процент']:>5.2f}% ошибок ({r['выявлено_ошибок']} из {r['всего_записей']} записей)\n"

    (output_dir / "model_metrics.txt").write_text(report_text, encoding="utf-8")
    return binary_clf, subagent_risk
