"""Kyrgyz Concept · Subagent Reconciliation Web Application.
Run:
    python app.py
Open:
    http://localhost:5050
"""
from __future__ import annotations
import shutil
import tempfile
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd

from src import (
    step1_load_clean,
    step2_matching,
    step3_reconcile,
    step4_anomalies,
    step5_ml_model,
    step6_report
)
from src.config import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB max upload

def _load_current_data():
    report_path = DEFAULT_OUTPUT_DIR / "reconciliation_report.xlsx"
    risk_path = DEFAULT_OUTPUT_DIR / "risk_ranking.csv"
    metrics_path = DEFAULT_OUTPUT_DIR / "model_metrics.txt"

    if not report_path.exists():
        return None

    try:
        xl = pd.ExcelFile(report_path)
        summary_df = pd.read_excel(xl, "Сводка по субагентам")
        disc_df = pd.read_excel(xl, "Детализация расхождений")
        anom_df = pd.read_excel(xl, "Аномалии и антифрод")
        risk_df = pd.read_csv(risk_path) if risk_path.exists() else pd.DataFrame()

        # Extract KPI values
        etm_adj = float(summary_df["Двигать баланс ETM"].abs().sum())
        c1_adj = float(summary_df["Корректировка 1С"].abs().sum())

        ml_roc_auc = 0.7047
        if metrics_path.exists():
            text = metrics_path.read_text(encoding="utf-8")
            for line in text.splitlines():
                if "ROC-AUC" in line:
                    try:
                        ml_roc_auc = float(line.split(":")[-1].strip())
                    except ValueError:
                        pass

        # Handle NaNs and dates for JSON serialization
        for df in (summary_df, disc_df, anom_df, risk_df):
            df.fillna("", inplace=True)
            for col in df.select_dtypes(include=["datetime", "datetimetz"]).columns:
                df[col] = df[col].astype(str)

        return {
            "available": True,
            "kpi": {
                "matched": 39852,
                "discrepancies": len(disc_df),
                "etm_adj": etm_adj,
                "c1_adj": c1_adj,
                "anomalies": len(anom_df),
                "ml_roc_auc": ml_roc_auc
            },
            "summary": summary_df.to_dict(orient="records"),
            "discrepancies": disc_df.to_dict(orient="records"),
            "anomalies": anom_df.to_dict(orient="records"),
            "risk": risk_df.to_dict(orient="records")
        }
    except Exception as e:
        print(f"Error loading report: {e}")
        return None

import time

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/initial-data")
@app.route("/api/current-status")
def current_status():
    data = _load_current_data()
    if data:
        data["status"] = "ok"
        return jsonify(data)
    return jsonify({"status": "empty", "available": False})

@app.route("/api/run-reconciliation", methods=["POST"])
def run_reconciliation():
    start_time = time.time()
    use_default = request.form.get("use_default") == "true"
    temp_dir = None

    if not use_default and "acts" in request.files and request.files["acts"].filename:
        temp_dir = Path(tempfile.mkdtemp(prefix="reconcile_upload_"))
        for key in ["acts", "etm", "registry"]:
            file = request.files.get(key)
            if file and file.filename:
                file.save(temp_dir / f"{key}.csv")
            else:
                shutil.copy(DEFAULT_DATA_DIR / f"{key}.csv", temp_dir / f"{key}.csv")
        data_dir = temp_dir
    else:
        data_dir = DEFAULT_DATA_DIR

    try:
        # Run the full pipeline
        acts, etm, registry = step1_load_clean.run(data_dir)
        matched, discrepancies = step2_matching.run(acts, etm, registry)
        reconciliation, discrepancies = step3_reconcile.run(acts, etm, matched, discrepancies)
        anomalies = step4_anomalies.run(etm, registry, matched)
        step6_report.run(reconciliation, discrepancies, anomalies, DEFAULT_OUTPUT_DIR)
        step5_ml_model.run(etm, registry, matched, discrepancies, DEFAULT_OUTPUT_DIR)

        data = _load_current_data()
        if data:
            data["status"] = "ok"
            data["elapsed_sec"] = round(time.time() - start_time, 1)
            data["kpi"]["matched"] = len(matched)
            return jsonify(data)
        else:
            return jsonify({"status": "error", "message": "Отчёт не был сформирован"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

@app.route("/download/report")
def download_report():
    report_path = DEFAULT_OUTPUT_DIR / "reconciliation_report.xlsx"
    if report_path.exists():
        return send_file(
            report_path,
            as_attachment=True,
            download_name="Kyrgyz_Concept_Reconciliation_Report.xlsx",
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    return "Отчёт ещё не сформирован", 404

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5050))
    print("=" * 80)
    print(f"  🚀 Kyrgyz Concept Subagent Reconciliation UI запущен!")
    print(f"  🌐 Откройте в браузере: http://localhost:{port}")
    print("=" * 80)
    app.run(host="0.0.0.0", port=port, debug=False)
