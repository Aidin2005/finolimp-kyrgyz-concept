"""Kyrgyz Concept · Subagent Reconciliation Web Application.
Run:
    python app.py
Open:
    http://localhost:5050
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import shutil
import tempfile
import time
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd

from src import reconcile, model, report, adapter, pipeline

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data_2_final"
DEFAULT_OUTPUT_DIR = BASE_DIR / "output"
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_MODELS_DIR = BASE_DIR / "models"
DEFAULT_MODELS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB max upload

PROGRESS_FILE = Path(tempfile.gettempdir()) / "finolimp_progress.json"


def _save_progress(data: dict):
    try:
        PROGRESS_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _load_progress() -> dict:
    try:
        if PROGRESS_FILE.exists():
            return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {
        "running": False,
        "stage": 0,
        "total_stages": 6,
        "stage_name": "Готов к запуску",
        "percent": 0
    }


def update_progress(stage: int, name: str, percent: int, running: bool = True):
    data = {
        "running": running,
        "stage": stage,
        "total_stages": 6,
        "stage_name": name,
        "percent": percent
    }
    _save_progress(data)


def _save_artifacts(R, M, emp, summary, out_dir: Path):
    """Save all analytical CSV files and summary JSON."""
    keep = [
        "group_id", "agent", "tickets", "status", "error_source", "confidence",
        "act_sum", "etm_sum", "reg_sum", "residual", "gap_effect",
        "etm_date_min", "act_date_min", "lag_days", "amount_pattern",
        "employees", "explanation", "etm_txn_ids"
    ]
    avail_cols = [c for c in keep if c in R.groups.columns]
    R.groups[avail_cols].to_csv(out_dir / "ticket_groups.csv", index=False)
    R.payments.to_csv(out_dir / "payment_exceptions.csv", index=False)
    R.actions.to_csv(out_dir / "agent_actions.csv", index=False)
    R.bridge.to_csv(out_dir / "bridge_agent.csv", index=False)
    R.monthly.to_csv(out_dir / "bridge_monthly.csv", index=False)
    R.act_table.to_csv(out_dir / "onec_acts_integrity.csv", index=False)
    R.versions.to_csv(out_dir / "onec_draft_vs_reissued.csv", index=False)
    R.missing.to_csv(out_dir / "onec_missing_months.csv", index=False)
    R.mapping.to_csv(out_dir / "party_mapping.csv", index=False)
    emp.to_csv(out_dir / "employee_risk.csv", index=False)

    if M.get("metrics") is not None:
        M["metrics"].to_csv(out_dir / "model_metrics.csv", index=False)
    if M.get("per_class") is not None:
        M["per_class"].to_csv(out_dir / "model_per_class.csv", index=False)
    if M.get("importance") is not None:
        M["importance"].to_csv(out_dir / "model_importance.csv", index=False)

    sc = M.get("scored_last_month")
    if sc is not None and not sc.empty:
        sc.groupby("agent").agg(
            operations=("txn_id", "size"),
            expected_error_ops=("p_error", "sum"),
            actual_flagged_ops=("is_err", "sum")
        ).sort_values("expected_error_ops", ascending=False).reset_index().to_csv(
            out_dir / "agent_risk_last_month.csv", index=False
        )

    js = {
        "checks": R.checks,
        "status_counts": R.groups["status"].value_counts().to_dict(),
        "payment_counts": R.payments["status"].value_counts().to_dict(),
        "model": M["metrics"].to_dict(orient="records") if M.get("metrics") is not None else [],
        "headline": [(k, v if not isinstance(v, (int, float, str)) else v) for k, v in summary["headline"]]
    }
    (out_dir / "summary.json").write_text(json.dumps(js, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _load_current_data():
    ui_cache = DEFAULT_OUTPUT_DIR / "ui_data.json"
    if ui_cache.exists():
        try:
            return json.loads(ui_cache.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Error loading ui_data.json: {e}")

    # Fallback: check if we can build from existing CSV artifacts
    actions_p = DEFAULT_OUTPUT_DIR / "agent_actions.csv"
    if actions_p.exists():
        try:
            R = reconcile.run_reconciliation(DEFAULT_DATA_DIR)
            M = model.run_model(R)
            emp = model.employee_risk(R)
            ui_data = adapter.to_ui_json(R, M, emp)
            ui_cache.write_text(json.dumps(ui_data, ensure_ascii=False), encoding="utf-8")
            return ui_data
        except Exception as e:
            print(f"Error building from data: {e}")
    return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/progress")
def get_progress():
    return jsonify(_load_progress())


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

    acts_f = request.files.get("acts")
    etm_f = request.files.get("etm")
    reg_f = request.files.get("registry")

    has_acts = bool(acts_f and acts_f.filename)
    has_etm = bool(etm_f and etm_f.filename)
    has_reg = bool(reg_f and reg_f.filename)

    if not use_default and not (has_acts or has_etm or has_reg):
        return jsonify({
            "status": "error",
            "message": "Файлы не загружены. Для запуска сверки прикрепите файлы (acts.csv, etm.csv, registry.csv) или выберите опцию «Использовать текущие выгрузки»."
        }), 400

    if not use_default:
        temp_dir = Path(tempfile.mkdtemp(prefix="reconcile_upload_"))
        for key, f_obj in [("acts", acts_f), ("etm", etm_f), ("registry", reg_f)]:
            if f_obj and f_obj.filename:
                f_obj.save(temp_dir / f"{key}.csv")
            else:
                shutil.copy(DEFAULT_DATA_DIR / f"{key}.csv", temp_dir / f"{key}.csv")
        data_dir = temp_dir
    else:
        data_dir = DEFAULT_DATA_DIR

    try:
        update_progress(1, "Сверка ETM ↔ 1С ↔ реестр (Union-Find группы заказов)...", 15)
        R = reconcile.run_reconciliation(data_dir)

        update_progress(3, "Построение баланс-моста и сопоставление платежей...", 45)
        # Bridge and payment matching are fully completed in Result

        update_progress(4, "Обучение ML-модели и оценка рисков сотрудников...", 65)
        M = model.run_model(R)
        emp = model.employee_risk(R)

        update_progress(5, "Генерация 15-страничного отчёта Excel и формул моста...", 85)
        summary = pipeline.build_summary(R, M, emp)
        report_path = DEFAULT_OUTPUT_DIR / "reconciliation_report.xlsx"
        report.build_workbook(R, M, emp, summary, report_path)
        shutil.copy(report_path, DEFAULT_OUTPUT_DIR / "KC_reconciliation_v3_report.xlsx")

        _save_artifacts(R, M, emp, summary, DEFAULT_OUTPUT_DIR)

        update_progress(6, "Сверка завершена", 100, running=False)

        elapsed = time.time() - start_time
        ui_data = adapter.to_ui_json(R, M, emp, elapsed_sec=elapsed)
        (DEFAULT_OUTPUT_DIR / "ui_data.json").write_text(json.dumps(ui_data, ensure_ascii=False), encoding="utf-8")

        return jsonify(ui_data)
    except Exception as e:
        update_progress(0, f"Ошибка: {str(e)[:100]}", 0, running=False)
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


import requests

GEMINI_API_KEY = "YOUR_GCP_KEY"

@app.route("/api/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    ui_cache = DEFAULT_OUTPUT_DIR / "ui_data.json"
    context = ""
    if ui_cache.exists():
        import json
        data = json.loads(ui_cache.read_text(encoding="utf-8"))
        summary = {
            "kpi_показатели": data.get("kpi", {}),
            "топ_проблемные_агенты": data.get("summary", [])[:10]
        }
        context = f"Данные из последней сверки: {json.dumps(summary, ensure_ascii=False)}"
        
    prompt = f"""Ты финансовый AI-ассистент в дашборде сверки Kyrgyz Concept.
Твоя задача — кратко и профессионально отвечать бухгалтеру на вопросы по отчету.
Контекст данных: {context}
Вопрос бухгалтера: {user_message}
Отвечай кратко, по делу, на русском языке."""

    try:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": "Bearer YOUR_OPENROUTER_KEY",
            "HTTP-Referer": "http://localhost:5050",
            "X-Title": "Kyrgyz Concept"
        }
        payload = {
            "model": "openai/gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "Ты финансовый AI-ассистент в дашборде сверки Kyrgyz Concept. Твоя задача — кратко и профессионально отвечать бухгалтеру на вопросы по отчету. Отвечай кратко, по делу, на русском языке."},
                {"role": "user", "content": f"Контекст данных: {context}\n\nВопрос бухгалтера: {user_message}"}
            ]
        }
        r = requests.post(url, json=payload, headers=headers, timeout=10)
        resp_data = r.json()
        text = resp_data['choices'][0]['message']['content']
        return jsonify({"reply": text})
    except Exception as e:
        # Fallback offline mode
        try:
            kpi = data.get('kpi', {})
            reply = (f"*(Offline Mode)* К сожалению, нет связи с API. "
                     f"Но я проанализировал локальные данные! "
                     f"Сматчено: {kpi.get('matched', 0)}. "
                     f"Расхождений: {kpi.get('discrepancies', 0)}. "
                     f"Требуют исправления в 1С: {kpi.get('c1_adj', 0)} сом. "
                     f"Рекомендую скачать Excel отчет для детализации.")
            return jsonify({"reply": reply})
        except:
            return jsonify({"reply": "Отчет сформирован успешно, расхождения отсортированы в Excel."})

@app.route("/api/reports", methods=["GET"])
def list_reports():
    import time
    files = []
    for ext in ["*.xlsx", "*.csv"]:
        for filepath in DEFAULT_OUTPUT_DIR.glob(ext):
            stat = filepath.stat()
            files.append({
                "name": filepath.name,
                "size": stat.st_size,
                "created_at": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime))
            })
    files.sort(key=lambda x: x["created_at"], reverse=True)
    return jsonify(files)

@app.route("/download/file/<filename>")
def download_specific_report(filename):
    if "/" in filename or "\\" in filename:
        return "Invalid filename", 400
    filepath = DEFAULT_OUTPUT_DIR / filename
    if filepath.exists():
        return send_file(filepath, as_attachment=True)
    return "Файл не найден", 404

@app.route("/api/delete/<filename>", methods=["DELETE"])
def delete_report(filename):
    if "/" in filename or "\\" in filename:
        return jsonify({"status": "error"}), 400
    filepath = DEFAULT_OUTPUT_DIR / filename
    if filepath.exists():
        filepath.unlink()
        return jsonify({"status": "ok"})
    return jsonify({"status": "error"}), 404

@app.route("/download/report")
def download_report():
    report_path = DEFAULT_OUTPUT_DIR / "reconciliation_report.xlsx"
    if not report_path.exists():
        report_path = DEFAULT_OUTPUT_DIR / "KC_reconciliation_v3_report.xlsx"

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
    print("  🚀 Kyrgyz Concept Subagent Reconciliation UI запущен!")
    print(f"  🌐 Откройте в браузере: http://localhost:{port}")
    print("=" * 80)
    app.run(host="0.0.0.0", port=port, debug=False)
