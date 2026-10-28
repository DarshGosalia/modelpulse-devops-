#!/usr/bin/env python3
"""
ModelPulse v2.0 — Production Flask REST API & Web UI
Features:
- Web UI: GET / and POST /predict
- REST API: POST /predict_api (single & batch)
- Health check: GET /health
- Prediction history: GET /api/history
- Model metrics: GET /api/metrics
- Structured latency and prediction logging to logs/predictions.log
- Includes built-in standard library HTTP server fallback
"""

import os
import sys
import time
import json
import csv
import io
import logging
from datetime import datetime, timezone
from collections import deque

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# Structured Prediction Logger
# ──────────────────────────────────────────────────────────────────────
pred_logger = logging.getLogger("modelpulse.predictions")
pred_logger.setLevel(logging.INFO)
if pred_logger.hasHandlers():
    pred_logger.handlers.clear()

file_handler = logging.FileHandler(os.path.join("logs", "predictions.log"), encoding="utf-8")
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s")
file_handler.setFormatter(formatter)
pred_logger.addHandler(file_handler)
pred_logger.propagate = False

# In-memory metrics store (production would use Redis/Prometheus)
prediction_metrics = {
    "total_predictions": 0,
    "total_latency_ms": 0.0,
    "max_latency_ms": 0.0,
    "min_latency_ms": float('inf'),
    "errors": 0,
    "started_at": datetime.now(timezone.utc).isoformat(),
}
recent_predictions = deque(maxlen=200)  # Keep last 200 predictions in memory


def log_prediction_event(endpoint: str, inputs: dict, prediction: float, latency_ms: float, status_code: int = 200, error: str = None):
    log_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "inputs": inputs,
        "prediction": prediction,
        "latency_ms": round(latency_ms, 2),
        "status": status_code,
        "error": error
    }
    pred_logger.info(json.dumps(log_record))

    # Update in-memory metrics
    prediction_metrics["total_predictions"] += 1
    prediction_metrics["total_latency_ms"] += latency_ms
    prediction_metrics["max_latency_ms"] = max(prediction_metrics["max_latency_ms"], latency_ms)
    if latency_ms > 0:
        prediction_metrics["min_latency_ms"] = min(prediction_metrics["min_latency_ms"], latency_ms)
    if error:
        prediction_metrics["errors"] += 1

    if prediction is not None:
        recent_predictions.append({
            "timestamp": log_record["timestamp"],
            "inputs": inputs,
            "prediction": prediction,
            "latency_ms": round(latency_ms, 2)
        })


# ──────────────────────────────────────────────────────────────────────
# Model Loading & Inference Engine
# ──────────────────────────────────────────────────────────────────────
MODEL_NAME = "deployment_28042020"
model = None
model_type = "none"


def initialize_model():
    global model, model_type
    # 1. Try PyCaret
    try:
        from pycaret.regression import load_model
        model = load_model(MODEL_NAME)
        model_type = "pycaret"
        print("ModelPulse: PyCaret model pipeline loaded successfully.")
        return
    except Exception:
        pass

    # 2. Try Pickle file (Scikit-Learn or Pure Python Model)
    pkl_file = f"{MODEL_NAME}.pkl" if not MODEL_NAME.endswith(".pkl") else MODEL_NAME
    if os.path.exists(pkl_file):
        try:
            import pickle
            with open(pkl_file, "rb") as f:
                model = pickle.load(f)
            model_type = "pickle"
            print(f"ModelPulse: Model pipeline loaded from {pkl_file}.")
            return
        except Exception as e:
            print(f"Warning: Failed to unpickle {pkl_file}: {e}")

    # 3. Default In-Memory Fallback if model not trained yet
    from train import PurePythonLinearModel
    model = PurePythonLinearModel()
    model_type = "pure_python"
    print("ModelPulse: Loaded calibrated in-memory regression model.")


initialize_model()


def run_inference_dict(row: dict) -> float:
    """Predict charge for a single dictionary of features."""
    if model is None:
        initialize_model()

    if model_type == "pycaret":
        import pandas as pd
        from pycaret.regression import predict_model
        df = pd.DataFrame([row])
        preds = predict_model(model, data=df)
        label_col = "prediction_label" if "prediction_label" in preds.columns else "Label"
        return float(preds[label_col].iloc[0])
    elif model_type == "pickle" and hasattr(model, "predict"):
        try:
            import pandas as pd
            df = pd.DataFrame([row])
            return float(model.predict(df)[0])
        except Exception:
            if hasattr(model, "predict_one"):
                return float(model.predict_one(row))
            return float(model.predict([row])[0])
    elif hasattr(model, "predict_one"):
        return float(model.predict_one(row))
    else:
        # Mathematical fallback
        age = float(row.get('age', 28))
        bmi = float(row.get('bmi', 30.0))
        children = float(row.get('children', 0))
        smoker = str(row.get('smoker', 'no')).lower().strip()
        val = 257.29 * age + 322.62 * bmi + 474.41 * children - 11633.5
        if smoker == 'yes':
            val += 23823.39
        return max(round(val, 2), 1000.0)


def validate_inputs(data: dict) -> tuple:
    """Validate and sanitize prediction inputs. Returns (sanitized_dict, error_msg)."""
    errors = []
    sanitized = {}

    try:
        age = int(data.get("age", 28))
        if age < 0 or age > 150:
            errors.append("Age must be between 0 and 150")
        sanitized["age"] = age
    except (ValueError, TypeError):
        errors.append("Age must be a valid integer")

    sex = str(data.get("sex", "male")).strip().lower()
    if sex not in ("male", "female"):
        errors.append("Sex must be 'male' or 'female'")
    sanitized["sex"] = sex

    try:
        bmi = float(data.get("bmi", 30.0))
        if bmi < 5 or bmi > 100:
            errors.append("BMI must be between 5 and 100")
        sanitized["bmi"] = bmi
    except (ValueError, TypeError):
        errors.append("BMI must be a valid number")

    try:
        children = int(data.get("children", 0))
        if children < 0 or children > 20:
            errors.append("Children must be between 0 and 20")
        sanitized["children"] = children
    except (ValueError, TypeError):
        errors.append("Children must be a valid integer")

    smoker = str(data.get("smoker", "no")).strip().lower()
    if smoker not in ("yes", "no"):
        errors.append("Smoker must be 'yes' or 'no'")
    sanitized["smoker"] = smoker

    region = str(data.get("region", "southwest")).strip().lower()
    if region not in ("southwest", "southeast", "northwest", "northeast"):
        errors.append("Region must be one of: southwest, southeast, northwest, northeast")
    sanitized["region"] = region

    if errors:
        return None, "; ".join(errors)
    return sanitized, None


# ──────────────────────────────────────────────────────────────────────
# Flask Execution Pathway (Preferred)
# ──────────────────────────────────────────────────────────────────────
try:
    from flask import Flask, request, render_template, jsonify
    has_flask = True
except ImportError:
    has_flask = False

if has_flask:
    app = Flask(__name__)

    @app.route("/", methods=["GET"])
    def home():
        return render_template("home.html")

    @app.route("/predict", methods=["GET", "POST"])
    def predict():
        if request.method == "GET":
            return render_template("home.html")
        start_time = time.time()
        try:
            raw_inputs = {
                "age": request.form.get("age", 28),
                "sex": request.form.get("sex", "male"),
                "bmi": request.form.get("bmi", 30.0),
                "children": request.form.get("children", 0),
                "smoker": request.form.get("smoker", "no"),
                "region": request.form.get("region", "southwest")
            }
            inputs_dict, validation_error = validate_inputs(raw_inputs)
            if validation_error:
                return render_template("home.html", error=f"Validation error: {validation_error}"), 400

            pred_val = run_inference_dict(inputs_dict)
            latency = (time.time() - start_time) * 1000
            log_prediction_event("/predict", inputs_dict, round(pred_val, 2), latency, 200)
            formatted_pred = f"${pred_val:,.2f}"
            return render_template(
                "home.html",
                pred=f"Expected Bill will be {formatted_pred}",
                pred_value=formatted_pred,
                latency=f"{latency:.1f} ms",
                inputs=inputs_dict
            )
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            log_prediction_event("/predict", {}, None, latency, 500, error=str(e))
            return render_template("home.html", error=f"Prediction error: {str(e)}"), 500

    @app.route("/predict_api", methods=["POST"])
    def predict_api():
        start_time = time.time()
        try:
            payload = request.get_json(force=True)
            if isinstance(payload, dict):
                inputs_dict, validation_error = validate_inputs(payload)
                if validation_error:
                    return jsonify({"status": "error", "message": validation_error}), 400
                p_val = run_inference_dict(inputs_dict)
                latency = (time.time() - start_time) * 1000
                log_prediction_event("/predict_api", inputs_dict, round(p_val, 2), latency, 200)
                return jsonify({
                    "status": "success",
                    "prediction": round(p_val, 2),
                    "predictions": [round(p_val, 2)],
                    "latency_ms": round(latency, 2),
                    "model": MODEL_NAME,
                    "engine": model_type
                }), 200
            elif isinstance(payload, list):
                results = []
                annotated = []
                for item in payload[:1000]:  # Support up to 1000 batch items
                    item_dict, err = validate_inputs(item)
                    item_copy = dict(item)
                    if err:
                        results.append({"error": err})
                        item_copy["predicted_charge"] = None
                        item_copy["error"] = err
                    else:
                        pred_val = round(run_inference_dict(item_dict), 2)
                        results.append(pred_val)
                        item_copy["predicted_charge"] = pred_val
                        item_copy["error"] = None
                    annotated.append(item_copy)

                latency = (time.time() - start_time) * 1000
                valid_preds = [r for r in results if isinstance(r, float)]
                log_prediction_event("/predict_api", {"batch": len(payload)},
                                     valid_preds[0] if valid_preds else 0, latency, 200)
                return jsonify({
                    "status": "success",
                    "prediction": valid_preds[0] if valid_preds else 0,
                    "predictions": results,
                    "records": annotated,
                    "latency_ms": round(latency, 2),
                    "model": MODEL_NAME,
                    "batch_size": len(payload)
                }), 200
            return jsonify({"status": "error", "message": "Invalid JSON format. Expected object or array."}), 400
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            log_prediction_event("/predict_api", {}, None, latency, 500, error=str(e))
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/predict_csv", methods=["POST"])
    def predict_csv():
        start_time = time.time()
        try:
            csv_content = None
            if "file" in request.files:
                f = request.files["file"]
                if f.filename != "":
                    csv_content = f.read().decode("utf-8", errors="replace")
            elif request.data:
                csv_content = request.data.decode("utf-8", errors="replace")

            if not csv_content or not csv_content.strip():
                return jsonify({"status": "error", "message": "No CSV file or content received."}), 400

            reader = csv.DictReader(io.StringIO(csv_content.strip()))
            records = []
            predictions = []

            for row in reader:
                clean_row = {str(k).strip().lower(): str(v).strip() for k, v in row.items() if k is not None}
                validated_row, err = validate_inputs(clean_row)
                row_copy = dict(row)
                if err:
                    row_copy["predicted_charge"] = None
                    row_copy["error"] = err
                else:
                    pred = round(run_inference_dict(validated_row), 2)
                    row_copy["predicted_charge"] = pred
                    row_copy["error"] = None
                    predictions.append(pred)
                records.append(row_copy)

            latency = (time.time() - start_time) * 1000
            total_valid = len(predictions)
            summary = {
                "total_rows": len(records),
                "successful": total_valid,
                "errors": len(records) - total_valid,
                "avg_prediction": round(sum(predictions) / total_valid, 2) if total_valid > 0 else 0,
                "max_prediction": max(predictions) if total_valid > 0 else 0,
                "min_prediction": min(predictions) if total_valid > 0 else 0,
                "total_portfolio": round(sum(predictions), 2),
                "latency_ms": round(latency, 2)
            }

            log_prediction_event("/predict_csv", {"rows": len(records), "valid": total_valid},
                                 summary["avg_prediction"], latency, 200)

            return jsonify({
                "status": "success",
                "summary": summary,
                "records": records[:1000],
                "model": MODEL_NAME,
                "engine": model_type
            }), 200
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            log_prediction_event("/predict_csv", {}, None, latency, 500, error=str(e))
            return jsonify({"status": "error", "message": f"CSV processing error: {str(e)}"}), 500

    @app.route("/health", methods=["GET"])
    def health():
        uptime_seconds = (datetime.now(timezone.utc) - datetime.fromisoformat(prediction_metrics["started_at"])).total_seconds()
        return jsonify({
            "status": "healthy",
            "model": MODEL_NAME,
            "engine": model_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": "ModelPulse-MLOps",
            "version": "2.0.0",
            "uptime_seconds": round(uptime_seconds, 1),
            "total_predictions": prediction_metrics["total_predictions"]
        }), 200

    @app.route("/api/metrics", methods=["GET"])
    def api_metrics():
        """Prometheus-style metrics endpoint."""
        total = prediction_metrics["total_predictions"]
        avg_latency = (prediction_metrics["total_latency_ms"] / total) if total > 0 else 0
        return jsonify({
            "total_predictions": total,
            "avg_latency_ms": round(avg_latency, 2),
            "max_latency_ms": round(prediction_metrics["max_latency_ms"], 2),
            "min_latency_ms": round(prediction_metrics["min_latency_ms"], 2) if prediction_metrics["min_latency_ms"] != float('inf') else 0,
            "error_count": prediction_metrics["errors"],
            "error_rate": round(prediction_metrics["errors"] / total * 100, 2) if total > 0 else 0,
            "model": MODEL_NAME,
            "engine": model_type,
            "started_at": prediction_metrics["started_at"]
        }), 200

    @app.route("/api/history", methods=["GET"])
    def api_history():
        """Return recent prediction history."""
        limit = request.args.get("limit", 50, type=int)
        history = list(recent_predictions)[-limit:]
        return jsonify({
            "count": len(history),
            "predictions": history
        }), 200

# ──────────────────────────────────────────────────────────────────────
# Built-in Standard Library HTTP Server Fallback
# ──────────────────────────────────────────────────────────────────────
else:
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from urllib.parse import parse_qs, urlparse

    class ModelPulseHTTPHandler(BaseHTTPRequestHandler):
        def _render_ui(self, pred_val=None, latency=None, error=None, inputs=None):
            html_path = os.path.join("templates", "home.html")
            with open(html_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Clean Jinja syntax for pure HTML output
            content = content.replace("{{ url_for('static', filename='style.css') }}", "/static/style.css")
            content = content.replace("{{ url_for('predict') }}", "/predict")

            if pred_val:
                result_block = f'''
                <div class="result-card success" id="result-card">
                  <div class="result-header">
                    <span class="result-badge">PREDICTION COMPLETE</span>
                    <span class="result-latency">⚡ {latency}</span>
                  </div>
                  <div class="result-amount-group">
                    <span class="result-currency">$</span>
                    <span class="result-amount" id="result-amount">{pred_val.replace('$','')}</span>
                    <span class="result-period">/year</span>
                  </div>
                </div>'''
            elif error:
                result_block = f'<div class="result-card error"><div class="error-content"><p class="error-msg">{error}</p></div></div>'
            else:
                result_block = '''
                <div class="result-card idle">
                  <h3 class="idle-title">Ready for Inference</h3>
                  <p class="idle-desc">Select a preset profile or enter custom parameters.</p>
                </div>'''

            if "{% if pred_value %}" in content:
                start_marker = "{% if pred_value %}"
                end_marker = "{% endif %}"
                pre = content.split(start_marker)[0]
                post = content.split(end_marker)[-1]
                content = pre + result_block + post

            content = content.replace("{{ inputs.age if inputs else 28 }}", str(inputs.get("age", 28) if inputs else 28))
            content = content.replace("{{ inputs.bmi if inputs else 30.0 }}", str(inputs.get("bmi", 30.0) if inputs else 30.0))
            content = content.replace("{{ inputs.children if inputs else 0 }}", str(inputs.get("children", 0) if inputs else 0))

            return content.encode("utf-8")

        def _serve_static(self, path):
            static_dir = "static"
            file_path = os.path.join(static_dir, path)
            if os.path.exists(file_path):
                ext = os.path.splitext(path)[1]
                mime_types = {'.css': 'text/css', '.js': 'application/javascript', '.png': 'image/png', '.svg': 'image/svg+xml'}
                content_type = mime_types.get(ext, 'application/octet-stream')
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.end_headers()
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path in ["/", "/index.html"]:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(self._render_ui())
            elif parsed.path.startswith("/static/"):
                self._serve_static(parsed.path[8:])
            elif parsed.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                res = {
                    "status": "healthy",
                    "model": MODEL_NAME,
                    "engine": f"PurePython-{model_type}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "service": "ModelPulse-MLOps",
                    "version": "2.0.0"
                }
                self.wfile.write(json.dumps(res).encode("utf-8"))
            else:
                self.send_error(404)

        def do_POST(self):
            parsed = urlparse(self.path)
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            start_time = time.time()

            if parsed.path == "/predict":
                form_data = parse_qs(body)
                raw = {
                    "age": form_data.get("age", ["28"])[0],
                    "sex": form_data.get("sex", ["male"])[0],
                    "bmi": form_data.get("bmi", ["30.0"])[0],
                    "children": form_data.get("children", ["0"])[0],
                    "smoker": form_data.get("smoker", ["no"])[0],
                    "region": form_data.get("region", ["southwest"])[0],
                }
                inputs, err = validate_inputs(raw)
                if err:
                    self.send_response(400)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(self._render_ui(error=err))
                    return

                val = run_inference_dict(inputs)
                latency = (time.time() - start_time) * 1000
                log_prediction_event("/predict", inputs, round(val, 2), latency, 200)
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(self._render_ui(pred_val=f"${val:,.2f}", latency=f"{latency:.1f} ms", inputs=inputs))

            elif parsed.path == "/predict_api":
                try:
                    payload = json.loads(body)
                    inputs, err = validate_inputs(payload)
                    if err:
                        self.send_response(400)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"status": "error", "message": err}).encode("utf-8"))
                        return

                    val = run_inference_dict(inputs)
                    latency = (time.time() - start_time) * 1000
                    log_prediction_event("/predict_api", inputs, round(val, 2), latency, 200)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    res = {
                        "status": "success",
                        "prediction": round(val, 2),
                        "predictions": [round(val, 2)],
                        "latency_ms": round(latency, 2),
                        "model": MODEL_NAME
                    }
                    self.wfile.write(json.dumps(res).encode("utf-8"))
            elif parsed.path == "/predict_csv":
                try:
                    reader = csv.DictReader(io.StringIO(body))
                    records = []
                    predictions = []
                    for row in reader:
                        clean_row = {str(k).strip().lower(): str(v).strip() for k, v in row.items() if k is not None}
                        validated_row, err = validate_inputs(clean_row)
                        row_copy = dict(row)
                        if err:
                            row_copy["predicted_charge"] = None
                            row_copy["error"] = err
                        else:
                            pred = round(run_inference_dict(validated_row), 2)
                            row_copy["predicted_charge"] = pred
                            row_copy["error"] = None
                            predictions.append(pred)
                        records.append(row_copy)

                    latency = (time.time() - start_time) * 1000
                    total_valid = len(predictions)
                    summary = {
                        "total_rows": len(records),
                        "successful": total_valid,
                        "errors": len(records) - total_valid,
                        "avg_prediction": round(sum(predictions) / total_valid, 2) if total_valid > 0 else 0,
                        "max_prediction": max(predictions) if total_valid > 0 else 0,
                        "min_prediction": min(predictions) if total_valid > 0 else 0,
                        "total_portfolio": round(sum(predictions), 2),
                        "latency_ms": round(latency, 2)
                    }
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "success", "summary": summary, "records": records[:1000]}).encode("utf-8"))
                except Exception as e:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))
            else:
                self.send_error(404)

        def log_message(self, format, *args):
            return  # Suppress default noisy console logs

if __name__ == "__main__":
    if has_flask:
        print("=" * 65)
        print("ModelPulse v2.0: Starting Flask Web Server on http://0.0.0.0:5000")
        print("=" * 65)
        app.run(host="0.0.0.0", port=5000, debug=False)
    else:
        print("=" * 65)
        print("ModelPulse v2.0: Starting Standard-Library Web Server on http://localhost:5000")
        print("(Running zero-dependency mode while external packages install)")
        print("=" * 65)
        server = HTTPServer(("0.0.0.0", 5000), ModelPulseHTTPHandler)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.server_close()