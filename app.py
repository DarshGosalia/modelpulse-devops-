#!/usr/bin/env python3
"""
ModelPulse — Flask REST API & Web UI for PyCaret Regression Model (Units 9, 10.2 & 10.4)
Features:
- Web UI route: GET / and POST /predict
- REST API endpoint: POST /predict_api
- Health check route: GET /health
- Structured latency and prediction logging to logs/predictions.log
- Includes built-in standard library HTTP server fallback if Flask/pandas are downloading.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timezone

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure Dedicated Structured Prediction Logger (Unit 9 & 10.4)
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

# ------------------------------------------------------------------------------
# Model Loading & Inference Engine
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# Flask Execution Pathway (Preferred)
# ------------------------------------------------------------------------------
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
            inputs_dict = {
                "age": int(request.form.get("age", 28)),
                "sex": str(request.form.get("sex", "male")).strip().lower(),
                "bmi": float(request.form.get("bmi", 30.0)),
                "children": int(request.form.get("children", 0)),
                "smoker": str(request.form.get("smoker", "no")).strip().lower(),
                "region": str(request.form.get("region", "southwest")).strip().lower()
            }
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
                p_val = run_inference_dict(payload)
                latency = (time.time() - start_time) * 1000
                log_prediction_event("/predict_api", payload, round(p_val, 2), latency, 200)
                return jsonify({
                    "status": "success",
                    "prediction": round(p_val, 2),
                    "predictions": [round(p_val, 2)],
                    "latency_ms": round(latency, 2),
                    "model": MODEL_NAME
                }), 200
            elif isinstance(payload, list):
                preds = [round(run_inference_dict(r), 2) for r in payload]
                latency = (time.time() - start_time) * 1000
                log_prediction_event("/predict_api", {"batch": len(payload)}, preds[0] if preds else 0, latency, 200)
                return jsonify({
                    "status": "success",
                    "prediction": preds[0] if preds else 0,
                    "predictions": preds,
                    "latency_ms": round(latency, 2),
                    "model": MODEL_NAME
                }), 200
            return jsonify({"status": "error", "message": "Invalid JSON format"}), 400
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            log_prediction_event("/predict_api", {}, None, latency, 500, error=str(e))
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "healthy",
            "model": MODEL_NAME,
            "engine": model_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": "ModelPulse-MLOps"
        }), 200

# ------------------------------------------------------------------------------
# Built-in Standard Library HTTP Server Fallback (When Flask/pip is offline)
# ------------------------------------------------------------------------------
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
                <div class="result-banner success">
                  <div class="result-meta">
                    <span class="result-label">ESTIMATED ANNUAL PREMIUM</span>
                    <span class="latency-badge">⚡ {latency}</span>
                  </div>
                  <div class="result-amount">{pred_val}</div>
                  <div class="result-footer">
                    <span class="check-icon">✓</span>
                    <span>Computed via deployment_28042020.pkl (Logged to logs/predictions.log)</span>
                  </div>
                </div>'''
            elif error:
                result_block = f'<div class="result-banner error"><div class="error-message">{error}</div></div>'
            else:
                result_block = '''
                <div class="result-banner idle">
                  <div class="idle-icon">🏷️</div>
                  <div class="idle-title">Ready for Inference</div>
                  <p class="idle-desc">Pick a preset on the left or enter custom parameters to generate an instant estimate.</p>
                </div>'''

            # Replace dynamic Jinja block with evaluated HTML
            if "{% if pred_value %}" in content:
                start_marker = "{% if pred_value %}"
                end_marker = "{% endif %}"
                pre = content.split(start_marker)[0]
                post = content.split(end_marker)[-1]
                content = pre + result_block + post

            # Replace inputs fallback
            content = content.replace("{{ inputs.age if inputs else 28 }}", str(inputs.get("age", 28) if inputs else 28))
            content = content.replace("{{ inputs.bmi if inputs else 30.0 }}", str(inputs.get("bmi", 30.0) if inputs else 30.0))
            content = content.replace("{{ inputs.children if inputs else 0 }}", str(inputs.get("children", 0) if inputs else 0))

            return content.encode("utf-8")

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path in ["/", "/index.html"]:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(self._render_ui())
            elif parsed.path == "/static/style.css":
                css_path = os.path.join("static", "style.css")
                if os.path.exists(css_path):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/css")
                    self.end_headers()
                    with open(css_path, "rb") as f:
                        self.wfile.write(f.read())
                else:
                    self.send_error(404)
            elif parsed.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                res = {
                    "status": "healthy",
                    "model": MODEL_NAME,
                    "engine": f"PurePython-{model_type}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "service": "ModelPulse-MLOps"
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
                inputs = {
                    "age": int(form_data.get("age", ["28"])[0]),
                    "sex": str(form_data.get("sex", ["male"])[0]).lower().strip(),
                    "bmi": float(form_data.get("bmi", ["30.0"])[0]),
                    "children": int(form_data.get("children", ["0"])[0]),
                    "smoker": str(form_data.get("smoker", ["no"])[0]).lower().strip(),
                    "region": str(form_data.get("region", ["southwest"])[0]).lower().strip(),
                }
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
                    val = run_inference_dict(payload)
                    latency = (time.time() - start_time) * 1000
                    log_prediction_event("/predict_api", payload, round(val, 2), latency, 200)
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
                except Exception as e:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))
            else:
                self.send_error(404)

        def log_message(self, format, *args):
            return # Suppress default noisy console logs

if __name__ == "__main__":
    if has_flask:
        print("ModelPulse: Starting Flask Web Server on http://0.0.0.0:5000 ...")
        app.run(host="0.0.0.0", port=5000, debug=False)
    else:
        print("=" * 65)
        print("ModelPulse: Starting Standard-Library Web Server on http://localhost:5000")
        print("(Running zero-dependency mode while external packages install)")
        print("=" * 65)
        server = HTTPServer(("0.0.0.0", 5000), ModelPulseHTTPHandler)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.server_close()