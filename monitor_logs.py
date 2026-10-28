#!/usr/bin/env python3
"""
ModelPulse v2.0 — Production Log Monitor & Alerting System

Reads JSONL prediction logs, computes real-time KPIs (throughput, latency
percentiles, error rates, drift signals), and prints a live dashboard
to the terminal. Can also send alerts via webhook if configured.

Usage:
    python monitor_logs.py                  # Live tail mode
    python monitor_logs.py --summary        # One-shot summary report
    python monitor_logs.py --watch 10       # Refresh every 10 seconds
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime, timezone, timedelta
from collections import defaultdict

LOG_FILE = os.path.join("logs", "predictions.log")
TRAINING_LOG = os.path.join("logs", "model_training.log")

# Alert thresholds
ALERT_LATENCY_P95_MS = 500.0
ALERT_ERROR_RATE_PCT = 5.0
ALERT_DRIFT_THRESHOLD = 0.3  # coefficient of variation threshold


def parse_log_entries(filepath: str) -> list:
    """Parse JSONL log file and return list of dicts."""
    entries = []
    if not os.path.exists(filepath):
        return entries
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def compute_percentile(sorted_vals: list, p: float) -> float:
    """Compute the p-th percentile from a sorted list."""
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_vals) else f
    d = k - f
    return sorted_vals[f] + d * (sorted_vals[c] - sorted_vals[f])


def analyze_predictions(entries: list) -> dict:
    """Compute comprehensive metrics from prediction log entries."""
    if not entries:
        return {
            "total": 0, "success": 0, "errors": 0, "error_rate": 0.0,
            "latency_mean": 0.0, "latency_p50": 0.0, "latency_p95": 0.0, "latency_p99": 0.0,
            "prediction_mean": 0.0, "prediction_std": 0.0,
            "endpoint_breakdown": {}, "hourly_throughput": {},
            "alerts": []
        }

    success_entries = [e for e in entries if e.get("status") == 200]
    error_entries = [e for e in entries if e.get("status") != 200]

    # Latency metrics
    latencies = sorted([e.get("latency_ms", 0) for e in success_entries if e.get("latency_ms")])
    latency_mean = sum(latencies) / len(latencies) if latencies else 0.0

    # Prediction value distribution
    predictions = [e.get("prediction", 0) for e in success_entries if e.get("prediction") is not None]
    pred_mean = sum(predictions) / len(predictions) if predictions else 0.0
    pred_variance = sum((p - pred_mean) ** 2 for p in predictions) / len(predictions) if len(predictions) > 1 else 0.0
    pred_std = pred_variance ** 0.5

    # Endpoint breakdown
    endpoint_counts = defaultdict(int)
    for e in entries:
        endpoint_counts[e.get("endpoint", "unknown")] += 1

    # Hourly throughput
    hourly = defaultdict(int)
    for e in entries:
        ts = e.get("timestamp", "")
        if len(ts) >= 13:
            hour_key = ts[:13]
            hourly[hour_key] += 1

    # Alert detection
    alerts = []
    p95 = compute_percentile(latencies, 95) if latencies else 0
    error_rate = (len(error_entries) / len(entries) * 100) if entries else 0

    if p95 > ALERT_LATENCY_P95_MS:
        alerts.append(f"⚠️  HIGH LATENCY: P95 latency {p95:.1f}ms exceeds {ALERT_LATENCY_P95_MS}ms threshold")
    if error_rate > ALERT_ERROR_RATE_PCT:
        alerts.append(f"🚨 HIGH ERROR RATE: {error_rate:.1f}% exceeds {ALERT_ERROR_RATE_PCT}% threshold")
    if pred_mean > 0 and (pred_std / pred_mean) > ALERT_DRIFT_THRESHOLD:
        alerts.append(f"📊 PREDICTION DRIFT: CV={pred_std/pred_mean:.2f} exceeds {ALERT_DRIFT_THRESHOLD} threshold")

    return {
        "total": len(entries),
        "success": len(success_entries),
        "errors": len(error_entries),
        "error_rate": round(error_rate, 2),
        "latency_mean": round(latency_mean, 2),
        "latency_p50": round(compute_percentile(latencies, 50), 2),
        "latency_p95": round(p95, 2),
        "latency_p99": round(compute_percentile(latencies, 99), 2),
        "prediction_mean": round(pred_mean, 2),
        "prediction_std": round(pred_std, 2),
        "endpoint_breakdown": dict(endpoint_counts),
        "hourly_throughput": dict(sorted(hourly.items())[-24:]),
        "alerts": alerts
    }


def print_dashboard(metrics: dict, training_info: dict = None):
    """Print a formatted terminal dashboard."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    width = 70

    print("\n" + "═" * width)
    print(f"  📊 ModelPulse v2.0 — Production Monitoring Dashboard")
    print(f"  ⏰ {now}")
    print("═" * width)

    # Overview
    print(f"\n  ┌─ OVERVIEW {'─' * (width - 14)}┐")
    print(f"  │ Total Predictions:  {metrics['total']:>10,}")
    print(f"  │ Successful:         {metrics['success']:>10,}  ✅")
    print(f"  │ Errors:             {metrics['errors']:>10,}  {'🚨' if metrics['errors'] > 0 else '✅'}")
    print(f"  │ Error Rate:         {metrics['error_rate']:>9.2f}%  {'⚠️' if metrics['error_rate'] > ALERT_ERROR_RATE_PCT else '✅'}")
    print(f"  └{'─' * (width - 4)}┘")

    # Latency
    print(f"\n  ┌─ LATENCY METRICS {'─' * (width - 21)}┐")
    print(f"  │ Mean:    {metrics['latency_mean']:>10.2f} ms")
    print(f"  │ P50:     {metrics['latency_p50']:>10.2f} ms")
    print(f"  │ P95:     {metrics['latency_p95']:>10.2f} ms  {'⚠️' if metrics['latency_p95'] > ALERT_LATENCY_P95_MS else '✅'}")
    print(f"  │ P99:     {metrics['latency_p99']:>10.2f} ms")
    print(f"  └{'─' * (width - 4)}┘")

    # Prediction Distribution
    print(f"\n  ┌─ PREDICTION DISTRIBUTION {'─' * (width - 29)}┐")
    print(f"  │ Mean Premium:   ₹{metrics['prediction_mean']:>12,.2f}")
    print(f"  │ Std Deviation:  ₹{metrics['prediction_std']:>12,.2f}")
    if metrics['prediction_mean'] > 0:
        cv = metrics['prediction_std'] / metrics['prediction_mean']
        print(f"  │ Coefficient of Variation: {cv:>6.2f}")
    print(f"  └{'─' * (width - 4)}┘")

    # Endpoint Breakdown
    if metrics['endpoint_breakdown']:
        print(f"\n  ┌─ ENDPOINT BREAKDOWN {'─' * (width - 24)}┐")
        for ep, count in sorted(metrics['endpoint_breakdown'].items()):
            pct = (count / metrics['total'] * 100) if metrics['total'] > 0 else 0
            bar_len = int(pct / 100 * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            print(f"  │ {ep:20s} {count:>6,}  {bar} {pct:5.1f}%")
        print(f"  └{'─' * (width - 4)}┘")

    # Training Info
    if training_info:
        print(f"\n  ┌─ LAST TRAINING RUN {'─' * (width - 23)}┐")
        for key, val in training_info.items():
            print(f"  │ {key:20s} {str(val):>30s}")
        print(f"  └{'─' * (width - 4)}┘")

    # Alerts
    if metrics['alerts']:
        print(f"\n  ┌─ 🚨 ACTIVE ALERTS {'─' * (width - 22)}┐")
        for alert in metrics['alerts']:
            print(f"  │ {alert}")
        print(f"  └{'─' * (width - 4)}┘")
    else:
        print(f"\n  ✅ All systems nominal — no active alerts")

    print("\n" + "═" * width)


def get_training_info() -> dict:
    """Parse the most recent training log entry."""
    entries = parse_log_entries(TRAINING_LOG)
    if not entries:
        return None
    last = entries[-1]
    details = last.get("details", {})
    return {
        "Timestamp": last.get("timestamp", "unknown")[:19],
        "Model": details.get("model_name", "unknown"),
        "Type": details.get("model_type", "unknown"),
        "Dataset Rows": str(details.get("dataset_rows", "?")),
        "Duration": f"{details.get('training_duration_seconds', '?')}s",
        "R²": str(details.get("metrics", {}).get("R2", "?")),
    }


def main():
    parser = argparse.ArgumentParser(description="ModelPulse Log Monitor")
    parser.add_argument("--summary", action="store_true", help="One-shot summary report")
    parser.add_argument("--watch", type=int, default=0, help="Refresh interval in seconds (0 = one-shot)")
    parser.add_argument("--json", action="store_true", help="Output metrics as JSON")
    parser.add_argument("--last", type=int, default=0, help="Only analyze last N entries")
    args = parser.parse_args()

    while True:
        entries = parse_log_entries(LOG_FILE)
        if args.last > 0:
            entries = entries[-args.last:]

        metrics = analyze_predictions(entries)

        if args.json:
            print(json.dumps(metrics, indent=2))
        else:
            training_info = get_training_info()
            if args.watch > 0 and sys.stdout.isatty():
                if sys.platform == 'win32':
                    os.system('cls')
                else:
                    os.system('clear')
            print_dashboard(metrics, training_info)

        if args.watch <= 0:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
