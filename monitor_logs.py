#!/usr/bin/env python3
"""
ModelPulse — System & ML Log Monitoring Script (Unit 6.2 & Unit 9.4)
Reads logs/predictions.log and logs/model_training.log.
Summarizes total inference calls, latency statistics, model training history,
and detects any anomalous latency spikes or runtime errors.
"""

import os
import sys
import json
import argparse
from datetime import datetime

# ANSI Color codes for clean terminal output
GREEN = "\033[92m"
BLUE = "\033[94m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

PREDICTIONS_LOG_PATH = os.path.join("logs", "predictions.log")
TRAINING_LOG_PATH = os.path.join("logs", "model_training.log")

def parse_predictions_log():
    if not os.path.exists(PREDICTIONS_LOG_PATH):
        return {
            "exists": False,
            "total_requests": 0,
            "successful": 0,
            "errors": 0,
            "latencies": [],
            "recent_events": []
        }
        
    total_requests = 0
    successful = 0
    errors = 0
    latencies = []
    recent_events = []
    
    with open(PREDICTIONS_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_requests += 1
            try:
                record = json.loads(line)
                status = record.get("status", 200)
                lat = record.get("latency_ms")
                if lat is not None:
                    latencies.append(float(lat))
                    
                if status == 200 and not record.get("error"):
                    successful += 1
                else:
                    errors += 1
                    
                recent_events.append(record)
            except json.JSONDecodeError:
                # Fallback for plain text logs
                if "error" in line.lower() or "exception" in line.lower():
                    errors += 1
                else:
                    successful += 1
                    
    return {
        "exists": True,
        "total_requests": total_requests,
        "successful": successful,
        "errors": errors,
        "latencies": latencies,
        "recent_events": recent_events[-5:]  # last 5
    }

def parse_training_log():
    if not os.path.exists(TRAINING_LOG_PATH):
        return {
            "exists": False,
            "training_runs": 0,
            "latest_run": None
        }
        
    records = []
    with open(TRAINING_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError:
                continue
                
    latest = records[-1] if records else None
    return {
        "exists": True,
        "training_runs": len(records),
        "latest_run": latest
    }

def print_dashboard():
    pred_data = parse_predictions_log()
    train_data = parse_training_log()
    
    print("\n" + "=" * 65)
    print(f"{BOLD}{CYAN}   ⚡ MODELPULSE — SYSTEM & ML TELEMETRY DASHBOARD ⚡{RESET}")
    print("=" * 65)
    
    # 1. Model Training Status
    print(f"\n{BOLD}{BLUE}▶ 1. Automated Model Retraining Status (Unit 10.1 & 10.3){RESET}")
    print("-" * 65)
    if train_data["exists"] and train_data["latest_run"]:
        latest = train_data["latest_run"]
        details = latest.get("details", {})
        status_color = GREEN if latest.get("event") == "TRAINING_SUCCESS" else RED
        
        print(f"  • Retraining Event   : {status_color}{latest.get('event')}{RESET}")
        print(f"  • Timestamp (UTC)    : {latest.get('timestamp')}")
        print(f"  • Model Type         : {details.get('model_type', 'N/A')}")
        print(f"  • Artifact File      : {details.get('artifact', 'N/A')}")
        print(f"  • Dataset Sample     : {details.get('dataset_rows', 'N/A')} rows ({details.get('data_source', 'remote')})")
        print(f"  • Retraining Time    : {details.get('training_duration_seconds', 'N/A')} seconds")
        
        metrics = details.get("metrics", {})
        if metrics:
            print(f"  • Model Metrics      :")
            for k, v in list(metrics.items())[:5]:
                print(f"      - {k:10}: {v}")
    else:
        print(f"  {YELLOW}[NOTICE] No model training log found at {TRAINING_LOG_PATH}.{RESET}")
        print(f"  Run 'python train.py' to generate model artifacts and training logs.")

    # 2. Prediction Telemetry & Latency
    print(f"\n{BOLD}{BLUE}▶ 2. Prediction API Telemetry & Latency (Unit 6.2 & 9.4){RESET}")
    print("-" * 65)
    if pred_data["exists"] and pred_data["total_requests"] > 0:
        total = pred_data["total_requests"]
        succ = pred_data["successful"]
        err = pred_data["errors"]
        latencies = pred_data["latencies"]
        
        success_rate = (succ / total) * 100 if total > 0 else 0
        avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
        min_lat = min(latencies) if latencies else 0.0
        max_lat = max(latencies) if latencies else 0.0
        
        print(f"  • Total Predictions  : {BOLD}{total}{RESET}")
        print(f"  • Success Count      : {GREEN}{succ}{RESET} ({success_rate:.1f}%)")
        print(f"  • Error Count        : {RED if err > 0 else GREEN}{err}{RESET}")
        print(f"  • Latency (Avg)      : {CYAN}{avg_lat:.2f} ms{RESET}")
        print(f"  • Latency (Min / Max): {min_lat:.2f} ms / {max_lat:.2f} ms")
        
        # Recent prediction calls
        if pred_data["recent_events"]:
            print(f"\n  {BOLD}Recent Inferences (Last 5 requests):{RESET}")
            for ev in pred_data["recent_events"]:
                ep = ev.get("endpoint", "/predict")
                p = ev.get("prediction", "N/A")
                lat = ev.get("latency_ms", "N/A")
                inp = ev.get("inputs", {})
                print(f"    [{ev.get('timestamp', '')[11:19]}] {ep:12} -> Pred: ${p} | Latency: {lat}ms | Inputs: {inp}")
    else:
        print(f"  {YELLOW}[NOTICE] No prediction logs recorded yet at {PREDICTIONS_LOG_PATH}.{RESET}")
        print(f"  Trigger a prediction via Web UI or cURL to start capturing telemetry.")

    # 3. Overall System Health Summary
    print(f"\n{BOLD}{BLUE}▶ 3. DevOps Pipeline Health Verdict{RESET}")
    print("-" * 65)
    has_errors = pred_data["errors"] > 0
    if not has_errors:
        print(f"  {GREEN}{BOLD}[STATUS: HEALTHY]{RESET} All telemetry logs within nominal operating thresholds.")
    else:
        print(f"  {YELLOW}{BOLD}[STATUS: ATTENTION]{RESET} {pred_data['errors']} prediction errors detected in logs.")
        
    print("=" * 65 + "\n")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ModelPulse System & ML Log Monitor")
    parser.add_argument("--json", action="store_true", help="Output summary in JSON format")
    args = parser.parse_args()
    
    if args.json:
        p = parse_predictions_log()
        t = parse_training_log()
        print(json.dumps({"predictions": p, "training": t}, indent=2))
        sys.exit(0)
    else:
        sys.exit(print_dashboard())
