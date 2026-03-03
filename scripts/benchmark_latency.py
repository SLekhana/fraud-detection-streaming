"""
Latency Benchmark: Fraud Detection API vs Batch Processing
============================================================
Measures p50/p95/p99 inference latency for:
  1. Real-time API scoring (single transaction)
  2. Batch processing (same transactions processed together)
  3. Direct model inference (no HTTP overhead)

Proves "sub-second latency with 40% reduction over batch" resume claim.

Usage:
    # Start API first: docker-compose up -d
    python scripts/benchmark_latency.py --n 500 --api-url http://localhost:8000

Output: benchmark_results.json + printed table
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Sample transactions ──────────────────────────────────────────────────────

SAMPLE_TRANSACTIONS = [
    {"TransactionAmt": 49.99, "ProductCD": "W", "card1": 9500, "card4": "visa",
     "addr1": 325.0, "addr2": 87.0, "TransactionDT": 86400},
    {"TransactionAmt": 2500.00, "ProductCD": "H", "card1": 4150, "card4": "mastercard",
     "addr1": 204.0, "addr2": 299.0, "TransactionDT": 172800},
    {"TransactionAmt": 15.50, "ProductCD": "C", "card1": 7723, "card4": "visa",
     "addr1": 100.0, "addr2": 100.0, "TransactionDT": 259200},
    {"TransactionAmt": 999.00, "ProductCD": "S", "card1": 3311, "card4": "discover",
     "addr1": 450.0, "addr2": 112.0, "TransactionDT": 345600},
    {"TransactionAmt": 75.25, "ProductCD": "R", "card1": 8842, "card4": "visa",
     "addr1": 375.0, "addr2": 375.0, "TransactionDT": 432000},
]


def percentile(data: list, p: float) -> float:
    return float(np.percentile(data, p))


# ─── Benchmark modes ──────────────────────────────────────────────────────────

def benchmark_realtime_api(api_url: str, n: int) -> dict:
    """
    Real-time: score each transaction individually via HTTP POST /score.
    Simulates production streaming inference.
    """
    print(f"\n[RT-API] Benchmarking real-time API ({n} requests) ...")
    url = f"{api_url}/score"
    latencies = []
    errors = 0

    for i in range(n):
        tx = SAMPLE_TRANSACTIONS[i % len(SAMPLE_TRANSACTIONS)]
        try:
            t0 = time.perf_counter()
            resp = requests.post(url, json=tx, timeout=10)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if resp.status_code == 200:
                latencies.append(elapsed_ms)
            else:
                errors += 1
        except Exception as e:
            errors += 1

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n} requests completed ...")

    if not latencies:
        return {"error": "All requests failed", "errors": errors}

    return {
        "mode": "real-time API (per-transaction HTTP)",
        "n_requests": n,
        "errors": errors,
        "p50_ms": round(percentile(latencies, 50), 2),
        "p75_ms": round(percentile(latencies, 75), 2),
        "p95_ms": round(percentile(latencies, 95), 2),
        "p99_ms": round(percentile(latencies, 99), 2),
        "mean_ms": round(statistics.mean(latencies), 2),
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
        "throughput_rps": round(n / (sum(latencies) / 1000), 1),
    }


def benchmark_batch_api(api_url: str, n_batches: int = 20, batch_size: int = 50) -> dict:
    """
    Batch: score transactions in batches via HTTP POST /score/batch.
    Simulates batch processing — latency measured as time-per-transaction.
    """
    print(f"\n[BATCH] Benchmarking batch API ({n_batches} batches × {batch_size} txns) ...")
    url = f"{api_url}/score/batch"
    per_tx_latencies = []
    errors = 0

    for i in range(n_batches):
        batch = [SAMPLE_TRANSACTIONS[j % len(SAMPLE_TRANSACTIONS)] for j in range(batch_size)]
        payload = {"transactions": batch, "include_explanations": False}
        try:
            t0 = time.perf_counter()
            resp = requests.post(url, json=payload, timeout=60)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if resp.status_code == 200:
                per_tx_ms = elapsed_ms / batch_size
                per_tx_latencies.extend([per_tx_ms] * batch_size)
            else:
                errors += 1
        except Exception as e:
            errors += 1

    if not per_tx_latencies:
        return {"error": "All batch requests failed", "errors": errors}

    return {
        "mode": f"batch API ({batch_size} txns/batch, amortized per-tx)",
        "n_batches": n_batches,
        "batch_size": batch_size,
        "total_transactions": n_batches * batch_size,
        "errors": errors,
        "p50_ms_per_tx": round(percentile(per_tx_latencies, 50), 2),
        "p75_ms_per_tx": round(percentile(per_tx_latencies, 75), 2),
        "p95_ms_per_tx": round(percentile(per_tx_latencies, 95), 2),
        "p99_ms_per_tx": round(percentile(per_tx_latencies, 99), 2),
        "mean_ms_per_tx": round(statistics.mean(per_tx_latencies), 2),
    }


def benchmark_direct_model(n: int) -> dict:
    """
    Direct model inference (no HTTP, no network overhead).
    Measures pure ML computation time.
    """
    print(f"\n[DIRECT] Benchmarking direct model inference ({n} predictions) ...")
    from app.core.ensemble import FraudEnsemble
    from app.core.features import build_features, get_feature_columns

    try:
        model = FraudEnsemble.load("models/")
    except Exception as e:
        return {"error": f"Could not load model: {e}"}

    latencies = []
    for i in range(n):
        tx = SAMPLE_TRANSACTIONS[i % len(SAMPLE_TRANSACTIONS)]
        try:
            t0 = time.perf_counter()
            df = pd.DataFrame([tx])
            df = build_features(df, fast=False)
            feat_cols = get_feature_columns(df)
            X = df[feat_cols].fillna(0).values
            _ = model.predict_proba(X)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed_ms)
        except Exception:
            pass

    if not latencies:
        return {"error": "All direct predictions failed"}

    return {
        "mode": "direct model inference (no HTTP)",
        "n_predictions": n,
        "p50_ms": round(percentile(latencies, 50), 2),
        "p75_ms": round(percentile(latencies, 75), 2),
        "p95_ms": round(percentile(latencies, 95), 2),
        "p99_ms": round(percentile(latencies, 99), 2),
        "mean_ms": round(statistics.mean(latencies), 2),
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_benchmark(api_url: str, n: int, output_path: str) -> None:
    print(f"{'='*60}")
    print("  Fraud Detection Latency Benchmark")
    print(f"  API: {api_url}  |  N={n}")
    print(f"{'='*60}")

    results = {}

    # Check API is up
    try:
        resp = requests.get(f"{api_url}/health", timeout=5)
        if resp.status_code != 200:
            print(f"[WARNING] API health check failed: {resp.status_code}")
    except Exception as e:
        print(f"[WARNING] API not reachable at {api_url}: {e}")
        print("  Run: docker-compose up -d (or start the API locally)")

    # 1. Real-time API
    rt = benchmark_realtime_api(api_url, n)
    results["realtime_api"] = rt

    # 2. Batch API
    batch = benchmark_batch_api(api_url, n_batches=20, batch_size=50)
    results["batch_api"] = batch

    # 3. Direct model
    direct = benchmark_direct_model(min(n, 200))
    results["direct_model"] = direct

    # ─── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"{'Mode':<40} {'p50':>8} {'p95':>8} {'p99':>8}  {'Mean':>8}")
    print(f"{'─'*70}")

    for key, r in results.items():
        if "error" in r:
            print(f"  {r.get('mode', key):<38}  ERROR: {r['error']}")
            continue
        mode = r.get("mode", key)[:39]
        # Use per-tx latency for batch
        p50 = r.get("p50_ms", r.get("p50_ms_per_tx", 0))
        p95 = r.get("p95_ms", r.get("p95_ms_per_tx", 0))
        p99 = r.get("p99_ms", r.get("p99_ms_per_tx", 0))
        mean = r.get("mean_ms", r.get("mean_ms_per_tx", 0))
        print(f"  {mode:<38}  {p50:>7.1f}ms {p95:>7.1f}ms {p99:>7.1f}ms {mean:>7.1f}ms")

    print(f"{'='*70}")

    # Compute reduction: real-time vs batch
    if "error" not in rt and "error" not in batch:
        rt_p50 = rt.get("p50_ms", 0)
        batch_p50 = batch.get("p50_ms_per_tx", 0)
        if batch_p50 > 0:
            reduction_pct = (batch_p50 - rt_p50) / batch_p50 * 100
            print(f"\n[Summary] Real-time p50 vs batch per-tx p50: {rt_p50:.1f}ms vs {batch_p50:.1f}ms")
            print(f"[Summary] Latency reduction (real-time vs batch): {reduction_pct:.1f}%")
            results["_summary"] = {
                "realtime_p50_ms": rt_p50,
                "batch_per_tx_p50_ms": batch_p50,
                "latency_reduction_pct": round(reduction_pct, 1),
                "sub_second": rt_p50 < 1000,
            }

    # Save
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] Results → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark fraud detection API latency")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=300, help="Number of real-time requests")
    parser.add_argument("--output", default="models/benchmark_results.json")
    args = parser.parse_args()

    run_benchmark(api_url=args.api_url, n=args.n, output_path=args.output)
