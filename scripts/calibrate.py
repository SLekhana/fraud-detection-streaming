"""
Isotonic calibration for fraud detection ensemble.

Why isotonic calibration?
  XGBoost with scale_pos_weight produces poorly calibrated probabilities —
  scores cluster below 0.40 even for true fraud. Isotonic regression learns
  a monotonic mapping from raw XGBoost scores → true probabilities.

  This is a standard MLOps practice (Platt scaling / isotonic calibration)
  used in production fraud systems.

Usage:
    python scripts/calibrate.py \
        --model-dir models/v2/ \
        --data-path data/test_features.parquet \
        --output models/v2/

Steps:
    1. Load v2 model (uncalibrated)
    2. Split test set: 50% calibration, 50% evaluation (no overlap)
    3. Fit IsotonicRegression on calibration split
    4. Evaluate before/after on evaluation split
    5. Show threshold sweep comparison
    6. Save calibrator.pkl to model directory
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.core.ensemble import FraudEnsemble
from app.core.features import get_feature_columns


def evaluate_metrics(proba, y, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    p = precision_score(y, pred, zero_division=0)
    r = recall_score(y, pred, zero_division=0)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    auc_pr = average_precision_score(y, proba)
    auc_roc = roc_auc_score(y, proba)
    return {"precision": p, "recall": r, "f1": f1, "auc_pr": auc_pr, "auc_roc": auc_roc}


def threshold_sweep(proba, y, label=""):
    print(f"\n{'─'*60}")
    print(f"Threshold sweep — {label}")
    print(f"{'─'*60}")
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}  {'vs XGB+SMOTE':>14}")
    print(f"{'─'*60}")
    baseline = 0.2163
    for t in np.arange(0.30, 0.81, 0.05):
        pred = (proba >= t).astype(int)
        if pred.sum() == 0:
            print(f"{t:>10.2f}  (no predictions above threshold)")
            continue
        p = precision_score(y, pred, zero_division=0)
        r = recall_score(y, pred, zero_division=0)
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        impr = (p - baseline) / baseline * 100
        marker = " ← 15%+ ✅" if impr >= 15 else ""
        print(f"{t:>10.2f} {p:>10.4f} {r:>10.4f} {f1:>10.4f}  ({impr:+.1f}%){marker}")


def calibrate(model_dir: str, data_path: str, output_dir: str) -> None:
    # ── Load model ──────────────────────────────────────────────────────────
    print(f"[Calibrate] Loading model from {model_dir} ...")
    model = FraudEnsemble.load(model_dir)
    print(f"[Calibrate] Calibrator already present: {model.calibrator is not None}")

    # ── Load test features ───────────────────────────────────────────────────
    print(f"[Calibrate] Loading test data from {data_path} ...")
    test_df = pd.read_parquet(data_path)
    feat_cols = get_feature_columns(test_df)
    X = test_df[feat_cols].fillna(0).values
    y = test_df["isFraud"].values
    print(f"[Calibrate] Test set: {len(y):,} samples ({y.sum():,} fraud, {y.mean():.3%} rate)")

    # ── Split into calibration / evaluation sets ─────────────────────────────
    # IMPORTANT: calibration set must NOT overlap with evaluation set
    X_cal, X_eval, y_cal, y_eval = train_test_split(
        X, y, test_size=0.5, stratify=y, random_state=42
    )
    print(f"[Calibrate] Calibration: {len(y_cal):,} | Evaluation: {len(y_eval):,}")

    # ── Get raw (uncalibrated) probabilities ─────────────────────────────────
    # Temporarily disable calibrator to get raw scores
    saved_calibrator = model.calibrator
    model.calibrator = None

    print("\n[Calibrate] Getting raw probabilities ...")
    raw_cal = model.predict_proba(X_cal)
    raw_eval = model.predict_proba(X_eval)

    print(f"[Before Calibration] Prob distribution on eval set:")
    print(f"  min={raw_eval.min():.4f}  p10={np.percentile(raw_eval,10):.4f}  "
          f"p50={np.percentile(raw_eval,50):.4f}  p90={np.percentile(raw_eval,90):.4f}  "
          f"max={raw_eval.max():.4f}")
    print(f"  % scores > 0.40: {(raw_eval > 0.40).mean():.2%}")
    print(f"  % scores > 0.50: {(raw_eval > 0.50).mean():.2%}")

    # ── Fit isotonic calibration ─────────────────────────────────────────────
    print("\n[Calibrate] Fitting IsotonicRegression ...")
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_cal, y_cal)
    model.calibrator = calibrator

    # ── Calibrated probabilities ─────────────────────────────────────────────
    cal_eval = model.predict_proba(X_eval)

    print(f"\n[After Calibration] Prob distribution on eval set:")
    print(f"  min={cal_eval.min():.4f}  p10={np.percentile(cal_eval,10):.4f}  "
          f"p50={np.percentile(cal_eval,50):.4f}  p90={np.percentile(cal_eval,90):.4f}  "
          f"max={cal_eval.max():.4f}")
    print(f"  % scores > 0.40: {(cal_eval > 0.40).mean():.2%}")
    print(f"  % scores > 0.50: {(cal_eval > 0.50).mean():.2%}")

    # ── Before/After metrics at threshold=0.5 ────────────────────────────────
    before = evaluate_metrics(raw_eval, y_eval)
    after = evaluate_metrics(cal_eval, y_eval)

    print(f"\n{'='*60}")
    print(f"  Metric comparison at threshold=0.50")
    print(f"{'='*60}")
    print(f"  {'Metric':<12} {'Before':>10} {'After':>10} {'Delta':>10}")
    print(f"  {'─'*44}")
    for k in ["precision", "recall", "f1", "auc_pr", "auc_roc"]:
        delta = after[k] - before[k]
        marker = " ↑" if delta > 0.001 else (" ↓" if delta < -0.001 else "")
        print(f"  {k:<12} {before[k]:>10.4f} {after[k]:>10.4f} {delta:>+10.4f}{marker}")
    print(f"{'='*60}")

    # ── Threshold sweeps ─────────────────────────────────────────────────────
    threshold_sweep(raw_eval, y_eval, label="BEFORE calibration")
    threshold_sweep(cal_eval, y_eval, label="AFTER calibration")

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    model.save(output_dir)

    # Save calibration results
    results = {
        "before_calibration": {k: round(v, 4) for k, v in before.items()},
        "after_calibration": {k: round(v, 4) for k, v in after.items()},
        "prob_distribution_before": {
            "p10": round(float(np.percentile(raw_eval, 10)), 4),
            "p50": round(float(np.percentile(raw_eval, 50)), 4),
            "p90": round(float(np.percentile(raw_eval, 90)), 4),
            "pct_above_0.5": round(float((raw_eval > 0.50).mean()), 4),
        },
        "prob_distribution_after": {
            "p10": round(float(np.percentile(cal_eval, 10)), 4),
            "p50": round(float(np.percentile(cal_eval, 50)), 4),
            "p90": round(float(np.percentile(cal_eval, 90)), 4),
            "pct_above_0.5": round(float((cal_eval > 0.50).mean()), 4),
        },
        "calibration_set_size": int(len(y_cal)),
        "evaluation_set_size": int(len(y_eval)),
    }
    results_path = os.path.join(output_dir, "calibration_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Save] Calibrator saved to {output_dir}/calibrator.pkl")
    print(f"[Save] Results saved to {results_path}")
    print("\n[Done] Calibration complete ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit isotonic calibration on trained ensemble")
    parser.add_argument("--model-dir", default="models/v2/", help="Directory with trained model")
    parser.add_argument("--data-path", default="data/test_features.parquet", help="Test parquet")
    parser.add_argument("--output", default="models/v2/", help="Directory to save calibrator")
    args = parser.parse_args()
    calibrate(args.model_dir, args.data_path, args.output)
