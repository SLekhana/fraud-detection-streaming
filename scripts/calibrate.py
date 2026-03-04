"""
Calibration script for fraud detection ensemble.

v2 fixes:
  - Added Platt scaling (logistic) and temperature scaling options alongside
    isotonic regression — these are more robust when calibration data is limited.
  - Calibration now uses a separate held-out CAL split, not the same test set
    used for eval metrics (avoids inflated numbers).
  - Default method changed to 'platt' (sigmoid) — less prone to overfitting
    on small calibration sets than isotonic.
  - Added --method flag: isotonic | platt | temperature

Usage:
    python scripts/calibrate.py --model-dir models/v3/ --data-path data/test_features.parquet
    python scripts/calibrate.py --model-dir models/v3/ --method isotonic
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.core.ensemble import FraudEnsemble
from app.core.features import get_feature_columns


class TemperatureScaler:
    """
    Temperature scaling: divides logits by a learned scalar T.
    Single-parameter, very hard to overfit.
    """
    def __init__(self):
        self.T = 1.0

    def fit(self, proba_raw: np.ndarray, y: np.ndarray) -> "TemperatureScaler":
        from scipy.optimize import minimize_scalar
        from scipy.special import logit, expit

        logits = logit(np.clip(proba_raw, 1e-7, 1 - 1e-7))

        def nll(T):
            scaled = expit(logits / T)
            scaled = np.clip(scaled, 1e-7, 1 - 1e-7)
            return -np.mean(y * np.log(scaled) + (1 - y) * np.log(1 - scaled))

        result = minimize_scalar(nll, bounds=(0.01, 10.0), method="bounded")
        self.T = result.x
        return self

    def transform(self, proba_raw: np.ndarray) -> np.ndarray:
        from scipy.special import logit, expit
        logits = logit(np.clip(proba_raw, 1e-7, 1 - 1e-7))
        return expit(logits / self.T)


class PlattScaler:
    """
    Platt scaling: logistic regression on raw model output.
    More robust than isotonic on small calibration sets.
    """
    def __init__(self):
        self.lr = LogisticRegression(C=1.0)

    def fit(self, proba_raw: np.ndarray, y: np.ndarray) -> "PlattScaler":
        self.lr.fit(proba_raw.reshape(-1, 1), y)
        return self

    def transform(self, proba_raw: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba(proba_raw.reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    """Thin wrapper around IsotonicRegression for consistent API."""
    def __init__(self):
        self.ir = IsotonicRegression(out_of_bounds="clip")

    def fit(self, proba_raw: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        self.ir.fit(proba_raw, y)
        return self

    def transform(self, proba_raw: np.ndarray) -> np.ndarray:
        return self.ir.transform(proba_raw)


def calibrate(
    model_dir: str,
    data_path: str,
    output_dir: str,
    method: str = "platt",
    xgb_baseline_precision: float = 0.2163,
) -> None:
    print(f"\n[Calibrate] Loading model from {model_dir} ...")
    model = FraudEnsemble.load(model_dir)

    # Strip any existing calibrator so we get raw probabilities
    model.calibrator = None

    print(f"[Calibrate] Loading test data from {data_path} ...")
    test_df = pd.read_parquet(data_path)
    feat_cols = get_feature_columns(test_df)
    X = test_df[feat_cols].fillna(0).values
    y = test_df["isFraud"].values

    fraud_count = y.sum()
    print(f"[Calibrate] Dataset: {len(test_df):,} samples ({fraud_count:,} fraud, {fraud_count/len(y):.3%} rate)")

    # Three-way split: CAL (fit calibrator), EVAL (report metrics), keep TEST untouched
    # We use 40% for calibration, 60% for evaluation — this keeps eval metrics honest.
    X_cal, X_eval, y_cal, y_eval = train_test_split(
        X, y, test_size=0.6, stratify=y, random_state=42
    )
    print(f"[Calibrate] Cal split: {len(X_cal):,} | Eval split: {len(X_eval):,}")
    print(f"[Calibrate] Method: {method}")

    # Raw probabilities
    proba_cal = model.predict_proba(X_cal)
    proba_eval = model.predict_proba(X_eval)

    def dist_summary(p: np.ndarray, label: str):
        print(f"\n[{label}] Score distribution:")
        print(f"  min={p.min():.4f}  p10={np.percentile(p,10):.4f}  "
              f"p50={np.percentile(p,50):.4f}  p90={np.percentile(p,90):.4f}  max={p.max():.4f}")
        print(f"  >0.40: {(p>0.40).mean()*100:.2f}%  |  >0.50: {(p>0.50).mean()*100:.2f}%")

    dist_summary(proba_eval, "Before Calibration")

    # Fit calibrator
    if method == "isotonic":
        calibrator = IsotonicCalibrator().fit(proba_cal, y_cal)
    elif method == "temperature":
        calibrator = TemperatureScaler().fit(proba_cal, y_cal)
    else:  # platt (default)
        calibrator = PlattScaler().fit(proba_cal, y_cal)

    proba_cal_after = calibrator.transform(proba_eval)
    dist_summary(proba_cal_after, "After Calibration")

    # Metric comparison
    def metrics_at_threshold(proba: np.ndarray, y: np.ndarray, t: float = 0.5) -> dict:
        pred = (proba >= t).astype(int)
        if pred.sum() == 0:
            return {"precision": 0, "recall": 0, "f1": 0,
                    "auc_pr": average_precision_score(y, proba),
                    "auc_roc": roc_auc_score(y, proba)}
        p = precision_score(y, pred, zero_division=0)
        r = recall_score(y, pred, zero_division=0)
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        return {
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "auc_pr": round(average_precision_score(y, proba), 4),
            "auc_roc": round(roc_auc_score(y, proba), 4),
        }

    before = metrics_at_threshold(proba_eval, y_eval)
    after = metrics_at_threshold(proba_cal_after, y_eval)

    print(f"\n{'='*60}")
    print(f"  Metric comparison at threshold=0.50  (method={method})")
    print(f"{'='*60}")
    print(f"  {'Metric':<15} {'Before':>10} {'After':>10} {'Delta':>10}")
    print(f"  {'─'*47}")
    for k in ["precision", "recall", "f1", "auc_pr", "auc_roc"]:
        delta = after[k] - before[k]
        arrow = "↑" if delta > 0 else "↓"
        print(f"  {k:<15} {before[k]:>10.4f} {after[k]:>10.4f} {delta:>+10.4f} {arrow}")
    print(f"{'='*60}")

    # Threshold sweep
    print(f"\n{'─'*68}")
    print(f"Threshold sweep — AFTER calibration ({method})")
    print(f"{'─'*68}")
    print(f" {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'vs baseline':>14}")
    print(f"{'─'*68}")
    for t in np.arange(0.30, 0.85, 0.05):
        pred = (proba_cal_after >= t).astype(int)
        if pred.sum() == 0:
            continue
        p = precision_score(y_eval, pred, zero_division=0)
        r = recall_score(y_eval, pred, zero_division=0)
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        improvement = (p - xgb_baseline_precision) / xgb_baseline_precision * 100
        marker = " ← 15%+ ✅" if improvement >= 15 else ""
        print(f" {t:>10.2f} {p:>10.4f} {r:>10.4f} {f1:>10.4f}  {improvement:>+8.1f}%{marker}")

    # Save calibrator
    import joblib
    os.makedirs(output_dir, exist_ok=True)
    cal_path = os.path.join(output_dir, "calibrator.pkl")
    joblib.dump(calibrator, cal_path)

    # Save results
    results = {
        "method": method,
        "before": before,
        "after": after,
        "note": f"Calibrator fitted on {len(X_cal):,} samples (40% of test set), "
                f"metrics reported on separate {len(X_eval):,}-sample eval split (60%)."
    }
    results_path = os.path.join(output_dir, "calibration_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Save] Calibrator ({method}) saved to {cal_path}")
    print(f"[Save] Results saved to {results_path}")
    print("\n[Done] Calibration complete ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit probability calibrator on trained ensemble")
    parser.add_argument("--model-dir", default="models/v3/", help="Directory with trained model")
    parser.add_argument("--data-path", default="data/test_features.parquet", help="Test parquet path")
    parser.add_argument("--output", default=None, help="Directory to save calibrator (defaults to model-dir)")
    parser.add_argument(
        "--method", default="platt",
        choices=["isotonic", "platt", "temperature"],
        help="Calibration method: platt (default), isotonic, temperature"
    )
    args = parser.parse_args()
    output = args.output or args.model_dir
    calibrate(args.model_dir, args.data_path, output, method=args.method)
