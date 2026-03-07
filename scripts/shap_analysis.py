"""
SHAP global feature importance analysis.

v2 fixes:
  - Sample size increased to 10,000 (was 2,000) for more stable importance estimates.
    At 2,000 samples only ~67 fraud cases were present; 10,000 gives ~350 fraud cases.
  - Added per-feature variance (std of |SHAP|) to flag unstable estimates.
  - Results saved to results_shap.json with top 30 global, 15 fraud, 15 legit.

Usage:
    python scripts/shap_analysis.py --model-dir models/v3/ --data-path data/test_features.parquet
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
import shap

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_shap_analysis(
    model_dir: str,
    data_path: str,
    n_samples: int = 10000,
    output: str = "results_shap.json",
) -> None:
    print(f"\n[SHAP] Loading model from {model_dir} ...")
    from app.core.ensemble import FraudEnsemble
    from app.core.features import get_feature_columns

    model = FraudEnsemble.load(model_dir)

    print(f"[SHAP] Loading test data from {data_path} ...")
    test_df = pd.read_parquet(data_path)
    feat_cols = get_feature_columns(test_df)

    # Sample — stratified to ensure adequate fraud cases
    fraud_df = test_df[test_df["isFraud"] == 1]
    legit_df = test_df[test_df["isFraud"] == 0]
    n_fraud = min(len(fraud_df), int(n_samples * test_df["isFraud"].mean() * 3))
    n_legit = min(len(legit_df), n_samples - n_fraud)

    sample_df = pd.concat([
        fraud_df.sample(n=n_fraud, random_state=42),
        legit_df.sample(n=n_legit, random_state=42),
    ]).sample(frac=1, random_state=42)  # shuffle

    X = sample_df[feat_cols].fillna(0).values
    y = sample_df["isFraud"].values
    print(f"[SHAP] Computing on {len(sample_df):,} samples ({y.sum():,} fraud, {y.mean():.2%} rate) ...")

    # Use model's internal XGB + scaler (bypass calibrator for SHAP)
    X_scaled = model.scaler.transform(X)
    if model.use_ae and model.ae_trainer is not None:
        X_aug = model._augment(X_scaled)
    else:
        X_aug = X_scaled

    feat_names = model.feature_names if model.feature_names else feat_cols

    t0 = time.time()
    explainer = shap.TreeExplainer(model.xgb_model)
    shap_values = explainer.shap_values(X_aug)
    print(f"[SHAP] Done in {time.time() - t0:.1f}s")

    # Trim feature names to SHAP output width
    n_feats = shap_values.shape[1]
    feat_names_trimmed = (feat_names + [f"f{i}" for i in range(n_feats)])[:n_feats]

    # Global importance
    mean_abs = np.abs(shap_values).mean(axis=0)
    std_abs = np.abs(shap_values).std(axis=0)
    importance_df = pd.DataFrame({
        "feature": feat_names_trimmed,
        "shap_mean": mean_abs,
        "shap_std": std_abs,
    }).sort_values("shap_mean", ascending=False).reset_index(drop=True)

    # Fraud / legit split
    fraud_shap = np.abs(shap_values[y == 1]).mean(axis=0)
    legit_shap = np.abs(shap_values[y == 0]).mean(axis=0)
    fraud_imp = pd.DataFrame({"feature": feat_names_trimmed, "shap": fraud_shap}).sort_values("shap", ascending=False)
    legit_imp = pd.DataFrame({"feature": feat_names_trimmed, "shap": legit_shap}).sort_values("shap", ascending=False)

    # Print report
    bar_max = importance_df["shap_mean"].max()
    print(f"\n{'='*68}")
    print(f"  Top 20 Features by Mean |SHAP|  (n={len(sample_df):,}, fraud={y.sum():,})")
    print(f"{'='*68}")
    for _, row in importance_df.head(20).iterrows():
        bar = "█" * int(30 * row["shap_mean"] / bar_max)
        stable = "" if row["shap_std"] < row["shap_mean"] * 0.5 else " ⚠ unstable"
        print(f"  {row['feature']:25s}  {row['shap_mean']:.4f}  {bar}{stable}")

    print(f"\n  Top 10 — FRAUD transactions (n={y.sum():,}):")
    for _, r in fraud_imp.head(10).iterrows():
        print(f"    {r['feature']:25s}  {r['shap']:.4f}")

    print(f"\n  Top 10 — LEGIT transactions (n={(y==0).sum():,}):")
    for _, r in legit_imp.head(10).iterrows():
        print(f"    {r['feature']:25s}  {r['shap']:.4f}")

    # Save
    results = {
        "n_samples": len(sample_df),
        "n_fraud": int(y.sum()),
        "n_legit": int((y == 0).sum()),
        "global": importance_df.head(30).to_dict("records"),
        "fraud": fraud_imp.head(15).to_dict("records"),
        "legit": legit_imp.head(15).to_dict("records"),
    }
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Save] {output} written ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAP global feature importance analysis")
    parser.add_argument("--model-dir", default="models/v3/", help="Model directory")
    parser.add_argument("--data-path", default="data/test_features.parquet", help="Test parquet path")
    parser.add_argument("--n-samples", type=int, default=10000, help="Sample size (default 10,000)")
    parser.add_argument("--output", default="results_shap.json", help="Output JSON path")
    args = parser.parse_args()
    run_shap_analysis(args.model_dir, args.data_path, args.n_samples, args.output)
