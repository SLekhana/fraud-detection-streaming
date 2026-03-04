"""
Training script for the fraud detection ensemble.

v2 fixes applied:
  - card_stats computed on TRAIN only, passed to build_features() for test set (no leakage)
  - risk_profile computed on TRAIN only (was already correct, now explicit)
  - use_ae=False by default — pure cost-sensitive XGBoost based on ablation + SHAP findings
  - Extended feature set: V1-V20 (was V1-V6) + numeric identity columns id_01..id_11
  - Added --use-ae flag to optionally re-enable the AutoEncoder for comparison
  - SHAP sample size increased to 10,000 (was 2,000) for more stable importance estimates

Usage:
    python scripts/train.py --data-dir data/ --model-dir models/v3/
    python scripts/train.py --data-dir data/ --model-dir models/v3_ae/ --use-ae
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
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.ensemble import FraudEnsemble
from app.core.features import (
    build_features,
    get_feature_columns,
    build_merchant_risk_profile,
    compute_card_stats,
)


def load_data(data_dir: str) -> pd.DataFrame:
    """Load and merge IEEE-CIS transaction + identity data."""
    tx_path = os.path.join(data_dir, "train_transaction.csv")
    id_path = os.path.join(data_dir, "train_identity.csv")

    print(f"[Data] Loading transactions from {tx_path} ...")
    tx_df = pd.read_csv(tx_path)
    print(f"[Data] Loaded {len(tx_df):,} transactions")

    if os.path.exists(id_path):
        print(f"[Data] Loading identity from {id_path} ...")
        id_df = pd.read_csv(id_path)
        df = tx_df.merge(id_df, on="TransactionID", how="left")
        print(f"[Data] Merged: {len(df):,} rows")
        # Report identity join rate
        identity_rate = id_df["TransactionID"].isin(tx_df["TransactionID"]).mean()
        print(f"[Data] Identity join rate: {identity_rate:.1%}")
    else:
        print("[Data] No identity file found — proceeding without identity features")
        df = tx_df

    fraud_rate = df["isFraud"].mean()
    print(
        f"[Data] Fraud rate: {fraud_rate:.4f} "
        f"({df['isFraud'].sum():,} fraud / {len(df):,} total)"
    )
    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run feature engineering and return (train_df, test_df, risk_profile, card_stats).

    FIX: card_stats and risk_profile are computed on TRAINING data only,
    then passed into build_features() for the test set to prevent leakage.
    """
    # Train / test split FIRST (before computing any statistics)
    train_df_raw, test_df_raw = train_test_split(
        df, test_size=0.2, stratify=df["isFraud"], random_state=42
    )
    print(f"[Split] Train raw: {len(train_df_raw):,} | Test raw: {len(test_df_raw):,}")

    # Compute statistics on TRAINING data only
    print("[Features] Building merchant risk profile (train only) ...")
    risk_profile = build_merchant_risk_profile(train_df_raw)

    print("[Features] Computing card statistics (train only) ...")
    card_stats = compute_card_stats(train_df_raw)
    print(f"[Features] {len(card_stats):,} unique cards in training set")

    # Feature engineering — train
    print("[Features] Engineering train features ...")
    t0 = time.time()
    train_df = build_features(train_df_raw, risk_profile=risk_profile, card_stats=card_stats, fast=True)
    print(f"[Features] Train done in {time.time() - t0:.1f}s")

    # Feature engineering — test (uses train statistics, no leakage)
    print("[Features] Engineering test features (using train stats) ...")
    t0 = time.time()
    test_df = build_features(test_df_raw, risk_profile=risk_profile, card_stats=card_stats, fast=True)
    print(f"[Features] Test done in {time.time() - t0:.1f}s")

    feat_cols = get_feature_columns(train_df)
    print(f"[Features] {len(feat_cols)} feature columns selected")

    return train_df, test_df, risk_profile, card_stats


def train(
    data_dir: str,
    model_dir: str,
    epochs: int = 50,
    smote: bool = False,
    use_ae: bool = False,
    cross_validate: bool = False,
) -> None:
    t_start = time.time()

    # 1. Load data
    df = load_data(data_dir)

    # 2. Feature engineering (leak-free)
    train_df, test_df, risk_profile, card_stats = prepare_features(df)

    feat_cols = get_feature_columns(train_df)
    X_train = train_df[feat_cols].fillna(0).values
    y_train = train_df["isFraud"].values
    X_test = test_df[feat_cols].fillna(0).values
    y_test = test_df["isFraud"].values

    ae_label = "AE+XGB" if use_ae else "XGB (cost-sensitive)"
    print(f"\n[Config] Architecture: {ae_label}")
    print(f"[Config] SMOTE: {smote} | Features: {len(feat_cols)}")

    # 3. Optional cross-validation
    if cross_validate:
        print("\n[CV] Running 5-fold stratified cross-validation ...")
        cv_model = FraudEnsemble(
            input_dim=len(feat_cols),
            smote=smote,
            use_ae=use_ae,
        )
        cv_results = cv_model.cross_validate(X_train, y_train, n_splits=5)
        cv_auc_pr = [r["auc_pr"] for r in cv_results]
        print(f"[CV] AUC-PR: {np.mean(cv_auc_pr):.4f} ± {np.std(cv_auc_pr):.4f}")

    # 4. Train full model
    print(f"\n[Train] Training ensemble (input_dim={len(feat_cols)}, use_ae={use_ae}) ...")
    model = FraudEnsemble(
        input_dim=len(feat_cols),
        ae_epochs=epochs,
        smote=smote,
        use_ae=use_ae,
    )
    model.fit(X_train, y_train, feature_names=feat_cols, verbose=True)

    # 5. Evaluate
    print("\n[Eval] Evaluating on held-out test set ...")
    metrics = model.evaluate(X_test, y_test)

    print(f"\n{'='*50}")
    print(f"  AUC-PR  : {metrics['auc_pr']:.4f}")
    print(f"  AUC-ROC : {metrics['auc_roc']:.4f}")
    fraud_rep = metrics["classification_report"].get("1", {})
    print(f"  Precision: {fraud_rep.get('precision', 0):.4f}")
    print(f"  Recall   : {fraud_rep.get('recall', 0):.4f}")
    print(f"  F1       : {fraud_rep.get('f1-score', 0):.4f}")
    print(f"  Features : {len(feat_cols)}")
    print(f"  use_ae   : {use_ae}")
    print(f"{'='*50}")

    # 6. Save model + metadata
    os.makedirs(model_dir, exist_ok=True)
    model.save(model_dir)

    # Save risk profile and card stats alongside model for inference
    risk_profile.to_parquet(os.path.join(model_dir, "risk_profile.parquet"), index=False)
    card_stats.to_parquet(os.path.join(model_dir, "card_stats.parquet"), index=False)
    print(f"\n[Save] Model + risk_profile + card_stats saved to {model_dir}/")

    # Save metrics
    with open(os.path.join(model_dir, "eval_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # 7. Save test features for /evaluate endpoint
    test_df_out = test_df.copy()
    test_df_out["isFraud"] = y_test
    test_df_out[feat_cols + ["isFraud"]].to_parquet(
        os.path.join(data_dir, "test_features.parquet"), index=False
    )
    print(f"[Save] Test features saved to {data_dir}/test_features.parquet")

    print(f"\n[Done] Total training time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fraud detection ensemble")
    parser.add_argument("--data-dir", default="data/", help="Directory with IEEE-CIS CSVs")
    parser.add_argument("--model-dir", default="models/v3/", help="Directory to save model")
    parser.add_argument("--epochs", type=int, default=50, help="AutoEncoder epochs (ignored if --use-ae not set)")
    parser.add_argument("--smote", action="store_true", help="Enable SMOTE (default: off, use cost-sensitive instead)")
    parser.add_argument("--use-ae", action="store_true", help="Enable AutoEncoder stage (default: off based on SHAP findings)")
    parser.add_argument("--cross-validate", action="store_true", help="Run 5-fold CV first")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        epochs=args.epochs,
        smote=args.smote,
        use_ae=args.use_ae,
        cross_validate=args.cross_validate,
    )
