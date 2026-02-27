"""
Training script for the fraud detection ensemble.

Usage:
    python scripts/train.py --data-dir data/ --model-dir models/ --epochs 50

Steps:
    1. Load IEEE-CIS train_transaction.csv + train_identity.csv
    2. Merge and run feature engineering pipeline
    3. Train AutoEncoder on legit transactions
    4. Train XGBoost stacked on AE features
    5. Evaluate on held-out test split
    6. Save model artefacts to models/
    7. Save test features to data/test_features.parquet
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
from app.core.features import build_features, get_feature_columns, build_merchant_risk_profile


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
    else:
        df = tx_df

    fraud_rate = df["isFraud"].mean()
    print(
        f"[Data] Fraud rate: {fraud_rate:.4f} "
        f"({df['isFraud'].sum():,} fraud / {len(df):,} total)"
    )
    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run feature engineering and return (train_df, test_df)."""
    print("[Features] Building merchant risk profile ...")
    risk_profile = build_merchant_risk_profile(df)

    print("[Features] Engineering features (vectorised) ...")
    t0 = time.time()
    df = build_features(df, risk_profile=risk_profile, fast=True)
    print(f"[Features] Done in {time.time() - t0:.1f}s")

    feat_cols = get_feature_columns(df)
    print(f"[Features] {len(feat_cols)} feature columns selected")

    # Train / test split (stratified)
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["isFraud"], random_state=42
    )
    print(f"[Split] Train: {len(train_df):,} | Test: {len(test_df):,}")
    return train_df, test_df


def train(
    data_dir: str,
    model_dir: str,
    epochs: int = 50,
    smote: bool = True,
    cross_validate: bool = False,
) -> None:
    t_start = time.time()

    # 1. Load data
    df = load_data(data_dir)

    # 2. Feature engineering
    train_df, test_df = prepare_features(df)

    feat_cols = get_feature_columns(train_df)
    X_train = train_df[feat_cols].fillna(0).values
    y_train = train_df["isFraud"].values
    X_test = test_df[feat_cols].fillna(0).values
    y_test = test_df["isFraud"].values

    # 3. Cross-validation (optional — slow but thorough)
    if cross_validate:
        print("\n[CV] Running 5-fold stratified cross-validation ...")
        cv_model = FraudEnsemble(input_dim=len(feat_cols), smote=smote)
        cv_results = cv_model.cross_validate(X_train, y_train, n_splits=5)
        cv_auc_pr = [r["auc_pr"] for r in cv_results]
        print(
            f"[CV] AUC-PR: {np.mean(cv_auc_pr):.4f} ± {np.std(cv_auc_pr):.4f}"
        )

    # 4. Train full model
    print(f"\n[Train] Training ensemble (input_dim={len(feat_cols)}, epochs={epochs}) ...")
    model = FraudEnsemble(input_dim=len(feat_cols), ae_epochs=epochs, smote=smote)
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
    print(f"{'='*50}")

    # 6. Save model
    os.makedirs(model_dir, exist_ok=True)
    model.save(model_dir)
    print(f"\n[Save] Model saved to {model_dir}/")

    # Save metrics
    with open(os.path.join(model_dir, "eval_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # 7. Save test features for /evaluate endpoint
    test_df["isFraud"] = y_test
    test_df[feat_cols + ["isFraud"]].to_parquet(
        os.path.join(data_dir, "test_features.parquet"), index=False
    )
    print(f"[Save] Test features saved to {data_dir}/test_features.parquet")

    print(f"\n[Done] Total training time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fraud detection ensemble")
    parser.add_argument("--data-dir", default="data/", help="Directory with IEEE-CIS CSVs")
    parser.add_argument("--model-dir", default="models/", help="Directory to save model")
    parser.add_argument("--epochs", type=int, default=50, help="AutoEncoder epochs")
    parser.add_argument("--no-smote", action="store_true", help="Disable SMOTE oversampling")
    parser.add_argument("--cross-validate", action="store_true", help="Run 5-fold CV first")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        epochs=args.epochs,
        smote=not args.no_smote,
        cross_validate=args.cross_validate,
    )
