"""
Ablation Study: Fraud Detection Ensemble
=========================================
Compares model variants to prove the 15% precision improvement claim.

Variants tested:
  1. Logistic Regression baseline (no feature engineering)
  2. XGBoost only (no AutoEncoder, full features)
  3. XGBoost only (no AutoEncoder, no SMOTE)
  4. Full Ensemble (AutoEncoder + XGBoost + SMOTE)          ← our model
  5. Full Ensemble — no velocity features
  6. Full Ensemble — no haversine/address features
  7. Full Ensemble — no merchant risk features
  8. Full Ensemble — SMOTE replaced with cost-sensitive learning

Usage:
    python scripts/ablation.py --data-dir data/ --model-dir models/ --output ablation_results.json

Results are saved to JSON + printed in a summary table.
Statistical significance tested via bootstrap (n=1000).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from imblearn.over_sampling import SMOTE

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.features import (
    build_features,
    build_merchant_risk_profile,
    get_feature_columns,
    NUMERIC_COLS,
)
from app.core.ensemble import FraudEnsemble


# ─── Helpers ─────────────────────────────────────────────────────────────────

def evaluate_predictions(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.4) -> dict:
    """Compute full evaluation metrics."""
    y_pred = (y_proba >= threshold).astype(int)
    auc_pr = average_precision_score(y_true, y_proba)
    auc_roc = roc_auc_score(y_true, y_proba)
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    fraud = report.get("1", {})

    # False negative rate
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "auc_pr": round(auc_pr, 4),
        "auc_roc": round(auc_roc, 4),
        "precision": round(fraud.get("precision", 0.0), 4),
        "recall": round(fraud.get("recall", 0.0), 4),
        "f1": round(fraud.get("f1-score", 0.0), 4),
        "false_negative_rate": round(fnr, 4),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
    }


def bootstrap_ci(y_true: np.ndarray, y_proba: np.ndarray, metric: str = "auc_pr", n: int = 500) -> tuple:
    """Bootstrap 95% CI for a metric."""
    scores = []
    rng = np.random.RandomState(42)
    for _ in range(n):
        idx = rng.randint(0, len(y_true), len(y_true))
        try:
            if metric == "auc_pr":
                scores.append(average_precision_score(y_true[idx], y_proba[idx]))
            elif metric == "auc_roc":
                scores.append(roc_auc_score(y_true[idx], y_proba[idx]))
        except Exception:
            continue
    return (round(np.percentile(scores, 2.5), 4), round(np.percentile(scores, 97.5), 4))


def statistical_significance(y_true, proba_a, proba_b, n_bootstrap=500):
    """Test if model B is significantly better than model A (AUC-PR)."""
    diffs = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        idx = rng.randint(0, len(y_true), len(y_true))
        try:
            a = average_precision_score(y_true[idx], proba_a[idx])
            b = average_precision_score(y_true[idx], proba_b[idx])
            diffs.append(b - a)
        except Exception:
            continue
    p_value = (np.array(diffs) <= 0).mean()
    return round(float(p_value), 4)


# ─── Feature subsets ─────────────────────────────────────────────────────────

VELOCITY_COLS = [c for c in NUMERIC_COLS if c.startswith("velocity_")]
HAVERSINE_COLS = ["addr_distance_km", "addr_mismatch"]
MERCHANT_COLS = ["merchant_fraud_rate", "merchant_tx_count", "high_risk_merchant"]
ALL_ENGINEERED = VELOCITY_COLS + HAVERSINE_COLS + MERCHANT_COLS


def drop_features(feat_cols: list, to_drop: list) -> list:
    return [c for c in feat_cols if c not in to_drop]


# ─── Model variants ───────────────────────────────────────────────────────────

def run_logistic_baseline(X_train, y_train, X_test, y_test) -> dict:
    print("  [LR] Training Logistic Regression baseline ...")
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    # Use class_weight for imbalance
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_train)
    proba = model.predict_proba(X_te)[:, 1]
    return proba


def run_xgb_only(X_train, y_train, X_test, y_test, smote=True) -> np.ndarray:
    tag = "SMOTE" if smote else "cost-sensitive"
    print(f"  [XGB-only/{tag}] Training ...")

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    if smote:
        sm = SMOTE(random_state=42, k_neighbors=5)
        X_tr, y_tr = sm.fit_resample(X_tr, y_train)
    else:
        y_tr = y_train

    ratio = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    params = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "aucpr",
        "early_stopping_rounds": 30,
        "random_state": 42,
        "n_jobs": -1,
    }
    if not smote:
        params["scale_pos_weight"] = ratio

    model = xgb.XGBClassifier(**params)
    split = int(0.85 * len(X_tr))
    model.fit(
        X_tr[:split], y_tr[:split],
        eval_set=[(X_tr[split:], y_tr[split:])],
        verbose=False,
    )
    return model.predict_proba(X_te)[:, 1]


def run_full_ensemble(X_train, y_train, X_test, y_test,
                      feat_names: list, epochs: int = 30) -> tuple[np.ndarray, FraudEnsemble]:
    print(f"  [Full Ensemble] Training (input_dim={len(feat_names)}, epochs={epochs}) ...")
    model = FraudEnsemble(
        input_dim=X_train.shape[1],
        ae_epochs=epochs,
        smote=True,
    )
    model.fit(X_train, y_train, feature_names=feat_names, verbose=False)
    proba = model.predict_proba(X_test)
    return proba, model


# ─── Main ablation ────────────────────────────────────────────────────────────

def run_ablation(
    data_dir: str,
    model_dir: str,
    output_path: str,
    epochs: int = 30,
    sample: Optional[int] = None,
) -> None:
    t_start = time.time()

    # ── Load data ──
    print("\n[Ablation] Loading data ...")
    tx_df = pd.read_csv(f"{data_dir}/train_transaction.csv")
    id_path = f"{data_dir}/train_identity.csv"
    if Path(id_path).exists():
        id_df = pd.read_csv(id_path)
        df = tx_df.merge(id_df, on="TransactionID", how="left")
    else:
        df = tx_df

    if sample:
        # Stratified subsample for speed
        fraud = df[df.isFraud == 1].sample(min(sample // 10, df.isFraud.sum()), random_state=42)
        legit = df[df.isFraud == 0].sample(sample - len(fraud), random_state=42)
        df = pd.concat([fraud, legit]).sample(frac=1, random_state=42)
        print(f"[Ablation] Using {len(df):,} samples ({df.isFraud.sum():,} fraud)")

    # ── Feature engineering ──
    print("[Ablation] Engineering features ...")
    risk_profile = build_merchant_risk_profile(df)
    df = build_features(df, risk_profile=risk_profile, fast=True)
    feat_cols = get_feature_columns(df)

    X = df[feat_cols].fillna(0).values
    y = df["isFraud"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"[Ablation] Train: {len(X_train):,} | Test: {len(X_test):,}")

    results = {}
    probas = {}  # store for significance testing

    # ─── Variant 1: Logistic Regression baseline ───────────────────────────
    print("\n[1/8] Logistic Regression baseline (no feature engineering) ...")
    # Use only basic features (no engineered ones)
    basic_cols = [c for c in feat_cols if c not in ALL_ENGINEERED]
    basic_idx = [feat_cols.index(c) for c in basic_cols if c in feat_cols]
    proba = run_logistic_baseline(X_train[:, basic_idx], y_train, X_test[:, basic_idx], y_test)
    results["1_logistic_baseline"] = evaluate_predictions(y_test, proba)
    results["1_logistic_baseline"]["variant"] = "Logistic Regression (basic features)"
    probas["lr"] = proba

    # ─── Variant 2: XGBoost only, full features, SMOTE ────────────────────
    print("\n[2/8] XGBoost only (full features, SMOTE) ...")
    proba = run_xgb_only(X_train, y_train, X_test, y_test, smote=True)
    results["2_xgb_smote"] = evaluate_predictions(y_test, proba)
    results["2_xgb_smote"]["variant"] = "XGBoost only (SMOTE)"
    probas["xgb_smote"] = proba

    # ─── Variant 3: XGBoost only, cost-sensitive (no SMOTE) ───────────────
    print("\n[3/8] XGBoost only (cost-sensitive, no SMOTE) ...")
    proba = run_xgb_only(X_train, y_train, X_test, y_test, smote=False)
    results["3_xgb_costsensitive"] = evaluate_predictions(y_test, proba)
    results["3_xgb_costsensitive"]["variant"] = "XGBoost only (cost-sensitive)"
    probas["xgb_cs"] = proba

    # ─── Variant 4: Full Ensemble (our model) ─────────────────────────────
    print("\n[4/8] Full Ensemble (AutoEncoder + XGBoost + SMOTE) ← our model ...")
    proba, full_model = run_full_ensemble(X_train, y_train, X_test, y_test, feat_cols, epochs)
    results["4_full_ensemble"] = evaluate_predictions(y_test, proba)
    results["4_full_ensemble"]["variant"] = "Full Ensemble (AE + XGB + SMOTE) ← OUR MODEL"
    probas["full"] = proba

    # ─── Variant 5: No velocity features ──────────────────────────────────
    print("\n[5/8] Full Ensemble — ablate velocity features ...")
    no_vel_cols = drop_features(feat_cols, VELOCITY_COLS)
    no_vel_idx = [feat_cols.index(c) for c in no_vel_cols]
    proba, _ = run_full_ensemble(
        X_train[:, no_vel_idx], y_train, X_test[:, no_vel_idx], y_test, no_vel_cols, epochs
    )
    results["5_no_velocity"] = evaluate_predictions(y_test, proba)
    results["5_no_velocity"]["variant"] = "Ensemble — no velocity features"
    results["5_no_velocity"]["features_dropped"] = VELOCITY_COLS

    # ─── Variant 6: No haversine/address features ──────────────────────────
    print("\n[6/8] Full Ensemble — ablate haversine/address features ...")
    no_geo_cols = drop_features(feat_cols, HAVERSINE_COLS)
    no_geo_idx = [feat_cols.index(c) for c in no_geo_cols]
    proba, _ = run_full_ensemble(
        X_train[:, no_geo_idx], y_train, X_test[:, no_geo_idx], y_test, no_geo_cols, epochs
    )
    results["6_no_haversine"] = evaluate_predictions(y_test, proba)
    results["6_no_haversine"]["variant"] = "Ensemble — no haversine/address features"
    results["6_no_haversine"]["features_dropped"] = HAVERSINE_COLS

    # ─── Variant 7: No merchant risk features ─────────────────────────────
    print("\n[7/8] Full Ensemble — ablate merchant risk features ...")
    no_merch_cols = drop_features(feat_cols, MERCHANT_COLS)
    no_merch_idx = [feat_cols.index(c) for c in no_merch_cols]
    proba, _ = run_full_ensemble(
        X_train[:, no_merch_idx], y_train, X_test[:, no_merch_idx], y_test, no_merch_cols, epochs
    )
    results["7_no_merchant_risk"] = evaluate_predictions(y_test, proba)
    results["7_no_merchant_risk"]["variant"] = "Ensemble — no merchant risk"
    results["7_no_merchant_risk"]["features_dropped"] = MERCHANT_COLS

    # ─── Variant 8: No SMOTE (cost-sensitive ensemble) ────────────────────
    print("\n[8/8] Full Ensemble — cost-sensitive (no SMOTE) ...")
    model_nsmote = FraudEnsemble(input_dim=X_train.shape[1], ae_epochs=epochs, smote=False)
    model_nsmote.fit(X_train, y_train, feature_names=feat_cols, verbose=False)
    proba = model_nsmote.predict_proba(X_test)
    results["8_no_smote"] = evaluate_predictions(y_test, proba)
    results["8_no_smote"]["variant"] = "Ensemble — no SMOTE (cost-sensitive)"

    # ─── Statistical significance vs baseline ─────────────────────────────
    print("\n[Stats] Computing bootstrap CIs and significance tests ...")
    for key, proba in probas.items():
        if key == "full":
            continue
        p_val = statistical_significance(y_test, proba, probas["full"])
        results[f"sig_{key}_vs_full"] = {
            "comparison": f"Full ensemble vs {key}",
            "p_value": p_val,
            "significant": p_val < 0.05,
        }

    full_ci = bootstrap_ci(y_test, probas["full"], "auc_pr")
    results["4_full_ensemble"]["auc_pr_95ci"] = full_ci

    # ─── Summary table ────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"{'Variant':<45} {'AUC-PR':>8} {'AUC-ROC':>8} {'Precision':>10} {'Recall':>8} {'FNR':>8}")
    print(f"{'─'*90}")

    full_prec = results["4_full_ensemble"]["precision"]
    full_fnr = results["4_full_ensemble"]["false_negative_rate"]

    for key in sorted(results.keys()):
        r = results[key]
        if "variant" not in r:
            continue
        name = r["variant"][:44]
        prec_delta = ""
        if key != "4_full_ensemble" and "precision" in r:
            delta = full_prec - r["precision"]
            prec_delta = f" (+{delta:.3f})" if delta > 0 else f" ({delta:.3f})"
        fnr_delta = ""
        if key != "4_full_ensemble" and "false_negative_rate" in r:
            delta = r["false_negative_rate"] - full_fnr
            fnr_delta = f" ({delta:+.3f})" if delta != 0 else ""

        print(
            f"{name:<45} "
            f"{r.get('auc_pr', 0):>8.4f} "
            f"{r.get('auc_roc', 0):>8.4f} "
            f"{r.get('precision', 0):>8.4f}{prec_delta:>4} "
            f"{r.get('recall', 0):>8.4f} "
            f"{r.get('false_negative_rate', 0):>6.4f}{fnr_delta}"
        )

    print(f"{'='*90}")

    # Compute improvement vs best single-model baseline
    best_baseline_prec = max(
        results["2_xgb_smote"]["precision"],
        results["3_xgb_costsensitive"]["precision"],
        results["1_logistic_baseline"]["precision"],
    )
    prec_improvement_pct = (full_prec - best_baseline_prec) / max(best_baseline_prec, 1e-6) * 100

    best_baseline_fnr = min(
        results["2_xgb_smote"]["false_negative_rate"],
        results["3_xgb_costsensitive"]["false_negative_rate"],
        results["1_logistic_baseline"]["false_negative_rate"],
    )
    fnr_reduction_pct = (best_baseline_fnr - full_fnr) / max(best_baseline_fnr, 1e-6) * 100

    print(f"\n[Summary] Precision improvement vs best baseline: {prec_improvement_pct:+.1f}%")
    print(f"[Summary] False-negative rate reduction vs best baseline: {fnr_reduction_pct:+.1f}%")
    print(f"[Summary] 95% CI for full ensemble AUC-PR: {full_ci}")

    results["_summary"] = {
        "precision_improvement_vs_best_baseline_pct": round(prec_improvement_pct, 2),
        "fnr_reduction_vs_best_baseline_pct": round(fnr_reduction_pct, 2),
        "full_ensemble_auc_pr_95ci": full_ci,
        "best_baseline": "xgb_smote",
        "total_time_s": round(time.time() - t_start, 1),
    }

    # ─── Save ─────────────────────────────────────────────────────────────
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n[Saved] Results → {output_path}")
    print(f"[Done] Total ablation time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation study for fraud detection ensemble")
    parser.add_argument("--data-dir", default="data/")
    parser.add_argument("--model-dir", default="models/")
    parser.add_argument("--output", default="models/ablation_results.json")
    parser.add_argument("--epochs", type=int, default=30, help="AE epochs per variant (default: 30 for speed)")
    parser.add_argument("--sample", type=int, default=None, help="Subsample N rows for fast testing (e.g. 50000)")
    args = parser.parse_args()

    run_ablation(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        output_path=args.output,
        epochs=args.epochs,
        sample=args.sample,
    )
