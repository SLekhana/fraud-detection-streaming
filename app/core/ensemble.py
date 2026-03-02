"""
Stacked Ensemble: AutoEncoder anomaly score + XGBoost classifier.

Architecture:
  Stage 1 — AutoEncoder reconstruction error → anomaly_score feature
  Stage 1b — AutoEncoder bottleneck embeddings (16-dim) as extra features
  Stage 2 — XGBoost trained on [original_features + anomaly_score + embeddings]

SHAP explainability:
  TreeExplainer on XGBoost produces per-feature Shapley values for each
  flagged transaction, enabling the LLM agent to generate human-readable
  fraud justifications.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from app.core.autoencoder import AutoEncoderTrainer


# ─── Ensemble model ──────────────────────────────────────────────────────────


class FraudEnsemble:
    """
    Two-stage stacked ensemble for fraud detection.

    Stage 1: AutoEncoder trained on legit transactions only.
             Outputs reconstruction error (anomaly score) + bottleneck embedding.

    Stage 2: XGBoost classifier trained on:
             [original engineered features]
             + [autoencoder anomaly score]
             + [16-dim bottleneck embeddings]
    """

    def __init__(
        self,
        input_dim: int,
        ae_bottleneck: int = 16,
        ae_dropout: float = 0.2,
        ae_lr: float = 1e-3,
        ae_epochs: int = 50,
        xgb_params: Optional[dict] = None,
        smote: bool = True,
        scale_pos_weight: Optional[float] = None,
        device: Optional[str] = None,
    ):
        self.input_dim = input_dim
        self.smote = smote
        self.scale_pos_weight = scale_pos_weight

        # Stage 1
        self.ae_trainer = AutoEncoderTrainer(
            input_dim=input_dim,
            bottleneck=ae_bottleneck,
            dropout=ae_dropout,
            lr=ae_lr,
            device=device,
        )
        self.ae_epochs = ae_epochs
        self.scaler = StandardScaler()

        # Stage 2
        default_xgb = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "gamma": 1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "use_label_encoder": False,
            "eval_metric": "aucpr",
            "early_stopping_rounds": 30,
            "random_state": 42,
            "n_jobs": -1,
        }
        if xgb_params:
            default_xgb.update(xgb_params)
        if scale_pos_weight:
            default_xgb["scale_pos_weight"] = scale_pos_weight

        self.xgb_model = xgb.XGBClassifier(**default_xgb)
        self.explainer: Optional[shap.TreeExplainer] = None
        self.feature_names: list[str] = []

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[list[str]] = None,
        ae_patience: int = 10,
        verbose: bool = True,
    ) -> "FraudEnsemble":
        """
        Full training pipeline.
        1. Scale features.
        2. Train AutoEncoder on legit transactions.
        3. Calibrate anomaly threshold.
        4. Augment features with AE score + embeddings.
        5. Apply SMOTE if requested.
        6. Train XGBoost on augmented features.
        7. Build SHAP explainer.
        """
        if feature_names:
            self.feature_names = feature_names

        # 1. Scale
        X_scaled = self.scaler.fit_transform(X)

        # 2. Train AE on legit only
        X_legit = X_scaled[y == 0]
        X_val_ae = X_scaled[:5000] if len(X_scaled) > 5000 else X_scaled
        if verbose:
            print(
                f"[AE] Training on {len(X_legit):,} legit transactions "
                f"({X_legit.shape[1]} features)"
            )
        self.ae_trainer.fit(
            X_legit, X_val_ae, epochs=self.ae_epochs, patience=ae_patience, verbose=verbose
        )

        # 3. Calibrate threshold at 95th percentile of legit errors
        threshold = self.ae_trainer.calibrate_threshold(X_legit, percentile=95.0)
        if verbose:
            print(f"[AE] Anomaly threshold calibrated: {threshold:.6f}")

        # 4. Augment features
        X_aug = self._augment(X_scaled)
        names_aug = self._augmented_feature_names()

        # 5. SMOTE
        if self.smote:
            fraud_count = y.sum()
            legit_count = len(y) - fraud_count
            if verbose:
                print(
                    f"[SMOTE] Class ratio before: {legit_count:,} legit / "
                    f"{fraud_count:,} fraud (1:{legit_count // fraud_count})"
                )
            sm = SMOTE(random_state=42, k_neighbors=5)
            X_aug, y = sm.fit_resample(X_aug, y)
            if verbose:
                print(f"[SMOTE] After resampling: {X_aug.shape[0]:,} samples")

        # Auto scale_pos_weight if not provided
        if self.scale_pos_weight is None and not self.smote:
            ratio = (y == 0).sum() / max((y == 1).sum(), 1)
            self.xgb_model.set_params(scale_pos_weight=ratio)
            if verbose:
                print(f"[XGB] scale_pos_weight set to {ratio:.1f}")

        # 6. Train XGBoost
        split = int(0.85 * len(X_aug))
        X_tr, X_ev = X_aug[:split], X_aug[split:]
        y_tr, y_ev = y[:split], y[split:]

        if verbose:
            print(f"[XGB] Training on {len(X_tr):,} samples …")

        self.xgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_ev, y_ev)],
            verbose=False,
        )

        # 7. SHAP explainer
        self.explainer = shap.TreeExplainer(self.xgb_model)
        self.feature_names = names_aug

        if verbose:
            print("[Ensemble] Training complete ✓")

        return self

    # ── Predict ──────────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return fraud probability for each sample."""
        X_scaled = self.scaler.transform(X)
        X_aug = self._augment(X_scaled)
        return self.xgb_model.predict_proba(X_aug)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Return raw AutoEncoder reconstruction errors."""
        X_scaled = self.scaler.transform(X)
        return self.ae_trainer.predict_anomaly_score(X_scaled)

    # ── SHAP ─────────────────────────────────────────────────────────────────

    def explain(self, X: np.ndarray) -> dict:
        """
        Compute SHAP values for a batch of transactions.
        Returns dict with shap_values array and feature names.
        """
        if self.explainer is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X_scaled = self.scaler.transform(X)
        X_aug = self._augment(X_scaled)
        shap_values = self.explainer.shap_values(X_aug)
        return {
            "shap_values": shap_values,
            "feature_names": self.feature_names,
            "expected_value": float(self.explainer.expected_value),
        }

    def explain_single(self, x: np.ndarray) -> dict:
        """
        Explain a single transaction. Returns top features driving the score.
        x: 1D array of shape (input_dim,)
        """
        result = self.explain(x.reshape(1, -1))
        sv = result["shap_values"][0]
        names = result["feature_names"]

        # Sort by absolute SHAP value
        idx = np.argsort(np.abs(sv))[::-1]
        top = [
            {
                "feature": names[i] if i < len(names) else f"f{i}",
                "shap_value": float(sv[i]),
                "direction": "increases_fraud_risk" if sv[i] > 0 else "decreases_fraud_risk",
            }
            for i in idx[:10]
        ]
        return {
            "top_features": top,
            "expected_value": result["expected_value"],
            "sum_shap": float(sv.sum()),
        }

    # ── Evaluation ───────────────────────────────────────────────────────────

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Full evaluation: AUC-PR, AUC-ROC, precision/recall, confusion matrix."""
        proba = self.predict_proba(X)
        pred = (proba >= 0.5).astype(int)

        pr, rec, thresholds = precision_recall_curve(y, proba)
        auc_pr = average_precision_score(y, proba)
        auc_roc = roc_auc_score(y, proba)
        cm = confusion_matrix(y, pred)
        report = classification_report(y, pred, output_dict=True)

        return {
            "auc_pr": round(auc_pr, 4),
            "auc_roc": round(auc_roc, 4),
            "precision_at_0.5": round(report.get("1", {}).get("precision", 0), 4),
            "recall_at_0.5": round(report.get("1", {}).get("recall", 0), 4),
            "f1_at_0.5": round(report.get("1", {}).get("f1-score", 0), 4),
            "confusion_matrix": cm.tolist(),
            "precision_recall_curve": {
                "precision": pr.tolist()[:100],
                "recall": rec.tolist()[:100],
            },
            "classification_report": report,
        }

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        verbose: bool = True,
    ) -> list[dict]:
        """Stratified K-Fold cross-validation."""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        results = []
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            if verbose:
                print(f"[CV] Fold {fold + 1}/{n_splits}")
            fold_model = FraudEnsemble(
                input_dim=self.input_dim,
                smote=self.smote,
            )
            fold_model.fit(X[tr_idx], y[tr_idx], verbose=False)
            metrics = fold_model.evaluate(X[val_idx], y[val_idx])
            results.append(metrics)
            if verbose:
                print(
                    f"      AUC-PR={metrics['auc_pr']:.4f} | "
                    f"AUC-ROC={metrics['auc_roc']:.4f} | "
                    f"Recall={metrics['recall_at_0.5']:.4f}"
                )
        return results

    # ── Save/Load ────────────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        """Save all model artefacts to directory."""
        os.makedirs(directory, exist_ok=True)
        self.ae_trainer.save(os.path.join(directory, "autoencoder.pt"))
        self.xgb_model.save_model(os.path.join(directory, "xgboost.json"))
        import joblib
        joblib.dump(self.scaler, os.path.join(directory, "scaler.pkl"))
        meta = {
            "input_dim": self.input_dim,
            "feature_names": self.feature_names,
            "ae_threshold": self.ae_trainer.threshold,
        }
        with open(os.path.join(directory, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, directory: str, device: Optional[str] = None) -> "FraudEnsemble":
        """Load saved ensemble from directory."""
        with open(os.path.join(directory, "meta.json")) as f:
            meta = json.load(f)

        ensemble = cls(input_dim=meta["input_dim"], device=device)
        ensemble.ae_trainer = AutoEncoderTrainer.load(
            os.path.join(directory, "autoencoder.pt"), device=device
        )
        ensemble.xgb_model = xgb.XGBClassifier()
        ensemble.xgb_model.load_model(os.path.join(directory, "xgboost.json"))
        import joblib
        ensemble.scaler = joblib.load(os.path.join(directory, "scaler.pkl"))
        ensemble.feature_names = meta.get("feature_names", [])
        ensemble.explainer = shap.TreeExplainer(ensemble.xgb_model)
        return ensemble

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _augment(self, X_scaled: np.ndarray) -> np.ndarray:
        """Add AE anomaly score + bottleneck embeddings to feature matrix."""
        ae_score = self.ae_trainer.predict_anomaly_score(X_scaled).reshape(-1, 1)
        embeddings = self.ae_trainer.get_embeddings(X_scaled)
        return np.hstack([X_scaled, ae_score, embeddings])

    def _augmented_feature_names(self) -> list[str]:
        """Feature names for the augmented feature matrix."""
        original = self.feature_names if self.feature_names else [
            f"f{i}" for i in range(self.input_dim)
        ]
        ae_name = ["ae_anomaly_score"]
        embed_names = [f"ae_emb_{i}" for i in range(self.ae_trainer.model.bottleneck)]
        return original + ae_name + embed_names
