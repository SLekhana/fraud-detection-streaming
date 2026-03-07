"""
Stacked Ensemble: AutoEncoder anomaly score + XGBoost classifier.

v2 changes:
- use_ae flag: set False to run pure cost-sensitive XGBoost (no AutoEncoder).
  SHAP analysis confirmed ae_emb_* and ae_anomaly_score never appear in top 20
  features — the AE adds ~100ms inference latency for negligible model contribution.
  Pure XGBoost (cost-sensitive) achieved precision=0.4538 vs ensemble's 0.1655
  pre-calibration in ablation study.
- calibrator support (unchanged from v1 patch)
- meta.json now stores use_ae flag for correct load() behaviour

Architecture (use_ae=True):
  Stage 1 — AutoEncoder reconstruction error → anomaly_score feature
  Stage 1b — AutoEncoder bottleneck embeddings (16-dim) as extra features
  Stage 2 — XGBoost trained on [original_features + anomaly_score + embeddings]

Architecture (use_ae=False):
  Stage 1 — XGBoost trained directly on original_features with scale_pos_weight
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
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


import pickle


class _PlattRemapper(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "PlattScaler":
            from app.core import ensemble

            return ensemble.PlattScaler
        return super().find_class(module, name)


class PlattScaler:
    """Platt scaling calibrator. Must live in ensemble.py so joblib.load always finds it."""

    def __init__(self):
        from sklearn.linear_model import LogisticRegression

        self.lr = LogisticRegression()

    def fit(self, scores, labels):
        self.lr.fit(scores.reshape(-1, 1), labels)
        return self

    def predict_proba(self, scores):
        return self.lr.predict_proba(scores.reshape(-1, 1))[:, 1]


class FraudEnsemble:
    """
    Two-stage stacked ensemble for fraud detection.

    use_ae=True  → Stage 1: AutoEncoder; Stage 2: XGBoost on augmented features
    use_ae=False → Stage 1: XGBoost directly on original features (faster, often better)
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
        use_ae: bool = True,
    ):
        self.input_dim = input_dim
        self.smote = smote
        self.scale_pos_weight = scale_pos_weight
        self.use_ae = use_ae
        self.calibrator = None

        # Stage 1 (only instantiated if use_ae=True)
        if self.use_ae:
            self.ae_trainer = AutoEncoderTrainer(
                input_dim=input_dim,
                bottleneck=ae_bottleneck,
                dropout=ae_dropout,
                lr=ae_lr,
                device=device,
            )
        else:
            self.ae_trainer = None

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
        self.explainer = None
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
        if feature_names:
            self.feature_names = feature_names

        # 1. Scale
        X_scaled = self.scaler.fit_transform(X)

        if self.use_ae:
            # 2. Train AE on legit only
            X_legit = X_scaled[y == 0]
            X_val_ae = X_scaled[:5000] if len(X_scaled) > 5000 else X_scaled
            if verbose:
                print(
                    f"[AE] Training on {len(X_legit):,} legit transactions "
                    f"({X_legit.shape[1]} features)"
                )
            self.ae_trainer.fit(
                X_legit,
                X_val_ae,
                epochs=self.ae_epochs,
                patience=ae_patience,
                verbose=verbose,
            )
            threshold = self.ae_trainer.calibrate_threshold(X_legit, percentile=95.0)
            if verbose:
                print(f"[AE] Anomaly threshold calibrated: {threshold:.6f}")

            # 3. Augment features with AE output
            X_aug = self._augment(X_scaled)
            names_aug = self._augmented_feature_names()
        else:
            if verbose:
                print(
                    "[XGB] use_ae=False — skipping AutoEncoder, using original features only"
                )
            X_aug = X_scaled
            names_aug = (
                self.feature_names
                if self.feature_names
                else [f"f{i}" for i in range(self.input_dim)]
            )

        # 4. SMOTE (if enabled)
        if self.smote:
            fraud_count = y.sum()
            legit_count = len(y) - fraud_count
            if verbose:
                print(
                    f"[SMOTE] Class ratio before: {legit_count:,} legit / "
                    f"{fraud_count:,} fraud (1:{legit_count // max(fraud_count, 1)})"
                )
            sm = SMOTE(random_state=42, k_neighbors=5)
            X_aug, y = sm.fit_resample(X_aug, y)
            if verbose:
                print(f"[SMOTE] After resampling: {X_aug.shape[0]:,} samples")

        # Auto scale_pos_weight if not using SMOTE
        if self.scale_pos_weight is None and not self.smote:
            ratio = (y == 0).sum() / max((y == 1).sum(), 1)
            self.xgb_model.set_params(scale_pos_weight=ratio)
            if verbose:
                print(f"[XGB] scale_pos_weight set to {ratio:.1f}")

        # 5. Train XGBoost
        split = int(0.85 * len(X_aug))
        X_tr, X_ev = X_aug[:split], X_aug[split:]
        y_tr, y_ev = y[:split], y[split:]

        if verbose:
            print(f"[XGB] Training on {len(X_tr):,} samples …")

        self.xgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_ev, y_ev)],
            verbose=False,
        )

        # 6. SHAP explainer
        import shap

        self.explainer = shap.TreeExplainer(self.xgb_model)
        self.feature_names = names_aug

        if verbose:
            print("[Ensemble] Training complete ✓")

        return self

    # ── Predict ──────────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        X_aug = self._augment(X_scaled) if self.use_ae else X_scaled
        proba = self.xgb_model.predict_proba(X_aug)[:, 1]
        if self.calibrator is not None:
            proba = self.calibrator.predict_proba(proba)
        return proba

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        if not self.use_ae or self.ae_trainer is None:
            return np.zeros(len(X))
        X_scaled = self.scaler.transform(X)
        return self.ae_trainer.predict_anomaly_score(X_scaled)

    # ── SHAP ─────────────────────────────────────────────────────────────────

    def explain(self, X: np.ndarray) -> dict:
        if self.explainer is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X_scaled = self.scaler.transform(X)
        X_aug = self._augment(X_scaled) if self.use_ae else X_scaled
        shap_values = self.explainer.shap_values(X_aug)
        return {
            "shap_values": shap_values,
            "feature_names": self.feature_names,
            "expected_value": float(self.explainer.expected_value),
        }

    def explain_single(self, x: np.ndarray) -> dict:
        result = self.explain(x.reshape(1, -1))
        sv = result["shap_values"][0]
        names = result["feature_names"]
        idx = np.argsort(np.abs(sv))[::-1]
        top = [
            {
                "feature": names[i] if i < len(names) else f"f{i}",
                "shap_value": float(sv[i]),
                "direction": (
                    "increases_fraud_risk" if sv[i] > 0 else "decreases_fraud_risk"
                ),
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
        proba = self.predict_proba(X)
        pred = (proba >= 0.5).astype(int)
        pr, rec, _ = precision_recall_curve(y, proba)
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
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        results = []
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            if verbose:
                print(f"[CV] Fold {fold + 1}/{n_splits}")
            fold_model = FraudEnsemble(
                input_dim=self.input_dim,
                smote=self.smote,
                use_ae=self.use_ae,
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
        os.makedirs(directory, exist_ok=True)
        if self.use_ae and self.ae_trainer is not None:
            self.ae_trainer.save(os.path.join(directory, "autoencoder.pt"))
        self.xgb_model.save_model(os.path.join(directory, "xgboost.json"))
        import joblib

        joblib.dump(self.scaler, os.path.join(directory, "scaler.pkl"))
        meta = {
            "input_dim": self.input_dim,
            "feature_names": self.feature_names,
            "ae_threshold": (
                self.ae_trainer.threshold if self.use_ae and self.ae_trainer else None
            ),
            "use_ae": self.use_ae,
        }
        with open(os.path.join(directory, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        # Save calibrator if present
        cal_path = os.path.join(directory, "calibrator.pkl")
        if self.calibrator is not None:
            joblib.dump(self.calibrator, cal_path)

    @classmethod
    def load(cls, directory: str, device: Optional[str] = None) -> "FraudEnsemble":
        with open(os.path.join(directory, "meta.json")) as f:
            meta = json.load(f)

        use_ae = meta.get("use_ae", True)  # backward compatible default
        ensemble = cls(input_dim=meta["input_dim"], device=device, use_ae=use_ae)

        if use_ae:
            ae_path = os.path.join(directory, "autoencoder.pt")
            if os.path.exists(ae_path):
                ensemble.ae_trainer = AutoEncoderTrainer.load(ae_path, device=device)
            else:
                # Fallback: disable AE if file missing
                ensemble.use_ae = False
                ensemble.ae_trainer = None

        ensemble.xgb_model = xgb.XGBClassifier()
        ensemble.xgb_model.load_model(os.path.join(directory, "xgboost.json"))
        import joblib

        ensemble.scaler = joblib.load(os.path.join(directory, "scaler.pkl"))
        ensemble.feature_names = meta.get("feature_names", [])
        import shap

        ensemble.explainer = shap.TreeExplainer(ensemble.xgb_model)

        # Load calibrator if present (backward compatible)
        cal_path = os.path.join(directory, "calibrator.pkl")
        if os.path.exists(cal_path):
            with open(cal_path, "rb") as _f:
                import sys as _sys
                from app.core.ensemble import PlattScaler as _PS

                _sys.modules.setdefault("__main__", type(_sys)("__main__"))
                _sys.modules["__main__"].PlattScaler = _PS
                try:
                    ensemble.calibrator = joblib.load(_f)
                except Exception:
                    ensemble.calibrator = None

        return ensemble

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _augment(self, X_scaled: np.ndarray) -> np.ndarray:
        """Add AE anomaly score + bottleneck embeddings. Only called if use_ae=True."""
        if not self.use_ae or self.ae_trainer is None:
            return X_scaled
        ae_score = self.ae_trainer.predict_anomaly_score(X_scaled).reshape(-1, 1)
        embeddings = self.ae_trainer.get_embeddings(X_scaled)
        return np.hstack([X_scaled, ae_score, embeddings])

    def _augmented_feature_names(self) -> list[str]:
        original = (
            self.feature_names
            if self.feature_names
            else [f"f{i}" for i in range(self.input_dim)]
        )
        if not self.use_ae or self.ae_trainer is None:
            return original
        ae_name = ["ae_anomaly_score"]
        embed_names = [f"ae_emb_{i}" for i in range(self.ae_trainer.model.bottleneck)]
        return original + ae_name + embed_names
