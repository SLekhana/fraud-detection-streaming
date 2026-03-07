"""
Inference utilities for the fraud detection API.

Key problem solved:
  The model was trained on 52+ features including C1-C14, V1-V6, dist1, dist2
  which are NOT present in a typical API request (only TransactionAmt, ProductCD, etc.)
  
  This module:
  1. Loads expected feature names from models/meta.json
  2. Builds a zero-padded feature vector with the correct dimension
  3. Fills in whatever features ARE available from the request

This ensures the model always receives a vector of the correct shape.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from app.core.features import build_features


@lru_cache(maxsize=1)
def _load_expected_features(model_dir: str = "models") -> list[str]:
    """Load the feature names the model was trained on."""
    meta_path = Path(model_dir) / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        names = meta.get("feature_names", [])
        # feature_names in meta includes AE augmented names (ae_anomaly_score, ae_emb_*)
        # We want only the original feature names (before AE augmentation)
        # These are everything before "ae_anomaly_score"
        if "ae_anomaly_score" in names:
            original = names[: names.index("ae_anomaly_score")]
        else:
            original = names
        if original:
            return original
    # Fallback to NUMERIC_COLS
    from app.core.features import NUMERIC_COLS

    return NUMERIC_COLS


def build_inference_features(
    tx_dict: dict,
    model_dir: str = "models",
    risk_profile: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """
    Build a correctly-shaped feature vector for a single transaction.

    Missing columns are zero-filled so the model receives the same
    dimension it was trained on.

    Args:
        tx_dict: Raw transaction dict from the API request.
        model_dir: Path to saved model directory (has meta.json).
        risk_profile: Pre-computed merchant risk profile (optional).

    Returns:
        np.ndarray of shape (1, n_features) — ready for model.predict_proba()
    """
    expected_cols = _load_expected_features(model_dir)

    # Run feature engineering
    df = pd.DataFrame([tx_dict])
    df = build_features(df, risk_profile=risk_profile, fast=False)

    # Build output array with zeros for any missing column
    row = {}
    for col in expected_cols:
        if col in df.columns:
            row[col] = float(df[col].fillna(0).iloc[0])
        else:
            row[col] = 0.0

    return np.array([[row[c] for c in expected_cols]]), expected_cols


def get_inference_feature_names(model_dir: str = "models") -> list[str]:
    """Return the list of features the model expects."""
    return _load_expected_features(model_dir)
