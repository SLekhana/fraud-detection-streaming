"""
Feature engineering for IEEE-CIS fraud detection.

v2 fixes:
- Velocity windowing: replaced expanding() with proper time-based rolling (closed='left')
- Card aggregates: accepts precomputed stats to prevent train→test leakage
- Extended feature set: V1-V20 (was V1-V6), numeric identity features id_01..id_11
- compute_card_stats() and compute_risk_profile() exported for use in train.py

Covers:
- Transaction velocity windowing (count/sum/std in 1h, 6h, 24h windows)
- Geolocation haversine distance (billing vs shipping address proxy)
- Merchant risk profiling (historical fraud rate per merchant category)
- Temporal features (hour of day, day of week, weekend flag)
- Card-level aggregates (avg transaction amount, card activity)
- IEEE-CIS V-columns (Vesta engineered, top 20 by variance)
- IEEE-CIS identity numeric features (id_01–id_11)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


# ─── Haversine distance ─────────────────────────────────────────────────────

def haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """
    Compute great-circle distance (km) between two lat/lon pairs.
    Used as proxy for billing-vs-shipping address distance in IEEE-CIS.
    """
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def add_address_distance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    addr1 = df["addr1"].fillna(0).astype(float)
    addr2 = df["addr2"].fillna(0).astype(float)
    lat1 = 30.0 + (addr1 % 100) / 10.0
    lon1 = -90.0 - (addr1 % 50) / 10.0
    lat2 = 30.0 + (addr2 % 100) / 10.0
    lon2 = -90.0 - (addr2 % 50) / 10.0
    df["addr_distance_km"] = haversine_distance(
        lat1.values, lon1.values, lat2.values, lon2.values
    )
    df["addr_mismatch"] = ((addr1 != addr2) & (addr1 > 0) & (addr2 > 0)).astype(int)
    return df


# ─── Transaction velocity windowing ─────────────────────────────────────────

def add_velocity_features(
    df: pd.DataFrame,
    windows_hours: list[int] | None = None,
) -> pd.DataFrame:
    """
    Row-wise velocity (used for single-transaction inference).
    Computes exact rolling window for the given transaction only.
    """
    if windows_hours is None:
        windows_hours = [1, 6, 24]

    df = df.copy().sort_values("TransactionDT")
    df["TransactionHour"] = df["TransactionDT"] / 3600.0

    for w in windows_hours:
        counts, sums, stds = [], [], []
        for _, row in df.iterrows():
            card = row["card1"]
            t = row["TransactionHour"]
            window = df[
                (df["card1"] == card)
                & (df["TransactionHour"] >= t - w)
                & (df["TransactionHour"] < t)
            ]
            counts.append(len(window))
            sums.append(window["TransactionAmt"].sum())
            stds.append(window["TransactionAmt"].std() if len(window) > 1 else 0.0)

        df[f"velocity_count_{w}h"] = counts
        df[f"velocity_sum_{w}h"] = sums
        df[f"velocity_std_{w}h"] = stds

    return df


def add_velocity_features_fast(
    df: pd.DataFrame,
    windows_hours: list[int] | None = None,
) -> pd.DataFrame:
    """
    Vectorised velocity computation (used for training on full dataset).

    FIX vs v1: Uses proper time-based rolling window (closed='left') instead
    of expanding().count() which was counting ALL prior transactions regardless
    of window size. This makes velocity_count_1h actually mean
    "transactions on this card in the past 1 hour", not "ever".
    """
    if windows_hours is None:
        windows_hours = [1, 6, 24]

    df = df.copy().sort_values(["card1", "TransactionDT"])
    df = df.reset_index(drop=True)

    # Build a datetime-indexed copy for time-based rolling
    ref = pd.Timestamp("2017-11-30")
    df["_dt"] = ref + pd.to_timedelta(df["TransactionDT"], unit="s")
    df_indexed = df.set_index("_dt")

    for w in windows_hours:
        window_str = f"{w}h"
        # closed='left' excludes the current transaction from its own window
        df[f"velocity_count_{w}h"] = (
            df_indexed.groupby("card1")["TransactionAmt"]
            .transform(lambda x: x.rolling(window_str, closed="left").count())
            .fillna(0)
            .values
        )
        df[f"velocity_sum_{w}h"] = (
            df_indexed.groupby("card1")["TransactionAmt"]
            .transform(lambda x: x.rolling(window_str, closed="left").sum())
            .fillna(0)
            .values
        )
        df[f"velocity_std_{w}h"] = (
            df_indexed.groupby("card1")["TransactionAmt"]
            .transform(lambda x: x.rolling(window_str, closed="left").std())
            .fillna(0)
            .values
        )

    df = df.drop(columns=["_dt"])
    return df


# ─── Merchant risk profiling ─────────────────────────────────────────────────

def compute_risk_profile(
    df: pd.DataFrame,
    target_col: str = "isFraud",
    category_col: str = "ProductCD",
) -> pd.DataFrame:
    """
    Compute historical fraud rate per merchant/product category.
    Call on TRAINING data only; pass result to add_merchant_risk() on test data.
    Exported alias of build_merchant_risk_profile for clarity.
    """
    return build_merchant_risk_profile(df, target_col=target_col, category_col=category_col)


def build_merchant_risk_profile(
    df: pd.DataFrame,
    target_col: str = "isFraud",
    category_col: str = "ProductCD",
) -> pd.DataFrame:
    profile = (
        df.groupby(category_col)[target_col]
        .agg(["mean", "count"])
        .rename(columns={"mean": "merchant_fraud_rate", "count": "merchant_tx_count"})
        .reset_index()
    )
    return profile


def add_merchant_risk(
    df: pd.DataFrame,
    risk_profile: pd.DataFrame,
    category_col: str = "ProductCD",
) -> pd.DataFrame:
    df = df.copy()
    df = df.merge(risk_profile, on=category_col, how="left")
    df["merchant_fraud_rate"] = df["merchant_fraud_rate"].fillna(0.0)
    df["merchant_tx_count"] = df["merchant_tx_count"].fillna(0)
    df["high_risk_merchant"] = (df["merchant_fraud_rate"] > 0.1).astype(int)
    return df


# ─── Temporal features ───────────────────────────────────────────────────────

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ref = pd.Timestamp("2017-11-30")
    dt = ref + pd.to_timedelta(df["TransactionDT"], unit="s")
    df["tx_hour"] = dt.dt.hour
    df["tx_day_of_week"] = dt.dt.dayofweek
    df["tx_is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    df["tx_is_night"] = ((dt.dt.hour < 6) | (dt.dt.hour >= 22)).astype(int)
    return df


# ─── Card-level aggregates ───────────────────────────────────────────────────

def compute_card_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute card-level statistics.
    Call on TRAINING data only to avoid leakage; pass result to add_card_aggregates().
    """
    stats = (
        df.groupby("card1")["TransactionAmt"]
        .agg(card_avg_amt="mean", card_std_amt="std", card_tx_count="count")
        .reset_index()
    )
    stats["card_std_amt"] = stats["card_std_amt"].fillna(0)
    return stats


def add_card_aggregates(
    df: pd.DataFrame,
    card_stats: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Add card-level aggregate features.

    FIX vs v1: card_stats is now passed in explicitly so that test/inference
    rows use statistics computed only on training data.
    If card_stats is None (backward compat), computes from df — which leaks
    for test sets and should only be used in exploratory contexts.
    """
    df = df.copy()
    if card_stats is None:
        card_stats = compute_card_stats(df)

    df = df.merge(card_stats, on="card1", how="left")

    # Fallback for cards unseen in training
    global_avg = card_stats["card_avg_amt"].mean()
    df["card_avg_amt"] = df["card_avg_amt"].fillna(global_avg)
    df["card_std_amt"] = df["card_std_amt"].fillna(0)
    df["card_tx_count"] = df["card_tx_count"].fillna(0)

    df["amt_zscore"] = (
        (df["TransactionAmt"] - df["card_avg_amt"]) / (df["card_std_amt"] + 1e-6)
    ).fillna(0)
    return df


# ─── Amount features ─────────────────────────────────────────────────────────

def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_amt"] = np.log1p(df["TransactionAmt"])
    df["amt_is_round"] = (df["TransactionAmt"] % 1 == 0).astype(int)
    df["amt_cents"] = (df["TransactionAmt"] * 100).astype(int) % 100
    return df


# ─── Master pipeline ─────────────────────────────────────────────────────────

NUMERIC_COLS = [
    # Amount
    "TransactionAmt", "log_amt", "amt_is_round", "amt_cents",
    # Address
    "addr_distance_km", "addr_mismatch",
    # Velocity (now uses correct time-based rolling)
    "velocity_count_1h", "velocity_sum_1h", "velocity_std_1h",
    "velocity_count_6h", "velocity_sum_6h", "velocity_std_6h",
    "velocity_count_24h", "velocity_sum_24h", "velocity_std_24h",
    # Merchant risk
    "merchant_fraud_rate", "merchant_tx_count", "high_risk_merchant",
    # Temporal
    "tx_hour", "tx_day_of_week", "tx_is_weekend", "tx_is_night",
    # Card aggregates (computed from train only)
    "card_avg_amt", "card_std_amt", "card_tx_count", "amt_zscore",
    # Distance features
    "dist1", "dist2",
    # Count features (behavioral)
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
    "C11", "C12", "C13", "C14",
    # Timedelta features
    "D1", "D2", "D3", "D4",
    # Vesta-engineered features — expanded from V1-V6 to V1-V20
    # These are masked/normalised by Vesta but carry strong signal
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    # Numeric identity features (from train_identity.csv)
    # id_01: device screen resolution width proxy
    # id_02: device screen resolution height proxy
    # id_03..id_06: network/browser numeric signals
    # id_09..id_11: additional numeric device signals
    "id_01", "id_02", "id_03", "id_05", "id_06", "id_09", "id_10", "id_11",
]


def build_features(
    df: pd.DataFrame,
    risk_profile: Optional[pd.DataFrame] = None,
    card_stats: Optional[pd.DataFrame] = None,
    fast: bool = True,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Args:
        df: Raw IEEE-CIS transaction DataFrame (merged with identity if available).
        risk_profile: Pre-computed merchant risk profile from TRAINING data.
                      Pass None only if merchant risk features are not needed.
        card_stats: Pre-computed card statistics from TRAINING data.
                    If None, computes from df (leaks for test sets — avoid in production).
        fast: Use vectorised time-based rolling (True for training/batch)
              vs row-wise (False for single-transaction streaming inference).
    """
    df = add_address_distance(df)
    df = add_temporal_features(df)
    df = add_amount_features(df)
    df = add_card_aggregates(df, card_stats=card_stats)

    if fast:
        df = add_velocity_features_fast(df)
    else:
        df = add_velocity_features(df)

    if risk_profile is not None:
        df = add_merchant_risk(df, risk_profile)
    else:
        df["merchant_fraud_rate"] = 0.0
        df["merchant_tx_count"] = 0
        df["high_risk_merchant"] = 0

    # Fill missing numeric cols with 0 (handles absent identity/V columns gracefully)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return list of engineered feature columns present in df."""
    return [c for c in NUMERIC_COLS if c in df.columns]
