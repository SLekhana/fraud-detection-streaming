"""
Feature engineering for IEEE-CIS fraud detection.

Covers:
- Transaction velocity windowing (count/sum/std in 1h, 6h, 24h windows)
- Geolocation haversine distance (billing vs shipping address proxy)
- Merchant risk profiling (historical fraud rate per merchant category)
- Temporal features (hour of day, day of week, weekend flag)
- Card-level aggregates (avg transaction amount, card activity)
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
    Used as proxy for billing-vs-shipping address distance in IEEE-CIS
    (addr1/addr2 mapped to approximate coordinates via zip-code lookup).
    """
    R = 6371.0  # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def add_address_distance(df: pd.DataFrame) -> pd.DataFrame:
    """
    IEEE-CIS addr1/addr2 are ZIP codes. We approximate lat/lon from ZIP
    modular arithmetic as a deterministic proxy (real deployment would use
    a ZIP→lat/lon lookup table).
    """
    df = df.copy()
    addr1 = df["addr1"].fillna(0).astype(float)
    addr2 = df["addr2"].fillna(0).astype(float)

    # Deterministic ZIP → approximate lat/lon proxy
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
    For each card (card1), compute transaction velocity in rolling windows:
    - count of transactions
    - sum of transaction amounts
    - std of transaction amounts

    IEEE-CIS TransactionDT is seconds since a reference date; we convert
    to hours for windowing.
    """
    if windows_hours is None:
        windows_hours = [1, 6, 24]

    df = df.copy().sort_values("TransactionDT")
    df["TransactionHour"] = df["TransactionDT"] / 3600.0

    for w in windows_hours:
        count_col = f"velocity_count_{w}h"
        sum_col = f"velocity_sum_{w}h"
        std_col = f"velocity_std_{w}h"

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

        df[count_col] = counts
        df[sum_col] = sums
        df[std_col] = stds

    return df


def add_velocity_features_fast(
    df: pd.DataFrame,
    windows_hours: list[int] | None = None,
) -> pd.DataFrame:
    """
    Vectorised velocity computation (used for training on full dataset).
    Groups by card1 and uses rolling window on sorted timestamps.
    """
    if windows_hours is None:
        windows_hours = [1, 6, 24]

    df = df.copy().sort_values(["card1", "TransactionDT"])
    df["TransactionHour"] = df["TransactionDT"] / 3600.0
    df = df.reset_index(drop=True)

    for w in windows_hours:
        window_size = w * 3600  # back to seconds for comparison

        # Use expanding window per card as approximation for rolling
        grp = df.groupby("card1")["TransactionAmt"]
        df[f"velocity_count_{w}h"] = (
            grp.transform(lambda x: x.expanding().count()) - 1
        ).clip(lower=0)
        df[f"velocity_sum_{w}h"] = grp.transform(
            lambda x: x.expanding().sum() - x
        )
        df[f"velocity_std_{w}h"] = grp.transform(
            lambda x: x.expanding().std().fillna(0)
        )

    return df


# ─── Merchant risk profiling ─────────────────────────────────────────────────

def build_merchant_risk_profile(
    df: pd.DataFrame,
    target_col: str = "isFraud",
    category_col: str = "ProductCD",
) -> pd.DataFrame:
    """
    Compute historical fraud rate per merchant/product category.
    Returns a risk profile DataFrame (used to join onto test data).
    """
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
    """Join merchant risk profile onto transaction DataFrame."""
    df = df.copy()
    df = df.merge(risk_profile, on=category_col, how="left")
    df["merchant_fraud_rate"] = df["merchant_fraud_rate"].fillna(0.0)
    df["merchant_tx_count"] = df["merchant_tx_count"].fillna(0)
    df["high_risk_merchant"] = (df["merchant_fraud_rate"] > 0.1).astype(int)
    return df


# ─── Temporal features ───────────────────────────────────────────────────────

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    IEEE-CIS TransactionDT = seconds elapsed since 2017-11-30.
    Extract hour of day, day of week, weekend flag.
    """
    df = df.copy()
    ref = pd.Timestamp("2017-11-30")
    dt = ref + pd.to_timedelta(df["TransactionDT"], unit="s")
    df["tx_hour"] = dt.dt.hour
    df["tx_day_of_week"] = dt.dt.dayofweek
    df["tx_is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    df["tx_is_night"] = ((dt.dt.hour < 6) | (dt.dt.hour >= 22)).astype(int)
    return df


# ─── Card-level aggregates ───────────────────────────────────────────────────

def add_card_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Per-card historical averages for amount deviation detection."""
    df = df.copy()
    card_stats = (
        df.groupby("card1")["TransactionAmt"]
        .agg(card_avg_amt="mean", card_std_amt="std", card_tx_count="count")
        .reset_index()
    )
    card_stats["card_std_amt"] = card_stats["card_std_amt"].fillna(0)
    df = df.merge(card_stats, on="card1", how="left")
    df["amt_zscore"] = (
        (df["TransactionAmt"] - df["card_avg_amt"]) / (df["card_std_amt"] + 1e-6)
    ).fillna(0)
    return df


# ─── Amount features ─────────────────────────────────────────────────────────

def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Log-transform and round-amount flag."""
    df = df.copy()
    df["log_amt"] = np.log1p(df["TransactionAmt"])
    df["amt_is_round"] = (df["TransactionAmt"] % 1 == 0).astype(int)
    df["amt_cents"] = (df["TransactionAmt"] * 100).astype(int) % 100
    return df


# ─── Master pipeline ─────────────────────────────────────────────────────────

NUMERIC_COLS = [
    "TransactionAmt", "log_amt", "amt_is_round", "amt_cents",
    "addr_distance_km", "addr_mismatch",
    "velocity_count_1h", "velocity_sum_1h", "velocity_std_1h",
    "velocity_count_6h", "velocity_sum_6h", "velocity_std_6h",
    "velocity_count_24h", "velocity_sum_24h", "velocity_std_24h",
    "merchant_fraud_rate", "merchant_tx_count", "high_risk_merchant",
    "tx_hour", "tx_day_of_week", "tx_is_weekend", "tx_is_night",
    "card_avg_amt", "card_std_amt", "card_tx_count", "amt_zscore",
    "dist1", "dist2",
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
    "C11", "C12", "C13", "C14",
    "D1", "D2", "D3", "D4",
    "V1", "V2", "V3", "V4", "V5", "V6",
]


def build_features(
    df: pd.DataFrame,
    risk_profile: Optional[pd.DataFrame] = None,
    fast: bool = True,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Args:
        df: Raw IEEE-CIS transaction DataFrame.
        risk_profile: Pre-computed merchant risk profile (from training set).
        fast: Use vectorised velocity (True for training) vs row-wise (False for streaming).
    """
    df = add_address_distance(df)
    df = add_temporal_features(df)
    df = add_amount_features(df)
    df = add_card_aggregates(df)

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

    # Fill missing numeric cols
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return list of engineered feature columns present in df."""
    return [c for c in NUMERIC_COLS if c in df.columns]
