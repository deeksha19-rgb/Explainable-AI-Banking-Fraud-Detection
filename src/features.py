"""Shared feature definitions and the scikit-learn pipeline builder.

Feature engineering lives *inside* the pipeline (FunctionTransformer), so the
model's inputs are the raw transaction columns. That keeps permutation
importance and per-transaction explanations in terms of fields a bank analyst
actually recognises.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# Raw input columns the model sees. `fraud_type` and `is_fraud` are deliberately
# excluded: they are labels / analysis fields, not things known at scoring time.
NUMERIC_RAW = [
    "amount",
    "hour",
    "distance_from_home_km",
    "txn_count_1h",
    "txn_count_24h",
    "failed_attempts_24h",
    "account_age_days",
    "avg_amount_30d",
    "balance_before",
    "is_international",
    "is_new_device",
]
CATEGORICAL_RAW = ["channel", "merchant_category"]
RAW_FEATURES = NUMERIC_RAW + CATEGORICAL_RAW

ENGINEERED = ["amount_to_avg_ratio", "amount_to_balance_ratio", "is_night"]
NUMERIC_ALL = NUMERIC_RAW + ENGINEERED

# Human-friendly grouping used to summarise explanations for analysts.
FEATURE_GROUPS = {
    "Unusual amount": ["amount", "avg_amount_30d", "balance_before"],
    "Velocity": ["txn_count_1h", "txn_count_24h"],
    "Device & authentication": ["is_new_device", "failed_attempts_24h"],
    "Location": ["distance_from_home_km", "is_international"],
    "Time of day": ["hour"],
    "Channel & merchant": ["channel", "merchant_category"],
    "Account profile": ["account_age_days"],
}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive ratio / flag features from raw columns (pure, stateless)."""
    out = df[RAW_FEATURES].copy()
    out["amount_to_avg_ratio"] = out["amount"] / out["avg_amount_30d"].clip(lower=1.0)
    out["amount_to_balance_ratio"] = out["amount"] / out["balance_before"].clip(lower=1.0)
    out["is_night"] = ((out["hour"] >= 23) | (out["hour"] <= 5)).astype(int)
    return out


def _preprocessor(scale: bool) -> ColumnTransformer:
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [
            ("num", num, NUMERIC_ALL),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_RAW,
            ),
        ]
    )


def build_candidates(seed: int = 42) -> dict[str, Pipeline]:
    """Three scikit-learn models: one transparent baseline, two stronger ones."""
    fe = lambda: FunctionTransformer(add_features)  # noqa: E731
    return {
        "Logistic Regression": Pipeline(
            [
                ("fe", fe()),
                ("prep", _preprocessor(scale=True)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=3000, class_weight="balanced", C=1.0, random_state=seed
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("fe", fe()),
                ("prep", _preprocessor(scale=False)),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=250,
                        min_samples_leaf=3,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("fe", fe()),
                ("prep", _preprocessor(scale=False)),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        learning_rate=0.08,
                        max_iter=300,
                        class_weight="balanced",
                        early_stopping=True,
                        validation_fraction=0.15,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }
