"""Feature definitions for the PaySim mobile-money fraud dataset.

Only fields available before the transaction outcome are used by the primary
model. Balance fields and isFlaggedFraud are intentionally excluded.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

RAW_FEATURES = ["step", "type", "amount"]
NUMERIC_FEATURES = ["step", "amount", "hour_of_day"]
CATEGORICAL_FEATURES = ["type"]
FEATURE_GROUPS = {
    "Transaction amount": ["amount"],
    "Simulation time": ["step", "hour_of_day"],
    "Transaction type": ["type"],
}

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df[RAW_FEATURES].copy()
    out["step"] = pd.to_numeric(out["step"], errors="coerce")
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    out["type"] = out["type"].astype("string").fillna("UNKNOWN").astype(str)
    # PaySim step is an hourly simulation index, not a wall-clock timestamp.
    out["hour_of_day"] = out["step"] % 24
    return out

class PaySimFeatureTransformer(BaseEstimator, TransformerMixin):
    """Small sklearn-compatible transformer without relying on pickle lambdas."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return add_features(X)
    def get_feature_names_out(self, input_features=None):
        return NUMERIC_FEATURES + CATEGORICAL_FEATURES

def _preprocessor(scale_numeric=True):
    numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler() if scale_numeric else "passthrough")
    categorical = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])

def build_candidates(seed=42):
    return {
        "Logistic Regression": Pipeline([
            ("features", PaySimFeatureTransformer()),
            ("prep", _preprocessor(True)),
            ("model", LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed)),
        ]),
        "Histogram Gradient Boosting": Pipeline([
            ("features", PaySimFeatureTransformer()),
            ("prep", _preprocessor(False)),
            ("model", HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, max_leaf_nodes=31, random_state=seed)),
        ]),
    }
