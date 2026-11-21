"""Model-agnostic explainability using only numpy / pandas / scikit-learn.

Two views:

1. Global  - permutation importance (sklearn.inspection) on the raw input fields.
2. Local   - Monte-Carlo Shapley values per transaction. Each field is switched
             from a "typical legitimate transaction" baseline to the real value
             in random orders; the average change in fraud probability is that
             field's contribution. Contributions add up exactly to
             score(transaction) - score(baseline), and are then rolled up into
             analyst-friendly groups ("Velocity", "Location", ...) with a
             plain-English reason for each.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import CATEGORICAL_RAW, FEATURE_GROUPS, NUMERIC_RAW, RAW_FEATURES


def make_baseline(train_legit: pd.DataFrame) -> pd.Series:
    """A 'typical legitimate transaction': median of numerics, mode of categoricals."""
    base = {}
    for col in NUMERIC_RAW:
        base[col] = float(train_legit[col].median())
    for col in CATEGORICAL_RAW:
        base[col] = train_legit[col].mode().iloc[0]
    return pd.Series(base)[RAW_FEATURES]


def shapley_contributions(
    model, X: pd.DataFrame, baseline: pd.Series, n_perm: int = 30, seed: int = 0
) -> pd.DataFrame:
    """Monte-Carlo Shapley contributions (in probability points) per raw feature."""
    rng = np.random.default_rng(seed)
    X = X[RAW_FEATURES].reset_index(drop=True)
    n = len(X)
    contrib = pd.DataFrame(0.0, index=range(n), columns=RAW_FEATURES)
    base_df = pd.DataFrame([baseline.to_dict()] * n)[RAW_FEATURES]
    base_score = float(model.predict_proba(base_df.iloc[:1])[0, 1])

    for _ in range(n_perm):
        order = rng.permutation(RAW_FEATURES)
        cur = base_df.copy()
        prev = np.full(n, base_score)
        for feat in order:
            cur[feat] = X[feat].to_numpy()
            pred = model.predict_proba(cur)[:, 1]
            contrib[feat] += pred - prev
            prev = pred
    return contrib / n_perm


def group_contributions(contrib: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({g: contrib[cols].sum(axis=1) for g, cols in FEATURE_GROUPS.items()})


def _reason_text(group: str, row: pd.Series) -> str:
    if group == "Unusual amount":
        ratio = row["amount"] / max(row["avg_amount_30d"], 1.0)
        text = f"Amount ₹{row['amount']:,.0f} is {ratio:.1f}x the customer's 30-day average"
        drain = row["amount"] / max(row["balance_before"], 1.0)
        if drain >= 0.6:
            text += f" and drains {drain:.0%} of the balance"
        return text
    if group == "Velocity":
        return f"{int(row['txn_count_1h'])} transactions in the last hour ({int(row['txn_count_24h'])} in 24h)"
    if group == "Device & authentication":
        bits = []
        if row["is_new_device"] >= 0.5:
            bits.append("new/unrecognised device")
        if row["failed_attempts_24h"] >= 1:
            bits.append(f"{int(row['failed_attempts_24h'])} failed login attempts in 24h")
        return ", ".join(bits).capitalize() if bits else "Device / authentication pattern"
    if group == "Location":
        bits = [f"{row['distance_from_home_km']:,.0f} km from home"]
        if row["is_international"] >= 0.5:
            bits.append("international transaction")
        return ", ".join(bits).capitalize()
    if group == "Time of day":
        return f"Transaction at {int(row['hour']):02d}:00 (unusual hour)"
    if group == "Channel & merchant":
        return f"{row['channel']} payment at a {row['merchant_category']} merchant"
    if group == "Account profile":
        return f"Relatively new account ({int(row['account_age_days'])} days old)"
    return group


def top_reasons(X: pd.DataFrame, groups: pd.DataFrame, k: int = 3, min_contrib: float = 0.02):
    """Return list[str] of the top-k positive risk drivers per transaction."""
    X = X.reset_index(drop=True)
    out = []
    for i in range(len(X)):
        g = groups.iloc[i].sort_values(ascending=False)
        picked = [name for name, v in g.items() if v >= min_contrib][:k]
        out.append(" | ".join(_reason_text(name, X.iloc[i]) for name in picked))
    return out
