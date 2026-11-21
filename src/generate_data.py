"""Generate a realistic *synthetic* bank transaction dataset (amounts in INR).

No real customer data is used. Fraud is injected as four behavioural patterns
plus a "subtle" group that looks almost normal, and a slice of legitimate
transactions is made to look suspicious (big purchases, genuine travel) so the
problem is not trivially separable.

Run:  python -m src.generate_data
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "transactions.csv"

CHANNELS = ["POS", "Online", "UPI", "ATM"]
CATEGORIES = [
    "Grocery", "Fuel", "Restaurants", "Electronics", "Travel",
    "Utilities", "Entertainment", "Jewellery", "Money Transfer", "Gaming",
]
CAT_P_LEGIT = [0.24, 0.12, 0.16, 0.07, 0.06, 0.12, 0.08, 0.02, 0.09, 0.04]


def generate(n: int = 60_000, n_customers: int = 3_000, fraud_rate: float = 0.02, seed: int = 42):
    rng = np.random.default_rng(seed)

    # ---- customers -------------------------------------------------------
    cust = pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(n_customers)],
            "avg_amount_30d": rng.lognormal(7.3, 0.6, n_customers).round(2),
            "account_age_days": rng.integers(30, 4000, n_customers),
        }
    )
    c = cust.iloc[rng.integers(0, n_customers, n)].reset_index(drop=True)
    avg = c["avg_amount_30d"].to_numpy()

    is_fraud = np.zeros(n, dtype=bool)
    is_fraud[: int(n * fraud_rate)] = True
    rng.shuffle(is_fraud)

    # ---- legitimate behaviour (defaults for every row) -------------------
    hour_p = np.array([1, .5, .3, .3, .5, 1, 2, 4, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 7, 5, 3, 2], float)
    hour = rng.choice(24, n, p=hour_p / hour_p.sum())
    channel = rng.choice(CHANNELS, n, p=[0.35, 0.25, 0.32, 0.08])
    category = rng.choice(CATEGORIES, n, p=CAT_P_LEGIT)
    amount = avg * rng.lognormal(0, 0.5, n)
    balance = avg * rng.lognormal(2.2, 0.7, n)
    distance = rng.exponential(6, n)
    international = (rng.random(n) < 0.012).astype(int)
    new_device = (rng.random(n) < 0.04).astype(int)
    c1h = rng.poisson(0.25, n)
    c24h = c1h + rng.poisson(2.0, n)
    failed = rng.poisson(0.08, n)

    # legit "look-alikes": big purchases and genuine travel
    big = (~is_fraud) & (rng.random(n) < 0.03)
    amount[big] = avg[big] * rng.uniform(3, 8, big.sum())
    category[big] = rng.choice(["Electronics", "Jewellery", "Travel"], big.sum())
    trav = (~is_fraud) & (rng.random(n) < 0.015)
    distance[trav] = rng.uniform(150, 1500, trav.sum())
    international[trav] = (rng.random(trav.sum()) < 0.5).astype(int)
    new_device[trav] = (rng.random(trav.sum()) < 0.3).astype(int)

    # ---- fraud patterns ---------------------------------------------------
    fraud_type = np.array([""] * n, dtype=object)
    idx = np.flatnonzero(is_fraud)
    types = rng.choice(
        ["card_testing_burst", "account_takeover", "international_misuse", "atm_cashout", "subtle"],
        len(idx), p=[0.22, 0.25, 0.20, 0.13, 0.20],
    )
    fraud_type[idx] = types

    def rows(kind):
        return idx[types == kind]

    r = rows("card_testing_burst")  # rapid small/medium online purchases on a new device
    channel[r] = rng.choice(["Online", "UPI"], len(r), p=[0.8, 0.2])
    category[r] = rng.choice(["Gaming", "Electronics", "Entertainment", "Money Transfer"], len(r))
    c1h[r] = rng.integers(3, 9, len(r)); c24h[r] = c1h[r] + rng.integers(2, 10, len(r))
    new_device[r] = (rng.random(len(r)) < 0.75).astype(int)
    amount[r] = avg[r] * rng.uniform(0.3, 2.5, len(r))
    hour[r] = rng.choice(24, len(r), p=np.roll(hour_p, 8) / hour_p.sum())

    r = rows("account_takeover")  # new device, failed logins, large transfer draining balance
    channel[r] = rng.choice(["UPI", "Online"], len(r))
    category[r] = "Money Transfer"
    new_device[r] = (rng.random(len(r)) < 0.9).astype(int)
    failed[r] = rng.integers(1, 7, len(r))
    balance[r] = avg[r] * rng.lognormal(2.0, 0.5, len(r))
    amount[r] = balance[r] * rng.uniform(0.6, 0.98, len(r))
    hour[r] = rng.choice(24, len(r), p=np.roll(hour_p, 6) / hour_p.sum())

    r = rows("international_misuse")  # card used abroad, far from home
    international[r] = (rng.random(len(r)) < 0.85).astype(int)
    distance[r] = rng.uniform(300, 9000, len(r))
    channel[r] = rng.choice(["POS", "Online"], len(r))
    category[r] = rng.choice(["Electronics", "Jewellery", "Travel", "Gaming"], len(r))
    amount[r] = avg[r] * rng.uniform(2, 9, len(r))
    new_device[r] = (rng.random(len(r)) < 0.35).astype(int)

    r = rows("atm_cashout")  # large night-time ATM withdrawals far from home
    channel[r] = "ATM"; category[r] = "Money Transfer"
    distance[r] = rng.uniform(40, 2000, len(r))
    hour[r] = rng.choice([22, 23, 0, 1, 2, 3, 4], len(r))
    amount[r] = np.minimum(avg[r] * rng.uniform(3, 10, len(r)), 40_000)
    c1h[r] = rng.integers(1, 5, len(r)); c24h[r] = c1h[r] + rng.integers(0, 5, len(r))

    r = rows("subtle")  # almost normal: only mildly elevated
    channel[r] = rng.choice(["Online", "UPI", "POS"], len(r))
    amount[r] = avg[r] * rng.uniform(1.2, 3.5, len(r))
    new_device[r] = (rng.random(len(r)) < 0.3).astype(int)
    c1h[r] = rng.poisson(1.0, len(r)); c24h[r] = c1h[r] + rng.poisson(2.5, len(r))

    # ---- finalise ----------------------------------------------------------
    amount = np.minimum(amount, balance * 0.98).round(2)
    balance = balance.round(2)
    c24h = np.maximum(c24h, c1h)

    start = pd.Timestamp("2026-07-01")
    ts = (
        start
        + pd.to_timedelta(rng.integers(0, 90, n), unit="D")
        + pd.to_timedelta(hour, unit="h")
        + pd.to_timedelta(rng.integers(0, 60, n), unit="m")
        + pd.to_timedelta(rng.integers(0, 60, n), unit="s")
    )

    df = pd.DataFrame(
        {
            "customer_id": c["customer_id"],
            "timestamp": ts,
            "amount": amount,
            "channel": channel,
            "merchant_category": category,
            "hour": hour,
            "distance_from_home_km": distance.round(1),
            "is_international": international,
            "is_new_device": new_device,
            "txn_count_1h": c1h,
            "txn_count_24h": c24h,
            "failed_attempts_24h": failed,
            "avg_amount_30d": avg,
            "account_age_days": c["account_age_days"],
            "balance_before": balance,
            "fraud_type": fraud_type,  # analysis only - never a model input
            "is_fraud": is_fraud.astype(int),
        }
    )
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.insert(0, "transaction_id", [f"TXN{i:07d}" for i in range(1, len(df) + 1)])
    return df


if __name__ == "__main__":
    df = generate()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Wrote {len(df):,} transactions to {DATA_PATH}")
    print(f"Fraud rate: {df.is_fraud.mean():.2%}")
    print(df[df.is_fraud == 1].fraud_type.value_counts().to_string())
