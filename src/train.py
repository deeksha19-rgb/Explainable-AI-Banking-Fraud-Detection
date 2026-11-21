"""Train, evaluate and explain the fraud model.

Run:  python -m src.train

Outputs (in ./outputs):
    model.joblib            fitted scikit-learn pipeline (best model)
    baseline.json           'typical legitimate transaction' used for explanations
    metrics.json            model comparison, threshold, confusion matrix, curves, ...
    scored_transactions.csv held-out test transactions with score, band, reasons
    feature_importance.csv  global permutation importance
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .explain import group_contributions, make_baseline, shapley_contributions, top_reasons
from .features import FEATURE_GROUPS, RAW_FEATURES, build_candidates

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "transactions.csv"
OUT = ROOT / "outputs"
SEED = 42


def pick_threshold(y, score, beta: float = 2.0) -> float:
    """Threshold that maximises F-beta on validation data (beta=2 favours recall,
    because a missed fraud costs far more than an extra manual review)."""
    prec, rec, thr = precision_recall_curve(y, score)
    f = (1 + beta**2) * prec[:-1] * rec[:-1] / np.maximum(beta**2 * prec[:-1] + rec[:-1], 1e-12)
    return float(thr[int(np.argmax(f))])


def evaluate(y, score, thr) -> dict:
    pred = (score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred)),
        "f2": float(fbeta_score(y, pred, beta=2)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(DATA, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Time-based split: train on the past, validate, then test on the most recent
    # period - this mimics deployment and avoids look-ahead leakage.
    n = len(df)
    i_val, i_test = int(n * 0.70), int(n * 0.80)
    train, val, test = df.iloc[:i_val], df.iloc[i_val:i_test], df.iloc[i_test:].copy()
    print(f"train={len(train):,}  val={len(val):,}  test={len(test):,}  "
          f"(fraud rate train {train.is_fraud.mean():.2%}, test {test.is_fraud.mean():.2%})")

    # ---- model comparison (selection on validation PR-AUC) ------------------
    comparison, fitted = [], {}
    for name, pipe in build_candidates(SEED).items():
        t0 = time.time()
        pipe.fit(train[RAW_FEATURES], train.is_fraud)
        s_val = pipe.predict_proba(val[RAW_FEATURES])[:, 1]
        thr = pick_threshold(val.is_fraud, s_val)
        s_test = pipe.predict_proba(test[RAW_FEATURES])[:, 1]
        row = {"model": name, "threshold": thr,
               "val_pr_auc": float(average_precision_score(val.is_fraud, s_val)),
               **{f"test_{k}": v for k, v in evaluate(test.is_fraud, s_test, thr).items()}}
        comparison.append(row)
        fitted[name] = (pipe, thr, s_test)
        print(f"{name:20s} val PR-AUC {row['val_pr_auc']:.3f} | test PR-AUC {row['test_pr_auc']:.3f} "
              f"recall {row['test_recall']:.3f} precision {row['test_precision']:.3f} ({time.time()-t0:.0f}s)")

    best = max(comparison, key=lambda r: r["val_pr_auc"])["model"]
    model, thr, score = fitted[best]
    print(f"\nSelected model: {best}  threshold={thr:.3f}")

    # ---- final test metrics, curves, business impact ---------------------
    y = test.is_fraud.to_numpy()
    final = evaluate(y, score, thr)
    fpr, tpr, _ = roc_curve(y, score)
    prec, rec, _ = precision_recall_curve(y, score)
    def thin(a, target=200):
        """Downsample a curve for the dashboard but always keep its final point."""
        idx = list(range(0, len(a), max(1, len(a) // target)))
        if idx[-1] != len(a) - 1:
            idx.append(len(a) - 1)
        return idx

    roc_idx, pr_idx = thin(fpr), thin(prec)

    flagged = score >= thr
    fraud_amt = float(test.loc[y == 1, "amount"].sum())
    caught_amt = float(test.loc[flagged & (y == 1), "amount"].sum())
    by_type = (
        test[test.is_fraud == 1].assign(caught=flagged[test.is_fraud.to_numpy() == 1])
        .groupby("fraud_type").agg(frauds=("caught", "size"), caught=("caught", "sum")).reset_index()
    )
    by_type["recall"] = by_type["caught"] / by_type["frauds"]

    # ---- bands, explanations ------------------------------------------------
    test["risk_score"] = score
    test["risk_band"] = np.where(score >= thr, "High", np.where(score >= thr * 0.5, "Medium", "Low"))
    test["flagged"] = flagged.astype(int)

    baseline = make_baseline(train[train.is_fraud == 0])
    explain_mask = (test["risk_band"] != "Low").to_numpy()
    X_exp = test.loc[explain_mask, RAW_FEATURES]
    print(f"Explaining {len(X_exp):,} medium/high-risk transactions (Monte-Carlo Shapley)...")
    t0 = time.time()
    contrib = shapley_contributions(model, X_exp, baseline, n_perm=24, seed=SEED)
    groups = group_contributions(contrib)
    print(f"  done in {time.time()-t0:.0f}s")

    test["top_reasons"] = ""
    test.loc[explain_mask, "top_reasons"] = top_reasons(X_exp, groups)
    for g in FEATURE_GROUPS:
        test[f"contrib_{g}"] = np.nan
        test.loc[explain_mask, f"contrib_{g}"] = groups[g].to_numpy().round(4)
    test["baseline_score"] = float(model.predict_proba(pd.DataFrame([baseline])[RAW_FEATURES])[0, 1])

    # ---- global importance ---------------------------------------------------
    pi = permutation_importance(
        model, test[RAW_FEATURES], y, scoring="average_precision",
        n_repeats=5, random_state=SEED, n_jobs=-1,
    )
    imp = (pd.DataFrame({"feature": RAW_FEATURES, "importance": pi.importances_mean,
                         "std": pi.importances_std})
           .sort_values("importance", ascending=False))
    group_of = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
    imp["group"] = imp["feature"].map(group_of)

    # ---- daily alert volume --------------------------------------------------
    daily = (test.assign(date=test.timestamp.dt.date.astype(str))
             .groupby("date").agg(transactions=("flagged", "size"), alerts=("flagged", "sum"),
                                  confirmed=("is_fraud", "sum")).reset_index())

    metrics = {
        "selected_model": best,
        "threshold": thr,
        "split": {"train": len(train), "val": len(val), "test": len(test),
                  "test_start": str(test.timestamp.min()), "test_end": str(test.timestamp.max())},
        "comparison": comparison,
        "test": final,
        "business": {
            "fraud_amount": fraud_amt, "caught_amount": caught_amt,
            "caught_share": caught_amt / fraud_amt if fraud_amt else 0.0,
            "alerts": int(flagged.sum()), "alert_rate": float(flagged.mean()),
            "test_transactions": int(len(test)),
        },
        "roc": {"fpr": fpr[roc_idx].round(4).tolist(), "tpr": tpr[roc_idx].round(4).tolist()},
        "pr": {"precision": prec[pr_idx].round(4).tolist(), "recall": rec[pr_idx].round(4).tolist()},
        "recall_by_fraud_type": by_type.round(4).to_dict(orient="records"),
        "daily": daily.to_dict(orient="records"),
    }

    joblib.dump(model, OUT / "model.joblib")
    (OUT / "baseline.json").write_text(json.dumps(baseline.to_dict(), indent=2))
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    imp.to_csv(OUT / "feature_importance.csv", index=False)
    cols = ["transaction_id", "customer_id", "timestamp", "amount", "channel", "merchant_category",
            "hour", "distance_from_home_km", "is_international", "is_new_device", "txn_count_1h",
            "txn_count_24h", "failed_attempts_24h", "avg_amount_30d", "balance_before",
            "account_age_days", "risk_score", "risk_band", "flagged", "top_reasons", "baseline_score",
            *[f"contrib_{g}" for g in FEATURE_GROUPS], "is_fraud", "fraud_type"]
    test[cols].sort_values("risk_score", ascending=False).to_csv(OUT / "scored_transactions.csv", index=False)

    print("\n=== TEST RESULTS (most recent 20% of time) ===")
    print(f"ROC-AUC {final['roc_auc']:.3f} | PR-AUC {final['pr_auc']:.3f} | "
          f"Precision {final['precision']:.3f} | Recall {final['recall']:.3f} | F2 {final['f2']:.3f}")
    print(f"TP {final['tp']}  FP {final['fp']}  FN {final['fn']}  TN {final['tn']}")
    print(f"Fraud value caught: {metrics['business']['caught_share']:.1%} of ₹{fraud_amt:,.0f}")
    print("\nTop global drivers:\n" + imp.head(6)[["feature", "importance"]].to_string(index=False))
    print("\nRecall by fraud type:\n" + by_type.to_string(index=False))


if __name__ == "__main__":
    main()
