"""Train/evaluate the PaySim version without modifying the original dataset or outputs.

Run from the project root: python -m src.train_paysim
Expected input: data/raw/PS_20174392719_1491204439457_log.csv
"""
from __future__ import annotations
import json, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, confusion_matrix, fbeta_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score)
from .paysim_features import RAW_FEATURES, FEATURE_GROUPS, add_features, build_candidates

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT = ROOT / "outputs_paysim"
SEED = 42
TARGET_COLUMNS = ["step", "type", "amount", "isFraud"]

def locate_csv():
    preferred = RAW_DIR / "PS_20174392719_1491204439457_log.csv"
    if preferred.exists(): return preferred
    matches = sorted(RAW_DIR.glob("*.csv"))
    if len(matches) == 1: return matches[0]
    raise FileNotFoundError(f"Put the original PaySim CSV in {RAW_DIR}. Expected the Kaggle CSV, or exactly one CSV file there.")

def choose_threshold(y, score, beta=2.0):
    p, r, thresholds = precision_recall_curve(y, score)
    if len(thresholds) == 0: return 0.5
    f = (1 + beta**2) * p[:-1] * r[:-1] / np.maximum(beta**2 * p[:-1] + r[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f))])

def metrics(y, score, threshold):
    pred = (score >= threshold).astype("int8")
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"roc_auc": float(roc_auc_score(y, score)), "pr_auc": float(average_precision_score(y, score)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f2": float(fbeta_score(y, pred, beta=2, zero_division=0)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

def main():
    OUT.mkdir(exist_ok=True)
    csv_path = locate_csv()
    print(f"Reading selected columns from: {csv_path}")
    # Read only four needed columns to reduce memory use; the original CSV remains untouched.
    df = pd.read_csv(csv_path, usecols=TARGET_COLUMNS,
                     dtype={"step": "int32", "type": "category", "amount": "float32", "isFraud": "int8"})
    if not set(df["isFraud"].dropna().unique()).issubset({0, 1}):
        raise ValueError("isFraud must contain only 0/1 labels.")
    # Dataset is usually ordered by step; stable sort makes the chronological split explicit.
    df = df.sort_values("step", kind="stable").reset_index(drop=True)
    n = len(df)
    unique_steps = np.sort(df.step.unique())
    step_train_end = int(unique_steps[min(int(len(unique_steps) * .70), len(unique_steps) - 1)])
    step_val_end = int(unique_steps[min(int(len(unique_steps) * .80), len(unique_steps) - 1)])
    # Split by whole simulation steps so transactions from the same hour are not split across sets.
    train = df[df.step <= step_train_end].copy()
    val = df[(df.step > step_train_end) & (df.step <= step_val_end)].copy()
    test = df[df.step > step_val_end].copy()
    print(f"Rows: {n:,}; fraud rate: {df.isFraud.mean():.5%}")
    print(f"Chronological split: train={len(train):,}, validation={len(val):,}, test={len(test):,}")

    # Keep every fraud row in training but cap legitimate rows at 30:1 for a practical
    # laptop-friendly first run. Weights correct for the legitimate-row sampling rate.
    fraud_train = train[train.isFraud == 1]
    legit_train = train[train.isFraud == 0]
    max_legit = min(len(legit_train), max(len(fraud_train) * 30, 10000))
    if max_legit < len(legit_train):
        legit_sample = legit_train.sample(n=max_legit, random_state=SEED)
    else:
        legit_sample = legit_train
    sampled = pd.concat([fraud_train, legit_sample]).sort_values("step", kind="stable")
    neg_weight = len(legit_train) / max(len(legit_sample), 1)
    sample_weight = np.where(sampled.isFraud.to_numpy() == 0, neg_weight, 1.0)
    print(f"Training subset: {len(sampled):,} rows (all {len(fraud_train):,} training fraud rows + sampled legitimate rows)")

    Xtr, ytr = sampled[RAW_FEATURES], sampled.isFraud.astype(int)
    Xv, yv = val[RAW_FEATURES], val.isFraud.astype(int)
    Xt, yt = test[RAW_FEATURES], test.isFraud.astype(int)
    results, fitted = [], {}
    for name, model in build_candidates(SEED).items():
        start = time.time(); print(f"Fitting {name}...")
        # Importance weights restore the negative class's original sampling frequency.
        model.fit(Xtr, ytr, model__sample_weight=sample_weight)
        val_score = model.predict_proba(Xv)[:, 1]
        threshold = choose_threshold(yv, val_score)
        test_score = model.predict_proba(Xt)[:, 1]
        row = {"model": name, "threshold": threshold,
               "validation_pr_auc": float(average_precision_score(yv, val_score)),
               **{f"test_{k}": v for k, v in metrics(yt, test_score, threshold).items()}}
        results.append(row); fitted[name] = (model, threshold, test_score)
        print(f"  validation PR-AUC={row['validation_pr_auc']:.5f}; test PR-AUC={row['test_pr_auc']:.5f}; {time.time()-start:.0f}s")

    best_name = max(results, key=lambda r: r["validation_pr_auc"])["model"]
    model, threshold, test_score = fitted[best_name]
    final = metrics(yt, test_score, threshold)
    y = yt.to_numpy(); flagged = test_score >= threshold
    test_out = test.copy()
    test_out["risk_score"] = test_score
    test_out["risk_band"] = np.where(test_score >= threshold, "High", np.where(test_score >= threshold * .5, "Medium", "Low"))
    test_out["flagged"] = flagged.astype("int8")
    test_out.rename(columns={"isFraud": "is_fraud"}, inplace=True)
    # Save a compact full held-out scoring file; do not copy account identifiers into model inputs.
    test_out.to_csv(OUT / "scored_transactions.csv", index=False)

    # Global permutation importance on a bounded sample to keep runtime manageable.
    rng = np.random.default_rng(SEED)
    take = min(50000, len(test))
    ix = rng.choice(len(test), size=take, replace=False) if take else np.array([], dtype=int)
    X_imp = Xt.iloc[ix]; y_imp = yt.iloc[ix]
    pi = permutation_importance(model, X_imp, y_imp, scoring="average_precision", n_repeats=3, random_state=SEED, n_jobs=1)
    imp = pd.DataFrame({"feature": RAW_FEATURES, "importance": pi.importances_mean, "std": pi.importances_std})
    imp["group"] = imp.feature.map({f: g for g, fields in FEATURE_GROUPS.items() for f in fields})
    imp.sort_values("importance", ascending=False).to_csv(OUT / "feature_importance.csv", index=False)

    metrics_obj = {
        "dataset": "PaySim Synthetic Financial Datasets for Fraud Detection (Kaggle)",
        "data_note": "Synthetic mobile-money transactions generated by PaySim; not genuine bank transaction records.",
        "input_file": csv_path.name,
        "target": "isFraud",
        "excluded_from_primary_model": ["isFlaggedFraud", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "nameOrig", "nameDest"],
        "features": RAW_FEATURES + ["hour_of_day (derived as step % 24)"],
        "training_sampling": {"method": "all training fraud rows + up to 30 legitimate rows per fraud row", "sampled_train_rows": int(len(sampled)), "full_train_rows": int(len(train)), "legitimate_weight": float(neg_weight)},
        "split": {"method": "chronological by whole simulation step (no step crosses split boundaries)", "train": len(train), "validation": len(val), "test": len(test), "train_step_max": int(train.step.max()), "validation_step_min": int(val.step.min()), "validation_step_max": int(val.step.max()), "test_step_min": int(test.step.min()), "test_step_max": int(test.step.max())},
        "selected_model": best_name, "threshold": threshold, "comparison": results, "test": final,
        "business": {"alerts": int(flagged.sum()), "alert_rate": float(flagged.mean()), "test_transactions": int(len(test)),
                     "fraud_transactions": int((y == 1).sum()), "fraud_value": float(test.loc[y == 1, "amount"].sum()),
                     "caught_fraud_value": float(test.loc[flagged & (y == 1), "amount"].sum())},
        "importance_sample_rows": int(take),
    }
    joblib.dump(model, OUT / "model.joblib")
    (OUT / "metrics.json").write_text(json.dumps(metrics_obj, indent=2))
    (OUT / "baseline.json").write_text(json.dumps({"step": int(train[train.isFraud == 0].step.median()), "type": str(train[train.isFraud == 0]["type"].mode().iloc[0]), "amount": float(train[train.isFraud == 0].amount.median())}, indent=2))
    print("\n=== HELD-OUT TEST RESULTS (PaySim) ===")
    print(f"Selected model: {best_name}")
    print(f"ROC-AUC {final['roc_auc']:.4f} | PR-AUC {final['pr_auc']:.4f} | Precision {final['precision']:.3f} | Recall {final['recall']:.3f}")
    print(f"TP {final['tp']} FP {final['fp']} FN {final['fn']} TN {final['tn']}")
    print(f"PaySim outputs saved to: {OUT}")

if __name__ == "__main__": main()
