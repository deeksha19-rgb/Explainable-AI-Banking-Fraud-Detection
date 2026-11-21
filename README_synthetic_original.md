# Explainable Bank Transaction Fraud Detection

An end-to-end fraud detection system built with **Python and scikit-learn**: it scores bank
transactions, explains *why* each suspicious one was flagged, and presents everything in a dashboard
with model-performance views.

> All data is **synthetic** (amounts in INR). No real customer data is used.

## Results (held-out, most recent 20% of time)

| Metric | Value |
|---|---|
| Model | Random Forest (selected on validation PR-AUC) |
| ROC-AUC / PR-AUC | 0.986 / 0.909 |
| Precision / Recall | 90.0% / 84.4% |
| Alerts raised | 241 of 12,000 transactions (2.0%) |
| Fraud value caught | 90.9% of ₹19.2 lakh |

Recall by fraud pattern: ATM cash-out 100%, card-testing bursts 100%, international misuse 100%,
account takeover 99%, **subtle fraud only 11%** (see Limitations).

## Quick start

```bash
pip install -r requirements.txt
python -m src.generate_data          # 60,000 synthetic transactions -> data/transactions.csv
python -m src.train                  # train, evaluate, explain  -> outputs/
streamlit run app.py                 # interactive dashboard
python -m src.build_static_dashboard # optional: single-file HTML dashboard -> outputs/dashboard.html
```

## How it works

```
generate_data.py  ->  train.py  ->  outputs/{model.joblib, metrics.json, scored_transactions.csv, ...}
                                          |
                          app.py (Streamlit)   build_static_dashboard.py (HTML)
```

**Data.** 60,000 transactions over 90 days, 2% fraud. Four fraud behaviours (card-testing bursts,
account takeover, international misuse, ATM cash-out) plus a "subtle" group that looks almost normal.
Some legitimate transactions are deliberately suspicious-looking (big purchases, genuine travel) so
the problem isn't trivially separable.

**Features.** Raw fields (amount, hour, distance from home, velocity counts, failed logins, new
device, international flag, balance, account age, channel, merchant category) plus engineered
`amount_to_avg_ratio`, `amount_to_balance_ratio` and `is_night`. Feature engineering lives inside the
scikit-learn `Pipeline`, so the model's inputs are the raw fields analysts recognise.
The `fraud_type` label is *never* given to the model.

**Validation.** Time-based split (70% train / 10% validation / 20% test) to mimic deployment and avoid
look-ahead leakage. Three models are compared (Logistic Regression, Random Forest, Gradient Boosting);
selection uses PR-AUC because accuracy is meaningless at 2% fraud. The alert threshold maximises
F2 on validation data (recall-weighted: a missed fraud costs more than an extra review).

**Explainability.**
- *Global*: permutation importance on the test set (`sklearn.inspection`).
- *Local*: Monte-Carlo **Shapley values** per transaction, implemented from scratch with only
  NumPy/pandas/scikit-learn (no SHAP dependency). Each field is switched from a "typical legitimate
  transaction" baseline to its real value in random orders; contributions sum exactly to
  `score - baseline score`. They are rolled up into analyst-friendly groups (Unusual amount, Velocity,
  Device & authentication, Location, Time of day, Channel & merchant, Account profile) with a plain-English
  reason, e.g. *"Amount ₹12,869 is 10.3x the customer's 30-day average and drains 69% of the balance"*.

## Dashboard (Streamlit)

| Tab | What it shows |
|---|---|
| Overview | KPIs, alerts vs confirmed fraud by day, alerts by channel, what drives alerts overall |
| Suspicious transactions | Filterable alert queue + per-transaction drill-down with reasons and contribution chart |
| Model performance | Model comparison, ROC, precision-recall, confusion matrix, recall by fraud pattern |
| Explainability | Global feature importance and method notes |
| Score a transaction | Enter a new transaction, get a live score and explanation |

## Project layout

```
app.py                         Streamlit dashboard
src/features.py                feature engineering + model pipelines
src/generate_data.py           synthetic data generator
src/train.py                   training, evaluation, explanations
src/explain.py                 Shapley explanations + reason text
src/build_static_dashboard.py  single-file HTML dashboard
```

## Limitations and next steps

- **Subtle fraud is mostly missed** (11% recall). It needs richer signals: merchant history, device
  fingerprints, per-customer behavioural baselines, graph features.
- Synthetic data is cleaner than reality; real deployments face concept drift, label delay and
  chargeback-based labels. Retraining cadence and drift monitoring would be needed.
- Scores are not calibrated probabilities; add `CalibratedClassifierCV` if they will drive cost-based rules.
- Next ideas: cost-sensitive threshold using per-transaction amount, SHAP comparison, REST scoring API,
  analyst feedback loop (confirm/dismiss alerts) to retrain.
