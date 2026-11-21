"""Streamlit dashboard for the PaySim-trained model.
Run after training: python -m streamlit run app_paysim.py
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs_paysim"
st.set_page_config(page_title="PaySim Fraud Lab", page_icon="🛡️", layout="wide")
st.title("Explainable AI · PaySim Fraud Detection")
st.caption("Research prototype using synthetic PaySim mobile-money transactions—not genuine bank transaction records.")

metrics_path = OUT / "metrics.json"
model_path = OUT / "model.joblib"
scored_path = OUT / "scored_transactions.csv"
if not all(p.exists() for p in (metrics_path, model_path, scored_path)):
    st.warning("PaySim model outputs are not available yet. Place the CSV in data/raw/ and run `python -m src.train_paysim` first.")
    st.stop()

metrics = json.loads(metrics_path.read_text())
model = joblib.load(model_path)
scored = pd.read_csv(scored_path)
threshold = float(metrics["threshold"])

tabs = st.tabs(["Overview", "Alert queue", "Model performance", "Explainability", "Score a transaction"])
with tabs[0]:
    m = metrics["test"]
    cols = st.columns(4)
    cols[0].metric("Test PR-AUC", f"{m['pr_auc']:.3f}")
    cols[1].metric("Precision", f"{m['precision']:.1%}")
    cols[2].metric("Recall", f"{m['recall']:.1%}")
    cols[3].metric("Flagged in test", f"{metrics['business']['alert_rate']:.1%}")
    st.write(f"**Selected model:** {metrics['selected_model']} · **Threshold:** {threshold:.5f}")
    st.write(f"Chronological test window: simulation steps {metrics['split']['test_step_min']:,}–{metrics['split']['test_step_max']:,}.")
    st.info("These are results from the held-out portion of PaySim, after you run the training script. They are not live bank performance estimates.")
    st.dataframe(pd.DataFrame(metrics["comparison"]), use_container_width=True)
with tabs[1]:
    bands = st.multiselect("Risk band", ["High", "Medium", "Low"], default=["High"])
    view = scored[scored.risk_band.isin(bands)].sort_values("risk_score", ascending=False)
    st.caption(f"Showing {min(len(view), 5000):,} of {len(view):,} matching held-out transactions.")
    st.dataframe(view.head(5000), use_container_width=True, hide_index=True)
    st.download_button("Download displayed alerts CSV", view.head(5000).to_csv(index=False), "paysim_alerts.csv", "text/csv")
with tabs[2]:
    m = metrics["test"]
    cm = pd.DataFrame([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], index=["Actual legitimate", "Actual fraud"], columns=["Predicted legitimate", "Predicted fraud"])
    st.subheader("Confusion matrix")
    st.dataframe(cm, use_container_width=True)
    st.subheader("Model comparison")
    comparison = pd.DataFrame(metrics["comparison"])
    st.dataframe(comparison, use_container_width=True, hide_index=True)
with tabs[3]:
    st.subheader("Global feature importance")
    imp_path = OUT / "feature_importance.csv"
    if imp_path.exists():
        imp = pd.read_csv(imp_path).sort_values("importance", ascending=True)
        st.plotly_chart(px.bar(imp, x="importance", y="feature", orientation="h", error_x="std", title="Permutation importance (sampled held-out transactions)"), use_container_width=True)
    st.markdown("**How to interpret:** permutation importance estimates how much held-out PR-AUC changes when one feature is shuffled. It is a global diagnostic, not proof of causation.")
    st.markdown("The primary model uses `step`, `type`, `amount`, plus derived `hour_of_day = step % 24`. It excludes the existing `isFlaggedFraud` rule flag, account IDs, and before/after balance fields.")
with tabs[4]:
    st.write("Enter a hypothetical transaction. This is a model score, not a real payment authorization.")
    step = st.number_input("Simulation step (hour index)", min_value=1, max_value=100000, value=1000, step=1)
    typ = st.selectbox("Transaction type", ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
    amount = st.number_input("Amount", min_value=0.0, value=10000.0, step=100.0)
    if st.button("Score transaction", type="primary"):
        x = pd.DataFrame([{"step": int(step), "type": typ, "amount": float(amount)}])
        score = float(model.predict_proba(x)[0, 1])
        st.metric("Model score", f"{score:.2%}")
        st.write("**Risk band:** " + ("High / review" if score >= threshold else "Below selected alert threshold"))
        # Approximate one-feature-at-a-time perturbations relative to a typical legitimate baseline.
        baseline_path = OUT / "baseline.json"
        base = json.loads(baseline_path.read_text())
        base_df = pd.DataFrame([base])
        base_score = float(model.predict_proba(base_df)[0, 1])
        st.caption(f"Typical-legitimate baseline score: {base_score:.2%}. The comparisons below are approximate, not exact SHAP values.")
        deltas = []
        for feat in ["step", "type", "amount"]:
            perturbed = base_df.copy()
            perturbed.loc[0, feat] = x.loc[0, feat]
            s = float(model.predict_proba(perturbed)[0, 1])
            deltas.append({"feature": feat, "score change vs baseline": s - base_score})
        st.dataframe(pd.DataFrame(deltas).sort_values("score change vs baseline", ascending=False), use_container_width=True, hide_index=True)
