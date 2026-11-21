"""Explainable Bank Transaction Fraud Detection - Streamlit dashboard.

Run:  python -m streamlit run app.py
(first run `python -m src.generate_data` and `python -m src.train`)
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.explain import group_contributions, shapley_contributions, top_reasons  # noqa: E402
from src.features import FEATURE_GROUPS, RAW_FEATURES  # noqa: E402
from src.generate_data import CATEGORIES, CHANNELS  # noqa: E402

OUT = ROOT / "outputs"

# One-hue theme: deep teal surfaces, a single mint accent, tints of the same hue for everything else.
BG, CARD, CARD2, LINE = "#062a30", "#0b3a42", "#104a54", "#1b6470"
INK, INK2 = "#e6fbf7", "#9cc7c4"
ACCENT, SOFT = "#5ff0d3", "#4f8f98"
GRID = "rgba(156,199,196,0.16)"

st.set_page_config(page_title="Fraud Detection Console", page_icon="🛡️", layout="wide")

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
:root {{ --bg:{BG}; --card:{CARD}; --card2:{CARD2}; --line:{LINE}; --ink:{INK}; --ink2:{INK2}; --accent:{ACCENT}; --soft:{SOFT}; }}

.stApp {{ background: linear-gradient(180deg, #08333a 0%, {BG} 38%, #04191d 100%); }}
.stApp, .stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp button,
.stApp input, .stApp textarea, .stApp [data-baseweb="select"] {{ font-family: 'Manrope', system-ui, sans-serif; }}
#MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; height: 0; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 4rem; max-width: 1180px; }}

h2, h3 {{ color: var(--ink); font-weight: 700; letter-spacing: -0.01em; }}
[data-testid="stCaptionContainer"], .stApp small {{ color: var(--ink2); }}

/* hero */
.hero {{ display: grid; grid-template-columns: 1.25fr 1fr; gap: 28px; align-items: stretch;
  background: radial-gradient(900px 300px at 8% -10%, rgba(95,240,211,0.16), transparent 60%), var(--card);
  border: 1px solid var(--line); border-radius: 26px; padding: 34px 36px; margin-bottom: 22px; }}
.hero h1 {{ font-size: clamp(2.1rem, 4.2vw, 3.2rem); line-height: 1.04; font-weight: 800; letter-spacing: -0.03em;
  margin: 0 0 14px 0; color: var(--ink); padding: 0; }}
.hero p.lead {{ font-size: 1.08rem; line-height: 1.55; color: var(--ink2); max-width: 34ch; margin: 0 0 18px 0; }}
.hero p.meta {{ font-size: 0.85rem; color: var(--soft); margin: 0; line-height: 1.6; }}
.alert {{ background: #07313a; border: 1px solid var(--line); border-radius: 18px; padding: 22px 24px; }}
.alert .top {{ display: flex; justify-content: space-between; align-items: center; color: var(--ink2); font-size: 0.9rem; }}
.alert .amt {{ font-size: 2.3rem; font-weight: 800; letter-spacing: -0.02em; margin: 6px 0 2px; color: var(--ink); }}
.alert .sub {{ color: var(--ink2); font-size: 0.9rem; margin-bottom: 14px; }}
.chip {{ background: var(--accent); color: #032a2e; font-weight: 800; border-radius: 999px; padding: 3px 12px; font-size: 0.85rem; }}
.alert ul {{ margin: 0; padding-left: 18px; color: var(--ink); font-size: 0.93rem; line-height: 1.5; }}
.alert li {{ margin-bottom: 6px; }}
.alert li::marker {{ color: var(--accent); }}
@media (max-width: 820px) {{ .hero {{ grid-template-columns: 1fr; padding: 24px 20px; }} }}

/* tabs */
[data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid var(--line); }}
button[data-baseweb="tab"] {{ background: transparent; border-radius: 10px 10px 0 0; padding: 10px 16px; color: var(--ink2); font-weight: 600; }}
button[data-baseweb="tab"]:hover {{ color: var(--ink); background: rgba(95,240,211,0.06); }}
button[data-baseweb="tab"][aria-selected="true"] {{ color: var(--accent); }}
[data-baseweb="tab-highlight"] {{ background-color: var(--accent) !important; height: 3px !important; }}
[data-baseweb="tab-border"] {{ background-color: transparent !important; }}

/* metrics */
[data-testid="stMetric"] {{ background: var(--card); border: 1px solid var(--line); border-radius: 16px; padding: 16px 18px; min-height: 124px; }}
[data-testid="stMetricDelta"] > div {{ white-space: normal; overflow: visible; text-overflow: clip; }}
[data-testid="stMetricLabel"] p {{ color: var(--ink2); font-weight: 600; }}
[data-testid="stMetricValue"] {{ font-weight: 800; letter-spacing: -0.02em; color: var(--ink); }}
[data-testid="stMetricDelta"] {{ color: var(--accent) !important; }}
[data-testid="stMetricDelta"] svg {{ display: none; }}

/* chart / detail cards */
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"] [data-testid="stPlotlyChart"]),
div[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 18px; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ background: var(--card); border-color: var(--line) !important; }}

/* inputs, buttons, tables */
[data-baseweb="select"] > div, [data-baseweb="input"], [data-testid="stNumberInputContainer"] {{ background: #07313a; border-color: var(--line); }}
span[data-baseweb="tag"] {{ background: var(--card2) !important; color: var(--ink); }}
button[kind="primary"], [data-testid="stBaseButton-primary"] {{ background: var(--accent); color: #032a2e; font-weight: 800; border: 0; border-radius: 12px; padding: 0.55rem 1.4rem; }}
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {{ background: #8af5e0; color: #032a2e; }}
[data-testid="stDataFrame"] {{ border: 1px solid var(--line); border-radius: 14px; overflow: hidden; }}
[data-testid="stAlert"] {{ background: var(--card); border: 1px solid var(--line); border-left: 4px solid var(--accent); border-radius: 12px; color: var(--ink); }}
[data-testid="stAlert"] p {{ color: var(--ink); }}
a {{ color: var(--accent); }}
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------- data
@st.cache_data
def load_outputs():
    metrics = json.loads((OUT / "metrics.json").read_text())
    scored = pd.read_csv(OUT / "scored_transactions.csv", parse_dates=["timestamp"])
    imp = pd.read_csv(OUT / "feature_importance.csv")
    return metrics, scored, imp


@st.cache_resource
def load_model():
    return joblib.load(OUT / "model.joblib"), pd.Series(json.loads((OUT / "baseline.json").read_text()))


if not (OUT / "model.joblib").exists():
    st.error("No trained model found. Run `python -m src.generate_data`, then `python -m src.train`.")
    st.stop()

metrics, scored, importance = load_outputs()
model, baseline = load_model()
T = metrics["test"]
BIZ = metrics["business"]
GROUPS = list(FEATURE_GROUPS)


def style(fig, height=330):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Manrope, system-ui, sans-serif", size=13, color=INK),
        legend=dict(orientation="h", y=1.14, x=0, font=dict(color=INK2)),
        hoverlabel=dict(bgcolor=CARD2, font_color=INK, bordercolor=LINE),
    )
    fig.update_xaxes(showgrid=False, linecolor=LINE, tickfont=dict(color=INK2), title_font=dict(color=INK2))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont=dict(color=INK2), title_font=dict(color=INK2))
    return fig


def contribution_chart(contrib: pd.Series, height=300):
    """Bright bars push risk up, muted bars pull it down (percentage points)."""
    c = contrib.sort_values()
    fig = go.Figure(go.Bar(
        x=c.values * 100, y=c.index, orientation="h",
        marker_color=[ACCENT if v > 0 else SOFT for v in c.values],
        text=[f"{v*100:+.1f} pts" for v in c.values], textposition="outside", cliponaxis=False,
        textfont=dict(color=INK),
        hovertemplate="%{y}: %{x:+.1f} percentage points<extra></extra>",
    ))
    fig.update_traces(marker_cornerradius=4)
    style(fig, height)
    fig.update_xaxes(title="Change in fraud probability (percentage points)", zeroline=True,
                     zerolinecolor=LINE, showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


def explain_one(row: pd.DataFrame, n_perm=40):
    contrib = shapley_contributions(model, row, baseline, n_perm=n_perm, seed=1)
    groups = group_contributions(contrib)
    return groups.iloc[0], top_reasons(row, groups, k=4, min_contrib=0.01)[0]


# --------------------------------------------------------------------- hero
top = scored.iloc[0]
top_reasons_list = [r for r in str(top.top_reasons).split(" | ") if r and r != "nan"][:3]
reasons_html = "".join(f"<li>{html.escape(r)}</li>" for r in top_reasons_list)
st.markdown(
    f"""
<div class="hero">
  <div>
    <h1>Fraud detection console</h1>
    <p class="lead">Every transaction gets a risk score. Every suspicious one comes with the reasons behind it.</p>
    <p class="meta">All data is synthetic.</p>
  </div>
  <div class="alert">
    <div class="top"><span>Highest-risk alert</span><span class="chip">Risk {top.risk_score:.0%}</span></div>
    <div class="amt">₹{top.amount:,.0f}</div>
    <div class="sub">{html.escape(str(top.channel))} payment, {html.escape(str(top.merchant_category))}, {html.escape(str(top.transaction_id))}</div>
    <ul>{reasons_html}</ul>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

tab_over, tab_susp, tab_perf, tab_xai, tab_live = st.tabs(
    ["Overview", "Suspicious transactions", "Model performance", "Explainability", "Score a transaction"]
)

# ----------------------------------------------------------------- overview
with tab_over:
    st.write("")
    c = st.columns(5)
    c[0].metric("Transactions scored", f"{BIZ['test_transactions']:,}")
    c[1].metric("Alerts raised", f"{BIZ['alerts']:,}", f"{BIZ['alert_rate']:.2%} of volume", delta_color="off")
    c[2].metric("Precision", f"{T['precision']:.1%}", help="Of the alerts raised, the share that were real fraud.")
    c[3].metric("Recall", f"{T['recall']:.1%}", help="Of all real fraud, the share that was caught.")
    c[4].metric("Fraud value caught", f"{BIZ['caught_share']:.1%}",
                f"₹{BIZ['caught_amount']/1e5:.1f} lakh of ₹{BIZ['fraud_amount']/1e5:.1f} lakh", delta_color="off")
    st.write("")

    left, right = st.columns([3, 2])
    with left, st.container(border=True):
        st.subheader("Alerts and confirmed fraud by day")
        d = pd.DataFrame(metrics["daily"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d["date"], y=d["alerts"], name="Alerts raised",
                                 line=dict(color=ACCENT, width=2.5), mode="lines"))
        fig.add_trace(go.Scatter(x=d["date"], y=d["confirmed"], name="Confirmed fraud",
                                 line=dict(color=INK2, width=2, dash="dot"), mode="lines"))
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(style(fig), width="stretch")
    with right, st.container(border=True):
        st.subheader("Alerts by channel")
        al = scored[scored.flagged == 1]
        ch = al.groupby("channel").size().reindex(CHANNELS).fillna(0)
        fig = go.Figure(go.Bar(x=ch.index, y=ch.values, marker_color=ACCENT,
                               text=ch.values.astype(int), textposition="outside", textfont=dict(color=INK)))
        fig.update_traces(marker_cornerradius=4)
        st.plotly_chart(style(fig), width="stretch")

    with st.container(border=True):
        st.subheader("What drives the alerts")
        al = scored[scored.flagged == 1]
        g = pd.DataFrame({g: al[f"contrib_{g}"] for g in GROUPS})
        share = (g.clip(lower=0).sum() / g.clip(lower=0).sum().sum()).sort_values()
        fig = go.Figure(go.Bar(x=share.values * 100, y=share.index, orientation="h", marker_color=ACCENT,
                               text=[f"{v:.0%}" for v in share.values], textposition="outside",
                               textfont=dict(color=INK), cliponaxis=False))
        fig.update_traces(marker_cornerradius=4)
        fig.update_xaxes(title="Share of total risk contribution across all alerts (%)", range=[0, share.max() * 130])
        st.plotly_chart(style(fig, 300), width="stretch")

# ------------------------------------------------------- suspicious txns
with tab_susp:
    st.write("")
    f1, f2, f3, f4 = st.columns([1.4, 1.2, 1.2, 1.2])
    bands = f1.multiselect("Risk band", ["High", "Medium"], default=["High"])
    chans = f2.multiselect("Channel", CHANNELS, default=CHANNELS)
    min_amt = f3.number_input("Minimum amount (₹)", min_value=0, value=0, step=1000)
    cats = f4.multiselect("Merchant category", CATEGORIES, default=CATEGORIES)

    view = scored[scored.risk_band.isin(bands) & scored.channel.isin(chans)
                  & scored.merchant_category.isin(cats) & (scored.amount >= min_amt)]
    st.caption(f"{len(view):,} transactions match, sorted by risk score")
    st.dataframe(
        view[["transaction_id", "timestamp", "customer_id", "amount", "channel", "merchant_category",
              "risk_score", "risk_band", "top_reasons"]],
        hide_index=True, width="stretch", height=320,
        column_config={
            "transaction_id": st.column_config.TextColumn("Transaction"),
            "customer_id": st.column_config.TextColumn("Customer"),
            "channel": st.column_config.TextColumn("Channel"),
            "merchant_category": st.column_config.TextColumn("Merchant"),
            "risk_band": st.column_config.TextColumn("Band"),
            "risk_score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=1, format="%.2f"),
            "amount": st.column_config.NumberColumn("Amount (₹)", format="%.0f"),
            "timestamp": st.column_config.DatetimeColumn("Time", format="DD MMM, HH:mm"),
            "top_reasons": st.column_config.TextColumn("Why it was flagged", width="large"),
        },
    )

    st.subheader("Investigate a transaction")
    if view.empty:
        st.info("No transactions match these filters. Widen the risk band or lower the minimum amount.")
    else:
        tid = st.selectbox("Transaction", view["transaction_id"].tolist())
        r = view[view.transaction_id == tid].iloc[0]
        a, b = st.columns([2, 3])
        with a, st.container(border=True):
            st.markdown(f"**{r.transaction_id}**, customer {r.customer_id}")
            st.metric("Fraud risk score", f"{r.risk_score:.1%}", f"{r.risk_band} risk", delta_color="off")
            st.markdown("**Why this was flagged**")
            for reason in str(r.top_reasons).split(" | "):
                st.markdown(f"- {reason}")
            st.caption(f"A typical legitimate transaction scores about {r.baseline_score:.1%}. "
                       "The bars add up to the gap between that and this score.")
            st.caption(f"Ground truth (synthetic data): {'fraud' if r.is_fraud else 'legitimate'}")
        with b, st.container(border=True):
            st.plotly_chart(contribution_chart(pd.Series({g: r[f"contrib_{g}"] for g in GROUPS})),
                            width="stretch")

# ------------------------------------------------------ model performance
with tab_perf:
    st.write("")
    st.subheader("Model comparison")
    st.caption("Trained on the first 70% of the timeline and chosen on validation PR-AUC, which is the right "
               "metric when fraud is about 2% of traffic. Scored once on the most recent 20%.")
    cmp_df = pd.DataFrame(metrics["comparison"])
    show = cmp_df[["model", "val_pr_auc", "test_pr_auc", "test_roc_auc", "test_precision", "test_recall", "test_f2"]]
    show.columns = ["Model", "Val PR-AUC", "Test PR-AUC", "Test ROC-AUC", "Precision", "Recall", "F2"]
    st.dataframe(show.style.format({c: "{:.3f}" for c in show.columns[1:]})
                 .apply(lambda s: ["font-weight:700" if v == metrics["selected_model"] else "" for v in s],
                        subset=["Model"]),
                 hide_index=True, width="stretch")
    st.write("")

    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.subheader("ROC curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=SOFT, dash="dot"),
                                 name="Random guess"))
        fig.add_trace(go.Scatter(x=metrics["roc"]["fpr"], y=metrics["roc"]["tpr"], mode="lines",
                                 line=dict(color=ACCENT, width=2.5), name=f"AUC {T['roc_auc']:.3f}"))
        fig.update_xaxes(title="False positive rate"); fig.update_yaxes(title="True positive rate")
        st.plotly_chart(style(fig), width="stretch")
    with c2, st.container(border=True):
        st.subheader("Precision and recall")
        fig = go.Figure(go.Scatter(x=metrics["pr"]["recall"], y=metrics["pr"]["precision"], mode="lines",
                                   line=dict(color=ACCENT, width=2.5), name=f"PR-AUC {T['pr_auc']:.3f}"))
        fig.add_trace(go.Scatter(x=[T["recall"]], y=[T["precision"]], mode="markers",
                                 marker=dict(color=INK, size=11, line=dict(color=ACCENT, width=3)),
                                 name=f"Operating point (threshold {metrics['threshold']:.2f})"))
        fig.update_xaxes(title="Recall"); fig.update_yaxes(title="Precision", range=[0, 1.02])
        st.plotly_chart(style(fig), width="stretch")

    c3, c4 = st.columns(2)
    with c3, st.container(border=True):
        st.subheader("Confusion matrix (test set)")
        z = [[T["tn"], T["fp"]], [T["fn"], T["tp"]]]
        lab = [["True negative", "False alarm"], ["Missed fraud", "Caught fraud"]]
        fig = go.Figure(go.Heatmap(
            z=np.log10(np.array(z) + 1), x=["Predicted legit", "Predicted fraud"], y=["Actually legit", "Actually fraud"],
            colorscale=[[0, "#0f4a54"], [1, "#1f8f96"]], showscale=False,
            text=[[f"{lab[i][j]}<br><b>{z[i][j]:,}</b>" for j in range(2)] for i in range(2)],
            texttemplate="%{text}", textfont=dict(color="#ffffff", size=14), hoverinfo="skip", xgap=3, ygap=3))
        fig.update_yaxes(autorange="reversed", showgrid=False)
        st.plotly_chart(style(fig, 300), width="stretch")
    with c4, st.container(border=True):
        st.subheader("Recall by fraud pattern")
        bt = pd.DataFrame(metrics["recall_by_fraud_type"]).sort_values("recall")
        fig = go.Figure(go.Bar(x=bt["recall"] * 100, y=bt["fraud_type"].str.replace("_", " "), orientation="h",
                               marker_color=ACCENT, textfont=dict(color=INK),
                               text=[f"{v:.0%} ({c}/{n})" for v, c, n in
                                     zip(bt["recall"], bt["caught"], bt["frauds"])],
                               textposition="outside", cliponaxis=False))
        fig.update_traces(marker_cornerradius=4)
        fig.update_xaxes(range=[0, 135], title="% of fraud caught")
        st.plotly_chart(style(fig, 300), width="stretch")
    st.info("**Known limitation:** the subtle pattern (mildly elevated amounts, otherwise normal behaviour) is "
            "mostly missed. Catching it needs richer signals such as merchant history, device fingerprints or "
            "customer-level behaviour profiles.")

# ------------------------------------------------------------ explainability
with tab_xai:
    st.write("")
    with st.container(border=True):
        st.subheader("What the model relies on overall")
        st.caption("Permutation importance on the test set: how much PR-AUC drops when a field is shuffled.")
        imp = importance.sort_values("importance")
        fig = go.Figure(go.Bar(x=imp["importance"], y=imp["feature"], orientation="h", marker_color=ACCENT,
                               error_x=dict(type="data", array=imp["std"], color=SOFT)))
        fig.update_traces(marker_cornerradius=4)
        fig.update_xaxes(title="Drop in PR-AUC when shuffled")
        st.plotly_chart(style(fig, 440), width="stretch")
    with st.container(border=True):
        st.subheader("Why one transaction was flagged")
        st.markdown(
            """
Every flagged transaction gets a Shapley-value explanation. Each field is switched from a "typical legitimate
transaction" to its real value in random orders, and the average change in fraud probability is that field's
contribution. The contributions add up exactly to the gap between the transaction's score and the baseline score.
They are grouped into terms an analyst uses: *unusual amount, velocity, device and authentication, location,
time of day, channel and merchant, account profile*.

The method works with any scikit-learn model, so the same code explains the logistic regression, random forest
or gradient boosting version.
"""
        )

# ----------------------------------------------------------------- live scoring
PRESETS = {
    "Everyday purchase": dict(amount=1800, channel="POS", cat="Grocery", hour=18, dist=3.0, intl=0, dev=0,
                              c1=0, c24=2, fail=0, avg=1500, age=900, bal=24000),
    "Account takeover": dict(amount=27000, channel="UPI", cat="Money Transfer", hour=2, dist=8.0, intl=0, dev=1,
                             c1=1, c24=2, fail=3, avg=1600, age=500, bal=30000),
    "Foreign card misuse": dict(amount=14000, channel="POS", cat="Jewellery", hour=15, dist=4200.0, intl=1, dev=0,
                                c1=1, c24=2, fail=0, avg=1700, age=1200, bal=60000),
}
with tab_live:
    st.write("")
    st.subheader("Score a new transaction")
    preset = st.selectbox("Start from an example", list(PRESETS))
    p = PRESETS[preset]
    k = preset.replace(" ", "_")
    with st.container(border=True):
        a, b, c = st.columns(3)
        amount = a.number_input("Amount (₹)", 1.0, 1_000_000.0, float(p["amount"]), 100.0, key=f"amt{k}")
        channel = a.selectbox("Channel", CHANNELS, index=CHANNELS.index(p["channel"]), key=f"ch{k}")
        category = a.selectbox("Merchant category", CATEGORIES, index=CATEGORIES.index(p["cat"]), key=f"cat{k}")
        avg = a.number_input("Customer's 30-day average amount (₹)", 1.0, 1_000_000.0, float(p["avg"]), 100.0, key=f"av{k}")
        hour = b.slider("Hour of day", 0, 23, p["hour"], key=f"hr{k}")
        dist = b.number_input("Distance from home (km)", 0.0, 20000.0, float(p["dist"]), 5.0, key=f"di{k}")
        bal = b.number_input("Balance before (₹)", 1.0, 100_000_000.0, float(p["bal"]), 1000.0, key=f"ba{k}")
        intl = b.checkbox("International", bool(p["intl"]), key=f"in{k}")
        dev = b.checkbox("New or unrecognised device", bool(p["dev"]), key=f"dv{k}")
        c1 = c.number_input("Transactions in the last hour", 0, 50, p["c1"], key=f"c1{k}")
        c24 = c.number_input("Transactions in the last 24 hours", 0, 200, max(p["c24"], p["c1"]), key=f"c24{k}")
        fail = c.number_input("Failed attempts in 24 hours", 0, 20, p["fail"], key=f"fa{k}")
        age = c.number_input("Account age (days)", 1, 20000, p["age"], key=f"ag{k}")

    row = pd.DataFrame([{
        "amount": amount, "hour": hour, "distance_from_home_km": dist, "txn_count_1h": c1,
        "txn_count_24h": max(c24, c1), "failed_attempts_24h": fail, "account_age_days": age,
        "avg_amount_30d": avg, "balance_before": bal, "is_international": int(intl),
        "is_new_device": int(dev), "channel": channel, "merchant_category": category,
    }])[RAW_FEATURES]

    if st.button("Score transaction", type="primary"):
        score = float(model.predict_proba(row)[0, 1])
        thr = metrics["threshold"]
        band = "High" if score >= thr else "Medium" if score >= thr * 0.5 else "Low"
        s1, s2 = st.columns([2, 3])
        with s1, st.container(border=True):
            st.metric("Fraud risk score", f"{score:.1%}", f"{band} risk, alert threshold {thr:.0%}",
                      delta_color="off")
            if band == "High":
                st.error("Flag for review or ask for extra authentication.")
            elif band == "Medium":
                st.warning("Elevated but below the alert threshold. Keep monitoring.")
            else:
                st.success("Looks normal.")
            groups, reason_text = explain_one(row)
            if reason_text:
                st.markdown("**Top risk drivers**")
                for r_ in reason_text.split(" | "):
                    st.markdown(f"- {r_}")
        with s2, st.container(border=True):
            st.plotly_chart(contribution_chart(groups), width="stretch")