"""CascadeGuard dashboard:  streamlit run app.py"""
import json, os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import G, POS, SERVICES, STRATEGIES, H
from simulator import simulate, DOWN, DEGRADED
from dataset import build_features, FEATURES
from analysis import detect_root_cause, compare_strategies, onset_table
import train

st.set_page_config(page_title="CascadeGuard", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_model():
    if not os.path.exists(train.MODEL_PATH):
        with st.spinner("First run: training the model (about 30 seconds)..."):
            train.main()
    return joblib.load(train.MODEL_PATH)


@st.cache_data
def load_metrics():
    with open(train.METRICS_PATH) as fh:
        return json.load(fh)


bundle = load_model()
model, thr = bundle["model"], bundle["threshold"]
metrics = load_metrics()

# ---------------- sidebar ----------------
st.sidebar.title("🛡️ CascadeGuard")
st.sidebar.caption("Predict · Trace · Recover")
root = st.sidebar.selectbox("Service to break", [s for s in SERVICES if s != "frontend"],
                            index=SERVICES.index("db") - 1)
fail_at = st.sidebar.slider("Failure starts at minute", 15, 60, 30)
seed = st.sidebar.number_input("Random seed", 0, 9999, 5)
apply_delay = st.sidebar.slider("Minutes until recovery action is applied", 0, 15, 3,
                                help="Earlier action (thanks to early prediction) means less damage.")
T = 100


@st.cache_data
def run_sim(root, fail_at, seed):
    df = simulate(root, T=T, fail_at=fail_at, seed=int(seed))
    feats = build_features(df, with_labels=False)
    feats["risk"] = model.predict_proba(feats[FEATURES])[:, 1]
    feats.loc[feats.status == DOWN, "risk"] = 1.0
    return df, feats


df, feats = run_sim(root, fail_at, seed)

st.title("CascadeGuard: cascading failure prediction")
st.caption(f"Injected failure: **{root}** at minute **{fail_at}**. "
           f"Model alerts when predicted risk of failing within {H} minutes ≥ {thr:.2f}.")

t_now = st.slider("Minute", 0, T - 1, min(fail_at + 8, T - 1))

# ---------------- state at t_now ----------------
snap = feats[feats.t == t_now].set_index("service")
rc = detect_root_cause(df, t_now)
rc_nodes = {rc["root"]} if rc else set()
rc_edges = set(map(tuple, rc["edges"])) if rc else set()

fig = go.Figure()
for a, b in G.edges:   # failure propagates from dependency (b) up to dependent (a)
    hot = (a, b) in rc_edges
    fig.add_annotation(
        x=POS[a][0], y=POS[a][1], ax=POS[b][0], ay=POS[b][1],
        xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3,
        arrowwidth=3 if hot else 1.2, arrowcolor="#dc2626" if hot else "#9ca3af",
        standoff=22, startstandoff=22)
xs = [POS[s][0] for s in G.nodes]
ys = [POS[s][1] for s in G.nodes]
risk = [float(snap.loc[s, "risk"]) for s in G.nodes]
stat = [int(snap.loc[s, "status"]) for s in G.nodes]
label = {0: "OK", 1: "DEGRADED", 2: "DOWN"}
fig.add_trace(go.Scatter(
    x=xs, y=ys, mode="markers+text", text=list(G.nodes), textposition="bottom center",
    marker=dict(size=46, color=risk, colorscale="RdYlGn_r", cmin=0, cmax=1,
                line=dict(width=[5 if s == DOWN else 1 for s in stat],
                          color=["#111827" if s == DOWN else "#ffffff" for s in stat]),
                colorbar=dict(title="Risk", thickness=12)),
    customdata=[[label[s], r] for s, r in zip(stat, risk)],
    hovertemplate="<b>%{text}</b><br>status: %{customdata[0]}<br>risk: %{customdata[1]:.2f}<extra></extra>"))
if rc:
    fig.add_trace(go.Scatter(
        x=[POS[r][0] for r in rc_nodes], y=[POS[r][1] for r in rc_nodes], mode="markers",
        marker=dict(size=66, symbol="circle-open", color="#f59e0b", line=dict(width=5, color="#f59e0b")),
        hoverinfo="skip", name="Root cause"))
fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                  xaxis=dict(visible=False, range=[-3, 3]), yaxis=dict(visible=False, range=[-0.6, 4.5]))

left, right = st.columns([3, 2])
with left:
    st.subheader("Dependency graph at this minute")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Colour = predicted failure risk · black ring = already down · "
               "gold ring = traced root cause · red arrows = failure path")
with right:
    st.subheader("Root cause")
    if rc:
        st.metric("Traced origin", rc["root"].upper(), f"{len(rc['affected'])} downstream services affected")
        st.write(f"True origin: **{root}** → " + ("✅ correct" if rc["root"] == root else "❌ wrong"))
    else:
        st.info("No anomaly yet.")
    st.subheader("Top at-risk services")
    top = snap[snap.status != DOWN].sort_values("risk", ascending=False).head(5)
    st.dataframe(pd.DataFrame({"service": top.index, "risk": top.risk.round(2),
                               "alert": np.where(top.risk >= thr, "🚨 ALERT", "")}),
                 hide_index=True, use_container_width=True)
    st.subheader("Incident timeline")
    tl = onset_table(df, t_now)
    st.dataframe(tl, hide_index=True, use_container_width=True, height=180)

# ---------------- timeline chart ----------------
st.subheader("Services down vs. alerts over time")
down_n = df[df.status == DOWN].groupby("t").size().reindex(range(T), fill_value=0)
alert_n = feats[(feats.status != DOWN) & (feats.risk >= thr)].groupby("t").size().reindex(range(T), fill_value=0)
ch = go.Figure()
ch.add_trace(go.Scatter(x=down_n.index, y=down_n.values, name="Services DOWN", line=dict(color="#7f1d1d")))
ch.add_trace(go.Scatter(x=alert_n.index, y=alert_n.values, name="Services flagged at risk", line=dict(color="#f59e0b")))
ch.add_vline(x=t_now, line_dash="dot")
ch.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="minute", yaxis_title="count")
st.plotly_chart(ch, use_container_width=True)

# ---------------- recovery comparison ----------------
st.subheader("Recovery strategy comparison")
st.caption("Same failure re-simulated under each strategy (30 runs each). "
           "Loss = down service-min + 0.4 × degraded service-min + action cost. Lower is better.")


@st.cache_data
def strat(root, fail_at, delay):
    return compare_strategies(root, fail_at=fail_at, T=T, n_seeds=30, apply_delay=delay)


res = strat(root, fail_at, apply_delay)
c1, c2 = st.columns([3, 2])
with c1:
    st.dataframe(res.drop(columns="strategy"), hide_index=True, use_container_width=True)
with c2:
    bar = go.Figure(go.Bar(x=res.total_loss, y=res.label, orientation="h",
                           marker_color=["#16a34a"] + ["#9ca3af"] * (len(res) - 1)))
    bar.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis=dict(autorange="reversed"), xaxis_title="total loss")
    st.plotly_chart(bar, use_container_width=True)
best = res.iloc[0]
saved = res[res.strategy == "none"].total_loss.iloc[0] - best.total_loss
st.success(f"✅ Recommended: **{best.label}**. Cuts total loss by {saved:.0f} vs. doing nothing.")

# ---------------- model evidence ----------------
with st.expander("📊 Model evidence on unseen test runs (vs. naive baseline)"):
    m = pd.DataFrame({"AI model (graph features)": metrics["model"],
                      "Baseline (own error rate > threshold)": metrics["baseline"]})
    st.dataframe(m, use_container_width=True)
    st.bar_chart(pd.Series(metrics["feature_importance"]).head(8))
