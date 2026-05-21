import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import time
import os

import plotly.graph_objects as go

from model import DDoSNet, LSTMModel, RLAgent
from sklearn.preprocessing import StandardScaler

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Security Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# RESPONSIVE CSS
# =========================================================

st.markdown("""
<style>

/* =====================================================
GLOBAL
===================================================== */

.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    padding-left: 1rem;
    padding-right: 1rem;
    max-width: 100%;
}

/* =====================================================
METRIC CARDS
===================================================== */

[data-testid="metric-container"] {

    background-color: #111111;

    border: 1px solid #333333;

    padding: 12px;

    border-radius: 12px;

    text-align: center;
}

/* =====================================================
TABS
===================================================== */

button[data-baseweb="tab"] {

    font-size: 14px;

    padding: 10px;

    margin: 2px;
}

/* =====================================================
GRAPHS
===================================================== */

.stPlotlyChart {

    border-radius: 12px;

    overflow: hidden;
}

/* =====================================================
LOG CONTAINER
===================================================== */

.scroll-log {

    height: 65vh;

    overflow-y: auto;

    padding: 12px;

    border-radius: 12px;

    background-color: #111111;

    border: 1px solid #444444;
}

/* =====================================================
LOG CARDS
===================================================== */

.log-card {

    padding: 10px;

    margin-bottom: 10px;

    border-radius: 10px;

    background-color: #1e1e1e;

    color: white;

    font-size: 14px;
}

/* =====================================================
TABLET
===================================================== */

@media (max-width: 1024px) {

    h1 {
        font-size: 28px !important;
    }

    h2 {
        font-size: 22px !important;
    }

    h3 {
        font-size: 18px !important;
    }

    button[data-baseweb="tab"] {

        font-size: 12px;
    }

    .scroll-log {

        height: 60vh;
    }
}

/* =====================================================
MOBILE
===================================================== */

@media (max-width: 768px) {

    .block-container {

        padding-left: 0.5rem;
        padding-right: 0.5rem;
    }

    h1 {

        font-size: 22px !important;

        text-align: center;
    }

    h2 {

        font-size: 18px !important;
    }

    h3 {

        font-size: 16px !important;
    }

    button[data-baseweb="tab"] {

        font-size: 11px;

        padding: 6px;
    }

    [data-testid="metric-container"] {

        padding: 8px;
    }

    .scroll-log {

        height: 55vh;

        font-size: 12px;
    }

    .log-card {

        font-size: 12px;
    }
}

/* =====================================================
SMALL MOBILE
===================================================== */

@media (max-width: 480px) {

    h1 {

        font-size: 18px !important;
    }

    button[data-baseweb="tab"] {

        font-size: 10px;

        padding: 4px;
    }

    .scroll-log {

        height: 50vh;
    }
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================

st.title("🚀 AI vs Rule-Based Network Security Dashboard")

st.caption("📱 Responsive Dashboard for All Devices")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ Controls")

speed = st.sidebar.slider(
    "Simulation Speed",
    0.01,
    1.0,
    0.1
)

GRAPH_UPDATE_INTERVAL = st.sidebar.slider(
    "Graph Refresh Interval",
    1,
    20,
    10
)

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_parquet(
        "clean_ddos_dataset.parquet"
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna()

    return df

df = load_data()

# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    ddos_model = DDoSNet(input_size=5)

    ddos_model.load_state_dict(
        torch.load("ddos_model.pth")
    )

    ddos_model.eval()

    lstm_model = LSTMModel()

    lstm_model.load_state_dict(
        torch.load("lstm_model.pth")
    )

    lstm_model.eval()

    rl = RLAgent()

    return ddos_model, lstm_model, rl

ddos_model, lstm_model, rl = load_models()

# =========================================================
# FEATURES
# =========================================================

features = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Flow Bytes/s",
    "Flow Packets/s"
]

features = [
    f for f in features
    if f in df.columns
]

scaler = StandardScaler()

df[features] = scaler.fit_transform(
    df[features]
)

# =========================================================
# TOP NAVIGATION
# =========================================================

tabs = st.tabs([
    "📊 Dashboard",
    "📈 Traffic",
    "🤖 ML vs Rule",
    "🔥 Probability",
    "📜 Logs"
])

# =========================================================
# DASHBOARD TAB
# =========================================================

with tabs[0]:

    st.subheader("📊 Real-Time Metrics")

    top1, top2 = st.columns(2)

    mid1, mid2 = st.columns(2)

    bottom = st.container()

# =========================================================
# GRAPH TABS
# =========================================================

with tabs[1]:

    traffic_chart = st.empty()

with tabs[2]:

    comparison_chart = st.empty()

with tabs[3]:

    probability_chart = st.empty()

with tabs[4]:

    st.subheader("📜 Live Detection Console")

    log_div = st.empty()

# =========================================================
# STORAGE
# =========================================================

traffic_history = []

ml_history = []

rule_history = []

attack_prob_history = []

logs_html = ""

seq = []

SEQ_LEN = 10

ml_detected = 0

rule_detected = 0

normal_count = 0

# =========================================================
# MAIN LOOP
# =========================================================

for i in range(min(500, len(df))):

    row = df.iloc[i]

    # =====================================================
    # TRAFFIC
    # =====================================================

    traffic = row["Flow Bytes/s"]

    traffic_history.append(
        float(traffic)
    )

    # =====================================================
    # LSTM
    # =====================================================

    seq.append([traffic])

    if len(seq) > SEQ_LEN:
        seq.pop(0)

    if len(seq) == SEQ_LEN:

        x = torch.tensor(
            seq,
            dtype=torch.float32
        ).unsqueeze(0)

        pred = lstm_model(x).item()

    else:

        pred = 0

    # =====================================================
    # ML DETECTION
    # =====================================================

    features_input = torch.tensor([
        row["Flow Duration"],
        row["Total Fwd Packets"],
        row["Total Backward Packets"],
        row["Flow Bytes/s"],
        row["Flow Packets/s"]
    ], dtype=torch.float32).unsqueeze(0)

    out = ddos_model(features_input)

    probabilities = F.softmax(
        out,
        dim=1
    )

    attack_prob = probabilities[0][1].item()

    if attack_prob > 0.55:

        ml_detection = "🚨 DDoS"

        ml_detected += 1

    else:

        ml_detection = "✅ Normal"

        normal_count += 1

    # =====================================================
    # RULE DETECTION
    # =====================================================

    if (
        row["Flow Packets/s"] > 1.2
        or row["Total Fwd Packets"] > 1.5
        or abs(traffic) > 2.0
    ):

        rule_detection = "🚨 DDoS"

        rule_detected += 1

    else:

        rule_detection = "✅ Normal"

    # =====================================================
    # RL AGENT
    # =====================================================

    state = rl.get_state(traffic)

    action_idx = rl.choose_action(state)

    action = rl.actions[action_idx]

    reward = rl.reward(
        ml_detection == "🚨 DDoS",
        action_idx
    )

    rl.update(
        state,
        action_idx,
        reward,
        state
    )

    # =====================================================
    # STORE
    # =====================================================

    ml_history.append(
        1 if ml_detection == "🚨 DDoS"
        else 0
    )

    rule_history.append(
        1 if rule_detection == "🚨 DDoS"
        else 0
    )

    attack_prob_history.append(
        attack_prob * 100
    )

    # =====================================================
    # METRICS
    # =====================================================

    with tabs[0]:

        top1.metric(
            "Traffic",
            f"{traffic:.2f}"
        )

        top2.metric(
            "ML Attack %",
            f"{attack_prob*100:.1f}%"
        )

        mid1.metric(
            "ML Detections",
            ml_detected
        )

        mid2.metric(
            "Rule Detections",
            rule_detected
        )

        with bottom:

            st.metric(
                "RL Action",
                action
            )

    # =====================================================
    # GRAPH UPDATES
    # =====================================================

    if i % GRAPH_UPDATE_INTERVAL == 0:

        # =================================================
        # TRAFFIC GRAPH
        # =================================================

        fig1 = go.Figure()

        fig1.add_trace(
            go.Scatter(
                y=traffic_history,
                mode='lines',
                name='Traffic'
            )
        )

        fig1.update_layout(
            title="📈 Live Traffic Flow",
            height=350
        )

        traffic_chart.plotly_chart(
            fig1,
            use_container_width=True
        )

        # =================================================
        # ML VS RULE GRAPH
        # =================================================

        fig2 = go.Figure()

        fig2.add_trace(
            go.Scatter(
                y=ml_history,
                mode='lines',
                name='ML Detection'
            )
        )

        fig2.add_trace(
            go.Scatter(
                y=rule_history,
                mode='lines',
                name='Rule Detection'
            )
        )

        fig2.update_layout(
            title="🤖 ML vs 📏 Rule-Based",
            height=350
        )

        comparison_chart.plotly_chart(
            fig2,
            use_container_width=True
        )

        # =================================================
        # PROBABILITY GRAPH
        # =================================================

        fig3 = go.Figure()

        fig3.add_trace(
            go.Scatter(
                y=attack_prob_history,
                mode='lines',
                name='Attack Probability'
            )
        )

        fig3.update_layout(
            title="🔥 Attack Probability %",
            height=350
        )

        probability_chart.plotly_chart(
            fig3,
            use_container_width=True
        )

    # =====================================================
    # LOG DIV
    # =====================================================

    color = "#ff4b4b"

    if ml_detection == "✅ Normal":
        color = "#00c853"

    logs_html += f"""

    <div class="log-card"
    style="border-left:6px solid {color};">

    <b>Time:</b> {i}
    <br>

    <b>Traffic:</b> {traffic:.2f}
    <br>

    <b>ML Detection:</b> {ml_detection}
    <br>

    <b>Rule Detection:</b> {rule_detection}
    <br>

    <b>Attack Probability:</b> {attack_prob*100:.2f}%
    <br>

    <b>RL Action:</b> {action}

    </div>
    """

    log_div.markdown(
        f"""
        <div class="scroll-log">
        {logs_html}
        </div>
        """,
        unsafe_allow_html=True
    )

    # =====================================================
    # DELAY
    # =====================================================

    time.sleep(
        max(speed, 0.05)
    )