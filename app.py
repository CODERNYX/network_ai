import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import time
import os

import plotly.graph_objects as go
import plotly.express as px

from model import DDoSNet, LSTMModel, RLAgent
from sklearn.preprocessing import StandardScaler

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI vs Rule-Based DDoS Dashboard",
    layout="wide"
)

st.title("🚀 AI vs Rule-Based Network Security Dashboard")

# =========================================================
# SIDEBAR CONTROLS
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
    8
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
# TOP NAVIGATION BAR
# =========================================================

tabs = st.tabs([
    "📊 Dashboard",
    "📈 Traffic Graph",
    "🤖 ML vs Rule",
    "🔥 Attack Probability",
    "📜 Live Logs"
])

# =========================================================
# METRICS
# =========================================================

with tabs[0]:

    st.subheader("📊 Real-Time Metrics")

    m1, m2, m3, m4, m5 = st.columns(5)

    dashboard_placeholder = st.empty()

# =========================================================
# TRAFFIC GRAPH TAB
# =========================================================

with tabs[1]:

    traffic_chart = st.empty()

# =========================================================
# ML VS RULE GRAPH TAB
# =========================================================

with tabs[2]:

    comparison_chart = st.empty()

# =========================================================
# ATTACK PROBABILITY TAB
# =========================================================

with tabs[3]:

    probability_chart = st.empty()

# =========================================================
# LIVE LOG TAB
# =========================================================

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

    traffic += np.random.normal(0, 0.25)

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

    # =====================================================
    # RULE DETECTION
    # =====================================================

    if (
        row["Flow Packets/s"] > 1.2
        or row["Total Fwd Packets"] > 1.5
        or abs(traffic) > 1.8
    ):

        rule_detection = "🚨 DDoS"

        rule_detected += 1

    else:

        rule_detection = "✅ Normal"

        normal_count += 1

    # =====================================================
    # RL AGENT
    # =====================================================

    state = rl.get_state(
        abs(traffic)
    )

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

        m1.metric(
            "Traffic",
            f"{traffic:.2f}"
        )

        m2.metric(
            "ML Attack %",
            f"{attack_prob*100:.1f}%"
        )

        m3.metric(
            "ML Detections",
            ml_detected
        )

        m4.metric(
            "Rule Detections",
            rule_detected
        )

        m5.metric(
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
            height=500
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
            title="🤖 ML vs 📏 Rule-Based Detection",
            height=500
        )

        comparison_chart.plotly_chart(
            fig2,
            use_container_width=True
        )

        # =================================================
        # ATTACK PROBABILITY
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
            height=500
        )

        probability_chart.plotly_chart(
            fig3,
            use_container_width=True
        )

    # =====================================================
    # LIVE LOG CONSOLE
    # =====================================================

    color = "#ff4b4b"

    if ml_detection == "✅ Normal":
        color = "#00c853"

    logs_html += f"""

    <div style="
        padding:10px;
        margin-bottom:10px;
        border-radius:10px;
        background-color:#1e1e1e;
        border-left:6px solid {color};
        color:white;
    ">

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
        <div style="
            height:600px;
            overflow-y:scroll;
            padding:10px;
            border:2px solid #444;
            border-radius:10px;
            background-color:#111111;
        ">
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