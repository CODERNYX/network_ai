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
    page_title="AI vs Rule-Based DDoS Detection",
    layout="wide"
)

st.title("🚀 AI vs Rule-Based Network Security Dashboard")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ Dashboard Controls")

speed = st.sidebar.slider(
    "Simulation Speed",
    0.01,
    1.0,
    0.15
)

PAGE_SIZE = st.sidebar.slider(
    "Logs Per Page",
    5,
    50,
    10
)

GRAPH_UPDATE_INTERVAL = st.sidebar.slider(
    "Graph Refresh Interval",
    1,
    20,
    8
)

# =========================================================
# DEBUGGING INFO
# =========================================================

st.sidebar.subheader("📁 File Status")

st.sidebar.write(os.getcwd())

st.sidebar.write(os.listdir())

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
# METRICS
# =========================================================

st.subheader("📊 Real-Time Detection Metrics")

m1, m2, m3, m4, m5 = st.columns(5)

# =========================================================
# CHART PLACEHOLDERS
# =========================================================

ml_chart = st.empty()

comparison_chart = st.empty()

prob_chart = st.empty()

col1, col2 = st.columns(2)

pie_chart = col1.empty()

bar_chart = col2.empty()

# =========================================================
# LOG TABLE
# =========================================================

st.subheader("📜 Detection Logs")

log_placeholder = st.empty()

# =========================================================
# STORAGE
# =========================================================

traffic_history = []

ml_history = []

rule_history = []

attack_prob_history = []

logs = []

seq = []

SEQ_LEN = 10

ml_detected = 0

rule_detected = 0

normal_count = 0

# =========================================================
# PAGE CONTROL
# =========================================================

page = st.sidebar.number_input(
    "📄 Log Page",
    min_value=1,
    value=1,
    step=1,
    key="log_page"
)

# =========================================================
# MAIN LOOP
# =========================================================

for i in range(min(300, len(df))):

    row = df.iloc[i]

    # =====================================================
    # TRAFFIC
    # =====================================================

    traffic = row["Flow Bytes/s"]

    traffic += np.random.normal(0, 0.25)

    traffic_history.append(float(traffic))

    # =====================================================
    # LSTM PREDICTION
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

    # =====================================================
    # ML LOGIC
    # =====================================================

    if attack_prob > 0.55:

        ml_detection = "DDoS"

        ml_detected += 1

    else:

        ml_detection = "Normal"

    # =====================================================
    # RULE-BASED SYSTEM
    # =====================================================

    if (
        row["Flow Packets/s"] > 1.2
        or row["Total Fwd Packets"] > 1.5
        or abs(traffic) > 1.8
    ):

        rule_detection = "DDoS"

        rule_detected += 1

    else:

        rule_detection = "Normal"

        normal_count += 1

    # =====================================================
    # RL AGENT
    # =====================================================

    state = max(
        0,
        min(int(abs(traffic) / 200), 9)
    )

    action_idx = rl.choose_action(state)

    action = rl.actions[action_idx]

    reward = rl.reward(
        ml_detection == "DDoS",
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
        1 if ml_detection == "DDoS"
        else 0
    )

    rule_history.append(
        1 if rule_detection == "DDoS"
        else 0
    )

    attack_prob_history.append(
        attack_prob * 100
    )

    logs.append({

        "Time": i,

        "Traffic":
            round(float(traffic), 2),

        "ML Detection":
            ml_detection,

        "Rule Detection":
            rule_detection,

        "Attack Probability":
            round(
                attack_prob * 100,
                2
            ),

        "RL Action":
            action
    })

    # =====================================================
    # METRICS
    # =====================================================

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
        # LIVE TRAFFIC
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
            xaxis_title="Time",
            yaxis_title="Traffic",
            height=400
        )

        ml_chart.plotly_chart(
            fig1,
            use_container_width=True
        )

        # =================================================
        # ML VS RULE COMPARISON
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
                name='Rule-Based Detection'
            )
        )

        fig2.update_layout(
            title="🤖 ML vs 📏 Rule-Based Detection",
            xaxis_title="Time",
            yaxis_title="Detection",
            height=400
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
            xaxis_title="Time",
            yaxis_title="Probability %",
            height=400
        )

        prob_chart.plotly_chart(
            fig3,
            use_container_width=True
        )

        # =================================================
        # PIE CHART
        # =================================================

        pie_fig = px.pie(
            names=[
                "ML DDoS",
                "Rule DDoS",
                "Normal"
            ],
            values=[
                ml_detected,
                rule_detected,
                normal_count
            ],
            title="🚨 Detection Distribution"
        )

        pie_fig.update_layout(
            height=400
        )

        pie_chart.plotly_chart(
            pie_fig,
            use_container_width=True
        )

        # =================================================
        # BAR CHART
        # =================================================

        bar_fig = px.bar(
            x=[
                "ML",
                "Rule-Based"
            ],
            y=[
                ml_detected,
                rule_detected
            ],
            title="📊 Detection Count Comparison"
        )

        bar_fig.update_layout(
            height=400
        )

        bar_chart.plotly_chart(
            bar_fig,
            use_container_width=True
        )

    # =====================================================
    # PAGINATION
    # =====================================================

    log_df = pd.DataFrame(logs)

    total_pages = max(
        1,
        (len(log_df) // PAGE_SIZE) + 1
    )

    if page > total_pages:
        page = total_pages

    start_idx = (
        (page - 1) * PAGE_SIZE
    )

    end_idx = start_idx + PAGE_SIZE

    paginated_df = log_df.iloc[
        start_idx:end_idx
    ]

    log_placeholder.dataframe(
        paginated_df,
        use_container_width=True
    )

    # =====================================================
    # DELAY
    # =====================================================

    time.sleep(
        max(speed, 0.05)
    )