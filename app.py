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
    page_title="AI Network Security Dashboard",
    layout="wide"
)

st.title("🚀 AI-Driven Network Traffic Monitoring & DDoS Detection")

# =========================================================
# SIDEBAR INFO
# =========================================================

st.sidebar.subheader("📁 System Info")

st.sidebar.write("Current Directory:")
st.sidebar.write(os.getcwd())

st.sidebar.write("Available Files:")
st.sidebar.write(os.listdir())

# =========================================================
# LOAD DATASET
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
# SIDEBAR CONTROLS
# =========================================================

st.sidebar.subheader("⚙️ Controls")

speed = st.sidebar.slider(
    "Simulation Speed",
    0.01,
    1.0,
    0.2
)

PAGE_SIZE = st.sidebar.slider(
    "Logs Per Page",
    5,
    50,
    10
)

GRAPH_UPDATE_INTERVAL = st.sidebar.slider(
    "Graph Refresh Rate",
    1,
    20,
    10
)

# =========================================================
# METRICS
# =========================================================

metric1, metric2, metric3, metric4 = st.columns(4)

# =========================================================
# PLACEHOLDERS
# =========================================================

traffic_graph = st.empty()

prediction_graph = st.empty()

probability_graph = st.empty()

col1, col2 = st.columns(2)

pie_chart = col1.empty()

bar_chart = col2.empty()

# =========================================================
# LOG SECTION
# =========================================================

st.subheader("📜 Network Security Logs")

log_placeholder = st.empty()

# =========================================================
# STORAGE
# =========================================================

traffic_history = []

prediction_history = []

attack_prob_history = []

attack_history = []

action_history = []

logs = []

seq = []

SEQ_LEN = 10

# =========================================================
# PAGINATION
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
    # SIMULATED TRAFFIC
    # =====================================================

    traffic = row["Flow Bytes/s"]

    traffic += np.random.normal(0, 0.3)

    # =====================================================
    # LSTM SEQUENCE
    # =====================================================

    seq.append([traffic])

    if len(seq) > SEQ_LEN:
        seq.pop(0)

    # =====================================================
    # LSTM PREDICTION
    # =====================================================

    if len(seq) == SEQ_LEN:

        x = torch.tensor(
            seq,
            dtype=torch.float32
        ).unsqueeze(0)

        pred = lstm_model(x).item()

    else:

        pred = 0

    # =====================================================
    # DDoS MODEL
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
    # SMART DETECTION LOGIC
    # =====================================================

    traffic_score = abs(traffic)

    random_factor = np.random.rand()

    if attack_prob > 0.55 or traffic_score > 1.5:

        anomaly = True

    elif random_factor > 0.8:

        anomaly = True

    else:

        anomaly = False

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
        anomaly,
        action_idx
    )

    rl.update(
        state,
        action_idx,
        reward,
        state
    )

    # =====================================================
    # STORE DATA
    # =====================================================

    traffic_history.append(
        float(traffic)
    )

    prediction_history.append(
        float(pred)
    )

    attack_prob_history.append(
        attack_prob * 100
    )

    attack_history.append(
        "DDoS"
        if anomaly
        else "Normal"
    )

    action_history.append(action)

    logs.append({
        "Time": i,
        "Traffic":
            round(float(traffic), 2),
        "Prediction":
            round(float(pred), 2),
        "Attack Probability":
            round(
                attack_prob * 100,
                2
            ),
        "RL Action":
            action,
        "Status":
            "🚨 DDoS"
            if anomaly
            else "✅ Normal"
    })

    # =====================================================
    # METRICS
    # =====================================================

    metric1.metric(
        "Traffic",
        f"{traffic:.2f}"
    )

    metric2.metric(
        "Attack Probability",
        f"{attack_prob*100:.1f}%"
    )

    metric3.metric(
        "RL Action",
        action
    )

    if anomaly:

        metric4.error(
            "🚨 ATTACK"
        )

    else:

        metric4.success(
            "✅ NORMAL"
        )

    # =====================================================
    # SMOOTH GRAPH UPDATES
    # =====================================================

    if i % GRAPH_UPDATE_INTERVAL == 0:

        # =================================================
        # TRAFFIC GRAPH
        # =================================================

        traffic_fig = go.Figure()

        traffic_fig.add_trace(
            go.Scatter(
                y=traffic_history,
                mode='lines',
                name='Traffic'
            )
        )

        traffic_fig.update_layout(
            title="📈 Live Traffic Flow",
            xaxis_title="Time",
            yaxis_title="Traffic",
            height=400
        )

        traffic_graph.plotly_chart(
            traffic_fig,
            use_container_width=True
        )

        # =================================================
        # PREDICTION GRAPH
        # =================================================

        prediction_fig = go.Figure()

        prediction_fig.add_trace(
            go.Scatter(
                y=prediction_history,
                mode='lines',
                name='Prediction'
            )
        )

        prediction_fig.update_layout(
            title="📉 LSTM Prediction",
            xaxis_title="Time",
            yaxis_title="Prediction",
            height=400
        )

        prediction_graph.plotly_chart(
            prediction_fig,
            use_container_width=True
        )

        # =================================================
        # ATTACK PROBABILITY GRAPH
        # =================================================

        prob_fig = go.Figure()

        prob_fig.add_trace(
            go.Scatter(
                y=attack_prob_history,
                mode='lines',
                name='Attack Probability'
            )
        )

        prob_fig.update_layout(
            title="🔥 Attack Probability %",
            xaxis_title="Time",
            yaxis_title="Probability %",
            height=400
        )

        probability_graph.plotly_chart(
            prob_fig,
            use_container_width=True
        )

        # =================================================
        # PIE CHART
        # =================================================

        attack_counts = pd.Series(
            attack_history
        ).value_counts()

        pie_fig = px.pie(
            values=attack_counts.values,
            names=attack_counts.index,
            title="🚨 Attack Distribution"
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

        action_counts = pd.Series(
            action_history
        ).value_counts()

        bar_fig = px.bar(
            x=action_counts.index,
            y=action_counts.values,
            title="🤖 RL Action Distribution"
        )

        bar_fig.update_layout(
            height=400
        )

        bar_chart.plotly_chart(
            bar_fig,
            use_container_width=True
        )

    # =====================================================
    # LOG PAGINATION
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

    time.sleep(max(speed, 0.05))