import streamlit as st
import pandas as pd
import numpy as np
import torch
import time
import plotly.graph_objects as go

from model import DDoSNet, LSTMModel, RLAgent
from sklearn.preprocessing import StandardScaler, MinMaxScaler
st.set_page_config(page_title="AI Network Security Dashboard", layout="wide")

st.title("🚀 AI-Driven Network Traffic Monitoring & DDoS Detection")
@st.cache_data
def load_data():
    df = pd.read_parquet("clean_ddos_dataset.parquet")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

df = load_data()
@st.cache_resource
def load_models():

    ddos_model = DDoSNet(input_size=5)
    ddos_model.load_state_dict(torch.load("ddos_model.pth"))
    ddos_model.eval()

    lstm_model = LSTMModel()
    lstm_model.load_state_dict(torch.load("lstm_model.pth"))
    lstm_model.eval()

    rl = RLAgent()

    return ddos_model, lstm_model, rl

ddos_model, lstm_model, rl = load_models()
features = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Flow Bytes/s",
    "Flow Packets/s"
]

features = [f for f in features if f in df.columns]

scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])
placeholder = st.empty()
chart = st.line_chart()
seq = []
SEQ_LEN = 10

for i in range(200):

    row = df.iloc[i]

    traffic = row["Flow Bytes/s"]

    seq.append([traffic])

    if len(seq) > SEQ_LEN:
        seq.pop(0)
    if len(seq) == SEQ_LEN:

        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)

        pred = lstm_model(x).item()
    else:
        pred = 0
    features_input = torch.tensor([
        row["Flow Duration"],
        row["Total Fwd Packets"],
        row["Total Backward Packets"],
        row["Flow Bytes/s"],
        row["Flow Packets/s"]
    ], dtype=torch.float32).unsqueeze(0)

    out = ddos_model(features_input)

    attack = torch.argmax(out).item()

    anomaly = attack == 1
    state = rl.get_state(traffic)
    action_idx = rl.choose_action(state)

    action = rl.actions[action_idx]

    reward = rl.reward(anomaly, action_idx)
    rl.update(state, action_idx, reward, state)
    placeholder.metric(
        label="Traffic Flow",
        value=f"{traffic:.2f}",
        delta=f"Prediction: {pred:.2f}"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("State", state)
    col2.metric("Action (RL)", action)

    if anomaly:
        col3.error("🚨 DDoS DETECTED")
    else:
        col3.success("✅ NORMAL TRAFFIC")
    chart.add_rows([traffic])

    time.sleep(0.2)
