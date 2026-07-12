import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import time
import importlib

import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler


# =========================================================
# FORCE RELOAD MODEL
# =========================================================

import model

importlib.reload(model)


DDoSNet = model.DDoSNet
LSTMModel = model.LSTMModel
RLAgent = model.RLAgent



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

st.markdown(
"""
<style>


.block-container {

    padding-top:1rem;

    padding-left:1rem;

    padding-right:1rem;

}


[data-testid="metric-container"] {

    background:#111111;

    border:1px solid #333333;

    padding:12px;

    border-radius:12px;

    text-align:center;

}



.stPlotlyChart {

    border-radius:12px;

}


.scroll-log {

    height:65vh;

    overflow-y:auto;

    padding:12px;

    background:#111111;

    border-radius:12px;

    border:1px solid #444444;

}


.log-card {

    padding:10px;

    margin-bottom:10px;

    background:#1e1e1e;

    color:white;

    border-radius:10px;

    font-size:14px;

}


</style>

""",

unsafe_allow_html=True

)



# =========================================================
# TITLE
# =========================================================

st.title(
"🚀 AI vs Rule-Based Network Security Dashboard"
)


st.caption(
"AI Driven SDN Security Monitoring using DDoSNet + LSTM + Reinforcement Learning"
)



# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header(
"⚙️ Controls"
)



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
# LOAD DATASET
# =========================================================


@st.cache_data

def load_data():


    df = pd.read_parquet(

        "clean_ddos_dataset.parquet"

    )


    df = df.replace(

        [np.inf,-np.inf],

        np.nan

    )


    df = df.dropna()


    return df



df = load_data()



original_df = df.copy()



# =========================================================
# LOAD AI MODELS
# =========================================================


@st.cache_resource

def load_models():


    # ----------------------------
    # DDoS Model
    # ----------------------------

    ddos_model = DDoSNet(

        input_size=5

    )


    ddos_model.load_state_dict(

        torch.load(

            "ddos_model.pth",

            map_location="cpu"

        )

    )


    ddos_model.eval()



    # ----------------------------
    # LSTM Model
    # ----------------------------


    lstm_model = LSTMModel()



    lstm_model.load_state_dict(

        torch.load(

            "lstm_model.pth",

            map_location="cpu"

        )

    )


    lstm_model.eval()



    # ----------------------------
    # RL Agent
    # ----------------------------

    rl = RLAgent()



    return (

        ddos_model,

        lstm_model,

        rl

    )



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

    x for x in features

    if x in df.columns

]



# =========================================================
# SCALE FEATURES
# =========================================================


scaler = StandardScaler()



df[features] = scaler.fit_transform(

    df[features]

)





# =========================================================
# TABS
# =========================================================


tabs = st.tabs(

[

"📊 Dashboard",

"📈 Traffic",

"🤖 ML vs Rule",

"🔥 Probability",

"🧠 RL Decision",

"📜 Logs",

"🚀 Technology Comparison"

]

)





# =========================================================
# DASHBOARD PLACEHOLDERS
# =========================================================


with tabs[0]:


    st.subheader(
    "📊 Real-Time Network Metrics"
    )


    top1,top2 = st.columns(2)


    mid1,mid2 = st.columns(2)


    bottom = st.container()





# =========================================================
# GRAPH PLACEHOLDERS
# =========================================================


with tabs[1]:

    traffic_chart = st.empty()



with tabs[2]:

    comparison_chart = st.empty()



with tabs[3]:

    probability_chart = st.empty()



with tabs[4]:

    rl_chart = st.empty()



with tabs[5]:

    st.subheader(
    "📜 Live Detection Console"
    )

    log_div = st.empty()

# =========================================================
# STORAGE
# =========================================================


traffic_history = []

ml_history = []

rule_history = []

attack_prob_history = []

rl_action_history = []

logs_html = ""


seq = []

SEQ_LEN = 10



ml_detected = 0

rule_detected = 0

normal_count = 0





# =========================================================
# MAIN PROCESSING LOOP
# =========================================================


for i in range(

    min(500,len(df))

):


    row = df.iloc[i]


    orig_row = original_df.iloc[i]



    # =====================================================
    # TRAFFIC EXTRACTION
    # =====================================================


    traffic = abs(

        float(

            orig_row["Flow Bytes/s"]

        )

    )


    traffic_history.append(

        traffic

    )




    # =====================================================
    # LSTM TRAFFIC PREDICTION
    # =====================================================


    seq.append(

        [traffic]

    )


    if len(seq) > SEQ_LEN:

        seq.pop(0)




    if len(seq)==SEQ_LEN:


        lstm_input = torch.tensor(

            seq,

            dtype=torch.float32

        ).unsqueeze(0)



        predicted_traffic = lstm_model(

            lstm_input

        ).item()



    else:


        predicted_traffic = 0




    # =====================================================
    # DDOS DETECTION USING DDoSNet
    # =====================================================



    input_features = torch.tensor(

    [

        row["Flow Duration"],

        row["Total Fwd Packets"],

        row["Total Backward Packets"],

        row["Flow Bytes/s"],

        row["Flow Packets/s"]

    ],

    dtype=torch.float32

    ).unsqueeze(0)





    output = ddos_model(

        input_features

    )



    probability = F.softmax(

        output,

        dim=1

    )



    attack_prob = probability[0][1].item()



    # =====================================================
    # ML DECISION
    # =====================================================



    if attack_prob > 0.50:


        ml_detection = "🚨 DDoS"


        ml_detected += 1



    else:


        ml_detection = "✅ Normal"


        normal_count += 1





    # =====================================================
    # RULE BASED DETECTION
    # =====================================================


    if (

        orig_row["Flow Packets/s"] > 10000

        or

        orig_row["Total Fwd Packets"] > 5000

        or

        orig_row["Flow Bytes/s"] > 1000000

    ):


        rule_detection = "🚨 DDoS"


        rule_detected += 1



    else:


        rule_detection = "✅ Normal"





    # =====================================================
    # REINFORCEMENT LEARNING AGENT
    # =====================================================


    # Generate state

    state = rl.get_state(

        traffic,

        attack_prob

    )



    # Select action

    action_index = rl.choose_action(

        state

    )



    # Convert index to name

    action = rl.get_action_name(

        action_index

    )



    # Calculate reward

    reward = rl.calculate_reward(

        ml_detection=="🚨 DDoS",

        action_index

    )



    # Next state

    next_state = rl.get_state(

        predicted_traffic,

        attack_prob

    )



    # Update Q-table

    rl.update(

        state,

        action_index,

        reward,

        next_state

    )



    rl_action_history.append(

        action

    )





    # =====================================================
    # STORE RESULTS
    # =====================================================


    ml_history.append(

        1 if ml_detection=="🚨 DDoS"

        else 0

    )



    rule_history.append(

        1 if rule_detection=="🚨 DDoS"

        else 0

    )



    attack_prob_history.append(

        attack_prob*100

    )





    # =====================================================
    # DASHBOARD METRICS
    # =====================================================



    with tabs[0]:


        top1.metric(

            "Traffic",

            f"{traffic:.2f}"

        )


        top2.metric(

            "Attack Probability",

            f"{attack_prob*100:.2f}%"

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

                "🧠 RL Action",

                action

            )



            st.caption(

                f"Reward : {reward}"

            )



    # =====================================================
    # GRAPH UPDATE
    # =====================================================


    if i % GRAPH_UPDATE_INTERVAL == 0:



        # =================================================
        # TRAFFIC GRAPH
        # =================================================


        fig1 = go.Figure()



        fig1.add_trace(

            go.Scatter(

                y=traffic_history,

                mode="lines",

                name="Traffic"

            )

        )



        fig1.update_layout(

            title="📈 Live Network Traffic",

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

                mode="lines",

                name="AI Detection"

            )

        )



        fig2.add_trace(

            go.Scatter(

                y=rule_history,

                mode="lines",

                name="Rule Detection"

            )

        )



        fig2.update_layout(

            title="🤖 AI Detection vs Rule Detection",

            height=350

        )



        comparison_chart.plotly_chart(

            fig2,

            use_container_width=True

        )






        # =================================================
        # ATTACK PROBABILITY GRAPH
        # =================================================



        fig3 = go.Figure()



        fig3.add_trace(

            go.Scatter(

                y=attack_prob_history,

                mode="lines",

                name="Attack Probability"

            )

        )



        fig3.update_layout(

            title="🔥 DDoS Attack Probability",

            yaxis_title="Probability (%)",

            height=350

        )



        probability_chart.plotly_chart(

            fig3,

            use_container_width=True

        )







        # =================================================
        # RL ACTION GRAPH
        # =================================================



        action_values = []



        for action in rl_action_history:


            if action == "NORMAL":

                action_values.append(0)


            elif action == "MONITOR":

                action_values.append(1)


            else:

                action_values.append(2)




        fig4 = go.Figure()



        fig4.add_trace(

            go.Scatter(

                y=action_values,

                mode="lines+markers",

                name="RL Decision"

            )

        )



        fig4.update_layout(

            title="🧠 Reinforcement Learning Decision",

            height=350,

            yaxis=dict(

                tickmode="array",

                tickvals=[0,1,2],

                ticktext=[

                    "NORMAL",

                    "MONITOR",

                    "THROTTLE"

                ]

            )

        )



        rl_chart.plotly_chart(

            fig4,

            use_container_width=True

        )







    # =====================================================
    # LIVE LOGS
    # =====================================================


    log_color = "#00c853"



    if ml_detection=="🚨 DDoS":

        log_color="#ff4b4b"




    logs_html += f"""

    <div class="log-card"

    style="border-left:6px solid {log_color};">


    <b>Packet:</b> {i}

    <br>


    <b>Traffic:</b>

    {traffic:.2f}


    <br>


    <b>ML Detection:</b>

    {ml_detection}


    <br>


    <b>Rule Detection:</b>

    {rule_detection}


    <br>


    <b>Attack Probability:</b>

    {attack_prob*100:.2f}%


    <br>


    <b>LSTM Prediction:</b>

    {predicted_traffic:.2f}


    <br>


    <b>RL Decision:</b>

    {action}


    <br>


    <b>Reward:</b>

    {reward}


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
    # SIMULATION SPEED
    # =====================================================


    time.sleep(

        max(speed,0.05)

    )





# =========================================================
# TECHNOLOGY COMPARISON
# =========================================================


with tabs[6]:


    st.subheader(

        "🚀 Existing Technology vs Proposed AI Security System"

    )



    comparison_data = {


        "Feature":[


            "Signature Based Detection",

            "Rule Based IDS",

            "Machine Learning Detection",

            "Traffic Prediction",

            "Reinforcement Learning",

            "Adaptive Response",

            "Real-Time Dashboard",

            "Edge Computing Support",

            "Cloud Integration",

            "Automatic Mitigation"


        ],



        "Traditional IDS":[


            1,

            1,

            0,

            0,

            0,

            0,

            0,

            0,

            0,

            0


        ],



        "ML IDS":[


            0,

            0,

            1,

            0,

            0,

            0,

            1,

            0,

            1,

            0


        ],



        "Proposed System":[


            1,

            1,

            1,

            1,

            1,

            1,

            1,

            1,

            1,

            1


        ]

    }




    comparison_df = pd.DataFrame(

        comparison_data

    )



    st.dataframe(

        comparison_df,

        use_container_width=True

    )





# =========================================================
# FINAL SUMMARY
# =========================================================


st.success(

    "✅ AI Security Simulation Completed Successfully"

)


st.write(

    "Total Packets Analysed:",

    min(500,len(df))

)


st.write(

    "ML DDoS Detections:",

    ml_detected

)


st.write(

    "Rule Based Detections:",

    rule_detected

)
