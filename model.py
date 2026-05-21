import torch
import torch.nn as nn
import numpy as np

# =========================================================
# 1. DDoS DETECTION MODEL
# =========================================================

class DDoSNet(nn.Module):

    def __init__(self, input_size=5):

        super().__init__()

        self.model = nn.Sequential(

            nn.Linear(input_size, 64),

            nn.ReLU(),

            nn.Linear(64, 32),

            nn.ReLU(),

            nn.Linear(32, 2)   # Normal / Attack
        )

    def forward(self, x):

        return self.model(x)

# =========================================================
# 2. TRAFFIC PREDICTION MODEL
# =========================================================

class LSTMModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM(

            input_size=1,

            hidden_size=64,

            num_layers=2,

            batch_first=True
        )

        self.fc = nn.Linear(64, 1)

    def forward(self, x):

        out, _ = self.lstm(x)

        return self.fc(out[:, -1, :])

# =========================================================
# 3. REINFORCEMENT LEARNING AGENT
# =========================================================

class RLAgent:

    def __init__(self):

        # =================================================
        # ACTIONS
        # =================================================

        self.actions = [

            "NORMAL",

            "MONITOR",

            "THROTTLE"
        ]

    # =====================================================
    # GET STATE
    # =====================================================

    def get_state(

        self,

        traffic,

        attack_probability
    ):

        traffic = abs(float(traffic))

        probability = float(attack_probability)

        # =================================================
        # LOW RISK
        # =================================================

        if probability < 0.40 and traffic < 1.0:

            return "LOW"

        # =================================================
        # MEDIUM RISK
        # =================================================

        elif probability < 0.70 and traffic < 2.5:

            return "MEDIUM"

        # =================================================
        # HIGH RISK
        # =================================================

        else:

            return "HIGH"

    # =====================================================
    # ACTION SELECTION
    # =====================================================

    def choose_action(self, state):

        # =================================================
        # SAFE TRAFFIC
        # =================================================

        if state == "LOW":

            return 0

        # =================================================
        # SUSPICIOUS TRAFFIC
        # =================================================

        elif state == "MEDIUM":

            return 1

        # =================================================
        # ATTACK TRAFFIC
        # =================================================

        else:

            return 2

    # =====================================================
    # REWARD FUNCTION
    # =====================================================

    def reward(self, anomaly, action):

        # =================================================
        # NORMAL TRAFFIC
        # =================================================

        if not anomaly:

            # NORMAL ACTION
            if action == 0:

                return 10

            # MONITOR ACTION
            elif action == 1:

                return 3

            # THROTTLE ACTION
            else:

                return -10

        # =================================================
        # ATTACK TRAFFIC
        # =================================================

        else:

            # THROTTLE ACTION
            if action == 2:

                return 10

            # MONITOR ACTION
            elif action == 1:

                return 4

            # NORMAL ACTION
            else:

                return -10

    # =====================================================
    # UPDATE FUNCTION
    # =====================================================

    def update(

        self,

        state,

        action,

        reward,

        next_state
    ):

        # =================================================
        # DETERMINISTIC RL SYSTEM
        # =================================================

        # No Q-table update needed because
        # actions are selected based on
        # real-time traffic + ML probability

        pass

    # =====================================================
    # OPTIONAL DEBUG FUNCTION
    # =====================================================

    def print_status(

        self,

        traffic,

        probability,

        state,

        action
    ):

        print("\n==============================")

        print(f"Traffic: {traffic}")

        print(f"Attack Probability: {probability}")

        print(f"State: {state}")

        print(f"Action: {action}")

        print("==============================\n")