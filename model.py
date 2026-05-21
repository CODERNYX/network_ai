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
        # Q TABLE
        # =================================================

        self.q_table = np.zeros((10, 3))

        # =================================================
        # ACTIONS
        # =================================================

        self.actions = [

            "NORMAL",

            "MONITOR",

            "THROTTLE"
        ]

    # =====================================================
    # GET STATE BASED ON TRAFFIC
    # =====================================================

    def get_state(self, traffic):

        traffic = abs(float(traffic))

        # =================================================
        # VERY LOW TRAFFIC
        # =================================================

        if traffic < 0.5:

            return 0

        # =================================================
        # LOW TRAFFIC
        # =================================================

        elif traffic < 1.0:

            return 2

        # =================================================
        # MEDIUM TRAFFIC
        # =================================================

        elif traffic < 2.0:

            return 5

        # =================================================
        # HIGH TRAFFIC
        # =================================================

        elif traffic < 3.5:

            return 7

        # =================================================
        # VERY HIGH TRAFFIC
        # =================================================

        else:

            return 9

    # =====================================================
    # ACTION SELECTION
    # =====================================================

    def choose_action(self, state):

        # =================================================
        # NORMAL ACTION
        # =================================================

        if state <= 2:

            return 0

        # =================================================
        # MONITOR ACTION
        # =================================================

        elif state <= 6:

            return 1

        # =================================================
        # THROTTLE ACTION
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

            # NORMAL action
            if action == 0:

                return 8

            # MONITOR action
            elif action == 1:

                return 3

            # THROTTLE during normal traffic
            else:

                return -5

        # =================================================
        # ATTACK TRAFFIC
        # =================================================

        else:

            # THROTTLE during attack
            if action == 2:

                return 10

            # MONITOR during attack
            elif action == 1:

                return 4

            # NORMAL during attack
            else:

                return -10

    # =====================================================
    # Q LEARNING UPDATE
    # =====================================================

    def update(

        self,

        state,

        action,

        reward,

        next_state,

        alpha=0.1,

        gamma=0.9
    ):

        best_next = np.max(
            self.q_table[next_state]
        )

        self.q_table[state, action] += alpha * (

            reward +

            gamma * best_next -

            self.q_table[state, action]
        )

    # =====================================================
    # PRINT Q TABLE
    # =====================================================

    def print_q_table(self):

        print("\n========== Q TABLE ==========")

        print(self.q_table)

        print("=============================\n")