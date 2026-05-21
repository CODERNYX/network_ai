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
    # GET STATE
    # =====================================================

    def get_state(self, traffic):

        state = int(abs(traffic) / 200)

        return max(0, min(state, 9))

    # =====================================================
    # DETERMINISTIC ACTION SELECTION
    # =====================================================

    def choose_action(self, state):

        # =================================================
        # LOW TRAFFIC
        # =================================================

        if state <= 2:

            return 0      # NORMAL

        # =================================================
        # MEDIUM TRAFFIC
        # =================================================

        elif state <= 5:

            return 1      # MONITOR

        # =================================================
        # HIGH TRAFFIC
        # =================================================

        else:

            return 2      # THROTTLE

    # =====================================================
    # REWARD FUNCTION
    # =====================================================

    def reward(self, anomaly, action):

        # =================================================
        # NORMAL TRAFFIC
        # =================================================

        if not anomaly:

            if action == 0:

                return 8

            elif action == 1:

                return 3

            else:

                return -5

        # =================================================
        # ATTACK TRAFFIC
        # =================================================

        else:

            if action == 2:

                return 10

            elif action == 1:

                return 4

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

        print("\n===== Q TABLE =====")

        print(self.q_table)