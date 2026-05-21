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

        # Q-table:
        # 10 states
        # 3 actions
        self.q_table = np.zeros((10, 3))

        # Available actions
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
    # EPSILON-GREEDY ACTION SELECTION
    # =====================================================

    def choose_action(self, state, epsilon=0.3):

        # Exploration:
        # randomly try actions

        if np.random.rand() < epsilon:

            return np.random.randint(
                0,
                len(self.actions)
            )

        # Exploitation:
        # choose best known action

        return np.argmax(
            self.q_table[state]
        )

    # =====================================================
    # REWARD FUNCTION
    # =====================================================

    def reward(self, anomaly, action):

        # =================================================
        # NORMAL TRAFFIC
        # =================================================

        if not anomaly:

            # NORMAL action during normal traffic
            if action == 0:

                return 8

            # MONITOR action during normal traffic
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
    # Q-LEARNING UPDATE
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
    # OPTIONAL:
    # VIEW Q-TABLE
    # =====================================================

    def print_q_table(self):

        print("\n===== Q TABLE =====")

        print(self.q_table)