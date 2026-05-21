import torch
import torch.nn as nn

# =========================================================
# DDoS DETECTION MODEL
# =========================================================

class DDoSNet(nn.Module):

    def __init__(self, input_size=5):

        super().__init__()

        self.model = nn.Sequential(

            nn.Linear(input_size, 64),

            nn.ReLU(),

            nn.Linear(64, 32),

            nn.ReLU(),

            nn.Linear(32, 2)
        )

    def forward(self, x):

        return self.model(x)

# =========================================================
# LSTM TRAFFIC PREDICTION
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
# RL AGENT
# =========================================================

class RLAgent:

    def __init__(self):

        self.actions = [

            "NORMAL",

            "MONITOR",

            "THROTTLE"
        ]

    # =====================================================
    # GET STATE
    # =====================================================

    def get_state(self,traffic,attack_probability):

        traffic = abs(float(traffic))

        probability = float(attack_probability)

        # SAFE TRAFFIC

        if probability < 0.40 and traffic < 1.0:

            return "LOW"

        # SUSPICIOUS TRAFFIC

        elif probability < 0.70 and traffic < 2.5:

            return "MEDIUM"

        # DANGEROUS TRAFFIC

        else:

            return "HIGH"

    # =====================================================
    # CHOOSE ACTION
    # =====================================================

    def choose_action(self, state):

        if state == "LOW":

            return 0

        elif state == "MEDIUM":

            return 1

        else:

            return 2

    # =====================================================
    # REWARD
    # =====================================================

    def reward(self, anomaly, action):

        if not anomaly:

            if action == 0:

                return 10

            elif action == 1:

                return 3

            else:

                return -10

        else:

            if action == 2:

                return 10

            elif action == 1:

                return 4

            else:

                return -10

    # =====================================================
    # UPDATE
    # =====================================================

    def update(

        self,

        state,

        action,

        reward,

        next_state
    ):

        pass