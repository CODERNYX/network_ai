import torch
import torch.nn as nn
import numpy as np

# =====================================
# 1. DDoS DETECTION MODEL (Neural Net)
# =====================================

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


# =====================================
# 2. TRAFFIC PREDICTION MODEL (LSTM)
# =====================================

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


# =====================================
# 3. REINFORCEMENT LEARNING AGENT
# =====================================

class RLAgent:

    def __init__(self):
        self.q_table = np.zeros((10, 3))
        self.actions = ["NORMAL", "MONITOR", "THROTTLE"]

    def get_state(self, traffic):
        return min(int(traffic / 200), 9)

    def choose_action(self, state):
        return np.argmax(self.q_table[state])

    def reward(self, anomaly, action):

        if anomaly and action == 2:
            return 10
        elif anomaly and action != 2:
            return -10
        elif not anomaly and action == 2:
            return -5
        else:
            return 1

    def update(self, state, action, reward, next_state, alpha=0.1, gamma=0.9):

        best_next = np.max(self.q_table[next_state])

        self.q_table[state, action] += alpha * (
            reward + gamma * best_next - self.q_table[state, action]
        )
