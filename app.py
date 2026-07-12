import torch
import torch.nn as nn
import numpy as np


# =========================================================
# 1. DDOS DETECTION MODEL
# =========================================================

class DDoSNet(nn.Module):

    def __init__(self, input_size=5):

        super().__init__()

        self.model = nn.Sequential(

            nn.Linear(input_size, 64),

            nn.ReLU(),

            nn.Linear(64, 32),

            nn.ReLU(),

            nn.Linear(32, 2)     # Normal / Attack
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


        self.fc = nn.Linear(

            64,

            1

        )


    def forward(self, x):

        out, _ = self.lstm(x)

        prediction = self.fc(

            out[:, -1, :]

        )

        return prediction




# =========================================================
# 3. REINFORCEMENT LEARNING AGENT
# =========================================================

class RLAgent:


    def __init__(self):


        # =================================================
        # Q TABLE
        #
        # 10 traffic states
        # 3 possible actions
        #
        # Action:
        # 0 -> NORMAL
        # 1 -> MONITOR
        # 2 -> THROTTLE
        #
        # =================================================

        self.q_table = np.zeros(

            (10,3)

        )


        # Available actions

        self.actions = [

            "NORMAL",

            "MONITOR",

            "THROTTLE"

        ]


        # =================================================
        # Q Learning Parameters
        # =================================================

        self.alpha = 0.1       # Learning rate

        self.gamma = 0.9       # Discount factor


        # Exploration parameters

        self.epsilon = 1.0

        self.epsilon_decay = 0.95

        self.min_epsilon = 0.05




    # =====================================================
    # STATE GENERATION
    # =====================================================

    def get_state(

            self,

            traffic,

            attack_prob

        ):


        traffic = abs(float(traffic))



        # Traffic level mapping

        if traffic < 0.5:

            state = 0


        elif traffic < 1.0:

            state = 2


        elif traffic < 2.0:

            state = 5


        elif traffic < 3.5:

            state = 7


        else:

            state = 9




        # Increase severity if attack probability high

        if attack_prob > 0.7:

            state = min(

                9,

                state + 1

            )



        return state




    # =====================================================
    # CHOOSE ACTION USING RL
    # =====================================================

    def choose_action(

            self,

            state

        ):


        # =================================================
        # Exploration
        # Randomly try all actions
        # =================================================

        if np.random.random() < self.epsilon:


            action = np.random.choice(

                [0,1,2]

            )



        # =================================================
        # Exploitation
        # Select highest Q value action
        # =================================================

        else:


            action = np.argmax(

                self.q_table[state]

            )



        return action





    # =====================================================
    # RETURN ACTION NAME
    # =====================================================

    def get_action_name(

            self,

            action

        ):


        return self.actions[action]





    # =====================================================
    # REWARD FUNCTION
    # =====================================================

    def calculate_reward(

            self,

            anomaly,

            action

        ):



        # =================================================
        # Normal Traffic
        # =================================================

        if anomaly == False:



            if action == 0:

                # Correct normal decision

                return 8



            elif action == 1:

                # Monitoring unnecessary

                return 3



            else:

                # Wrong throttling

                return -5




        # =================================================
        # Attack Traffic
        # =================================================

        else:



            if action == 2:


                # Correct mitigation

                return 10



            elif action == 1:


                # Partial protection

                return 4



            else:


                # Missed attack

                return -10





    # =====================================================
    # Q TABLE UPDATE
    # =====================================================

    def update(

            self,

            state,

            action,

            reward,

            next_state

        ):



        current_q = self.q_table[

            state,

            action

        ]



        max_future_q = np.max(

            self.q_table[next_state]

        )



        new_q = current_q + self.alpha * (

            reward

            +

            self.gamma * max_future_q

            -

            current_q

        )



        self.q_table[

            state,

            action

        ] = new_q




        # Reduce exploration

        self.epsilon = max(

            self.min_epsilon,

            self.epsilon * self.epsilon_decay

        )





    # =====================================================
    # DISPLAY Q TABLE
    # =====================================================

    def print_q_table(self):


        print("\n========== Q TABLE ==========")

        print(

            self.q_table

        )

        print(

            "=============================\n"

        )





# =========================================================
# TEST RL AGENT
# =========================================================

if __name__ == "__main__":


    agent = RLAgent()



    print("\nRL ACTION TEST\n")



    for i in range(20):


        traffic = np.random.uniform(

            0,

            5

        )


        attack_probability = np.random.uniform(

            0,

            1

        )



        state = agent.get_state(

            traffic,

            attack_probability

        )



        action = agent.choose_action(

            state

        )



        print(

            "Traffic:",

            round(traffic,2),

            "Attack Probability:",

            round(attack_probability,2),

            "State:",

            state,

            "Action:",

            agent.get_action_name(action)

        )



    agent.print_q_table()
