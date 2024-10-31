import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# Import SUMO and TraCI modules
import os
import sys
import optparse
import random

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Replay memory
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = self.model(state)
        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            q_values = self.model(state)
            next_q_values = self.model(next_state)
            target = reward + (1 - done) * self.gamma * next_q_values.max(dim=1)[0]
            target = target.unsqueeze(1)
            target_f = q_values.clone()
            target_f.scatter_(1, torch.LongTensor([[action]]), target)
            self.optimizer.zero_grad()
            loss = self.criterion(q_values, target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def get_state():
    # Retrieve state information from SUMO (e.g., number of vehicles in each lane)
    # You can use TraCI commands to query the current state of the simulation
    # For example, you might want to retrieve the number of vehicles waiting at each traffic light
    # and any other relevant features that might help the agent make decisions
    # Return state as a numpy array
    state = []
    for tls_id in traci.trafficlight.getIDList():
        lanes = traci.trafficlight.getControlledLanes(tls_id)
        vehicles_waiting = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in lanes)
        state.append(vehicles_waiting)
    return np.array(state)

def get_reward():
    # Calculate reward based on the current state
    # You can define the reward function based on the average time a car stops during the red signal
    # For example, you might want to penalize the agent based on the total number of vehicles waiting during the red signal
    # Return a scalar reward value
    reward = 0
    for tls_id in traci.trafficlight.getIDList():
        lanes = traci.trafficlight.getControlledLanes(tls_id)
        for lane in lanes:
            waiting_time = traci.lane.getWaitingTime(lane)
            reward -= waiting_time  # Penalize waiting time during red signal
    return reward

def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options

# Environment setup
state_size = 8  # Example state size
action_size = 4  # Example action size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = DQNAgent(state_size, action_size)
batch_size = 32

options = get_options()

if options.nogui:
        sumoBinary = checkBinary('sumo')
else:
    sumoBinary = checkBinary('sumo-gui')


# Training loop
EPISODES = 1000
for e in range(EPISODES):
    # Reset the environment, get initial state
    # traci.start(["sumo-gui", "-c", "your_sumo_config_file.sumocfg"])
    traci.start([sumoBinary, "-c", "/home/poison/RL/Final/cross/cross.sumocfg",
                             "--tripinfo-output", "tripinfo.xml"])
    state = get_state()
    for time in range(100):  # Adjust based on simulation time steps
        # Choose action
        action = agent.act(state)

        # Take action in SUMO (e.g., change traffic light phase)
        # Implement your code to change traffic light phase based on the chosen action
        for tls_id in traci.trafficlight.getIDList():
            # Map action to traffic light phase
            phase_map = {0: "GrGr", 1: "yryr", 2: "rGrG", 3: "ryry"}
            traci.trafficlight.setPhase(tls_id, phase_map[action])

        # Get reward
        reward = get_reward()

        # Get next state
        next_state = get_state()

        # Check if episode is done
        done = False  # Define termination condition

        # Store experience in replay memory
        agent.remember(state, action, reward, next_state, done)

        # Update state
        state = next_state

        # Perform experience replay
        agent.replay(batch_size)

        # Check if episode is done
        if done:
            break

    # Print episode info, save model checkpoints, etc.
    print("Episode: {}/{}, Epsilon: {:.2}".format(e, EPISODES, agent.epsilon))
    if e % 10 == 0:
        torch.save(agent.model.state_dict(), "model_{}.pt".format(e))

    traci.close()
