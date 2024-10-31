import logging
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque

# Import SUMO and TraCI modules
import os
import sys

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci

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

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
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
            return None
        minibatch = random.sample(self.memory, batch_size)
        losses = []
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            q_values = self.model(state)
            next_q_values = self.model(next_state)
            target = reward + (1 - done) * self.gamma * next_q_values.max(dim=1)[0]
            target = target.unsqueeze(1)
            target_f = q_values.clone()
            target_f.scatter_(1, torch.LongTensor([[action]]).to(device), target)
            self.optimizer.zero_grad()
            loss = self.criterion(target_f, q_values)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return sum(losses) / len(losses)

if __name__ == "__main__":
    logging.basicConfig(filename='simulation.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    state_size = 10  # Define based on your environment
    action_size = 4  # Define based on your environment
    agent = DQNAgent(state_size, action_size)
    batch_size = 32
    EPISODES = 3

    cumulative_rewards = []
    average_losses = []
    steps_per_episode = []

    for e in range(EPISODES):
        total_reward = 0
        total_loss = 0
        step_count = 0

        try:
            traci.start(["sumo-gui", "-c", "twoone/twoone.sumocfg"])
            state = np.random.rand(state_size)  # Replace with actual state initialization
            while True:
                action = agent.act(state)
                # Implement your SUMO logic here
                reward = np.random.rand()  # Replace with actual reward calculation
                next_state = np.random.rand(state_size)  # Replace with actual state transition
                done = step_count > 1000  # Implement your termination logic
                agent.remember(state, action, reward, next_state, done)
                loss = agent.replay(batch_size)
                if loss is not None:
                    total_loss += loss

                state = next_state
                total_reward += reward
                step_count += 1
                if done:
                    break

            cumulative_rewards.append(total_reward)
            average_losses.append(total_loss / step_count if step_count > 0 else 0)
            steps_per_episode.append(step_count)

            if (e + 1) % 10 == 0:
                print(f"Episode: {e+1}/{EPISODES}, Reward: {total_reward}, Average Loss: {average_losses[-1]}, Steps: {step_count}, Epsilon: {agent.epsilon:.2f}")

        except Exception as err:
            logging.error(f"Error during episode {e+1}: {err}")

        finally:
            traci.close()

    # Plotting the results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.plot(cumulative_rewards)
    plt.title('Cumulative Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')

    plt.subplot(1, 3, 2)
    plt.plot(average_losses)
    plt.title('Average Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Loss')

    plt.subplot(1, 3, 3)
    plt.plot(steps_per_episode)
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')

    plt.tight_layout()
    plt.show()