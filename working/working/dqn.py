import logging
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Import SUMO and TraCI modules
import os
import sys

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci

MAX_TRAFFIC_FLOW_REWARD = 10
QUEUE_LENGTH_PENALTY = -5
DELAY_PENALTY = 5

hidden_size = 128


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Replay memory
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DuelingDQN(state_size, action_size).to(device)
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
            target_f.scatter_(1, torch.LongTensor([[action]]).to(device), target)
            self.optimizer.zero_grad()
            loss = self.criterion(q_values, target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingDQN, self).__init__()
        self.hidden_size = hidden_size  # Store for clarity

        # Common feature layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # Value stream
        self.fc_value = nn.Linear(hidden_size, 1)

        # Advantage stream
        self.fc_advantage = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        # Split into value and advantage streams
        value = self.fc_value(x)
        advantage = self.fc_advantage(x)

        # Combine streams to get Q-values
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values
   
def generate_routefile(seed):
    random.seed(seed)
    N = 3600

    pWE= 1. / 10
    pEW = 1. / 11
    pNS = 1. / 10
    pSN = 1. / 11

    with open("working/route_gen.rou.xmll","w") as routes:
        print("""<routes>
        <vType id="car" vClass="passenger" length="4" maxSpeed="25.0" accel="2.6" decel="4.5"/>
        <vType id="truck" vClass="truck" length="10" maxSpeed="20.0" accel="1.2" decel="2.5"/>

        <route id="r_0" edges="3i 1o"/>
        <route id="r_1" edges="3i 4o"/>
        <route id="r_10" edges="2i 3o"/>
        <route id="r_11" edges="2i 4o"/>
        <route id="r_2" edges="3i 2o"/>
        <route id="r_3" edges="4i 3o"/>
        <route id="r_4" edges="4i 2o"/>
        <route id="r_5" edges="4i 1o"/>
        <route id="r_6" edges="1i 2o"/>
        <route id="r_7" edges="1i 4o"/>
        <route id="r_8" edges="1i 3o"/>
        <route id="r_9" edges="2i 1o"/>""", file=routes)

        vehicle_num = 0
        vclasses = ["car","truck"]
        routes_dict = {'WE':['r_6','r_7','r_8'],'SN':['r_0','r_1','r_2'],'EW':['r_3','r_4','r_5'],'NS':['r_9','r_10','r_11']}

        weights_vclass = [10,1]
        weights_route = [1,1,1]

        for i in range(N):

            if random.uniform(0, 1) < pWE:

                vclass_type = random.choices(vclasses,weights=weights_vclass)[0]
                v_route = random.choices(routes_dict['WE'],weights = weights_route)[0]
        
                print(f'    <vehicle id="WE_{i}" type="{vclass_type}" route="{v_route}" depart="{str(i)}" />',file = routes)
                vehicle_num += 1
            
            if random.uniform(0, 1) < pEW:

                vclass_type = random.choices(vclasses,weights=weights_vclass)[0]
                v_route = random.choices(routes_dict['EW'],weights=weights_route)[0]

                print(f'    <vehicle id="EW_{i}" type="{vclass_type}" route="{v_route}" depart="{str(i)}" color= "1,0,0"/>',file = routes)
                vehicle_num += 1
            
            if random.uniform(0, 1) < pNS:

                vclass_type = random.choices(vclasses,weights=weights_vclass)[0]
                v_route = random.choices(routes_dict['NS'],weights=weights_route)[0]

                print(f'    <vehicle id="NS_{i}" type="{vclass_type}" route="{v_route}" depart="{str(i)}" color="0,1,0"/>',file = routes)
                vehicle_num += 1
            
            if random.uniform(0, 1) < pSN:

                vclass_type = random.choices(vclasses,weights=weights_vclass)[0]
                v_route = random.choices(routes_dict['SN'],weights=weights_route)[0]
        
                print(f'    <vehicle id="SN_{i}" type="{vclass_type}" route="{v_route}" depart="{str(i)}" color="0,0,1"/>',file = routes)
                vehicle_num += 1

        print("</routes>", file=routes)


def get_state():
    """
    Retrieve state information from SUMO.

    Returns:
        state (numpy.ndarray): Array containing state information.
    """
    state = []
    try:
        for tls_id in traci.trafficlight.getIDList():
            # print(tls_id)
            lanes = traci.trafficlight.getControlledLanes(tls_id)
            vehicles_waiting = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in lanes)
            state.append(vehicles_waiting)
    except Exception as e:
        logging.error(f"Error occurred during state retrieval: {e}")
    return np.array(state)

def get_traffic_light_ids():
    """
    Retrieve IDs of all traffic light signals in the SUMO simulation.

    Returns:
        tls_ids (list): List of traffic light signal IDs.
    """
    tls_ids = traci.trafficlight.getIDList()
    return tls_ids


def get_traffic_congestion_ratio(tls_id):
    # Get the total number of vehicles
    total_vehicles = len(traci.vehicle.getIDList())

    # Get the number of waiting vehicles at traffic lights
    waiting_vehicles = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in traci.trafficlight.getControlledLanes(tls_id))

    # Calculate traffic congestion ratio
    congestion_ratio = waiting_vehicles / total_vehicles if total_vehicles > 0 else 0

    return congestion_ratio

def get_average_queue_length(lanes):

    # Get the total queue length
    total_queue_length = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in lanes)

    # Calculate average queue length
    average_queue_length = total_queue_length / len(lanes) if len(lanes) > 0 else 0

    return average_queue_length

def get_average_waiting_time():
    # Get the list of vehicle IDs
    vehicle_ids = traci.vehicle.getIDList()

    # Get the total waiting time of vehicles
    total_waiting_time = sum(traci.vehicle.getWaitingTime(vehicle_id) for vehicle_id in vehicle_ids)

    # Calculate average waiting time
    average_waiting_time = total_waiting_time / len(vehicle_ids) if len(vehicle_ids) > 0 else 0

    return average_waiting_time

# def get_reward():
    """
    Calculate reward based on multiple metrics:
    1. Traffic Flow: Reward for maximizing traffic flow efficiency (minimizing congestion).
    2. Queue Length: Penalty for long vehicle queues at intersections.
    3. Delay or Waiting Time: Penalty for high average delay or waiting time experienced by vehicles.
    4. Safety: Not included in this function for simplicity.
    5. Energy Consumption: Not included in this function for simplicity.
    6. Emergency Vehicle Priority: Reward for efficiently prioritizing emergency vehicles.
    7. User Satisfaction: Not included in this function for simplicity.
    8. Adaptability: Not directly included in this function; it could be incorporated by encouraging dynamic adjustments.

    Returns:
        reward (float): Scalar reward value.
    """
    reward = 0
    try:
        # tls_ids = get_traffic_light_ids
        # print(len(tls_ids))
        # Traffic Flow: Reward for maximizing traffic flow efficiency
        # reward += MAX_TRAFFIC_FLOW_REWARD * (1 - get_traffic_congestion_ratio(tls_id))

        # Queue Length: Penalty for long vehicle queues at intersections
        # reward -= QUEUE_LENGTH_PENALTY * get_average_queue_length()

        # Delay or Waiting Time: Penalty for high average delay or waiting time experienced by vehicles
        reward -= DELAY_PENALTY * get_average_waiting_time()

    except Exception as e:
        logging.error(f"Error occurred during reward calculation: {e}")
    return reward

def get_reward():
    """
    Calculate reward based on multiple metrics:
    1. Traffic Flow: Reward for maximizing traffic flow efficiency (minimizing congestion).
    2. Queue Length: Penalty for long vehicle queues at intersections.
    3. Delay or Waiting Time: Penalty for high average delay or waiting time experienced by vehicles.
    4. Safety: Not included in this function for simplicity.
    5. Energy Consumption: Not included in this function for simplicity.
    6. Emergency Vehicle Priority: Reward for efficiently prioritizing emergency vehicles.
    7. User Satisfaction: Not included in this function for simplicity.
    8. Adaptability: Not directly included in this function; it could be incorporated by encouraging dynamic adjustments.

    Returns:
        reward (float): Scalar reward value.
    """
    reward = 0
    try:
        for tls_id in traci.trafficlight.getIDList():
            # print(tls_id)
            lanes = traci.trafficlight.getControlledLanes(tls_id)

            # Traffic Flow: Reward for maximizing traffic flow efficiency
            reward += MAX_TRAFFIC_FLOW_REWARD * (1 - get_traffic_congestion_ratio(tls_id))

            # Queue Length: Penalty for long vehicle queues at intersections
            reward -= QUEUE_LENGTH_PENALTY * get_average_queue_length(lanes)

            # Delay or Waiting Time: Penalty for high average delay or waiting time experienced by vehicles
            reward -= DELAY_PENALTY * get_average_waiting_time()

    except Exception as e:
        logging.error(f"Error occurred during reward calculation: {e}")
    return reward

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(filename='simulation.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

    # Environment setup
    state_size = 1  # Example state size
    action_size = 4  # Example action size
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    print("Running on Device:", device)
    agent = DQNAgent(state_size, action_size)
    batch_size = 32

    # Training loop
    EPISODES = 50

    total_rewards = []
    for episode in range(1,EPISODES+1):

        print("\nEpisode:{} started".format(episode))
        
        try:
            generate_routefile(episode)
            print("Route File generated successfully for episode:{}".format(episode))

        except Exception as e:
            logging.error("Error occured during route generation in episode{}:{}".format(episode,str(e)))

        try:
            # Start SUMO in no-GUI mode
            traci.start(["sumo", "-c", "working/fukk_1.sumocfg", "--no-warnings", "--quit-on-end"])

            state = get_state()
            total_reward = 0
            for time in range(1000):  # Adjust based on simulation time steps
                action = agent.act(state)

                for tls_id in traci.trafficlight.getIDList():
                    phase_map = {0: 0, 1: 1, 2: 2, 3: 3}
                    traci.trafficlight.setPhase(tls_id, phase_map[action])

                traci.simulationStep()

                reward = get_reward()
                total_reward += reward

                next_state = get_state()

                done = False

                agent.remember(state, action, reward, next_state, done)

                state = next_state

                agent.replay(batch_size)

                if done:
                    break

            print("Episode: {}/{}, Total Reward: {:.2f}, Epsilon: {:.2}".format(episode, EPISODES, total_reward, agent.epsilon))
            total_rewards.append(total_reward)
            if episode % 10 == 0:
                torch.save(agent.model.state_dict(), "model_{}.pt".format(episode))

        except Exception as e:
            logging.error("Error occurred during episode# {}: {}".format(episode, str(e)))

        finally:
            try:
                traci.close()
            except Exception as e:
                logging.error(f"Error occurred during close: {e}")

    plt.plot(total_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward per Episode')
    plt.show()
