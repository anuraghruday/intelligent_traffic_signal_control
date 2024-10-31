import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import gym
from gym import spaces
from stable_baselines3 import PPO,A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import traci
import sumolib
import numpy as np
import random
import logging
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='traffic_signal_eval_ppo.log',
                    filemode='w')
logger = logging.getLogger(__name__)

configfile_path = "twoone.sumocfg"
routefile_path = "route_gen.rou.xml"

def generate_routefile():
    # random.seed(1)
    N = 3600

    pWE= 1. / 10
    pEW = 1. / 11
    pNS = 1. / 10
    pSN = 1. / 11

    with open(routefile_path,"w") as routes:
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
        logger.info("total no of vechiles generated: {}".format(vehicle_num))

class CustomEnv(gym.Env):
    def __init__(self):

        self.action_space = spaces.Discrete(2)  # Red and green phases only
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(25,), dtype=np.float32)
        # self.seed = 0
        
        
    def reset(self,**kwargs):
        try:
            # logger.info("reset function entered")           
            self.step_counter = 0
            return self._get_observation()
        except Exception as e:
            logger.error("An error occurred while resetting the environment: %s", str(e))
            raise e

    def step(self, action):
        try:
            for idx, tl_id in enumerate(traci.trafficlight.getIDList()):
                traci.trafficlight.setPhase(tl_id, action)
            traci.simulationStep()
            self.step_counter += 1
            next_observation = self._get_observation()
            reward = self._calculate_reward()
            done = self._is_done()
            info = {}
            # logger.info("Step %d completed.", self.step_counter)
            return next_observation, reward, done, info
        except Exception as e:
            logger.error("An error occurred during the step: %s", str(e))
            raise e

    def _get_observation(self):
        try:  # Begin try block
            waiting_times = []
            densities = []
            phases = []

            for tl_id in traci.trafficlight.getIDList():
                phases.append(traci.trafficlight.getPhase(tl_id))
                lanes = traci.trafficlight.getControlledLanes(tl_id)
                for lane in lanes:
                    waiting_times.append(traci.lane.getLastStepHaltingNumber(lane))
                    densities.append(traci.lane.getLastStepOccupancy(lane))

            observation = np.concatenate((waiting_times, densities,phases)) 
            return observation

        except Exception as e:
            logger.error("An error occurred while getting observation: %s", str(e))
            raise e  

    def _calculate_reward(self):
        try:
            # Initialize reward components
            traffic_flow_reward = 0
            traffic_delay_penalty = 0
            queue_length_penalty = 0
            safety_reward = 0

            # Iterate over traffic lights
            for tl_id in traci.trafficlight.getIDList():
                lanes = traci.trafficlight.getControlledLanes(tl_id)
                for lane in lanes:
                    # Traffic flow reward: based on average speed of vehicles
                    traffic_flow_reward += traci.lane.getLastStepMeanSpeed(lane)

                    # traffic_flow_reward = int(traffic_flow_reward)

                    # Traffic delay penalty: based on waiting time of vehicles
                    traffic_delay_penalty += traci.lane.getLastStepHaltingNumber(lane)

                    # Queue length penalty: based on length of vehicle queues
                    queue_length_penalty += traci.lane.getLastStepVehicleNumber(lane)

                    # Safety reward: based on number of collisions
                    safety_reward += traci.simulation.getCollidingVehiclesNumber()

        
            # logger.info("rewards:{},{},{},{}".format(traffic_flow_reward,traffic_delay_penalty,queue_length_penalty,safety_reward))        

            # Combine individual rewards with appropriate weights
            total_reward = (
                traffic_flow_reward
                - traffic_delay_penalty 
                - queue_length_penalty 
                - safety_reward
            )

            return total_reward

        except Exception as e:
            logger.error("An error occurred while calculating reward: %s", str(e))
            raise e

    def _is_done(self):

        done_flag = self.step_counter >= 5000
        checkpoint = self.step_counter >=100
        emptyflag = False
        total_vehicles_present = 0

        if checkpoint:
            # Check if any lane has vehicles
            for tl_id in traci.trafficlight.getIDList():
                lanes = traci.trafficlight.getControlledLanes(tl_id)
                for lane in lanes:
                    total_vehicles_present += traci.lane.getLastStepVehicleNumber(lane)
            emptyflag = total_vehicles_present == 0

        if done_flag and emptyflag:
            done_reason = "Both max steps reached and no vehicles in any lane"
            logger.info("done reason:{}".format(done_reason))
        elif done_flag:
            done_reason = "Max steps reached"
            logger.info("done reason:{}".format(done_reason))
        elif emptyflag:
            done_reason = "No vehicles in any lane"
            logger.info("done reason:{}".format(done_reason))
        else:
            done_reason = "Neither condition met"
            # logger.info("done reason:{}".format(done_reason))
        done = done_flag or emptyflag
        return done
            
    def close(self):
        try:
            traci.close(False)
            logger.info("SUMO simulation closed.")
        except Exception as e:
            logger.error("An error occurred while closing SUMO simulation: %s", str(e))
            raise e
        
# Running reward statistics
running_reward_mean = 0
running_reward_std = 1e-6 

def normalize_reward(reward):
    global running_reward_mean, running_reward_std
    running_reward_mean = 0.95 * running_reward_mean + 0.05 * reward
    running_reward_std = np.sqrt(0.95 * running_reward_std**2 + 0.05 * (reward - running_reward_mean)**2)
    normalized_reward = (reward - running_reward_mean) / (running_reward_std + 1e-5)
    return normalized_reward

# Initialize environment
env = CustomEnv()
env = DummyVecEnv([lambda: env])


# Train PPO agent
# model = A2C('MlpPolicy', env, verbose=1,learning_rate=0.001,ent_coef=0.01)

# model = DQN.load("dqn_model_140")
model = PPO.load("/home/poison/RL/Final/PPO_Final/saved_models/traffic_signal_controller")


total_rewards = []

def evaluate_model(model, env, num_episodes=5):
    
    generate_routefile()
    total_eval_rewards = []
    for episode in range(1, num_episodes+1):
        traci.start(["sumo", "-c", configfile_path,"--no-warnings"])
        eval_episode_reward = 0
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            reward = normalize_reward(reward[0])
            eval_episode_reward += reward
        total_eval_rewards.append(eval_episode_reward)
        print("Eval Reward for the episdode {}: {}".format(episode,eval_episode_reward))
        logger.info("Eval Reward for the episdode {}: {}".format(episode,eval_episode_reward))
        traci.close()
    avg_reward = sum(total_eval_rewards) / num_episodes

    
    
    return total_eval_rewards, avg_reward

logger.info("Evaluation Started")

print("\n--------------Evalutaion Started------------------")
# Assuming 'model' and 'env' are already created
episode_rewards, avg_reward = evaluate_model(model, env, num_episodes=5)
print("Average Reward over 5 episodes: {}".format(avg_reward))

# Plot rewards per episode
plt.plot(episode_rewards, marker='o')
plt.xlabel('Episode')
plt.ylabel('Episode Reward')
plt.title('Reward per Episode (Evaluation)')
plt.grid(True)
plt.savefig('total_rewards_plots_eval_ppo.png')
plt.show()


    




