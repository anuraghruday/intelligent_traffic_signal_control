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
                    filename='traffic_signal.log',
                    filemode='w')
logger = logging.getLogger(__name__)

configfile_path = "fukk_1.sumocfg"
routefile_path = "route_gen.rou.xml"

def generate_routefile():
    # random.seed(seed)
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
model = PPO('MlpPolicy', env, verbose=1,learning_rate=0.001,ent_coef=0.01)
# model = PPO('CnnPolicy', env, verbose=1,)
# model = PPO('MlpPolicy', env, verbose=1, learning_rate=0.0003, n_steps=2048, batch_size=64,ent_coef=0.01,n_epochs=1,tensorboard_log=None)

# Set number of episodes
num_episodes = 1000
total_rewards = []


for episode in range(num_episodes):
    logger.info("\n")
    logger.info("Episode %d started.", episode + 1)
    episode_reward =0
    generate_routefile()  # Generate route file 
    logger.info("Route file generated")

    tripinfo_name = "tripinfo_epi{}.xml".format(episode+1)
    traci.start(["sumo", "-c", configfile_path,"--no-warnings", "--tripinfo-output", tripinfo_name])
    logger.info("SUMO simulation started.")

    # Resetting environment at the beginning of each episode

    obs= env.reset()
    logger.info("Environmnet reset done")
    done = False
    epi_steps = 0

    try:
        while not done:
            action, _ = model.predict(obs)
            # Take action in the environment
            next_obs, reward, done, info = env.step(action)
            # Update observation for next step
            obs = next_obs
            reward = normalize_reward(reward)
            episode_reward += reward
            epi_steps += 1
        logger.info("model learning started")
        # Train agent
        model.learn(total_timesteps=epi_steps, reset_num_timesteps=False)
        total_rewards.append(episode_reward)
        # losses.append(model.ep_info_buffer[0]['loss'])

        logger.info("Total reward for episode {}: {}, completed in steps: {}".format(episode + 1, episode_reward,epi_steps))
        print("Total reward for episode {}: {}, completed in steps: {}".format(episode + 1, episode_reward,epi_steps))
        env.close()
        logger.info("traci is closed for episode")

    except Exception as e:
        logger.error("An error occurred during episode {%d}: %s", episode + 1, str(e))
        raise e

    # Save trained model
    try:
        if episode%10 == 0:
            model.save("model_{}".format(episode+1))
            logger.info("Model saved successfully.")
    except Exception as e:
        logger.error("An error occurred while saving the model: %s", str(e))
        raise e


plt.plot(total_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Reward per Episode')
plt.savefig('total_rewards_plots.png')
plt.show()
