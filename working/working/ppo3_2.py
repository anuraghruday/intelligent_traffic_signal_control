import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import gym
import time
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import traci
from traci import TraCIException
import sumolib
import numpy as np
import random
import logging
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='traffic_signal.log',
                    filemode='w')
logger = logging.getLogger(__name__)

MAX_TRAFFIC_FLOW_PENALTY = -10
QUEUE_LENGTH_PENALTY = -50
DELAY_PENALTY = 5

def generate_routefile(seed):
    random.seed(seed)
    N = 5000

    pWE= 1. / 10
    pEW = 1. / 11
    pNS = 1. / 10
    pSN = 1. / 11

    with open("working/route_gen_13.rou.xml","w") as routes:
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

        # traci.start(["sumo", "-c", "/home/poison/RL/Final/test5/fukk_1.sumocfg", "--no-warnings", "--quit-on-end"])
        
        # self.traffic_lights = traci.trafficlight.getIDList()
        # self.num_traffic_lights = len(self.traffic_lights)
        # print(self.num_traffic_lights)
        self.action_space = spaces.Discrete(2)  # Red and green phases only
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(25,), dtype=np.float32)
        # traci.close()
        self.seed = 0
        
        
    # def reset(self,**kwargs):
        # try:
        #     logger.info("reset function entered")           
        #     self.step_counter = 0
        #     # self.seed +=1

        #     return self._get_observation()
        #     except Exception as e:
        #     logger.error("An error occurred while resetting the environment: %s", str(e))
        #     raise e

    def reset(self, **kwargs):
        try: 
            logger.info("reset function entered") 
            self.step_counter = 0
            # Print traffic light IDs for debugging
            # traffic_light_ids = traci.trafficlight.getIDList()
            # print("Traffic light IDs:", traffic_light_ids)
            
            # Reset traffic lights
            for tl_id in traci.trafficlight.getIDList():

                traci.trafficlight.setPhase(tl_id, 0)  # Assuming 0 is your desired phase

            return self._get_observation() 

        except Exception as e:
            logger.error("An error occurred while resetting the environment: %s", str(e))
        raise e

    # def step(self, action):
    #     try:
    #         for idx, tl_id in enumerate(traci.trafficlight.getIDList()):
    #             traci.trafficlight.setPhase(tl_id, action)
    #         traci.simulationStep()
    #         self.step_counter += 1
    #         next_observation = self._get_observation()
    #         reward = self._calculate_reward()
    #         done = self._is_done()
    #         info = {}
    #         logger.debug("Step %d completed.", self.step_counter)
    #         return next_observation, reward, done, info
    #     except Exception as e:
    #         logger.error("An error occurred during the step: %s", str(e))
    #         raise e

    #  def _get_observation(self):
    #     try:
    #         # Example: Get the number of waiting vehicles at each traffic light
    #         num_waiting = []
    #         phases = []
    #         for tl_id in traci.trafficlight.getIDList():
    #             phases.append(traci.trafficlight.getPhase(tl_id))
    #             lanes = traci.trafficlight.getControlledLanes(tl_id)
    #             for lane in lanes:
    #                 num_waiting.append(traci.lane.getLastStepHaltingNumber(lane))
    #         return np.array(num_waiting + phases)
    #     except Exception as e:
    #         logger.error("An error occurred while getting observation: %s", str(e))
    #         raise e

    def step(self, action):
        try:
            for idx, tl_id in enumerate(traci.trafficlight.getIDList()):
                traci.trafficlight.setPhase(tl_id, action)
            traci.simulationStep()
            self.step_counter += 1

            next_observation = self._get_observation()  # Get your rich observation 
            reward = self._calculate_reward() 
            done = self._is_done()
            info = {}
            logger.debug("Step %d completed.", self.step_counter)

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

            observation = np.concatenate((waiting_times, densities, phases))
            print(observation.shape) 
            return observation

        except Exception as e:
            logger.error("An error occurred while getting observation: %s", str(e))
            raise e  # Re-raise the exception for further handling

    #  def _calculate_reward(self):
    #     try:
    #         # Example: Reward based on the number of vehicles that passed the intersection
    #         total_reward = 0
    #         for tl_id in traci.trafficlight.getIDList():
    #             lanes = traci.trafficlight.getControlledLanes(tl_id)
    #             for lane in lanes:
    #                 total_reward += traci.lane.getLastStepVehicleNumber(lane)
    #         return total_reward
    #     except Exception as e:
    #         logger.error("An error occurred while calculating reward: %s", str(e))
    #         raise e

    def get_traffic_congestion_ratio(tls_id):
        # Get the total number of vehicles
        total_vehicles = len(traci.vehicle.getIDList())

        # Get the number of waiting vehicles at traffic lights
        waiting_vehicles = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in traci.trafficlight.getControlledLanes(tls_id))

        # Calculate traffic congestion ratio
        congestion_ratio = waiting_vehicles / total_vehicles if total_vehicles > 0 else 0

        return congestion_ratio
    
    def _calculate_reward(self):
        try:
            reward = 0

            waiting_time_history = []
            density_history = []
            
            # 1. Throughput Bonus
            base_throughput_reward = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in traci.lane.getIDList())
            reward += base_throughput_reward


            # 2. Congestion Penalties 
            for lane_id in traci.lane.getIDList():

                queue_length = traci.lane.getLastStepVehicleIDs(lane_id)  # Get vehicle IDs on the lane
                waiting_time = traci.lane.getLastStepHaltingNumber(lane_id) 
                density = traci.lane.getLastStepOccupancy(lane_id)
            
                waiting_time_history.append(waiting_time)
                density_history.append(density)

                current_wait_average = sum(waiting_time_history) / len(waiting_time_history)
                current_density_average = sum(density_history) / len(density_history)

                waiting_threshold = current_wait_average * 1.2  
                density_threshold = current_density_average * 1.2
            
                if waiting_time > waiting_threshold:
                    reward -= waiting_time * MAX_TRAFFIC_FLOW_PENALTY

                if density > density_threshold:
                    reward -= density * MAX_TRAFFIC_FLOW_PENALTY

                reward -= len(queue_length) * QUEUE_LENGTH_PENALTY

            return reward

        except Exception as e:
            logger.error("An error occurred while calculating reward: %s", str(e))
            raise e
        
    def _is_done(self):

        done_flag = self.step_counter >= 5000

        checkpoint = self.step_counter >= 100

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

# Initialize environment
env = CustomEnv()
env = DummyVecEnv([lambda: env])

# Train PPO agent
model = PPO('MlpPolicy', env, verbose=1)


# Set number of episodes
num_episodes = 5

total_rewards = []

for episode in range(num_episodes):
    logger.info("\nEpisode %d started.", episode + 1)
    episode_reward =0
    generate_routefile(episode)  # Generate route file 
    logger.info("Route file generated")
    # traci.start(["sumo", "-c", "/home/poison/RL/Final/test5/fukk_1.sumocfg","--no-warnings","--quit-on-end"])
    traci.start(["sumo-gui", "-c", "working/fukk_1.sumocfg","--no-warnings"])
    logger.info("SUMO simulation started.")
    done = False

    step = 0

    try:
        # Reset environment at the beginning of each episode

        obs = env.reset()
        logger.info("Environmnet reset done")
        
        # print(obs)
        

        while True:

            action, _ = model.predict(obs)
            # print(action)
            # print(action)
            # Take action in the environment
            try:
                next_obs, reward, done, info = env.step(action)
            except TraCIException: 
                logger.warning("TraCIException encountered. Retrying...")
                time.sleep(0.1)  # Small pause
                continue # Retry the step
            
            # Update observation for next step
            obs = next_obs
            episode_reward += reward

            step +=1
            logger.info("step {} is completed:".format(step))

            if done:
                # Close SUMO simulation
                try:
                    logger.info("episode is terminated")
                    break
                    
                except Exception as e:
                    logger.error("An error occurred while closing the environment: %s", str(e))
                    raise e
                
        logger.info("model learning started")
        # Train agent
        model.learn(total_timesteps=1, reset_num_timesteps=False)
        total_rewards.append(episode_reward)


        print("Total reward for episode {}: {}".format(episode + 1, episode_reward))
        env.close()
        logger.info("traci is closed for episode")

    except Exception as e:
        logger.error("An error occurred during episode {%d}: %s", episode + 1, str(e))
        raise e

# Save trained model
try:
    model.save("traffic_signal_controller")
    logger.info("Model saved successfully.")
except Exception as e:
    logger.error("An error occurred while saving the model: %s", str(e))
    raise e

plt.plot(total_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Reward per Episode')
plt.show()

# # Close SUMO simulation
# try:
#     env.close()
# except Exception as e:
#     logger.error("An error occurred while closing the environment: %s", str(e))
#     raise e
