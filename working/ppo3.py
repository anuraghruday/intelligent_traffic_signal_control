import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import traci
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



def generate_routefile(seed):
    random.seed(seed)
    N = 3600

    pWE= 1. / 10
    pEW = 1. / 11
    pNS = 1. / 10
    pSN = 1. / 11

    with open("route_gen_1.rou.xml","w") as routes:
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

class CustomEnv(gym.Env):
    def __init__(self):

        # traci.start(["sumo", "-c", "/home/poison/RL/Final/test5/fukk_1.sumocfg", "--no-warnings", "--quit-on-end"])
        
        # self.traffic_lights = traci.trafficlight.getIDList()
        # self.num_traffic_lights = len(self.traffic_lights)
        # print(self.num_traffic_lights)
        self.action_space = spaces.Discrete(1)  # Red and green phases only
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(12,), dtype=np.float32)
        # traci.close()
        self.seed = 0
        
        
    def reset(self,**kwargs):
        try:
            generate_routefile(self.seed)  # Generate route file            
            self.step_counter = 0
            self.seed +=1
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
            logger.debug("Step %d completed.", self.step_counter)
            return next_observation, reward, done, info
        except Exception as e:
            logger.error("An error occurred during the step: %s", str(e))
            raise e

    def _get_observation(self):
        try:
            # Example: Get the number of waiting vehicles at each traffic light
            num_waiting = []
            for tl_id in traci.trafficlight.getIDList():
                lanes = traci.trafficlight.getControlledLanes(tl_id)
                for lane in lanes:
                    num_waiting.append(traci.lane.getLastStepHaltingNumber(lane))
            return np.array(num_waiting)
        except Exception as e:
            logger.error("An error occurred while getting observation: %s", str(e))
            raise e

    def _calculate_reward(self):
        try:
            # Example: Reward based on the number of vehicles that passed the intersection
            total_reward = 0
            for tl_id in traci.trafficlight.getIDList():
                lanes = traci.trafficlight.getControlledLanes(tl_id)
                for lane in lanes:
                    total_reward += traci.lane.getLastStepVehicleNumber(lane)
            return total_reward
        except Exception as e:
            logger.error("An error occurred while calculating reward: %s", str(e))
            raise e

    def _is_done(self):
        try:
            # Example: Episode ends after a fixed number of steps
            done = self.step_counter >= 4200
            # print(self.step_counter)
            if done:
                
                logger.info("Episode finished after %d steps.", self.step_counter)
            return done
        except Exception as e:
            logger.error("An error occurred while checking if episode is done: %s", str(e))
            raise e

    def close(self):
        try:
            traci.close()
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
    logger.info("Episode %d started.", episode + 1)
    episode_reward =0
    traci.start(["sumo", "-c", "working/fukk_1.sumocfg","--no-warnings","--quit-on-end"])
    logger.info("SUMO simulation started.")
    done = False

    step = 0

    try:
        # Reset environment at the beginning of each episode

        obs= env.reset()
        logger.info("Environmnet reset done")
        
        # print(obs)
        

        while True:

            action, _ = model.predict(obs)
            # print(action)
            # Take action in the environment
            next_obs, reward, done, info = env.step(action)
            # Update observation for next step
            obs = next_obs
            episode_reward += reward

            step +=1

            logger.info("step {} is completed:".format(step))

            if done:
                # Close SUMO simulation
                try:
                    # env.close()
                    logger.info("Episode is completed successful")
                    break
                except Exception as e:
                    logger.error("An error occurred while closing the environment: %s", str(e))
                    raise e
                

        # Train agent
        model.learn(total_timesteps=1000, reset_num_timesteps=False)
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
