def _get_observation(self):
    try:
        waiting_times = []
        densities = []  
        phases = []

        for tl_id in traci.trafficlight.getIDList():
            phases.append(traci.trafficlight.getPhase(tl_id))
            lanes = traci.trafficlight.getControlledLanes(tl_id)
            for lane in lanes:
                waiting_times.append(traci.lane.getLastStepHaltingNumber(lane))
                densities.append(traci.lane.getLastStepOccupancy(lane))  

         # Normalize (optional)
         # ... (Code for normalization of waiting_times & densities if needed) ...    

        observation = np.concatenate((waiting_times, densities, phases))  
        return observation
    except Exception as e:
        logger.error("An error occurred while getting observation: %s", str(e))
        raise e

def _get_observation(self):
    try:
        num_waiting = []
        phases = []
        for tl_id in traci.trafficlight.getIDList():
            phases.append(traci.trafficlight.getPhase(tl_id)) 
            lanes = traci.trafficlight.getControlledLanes(tl_id)
            for lane in lanes:
                num_waiting.append(traci.lane.getLastStepHaltingNumber(lane))

        # Normalize waiting times (optional, uncomment if desirable)
        # max_waiting_time = ...  # Replace with the maximum possible value
        # num_waiting = np.array(num_waiting) / max_waiting_time  

        observation = np.array(num_waiting + phases) 
        return observation
    except Exception as e:
        logger.error("An error occurred while getting observation: %s", str(e))
        raise e
    

class CustomEnv(gym.Env):
    def __init__(self):
        # ... (Your existing initialization code) ...
        self.action_space = spaces.MultiDiscrete([2, 2, 2, 2])  # One action for each traffic light
        # ... (Rest of your initialization) ... 

    def step(self, action):
        try:
            for idx, action_for_light in enumerate(action):
                tl_id = traci.trafficlight.getIDList()[idx]
                traci.trafficlight.setPhase(tl_id, action_for_light)  # Apply the action
            traci.simulationStep()
            # ... (Rest of your 'step' function: observation, reward, etc.) ...
        except Exception as e:
            # ... (Your error handling) ...

# ... (Rest of your code: model creation, training loop, etc.) ...


def _calculate_reward(self):
    try:
        total_waiting_time = 0
        for lane in traci.lane.getIDList():
            total_waiting_time += traci.lane.getLastStepHaltingNumber(lane)

        # Reward decreases as waiting time increases 
        reward = -total_waiting_time 

        return reward
    except Exception as e:
        logger.error("An error occurred while calculating reward: %s", str(e))
        raise e