## Traffic Signal Control with Reinforcement Learning
# Overview

This project utilizes reinforcement learning techniques to optimize traffic signal control in a simulated environment. It employs the SUMO (Simulation of Urban MObility) traffic simulation framework along with the Stable Baselines3 library for RL implementation.
Requirements

    Python 3.x
    SUMO (Simulation of Urban MObility) traffic simulation framework
    Stable Baselines3 library
    Gym
    Matplotlib

Setup

    Install SUMO: SUMO Installation Guide
    Install required Python packages: pip install stable-baselines3 gym matplotlib
    Clone this repository: git clone <repository_url>
    Navigate to the repository directory: cd <repository_directory>

Usage

    Generate route file: Run generate_routefile() function to create the route file for SUMO simulation.
    Run the simulation: Execute the main script to start the SUMO simulation and train the RL agent. Example: python traffic_signal_rl.py
    Training: The script will train the RL agent using the specified algorithm (DQN, PPO, A2C, etc.) and update the model parameters.
    Evaluation: After training, the model can be evaluated using the evaluate_model() function. This will run the trained model in the simulation environment and provide evaluation metrics.

Files

    traffic_signal_rl.py: Main script containing the RL implementation.
    route_gen.rou.xml: XML file defining the vehicle routes for SUMO simulation.
    traffic_signal.log: Log file for recording simulation and training information.
    README.md: This README file providing an overview of the project.
    saved_models: Models are being saved at regular interval.
    tripinfo_files: Saves the trip of each episode for further analysis of output metrics.

Acknowledgements

    This project is inspired by the Deep Reinforcement Learning for Traffic Signal Control applications in the field of traffic signal optimization using reinforcement learning.
    Acknowledgment to the SUMO development team and the Stable Baselines3 contributors for their valuable tools and libraries.

