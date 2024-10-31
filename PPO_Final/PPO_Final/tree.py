import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

# Parse tripinfo files and extract relevant data
def parse_tripinfo(tripinfo_files):
    avg_speeds = []
    waiting_times = []
    time_losses = []

    for file in tripinfo_files:
        tree = ET.parse(file)
        root = tree.getroot()
        speeds = []
        waiting = 0
        time_loss = 0

        for trip in root.findall('tripinfo'):
            speeds.append(float(trip.get('arrivalSpeed')))
            waiting += float(trip.get('waitingTime'))
            time_loss += float(trip.get('timeLoss'))

        avg_speed = np.mean(speeds)
        avg_speeds.append(avg_speed)
        waiting_times.append(waiting)
        time_losses.append(time_loss)

    return avg_speeds, waiting_times, time_losses

# List of tripinfo files for 500 episodes
tripinfo_files = ["tripinfo_epi{}.xml".format(i) for i in range(1, 1001)]

# Parse tripinfo files and extract data
avg_speeds, waiting_times, time_losses = parse_tripinfo(tripinfo_files)

# Plotting
plt.figure(figsize=(10, 6))

# Average Speeds
plt.subplot(3, 1, 1)
plt.plot(avg_speeds, color='blue')
plt.title('Average Speeds of Vehicles')
plt.xlabel('Episode')
plt.ylabel('Average Speed')
plt.grid(True)

# Waiting Time
plt.subplot(3, 1, 2)
plt.plot(waiting_times, color='green')
plt.title('Waiting Time of Vehicles')
plt.xlabel('Episode')
plt.ylabel('Waiting Time')
plt.grid(True)

# Time Loss
plt.subplot(3, 1, 3)
plt.plot(time_losses, color='red')
plt.title('Time Loss of Vehicles')
plt.xlabel('Episode')
plt.ylabel('Time Loss')
plt.grid(True)

plt.tight_layout()
plt.show()
