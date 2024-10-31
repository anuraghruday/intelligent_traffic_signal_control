import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

outputfile = "vehicles_info_ppo.png"

def parse_tripinfo(tripinfo_files):
    avg_speeds = []
    waiting_times = []
    time_losses = []
    vehicle_counts = []
    queue_lengths = []
    intersection_performance = []
    travel_times = []

    for file in tripinfo_files:
        tree = ET.parse(file)
        root = tree.getroot()

        for tripinfo in root.findall('tripinfo'):
            avg_speeds.append(float(tripinfo.get('arrivalSpeed')))
            waiting_times.append(float(tripinfo.get('waitingTime')))
            time_losses.append(float(tripinfo.get('timeLoss')))
            travel_times.append(float(tripinfo.get('duration')))

        vehicle_counts.append(len(root.findall('tripinfo')))
        queue_lengths.append(sum(float(tripinfo.get('waitingCount')) for tripinfo in root.findall('tripinfo')))
        intersection_performance.append(sum(float(tripinfo.get('timeLoss')) for tripinfo in root.findall('tripinfo')))

    return avg_speeds, waiting_times, time_losses, vehicle_counts, queue_lengths, intersection_performance, travel_times

# List of tripinfo files for 500 episodes
tripinfo_files = ["/home/poison/RL/Final/working/tripinfo_epi{}.xml".format(i) for i in range(1, 1001)]

# Parse tripinfo files
avg_speeds, waiting_times, time_losses, vehicle_counts, queue_lengths, intersection_performance, travel_times = parse_tripinfo(tripinfo_files)

# Plotting
plt.figure(figsize=(10, 12))

plt.subplot(3, 2, 1)
plt.plot(avg_speeds, color='blue')
plt.xlabel('Episode')
plt.ylabel('Average Speed (m/s)')
plt.title('Average Speeds')

plt.subplot(3, 2, 2)
plt.plot(waiting_times, color='green')
plt.xlabel('Episode')
plt.ylabel('Waiting Time (s)')
plt.title('Waiting Times')

plt.subplot(3, 2, 3)
plt.plot(time_losses, color='orange')
plt.xlabel('Episode')
plt.ylabel('Time Loss (s)')
plt.title('Time Losses')

plt.subplot(3, 2, 4)
plt.plot(vehicle_counts, color='red')
plt.xlabel('Episode')
plt.ylabel('Number of Vehicles')
plt.title('Vehicle Counts')

plt.subplot(3, 2, 5)
plt.plot(queue_lengths, color='purple')
plt.xlabel('Episode')
plt.ylabel('Queue Length')
plt.title('Queue Lengths')

plt.subplot(3, 2, 6)
plt.plot(intersection_performance, color='brown')
plt.xlabel('Episode')
plt.ylabel('Intersection Performance')
plt.title('Intersection Performance')

plt.tight_layout()
plt.savefig(outputfile)
plt.show()
