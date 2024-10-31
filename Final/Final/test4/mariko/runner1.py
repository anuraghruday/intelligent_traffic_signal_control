import os
import traci
import traci.constants as tc
import xml.etree.ElementTree as ET

import os
import sys
import optparse
import random

# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa

def parse_flows(route_file):
    flows = {}
    tree = ET.parse(route_file)
    root = tree.getroot()
    for flow in root.findall('flow'):
        flow_id = flow.get('id')
        route_prob_distribution = {}
        for route in flow.findall('routeProbDistribution/route'):
            route_id = route.get('id')
            probability = float(route.get('probability'))
            route_prob_distribution[route_id] = probability
        flows[flow_id] = route_prob_distribution
    return flows

def run_simulation(route_file):
    sumo_binary = "sumo"  # Path to SUMO executable
    sumo_cmd = [sumo_binary, "-c", "mariko.sumocfg"]  # Path to your SUMO configuration file
    traci.start(sumo_cmd)

    flows = parse_flows(route_file)

    print(flows)

    for flow_id, route_prob_distribution in flows.items():
        # begin_time = int(flow_id.split('_')[1]) * 3600
        # end_time = (int(flow_id.split('_')[1]) + 1) * 3600
        vehicle_type = "car"
        depart_pos = "random"
        depart_speed_min = 10
        depart_speed_max = 20

        for route_id, probability in route_prob_distribution.items():
            traci.route.add(route_id, route_id.split('_')[1].split())
            traci.vehicle.addFlow(flow_id, 0, 3600, typeID=vehicle_type, routeID=route_id,
                                  departPos=depart_pos, departSpeed=(depart_speed_min, depart_speed_max), probability=probability)

    # Run simulation
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

    traci.close()

if __name__ == "__main__":
    route_file = "anjin1.rou.xml"
    run_simulation(route_file)
