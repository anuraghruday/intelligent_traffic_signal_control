from __future__ import absolute_import
from __future__ import print_function

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


def generate_routefile():
    random.seed(42)
    N = 3600

    pWE= 1. / 10
    pEW = 1. / 11
    pNS = 1. / 10
    pSN = 1. / 11

    with open("/home/poison/RL/Final/test5/route_gen.rou.xml","w") as routes:
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

        random.seed(42)

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


# # def run():
# #     """execute the TraCI control loop"""
# #     step = 0
# #     # we start with phase 2 where EW has green
# #     traci.trafficlight.setPhase("0", 2)
# #     while traci.simulation.getMinExpectedNumber() > 0:
# #         traci.simulationStep()
# #         if traci.trafficlight.getPhase("0") == 2:
# #             # we are not already switching
# #             if traci.inductionloop.getLastStepVehicleNumber("0") > 0:
# #                 # there is a vehicle from the north, switch
# #                 traci.trafficlight.setPhase("0", 3)
# #             else:
# #                 # otherwise try to keep green for EW
# #                 traci.trafficlight.setPhase("0", 2)
# #         step += 1
# #     traci.close()
# #     sys.stdout.flush()

# def get_options():
#     optParser = optparse.OptionParser()
#     optParser.add_option("--nogui", action="store_true",
#                          default=False, help="run the commandline version of sumo")
#     options, args = optParser.parse_args()
#     return options



# # this is the main entry point of this script
# if __name__ == "__main__":
#     options = get_options()

#     # this script has been called from the command line. It will start sumo as a
#     # server, then connect and run
#     if options.nogui:
#         sumoBinary = checkBinary('sumo')
#     else:
#         sumoBinary = checkBinary('sumo-gui')

#     # first, generate the route file for this simulation
#     generate_routefile()

#     # this is the normal way of using traci. sumo is started as a
#     # subprocess and then the python script connects and runs
#     traci.start([sumoBinary, "-c", "/home/poison/RL/Final/test5/fukk_1.sumocfg",
#                              "--tripinfo-output", "tripinfo_dqn.xml"])
#     # run()

generate_routefile()