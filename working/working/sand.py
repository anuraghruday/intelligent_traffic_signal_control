import random


vclasses = ["car","truck"]
routes_dict = {'WE':['r_6','r_7','r_8'],'SN':['r_0','r_1','r_2'],'EW':['r_3','r_4','r_5'],'NS':['r_9','r_10','r_11']}

pWE= 1. / 10
pEW = 1. / 11
pNS = 1. / 10
pSN = 1. / 11

vehNr = 0

weights_vclass = [10,1]
weights_route = [1,1,1]

random.seed(42)

for i in range(3600):

    if random.uniform(0, 1) < pWE:

        vclass_type = random.choices(vclasses,weights=weights_vclass)[0]
        v_route = random.choices(routes_dict['WE'],weights = weights_route)[0]
 
        print(f'    <vehicle id=right_{i} type={vclass_type} route={v_route} depart={str(i)} />')
        vehNr += 1
    
    if random.uniform(0, 1) < pEW:

        vclass_type = random.choices(vclasses,weights=weights_vclass)[0]
        v_route = random.choices(routes_dict['EW'],weights=weights_route)[0]

        print(f'    <vehicle id=right_{i} type={vclass_type} route={v_route} depart={str(i)} />')
        vehNr += 1
    
    if random.uniform(0, 1) < pNS:

        vclass_type = random.choices(vclasses,weights=weights_vclass)[0]
        v_route = random.choices(routes_dict['NS'],weights=weights_route)[0]

        print(f'    <vehicle id=right_{i} type={vclass_type} route={v_route} depart={str(i)} />' )
        vehNr += 1
    
    if random.uniform(0, 1) < pSN:

        vclass_type = random.choices(vclasses,weights=weights_vclass)[0]
        v_route = random.choices(routes_dict['SN'],weights=weights_route)[0]
 
        print(f'    <vehicle id=right_{i} type={vclass_type} route={v_route} depart={str(i)} />')
        vehNr += 1
              