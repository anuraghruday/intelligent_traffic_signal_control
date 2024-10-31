LIGHT_CONSTANTS = {
    "ACTION_DIR": {  # direction the car is facing in that queue
        "n": 0,
        "e": 1,
        "s": 2,
        "w": 3,
    },
    "TIME_BINS": {
        "zero": {"penalty": 0},
        "small": {"lowerBound": 0, "penalty": 2},
        "medium": {"lowerBound": lambda totalWait: max(totalWait/8, 10), "penalty": 25},
        "large": {"lowerBound": lambda totalWait: max(totalWait/3, 30), "penalty": 100},
    },  # large wait > 5
}

EPISODE_LEN = 600  # Number of minutes in a 10 hour day (so morning, rush hour, work day, and then night rush hour)
NUM_INTERVALS = 5

boundaryOne = EPISODE_LEN // NUM_INTERVALS
boundaryTwo = (EPISODE_LEN // NUM_INTERVALS) * 2
boundaryThree = (EPISODE_LEN // NUM_INTERVALS) * 3
boundaryFour = (EPISODE_LEN // NUM_INTERVALS) * 4

FILES = {
    "SAVE_FILE": "qTables/egreedy_normal.json",
    "LOAD_FILE": "qTables/egreedy_normal.json"
}

ENV_CONSTANTS = {
    # For normal route use 7*365 to train
    # for loop route use 15*10
    "NUM_YEARS": 1,
    "NUM_DAYS": 10,
    "EPISODE_LENGTH": EPISODE_LEN,
    "MAX_CARS": 20,
    "ROUTE": "normal", # route = {normal,loopy,simpleLoopy}
    "NUM_INTERVALS": NUM_INTERVALS,
    "LIGHT_POSITIONS": {"NW": 0, "NE": 1, "SE": 2, "SW": 3},
    "QUEUE_DIR": LIGHT_CONSTANTS["ACTION_DIR"],
    "RUSH_HOUR_TIMES": [
        (boundaryOne, boundaryTwo - 1),
        (boundaryThree, boundaryFour - 1),
    ],
    "TIME_INTERVALS": [
        (0, boundaryOne - 1),
        (boundaryOne, boundaryTwo - 1),
        (boundaryTwo, boundaryThree - 1),
        (boundaryThree, boundaryFour - 1),
        (boundaryFour, EPISODE_LEN - 1),
    ]
}

CAR_CONSTS = {
    "MAX_DELAY": 3,
    "CO2_PER_TICK": 0.208 # kg - based on average speed of 50km/h and average CO2 output of 250 g/km
}

# Used to determine the size of the state by considering each variable that makes up the state
# Parameterize construction of dictionaries
def stateEntity(quantity, numStates):
    return {"quantity": quantity, "numStates": numStates}


STATE_CONSTANTS = {
    "LIGHTS": stateEntity(4, 2),
    "QUEUES": stateEntity(
        8, 1 + len(LIGHT_CONSTANTS["TIME_BINS"])
    ),  # Add one to the length here because the length is the number of boundaries, which is one more than the number of bins
    "TIME_OF_DAY": stateEntity(1, NUM_INTERVALS),
    "POLICY": "egreedy", # policy = {softmax,egreedy}
}

