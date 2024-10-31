import matplotlib.pyplot as plt
import numpy as np
from config import ENV_CONSTANTS, CAR_CONSTS, FILES, STATE_CONSTANTS

def plot(rewardHistory, carsHistory):
        x = np.arange(1,len(rewardHistory)+1,1)
        plt.subplot(2, 1, 1)
        plt.title('One Day Example (Normal Route, {})'.format(STATE_CONSTANTS["POLICY"]))

        plt.plot(x,rewardHistory,label="Trained Agent")
        plt.ylabel('Cost Per Car')
        plt.legend(loc="best")

        plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        labelbottom=False) # labels along the bottom edge are off

        plt.subplot(2, 1, 2)
        plt.plot(x,carsHistory)
        plt.xlabel('Timestep')
        plt.ylabel('Number of Cars')
        plt.show()

def dualDailyPlot(trainedRewards,naiveRewards,trainedCarHistory,naiveCarHistory,trainedTravelTimes,naiveTravelTimes):
    assert(len(trainedRewards)==len(naiveRewards)==len(trainedCarHistory)==len(naiveCarHistory))

    _, ax = plt.subplots()
    episodeLength = ENV_CONSTANTS["EPISODE_LENGTH"]
    trainedRewards,naiveRewards = np.array(trainedRewards),np.array(naiveRewards)
    trainedRewards = np.mean(trainedRewards.reshape(-1, episodeLength), axis=1)
    naiveRewards = np.mean(naiveRewards.reshape(-1, episodeLength), axis=1)
    x = np.arange(1,len(trainedRewards)+1,1)
    s = plt.subplot(3, 1, 1)
    s.set_ylim([0, max(max(trainedRewards),max(naiveRewards))*1.1])
    plt.title('Daily Averages (Normal Route, {})'.format(STATE_CONSTANTS["POLICY"]))
    plt.plot(x,trainedRewards,label="Learning Agent")
    plt.plot(x,naiveRewards,label="Naive Agent")
    plt.ylabel('Wait Time/Day')
    plt.legend(loc="best")

    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    labelbottom=False) # labels along the bottom edge are off

    trainedCarHistory,naiveCarHistory = np.array(trainedCarHistory),np.array(naiveCarHistory)
    trainedCarHistory = np.mean(trainedCarHistory.reshape(-1, episodeLength), axis=1)
    naiveCarHistory = np.mean(naiveCarHistory.reshape(-1, episodeLength), axis=1)

    s2 = plt.subplot(3, 1, 2)
    s2.set_ylim([0, ENV_CONSTANTS["MAX_CARS"]+2])
    plt.plot(x,trainedCarHistory,label="Learning Agent")
    plt.plot(x,naiveCarHistory,label="Naive Agent")
    plt.legend(loc="best")
    plt.ylabel('Number of Cars')

    # trainedTravelTimes,naiveTravelTimes = np.array(trainedTravelTimes),np.array(naiveTravelTimes)
    # trainedTravelTimes = np.mean(trainedTravelTimes.reshape(-1, episodeLength), axis=1)
    # naiveTravelTimes = np.mean(naiveTravelTimes.reshape(-1, episodeLength), axis=1)
    s3 = plt.subplot(3, 1, 3)
    s3.set_ylim([0, max(max(trainedTravelTimes),max(naiveTravelTimes))*1.5])
    plt.plot(x,trainedTravelTimes,label="Learning Agent")
    plt.plot(x,naiveTravelTimes,label="Naive Agent")
    plt.legend(loc="best")
    plt.ylabel('Travel Time')
    plt.xlabel('Day (600 timesteps each)')
    plt.show()

def plotDays(rewardHistory,carsHistory, avgDailyWaitTimes):
    _, ax = plt.subplots()
    episodeLength = ENV_CONSTANTS["EPISODE_LENGTH"]
    rewardHistory,carsHistory = np.array(rewardHistory),np.array(carsHistory)
    dailyAverage = np.mean(rewardHistory.reshape(-1, episodeLength), axis=1)
    dailyCars = np.mean(carsHistory.reshape(-1, episodeLength), axis=1)

    s = plt.subplot(3, 1, 1)
    s.set_ylim([0, 5])
    plt.title('Daily Averages (Normal Route, {})'.format(STATE_CONSTANTS["POLICY"]))
    plt.plot(dailyAverage)
    plt.ylabel('Wait Time/Day')
    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    labelbottom=False) # labels along the bottom edge are off

    s2 = plt.subplot(3, 1, 2)
    s2.set_ylim([0, ENV_CONSTANTS["MAX_CARS"]+2])
    plt.plot(dailyCars)
    plt.ylabel('Number of Cars')

    s3 = plt.subplot(3, 1, 3)
    s3.set_ylim([0,max(avgDailyWaitTimes)*1.5])
    plt.plot(avgDailyWaitTimes)
    plt.ylabel('Travel Time')

    ymin, ymax = ax.get_ylim()
    plt.xlabel('Day (600 timesteps each)')
    plt.show()


def culminativeCO2(travelTimes):
        return np.cumsum([x * CAR_CONSTS["CO2_PER_TICK"] for x in travelTimes])

def plotCulminativeCO2(trainedTravelTimes,naiveTravelTimes):
    x = np.arange(1,len(trainedTravelTimes)+1,1)
    trainedCO2, naiveCO2 = culminativeCO2(trainedTravelTimes), culminativeCO2(naiveTravelTimes)
    plt.plot(x,trainedCO2,label="Learning Agent Carbon Emissions")
    plt.plot(x,naiveCO2,label="Naive Agent Carbon Emissions")
    plt.title('Culminative CO2 (kg) (Normal Route, {})'.format(STATE_CONSTANTS["POLICY"]))
    plt.ylabel('CO2 Emitted (kg)')
    plt.xlabel('Day (600 timesteps each)')
    plt.legend(loc="best")
    plt.show()

