from tkinter import *
import time as tm
from config import ENV_CONSTANTS,LIGHT_CONSTANTS,FILES
from Agent import Agent
import json
import numpy as np
from Main import Main

class Visualizer:

    def __init__(self):
        self.time = 0
        self.createCanvas()
        self.gui.update()
        self.gui.title("Traffic Light Simulation")
        self.lightOffset = 350
        self.padding = 150
        self.stateIsNew = False
        self.main = Main(visualizerCallback=self.updateFrame)
        self.environment = self.main.environment
        self.main.startSimulation()

        # with open(FILES["LOAD_FILE"]) as qTable:
            # self.agent.qTable = json.load(qTable)
           
    def updateFrame(self, newEnvironment, time, stateIsNew, avgCost, isRushHour):
        tm.sleep(0.5)
        self.environment = newEnvironment
        self.time = time
        self.canvas.delete("all")
        self.updateTrafficLights()
        self.createLabels(stateIsNew, avgCost, isRushHour)
        self.gui.update()

    def createLabels(self, stateIsNew, avgCost, isRushHour):
        newState = "New state!" if stateIsNew else ""
        self.canvas.create_text(60, 60, font=("Arial", 16), text=newState, fill="blue")
        self.canvas.create_text(60, 90, font=("Arial", 16), text="Avg cost:", fill="blue")
        self.canvas.create_text(130, 90, font=("Arial", 16), text="{:.2f}".format(avgCost), fill="blue")
        if isRushHour:
            self.canvas.create_text(90, 30, font=("Arial", 16), text="RUSH HOUR TIME!", fill="blue")

    def updateTrafficLights(self):
        for light in self.environment.lights:
            self.createTrafficLight(light, light.id)

    def createTrafficLight(self,light, lightId):
        xCenter, yCenter = self.drawTrafficLight(light, lightId)
        nsPenalty, ewPenalty = light.getWaitTimes(self.time, sum(self.environment.getCarWaits(self.time)))
        for queueIndex, queue in enumerate(light.queues):
            howBad = "grey"
            if queueIndex in [LIGHT_CONSTANTS["ACTION_DIR"]["n"],LIGHT_CONSTANTS["ACTION_DIR"]["s"]]:
                if nsPenalty == LIGHT_CONSTANTS["TIME_BINS"]["large"]["penalty"]:
                    howBad = "red"
                elif nsPenalty == LIGHT_CONSTANTS["TIME_BINS"]["medium"]["penalty"]:
                    howBad = "orange" 
                elif nsPenalty == LIGHT_CONSTANTS["TIME_BINS"]["small"]["penalty"]:
                    howBad = "green" 
            else:
                if ewPenalty == LIGHT_CONSTANTS["TIME_BINS"]["large"]["penalty"]:
                    howBad = "red"
                elif ewPenalty == LIGHT_CONSTANTS["TIME_BINS"]["medium"]["penalty"]:
                    howBad = "orange"            
                elif ewPenalty == LIGHT_CONSTANTS["TIME_BINS"]["small"]["penalty"]:
                    howBad = "green"           
            self.createLightQueue(queue, queueIndex, lightId, xCenter,yCenter,howBad)

    def drawTrafficLight(self, light, lightId):
        direction = light.directionIsNorthSouth
        position = self.getCoordinatesFromlightId(lightId)
        length = 60

        xLeft = position[0] * self.lightOffset + self.padding
        yTop = position[1] * self.lightOffset + self.padding
        xRight = xLeft + length
        yBottom = yTop + length
        xCenter = (xLeft + xRight) / 2
        yCenter = (yTop + yBottom) / 2

        if direction: # N/S
            self.canvas.create_line(xCenter, yTop, xCenter, yBottom, arrow=BOTH, width=4, arrowshape=(8,12,8))
        else: # E/W
            self.canvas.create_line(xLeft, yCenter, xRight, yCenter, arrow=BOTH, width=4, arrowshape=(8,12,8))
        self.canvas.create_text(xCenter+30,yCenter+30, font=("Arial", 16), text=str(light.id))

        return xCenter, yCenter

    def createLightQueue(self, queue, queueIndex, lightId, lightCenterX, lightCenterY,fill='g'):
        queueCenterX, queueCenterY = self.drawLightQueue(queue, queueIndex, lightCenterX, lightCenterY,fill,)
        # if queueIndex == ENV_CONSTANTS["QUEUE_DIR"]["n"] and lightId == ENV_CONSTANTS["LIGHT_POSITIONS"]["SE"]:
        numCars = queue.getNumCarsDriving()
        self.createCars(numCars, queueCenterX, queueCenterY, queueIndex, lightId)
            # print([car.delay for car in queue.cars])

    def drawLightQueue(self, queue, queueIndex, xCenter, yCenter,fill='g'):
        xOffset, yOffset = self.getQueueOffset(queueIndex)
        x = xCenter + xOffset
        y = yCenter + yOffset

        self.canvas.create_text(x, y, font=("Arial", 24), text=str(queue.getNumCarsWaiting()),fill=fill)
        return x, y

    def createCars(self, numCars, queueCenterX, queueCenterY, queueIndex, lightId):
        # if numCars:
        self.drawCars(numCars, queueCenterX, queueCenterY, queueIndex, lightId)
    
    def drawCars(self, numCars, endX, endY, queueIndex, lightId):
        if not numCars:
            return
        size = 100
        padding = 20

        line = None
        x1 = None
        y1 = None
        x2 = None
        y2 = None

        xText = None
        yText = None

        if queueIndex == ENV_CONSTANTS["QUEUE_DIR"]["n"]:
            x1 = endX
            y1 = endY + size
            x2 = endX
            y2 = endY + padding
            xText = (x1 + x2) / 2 + padding
            yText = (y1 + y2) / 2
        elif queueIndex == ENV_CONSTANTS["QUEUE_DIR"]["e"]:
            x1 = endX - size
            y1 = endY
            x2 = endX - padding
            y2 = endY
            xText = (x1 + x2) / 2
            yText = (y1 + y2) / 2 + padding
        elif queueIndex == ENV_CONSTANTS["QUEUE_DIR"]["s"]:
            x1 = endX
            y1 = endY - size
            x2 = endX
            y2 = endY - padding
            xText = (x1 + x2) / 2 + padding
            yText = (y1 + y2) / 2
        else:  # ENV_CONSTANTS["QUEUE_DIR"]["w"]
            x1 = endX + size
            y1 = endY
            x2 = endX + padding
            y2 = endY
            xText = (x1 + x2) / 2
            yText = (y1 + y2) / 2 + padding
            

        line = self.canvas.create_line(x1, y1, x2, y2, arrow=LAST, width=3, fill="blue")

        text = self.canvas.create_text(xText, yText, font=("Arial", 16), text=str(numCars), fill="blue")

    def getCoordinatesFromlightId(self, lightId):
        if lightId == "NW":
            return (0, 0)
        elif lightId == "NE":
            return (1,0)
        elif lightId == "SE":
            return (1, 1)
        else: # West
            return (0, 1)
    
    def getQueueOffset(self, queueIndex):
        offset = 50
        n = ENV_CONSTANTS["QUEUE_DIR"]["n"]
        e = ENV_CONSTANTS["QUEUE_DIR"]["e"]
        s = ENV_CONSTANTS["QUEUE_DIR"]["s"]

        if queueIndex == n:
            return 0, offset
        elif queueIndex == e:
            return -offset, 0
        elif queueIndex == s:
            return 0, -offset
        else:  # west
            return offset, 0
    
    def getCarOffset(self, progress, queueIndex):
        maxOffset = 30
        if queueIndex == 0:
            return 0, progress * maxOffset
        elif queueIndex == 1:
            return progress * maxOffset, 0
        elif queueIndex == 2:
            return 0, progress * maxOffset
        else:
            return progress * maxOffset, 0

    def createCanvas(self):
        self.gui = Tk()
        canvasSize = min(int(self.gui.winfo_screenheight() * 0.9), 800)
        self.gui.geometry("{}x{}".format(canvasSize, canvasSize))
        self.canvas = Canvas(self.gui, width=canvasSize,
                        height=canvasSize, background="#fafafa")
        self.canvas.pack()
        return self.canvas

if __name__ == "__main__":
    gui = Visualizer()
