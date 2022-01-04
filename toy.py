from random import *
from matplotlib import pyplot as plt
from math import *
import numpy as np

class Agent:
    def __init__(self, input, weights, outputNodes, target):
        self.input = input
        self.weights = weights
        self.outputNodes = outputNodes
        self.target = target
        self.eval = 0.0

    def mutate(self, min, max):
        # This function is to change a random weight much like genetics
        flag = randint(0, 1);
        index = randint(0, len(self.weights)-1)

        if (flag == 1):
            self.weights[index] = uniform(min, max) + self.weights[index]
            pass
        else:
            self.weights[index] = uniform(min, max) - self.weights[index]
            pass


    def activate(self):
        # For matrix multiplication the Y value must be the same
        inputMatrix = np.reshape(
            self.input,
            (
                1,
                len(self.input)
            )
        )

        weightMatrix = np.reshape(
            self.weights,
            (
                len(self.input),
                self.outputNodes # The number of weights is o*i
            )
        )

        self.output = np.matmul(
            inputMatrix,
            weightMatrix
        )[0] # 0 because np.matmul gives back an array

        # Now evaluate yourself with root mean square
        self.eval = 0.0
        for i in range(self.outputNodes):
            self.eval += pow(
                (self.target[i] -
                self.output[i]),
                2
            )

class Model:
    def __init__(self, inputNodes, outputNodes):
        self.input = np.array([])
        self.output = np.array([])
        self.targetOut = np.array([])
        self.weights = np.array([])

        self.inputNodes = inputNodes
        self.outputNodes = outputNodes
        self.weightNodes = inputNodes*outputNodes

    def initWeights(self, min, max):
        # Reset the weights
        self.weights = np.random.uniform(min, max, self.weightNodes)

    # Numpy reroute functions #
    def setInput(self, input):
        self.input = np.array(input)

    def setTargetOut(self, targetOut):
        self.targetOut = np.array(targetOut)
    # Numpy reroute functions #

    def train(self, epochs, batchSize, min, max):
        for i in range(epochs):
            agents = []
            for j in range(batchSize):
                agents.append(Agent(
                    self.input,
                    np.copy(self.weights),
                    self.outputNodes,
                    self.targetOut
                ))

            for AGENT in agents:
                AGENT.mutate(min, max)
                AGENT.activate()

            smartestAgent = agents[0]
            for AGENT in agents:
                if (AGENT.eval < smartestAgent.eval):
                    smartestAgent = AGENT
            self.weights = smartestAgent.weights

            print(f"GENERATION NUMBER {i}. AGENT EVAL {smartestAgent.eval}")
            

# SAMPLE CODE
n = Model(3, 5)
n.initWeights(0, 5)
n.setInput([0, 1, 2])
n.setTargetOut([0, 0, 0, 0, 0])
n.train(200, 2000, 1, 2)
