from mdp import MDP
from percept import Percept
import numpy as np
import math
from abc import ABC, abstractmethod
from tests.policytests import ProbabilityTest
from tests.policytests import ParametrizedTestCase
from tests.policytests import EpsilonTest
import unittest
import os
import subprocess
import sys

ALL_POSSIBLE_ACTIONS = (0, 1, 2, 3)

class LearningStrategy(ABC):

    def __init__(self):
        self._states = 1 #standaardwaarde, wordt later aangepast eens de omgeving bekend is
        self._mdp = MDP(1)
        self._qvalues = np.zeros((self._states, 4))
        self._vvalues = np.zeros(self._states)
        self._policy = np.ones((self._states, 4))/4
        self._learningRate = 0.8
        self._epsilonDecay = -0.005
        self._epsilon = 1.0
        self._epsilonMin = 0.01
        self._epsilonMax = 1.0
        self._count = 0


    def setStates(self, amount):
        self._states = amount
        #=====Aanpassen aan nieuwe aantal states======
        self._qvalues = np.zeros((self._states, 4))
        self._vvalues = np.zeros(self._states)
        self._policy = np.ones((self._states, 4)) / 4
        self._mdp.setStates(amount)

    def getStates(self):
        return self._states

    def getpolicy(self):
        return self._policy

    def getepsilon(self):
        return self._epsilon


    def learn(self, percept: Percept):
        self.evaluate(percept)
        r_max = np.max(self._mdp.matrix[::, ::, ::, 0])
        if (r_max != 0 and self._count % 100 == 0 and percept.final_state):
            self.printqvalues()
        self.improve()
        #suite = unittest.TestSuite()
        #suite.addTest(ParametrizedTestCase.parametrize(ProbabilityTest, param=self._policy))
        #runner = unittest.TextTestRunner(verbosity=2)
        #runner.run(suite)



    @abstractmethod
    def evaluate(self, percept: Percept):
        pass

    def printqvalues(self):
        print("======QVALUES AFTER PLAYING: " + self._count.__str__() + " EPISODES =========")
        print(self._qvalues)
        print("=============================================================")


    def improve(self):
        for s in range(0, self._states):
            bestAction = self.argMax(s)
            for a in range(0, 4):
                if (a == bestAction):
                    self._policy[s, a] = 1 - self._epsilon + (self._epsilon / 4)
                else:
                    self._policy[s, a] = self._epsilon / 4

            self._epsilon = self._epsilonMin + (self._epsilonMax - self._epsilonMin) * math.pow(math.e, self._epsilonDecay * self._count)

            #suite = unittest.TestSuite()
            #suite.addTest(ParametrizedTestCase.parametrize(EpsilonTest, param=self._epsilon))
            #runner = unittest.TextTestRunner(verbosity=40)
            #runner.run(suite)


    def next_action(self, state: int):
        action = np.random.choice(
            ALL_POSSIBLE_ACTIONS,
            p=self._policy[state]
        )

        return action

    def argMax(self, state : int):
        values = self._qvalues
        maxArray = np.where(values[state] == max(values[state]))
        return np.random.choice(maxArray[0])

    def print_policy(self):
        count = 0
        print("------------")
        for h in range(0, 4):
            for w in range(0, 4):
                print("|", end="")
                print(self._policy[count], end="")
                print("|", end="")
                count = count + 1
            print("")


    def setCount(self, x):
        self._count = x

