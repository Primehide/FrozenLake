from mdp import MDP
from percept import Percept
import numpy as np
import math
from abc import ABC, abstractmethod

ALL_POSSIBLE_ACTIONS = (0, 1, 2, 3)

class LearningStrategy(ABC):

    def __init__(self):
        self._mdp = MDP()
        self._qvalues = np.zeros((16, 4))
        self._vvalues = np.zeros(16)
        self._policy = np.ones((16,4))/4
        self._learningRate = 0.8
        self._epsilonDecay = -0.005
        self._epsilon = 1.0
        self._epsilonMin = 0.01
        self._epsilonMax = 1.0
        self._count = 0


    def learn(self, percept: Percept):
        self.evaluate(percept)
        self.improve()


    @abstractmethod
    def evaluate(self,percept: Percept):
        pass


    def improve(self):
        for s in range(0, 16):
            bestAction = self.argMax(s)
            for a in range(0, 4):
                if (a == bestAction):
                    self._policy[s, a] = 1 - self._epsilon + (self._epsilon / 4)
                else:
                    self._policy[s, a] = self._epsilon / 4

            self._epsilon = self._epsilonMin + (self._epsilonMax - self._epsilonMin) * math.pow(math.e, self._epsilonDecay * self._count)


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

    def print_rewards(self, height: int, width):
        count = 0
        print("------------")
        for h in range(0, height):
            for w in range(0, width):
                print("|", end="")
                if(count != 15):
                    print(0.0, end="")
                else:
                    print(1.0, end="")
                print("|", end="")
                count = count + 1
            print("")

    def setCount(self, x):
        self._count = x

