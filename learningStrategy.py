from mdp import MDP
from percept import Percept
from random import randint
from random import uniform
import numpy as np
import gym
from evaluation import Evaluation
import math

ALL_POSSIBLE_ACTIONS = (0, 1, 2, 3)

class LearningStrategy:

    def __init__(self):
        self._mdp = MDP()
        self._evaluation = Evaluation(mdp=self._mdp)
        self._epsilon = 1.0
        self._epsilonMin = 0.01
        self._epsilonMax = 1.0
        self.policy3 = np.ones((16, 4)) / 4
        self.policy2 = np.ones((16, 4)) / 4
        self.count = 0
        print("Rewards:")
        self.print_rewards(4, 4)
        print()

        # initial policy
        print("initial policy:")
        self.print_policy(4,4)

    @property
    def mdp(self) -> MDP:
        return self._mdp

    @property
    def getepsilon(self):
        return self._epsilon

    def setCount(self, x):
        self.count = x

    def next_action(self, state: int):
        action = np.random.choice(
            ALL_POSSIBLE_ACTIONS,
            p=self.policy3[state]
        )

        return action


    def learn(self, percept: Percept, done: bool):
        #self.mdp.update(percept)
        self.evaluate(percept, done)
        #values = self._evaluation.getqvalues
        self.improve()

    def getqvalues(self):
        return self._evaluation.getqvalues


    def evaluate(self,percept: Percept, done: bool):
        #self._evaluation.qLearning(percept)
        #self._evaluation.nstepqlearning(percept)
        self._evaluation.montecarlo(percept, done)

    def improve(self):
        for s in range(0, 16):
            bestAction = self.argMax(s)
            for a in range(0, 4):
                if (a == bestAction):
                    self.policy3[s, a] = 1 - self._epsilon + (self._epsilon / 4)
                    #self.policy2[s, a] = 1 - self._epsilon + (self._epsilon / 4)
                else:
                    self.policy3[s, a] = self._epsilon / 4
                    #self.policy2[s, a] = self._epsilon / 4

        self._epsilon = self._epsilonMin + (self._epsilonMax - self._epsilonMin) * math.pow(math.e, -0.006 * self.count)
        #print(self._epsilon)
        #self.print_policy(4, 4)






    def argMax(self, state : int):
        values = self._evaluation.getqvalues
        maxArray = np.where(self._evaluation.getqvalues[state] == max(values[state]))
        return np.random.choice(maxArray[0])



    def print_policy(self, height: int, width):
        count = 0
        print("------------")
        for h in range(0, height):
            for w in range(0, width):
                print("|", end="")
                print(self.policy3[count], end="")
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




