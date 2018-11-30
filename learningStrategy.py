from mdp import MDP
from percept import Percept
from random import randint
import numpy as np
import gym
from evaluation import Evaluation
import math

ALL_POSSIBLE_ACTIONS = (0, 1, 2, 3)

class LearningStrategy:

    def __init__(self):
        self._mdp = MDP()
        self._evaluation = Evaluation(mdp=self._mdp)
        self._epsilon = 1
        self._epsilonMin = 0.01
        self._epsilonMax = 1.0
        self.policy2 = np.ones((16, 4)) / 4
        self.count = 0
        #policy is random in het begin
        self.policy = {}
        for s in range(0, 16):
            self.policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

        print("Rewards:")
        self.print_rewards(4, 4)
        print()

        # initial policy
        print("initial policy:")
        self.print_policy(4,4)

    @property
    def mdp(self) -> MDP:
        return self._mdp

    def setCount(self, x):
        self.count = x

    def next_action(self, state: int):
        #print(self.policy2[state])
        return np.random.choice(
            ALL_POSSIBLE_ACTIONS,
            p=self.policy2[state]
        )


    def learn(self, percept: Percept):
        self.mdp.update(percept)
        self.evaluate(percept)
        #self.improve()

    def evaluate(self,percept: Percept):
        self._evaluation.qLearning(percept)

    def improve(self):
        for s in range(0, 16):
            bestAction = max(self._evaluation.qvalues[s])
            increased = False
            for a in range(0, 4):
                qval = self._evaluation.qvalues[s, a]
                if(qval == bestAction) and (increased == False):
                    #4 nog aan te passen aan aantal mogelijke acties
                    self.policy2[s, a] = 1 - self._epsilon + (self._epsilon / 4)
                    increased = True
                else:
                    self.policy2[s, a] = self._epsilon / 4
                increased == False

            self._epsilon = self._epsilonMin + (self._epsilonMax - self._epsilonMin) * math.pow(math.e, -0.9 * self.count)






    def print_policy(self, height: int, width):
        count = 0
        print("------------")
        for h in range(0, height):
            for w in range(0, width):
                print("|", end="")
                print(self.policy2[count], end="")
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




