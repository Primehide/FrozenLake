from percept import Percept
from mdp import MDP
import numpy as np

class Evaluation:
    def __init__(self, mdp: MDP):
        self._mdp = mdp
        self.qvalues = np.zeros((16, 4))
        self.vvalues = np.zeros(16)

    @property
    def qValues(self):
        return self.qvalues

    def qLearning(self, percept: Percept):
        self._mdp.update(percept)
        qprevious = self.qvalues[percept.current_state, percept.action]
        reward = percept.reward
        maxq = max(self.qvalues[percept.next_state])

        self.qvalues[percept.current_state, percept.action] = qprevious + self._mdp.learningRate * (reward + self._mdp.discountFactor * maxq)


        for v in range(0, 16):
            self.vvalues[v] = max(self.qvalues[v])

        print("q values")
        print(self.qvalues)


