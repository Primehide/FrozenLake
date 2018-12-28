from learningStrategies.learningStrategy import LearningStrategy
from percept import Percept
import numpy as np


class NStepQlearning(LearningStrategy):

    def __init__(self):
        super().__init__()
        self._perceptsBuffer = []
        self._N = 20

    def evaluate(self, percept: Percept):
        self._mdp.update(percept)
        self._perceptsBuffer.append(percept)

        if (self._perceptsBuffer.__len__() >= self._N):
            for p in self._perceptsBuffer:
                currentQ = self._qvalues[p.current_state, p.action]
                reward = p.reward
                maxq = np.max(self._qvalues[p.next_state, :])
                self._qvalues[p.current_state, p.action] = currentQ + self._learningRate * (
                        reward + self._mdp.discountFactor * maxq - currentQ)

            # print(self.qvalues2)
            self._perceptsBuffer.clear()


