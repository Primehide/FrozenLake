from learningStrategies.learningStrategy import LearningStrategy
from percept import Percept
import numpy as np

class MonteCarlo(LearningStrategy):

    def __init__(self):
        super().__init__()
        self._perceptsBuffer = []

    def evaluate(self, percept: Percept):
        self._mdp.update(percept)
        self._perceptsBuffer.append(percept)
        # print("adding percept")

        if (percept.final_state):
            # print("episode is done, updating q values")
            for p in self._perceptsBuffer:
                currentQ = self._qvalues[p.current_state, p.action]
                reward = p.reward
                maxq = np.max(self._qvalues[p.next_state, :])
                self._qvalues[p.current_state, p.action] = currentQ + self._learningRate * (
                        reward + self._mdp.discountFactor * maxq - currentQ)
            self._perceptsBuffer.clear()


