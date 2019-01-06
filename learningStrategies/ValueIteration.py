from learningStrategies.learningStrategy import LearningStrategy
from percept import Percept
import numpy as np

class ValueIteration(LearningStrategy):

    def __init__(self):
        super().__init__()
        self._precision = 0.001



    def evaluate(self, percept: Percept):
        self._mdp.update(percept)
        # max reward uit mdp halen
        r_max = np.max(self._mdp.matrix[::, ::, ::, 0])
        # delta even heel hoog zetten zodat de loop begint
        delta = np.inf

        while delta > self._precision * r_max * ((1 - self._mdp.discountFactor) / self._mdp.discountFactor):
            delta = 0
            # over elke state lopen, moet nog dynamische worden, niet vast 16
            for s in range(self.getStates()):
                # oude v value opslagen
                u = self._vvalues[s]
                # de grootste waarde word onze vwaarde
                self._qvalues[s] = self.value_function(s)
                self._vvalues[s] = np.max(self._qvalues[s])
                delta = max(delta, abs(u - self._vvalues[s]))

    def value_function(self, s):

        return [self._policy[s, a] * sum(
            [self._mdp.matrix[s, a, s_, 3] * (self._mdp.matrix[s, a, s_, 0] + self._mdp.discountFactor * self._vvalues[s_])
             for s_ in range(self.getStates())])
                for a in range(4)]


