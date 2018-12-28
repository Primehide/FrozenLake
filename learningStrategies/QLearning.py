from learningStrategies.learningStrategy import LearningStrategy
from percept import Percept
import numpy as np


class QLearning(LearningStrategy):

    def __init__(self):
        super().__init__()



    #Override van evaluate functie in super klasse
    def evaluate(self, percept: Percept):
        # mdp updaten
        self._mdp.update(percept=percept)
        # qvalues ophalen die er nu staan
        currentQ = self._qvalues[percept.current_state, percept.action]
        # de reward die we hebben gekregen door hier terecht te komen
        reward = percept.reward
        # de hoogste qwaarde van de state waar we zijn terecht gekomen
        maxq = np.max(self._qvalues[percept.next_state, :])

        # qwaardes voor deze state updaten
        self._qvalues[percept.current_state, percept.action] = currentQ + self._learningRate * (
                    reward + self._mdp.discountFactor * maxq - currentQ)

        for v in range(0, 16):
            self._vvalues[v] = max(self._qvalues[v])
        # print(self.qvalues)


