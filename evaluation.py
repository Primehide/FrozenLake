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
        #mdp updaten
        self._mdp.update(percept)
        #de vorige qvalues ophalen van de state waar we nu zijn terechtgekomen
        qprevious = self.qvalues[percept.current_state, percept.action]
        #de reward die we hebben gekregen door hier terecht te komen
        reward = percept.reward
        #de hoogste qwaarde van de state waar we zijn terecht gekomen
        maxq = max(self.qvalues[percept.next_state])

        #qwaardes voor deze state updaten
        self.qvalues[percept.current_state, percept.action] = qprevious + self._mdp.learningRate * (reward + self._mdp.discountFactor * maxq - qprevious)

        for v in range(0, 16):
            self.vvalues[v] = max(self.qvalues[v])

        #print(self.qvalues)
