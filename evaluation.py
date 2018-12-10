from percept import Percept
from mdp import MDP
import numpy as np

class Evaluation:
    def __init__(self, mdp: MDP):
        self._mdp = mdp
        self.qvalues = np.zeros((16, 4))
        self.vvalues = np.zeros(16)

    @property
    def getqvalues(self):
        #print(self.qvalues)
        return self.qvalues

    def qLearning(self, percept: Percept):
        #mdp updaten
        self._mdp.update(percept)
        #qvalues ophalen die er nu staan
        currentQ = self.qvalues[percept.current_state, percept.action]
        #qvalues van volgende state
        nextQ = self.qvalues[percept.next_state, percept.action]

        #print(nextQ)
        #de reward die we hebben gekregen door hier terecht te komen
        reward = percept.reward
        #de hoogste qwaarde van de state waar we zijn terecht gekomen
        maxq = max(self.qvalues[percept.next_state])

        #qwaardes voor deze state updaten
        self.qvalues[percept.current_state, percept.action] = currentQ + self._mdp.learningRate * (reward + self._mdp.discountFactor * (maxq - currentQ))


        for v in range(0, 16):
            self.vvalues[v] = max(self.qvalues[v])

        #print(self.qvalues)
