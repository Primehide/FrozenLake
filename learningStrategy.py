from mdp import MDP
from percept import Percept

class LearningStrategy:
    def __init__(self):
        self._mdp = MDP()

    @property
    def mdp(self) -> MDP:
        return self._mdp

    def learn(self, percept : Percept):
        self.mdp.update(percept)