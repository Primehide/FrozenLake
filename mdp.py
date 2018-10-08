import numpy as np
from percept import Percept
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)

class MDP:
    def __init__(self):
        self._tables = np.zeros((4, 16, 16, 4))
        print(self._tables.size)

    @property
    def matrix(self):
        return self._tables

    def update(self, percept : Percept):
        frequentie = self.matrix[percept.action, percept.current_state, percept.current_state, 1] + 1
        frequentienext = self.matrix[percept.action, percept.next_state, percept.next_state, 1] + 1
        #Update R(reward), zit in percept, element 0
        #Update Nsa(state actie frequentie), hoe vaak hebben we op deze staat deze actie uitgevoert
        #Update Ntsa(state s-actie-state s' frequencties
        #Update Ptsa (update transitiemodel, maw wat is da kans dat ik hier uitglij?)
        self.matrix[percept.action, percept.current_state, percept.next_state] = [percept.reward, frequentie, frequentienext, percept.transition]
