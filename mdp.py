import numpy as np
from percept import Percept
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)

class MDP:
    def __init__(self, states):
        #Dim 1 -> Current State
        #Dim 2 -> Actie
        #Dim 3 -> Next State
        #Dim 4 -> Waardes
        #Dim 4 -> Reward, Nsa, Ntsa, Ptsa
        self._states = states
        self._tables = np.zeros((states, 4, states, 4))
        self._discountFactor = 0.80

    def setStates(self, amount):
        self._states = amount
        self._tables = np.zeros((self._states, 4, self._states, 4))

    @property
    def matrix(self):
        return self._tables

    @property
    def discountFactor(self):
        return self._discountFactor

    def update(self, percept : Percept):
        #Update R(reward), zit in percept, element 0
        self._tables[percept.current_state, percept.action, percept.next_state, 0] = percept.reward
        #Update Nsa(state actie frequentie), hoe vaak hebben we op deze staat deze actie uitgevoert
        self._tables[percept.current_state, percept.action, ::, 1] += 1
        #Update Ntsa(state s-actie-state s' frequencties
        self._tables[percept.current_state, percept.action, percept.next_state, 2] += 1
        #Update Ptsa (update transitiemodel, maw wat is da kans dat ik hier uitglij?)
        #self.matrix[percept.current_state,percept.action,percept.next_state]
        #self.matrix[percept.current_state, percept.action, percept.next_state, 3] = 0

        Nsa = self._tables[percept.current_state, percept.action, percept.next_state, 1]
        Ntsa = self._tables[percept.current_state, percept.action, percept.next_state, 2]
        self._tables[percept.current_state, percept.action, percept.next_state, 3] = Ntsa / Nsa
