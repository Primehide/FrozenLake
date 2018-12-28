from percept import Percept
from mdp import MDP
import numpy as np

class Evaluation:
    def __init__(self, mdp: MDP, policy):
        self._mdp = mdp
        self.qvalues = np.zeros((16, 4))
        self.qvalues2 = np.zeros((16, 4))
        self.qvalues3 = np.zeros((16, 4))
        self.vvalues = np.zeros(16)
        self.perceptsBuffer = []
        self.N = 50
        self.precision = 0.01
        self._policy = policy


    @property
    def getqvalues(self):
        #print(self.qvalues)
        return self.qvalues

    @property
    def getvvalues(self):
        # print(self.qvalues)
        return self.vvalues

    def qLearning(self, percept: Percept):
        #mdp updaten
        self._mdp.update(percept)
        #qvalues ophalen die er nu staan
        currentQ = self.qvalues[percept.current_state, percept.action]
        #de reward die we hebben gekregen door hier terecht te komen
        reward = percept.reward
        #de hoogste qwaarde van de state waar we zijn terecht gekomen
        maxq = np.max(self.qvalues[percept.next_state, :])

        # qwaardes voor deze state updaten
        self.qvalues[percept.current_state, percept.action] = currentQ + self._mdp.learningRate * (reward + self._mdp.discountFactor * maxq - currentQ)



        for v in range(0, 16):
            self.vvalues[v] = max(self.qvalues[v])

        #print(self.qvalues)

    def nstepqlearning(self, percept: Percept):
        self._mdp.update(percept)
        self.perceptsBuffer.append(percept)

        if (self.perceptsBuffer.__len__() >= self.N):
            for p in self.perceptsBuffer:
                currentQ = self.qvalues2[p.current_state, p.action]
                reward = p.reward
                maxq = np.max(self.qvalues2[p.next_state, :])
                self.qvalues2[p.current_state, p.action] = currentQ + self._mdp.learningRate * (
                            reward + self._mdp.discountFactor * maxq - currentQ)

            #print(self.qvalues2)
            self.perceptsBuffer.clear()

    def montecarlo(self, percept: Percept, done: bool):
        self._mdp.update(percept)
        self.perceptsBuffer.append(percept)
        #print("adding percept")

        if (done):
            #print("episode is done, updating q values")
            for p in self.perceptsBuffer:
                currentQ = self.qvalues3[p.current_state, p.action]
                reward = p.reward
                maxq = np.max(self.qvalues3[p.next_state, :])
                self.qvalues3[p.current_state, p.action] = currentQ + self._mdp.learningRate * (
                        reward + self._mdp.discountFactor * maxq - currentQ)
            self.perceptsBuffer.clear()

    def vevaluete(self, percept: Percept):
        self._mdp.update(percept)
        #max reward uit mdp halen
        r_max = np.max(self._mdp.matrix[0, ::, ::, ::])
        #delta even heel hoog zetten zodat de loop begint
        delta = np.inf

        while delta > self.precision * r_max * (1 - self._mdp.discountFactor) / self._mdp.discountFactor:
            delta = 0
            #over elke state lopen, moet nog dynamische worden, niet vast 16
            for s in range(16):
                #oude v value opslagen
                u = self.vvalues[s]
                #value functie geeft 4 waardes terug
                #1 voor elke actie in die state
                self.qvalues[s] = self.value_function(s)
                #de grootste waarde word onze vwaarde
                self.vvalues[s] = np.max(self.qvalues[s])
                delta = max(delta, abs(u - self.vvalues[s]))
                #print(delta)
                #print(self.vvalues)

    def value_function(self, s):
        return [self._policy[s, a] * sum(
            [self._mdp.matrix[s, a, s_, 3] * (self._mdp.matrix[s, a, s_, 0] + self._mdp.discountFactor * self.vvalues[s_]) for s_ in
             range(16)]) for a in range(4)]





