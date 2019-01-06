from percept import Percept
from learningStrategies.learningStrategy import LearningStrategy
import gym
import unittest
try:
    import tests.unittesting
except:
    "just ignore it"


class Agent:
    def __init__(self, learningStrategy: LearningStrategy):
        self.env = gym.envs.make("FrozenLake-v0")
        self._states = self.env.observation_space.n  # aantal states in de omgeving
        self._strategy = learningStrategy #Bij aanmaken strategy meegeven
        self.episodes = 2000
        self.count = 0
        self.totalRewards = 0
        self._strategy.setStates(self._states)

    def getstates(self):
        return self._states

    def getcount(self):
        return self.count

    def getstrategy(self):
        return self._strategy



    def learn(self):
        while self.count < self.episodes:
            # env resetten
            # huidige status
            current_state = self.env.reset()
            self._strategy.setCount(self.count) #niet hier doen
            self.count = self.count + 1
            done = False
            while not done:
                # random actie uitvoeren
                action = self._strategy.next_action(current_state)
                # step actie geeft 4 waardes terug
                # 1 observation (object), 2 reward (float), 3 done(bool), 4 info dict
                step = self.env.step(action)
                # percept aanmaken
                percept = Percept(current_state, action, step[1], step[0], step[2])
                self.totalRewards += percept.reward
                # huidige state aanpassen
                current_state = percept.next_state
                # nakijken of we dood zijn, zoja is de episode klaar
                done = percept.final_state
                #print(done)
                self._strategy.learn(percept)


        print("Average reward: " + (self.totalRewards / self.count).__str__())
        self._strategy.print_policy()


