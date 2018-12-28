from percept import Percept
from learningStrategy import LearningStrategy
import gym


class Agent:
    def __init__(self):
        self._strategy = LearningStrategy()
        self.env = gym.envs.make("FrozenLake-v0")
        self.episodes = 2000
        self.count = 0
        self.totalRewards = 0

    def learn(self):
        while self.count < self.episodes:
            # env resetten
            # huidige status
            current_state = self.env.reset()
            self._strategy.setCount(self.count)
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
                self._strategy.learn(percept, done)
                if (percept.reward == 1):
                    print("found reward on episode: " + self.count.__str__())


        print("Average reward: " + (self.totalRewards / self.count).__str__())
        #print(self._strategy.getqvalues())
        #print(self._strategy.getepsilon)
        self._strategy.print_policy(4, 4)
        #print(self._strategy.mdp.matrix)

