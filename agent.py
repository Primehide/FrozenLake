from mdp import MDP
from percept import Percept
import gym


class Agent:
    def __init__(self):
        self._mdp = MDP()
        self.env = gym.envs.make("FrozenLake-v0")
        self.episodes = 8000
        self.count = 0

    @property
    def mdp(self) -> MDP:
        return self._mdp

    def learn(self):
        while self.count < self.episodes:
            # env resetten
            # huidige status
            current_state = self.env.reset()
            self.count = self.count + 1
            done = False
            while not done:
                # random actie uitvoeren
                action = self.env.action_space.sample()
                # step actie geeft 4 waardes terug
                # 1 observation (object), 2 reward (float), 3 done(bool), 4 info dict
                step = self.env.step(action)
                # percept aanmaken
                percept = Percept(current_state, action, step[1], step[0], step[2], step[3]['prob'])
                # huidige state aanpassen
                current_state = percept.next_state
                # nakijken of we dood zijn, zoja is de episode klaar
                done = percept.final_state
                self.mdp.update(percept)
        print(self.mdp.matrix[0])