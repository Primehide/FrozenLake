from agent import Agent
from learningStrategies.QLearning import QLearning
from learningStrategies.NStepQLearning import NStepQlearning
from learningStrategies.MonteCarlo import MonteCarlo
from learningStrategies.ValueIteration import ValueIteration
from tests import unittesting
import unittest

def method_picker(argument):
    switcher = {
        '1': ValueIteration(),
        '2': QLearning(),
        '3': NStepQlearning(),
        '4': MonteCarlo()
    }
    return switcher.get(argument, QLearning())



if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(unittesting.AITests("test_input"))
    suite.addTest(unittesting.AITests("test_states"))
    suite.addTest(unittesting.AITests("test_probability"))
    runner = unittest.TextTestRunner()
    runner.run(suite)

