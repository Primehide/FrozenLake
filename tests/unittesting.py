import unittest
from agent import Agent
from learningStrategies.QLearning import QLearning
from learningStrategies.NStepQLearning import NStepQlearning
from learningStrategies.MonteCarlo import MonteCarlo
from learningStrategies.ValueIteration import ValueIteration


class ParametrizedTestCase(unittest.TestCase):

    def __init__(self, methodName='runTest', param=None):
        super(ParametrizedTestCase, self).__init__(methodName)
        self.param = param

    @staticmethod
    def parametrize(testcase_klass, param=None):
        """ Create a suite containing all tests taken from the given
            subclass, passing them the parameter 'param'.
        """
        testloader = unittest.TestLoader()
        testnames = testloader.getTestCaseNames(testcase_klass)
        suite = unittest.TestSuite()
        for name in testnames:
            suite.addTest(testcase_klass(name, param=param))
        return suite

    @staticmethod
    def parametrize(testcase_klass, param=None):
        """ Create a suite containing all tests taken from the given
            subclass, passing them the parameter 'param'.
        """
        testloader = unittest.TestLoader()
        testnames = testloader.getTestCaseNames(testcase_klass)
        suite = unittest.TestSuite()
        for name in testnames:
            suite.addTest(testcase_klass(name, param=param))
        return suite


class TestSetup(ParametrizedTestCase):
    agent = Agent(learningStrategy=QLearning())

    def method_picker(self, argument):
        switcher = {
            '1': ValueIteration(),
            '2': QLearning(),
            '3': NStepQlearning(),
            '4': MonteCarlo()
        }
        return switcher.get(argument, QLearning())


    def test_input(self):
        print("")
        print("Welke evaluation wil je gebruiken")
        print("1) Value iteration")
        print("2) qlearning")
        print("3) nstep-qlearing")
        print("4) monte carlo")
        choice = input("Wat is uw keuze: ")

        learningStrategy = self.method_picker(choice)
        self.agent = Agent(learningStrategy)
        #self.agent.learn()

    def test_states(self):
        """Kijkt na of het aantal states voor dit spel correct is ingesteld. Frozenlakev0 heeft 16 states"""
        self.assertEqual(self.agent._states, 16)
        self.assertEqual(self.agent.getstrategy().getStates(), 16)

    def test_probability(self):
        """Test om na te kijken of de kansverdeling van alle acties per state 1 is."""
        for p in self.agent.getstrategy().getpolicy():
            probability = 0
            for v in p:
                probability += v
            self.assertEqual(probability, 10)

    def test_learn(self):
        self.agent.learn()