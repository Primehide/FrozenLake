import unittest

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

class ProbabilityTest(ParametrizedTestCase):
    def test_probability(self):
        """Test om na te kijken of de kansverdeling van alle acties per state 1 is."""
        for p in self.param:
            probability = 0
            for v in p:
                probability += v
            self.assertAlmostEqual(probability, 1)

class EpsilonTest(ParametrizedTestCase):
    epsilon = 1
    newEpsilon = 0
    def test_epsilon_decay(self):
        self.assertGreaterEqual(self.epsilon, self.param)
        self.epsilon = self.param






