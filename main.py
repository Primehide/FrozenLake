from agent import Agent
from learningStrategies.QLearning import QLearning
from learningStrategies.NStepQLearning import NStepQlearning
from learningStrategies.MonteCarlo import MonteCarlo
from learningStrategies.ValueIteration import ValueIteration

def method_picker(argument):
    switcher = {
        '1': ValueIteration(),
        '2': QLearning(),
        '3': NStepQlearning(),
        '4': MonteCarlo()
    }
    return switcher.get(argument, QLearning())



if __name__ == '__main__':
    print("Welke evaluation wil je gebruiken")
    print("1) Value iteration")
    print("2) qlearning")
    print("3) nstep-qlearing")
    print("4) monte carlo")
    input = input("Wat is uw keuze")
    print("Uw keuze is : ")
    print(input)
    learningStrategy = method_picker(input)
    agent = Agent(learningStrategy)
    agent.learn()

