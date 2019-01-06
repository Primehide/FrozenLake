from agent import Agent
from learningStrategies.QLearning import QLearning
from learningStrategies.NStepQLearning import NStepQlearning
from learningStrategies.MonteCarlo import MonteCarlo
from learningStrategies.ValueIteration import ValueIteration
import time

def method_picker(argument):
    switcher = {
        '1': ValueIteration(),
        '2': QLearning(),
        '3': NStepQlearning(),
        '4': MonteCarlo()
    }
    return switcher.get(argument, QLearning())



if __name__ == '__main__':
    print("")
    print("Welke evaluation wil je gebruiken")
    print("1) Value iteration")
    print("2) qlearning")
    print("3) nstep-qlearing")
    print("4) monte carlo")
    choice = input("Wat is uw keuze: ")

    learningStrategy = method_picker(choice)
    agent = Agent(learningStrategy)

    print("")
    print("The game map and its rewards:")
    agent.print_rewards()
    print("")
    print("intial policy: all randon, 25% chance")
    agent.getstrategy().print_policy()
    print("")
    print("the algorithm starts printing once it's found a reward.")
    print("game will start in 5 seconds...")
    print("")
    time.sleep(5)


    "Start learning"
    agent.learn()

    print("")
    print("Done playing, end policy: ")
    agent.getstrategy().print_policy()

