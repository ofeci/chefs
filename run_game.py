from ChefsHatGym.Agents.Agent_Naive_Random import AgentNaive_Random

from ChefsHatGym.Rewards import RewardOnlyWinning

from ChefsHatGym.env import ChefsHatEnv
from path_agent import PathAgent
from memory_agent import MemoryAgent
from my_agent import MyAgent
import gym
import itertools
from q_agent import QAgent, QTable
from dqn_agent import DQNAgent
import numpy as np
"""Game parameters"""
gameType = ChefsHatEnv.GAMETYPE["MATCHES"]
gameStopCriteria = 10

"""Player Parameters"""
# table = QTable()
# agent1 = AgentNaive_Random("Random1")

# agent1 = DQNAgent("MyAgent1")
# memory = ReplayMemory(100000)
agent1 = DQNAgent("1")
# agent1 = PathAgent("Path1")
# agent2 = MemoryAgent("Random2")
agent2 = DQNAgent("2")
agent3 = DQNAgent("3")
agent4 = DQNAgent("4")


agentNames = [agent1.name, agent2.name, agent3.name, agent4.name]
playersAgents = [agent1, agent2, agent3, agent4]

rewards = []
for r in playersAgents:
    rewards.append(r.getReward)

"""Experiment parameters"""
saveDirectory = "examples/"
verbose = False
saveLog = False
saveDataset = False


"""Setup environment"""
env = gym.make('chefshat-v0') #starting the game Environment
env.startExperiment(rewardFunctions=rewards, gameType=gameType, stopCriteria=gameStopCriteria, playerNames=agentNames, logDirectory=saveDirectory, verbose=verbose, saveDataset=True, saveLog=True)

"""Start Environment"""
wins = [0] * 4
episodes = 100
for a in range(episodes):
# for a in itertools.count(start=1):

    observations = env.reset()

    while not env.gameFinished:
        currentPlayer = playersAgents[env.currentPlayer]

        observations = env.getObservation()
        action = currentPlayer.getAction(observations)

        info = {"validAction": False}
        while not info["validAction"]:
            nextobs, reward, isMatchOver, info = env.step(action)
            currentPlayer.actionUpdate(observations, nextobs, action, reward, info)

        if isMatchOver:
            print("-------------")
            print("Match:" + str(info["matches"]))
            print("Score:" + str(info["score"]))
            print("Performance:" + str(info["performanceScore"]))
            if env.gameFinished:
                wins[np.argmax(np.array(info["score"]))] += 1
                print("Wins:" + str(wins))
            print("-------------")

agent1.save("models/agent1")
