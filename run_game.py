import pickle

from ChefsHatGym.Agents.Agent_Naive_Random import AgentNaive_Random

from ChefsHatGym.Rewards import RewardOnlyWinning
from ChefsHatGym.env import ChefsHatEnv
from my_agent import MyAgent
import gym
import itertools
from brilliant_agent import BrilliantAgent
import numpy as np

"""Game parameters"""
gameType = ChefsHatEnv.GAMETYPE["MATCHES"]
gameStopCriteria = 10

"""Player Parameters"""
# agent1 = MyAgent("1")
# agent1 = AgentNaive_Random("Random1")
# agent1 = BrilliantAgent("B1")
agent1 = BrilliantAgent("B1", saveModelIn="models", type="B1", continue_training=False)


agent2 = BrilliantAgent("B2", saveModelIn="models", type="V3", continue_training=False)
# agent2 = MyAgent("2")
# agent2 = DQNAgent("MyAgent2")


agent3 = BrilliantAgent("B3", saveModelIn="models", type="V2", continue_training=False)
# agent3 = MyAgent("3")
# agent3 = BrilliantAgent("B3")

agent4 = BrilliantAgent("B4", saveModelIn="models", type="V1", continue_training=False)
# agent4 = MyAgent("4")
# agent4 = BrilliantAgent("B4")


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
env = gym.make('chefshat-v0')
env.startExperiment(rewardFunctions=rewards, gameType=gameType, stopCriteria=gameStopCriteria, playerNames=agentNames, logDirectory=saveDirectory, verbose=verbose, saveDataset=True, saveLog=True)

"""Start Environment"""
wins = [0] * 4

episodes = 1
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
            # print("-------------")
            # print("Match:" + str(info["matches"]))
            # print("Score:" + str(info["score"]))
            # print("Performance:" + str(info["performanceScore"]))
            if env.gameFinished:
                wins[np.argmax(np.array(info["score"]))] += 1
                status = "Wins:" + str(wins) + " Rates:" + str([int(100 * w / sum(wins)) for w in wins])
                print(status)
            # print("-------------")

# agent1.save(dir="models")
# agent2.save(dir="models")
# agent3.save(dir="models")
# agent4.save(dir="models")
