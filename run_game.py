from ChefsHatGym.Agents import Agent_Naive_Random

from ChefsHatGym.Rewards import RewardOnlyWinning

from ChefsHatGym.env import ChefsHatEnv

from memory_agent import MemoryAgent
from my_agent import MyAgent
from dqn_agent import DQNAgent, ReplayMemory
import gym
import itertools
from q_agent import QAgent, QTable

"""Game parameters"""
gameType = ChefsHatEnv.GAMETYPE["MATCHES"]
gameStopCriteria = 10
rewardFunction = RewardOnlyWinning.RewardOnlyWinning()

"""Player Parameters"""
# table = QTable()
# agent1 = Agent_Naive_Random.AgentNaive_Random("Random1")
# agent2 = MyAgent("MyAgent2")

memory = ReplayMemory(100000)
agent1 = DQNAgent("MyAgent1", memory)
agent2 = MemoryAgent("Random2", memory)
agent3 = MemoryAgent("Random3", memory)
agent4 = MemoryAgent("Random4", memory)

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
episodes = 100


"""Setup environment"""
env = gym.make('chefshat-v0') #starting the game Environment
env.startExperiment(rewardFunctions=rewards, gameType=gameType, stopCriteria=gameStopCriteria, playerNames=agentNames, logDirectory=saveDirectory, verbose=verbose, saveDataset=True, saveLog=True)

"""Start Environment"""

# for a in range(episodes):
for a in itertools.count(start=1):

    observations = env.reset()

    while not env.gameFinished:
        currentPlayer = playersAgents[env.currentPlayer]

        observations = env.getObservation()
        action = currentPlayer.getAction(observations)

        info = {"validAction":False}
        while not info["validAction"]:
            nextobs, reward, isMatchOver, info = env.step(action)
            currentPlayer.actionUpdate(observations, nextobs, action, reward, info)

        if isMatchOver:
            print("-------------")
            print("Match:" + str(info["matches"]))
            print("Score:" + str(info["score"]))
            print("Performance:" + str(info["performanceScore"]))
            print("-------------")
