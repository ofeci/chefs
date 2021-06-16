import numpy
from ChefsHatGym.Agents import IAgent
from ChefsHatGym.Rewards import RewardOnlyWinning
import random
from ChefsHatGym.Agents import Agent_Naive_Random

import torch
import numpy as np

from dqn_agent import ReplayMemory


class MemoryAgent(Agent_Naive_Random.AgentNaive_Random):
    def __init__(self, name="Memory", memory=None):
        super(MemoryAgent, self).__init__(name)
        self.memory = memory or ReplayMemory(100000)

    def actionUpdate(self, observations, nextobs, action, reward, info):
        state = torch.from_numpy(observations[:28].astype(np.float32))
        next_state = torch.from_numpy(nextobs[:28].astype(np.float32))
        action = torch.tensor([np.argmax(action)])
        reward = torch.tensor([reward])

        # Store the transition in memory
        self.memory.push(state, action, next_state, reward)
