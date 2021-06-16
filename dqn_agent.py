import gym
import math
import random
import numpy as np
from collections import namedtuple, deque
from itertools import count
from ChefsHatGym.Rewards import RewardPerformanceScore, RewardOnlyWinning

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 5         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
# UPDATE_EVERY = 4        # how often to update the network


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, state_size=28, action_size=200, fc1_unit=100, fc2_unit=100):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_unit)
        self.fc2 = nn.Linear(fc1_unit, fc2_unit)
        self.fc3 = nn.Linear(fc2_unit, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent:

    def __init__(self, name="DQNAgent", memory=None):

        self.name = "DQN_" + name
        self.reward = RewardPerformanceScore.RewardPerformanceScore()

        self.memory = memory or ReplayMemory(100000)

        self.policy = DQN()
        self.target = DQN()

        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        self.criterion = nn.SmoothL1Loss()

        self.eps = 1

    def update_epsilon(self):
        self.eps = 0.999 * self.eps
        if self.eps < 0.1:
            self.eps = 0.1

    def getAction(self, observations):
        state = torch.from_numpy(observations[:28].astype(np.float32))
        possible_actions = observations[28:]
        itemindex = np.array(np.where(np.array(possible_actions) == 1))[0].tolist()
        random.shuffle(itemindex)

        with torch.no_grad():
            actions = self.policy(state)

        actions = [actions[index] for index in itemindex]
        best_action = np.argmax(actions)
        best_action = itemindex[best_action]

        best_action = best_action if random.random() > self.eps else itemindex[0]

        a = np.zeros(200)
        a[best_action] = 1

        return a

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return

        self.policy.train()
        self.target.eval()

        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)

        # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        # non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).view(BATCH_SIZE, -1)

        next_states = torch.cat(batch.next_state).view(BATCH_SIZE, -1)
        state_batch = torch.cat(batch.state).view(BATCH_SIZE, -1)
        action_batch = torch.cat(batch.action).view(BATCH_SIZE, -1)
        reward_batch = torch.cat(batch.reward).view(BATCH_SIZE, -1)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        # state_action_values = self.policy(state_batch)
        state_action_values = self.policy(state_batch).gather(1, action_batch)

        with torch.no_grad():
            labels_next = self.target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute the expected Q values
        expected_state_action_values = reward_batch + (labels_next * GAMMA)

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.policy, self.target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        =======
            local model (PyTorch model): weights will be copied from
            target model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(),
                                             local_model.parameters()):
            # target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)
            target_param.data.copy_(local_param.data)

    def actionUpdate(self, observations, nextobs, action, reward, info):
        state = torch.from_numpy(observations[:28].astype(np.float32))
        next_state = torch.from_numpy(nextobs[:28].astype(np.float32))
        action = torch.tensor([np.argmax(action)])
        reward = torch.tensor([reward])

        # Store the transition in memory
        self.memory.push(state, action, next_state, reward)

        self.optimize_model()

    def observeOthers(self, envInfo):
        pass

    def matchUpdate(self, envInfo):
        pass

    def getReward(self, info, stateBefore, stateAfter):
        thisPlayerPosition = info["thisPlayerPosition"]
        matchFinished = info["thisPlayerFinished"]
        thisPlayerIndex = info["thisPlayer"]
        performanceScore = info['performanceScore'][thisPlayerIndex]

        if matchFinished:
            self.update_epsilon()

        return self.reward.getReward(thisPlayerPosition, performanceScore, matchFinished)
