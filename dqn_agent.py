import pickle
import random
from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# DEVICE = torch.cuda.set_device(1)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BUFFER_SIZE = int(2e3)  # replay buffer size
BATCH_SIZE = 16  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 2e-3  # for soft update of target parameters
LR = 1e-3  # learning rate
UPDATE_EVERY = 4  # how often to update the network

BANNED = [199, 198]

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class MyReward:
    rewardName = "MyReward"
    ranks = [1, 0.5, -0.5, -1]

    def getReward(self, thisPlayerPosition, matchFinished, stateBefore, stateAfter):
        # s0 = sum([1 for card in stateBefore[11:28] if card == 0])
        # s1 = sum([1 for card in stateAfter[11:28] if card == 0])
        # diff = abs(s0 - s1)
        # reward = 0 if all([card == 0 for card in stateAfter[11:28]]) else -0.01
        reward = 0.001
        if matchFinished:
            # finalPoints = (2 - thisPlayerPosition)/2
            # reward = finalPoints
            reward = self.ranks[thisPlayerPosition]

        return reward


class DQN(nn.Module):

    def __init__(self, state_size=28, action_size=200, lin_size=[800, 1000, 800, 600]):
        super(DQN, self).__init__()
        self.state_size = state_size
        self.embed_dim = 5
        self.embed = nn.Embedding(14, self.embed_dim)
        self.classifier = nn.Sequential(nn.Linear(self.state_size * self.embed_dim, lin_size[0]),
                                        # nn.BatchNorm1d(lin_size[0]),
                                        nn.ReLU(),

                                        nn.Linear(lin_size[0], lin_size[1]),
                                        # nn.BatchNorm1d(lin_size[1]),
                                        nn.ReLU(),

                                        nn.Linear(lin_size[1], lin_size[2]),
                                        # nn.BatchNorm1d(lin_size[2]),
                                        nn.ReLU(),

                                        nn.Linear(lin_size[2], action_size))

    def forward(self, x):
        x = self.embed(x).view(-1, self.state_size * self.embed_dim)
        return self.classifier(x)


class DQNAgent:

    def __init__(self, name="Agent", memory=None):
        self.name = "DQN_" + name
        self.reward = MyReward()

        self.memory = memory or ReplayMemory(BUFFER_SIZE)

        self.policy = DQN().to(DEVICE)
        self.target = DQN().to(DEVICE)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        self.criterion = nn.SmoothL1Loss()

        self.last_state = None
        self.last_action = None
        self.last_reward = None

        self.step_count = 0

        self.eps = 1

    def save(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file):
        with open(file, 'rb') as f:
            agent = pickle.load(f)
            return agent

    def update_epsilon(self):
        self.eps = 0.999 * self.eps
        if self.eps < 0.1:
            self.eps = 0.1

    def get_state(self, observations):
        state = torch.from_numpy((observations[:28] * 13).astype(np.int)).to(DEVICE)
        return state

    def getAction(self, observations):
        state = self.get_state(observations)
        possible_actions = observations[28:]
        itemindex = np.array(np.where(np.array(possible_actions) == 1))[0].tolist()
        itemindex = [i for i in itemindex if i not in BANNED] or itemindex

        random.shuffle(itemindex)

        with torch.no_grad():
            self.policy.eval()
            actions = self.policy(state.unsqueeze(0)).squeeze()

        actions = [actions[index] for index in itemindex]
        best_action = int(np.argmax(actions))
        best_action = itemindex[best_action]

        best_action = best_action if random.random() > self.eps else itemindex[0]

        a = np.zeros(200)
        a[best_action] = 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    def actionUpdate(self, observations, nextobs, action, reward, info):
        state = self.get_state(observations)
        action = torch.tensor([np.argmax(action)]).to(DEVICE)
        reward = torch.tensor([reward]).to(DEVICE)

        if self.last_state is not None and int(self.last_action[0]) not in BANNED:
            self.memory.push(self.last_state, self.last_action, state, self.last_reward)
        self.last_state = state
        self.last_action = action
        self.last_reward = reward

        self.step_count += 1
        if self.step_count % UPDATE_EVERY == 0:
            self.optimize_model()
            self.step_count = 0

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

        return self.reward.getReward(thisPlayerPosition, matchFinished, stateBefore, stateAfter)
