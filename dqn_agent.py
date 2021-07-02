import pickle
import random
from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BUFFER_SIZE = int(2e3)  # replay buffer size
BATCH_SIZE = 16  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 3e-3  # for soft update of target parameters
LR = 1e-3  # learning rate
UPDATE_EVERY = 4  # how often to update the network

PASS = 199

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


class BrilliantReward:
    ranks = [1, 0.5, -0.5, -1]

    def getReward(self, thisPlayerPosition, performanceScore, matchFinished):
        reward = - 0.001
        if matchFinished:
            reward = self.ranks[thisPlayerPosition] + performanceScore
        return reward


class DQN(nn.Module):

    def __init__(self, embed_dim=15, state_size=28, action_size=200, lin_size=[800, 800, 600]):
        super(DQN, self).__init__()
        self.state_size = state_size
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(14, self.embed_dim)
        self.dropout = nn.Dropout(p=0.1)

        self.classifier = nn.Sequential(nn.Linear(self.state_size * self.embed_dim, lin_size[0]),
                                        nn.ReLU(),

                                        self.dropout,
                                        nn.Linear(lin_size[0], lin_size[1]),
                                        nn.ReLU(),

                                        self.dropout,
                                        nn.Linear(lin_size[1], action_size))

    def forward(self, x):
        x = self.embed(x).view(-1, self.state_size * self.embed_dim)
        return self.classifier(x)


class BrilliantAgent:

    def __init__(self, name="Agent", continue_training=True):
        self.name = "Brilliant_" + name

        self.continue_training = continue_training

        self.reward = BrilliantReward()

        self.memory = ReplayMemory(BUFFER_SIZE)

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
    def load(file, continue_training=False):
        with open(file, 'rb') as f:
            agent = pickle.load(f)
            agent.continue_training = continue_training
            return agent

    def update_epsilon(self):
        self.eps = 0.998 * self.eps if self.eps > 0.1 else 0.1

    def get_state(self, observations):
        state = torch.from_numpy((observations[:28] * 13).astype(np.int)).to(DEVICE)
        return state

    def getAction(self, observations):
        state = self.get_state(observations)
        possible_actions = observations[28:]
        itemindex = np.array(np.where(np.array(possible_actions) == 1))[0].tolist()

        if itemindex != [PASS]:
            itemindex.remove(PASS)

        with torch.no_grad():
            self.policy.eval()
            actions = self.policy(state.unsqueeze(0)).squeeze()

        actions = [actions[index] for index in itemindex]
        best_action = int(np.argmax(actions))
        best_action = itemindex[best_action]

        if random.random() > self.eps and self.continue_training:
            random.shuffle(itemindex)
            best_action = itemindex[0]

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

        batch = Transition(*zip(*transitions))

        next_states = torch.cat(batch.next_state).view(BATCH_SIZE, -1)
        state_batch = torch.cat(batch.state).view(BATCH_SIZE, -1)
        action_batch = torch.cat(batch.action).view(BATCH_SIZE, -1)
        reward_batch = torch.cat(batch.reward).view(BATCH_SIZE, -1)

        state_action_values = self.policy(state_batch).gather(1, action_batch)

        with torch.no_grad():
            labels_next = self.target(next_states).detach().max(1)[0].unsqueeze(1)

        expected_state_action_values = reward_batch + (labels_next * GAMMA)

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.policy, self.target, TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(),
                                             local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    def actionUpdate(self, observations, nextobs, action, reward, info):
        state = self.get_state(observations)
        action = torch.tensor([np.argmax(action)]).to(DEVICE)
        reward = torch.tensor([reward]).to(DEVICE)

        if self.last_state is not None and int(self.last_action[0]) != PASS:
            self.memory.push(self.last_state, self.last_action, state, self.last_reward)
        self.last_state = state
        self.last_action = action
        self.last_reward = reward

        self.step_count += 1
        if self.step_count % UPDATE_EVERY == 0 and self.continue_training:
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

        return self.reward.getReward(thisPlayerPosition, performanceScore, matchFinished)
