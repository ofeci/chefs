import numpy
from ChefsHatGym.Agents import IAgent
from ChefsHatGym.Rewards import RewardPerformanceScore, RewardOnlyWinning
import random
import numpy as np

class QTable:
    def __init__(self):
        self.q_table = {}

class QAgent:

    def __init__(self, name="QAgent", table: QTable = None):

        self.name = "Q_" + name
        self.reward = RewardPerformanceScore.RewardPerformanceScore()

        self.q_table = {} if table is None else table.q_table
        self.gamma = 0.9
        self.lr = 0.01
        self.eps = 1

    def get_best_action(self, observations):
        state = tuple(observations[:28])
        if state in self.q_table:
            actions = self.q_table[state]
            best_action = np.argmax(actions)
            return best_action
        return None

    def update_epsilon(self):
        self.eps = 0.99 * self.eps

    def getAction(self, observations):
        possible_actions = observations[28:]
        itemindex = numpy.array(numpy.where(numpy.array(possible_actions) == 1))[0].tolist()
        random.shuffle(itemindex)

        best_action = self.get_best_action(observations) if random.random() > self.eps else None
        best_action = best_action if best_action in possible_actions else None

        aIndex = best_action or itemindex[0]

        a = numpy.zeros(200)
        a[aIndex] = 1

        return a

    def actionUpdate(self, observations, nextobs, action, reward, info):
        state = tuple(observations[:28])
        next_state = tuple(nextobs[:28])
        action_index = np.argmax(action)

        if state not in self.q_table:
            self.q_table[state] = np.zeros(200)

        next_q = 0
        if next_state in self.q_table:
            next_q = max(self.q_table[next_state])

        target = reward + self.gamma * next_q

        self.q_table[state][action_index] += self.lr * (target - self.q_table[state][action_index])

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
