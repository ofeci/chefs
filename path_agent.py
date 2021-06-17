import numpy
from ChefsHatGym.Agents import IAgent
from ChefsHatGym.Rewards import RewardPerformanceScore, RewardOnlyWinning
import random
import numpy as np

from collections import Counter


class QTable:
    def __init__(self):
        self.q_table = {}


class PathAgent:

    def __init__(self, name="PathAgent"):

        self.name = "PATH_" + name
        self.reward = RewardPerformanceScore.RewardPerformanceScore()

        self.paths = Counter()
        self.total = Counter()
        self.eps = 1
        self.current_path = []

    def get_best_action(self, itemindex):
        actions = [self.paths[i]/self.total.get(i, 1) for i in itemindex]
        best = itemindex[np.argmax(np.array(actions))]
        return int(best)

    def getAction(self, observations):
        possible_actions = observations[28:]
        itemindex = numpy.array(numpy.where(numpy.array(possible_actions) == 1))[0].tolist()
        random.shuffle(itemindex)

        best_action = self.get_best_action(observations) if random.random() > self.eps else None
        best_action = best_action if best_action in possible_actions else None

        aIndex = best_action or itemindex[0]

        a = numpy.zeros(200)
        a[aIndex] = 1

        self.current_path.append(aIndex)
        return a

    def actionUpdate(self, observations, nextobs, action, reward, info):
        pass

    def observeOthers(self, envInfo):
        pass

    def matchUpdate(self, envInfo):
        pass

    def update_epsilon(self):
        self.eps = 0.9 * self.eps
        if self.eps < 0.1:
            self.eps = 0.1

    def getReward(self, info, stateBefore, stateAfter):
        thisPlayerPosition = info["thisPlayerPosition"]
        matchFinished = info["thisPlayerFinished"]
        thisPlayerIndex = info["thisPlayer"]
        performanceScore = info['performanceScore'][thisPlayerIndex]

        if matchFinished and thisPlayerPosition == 0:
            self.paths.update(self.current_path)
            self.current_path = []

        if matchFinished:
            self.update_epsilon()

        self.total.update(self.current_path)

        return self.reward.getReward(thisPlayerPosition, performanceScore, matchFinished)
