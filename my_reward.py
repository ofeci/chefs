class MyReward:
    rewardName = "MyReward"
    ranks = [1, 0.5, -0.5, -1]

    def getReward(self, thisPlayerPosition, performanceScore, matchFinished):
        reward = - 0.001
        if matchFinished:
            # finalPoints = (2 - thisPlayerPosition)/2
            # reward = finalPoints
            reward = self.ranks[thisPlayerPosition]

        return reward
