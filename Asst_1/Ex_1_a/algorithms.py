from utils import CFunction, UFunction
import math
import numpy as np
import random

class ActionEliminationAlgo():
    def __init__(self, delta, epsilon):
        self.r = 1
        self.omega = {1, 2, 3, 4, 5, 6}
        self.est_means = [None, None, None, None, None, None]
        self.received_rewards = [[] for k in range(6)]
        self.number_times_picked = [0, 0, 0, 0, 0, 0]
        self.delta = delta
        self.epsilon = epsilon
        self.epoch_done = False
        self.epoch_k = 0
        self.epoch_list = list(self.omega)
        random.shuffle(self.epoch_list)


    def update(self, arm_id, reward):
        self.received_rewards[arm_id - 1].append(reward)
        self.number_times_picked[arm_id - 1] += 1
        self.est_means[arm_id - 1] = np.mean(self.received_rewards[arm_id - 1])
        
        if self.epoch_done:
            self.removeBadArms()
            self.epoch_done = False
            self.epoch_k = 0
            self.epoch_list = list(self.omega)
            random.shuffle(self.epoch_list)


    def bound(self, arm_id):
        bound = CFunction(self.number_times_picked[arm_id - 1], self.delta, len(self.omega), self.epsilon)
        #print("Times picked arm " + str(arm_id) + " is: " + str(self.number_times_picked[arm_id - 1]))
        #print("Bound: " + str(bound))
        return bound


    def removeBadArms(self):
        curr_best_mean = max(self.est_means)
        curr_best_arm_id = self.est_means.index(curr_best_mean) + 1
        for arm_id in self.epoch_list:
            if arm_id != curr_best_arm_id:
                if curr_best_mean - self.bound(curr_best_arm_id) > self.est_means[arm_id - 1] + self.bound(arm_id):
                    self.omega.remove(arm_id)


    def nextAction(self):
        action = self.epoch_list[self.epoch_k % len(self.epoch_list)]
        self.epoch_k += 1
        if self.epoch_k >= len(self.omega) * self.r:
            self.epoch_done = True
            #print("Epoch done! ")
        return action


    def isDone(self):
        if len(self.omega) == 1:
            return True
        else:
            #print("Omega: " + str(self.omega))
            #print("Est means: " + str(self.est_means))
            return False

    def result(self):
        return list(self.omega)[0]





