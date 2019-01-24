from utils import CFunction, UFunction, secondLargest
import math
import numpy as np
import random


class ActionEliminationAlgo():
    def __init__(self, delta, epsilon):
        self.r = 1
        self.omega = {1, 2, 3, 4, 5, 6}
        self.est_means = [None for k in range(len(self.omega))]
        self.received_rewards = [[] for k in range(len(self.omega))]
        self.number_times_picked = [0 for k in range(len(self.omega))]
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
        est_means_with_bounds = [0 for k in range(len(self.omega))]
        bounds = [0 for k in range(len(self.omega))]
        for i in range(len(est_means_with_bounds)):
            bounds[i] = self.bound(i+1)
            est_means_with_bounds[i] = self.est_means[i] + bounds[i]
        curr_best = max(est_means_with_bounds)
        curr_best_arm_id = est_means_with_bounds.index(curr_best) + 1
        for arm_id in self.epoch_list:
            if arm_id != curr_best_arm_id:
                if curr_best - 2*bounds[curr_best_arm_id -1] > est_means_with_bounds[arm_id - 1]:
                    self.omega.remove(arm_id)
        #curr_best_mean = max(self.est_means)
        #curr_best_arm_id = self.est_means.index(curr_best_mean) + 1
        #for arm_id in self.epoch_list:
        #    if arm_id != curr_best_arm_id:
        #        if curr_best_mean - self.bound(curr_best_arm_id) > self.est_means[arm_id - 1] + self.bound(arm_id):
        #            self.omega.remove(arm_id)


    def nextAction(self):
        if len(self.epoch_list) == 1:
            return self.epoch_list[0]
        else:
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



class UCBAlgo():
    def __init__(self, delta, epsilon):
        self.omega = {1, 2, 3, 4, 5, 6}
        self.est_means = [None for k in range(len(self.omega))]
        self.means_with_bounds = [None for k in range(len(self.omega))]
        self.received_rewards = [[] for k in range(len(self.omega))]
        self.number_times_picked = [0 for k in range(len(self.omega))]
        self.delta = delta
        self.epsilon = epsilon
        self.omega_list = list(self.omega)
        self.initialisation_done = False
        self.initial_count = 0
        self.is_done = False
        self.end_result = None
        self.beta = 1.66


    def update(self, arm_id, reward):
        self.received_rewards[arm_id - 1].append(reward)
        self.number_times_picked[arm_id - 1] += 1
        self.est_means[arm_id - 1] = np.mean(self.received_rewards[arm_id - 1])

    def nextAction(self):
        if self.is_done:
            return self.end_result
        else:
            if not self.initialisation_done:
                action = self.omega_list[self.initial_count]
                self.initial_count += 1
                if self.initial_count >= len(self.omega_list):
                    self.initialisation_done = True
            else:
                for i in range(len(self.omega)):
                    self.means_with_bounds[i] = self.est_means[i] + self.bound(i+1)
                curr_best_mean_with_bounds = max(self.means_with_bounds)
                action_id = self.means_with_bounds.index(curr_best_mean_with_bounds)
                action = self.omega_list[action_id]
            return action

    def bound(self, arm_id):
        bound = (1+self.beta)*CFunction(self.number_times_picked[arm_id - 1], self.delta, len(self.omega), self.epsilon)
        return bound


    def isDone(self):
        curr_best_mean = max(self.est_means)
        best_mean_act = self.est_means.index(curr_best_mean) + 1
        curr_best_mean_minus_bound = curr_best_mean - bound(best_mean_act)

        second_best_mean_plus_bound = secondLargest(self.means_with_bounds)

        if curr_best_mean_minus_bound > second_best_mean_plus_bound:
            self.is_done = True
            self.end_result = best_mean_act
            print("Algo is done!")
            return True
        else:
            return False

    def result(self):
        return self.end_result