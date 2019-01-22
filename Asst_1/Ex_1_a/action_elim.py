from utils import C_function, U_function


class ActionEliminationAlgo():
    def __init__(self, delta, epsilon):
        self.r = 1
        self.omega = {1, 2, 3, 4, 5, 6}
        self.est_means = [None, None, None, None, None, None]
        self.received_rewards = [[] for k in range(6)]
        self.number_times_picked = [0, 0, 0, 0, 0, 0]
        self.delta = delta
        self.epsilon = epsilon
        self.initialized = 


    def update(self, arm_id, reward):
        self.received_rewards[arm_id - 1].append(reward)
        self.number_times_picked[arm_id - 1] += 1
        self.est_means[arm_id - 1] = mean(self.received_rewards[arm_id - 1])


    def bound(self, arm_id):
        return C_function(number_times_picked[arm_id - 1], self.delta, len(self.omega), self.epsilon)

    def remove_bad_arms(self):
        curr_best_mean = max(self.est_means)
        curr_best_arm_id = self.est_means.index(curr_best_mean) + 1
        for arm_id in self.omega:
            if arm_id != curr_best_arm_id:
                if curr_best_mean - self.bound(curr_best_arm_id) > self.est_means[arm_id - 1] + self.bound(arm_id):
                    self.omega.remove(arm_id)


    def next_action(self):





