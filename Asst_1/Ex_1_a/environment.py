import numpy as np
import math




class Environment():
    def __init__(self):
        self.means = [1, 0.8, 0.6, 0.4, 0.2, 0]
        self.variance = 0.25
        self.std_dev = math.sqrt(self.variance)

    def draw(self, arm_id):
        mean = self.means[arm_id - 1] # mu_1: 1.0    mu_2: 0.8  ... mu_6: 0.0
        reward = np.random.normal(loc=mean, scale = self.std_dev)
        return reward


