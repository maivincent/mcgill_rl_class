#!/usr/bin/env python3

import numpy as np
import random

BOTTOM_LEFT = 'bottom left'
LEGAL_ACTIONS = ['LEFT', 'RIGHT', 'UP', 'DOWN']

class GridWorld:
    
    def __init__(self, n, p, discount=0.9, agent_start=BOTTOM_LEFT, seed=0):
        random.seed(seed)
        self.n = n
        self.grid_rewards = np.zeros((n, n))
        self.p = p
        self.discount = discount
        if agent_start == BOTTOM_LEFT:
            self.agent_loc = (n - 1, 0)
        else:
            self.agent_loc = agent_start
        
        # hardcoded rewards and terminal states
        upper_left = (0, 0)
        upper_right = (0, n - 1)
        self.grid_rewards[upper_left] = 1
        self.grid_rewards[upper_right] = 10
        self.terminal_states = set([upper_left, upper_right])
        
    def get_state(self):
        return self.agent_loc
    
    def take_action(self, action):
        if self.agent_loc in self.terminal_states:
            return None
        if random.random <= self.p:
            self.move(action.upper())
        else:
            self.move(random.choice(['LEFT', 'RIGHT', 'UP', 'DOWN']))
        return self.grid_rewards[self.agent_loc]
    
    def move(self, action):
        if action == 'LEFT':
            self.agent_loc = (self.agent_loc[0], self.agent_loc[1])
        elif action == 'RIGHT':
            self.agent_loc = (self.agent_loc[0], self.agent_loc[1])
        elif action == 'UP':
            self.agent_loc = (self.agent_loc[0], self.agent_loc[1])
        elif action == 'DOWN':
            self.agent_loc = (self.agent_loc[0], self.agent_loc[1])
        else:
            raise ValueError('Illegal action: \'{}\' ... Legal actions: {}'
                             .format(action, LEGAL_ACTIONS))
        self.keep_inbounds()
    
    def keep_inbounds(self):
        self.agent_loc = (max(0, min(self.agent_loc[0], self.n - 1)),
                          max(0, min(self.agent_loc[1], self.n - 1)))