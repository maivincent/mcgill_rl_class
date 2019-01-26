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
        
    def grid_to_vector(self, grid):
        return grid.flatten()
    
    def vector_to_grid(self, v):
        return v.reshape(self.n, self.n)
    
    def get_transition_matrix(self, policy):
        '''
        Args:
            policy: |S| length vector of action indices
            
        Returns:
            |S| x |S| tranisition matrix for the policy
        '''
        assert len(policy) == self.n * self.n
        policy = self.vector_to_grid(policy)
        transitions = []
        for i in range(self.n):
            for j in range(self.n):
                transition_probs = np.zeros((self.n, self.n))
                self.set_move_probs(i, j, policy[i, j], transition_probs)
                transitions.append(self.grid_to_vector(transition_probs))
        return np.vstack(transitions)
    
    def set_move_probs(self, i, j, action_i, transition_probs):
        action_loc = self.get_inbounds_loc(i, j, LEGAL_ACTIONS[action_i])
        transition_probs[action_loc] = self.p
        for action in LEGAL_ACTIONS:
            random_loc = self.get_inbounds_loc(i, j, action)
            transition_probs[random_loc] += (1 - self.p) / len(LEGAL_ACTIONS)
        print(transition_probs)
    
    def get_inbounds_loc(self, i, j, action):
        if action == 'LEFT':
            dest = (i, j - 1)
        elif action == 'RIGHT':
            dest = (i, j + 1)
        elif action == 'UP':
            dest = (i - 1, j)
        elif action == 'DOWN':
            dest = (i + 1, j)
        else:
            raise ValueError('Illegal action: \'{}\' ... Legal actions: {}'
                             .format(action, LEGAL_ACTIONS))
        return (max(0, min(dest[0], self.n - 1)),
                max(0, min(dest[1], self.n - 1)))
    

# =============================================================================
#     Potentially to remove below
# =============================================================================
        
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
            self.agent_loc = (self.agent_loc[0], self.agent_loc[1] - 1)
        elif action == 'RIGHT':
            self.agent_loc = (self.agent_loc[0], self.agent_loc[1] + 1)
        elif action == 'UP':
            self.agent_loc = (self.agent_loc[0] - 1, self.agent_loc[1])
        elif action == 'DOWN':
            self.agent_loc = (self.agent_loc[0] + 1, self.agent_loc[1])
        else:
            raise ValueError('Illegal action: \'{}\' ... Legal actions: {}'
                             .format(action, LEGAL_ACTIONS))
        self.keep_inbounds()
    
    def keep_inbounds(self):
        self.agent_loc = (max(0, min(self.agent_loc[0], self.n - 1)),
                          max(0, min(self.agent_loc[1], self.n - 1)))