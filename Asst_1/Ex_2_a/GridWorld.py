#!/usr/bin/env python3

import numpy as np
import random

BOTTOM_LEFT = 'bottom left'
LEGAL_ACTIONS = ['LEFT', 'RIGHT', 'UP', 'DOWN']
TERMINAL_I = len(LEGAL_ACTIONS)

@np.vectorize
def int_policy_to_str_policy(action_i):
    if action_i == TERMINAL_I:
        return 'TERMINAL'
    elif 0 <= action_i < len(LEGAL_ACTIONS):
        return LEGAL_ACTIONS[action_i]
    else:
        raise ValueError(
                'Illegal action index: {} ... Legal action indices: {}-{}'
                         .format(action_i, 0, len(LEGAL_ACTIONS) - 1))
        

class GridWorld:
    
    def __init__(self, n, p, discount=0.9, agent_start=BOTTOM_LEFT, seed=None):
        if seed is not None:
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
        
    def get_rewards(self):
        return self.grid_to_vector(self.grid_rewards)
    
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
                if (i, j) not in self.terminal_states:
                    self.set_move_probs(i, j, policy[i, j], transition_probs)
                transitions.append(self.grid_to_vector(transition_probs))
        return np.vstack(transitions)
    
    def get_greedy_policy(self, values):
        '''
        Args:
            values: |S| length vector of state values
            
        Returns:
            policy: |S| length vector of action indices
        '''
        assert len(values) == self.n * self.n
        values = self.vector_to_grid(values)
        policy = np.zeros_like(values, dtype=int)
        for i in range(self.n):
            for j in range(self.n):
                if (i, j) in self.terminal_states:
                    policy[i, j] = TERMINAL_I
                else:
                    local_vals = [values[self.get_inbounds_loc(i, j, action)] \
                                         for action in LEGAL_ACTIONS]
                    policy[i, j] = np.argmax(local_vals)
        return self.grid_to_vector(policy)
    
    def vector_to_grid(self, v):
        return v.reshape(self.n, self.n)
                
    def grid_to_vector(self, grid):
        return grid.flatten()
    
    def set_move_probs(self, i, j, action_i, transition_probs):
        if action_i == TERMINAL_I:
            raise Exception('Terminal_i')
        action_loc = self.get_inbounds_loc(i, j, LEGAL_ACTIONS[action_i])
        transition_probs[action_loc] = self.p
        for action in LEGAL_ACTIONS:
            random_loc = self.get_inbounds_loc(i, j, action)
            transition_probs[random_loc] += (1 - self.p) / len(LEGAL_ACTIONS)
    
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
