#!/usr/bin/env python3

import numpy as np
from GridWorld import GridWorld

MAX_ITERS = 100000
LEGAL_ACTIONS = ['LEFT', 'RIGHT', 'UP', 'DOWN']


def policy_evaluation(transition, reward, discount=0.9, num_iterations=None, epsilon=.001):
    ''' Policy evaluation implementation
    Args:
        transition:     |S| x |S| ndarray transition conditioned on the policy
        reward:         |S| x 1 ndarray
        
    Returns:
        state values under the given policy
    '''
    num_states = len(reward)
    v_prev = np.zeros(num_states)
    
    if num_iterations is None:
        num_iterations = MAX_ITERS
    else:
        epsilon = None
        
    for _ in range(num_iterations):
        v = reward + discount * transition @ v_prev
        delta = np.max(np.abs(v - v_prev))
        if epsilon is not None and delta < epsilon:
            break
        
    return v


def policy_transition_matrix(policy, gridworld):
    ''' Given, a deterministic policy, return the transition dynamics where the
    desired action is taken with probability p and a random action is taken
    with probability (1 - p)
    Args:
        policy:         |S| length vector of action indices
        
    Returns:
        transition_matrix:  |S| x |S|
    '''
    return gridworld.get_transition_matrix(policy)


def policy_improvement(state_values, gridworld):
    ''' Get greedy policy from given state values
    Args:
        state_values:   |S| length vector of values
        
    Returns:
        policy:         |S| length vector of action indices
    '''
    return gridworld.get_greedy_policy(state_values)
    
