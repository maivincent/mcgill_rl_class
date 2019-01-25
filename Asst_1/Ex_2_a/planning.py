#!/usr/bin/env python3

import numpy as np

MAX_ITERS = 100000

def policy_evaluation(transition, reward, discount=0.9, num_iterations=None, epsilon=.001):
    ''' Policy evaluation implementation, without tensor product optimization
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
        
def policy_transition_matrix(policy, p):
    ''' Given, a deterministic policy, return the transition dynamics where the
    desired action is taken with probability p and a random action is taken
    with probability (1 - p)
    Args:
        policy:         |S| x |A| ndarray
        
    Returns:
        transition_matrix:  |S| x |S|
    '''
    num_states, num_actions = policy.shape
    policy = p * policy + (1 - p) / num_actions * np.ones_like(policy)
    # TODO
    