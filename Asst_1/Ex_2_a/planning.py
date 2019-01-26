#!/usr/bin/env python3

import numpy as np
from GridWorld import GridWorld

MAX_ITERS = 100000
LEGAL_ACTIONS = ['LEFT', 'RIGHT', 'UP', 'DOWN']
EPSILON = 10e-6


def modified_policy_iteration(gridworld, discount=0.9, 
                              num_eval_iters=None, epsilon_eval=None):
    ''' Generalized policy iteration implementation
    Args:
        transition:     |S| x |S| ndarray transition conditioned on the policy
        reward:         |S| x 1 ndarray
        
    Returns:
        state values under the given policy
    '''
    num_states = gridworld.n ** 2
    reward = gridworld.get_rewards()
    policy = np.random.choice(range(len(gridworld.LEGAL_ACTIONS)), num_states)
    
    while True:
        # policy evaluation
        transition = gridworld.get_transition_matrix(policy)
        v = modified_policy_evaluation(
                transition, reward, discount=discount, 
                num_iters=num_eval_iters, epsilon=epsilon_eval)
        
        # policy improvement
        new_policy = gridworld.get_greedy_policy(v)
        if policy == new_policy:
            break
    
    return policy

def modified_policy_evaluation(transition, reward, discount=0.9, 
                               num_iters=None, epsilon=None):
    ''' Modified policy evaluation implementation
    Args:
        transition:     |S| x |S| ndarray transition conditioned on the policy
        reward:         |S| x 1 ndarray
        
    Returns:
        state values under the given policy
    '''
    num_states = len(reward)
    v_prev = np.zeros(num_states)
    
    if num_iters is None and epsilon is None:
        epsilon = EPSILON
        num_iters = MAX_ITERS

    for _ in range(num_iters):
        v = reward + discount * transition @ v_prev
        delta = np.max(np.abs(v - v_prev))
        if epsilon is not None and delta < epsilon:
            break
    return v

def full_policy_evaluation(transition, reward, discount=0.9):
    return modified_policy_evaluation(transition, reward, discount=discount)

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
    
