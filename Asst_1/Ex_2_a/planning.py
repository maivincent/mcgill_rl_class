#!/usr/bin/env python3

import numpy as np
from GridWorld import LEGAL_ACTIONS
from GridWorld import int_policy_to_str_policy

MAX_ITERS = 100000
DISCOUNT = 0.9
EPSILON = 10e-6


def modified_policy_iteration(gridworld, discount=DISCOUNT, 
                              num_eval_iters=None, epsilon_eval=None, seed=0,
                              return_all_policies=False):
    ''' Generalized policy iteration

    Returns:
        state values under the given policy
    '''
    np.random.seed(seed)
    num_states = gridworld.n ** 2
    reward = gridworld.get_rewards()
    policy = np.random.choice(range(len(LEGAL_ACTIONS)), num_states)
    
    print('discount2:', discount)
    
    policies = []
    v = np.zeros(num_states)
    while True:
        # policy evaluation
        transition = gridworld.get_transition_matrix(policy)
        v = modified_policy_evaluation(
                transition, reward, v_init=v, discount=discount, 
                num_iters=num_eval_iters, epsilon=epsilon_eval)
        
#        print(v.reshape(gridworld.n, gridworld.n))
        
        # policy improvement
        new_policy = gridworld.get_greedy_policy(v)
        if np.all(policy == new_policy):
            break
        
        policy = new_policy
        policies.append(policy)
        
#        print(int_policy_to_str_policy(policy).reshape(gridworld.n, gridworld.n))
    
    if return_all_policies:
        return policies
    else:
        return policy


def policy_iteration(gridworld, discount=DISCOUNT, seed=0,
                     return_all_policies=False):
    return modified_policy_iteration(gridworld, discount,
                            num_eval_iters=MAX_ITERS, epsilon_eval=EPSILON,
                            seed=seed, return_all_policies=return_all_policies)
    

def value_iteration(gridworld, discount=DISCOUNT, seed=0,
                    return_all_policies=False):
    return modified_policy_iteration(gridworld, discount, num_eval_iters=1,
                            seed=seed, return_all_policies=return_all_policies)
    

def modified_policy_evaluation(transition, reward, v_init=None, discount=DISCOUNT, 
                               num_iters=None, epsilon=None):
    ''' Generalized policy evaluation
    Args:
        transition:     |S| x |S| ndarray transition conditioned on the policy
        reward:         |S| x 1 ndarray
        
    Returns:
        state values under the given policy
    '''
    num_states = len(reward)
    if v_init is None:
        v_prev = np.zeros(num_states)
    else:
        v_prev = v_init
    
    # default to full policy evaluation
    if num_iters is None and epsilon is None:
        epsilon = EPSILON
        num_iters = MAX_ITERS
        
    for _ in range(num_iters):
        v = reward + discount * transition @ v_prev
        delta = np.max(np.abs(v - v_prev))
        if epsilon is not None and delta < epsilon:
            break
        v_prev = v
    return v


def full_policy_evaluation(transition, reward, discount=DISCOUNT):
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
    
