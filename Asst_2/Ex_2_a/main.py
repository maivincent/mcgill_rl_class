#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import gym
from tilecoding.representation import TileCoding
from linapprox import mc_eval, td_lambda_eval

NUM_BINS = 10
NUM_TILINGS = 5
NUM_EPISODES = 200
LRS = [1/4, 1/8, 1/16]
NUM_SEEDS = 10
LAMBDAS = [0, 0.3, 0.7, 0.9, 1]
#LRS = [1/8]
#NUM_SEEDS = 2
#LAMBDAS = [.3]


def policy(env, state):
    x, y, velocity = state
    r = np.random.rand()
    neg_torque = env.action_space.low
    pos_torque = env.action_space.high
    if velocity < 0:
        if r < .9:
            action = neg_torque
        else:
            action = pos_torque
    elif velocity > 0:
        if r < .9:
            action = pos_torque
        else:
            action = neg_torque
    else:
        if r < .5:
            action = neg_torque
        else:
            action = pos_torque
    return action


def get_tile_coding(env):
    state_range = [env.observation_space.low, env.observation_space.high]
    tc = TileCoding(input_indices = [np.arange(env.observation_space.shape[0])],
    				ntiles = [NUM_BINS],
    				ntilings = [NUM_TILINGS],
                    bias_term=False,
    				hashing = None,
    				state_range = state_range,
    				rnd_stream = np.random.RandomState())
    return tc
    

def plot_vals(value_lists, labels, title):
    plt.figure()
    x = list(range(1, NUM_EPISODES + 1))
    for i, value_list in enumerate(value_lists):
        mean = np.mean(value_list, axis=0)
        std = np.std(value_list, axis=0)
        plt.plot(x, mean, label=labels[i])
        plt.fill_between(x, mean - std, mean + std, alpha=0.3)
    plt.title(title)
    plt.xlabel('number of episodes')
    plt.ylabel('initial state value under fixed policy')
    plt.legend()
    plt.show()
    plt.savefig('{}.pdf'.format(title), bbox_inches='tight')

       
if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    start_state = np.array([np.pi, 0])  # pendulum at bottom
#    start_state = np.array([0, 0])  # pendulum at top
#    start_state = np.array([np.pi/2, 0])    # pendulum to left
#    start_state = np.array([-np.pi/2, 0])   # pendulum to right
    for lambd in LAMBDAS:
        value_lists = []
        for lr in LRS:
            value_seeds = np.zeros((NUM_SEEDS, NUM_EPISODES))
            for seed in range(NUM_SEEDS):
                print('lr:', lr)
                np.random.seed(seed)
                tc = get_tile_coding(env)
#                value_seeds[seed, :] = mc_eval(env, policy, tc, lr, start_state,
#                                               NUM_EPISODES)
                value_seeds[seed, :] = td_lambda_eval(env, policy, tc, lr,
                           lambd, start_state, NUM_EPISODES, gamma=1)
            value_lists.append(value_seeds)
        plot_vals(value_lists, ['lr={}'.format(lr) for lr in LRS], 'TD({})'.format(lambd))
        
    