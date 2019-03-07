#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import gym
from tilecoding.representation import TileCoding
from linapprox import mc_eval, td_eval

NUM_BINS = 10
NUM_TILINGS = 5
NUM_EPISODES = 200
LRS = [1/4, 1/8, 1/16]
NUM_SEEDS = 10


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
    for i, value_list in enumerate(value_lists):
        plt.plot(value_list, label=labels[i])
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
    value_lists = []
    for seed in range(1):
        for lr in LRS:
            print('lr:', lr)
            np.random.seed(seed)
            tc = get_tile_coding(env)
            values = mc_eval(env, policy, tc, lr, start_state, NUM_EPISODES)
            value_lists.append(values)
        plot_vals(value_lists, ['lr={}'.format(lr) for lr in LRS], 'Monte Carlo')
        
    