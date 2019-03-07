#!/usr/bin/env python3

import numpy as np

MAX_TIME_STEPS = 300    # Pendulum env should finish after 200


def reset_state(env, start_state):
    env.reset()
    env.unwrapped.state = start_state
#    print(env.unwrapped.state)
    return env.unwrapped._get_obs()


def get_episode(env, policy, start_state):
    state = reset_state(env, start_state)
    states = [state]
    rewards = []
    for _ in range(MAX_TIME_STEPS):
#        env.render()
#        print(observation)
        action = policy(env, state)
        state, reward, done, info = env.step(action)
        states.append(state)
        rewards.append(reward)
        if done:
            break
    returns = np.cumsum(rewards[::-1])[::-1]
    return states, rewards, returns    


def mc_eval(env, policy, tc, lr, start_state, num_episodes):
    num_tilings = tc.tilings[0].ntilings
    d = tc.size
    w = np.random.uniform(-0.0001, 0.0001, size=d)
#    values = [np.sum(w[tc(start_state)])]
    values = []
    for i in range(num_episodes):
#        print('Episode:', i)
        states, rewards, returns = get_episode(env, policy, start_state)
        
        # transform from hidden state space to observation space
        start_obs = states[0]
        
        # Update weights
        for t in range(len(returns)):
            w_idxs = tc(states[t])   # the only weights that are used for update

            w[w_idxs] += lr / num_tilings * (returns[t] - np.sum(w[w_idxs]))
#        print(w[tc(start_obs)])
        values.append(np.sum(w[tc(start_obs)]))
    return values


def td_eval(env, policy, tc, lr, lambd):
    pass



#class RandomAgent(object):
#    """The world's simplest agent!"""
#    def __init__(self, action_space):
#        self.action_space = action_space
#
#    def act(self, observation, reward, done):
#        return self.action_space.sample()
    
    