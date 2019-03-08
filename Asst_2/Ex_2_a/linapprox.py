#!/usr/bin/env python3

import numpy as np
import sys

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


def td_lambda_eval(env, policy, tc, lr, lambd, start_state, num_episodes, gamma=1):
    num_tilings = tc.tilings[0].ntilings
    d = tc.size
    w = np.random.uniform(-0.0001, 0.0001, size=d)
#    values = [np.sum(w[tc(start_state)])]
    values = []
    for i in range(num_episodes):
        states, rewards, returns = get_episode(env, policy, start_state)
#        print(rewards)
#        print(np.sum(rewards))
#        sys.exit()
        
        # transform from hidden state space to observation space
        start_obs = states[0]
        
        # Update weights
        z = np.zeros(d)
        for t in range(len(rewards)):
            w_idxs = tc(states[t])
            grad = np.zeros(d)
            grad[w_idxs] = 1
            z = gamma * lambd * z + grad
            delta = rewards[t] + gamma * np.sum(w[tc(states[t + 1])]) - np.sum(w[w_idxs])
            w += lr / num_tilings * delta * z
#        print(w[tc(start_obs)])
        values.append(np.sum(w[tc(start_obs)]))
    return values
    