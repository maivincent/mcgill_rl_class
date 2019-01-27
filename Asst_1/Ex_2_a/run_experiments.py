#!/usr/bin/env python3

from GridWorld import GridWorld
from planning import modified_policy_iteration, policy_iteration, \
    value_iteration, full_policy_evaluation, policy_transition_matrix


def get_true_bottom_corner_values(policy, gridworld, discount):
    transition = policy_transition_matrix(policy, gridworld)
    reward = gridworld.get_rewards()
    values = full_policy_evaluation(transition, reward, discount)
    values = gridworld.vector_to_grid(values)
    
    n = gridworld.n
    bottom_left_value = round(values[n - 1, 0], 3)
    bottom_right_value = round(values[n - 1, n - 1], 3)
    return bottom_left_value, bottom_right_value

if __name__ == '__main__':
    discount = 0.9
    n_list = [5, 50]
    p_list = [0.9, 0.7]
    k_list = [20, 5, 3]
    for n in n_list:
        for p in p_list:
            print('{} x {} grid with p = {}\n'.format(n, n, p))
            gridworld = GridWorld(n, p)
            
            # policy iteration
            policies = policy_iteration(gridworld, discount, return_all_policies=True)
            bottom_corner_vals = [
                    get_true_bottom_corner_values(policy, gridworld, discount) 
                        for policy in policies]
            print('policy iteration \t(k = inf):\t{}'.format(bottom_corner_vals))
            
            # modied policy iteration
            for k in k_list:
                policies = modified_policy_iteration(gridworld, discount,
                                                     num_eval_iters=k,
                                                     return_all_policies=True)
                bottom_corner_vals = [
                    get_true_bottom_corner_values(policy, gridworld, discount) 
                        for policy in policies]
                print('modified policy iteration (k = {}):\t{}'
                      .format(k, bottom_corner_vals))
            
            # value iteration
            policies = value_iteration(gridworld, discount, return_all_policies=True)
            bottom_corner_vals = [
                    get_true_bottom_corner_values(policy, gridworld, discount) 
                        for policy in policies]
            print('value iteration \t(k = 1):\t{}'.format(bottom_corner_vals))
            
            print('\n')
