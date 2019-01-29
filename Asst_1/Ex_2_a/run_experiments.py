#!/usr/bin/env python3

from GridWorld import GridWorld
from planning import modified_policy_iteration, policy_iteration, \
    value_iteration, full_policy_evaluation, policy_transition_matrix
import matplotlib.pyplot as plt


def get_true_bottom_corner_values(policy, gridworld, discount):
    transition = policy_transition_matrix(policy, gridworld)
    reward = gridworld.get_rewards()
    values = full_policy_evaluation(transition, reward, discount)
    values = gridworld.vector_to_grid(values)
    
    n = gridworld.n
    bottom_left_value = values[n - 1, 0]
    bottom_right_value = values[n - 1, n - 1]
    return bottom_left_value, bottom_right_value

def get_bottom_corner_lists(policies, gridworld, discount): 
    bottom_left_values = []
    bottom_right_values = []
    for policy in policies:
        bottom_left_value, bottom_right_value = \
            get_true_bottom_corner_values(policy, gridworld, discount)
        bottom_left_values.append(bottom_left_value)
        bottom_right_values.append(bottom_right_value)
    return bottom_left_values, bottom_right_values

def plot_bottom_corner_vals(value_lists, labels, title):
    plt.figure()
    for i, value_list in enumerate(value_lists):
#        plt.plot(range(len(value_list)), value_list, label=labels[i])
        plt.plot(value_list, label=labels[i])
    plt.title(title)
    plt.xlabel('number of policy iterations')
    plt.ylabel('state value under policy')
    plt.legend()
    plt.show()
    plt.savefig('{}.pdf'.format(title), bbox_inches='tight')
    

if __name__ == '__main__':
    discount = 0.9
    n_list = [5, 50]
    p_list = [0.9, 0.7]
    k_list = [20, 10, 5, 3]
    
    for n in n_list:
        for p in p_list:
            title = '{} x {} grid with p = {}'.format(n, n, p)
            print(title)
            gridworld = GridWorld(n, p)
            
            bottom_left_list = []
            bottom_right_list = []
            labels = []
            
            # policy iteration
            policies = policy_iteration(gridworld, discount, return_all_policies=True)
            bottom_left_values, bottom_right_values = \
                get_bottom_corner_lists(policies, gridworld, discount)
            bottom_left_list.append(bottom_left_values)
            bottom_right_list.append(bottom_right_values)
            labels.append('policy iteration (k = inf)')
                        
            # modied policy iteration
            for k in k_list:
                policies = modified_policy_iteration(gridworld, discount,
                                                     num_eval_iters=k,
                                                     return_all_policies=True)
                bottom_left_values, bottom_right_values = \
                    get_bottom_corner_lists(policies, gridworld, discount)
                bottom_left_list.append(bottom_left_values)
                bottom_right_list.append(bottom_right_values)
                labels.append('k={} mod policy iteration'.format(k))
            
            # value iteration
            policies = value_iteration(gridworld, discount, return_all_policies=True)
            bottom_left_values, bottom_right_values = \
                get_bottom_corner_lists(policies, gridworld, discount)
            bottom_left_list.append(bottom_left_values)
            bottom_right_list.append(bottom_right_values)
            labels.append('value iteration (k = 1)')

            # plotting
            left_title = 'bottom left value in {}'.format(title)
            right_title = 'bottom right value in {}'.format(title)
            plot_bottom_corner_vals(bottom_left_list, labels, left_title)
            plot_bottom_corner_vals(bottom_right_list, labels, right_title)
