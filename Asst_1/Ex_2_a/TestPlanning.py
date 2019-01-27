#!/usr/bin/env python3

import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from GridWorld import GridWorld, int_policy_to_str_policy
from planning import policy_transition_matrix, full_policy_evaluation, \
    policy_iteration, value_iteration, modified_policy_iteration

class TestPlanning(unittest.TestCase):
    
    def setUp(self):
        self.n = 5
        self.p = 1
        self.gridworld = GridWorld(self.n, self.p)
        self.go_right_policy = np.ones(self.n * self.n, dtype=int)
        self.discount = 0.9
        self.large_discount = 0.2
        self.policy = np.array(
                [['TERMINAL', 'RIGHT', 'RIGHT', 'RIGHT', 'TERMINAL'],
                 ['RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'UP'],
                 ['RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'UP'],
                 ['RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'UP'],
                 ['RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'UP']])
        self.policy_large_discount = np.array(
                [['TERMINAL', 'LEFT', 'RIGHT', 'RIGHT', 'TERMINAL'],
                 ['UP', 'LEFT', 'RIGHT', 'RIGHT', 'UP'],
                 ['UP', 'LEFT', 'RIGHT', 'RIGHT', 'UP'],
                 ['UP', 'LEFT', 'RIGHT', 'RIGHT', 'UP'],
                 ['UP', 'LEFT', 'RIGHT', 'RIGHT', 'UP']])
    
    def test_transition_matrix(self):
        transition_rows = []
        for i in range(self.n):
            for j in range(self.n):
                transition_row = np.zeros((self.n, self.n))
                if (i, j) not in set([(0, 0), (0, self.n - 1)]):
                    transition_row[i, min(j + 1, self.n - 1)] = 1
                transition_rows.append(transition_row.flatten())
        expected = np.vstack(transition_rows)
    
        actual = policy_transition_matrix(
                self.go_right_policy, self.gridworld)
        assert_array_equal(expected, actual)
        
    def test_full_policy_eval(self):
        transition = policy_transition_matrix(
                self.go_right_policy, self.gridworld)
        reward = self.gridworld.get_rewards()
        actual = full_policy_evaluation(transition, reward, self.discount)
        
        expected = np.zeros((self.n, self.n))
        expected[0, :] = [10 * self.discount ** (self.n - 1 - i) 
                            for i in range(self.n)]
        expected[0, 0] = 1
        expected = expected.flatten()
        
        assert_array_almost_equal(expected, actual)
#        
    def test_policy_iteration(self):
        policy = policy_iteration(self.gridworld, self.discount)
        actual = int_policy_to_str_policy(policy).reshape(self.n, self.n)
        assert_array_equal(self.policy, actual)
        
    def test_policy_iteration_more_discount(self):
        policy = policy_iteration(self.gridworld, self.large_discount)
        actual = int_policy_to_str_policy(policy).reshape(self.n, self.n)
        assert_array_equal(self.policy_large_discount, actual)
        
if __name__ == '__main__':
    unittest.main()