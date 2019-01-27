#!/usr/bin/env python3

import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from GridWorld import GridWorld
from planning import policy_transition_matrix, full_policy_evaluation

class TestStringMethods(unittest.TestCase):
    
    def setUp(self):
        self.n = 5
        self.p = 1
        self.gridworld = GridWorld(self.n, self.p)
        self.go_right_policy = np.ones(self.n * self.n, dtype=int)
        self.discount = .9
    
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


if __name__ == '__main__':
    unittest.main()