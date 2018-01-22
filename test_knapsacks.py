from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import *

import unittest
import numpy as np
import knapsacks
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class TestKMultisetKnapsack(unittest.TestCase):
    def test_k_multiset_knapsack(self):
        values = np.array([1, 2, 3, 4.1, 5])
        weights = np.array([10, 2, 10, 7, 10])
        target_value = 6.01
        weight_bound = 9  # should be int
        res = knapsacks.k_multiset_knapsack(values=values, weights=weights, k=2, target_value=target_value,
                                            weight_bound=weight_bound)

        logger.info('res={}'.format(res))
        self.assertListEqual(res, [1, 3])
        self.assertGreater(values[res].sum(), target_value)  # value greater than target_value
        self.assertLessEqual(weights[res].sum(), weight_bound)  # weight less than or equal weight_bound

    def test_k_multiset_knapsack_numpy(self):
        values = np.array([1, 2, 3, 4.1, 5])
        weights = np.array([10, 2, 10, 7, 10])
        target_value = 6.01
        weight_bound = 9  # should be int
        res = knapsacks.k_multiset_knapsack_numpy(values=values, weights=weights, k=2, target_value=target_value,
                                            weight_bound=weight_bound)

        logger.info('res={}'.format(res))
        self.assertListEqual(res, [1, 3])
        self.assertGreater(values[res].sum(), target_value)  # value greater than target_value
        self.assertLessEqual(weights[res].sum(), weight_bound)  # weight less than or equal weight_bound

    def test_k_sequence_knapsack(self):
        values = np.array([[1, 2, 3, 4.1, 5], [1, 2, 3, 4.1, 5]]).transpose()
        item_weights = np.array([10, 2, 10, 7, 10])
        target_value = 6.01
        weight_bound = 9  # should be int
        penalties = np.array([1, 1])  # should be int
        res = knapsacks.k_sequnce_knapsack(values=values, item_weights=item_weights, penalties=penalties,
                                           target_value=target_value,
                                           weight_bound=weight_bound)

        logger.info('res={}'.format(res))

        res_value = np.sum([values[score_idx, ell] for ell, score_idx in enumerate(res)])
        panalized_weight = np.dot(item_weights[res], penalties)

        self.assertGreater(res_value, target_value)  # value greater than target_value
        self.assertLessEqual(panalized_weight, weight_bound)  # weight less than or equal weight_bound

    def test_k_sequence_knapsack2(self):
        values = np.array([[0.4598779912508748, 0.4598779919990039], [0.6085798198326219, 0.6085798198896052],
                           [0.6085798221182851, 0.6085798206190864], [0.6085798182073581, 0.6085798190092567]])
        item_weights = np.array([0, 1, 2, 3])
        target_value = 0.739461704232
        weight_bound = 6  # should be int
        penalties = np.array([2, 2])  # should be int
        res = knapsacks.k_sequnce_knapsack(values=values, item_weights=item_weights, penalties=penalties,
                                           target_value=target_value,
                                           weight_bound=weight_bound)

        logger.info('res={}'.format(res))
        # self.assertListEqual(res, [2, 2])

        res_value = np.sum([values[score_idx, ell] for ell, score_idx in enumerate(res)])
        panalized_weight = np.dot(  item_weights[res] , penalties)

        self.assertGreater(res_value, target_value)  # value greater than target_value
        self.assertLessEqual(panalized_weight, weight_bound)  # weight less than or equal weight_bound


if __name__ == '__main__':
    unittest.main()
