from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import *

import unittest
import numpy as np
import utils
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class TestUtils(unittest.TestCase):
    def test_calculate_awarded(self):
        m = 3
        config_mat = np.array([[0, 1], [2, 1], [2, 0]])
        initil_sigmas = np.array([10, 11, 12])
        weights = np.array([1, 2])
        alpha = utils.borda(m)  # borda
        awarded = utils.weighted_calculate_awarded(config_mat, alpha, weights, initil_sigmas)

        self.assertEquals(awarded.tolist(), [12, 15, 14])

    def test_polya_eggenberger_rankings(self):
        m = 5
        n = 6
        rand = np.random.RandomState(452452)

        res = utils.draw_polya_eggenberger_rankings(m, n, rand)
        np.testing.assert_array_equal(
            np.array([[4, 1, 3, 2, 0], [2, 3, 0, 4, 1], [4, 0, 2, 3, 1], [2, 3, 0, 4, 1], [0, 3, 4, 1, 2],
                      [0, 3, 4, 1, 2]]), res)

    def test_draw_zipf_weights(self):
        rand = np.random.RandomState(452452)
        weights = utils.draw_zipf_weights(5, owners=5, rand=rand)
        np.testing.assert_array_equal(np.array([4, 1]), weights)

    def test_draw_zipf_(self):
        rand = np.random.RandomState(452452)
        weights = utils.draw_zipf(5, rand=rand)
        np.testing.assert_array_equal(np.array([4, 1]), weights)

    def test_rankings_to_initial_sigmas_weighted(self):
        weights = np.array([1, 5])
        rankings = np.array([0, 1, 2, 2, 1, 0]).reshape((2, 3))
        sigmas = utils.rankings_to_initial_sigmas_weighted(rankings, weights)
        np.testing.assert_array_equal(sigmas, np.array([10, 6, 2]))

    if __name__ == '__main__':
        unittest.main()
