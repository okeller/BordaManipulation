from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import *

import unittest
import numpy as np

import reverse_algorithm
import utils
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class TestReverseAlgorithm(unittest.TestCase):
    def test_find_strategy(self):
        initial_sigmas = np.array([5, 6, 6, 6, 7], dtype=np.int32)
        weights = np.array([1, 1])
        k = len(weights)
        m = len(initial_sigmas)

        alpha = utils.borda(m)

        res = reverse_algorithm.find_strategy(initial_sigmas, alpha, weights)

        self.assertEquals(np.ravel(res).tolist(), [4, 0, 1, 3, 2, 2, 3, 1, 0, 4])

        # rray([[4, 0],
        #       [1, 3],
        #       [2, 2],
        #       [3, 1],
        #       [0, 4]])


if __name__ == '__main__':
    unittest.main()
