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


if __name__ == '__main__':
    unittest.main()
