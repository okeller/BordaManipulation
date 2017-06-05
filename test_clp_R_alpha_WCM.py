from __future__ import absolute_import, division, print_function, unicode_literals
import unittest
import logging
import sys

import numpy as np
from cvxopt import solvers
from nose.tools.nontrivial import nottest

import average_fit

import clp_R_alpha_WCM
import lp_solver
import utils

solvers.options['show_progress'] = False
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

import time

current_milli_time = lambda: int(round(time.time() * 1000))


class TestClpRAlphaWcm(unittest.TestCase):
    # @nottest
    def test_find_violated_constraints(self):
        y = np.array([2.5, 10., 20.])
        z = np.ravel(np.array([[0.5, 0.5, 3.0], [0.5, 0.5, 3.0]]).transpose())

        alpha = np.array([0, 1, 2])
        targets = np.array([2, 1, 0])
        weights = np.array([1, 1])

        coeff, name = clp_R_alpha_WCM.find_violated_constraints(y, z, targets, alpha, weights)

        self.assertEquals(name, [(u'C', 0, u'2,0')])

        coeff = np.ravel(coeff)
        y_coeff, z_coeff_flat = coeff[:3], coeff[3:]

        candidate = y_coeff.nonzero()[0][0]

        configuration_as_matrix = z_coeff_flat.reshape((3, 2))

        self.assertLessEqual(np.dot(np.dot(alpha, configuration_as_matrix), weights),
                             targets[candidate])  # check configuration is in C_i(T)

        self.assertGreater(np.dot(z_coeff_flat, z), y[candidate])  # values

    # @nottest
    def test_full(self):
        initial_sigmas = np.array([5, 6, 6, 6, 7], dtype=np.int32)
        alpha = np.arange(5)  # Borda
        weights = np.array([1, 1])
        m = len(initial_sigmas)

        assert isinstance(initial_sigmas, np.ndarray)
        gaps = utils.sigmas_to_gaps(initial_sigmas, np.max(initial_sigmas))
        logger.info('gaps={}'.format(gaps))

        init_gaps = gaps
        t1 = current_milli_time()
        frac_res, clp_res = clp_R_alpha_WCM.find_strategy(initial_sigmas, alpha, weights, mode='per_cand')
        t2 = current_milli_time()

        logger.warning('Took time {}ms, frac_res={}'.format(t2 - t1, frac_res))

        fractional_makespan = utils.weighted_makespan(frac_res, alpha, weights, initial_sigmas)
        clp_makespan = utils.weighted_makespan(clp_res, alpha, weights, initial_sigmas)

        logger.info(
            'weights={} m={}frac={} CLP={}'.format(weights, m, fractional_makespan, clp_makespan))
        # result_to_append = [n, k, m, trial, initial_sigmas, fractional_makespan, clp_makespan, af_makespan]
        # print(result_to_append)

    # @nottest
    def test_draw_interim_configs(self):
        input = [
            [(u'config_A1', 0.5), (u'config_A2', 0.5)],
            [(u'config_B1', 0.9), (u'config_B2', 0.1)]]

        output = clp_R_alpha_WCM.draw_interim_configs(input)
        self.assertEquals(len(output), 2)
        self.assertTrue(output[0].startswith(u'config_A'))
        self.assertTrue(output[1].startswith(u'config_B'))

    # @nottest
    def test_fix_rounding_result_weighted(self):
        k = 2
        m = 3

        config_mat = np.array([[0, 1], [2, 2], [2, 0]])

        initial_sigmas = np.array([0, 1, 1])
        weights = np.array([1, 1])
        alpha = np.arange(m)  # borda

        res_config_mat = clp_R_alpha_WCM.fix_rounding_result_weighted(config_mat, weights, initial_sigmas, alpha)

        self.assertEquals(np.ravel(res_config_mat).tolist(), [0, 1, 2, 2, 1, 0])


if __name__ == '__main__':
    unittest.main()
