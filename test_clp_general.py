from __future__ import absolute_import, division, print_function, unicode_literals
import unittest
import logging
import sys

import numpy as np
from cvxopt import solvers
from nose.tools.nontrivial import nottest

import average_fit

import clp_general
import lp_solver
import utils

solvers.options['show_progress'] = False
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

import time

current_milli_time = lambda: int(round(time.time() * 1000))


class TestClpGeneral(unittest.TestCase):
    def test_find_violated_constraints(self):
        y = np.array([2.5, 10., 20.])
        z = np.array([0.5, 0.5, 3.0])
        alpha = np.array([0, 1, 2])
        targets = np.array([2, 1, 0])

        coeff, name = clp_general.find_violated_constraints(y, z, targets, 2)

        self.assertEquals(name, [('C', 0, '1,0,1')])


        coeff = np.ravel(coeff)
        self.assertListEqual(coeff.tolist(), [-1., 0., 0., 1., 0., 1.])
        # coeff_ = np.array(coeff)
        user = coeff[:3].nonzero()[0][0]
        subset = coeff[3:].nonzero()[0]
        self.assertLessEqual(alpha[subset].sum(), targets[user]) # weights
        self.assertGreater(z[subset].sum(), y[user]) # values


    def test_full(self):
        initial_sigmas = np.array([5, 6, 6, 6, 7], dtype=np.int32)
        k = 2
        m = len(initial_sigmas)

        assert isinstance(initial_sigmas, np.ndarray)
        gaps = utils.sigmas_to_gaps(initial_sigmas, np.max(initial_sigmas))
        logger.info('gaps={}'.format(gaps))

        init_gaps = gaps
        t1 = current_milli_time()
        fractional_makespan, clp_res = clp_general.find_strategy(initial_sigmas, k, mode='per_cand')
        t2 = current_milli_time()

        logger.warning('Took time {}ms, fractional_makespan={}'.format(t2 - t1, fractional_makespan))

        # fractional_makespan = utils.makespan(initial_sigmas, frac_res)
        clp_makespan = utils.makespan(initial_sigmas, clp_res
                                      )
        gaps = utils.sigmas_to_gaps(initial_sigmas, clp_makespan)
        af_res = average_fit.solve_one_gaps(k, gaps, verbose=False)
        af_makespan = utils.makespan(initial_sigmas, af_res)
        logger.info(
            'k={} m={}frac={} CLP={} AF={}'.format(k, m, fractional_makespan, clp_makespan,
                                                   af_makespan))
        # result_to_append = [n, k, m, trial, initial_sigmas, fractional_makespan, clp_makespan, af_makespan]
        # print(result_to_append)


    def test_draw_interim_configs(self):
        input = [
            [(u'config_A1', 0.5), (u'config_A2', 0.5)],
            [(u'config_B1', 0.9), (u'config_B2', 0.1)]]

        output = clp_general.draw_interim_configs(input)
        self.assertEquals(len(output), 2)
        self.assertTrue(output[0].startswith(u'config_A'))
        self.assertTrue(output[1].startswith(u'config_B'))

    # @nottest
    def test_fix_rounding_result_weighted(self):
        k = 2
        m = 3

        config_mat = np.array([[1, 1, 0], [0, 0, 2], [1, 0, 1]])

        initial_sigmas = np.array([0, 1, 1])
        weights = np.array([1, 1])
        alpha = np.arange(m)  # borda

        res_config_mat = clp_general.fix_rounding_result(config_mat, k, initial_sigmas)

        self.assertEquals(np.ravel(res_config_mat).tolist(), [1, 1, 0, 0, 0, 2, 1, 1, 0])


if __name__ == '__main__':
    unittest.main()
