from __future__ import absolute_import, division, print_function, unicode_literals
import unittest
import logging
import sys

import numpy as np
from cvxopt import solvers
from mock import mock
from mock.mock import MagicMock
from nose.tools.nontrivial import nottest


import average_fit

import clp_R_alpha_UCM
import lp_solver
import utils

solvers.options['show_progress'] = False
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

import time

current_milli_time = lambda: int(round(time.time() * 1000))


class TestClpRAlphaUcm(unittest.TestCase):
    # @nottest
    def test_find_violated_constraints(self):
        y = np.array([2.5, 10., 20.])
        z = np.array([0.5, 0.5, 3.0])
        alpha = np.arange(3)
        targets = np.array([2, 1, 0])

        coeff, name = clp_R_alpha_UCM.find_violated_constraints(y, z, targets, alpha, 2)

        self.assertEquals(name, [('C', 0, '1,0,1')])

        coeff = np.ravel(coeff)
        self.assertListEqual(coeff.tolist(), [-1., 0., 0., 1., 0., 1.])
        # coeff_ = np.array(coeff)
        user = coeff[:3].nonzero()[0][0]
        subset = coeff[3:].nonzero()[0]
        self.assertLessEqual(alpha[subset].sum(), targets[user])  # weights
        self.assertGreater(z[subset].sum(), y[user])  # values

    # @mock.patch('clp_R_alpha_UCM.find_violated_constraints')
    # @mock.patch('clp_R_alpha_UCM.lp_solver')
    # def test_lp_solve_by_gaps(self, mock_lp_solver, mock_find_violated_constraints):
    #
    #
    #     mock_HomogenicLpSolver1 = MagicMock()
    #     mock_HomogenicLpSolver2 = MagicMock()
    #
    #     mock_lp_solver.HomogenicLpSolver.side_effect = [mock_HomogenicLpSolver1, mock_HomogenicLpSolver2]
    #
    #     mock_HomogenicLpSolver1.solve.return_value = None
    #     mock_HomogenicLpSolver1.x = MagicMock()
    #
    #     mock_HomogenicLpSolver2.solve.return_value = None
    #     mock_HomogenicLpSolver1.x = MagicMock()
    #
    #     mock_find_violated_constraints.side_effect = [([],[]),([],[])]
    #
    #     clp_R_alpha_UCM.lp_solve_by_gaps(2, np.arange(2), 2, np.array([0,0]), mode='per_cand')
    #
    #     mock_lp_solver.HomogenicLpSolver.assert_has_calls([([1.,1.,-2.-2.]),2])
    #     mock_find_violated_constraints.assert_has_calls([1,2])



    # @nottest
    def test_full(self):
        initial_sigmas = np.array([5, 6, 6, 6, 7], dtype=np.int32)
        k = 2
        m = len(initial_sigmas)

        alpha = utils.borda(m)  # borda

        assert isinstance(initial_sigmas, np.ndarray)
        gaps = utils.sigmas_to_gaps(initial_sigmas, np.max(initial_sigmas))
        logger.info('gaps={}'.format(gaps))

        init_gaps = gaps
        t1 = current_milli_time()
        fractional_makespan, clp_res = clp_R_alpha_UCM.find_strategy(initial_sigmas, alpha, k, mode='per_cand')
        t2 = current_milli_time()

        logger.warning('Took time {}ms, fractional_makespan={}'.format(t2 - t1, fractional_makespan))

        # fractional_makespan = utils.makespan(initial_sigmas, frac_res, alpha=alpha)
        clp_makespan = utils.makespan(initial_sigmas, clp_res, alpha=alpha)
        gaps = utils.sigmas_to_gaps(initial_sigmas, clp_makespan)
        af_res = average_fit.solve_one_gaps(k, gaps, verbose=False)
        af_makespan = utils.makespan(initial_sigmas, af_res, alpha=alpha)
        logger.info(
            'k={} m={} frac={} CLP={} AF={}'.format(k, m, fractional_makespan, clp_makespan,
                                                    af_makespan))

        # self.assertListEqual(clp_res.tolist(),
        #                      [[0, 0, 1, 1, 0],
        #                       [0, 1, 0, 1, 0],
        #                       [1, 0, 0, 0, 1],
        #                       [1, 0, 0, 0, 1],
        #                       [0, 1, 1, 0, 0]]
        #                      )
        self.assertAlmostEqual(fractional_makespan, 10.0, delta=1e-3)
        self.assertEquals(clp_makespan, 10)
        self.assertEquals(af_makespan, 12)

    # @nottest
    def test_full2(self):
        initial_sigmas = np.array([10, 12, 12, 12, 14], dtype=np.int32)
        alpha = np.arange(5) * 2  # kind-of-Borda

        exa = utils.makespan(initial_sigmas, np.array(
            [[0, 0, 1, 1, 0],
             [0, 1, 0, 1, 0],
             [1, 0, 0, 0, 1],
             [1, 0, 0, 0, 1],
             [0, 1, 1, 0, 0]]), alpha)

        k = 2
        m = len(initial_sigmas)

        assert isinstance(initial_sigmas, np.ndarray)
        gaps = utils.sigmas_to_gaps(initial_sigmas, np.max(initial_sigmas))
        logger.info('gaps={}'.format(gaps))

        init_gaps = gaps
        t1 = current_milli_time()
        fractional_makespan, clp_res = clp_R_alpha_UCM.find_strategy(initial_sigmas, alpha, k, mode='per_cand')
        t2 = current_milli_time()

        logger.warning('Took time {}ms, fractional_makespan={}'.format(t2 - t1, fractional_makespan))

        # fractional_makespan = utils.fractional_makespan(initial_sigmas, frac_res, alpha=alpha)
        clp_makespan = utils.makespan(initial_sigmas, clp_res, alpha=alpha)

        logger.info(
            'k={} m={} frac={} CLP={}'.format(k, m, fractional_makespan, clp_makespan))

        # self.assertListEqual(clp_res.tolist(),
        #                      [[0, 0, 1, 1, 0],
        #                       [0, 1, 0, 1, 0],
        #                       [1, 0, 0, 0, 1],
        #                       [1, 0, 0, 0, 1],
        #                       [0, 1, 1, 0, 0]]
        #                      )
        self.assertAlmostEqual(fractional_makespan, 20.0, delta=1e-3)
        self.assertEquals(clp_makespan, 20)

    # @nottest
    def test_draw_interim_configs(self):
        input = [
            [(u'config_A1', 0.5), (u'config_A2', 0.5)],
            [(u'config_B1', 0.9), (u'config_B2', 0.1)]]

        output = clp_R_alpha_UCM.draw_interim_configs(input)
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
        alpha = utils.borda(m)  # borda

        res_config_mat = clp_R_alpha_UCM.fix_rounding_result(config_mat, alpha, k, initial_sigmas)

        self.assertListEqual(np.ravel(res_config_mat).tolist(), [1, 1, 0, 0, 1, 1, 1, 0, 1])


if __name__ == '__main__':
    unittest.main()
