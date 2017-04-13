from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import sys

import numpy as np
from cvxopt import solvers

import average_fit
import clp
import clp_general
import lp_solver
import utils

solvers.options['show_progress'] = False
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


import time

current_milli_time = lambda: int(round(time.time() * 1000))

if __name__ == '__main__':
    initial_sigmas = np.array([5, 6, 6, 6, 7], dtype=np.int32)
    k = 2
    m = len(initial_sigmas)

    assert isinstance(initial_sigmas, np.ndarray)
    gaps = utils.sigmas_to_gaps(initial_sigmas, np.max(initial_sigmas))
    logger.info('gaps={}'.format(gaps))

    init_gaps = gaps
    t1 = current_milli_time()
    frac_res, clp_res = clp_general.find_strategy(initial_sigmas, k, mode='per_cand')
    t2 = current_milli_time()

    logger.warning('Took time {}ms, frac_res={}'.format(t2 - t1, frac_res))

    fractional_makespan = utils.makespan(initial_sigmas, frac_res)
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
