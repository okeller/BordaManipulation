from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np
from cvxopt import solvers

import average_fit
import clp
import utils

solvers.options['show_progress'] = False
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    initial_sigmas = np.array([5, 6, 6, 6, 7], dtype=np.int32)
    k = 2
    trial = 1
    n = 4
    m = 5

    assert isinstance(initial_sigmas, np.ndarray)
    gaps = utils.sigmas_to_gaps(initial_sigmas, np.max(initial_sigmas))
    logger.info('gaps={}'.format(gaps))

    init_gaps = gaps
    frac_res, clp_res = clp.find_strategy(initial_sigmas, k)
    fractional_makespan = utils.makespan(initial_sigmas, frac_res)
    clp_makespan = utils.makespan(initial_sigmas, clp_res
                                  )
    gaps = utils.sigmas_to_gaps(initial_sigmas, clp_makespan)
    af_res = average_fit.solve_one_gaps(k, gaps, verbose=False)
    af_makespan = utils.makespan(initial_sigmas, af_res)
    logger.info(
        'n={} k={} m={} trial={} frac={} CLP={} AF={}'.format(n, k, m, trial, fractional_makespan, clp_makespan,
                                                              af_makespan))
    # result_to_append = [n, k, m, trial, initial_sigmas, fractional_makespan, clp_makespan, af_makespan]
    # print(result_to_append)
