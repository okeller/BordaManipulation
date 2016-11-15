from __future__ import print_function

import logging

import numpy as np
import pandas as pd
from cvxopt import solvers
from sklearn.externals.joblib import delayed, Parallel

import average_fit
import clp
import utils

solvers.options['show_progress'] = False
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# k = 4
# n = 8
#
trials = 8
# start = 4
# end = 10



def run(n,k,m, trial, initial_sigmas):


    assert isinstance(initial_sigmas, np.ndarray)
    gaps = utils.sigmas_to_gaps(initial_sigmas, np.max(initial_sigmas))
    logger.info('gaps={}'.format(gaps))
    # af_makespan, clp_makespan = 0, 1
    init_gaps = gaps
    frac_res, clp_res = clp.find_strategy(initial_sigmas, k)
    fractional_makespan = utils.makespan(initial_sigmas, frac_res)
    clp_makespan = utils.makespan(initial_sigmas, clp_res
                                  )
    gaps = utils.sigmas_to_gaps(initial_sigmas, clp_makespan)
    af_res = average_fit.solve_one_gaps(k, gaps, verbose=False)
    af_makespan = utils.makespan(initial_sigmas, af_res)
    logger.info(
        'n={} k={} m={} trial={} frac={} CLP={} AF={}'.format(n, k, m, trial, fractional_makespan, clp_makespan, af_makespan))
    result_to_append = [n, k, m, trial, initial_sigmas, fractional_makespan, clp_makespan, af_makespan]
    return result_to_append



if __name__ == '__main__':


    for m in range(4, 65, 10):
        logm = int(np.math.ceil(np.math.log(m, 2)))
        max_logk = logm // 2
        for logn in range (2, max_logk+2):


            # m = 2 ** logm
            n = 2 ** logn
            k = n // 2

            logger.info('m={} n={}'.format(m, n))

            res = Parallel(n_jobs=1)\
                (delayed(run)(n,k,m, trial, utils.draw_uniform(m, n))
                 for trial in range(trials)
                 )

            # for m in trange(start, end):
            #     for trial in range(trials):
            #
            #
            #         result_to_append = run(n,k,m,trial)
            #         res.append(result_to_append)

            df = pd.DataFrame(data=res, columns=['n', 'k', 'm', 'trial', 'initial_sigmas', 'fractional_makespan', 'clp_makespan', 'af_makespan'])
            df.to_csv('output5/results-n{}-k{}-m{}-t{}.csv'.format(n, k, m, trials), index=False)


# means = np.mean(results,axis=1)
# stddevs = np.std(results,axis=1)
# import matplotlib.pyplot as plt
# plt.errorbar(range(start,end), means, yerr=stddevs)
# plt.show()
