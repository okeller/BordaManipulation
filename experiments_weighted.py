# from __future__ import absolute_import, division, print_function, unicode_literals

import sys

import logging
import os

from io import StringIO, BytesIO
import numpy as np
import pandas as pd
from cvxopt import solvers
from sklearn.externals.joblib import delayed, Parallel

import average_fit
import clp_R_alpha_WCM
import clp_general
import reverse_algorithm
import utils

solvers.options['show_progress'] = False
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# k = 4
# n = 8
#
trials = 20
# start = 4
# end = 10


folder = 'borda_weighted'
if folder and not os.path.exists(folder):
    os.makedirs(folder)


def run(n, alpha, weights, m, trial, initial_sigmas):
    assert isinstance(initial_sigmas, np.ndarray)
    gaps = utils.sigmas_to_gaps(initial_sigmas, np.max(initial_sigmas))
    logger.info('gaps={}'.format(gaps))
    # reverse_makespan, clp_makespan = 0, 1
    init_gaps = gaps
    fractional_makespan, clp_res = clp_R_alpha_WCM.find_strategy(initial_sigmas, alpha, weights, mode='per_cand')
    clp_makespan = utils.weighted_makespan(clp_res, alpha, weights, initial_sigmas )

    reverse_result = reverse_algorithm.find_strategy(initial_sigmas, alpha, weights)
    reverse_makespan = utils.weighted_makespan(reverse_result, alpha, weights, initial_sigmas)
    logger.info(
        'n={} k={} m={} trial={} frac={} CLP={} AF={}'.format(n, len(weights), m, trial, fractional_makespan,
                                                              clp_makespan,
                                                              reverse_makespan))
    result_to_append = [n, len(weights), m, trial, initial_sigmas, weights, fractional_makespan, clp_makespan,
                        reverse_makespan]

    df = pd.DataFrame(data=[result_to_append],
                      columns=['n', 'k', 'm', 'trial', 'initial_sigmas', 'weights', 'fractional_makespan',
                               'clp_makespan',
                               'reverse_makespan'])

    filename = 'results-m{}-k{}-n{}-t{}-tt{}.csv'.format(m, len(weights), n, trials, trial)

    if folder is None:  # s3
        import boto3
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, encoding='utf-8')
        s3_resource = boto3.resource('s3')
        s3_resource.Object('borda_weighted', filename).put(Body=csv_buffer.getvalue())
    else:
        abs_path = os.path.join(folder, filename)
        df.to_csv(abs_path, index=False)

    return result_to_append


if __name__ == '__main__':
    experiments = (delayed(run)(k * 2, utils.borda(m), utils.draw_weights(k), m, trial, utils.draw_uniform(m, k * 2))
                   for m in
                   range(4, 65, 10) for k in [2, 4, 6] for
                   trial in range(trials))

    res = Parallel(n_jobs=-1)(experiments)
