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
import clp_general
import utils

solvers.options['show_progress'] = False
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# k = 4
# n = 8
#
trials = 100
# start = 4
# end = 10


folder = None
if folder and not os.path.exists(folder):
    os.makedirs(folder)


def run(n, k, m, trial, initial_sigmas):
    assert isinstance(initial_sigmas, np.ndarray)
    try:
        gaps = utils.sigmas_to_gaps(initial_sigmas, np.max(initial_sigmas))
        logger.info('gaps={}'.format(gaps))
        # af_makespan, clp_makespan = 0, 1
        init_gaps = gaps
        fractional_makespan, clp_res = clp_general.find_strategy(initial_sigmas, k, mode='per_cand')
        clp_makespan = utils.makespan(initial_sigmas, clp_res
                                      )
        gaps = utils.sigmas_to_gaps(initial_sigmas, clp_makespan)
        af_res = average_fit.solve_one_gaps(k, gaps, verbose=False)
        af_makespan = utils.makespan(initial_sigmas, af_res)
        logger.info(
            'n={} k={} m={} trial={} frac={} CLP={} AF={}'.format(n, k, m, trial, fractional_makespan, clp_makespan,
                                                                  af_makespan))
        result_to_append = [n, k, m, trial, initial_sigmas, fractional_makespan, clp_makespan, af_makespan]

        df = pd.DataFrame(data=[result_to_append],
                          columns=['n', 'k', 'm', 'trial', 'initial_sigmas', 'fractional_makespan', 'clp_makespan',
                                   'af_makespan'])

        filename = 'results-m{}-k{}-n{}-t{}-tt{}.csv'.format(m, k, n, trials, trial)

        if folder is None:  # s3
            import boto3
            csv_buffer = BytesIO()
            df.to_csv(csv_buffer, encoding='utf-8')
            s3_resource = boto3.resource('s3')
            s3_resource.Object('borda-poya', filename).put(Body=csv_buffer.getvalue())
        else:
            abs_path = os.path.join(folder, filename)
            df.to_csv(abs_path, index=False)
        return result_to_append

    except:
        logger.error('Failed on params n={} k={} m={} trial={} sigmas={}'.format(n, k, m, trial, initial_sigmas))
        raise RuntimeError('Failed on params n={} k={} m={} trial={} sigmas={}'.format(n, k, m, trial, initial_sigmas))


if __name__ == '__main__':
    experiments = (delayed(run)(k * 2, k, m, trial, utils.draw_polya_eggenberger(m, k * 2)) for m in
                   range(10, 110, 10) for k in [2, 4, 6, 8] for trial in range(trials))

    res = Parallel(n_jobs=-1)(experiments)
