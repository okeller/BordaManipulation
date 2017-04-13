import time
import logging

from cvxopt import solvers

import clp

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
solvers.options['show_progress'] = False

current_milli_time = lambda: int(round(time.time() * 1000))

# gaps = [227, 187, 121, 168, 272, 175, 168, 226, 214, 214, 200, 167, 238, 176, 167, 198, 102, 245, 176, 133, 214, 209]
        # 151, 185, 117, 236, 153, 172, 133, 115, 156, 141, 132, 158, 144, 148, 197, 134, 138, 161, 93, 172, 163, 172]

gaps = [13, 12, 12, 17, 15, 0, 20, 19, 13, 14, 38, 0, 10, 13]
# gaps = [5,6,6,6,7]
m = len(gaps)
k = 2

t1 = current_milli_time()
clp.lp_solve_by_gaps_general(m, k, gaps, mode='per_cand')
t2 = current_milli_time()
logger.warning('General Took time {}ms'.format(t2 - t1))

logger.error('--------------------------------------------')

t1 = current_milli_time()
clp.lp_solve_by_gaps(m, k, gaps, mode='per_cand')
t2 = current_milli_time()
logger.warning('Took time {}ms'.format(t2 - t1))

