from __future__ import absolute_import, division, print_function, unicode_literals
import logging

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# pre-calculate for fast bounded Zipf:
# x_zipf = np.arange(1, 1024 + 1)
ZIPF_a = 2.15
# weights = x_zipf ** (-ZIPF_a)
zeta_dist = stats.zipf(ZIPF_a)


def borda(m):
    return np.arange(m)


def draw_zipf(num_voters, rand=None):
    if rand is None:
        rand = np.random.RandomState()

    return zeta_dist.rvs(size=num_voters, random_state=rand)
    # return np.random.zipf(a, size=num_voters)

def draw_zipf_weights(owners, W=None, return_len_k=False, rand=None):
    """
    Draws owners for W 'stocks' (out of k owners), such that in expectation, the amount of stocks owned by person i is proportional to
    i^(-a) for a=1.1 (Zipf law over the bounded set {0,1,...,k-1} )

    Args:
        W (int): The number units to distrobute between the owners
        owners (int): the number of owners
        rand (Optional[numpy.random.mtrand.RandomState]):

    Returns:

    """
    # return rand.randint(1, 3, k, dtype=int) if rand else np.random.randint(1, 3, k, dtype=int)
    if rand is None:
        rand = np.random.RandomState()
    if W is None:
        W = int(owners * np.ceil(np.log(owners)))

    _weights = weights[:owners].copy()
    _weights /= _weights.sum()  # normalize

    bounded_zipf = stats.rv_discrete(name='bounded_zipf', values=(np.arange(owners), _weights))

    sample = bounded_zipf.rvs(size=W, random_state=rand)  # drawn W numbers from the set {0,...,k-1}

    res = np.bincount(sample, minlength=owners)

    res = np.sort(res)[::-1]

    if not return_len_k:
        # now discard owner that recieved 0 stocks:
        indices_of_zeros = (res == 0).nonzero()[0]
        res = res[:indices_of_zeros[0]] if len(indices_of_zeros) else res

    return res


def draw_uniform(m, n, rand=None):
    """

    Args:
        m (int)
        n (int:
        rand (Optional [Union[ numpy.random.mtrand.RandomState, None]]):

    Returns:

    """
    rankings = draw_uniform_rankings(m, n, rand)
    return rankings_to_initial_sigmas(rankings)


def draw_uniform_weighted(m, n, weights, rand=None):
    """

    Args:
        m (int)
        n (int:
        rand (Optional [Union[ numpy.random.mtrand.RandomState, None]]):

    Returns:

    """
    if weights.shape != (n,):
        raise ValueError("dimension mismatch")
    rankings = draw_uniform_rankings(m, n, rand)
    return rankings_to_initial_sigmas_weighted(rankings, weights)


def draw_polya_eggenberger(m, n, rand=None):
    """

    Args:
        m (int)
        n (int:
        rand (Optional [Union[ numpy.random.mtrand.RandomState, None]]):

    Returns:

    """
    rankings = draw_polya_eggenberger_rankings(m, n, rand)
    return rankings_to_initial_sigmas(rankings)


def draw_uniform_rankings(m, n, rand=None):
    """

    Args:
       m (int)
       n (int:
       rand (numpy.random.mtrand.RandomState):

    Returns:

    """
    res = []
    for i in range(n):
        perm = rand.permutation(m) if rand else np.random.permutation(m)
        res.append(perm)
    return np.array(res, dtype=np.int)


def draw_polya_eggenberger_rankings(m, n, rand=None):
    """

    Args:
       m (int)
       n (int:
       rand (numpy.random.mtrand.RandomState):

    Returns:

    """

    if rand is None:
        rand = np.random.RandomState()
    repetitions = 0
    res = []
    meta_dist = [None]
    for i in range(n):
        perm = rand.choice(meta_dist)

        if perm is None:
            perm = rand.permutation(m)
        else:
            repetitions += 1
        meta_dist.append(perm)
        res.append(perm)

    logger.debug('Polya-Eggenberger repeated selections {} times'.format(repetitions))
    return np.array(res, dtype=np.int)


def remove_candidtate(rankings, cand):
    """

    :type cand: int
    :type rankings: nupmy.ndarray
    """
    res = np.zeros((rankings.shape[0], rankings.shape[1] - 1), dtype=int)


def rankings_to_initial_sigmas(rankings):
    """

    :type rankings: numpy.ndarray
    """
    initial_sigmas = np.zeros(rankings.shape[1], dtype=np.int)
    for i in range(rankings.shape[0]):
        perm = rankings[i]
        for j, perm_j in enumerate(perm):
            initial_sigmas[perm_j] += j
    return initial_sigmas


def rankings_to_initial_sigmas_weighted(rankings, weights):
    """

    :type rankings: numpy.ndarray
    """
    if rankings.shape[0] != weights.shape[0]:
        raise ValueError("dimension mismatch")
    initial_sigmas = np.zeros(rankings.shape[1], dtype=np.int)
    for i in range(rankings.shape[0]):
        perm = rankings[i]
        for j, perm_j in enumerate(perm):
            initial_sigmas[perm_j] += j * weights[i]
    return initial_sigmas


def calculate_awarded(config_mat, initial_sigmas=None, alpha=None):
    m = config_mat.shape[0]

    if alpha is None:
        alpha = borda(m)

    awarded = np.zeros(m, dtype=float)

    if initial_sigmas is not None:
        awarded += initial_sigmas

    for j in range(m):
        awarded += config_mat[:, j] * alpha[j]
    return awarded


def makespan(initial_sigmas, config_mat, alpha=None):
    return np.max(calculate_awarded(config_mat, initial_sigmas, alpha))


def weighted_makespan(config_mat, alpha, weights, initial_sigmas):
    return np.max(weighted_calculate_awarded(config_mat, alpha, weights, initial_sigmas))


def weighted_calculate_awarded(config_mat, alpha, weights, initial_sigmas=None):
    assert config_mat.shape[0] == len(alpha)
    scores = np.zeros(config_mat.shape[0], dtype=float)

    if initial_sigmas is not None:
        scores += initial_sigmas

    for i in range(config_mat.shape[0]):
        for ell in range(config_mat.shape[1]):
            scores[i] += weights[ell] * alpha[config_mat[i, ell]]

    return scores


def fractional_makespan(initial_sigmas, x_i_C2val, alpha=None):
    m = len(initial_sigmas)
    if alpha is None:
        alpha = borda(m)  # Borda
    assert len(initial_sigmas) == len(alpha)

    scores = np.zeros(len(initial_sigmas), dtype=float)

    scores += initial_sigmas

    for cand, weighted_configs in enumerate(x_i_C2val):
        for con_str, prob in weighted_configs:
            config = np.array([int(v) for v in con_str.split(',')], dtype=np.int32)

            scores[cand] += np.dot(config, alpha) * prob

    return np.max(scores)


def weighted_fractional_makespan(initial_sigmas, x_i_C2val, alpha, weights):
    assert len(initial_sigmas) == len(alpha)
    scores = np.zeros(len(initial_sigmas), dtype=float)

    scores += initial_sigmas

    for cand, weighted_configs in enumerate(x_i_C2val):
        for con_str, prob in weighted_configs:
            sequence = np.array([int(v) for v in con_str.split(',')], dtype=np.int32)

            seq_val = np.sum([alpha[sequence[ell]] * weights[ell] for ell in range(len(sequence))])

            scores[cand] += prob * seq_val

    return np.max(scores)


def sigmas_to_gaps(initial_sigmas, target):
    return target - initial_sigmas


def gaps_to_sigmas(gaps):
    return np.max(gaps) - gaps


def pretty_print(gaps, config_mat):
    """

    :type config_mat: numpy.ndarray
    """
    m = config_mat.shape[0]
    res = [[] for _ in range(m)]
    for cand, score in np.transpose(np.nonzero(config_mat)):
        count = config_mat[cand, score]
        for _ in range(count):
            res[cand].append(score)
    a = np.asarray(res)
    s = bcolors.OKBLUE + '\t'.join([str(g) for g in gaps]) + bcolors.ENDC + '\n'
    a = a.transpose()
    for i in range(a.shape[0]):
        s += '\t'.join([str(v) for v in a[i, :]]) + '\n'
    return s
