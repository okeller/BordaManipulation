
import numpy as np


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def draw_uniform(m,n, rand=None):
    rankings = draw_uniform_rankings(m, n, rand)
    return rankings_to_initial_sigmas(rankings)

def draw_uniform_rankings(m,n, rand = None):
    """

    :type rand: np.random.RandomState
    """
    res = []
    for i in range(n):
        perm = rand.permutation(m) if rand else np.random.permutation(m)
        res.append(perm)
    return np.array(res, dtype=int)


def remove_candidtate(rankings, cand):
    """

    :type cand: int
    :type rankings: nupmy.ndarray
    """
    res = np.zeros((rankings.shape[0], rankings.shape[1]-1))


def rankings_to_initial_sigmas(rankings):
    """

    :type rankings: numpy.ndarray
    """
    initial_sigmas = np.zeros(rankings.shape[1], dtype=int)
    for i in range(rankings.shape[0]):
        perm = rankings[i]
        for j, perm_j in enumerate(perm):
            initial_sigmas[perm_j] += j
    return initial_sigmas


def calculate_awarded(config_mat):
    m = config_mat.shape[0]
    awarded = np.zeros(m)
    for j in range(m):
        awarded += config_mat[:, j] * j
    return awarded


def makespan(initial_sigmas, config_mat):
    return np.max(initial_sigmas + calculate_awarded(config_mat))


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
