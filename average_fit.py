from __future__ import print_function
import logging
import numpy as np
import utils


def solve_one(k, initial_sigmas, target, verbose=False):
    """

    """

    init_gaps = [target - sigma for sigma in initial_sigmas]
    return solve_one_gaps(k, init_gaps, verbose)


def solve_one_gaps(k, init_gaps, verbose=False):
    m = len(init_gaps)

    score2mult = np.array([k] * m)

    init_gaps = np.array(init_gaps, dtype=float)

    gaps = init_gaps.copy()

    left = np.array([k] * m)
    potential = gaps / left

    num_scores = m * k
    config_mat = np.zeros((m, m), dtype=int)

    if verbose:
        strs = ['{}/{}={}'.format(g, l, p) for g, l, p in zip(gaps, left, potential)]
        print('\t\t'.join(strs))

    while num_scores > 0:
        # choose cand with highest potential
        cand = np.argmax(potential)

        # choose which score to give
        does_fit = np.arange(m) <= gaps[cand]
        still_left = score2mult > 0
        does_fit_and_still_left = np.logical_and(does_fit, still_left)
        score_to_give = None
        if np.any(does_fit_and_still_left):
            score_to_give = does_fit_and_still_left.nonzero()[0][-1]
        else:
            score_to_give = still_left.nonzero()[0][-1]

        # does_fit = np.array([ j <= gaps[cand] for j in range(m)])
        # still_left = np.array([score2mult[j] > 0 for j in range(m)])

        # if we have still score of this type to give, and it doesn't cause the candidate to overflow
        score2mult[score_to_give] -= 1
        config_mat[cand, score_to_give] += 1
        gaps[cand] -= score_to_give
        left[cand] -= 1

        # re-calculate potential
        if left[cand] > 0:
            potential[cand] = gaps[cand] / left[cand]
        else:
            potential[cand] = -1

        num_scores -= 1

        if verbose:
            strs = ['{}/{}={}'.format(g, l, p) for g, l, p in zip(gaps, left, potential)]
            strs[cand] = utils.bcolors.OKBLUE + str(score_to_give) + '->' + utils.bcolors.WARNING + strs[
                cand] + utils.bcolors.ENDC
            print('\t\t'.join(strs))
    logging.debug('final configs:')
    logging.debug(config_mat)

    return config_mat

    # awarded = utils.calculate_awarded(config_mat)
    #
    # if np.all(awarded <= init_gaps):
    #     return config_mat
    # else:
    #     return None


def find_strategy(initial_sigmas, k):
    # m = len(initial_sigmas)
    target = max(initial_sigmas)
    assert isinstance(target, int)

    # find target by one-sided binary search
    config_mat = solve_one(k, initial_sigmas, target)
    last_target = target

    ms = utils.makespan(initial_sigmas, config_mat)
    while ms > target:
        last_target = target
        target *= 2
        config_mat = solve_one(k, initial_sigmas, target)
        ms = utils.makespan(initial_sigmas, config_mat)

    # then find target by two-sided binary search
    lo, hi = last_target, target

    last_dual_feasible_solution = config_mat
    # binary search
    while lo < hi:
        mid = (lo + hi) // 2
        config_mat = solve_one(k, initial_sigmas, mid)
        ms = utils.makespan(initial_sigmas, config_mat)
        if ms > mid:
            lo = mid + 1
        else:
            last_dual_feasible_solution = config_mat
            hi = mid

    return (lo, last_dual_feasible_solution)
