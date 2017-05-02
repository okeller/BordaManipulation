from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import sys
# from future import standard_library
from builtins import (range,
                      zip, int)

import numpy as np
from cvxopt import matrix
from cvxopt.modeling import variable, sum, op, dot

import utils

# PY_OLD = sys.version_info[0] < 3

logger = logging.getLogger(__name__)



def backtrack(taken, weight_bound, multiplicity, weights):
    """
    Backtracks the table of `exclusive_knapsack` to produce the resulting multiset

    Args:
        taken (List[List[int]]): the table confirming whether an item was taken or not
        weight_bound (int):
        multiplicity (int):
        weights (List[int]):

    Returns:
        List[int]: A milti-subset as list

    """

    res = []
    w = weight_bound
    for ell in range(multiplicity, 0, -1):
        item = taken[w][ell]
        res.append(item)
        w -= weights[item]
    res.reverse()
    return res


def k_multiset_knapsack(values, weights, k, target_value, weight_bound, tol=0.001):
    """
    Solves an instance of the k-multiset knapsack
    Args:
        values (List[float]): a vector of item values
        weights (List[int]): a vector of item weights
        k (int): the size of the resulting multiset
        target_value (float): a lower bound on resulting subset value
        weight_bound (numpy.int32): an upper bound on resulting subset weight
        tol (float): a tolerance parameter

    Returns:
        List[int]: A multiset of items of size `k` represented as an list containing the histogram of score types.

    """

    assert len(values) == len(weights)

    num_types = len(values)

    mat = np.zeros((weight_bound + 1, k + 1), dtype=np.float32)
    last_taken = [[-1] * (k + 1) for w in range(weight_bound + 1)]
    # mat[:, 0] = 0  # init table - we always don't want to take the dummy item
    for w in range(weight_bound + 1):
        for ell in range(k + 1):
            if ell == 0:
                mat[w, ell] = 0
            else:
                mat[
                    w, ell] = -sys.maxsize  # since we still don't know better, the default is that current option is invalid
                last_taken[w][ell] = -1
                for item in range(0, num_types):
                    if w - weights[item] >= 0:  # if item is relevant, i.e., can be at all taken
                        if values[item] + mat[w - weights[item], ell - 1] > mat[w, ell]:  # if it given better value
                            mat[w, ell] = values[item] + mat[w - weights[item], ell - 1]
                            last_taken[w][ell] = item

    best_val = mat[weight_bound, k]

    if best_val > target_value + tol:
        logger.debug('best val: {} target: {}'.format(best_val, target_value))
        subset = backtrack(last_taken, weight_bound, k, weights)
        # assert best_val - tol < sum(values[item] for item in subset) < best_val + tol
        assert isinstance(subset, list)
        return subset
    else:
        return None


def aslist(mat):
    """

    Args:
        mat:

    Returns:

    """
    assert isinstance(mat, matrix)
    return [mat[i] for i in range(len(mat))]


def find_violated_constraints(y, z, targets, k, mode='one'):
    """
    This is the separation oracle. Given (y,z) representing a prosoped solution to the dual of the C-LP, it find
    a violated constrait or returns `None`.
    Args:
        y (cvxopt.modeling.variable): the y vector
        z (cvxopt.modeling.variable): the z vector
        targets (List[numpy.int32]): list of bounds for each candidate i representing T-sigma(i)
        k (int): the number of manipulators

    Returns:
        cvxopt.modeling.constraint: the violated constraint
    """
    assert mode in ['one', 'per_cand', 'per_cand_prune']
    if y.value is None or z.value is None:
        return None
    y_vals, z_vals = aslist(y.value), aslist(z.value)
    num_item_types = len(z_vals)

    natural_bound = (len(z_vals) - 1) * k

    constraints = []
    for i in range(len(targets)):
        # a violated constraint is such that y[i]>sum_of_subset_of(z_j's) while sum_of_subset_of(votes)<targets[i]
        subset = k_multiset_knapsack(values=z_vals, weights=range(num_item_types), k=k,
                                     target_value=y_vals[i],
                                     weight_bound=min(targets[i], natural_bound))


        if subset:
            vote_vector, _ = np.histogram(subset, bins=range(num_item_types + 1))
            assert len(vote_vector) == num_item_types
            vote_vector_rep = ','.join([str(v) for v in vote_vector])

            vote_vector_mat = matrix(np.array(vote_vector, dtype=np.float64), tc=str('d'))
            c = dot(vote_vector_mat, z) <= y[i]
            c.name = str('{}\t{}').format(i, vote_vector_rep)
            # c.name = (i, subset)
            constraints.append(c)  # the constraint itself
            if mode == 'one':
                break
            else:
                pass
    return constraints




def get_frac_config_mat(x_i_C2val):
    fractional_config_mat = np.zeros((len(x_i_C2val), len(x_i_C2val)), dtype=np.float32)

    for cand, weighted_configs in enumerate(x_i_C2val):
        for con_str, w in weighted_configs:
            con_array = np.array([int(v) for v in con_str.split(',')], dtype=np.int32)

            fractional_config_mat[cand, :] += w * con_array

    return fractional_config_mat


def draw_interim_configs(x_i_C2val):
    res = []
    for weighted_configs in x_i_C2val:
        weights = np.array([v for k, v in weighted_configs], dtype=np.float32)
        weights /= np.sum(weights)  # re-normalize in order to fix rounding issues

        configs = [k for k, v in weighted_configs]
        try:
            config = np.random.choice(configs, p=weights)
        except:
            logger.error('weights={}'.format(sum(weights)))
            raise ValueError('weights')
        res.append(config)
    return res


def lp_solve(m, k, sigmas, target, mode='one'):
    """
    Solved a C-LP instance
    Args:
        m (int): number of candidates
        k (int): number of manipulators
        sigmas (numpy.ndarray): the initial (i.e. non-manipulator) score for each candidate
        target (numpy.int32): the proposed bound on the final score of each candidate

    Returns:

    """
    gaps = [target - sigma for sigma in sigmas]
    return lp_solve_by_gaps(m, k, gaps, mode=mode)


def lp_solve_by_gaps(m, k, gaps, mode='one', tol=0.001):
    """

    Args:
        m (int):
        k (int):
        gaps (List[numpy.int32]):

    Returns:

    """
    assert mode in ['one', 'per_cand', 'per_cand_prune']
    y = variable(m, name=str('y'))
    z = variable(m, name=str('z'))
    obj_func = sum(y) - k * sum(z)
    constraints = [y >= 0, z >= 0]
    prog = op(obj_func, constraints)
    prog.solve()

    new_constraints = find_violated_constraints(y, z, gaps, k, mode=mode)
    while len(new_constraints) > 0:
        # logger.info('(y,z)={}{} status={}'.format(aslist(y.value), aslist(z.value), prog.status))



        logger.info('Adding {} constraints'.format(len(new_constraints)))
        constraints += new_constraints
        prog = op(obj_func, constraints)
        prog.solve(solver='default')

        if mode == 'per_cand_prune':
            non_pruned_constraints = []
            for c in constraints:
                if c.name and c.multiplier.value is not None and c.multiplier.value[0] > tol:
                    non_pruned_constraints.append(c)
                else:
                    non_pruned_constraints.append(c)
            logger.info('pruned {} constraints'.format(len(constraints) - len(non_pruned_constraints)))
            constraints = non_pruned_constraints

        # logger.warn('in status {}'.format(prog.status))

        new_constraints = find_violated_constraints(y, z, gaps, k, mode=mode)

    # logger.info('(y,z)={}{} status={}'.format(aslist(y.value), aslist(z.value), prog.status))
    logger.info('reached obj val: {}'.format(prog.objective.value()))
    logger.info('{} constraints were added'.format(len(constraints) - 2))

    if mode == 'per_cand_prune':
        non_pruned_constraints = []
        for c in constraints:
            if c.name and c.multiplier.value is not None and c.multiplier.value[0] > tol:
                non_pruned_constraints.append(c)
            else:
                non_pruned_constraints.append(c)
        logger.info('pruned {} constraints'.format(len(constraints) - len(non_pruned_constraints)))
        constraints = non_pruned_constraints

    if 'infeasible' in prog.status:
        logger.warn(prog.status)
        return prog.status
    logger.debug('{} {}'.format(y.value, z.value))
    x_i_C2val = [[] for _ in range(m)]
    for c in constraints:
        if c.name:
            i_str, subset_str = c.name.split()
            x_i_C2val[int(i_str)].append((subset_str, c.multiplier.value[0]))
    return x_i_C2val




def fix_rounding_result(config_mat, k, initial_sigmas):
    """

    :type initial_sigmas: numpy.ndarray
    :param initial_sigmas:
    :type config_mat: numpy.ndarray
    """
    m = len(initial_sigmas)
    awarded = utils.calculate_awarded(config_mat)
    awarded += initial_sigmas

    tuples = list(enumerate(awarded))
    tuples.sort(key=lambda t: t[1])
    candidate_order, _ = zip(*tuples)

    events = []
    for score in range(m):
        for cand in candidate_order:
            multiplicity = config_mat[cand, score]
            for i in range(multiplicity):
                events.append((score, cand))

    fixed_events = []
    for i, ev in enumerate(events):
        fixed_events.append((i // k, ev[1]))

    res = np.zeros((m, m), dtype=np.int32)
    for ev in fixed_events:
        score, cand = ev
        res[cand, score] += 1
    return res


def find_strategy_by_gaps(initial_gaps, k):
    initial_sigmas = -initial_gaps + np.max(initial_gaps)
    return find_strategy(initial_sigmas, k)


def find_strategy(initial_sigmas, k, mode='one'):
    """

    Args:
        initial_sigmas:
        k:

    Returns:

    """
    m = len(initial_sigmas)
    target = np.max(initial_sigmas)
    if target == 0:
        target = 1

    # find target by one-sided binary search
    logger.warning('target={}'.format(target))
    x_i_C2val = lp_solve(m, k, initial_sigmas, target, mode=mode)
    last_target = target
    while x_i_C2val == 'dual infeasible':

        last_target = target
        target *= 2
        logger.warning('target={}'.format(target))
        x_i_C2val = lp_solve(m, k, initial_sigmas, target, mode=mode)

    # then find target by two-sided binary search
    lo, hi = last_target, target

    last_dual_feasible_solution = x_i_C2val
    # binary search
    while lo < hi:

        mid = (lo + hi) // 2
        logger.warning('mid={}'.format(mid))
        x_i_C2val = lp_solve(m, k, initial_sigmas, mid, mode=mode)
        if x_i_C2val == 'dual infeasible':
            lo = mid + 1
        else:
            last_dual_feasible_solution = x_i_C2val
            hi = mid

    # target = lo
    # x_i_C2val = lp_solve(m, k, sigmas, target)

    x_i_C2val = last_dual_feasible_solution

    assert x_i_C2val != 'dual infeasible'

    logger.debug('bin search ended in {}'.format(hi))

    for i, weighted_configs in enumerate(x_i_C2val):
        logger.debug('{}: {}'.format(i, weighted_configs))

    frac_config_mat = get_frac_config_mat(x_i_C2val)

    frac_makespan = utils.makespan(initial_sigmas, frac_config_mat)
    logger.info('fractional makespan is {}'.format(frac_makespan))

    sum_votes = np.zeros(m, dtype=np.float32)

    # strs to arrays
    for i, weighted_configs in enumerate(x_i_C2val):
        for config, weight in weighted_configs:
            values = [int(val) for val in config.split(',')]
            vote_vector = np.array(values, dtype=np.int32)
            sum_votes += vote_vector * weight

    # print sum_votes

    logger.debug("start fixing loops")

    result_range = set()
    best_makespan = sys.maxsize
    best_config_mat = None
    for i in range(1000):

        res = draw_interim_configs(x_i_C2val)

        # turn the configs to lists of ints
        res = [[int(v) for v in con.split(',')] for con in res]

        frac_config_mat = np.array(res, dtype=np.int32)
        logger.debug('interim configs:')
        logger.debug(frac_config_mat)
        histogram = np.sum(frac_config_mat, axis=0)
        logger.debug('histogram={}'.format(histogram))

        cur_config_mat = fix_rounding_result(frac_config_mat, k, initial_sigmas)
        cur_makespan = utils.makespan(initial_sigmas, cur_config_mat)
        result_range.add(cur_makespan)
        if cur_makespan < best_makespan:
            best_makespan = cur_makespan
            best_config_mat = cur_config_mat
    logger.debug("end fixing loops result range {}".format(result_range))

    return frac_config_mat, best_config_mat
