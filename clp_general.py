from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import sys
# from future import standard_library
from builtins import (range,
                      zip, int)

import numpy as np
from cvxopt import matrix
from cvxopt.modeling import sum

# PY_OLD = sys.version_info[0] < 3
import lp_solver
import utils

logger = logging.getLogger(__name__)

try:
    from cy_knapsacks import k_multiset_knapsack
    logger.info("using Cython knpasacks")
except ImportError:
    logger.info("using vanilla knpasacks")
    from knapsacks import k_multiset_knapsack





# def backtrack(taken, weight_bound, multiplicity, weights):
#     """
#     Backtracks the table of `exclusive_knapsack` to produce the resulting multiset
#
#     Args:
#         taken (List[List[int]]): the table confirming whether an item was taken or not
#         weight_bound (int):
#         multiplicity (int):
#         weights (List[int]):
#
#     Returns:
#         List[int]: A milti-subset as list
#
#     """
#
#     res = []
#     w = weight_bound
#     for ell in range(multiplicity, 0, -1):
#         item = taken[w][ell]
#         res.append(item)
#         w -= weights[item]
#     res.reverse()
#     return res


# def k_multiset_knapsack(values, weights, k, target_value, weight_bound, tol=0.001):
#     """
#     Solves an instance of the k-multiset knapsack
#     Args:
#         values (List[float]): a vector of item values
#         weights (List[int]): a vector of item weights
#         k (int): the size of the resulting multiset
#         target_value (float): a lower bound on resulting subset value
#         weight_bound (numpy.int32): an upper bound on resulting subset weight
#         tol (float): a tolerance parameter
#
#     Returns:
#         List[int]: A multiset of items of size `k` represented as an list containing the histogram of score types.
#
#     """
#
#     assert len(values) == len(weights)
#
#     num_types = len(values)
#
#     mat = np.zeros((weight_bound + 1, k + 1), dtype=np.float32)
#     last_taken = [[-1] * (k + 1) for w in range(weight_bound + 1)]
#     # mat[:, 0] = 0  # init table - we always don't want to take the dummy item
#     for w in range(weight_bound + 1):
#         for ell in range(k + 1):
#             if ell == 0:
#                 mat[w, ell] = 0
#             else:
#                 mat[
#                     w, ell] = -sys.maxsize  # since we still don't know better, the default is that current option is invalid
#                 last_taken[w][ell] = -1
#                 for item in range(0, num_types):
#                     if w - weights[item] >= 0:  # if item is relevant, i.e., can be at all taken
#                         if values[item] + mat[w - weights[item], ell - 1] > mat[w, ell]:  # if it given better value
#                             mat[w, ell] = values[item] + mat[w - weights[item], ell - 1]
#                             last_taken[w][ell] = item
#
#     best_val = mat[weight_bound, k]
#
#     if best_val > target_value + tol:
#         logger.debug('best val: {} target: {}'.format(best_val, target_value))
#         subset = backtrack(last_taken, weight_bound, k, weights)
#         assert best_val - tol < sum(values[item] for item in subset) < best_val + tol
#         assert isinstance(subset, list)
#         return subset
#     else:
#         return None


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

    m = len(z)
    num_item_types = len(z)

    natural_bound = (len(z) - 1) * k

    constraints = []
    names = []
    for i in range(len(targets)):
        # a violated constraint is such that y[i]>sum_of_subset_of(z_j's) while sum_of_subset_of(votes)<targets[i]
        multiset = k_multiset_knapsack(values=z, weights=np.arange(num_item_types, dtype=np.int), k=k,
                                     target_value=y[i],
                                     weight_bound=min(targets[i], natural_bound))

        if multiset:
            configuration, _ = np.histogram(multiset, bins=range(num_item_types + 1))
            assert len(configuration) == num_item_types
            vote_vector_rep = ','.join([str(v) for v in configuration])

            y_component = np.zeros(m, dtype=float)
            y_component[i] = -1.0
            c = np.hstack((y_component, configuration))
            name = ('C', i, vote_vector_rep)
            # name = ('C', i, configuration)

            constraints.append(c)  # the constraint itself
            names.append(name)
            if mode == 'one':
                break
            else:
                pass
    return constraints, names


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
        weight_list = [v for k, v in weighted_configs]
        weight_list = [0.0 if v < 0.0 else v for v in weight_list]
        weights = np.array(weight_list, dtype=np.float32)

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
    # gaps = [target - sigma for sigma in sigmas]
    gaps = target - sigmas
    return lp_solve_by_gaps(m, k, gaps, mode=mode)


def lp_solve_by_gaps(m, k, gaps, mode='one', tol=0.000001):
    """

    Args:
        m (int):
        k (int):
        gaps (List[numpy.int32]):

    Returns:

    """
    assert mode in ['one', 'per_cand', 'per_cand_prune']
    A_trivial = -np.eye(2 * m, dtype=float)
    non_trivial_constraints = []
    non_trivial_const_names = []
    const_names_set = set()

    c = np.array(([1] * m) + ([-k] * m), dtype=float)

    var_names = [('y', i) for i in range(m)] + [('z', i) for i in range(m)]
    triv_const_names = [('trivial', kk, v) for kk, v in var_names]

    lp = lp_solver.HomogenicLpSolver(A_trivial, c, var_names=var_names, const_names=triv_const_names)
    logger.info('Solving...')
    lp.solve()
    logger.info('solved.')

    res = lp.x
    y = res[:m]
    z = res[m:]

    logger.info("finding constraints...")
    new_constraints, new_constraints_names = find_violated_constraints(y, z, gaps, k, mode=mode)
    logger.info("Found")
    while len(new_constraints) > 0:

        logger.info('Adding {} constraints'.format(len(new_constraints)))

        # an existing constraint might be sometimes added
        for constraint, constraint_name in zip(new_constraints, new_constraints_names):
             if constraint_name not in const_names_set:
                non_trivial_constraints.append(constraint)
                non_trivial_const_names.append(constraint_name)
                const_names_set.add(constraint_name)

        # non_trivial_constraints += new_constraints
        # non_trivial_const_names += new_constraints_names

        A = np.vstack([A_trivial] + non_trivial_constraints)
        const_names = triv_const_names + non_trivial_const_names

        lp = lp_solver.HomogenicLpSolver(A, c, var_names=var_names, const_names=const_names)
        logger.info('Solving...')
        lp.solve()
        logger.info('solved.')

        res = lp.x
        y = res[:m]
        z = res[m:]

        # when the dummy constraint is in effect, it is possible that active constraints would still have weight 0. I think
        # that this is due to numerical issues. Hence the `lp.status != lp_solver.UNBOUNDED`
        if mode == 'per_cand_prune' and lp.status != lp_solver.UNBOUNDED:
            non_pruned_constraints = []
            non_pruned_names = []
            for constraint, constraint_name in zip(non_trivial_constraints, non_trivial_const_names):
                if lp[constraint_name] is not None and lp[constraint_name] > tol:
                # if lp[constraint_name] > tol:
                    non_pruned_constraints.append(constraint)
                    non_pruned_names.append(constraint_name)
                else:
                    raise ValueError()                   # non_pruned_constraints.append(c)
                    # non_pruned_names.append(n)
            logger.info('pruned {} constraints'.format(len(non_trivial_constraints) - len(non_pruned_constraints)))
            non_trivial_constraints = non_pruned_constraints
            non_trivial_const_names = non_pruned_names

        # logger.warn('in status {}'.format(prog.status))

        new_constraints, new_constraints_names = find_violated_constraints(y, z, gaps, k, mode=mode)

    # logger.info('(y,z)={}{} status={}'.format(aslist(y.value), aslist(z.value), prog.status))

    logger.info('reached obj val: {}'.format(lp.objective))
    logger.info('{} constraints were added'.format(len(non_trivial_constraints)))

    status = lp.status
    if status in [lp_solver.INFEASIBLE, lp_solver.UNBOUNDED]:
        # logger.warning('status is {}'.format(status))
        return status
    logger.info('{} {}'.format(y, z))
    x_i_C2val = [[] for _ in range(m)]
    for constraint, constraint_name in zip(non_trivial_constraints, non_trivial_const_names):
        _, i, configuration = constraint_name
        x_i_C2val[i].append((configuration, lp[constraint_name]))
    return x_i_C2val


def fix_rounding_result(config_mat, k, initial_sigmas):
    """

    :type initial_sigmas: numpy.ndarray
    :param initial_sigmas:
    :type config_mat: numpy.ndarray
    """
    m = len(initial_sigmas)

    # this is a heauristic to break ties
    awarded = utils.calculate_awarded(config_mat, initial_sigmas)

    tuples = list(enumerate(awarded))
    tuples = sorted(tuples, key=lambda t: t[1], reverse=True)
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
    if mode not in ['one', 'per_cand', 'per_cand_prune']:
        raise ValueError("mode not in ['one', 'per_cand', 'per_cand_prune']")
    if not np.issubdtype(initial_sigmas.dtype, np.integer):
        raise ValueError('initial_sigmas should contain integers.')
        # if not np.issubdtype(np.type(k), np.integer):
        #     raise ValueError('k should be an integer.')

    m = len(initial_sigmas)

    ## hueristic search : try (m-1)/2 * k + average(initial_sigmas) - 1
    # heuristic_lo = int(float(m-1)/2 * k + initial_sigmas.mean() - 1)
    # x_i_C2val = lp_solve(m, k, initial_sigmas, heuristic_lo, mode=mode)
    # if x_i_C2val == lp_solver.UNBOUNDED:
    #     lo = heuristic_lo + 1
    # else:
    #     lo = np.max(initial_sigmas)

    initial_sigmas_sorted = np.sort(initial_sigmas)[::-1]
    lower_bounds = np.array([int(float(i-1)/2 * k + initial_sigmas_sorted[:i].mean())  for i in range(1,m+1)])

    lo = np.max(lower_bounds)

    hi = lo

    interval_size = 1

    # find target by one-sided binary search
    logger.warning('target={}'.format(hi))
    x_i_C2val = lp_solve(m, k, initial_sigmas, hi, mode=mode)

    while x_i_C2val == lp_solver.UNBOUNDED:
        lo = hi
        # target *= 2
        hi = hi + interval_size
        interval_size *= 2
        logger.warning('target={}'.format(hi))
        x_i_C2val = lp_solve(m, k, initial_sigmas, hi, mode=mode)

    # then find target by two-sided binary search
    # lo, hi = last_hi, hi

    last_dual_feasible_solution = x_i_C2val
    # binary search
    while lo < hi:

        mid = (lo + hi) // 2
        logger.warning('mid={}'.format(mid))
        x_i_C2val = lp_solve(m, k, initial_sigmas, mid, mode=mode)
        if x_i_C2val == lp_solver.UNBOUNDED:
            lo = mid + 1
        else:
            last_dual_feasible_solution = x_i_C2val
            hi = mid

    # target = lo
    # x_i_C2val = lp_solve(m, k, sigmas, target)

    x_i_C2val = last_dual_feasible_solution

    assert x_i_C2val != lp_solver.UNBOUNDED

    logger.debug('bin search ended in {}'.format(hi))

    for i, weighted_configs in enumerate(x_i_C2val):
        logger.debug('{}: {}'.format(i, weighted_configs))

    # frac_config_mat = get_frac_config_mat(x_i_C2val)
    # frac_makespan = utils.makespan(initial_sigmas, frac_config_mat)

    frac_makespan = utils.fractional_makespan(initial_sigmas, x_i_C2val)

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

        rounded_config_mat = np.array(res, dtype=np.int32)
        logger.debug('interim configs:')
        logger.debug(rounded_config_mat)
        histogram = np.sum(rounded_config_mat, axis=0)
        logger.debug('histogram={}'.format(histogram))

        cur_config_mat = fix_rounding_result(rounded_config_mat, k, initial_sigmas)
        cur_makespan = utils.makespan(initial_sigmas, cur_config_mat)
        result_range.add(cur_makespan)
        if cur_makespan < best_makespan:
            best_makespan = cur_makespan
            best_config_mat = cur_config_mat
    logger.debug("end fixing loops result range {}".format(result_range))

    return frac_makespan, best_config_mat
