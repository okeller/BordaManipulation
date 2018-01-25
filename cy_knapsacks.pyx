from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import *

import numpy as np
cimport numpy as np

import sys
import logging

DTYPE_INT = np.int
ctypedef np.int_t DTYPE_INT_t

DTYPE_FLOAT = np.float64
ctypedef np.float64_t DTYPE_FLOAT_t

logger = logging.getLogger(__name__)

def backtrack(np.ndarray[DTYPE_INT_t, ndim=2] taken, DTYPE_INT_t weight_bound, DTYPE_INT_t size,
              np.ndarray[DTYPE_INT_t, ndim=1] weights):
    """
    Backtracks the table of `exclusive_knapsack` to produce the resulting multiset

    Args:
        taken (List[List[int]]): the table confirming whether an item was taken or not
        weight_bound (int):
        size (int):
        weights (List[int]):

    Returns:
        List[int]: A multi-subset as list

    """

    cdef list res = []
    cdef int w, ell, item

    w = weight_bound
    for ell in range(size, 0, -1):
        item = taken[w, ell]
        res.append(item)
        w -= weights[item]
    res.reverse()
    return res

def backtrack_for_k_sequence(np.ndarray[DTYPE_INT_t, ndim=2] taken, DTYPE_INT_t weight_bound, DTYPE_INT_t size,
                             np.ndarray[DTYPE_INT_t, ndim=1] weights, np.ndarray[DTYPE_INT_t, ndim=1] pernalties):
    """
    Backtracks the table of `exclusive_knapsack` to produce the resulting multiset

    Args:
        taken (List[List[int]]): the table confirming whether an item was taken or not
        weight_bound (int):
        size (int):
        weights (List[int]):

    Returns:
        List[int]: A multi-subset as list

    """

    cdef list res = []
    cdef int w, ell, item

    res = []
    w = weight_bound
    for ell in range(size, 0, -1):
        item = taken[w, ell]
        res.append(item)
        w -= weights[item] * pernalties[ell - 1]

    return list(reversed(res))

def k_multiset_knapsack(np.ndarray[DTYPE_FLOAT_t, ndim=1] values, np.ndarray[DTYPE_INT_t, ndim=1] weights, int k,
                        float target_value, int weight_bound, float tol=0.001):
    """
    Solves an instance of the k-multiset knapsack
    Args:
        values (List[float]): a vector of item values
        weights (List[int]): a vector of item weights
        k (int): the size of the resulting multiset
        target_value (float): a lower bound on resulting subset value
        weight_bound (int): an upper bound on resulting subset weight
        tol (float): a tolerance parameter

    Returns:
        List[int]: A multiset of items of size `k` represented as an list containing the histogram of score types.

    """

    cdef int num_types, w, ell, item
    cdef float best_val
    cdef list subset
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] mat
    cdef np.ndarray[DTYPE_INT_t, ndim=2] last_taken = np.empty((weight_bound + 1, k + 1), dtype=np.int)
    last_taken.fill(-1)

    assert len(values) == len(weights)

    num_types = len(values)

    mat = np.zeros((weight_bound + 1, k + 1), dtype=DTYPE_FLOAT)
    # last_taken = [[-1] * (k + 1) for w in range(weight_bound + 1)]
    # mat[:, 0] = 0  # init table - we always don't want to take the dummy item
    for w in range(weight_bound + 1):
        for ell in range(k + 1):
            if ell == 0:
                mat[w, ell] = 0
            else:
                mat[
                    w, ell] = -sys.maxsize  # since we still don't know better, the default is that current option is invalid
                last_taken[w, ell] = -1
                for item in range(0, num_types):
                    if w - weights[item] >= 0:  # if item is relevant, i.e., can be at all taken
                        if values[item] + mat[w - weights[item], ell - 1] > mat[w, ell]:  # if it given better value
                            mat[w, ell] = values[item] + mat[w - weights[item], ell - 1]
                            last_taken[w, ell] = item

    best_val = mat[weight_bound, k]

    if best_val > target_value + tol:
        logger.debug('best val: {} target: {}'.format(best_val, target_value))
        subset = backtrack(last_taken, weight_bound, k, weights)
        # assert best_val - tol < sum(values[item] for item in subset) < best_val + tol
        assert isinstance(subset, list)
        return subset
    else:
        return None

def k_multiset_knapsack_numpy(np.ndarray[DTYPE_FLOAT_t, ndim=1] values, np.ndarray[DTYPE_INT_t, ndim=1] weights, int k,
                              float target_value, int weight_bound, float tol=0.001):
    """
    Solves an instance of the k-multiset knapsack
    Args:
        values (List[float]): a vector of item values
        weights (List[int]): a vector of item weights
        k (int): the size of the resulting multiset
        target_value (float): a lower bound on resulting subset value
        weight_bound (int): an upper bound on resulting subset weight
        tol (float): a tolerance parameter

    Returns:
        List[int]: A multiset of items of size `k` represented as an list containing the histogram of score types.

    """
    assert values.dtype == DTYPE_FLOAT
    assert weights.dtype == DTYPE_INT
    assert len(values) == len(weights)

    cdef int w, ell
    cdef np.ndarray ok_items
    cdef np.ndarray options_per_item
    cdef int item_taken
    cdef list subset

    cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] mat = np.empty((weight_bound + 1, k + 1), dtype=DTYPE_FLOAT)
    mat.fill(-sys.maxsize)
    mat[:, 0] = 0

    cdef np.ndarray[DTYPE_INT_t, ndim=2] last_taken = np.empty((weight_bound + 1, k + 1), dtype=np.int)
    last_taken.fill(-1)

    # last_taken = [[-1] * (k + 1) for w in range(weight_bound + 1)]

    # mat[:, 0] = 0  # init table - we always don't want to take the dummy item
    for w in range(weight_bound + 1):
        for ell in range(1, k + 1):

            # for item in range(0, num_types):
            #     if w - weights[item] >= 0:  # if item is relevant, i.e., can be at all taken
            #         if values[item] + mat[w - weights[item], ell - 1] > mat[w, ell]:  # if it given better value
            #             mat[w, ell] = values[item] + mat[w - weights[item], ell - 1]
            #             last_taken[w][ell] = item

            ok_items = w - weights >= 0
            if np.any(ok_items):
                options_per_item = np.where(ok_items, values + mat[w - weights, ell - 1], -sys.maxsize)
                item_taken = np.argmax(options_per_item)
                mat[w, ell] = options_per_item[item_taken]
                last_taken[w, ell] = item_taken

    cdef float best_val = mat[weight_bound, k]

    if best_val > target_value + tol:
        logger.debug('best val: {} target: {}'.format(best_val, target_value))
        subset = backtrack(last_taken, weight_bound, k, weights)
        # assert best_val - tol < sum(values[item] for item in subset) < best_val + tol
        assert isinstance(subset, list)
        return subset
    else:
        return None

def k_sequnce_knapsack(np.ndarray[DTYPE_FLOAT_t, ndim=2] values, np.ndarray[DTYPE_INT_t, ndim=1] item_weights,
                       np.ndarray[DTYPE_INT_t, ndim=1] penalties, float target_value, int weight_bound,
                       float tol=0.001):
    """
    Solves an instance of the k-multiset knapsack
    Args:
        values (List[float]): a vector of item values
        item_weights (List[int]): a vector of item weights
        k (int): the size of the resulting multiset
        target_value (float): a lower bound on resulting subset value
        weight_bound (int): an upper bound on resulting subset weight
        tol (float): a tolerance parameter

    Returns:
        List[int]: A multiset of items of size `k` represented as an list containing the histogram of score types.

    """

    # assert values.shape == (len(item_weights), len(penalties))
    assert values.shape[0] == item_weights.shape[0]
    assert values.shape[1] == penalties.shape[0]

    cdef int num_types, k, b_prime, seq_len, curr_seq_idx
    k = len(penalties)
    cdef float best_val
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] mat
    cdef list subset
    cdef np.ndarray[DTYPE_INT_t, ndim=2] last_taken = np.empty((weight_bound + 1, k + 1), dtype=np.int)
    last_taken.fill(-1)

    num_types = len(item_weights)


    mat = np.zeros((weight_bound + 1, k + 1), dtype=DTYPE_FLOAT)
    # last_taken = [[-1] * (k + 1) for b_prime in range(weight_bound + 1)]
    # mat[:, 0] = 0  # init table - we always don't want to take the dummy item
    for b_prime in range(weight_bound + 1):
        for seq_len in range(k + 1):
            if seq_len == 0:
                mat[b_prime, seq_len] = 0
            else:
                curr_seq_idx = seq_len - 1
                mat[b_prime, seq_len] = -sys.maxsize  # since we still don't know better, the default is that
                # current option is invalid

                last_taken[b_prime,seq_len] = -1
                for item in range(0, num_types):

                    cost_of_item = penalties[curr_seq_idx] * item_weights[item]
                    value_of_item = values[item, curr_seq_idx]

                    if b_prime - cost_of_item >= 0:  # if item is relevant, i.e., can be at all taken
                        if value_of_item + mat[b_prime - cost_of_item, seq_len - 1] > mat[b_prime, seq_len]:
                            # if it given better value

                            mat[b_prime, seq_len] = value_of_item + mat[b_prime - cost_of_item, seq_len - 1]
                            last_taken[b_prime][seq_len] = item

    best_val = mat[weight_bound, k]

    if best_val > target_value + tol:
        logger.debug('best val: {} target: {}'.format(best_val, target_value))
        subset = backtrack_for_k_sequence(last_taken, weight_bound, k, item_weights, penalties)
        assert isinstance(subset, list)
        return subset
    else:
        return None
