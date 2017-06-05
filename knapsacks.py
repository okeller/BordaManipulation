from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import *

import numpy as np
import sys
import logging

logger = logging.getLogger(__name__)


def backtrack(taken, weight_bound, size, weights):
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

    res = []
    w = weight_bound
    for ell in range(size, 0, -1):
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
        weight_bound (int): an upper bound on resulting subset weight
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


def k_sequnce_knapsack(values, item_weights, penalties, target_value, weight_bound, tol=0.001):
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

    assert values.shape == (len(item_weights), len(penalties))

    num_types = len(item_weights)
    k = len(penalties)

    mat = np.zeros((weight_bound + 1, k + 1), dtype=np.float32)
    last_taken = [[-1] * (k + 1) for b_prime in range(weight_bound + 1)]
    # mat[:, 0] = 0  # init table - we always don't want to take the dummy item
    for b_prime in range(weight_bound + 1):
        for seq_len in range(k + 1):
            if seq_len == 0:
                mat[b_prime, seq_len] = 0
            else:
                last_seq_idx = seq_len - 1
                mat[b_prime, seq_len] = -sys.maxsize  # since we still don't know better, the default is that
                # current option is invalid

                last_taken[b_prime][seq_len] = -1
                for item in range(0, num_types):
                    if b_prime - penalties[last_seq_idx] * item_weights[
                        item] >= 0:  # if item is relevant, i.e., can be at all taken
                        if values[item, last_seq_idx] + \
                                mat[b_prime - penalties[last_seq_idx] * item_weights[item], seq_len - 1] \
                                > mat[b_prime, seq_len]:  # if it given better value
                            mat[b_prime, seq_len] = values[item, last_seq_idx] \
                                                    + mat[b_prime
                                                          - penalties[last_seq_idx] * item_weights[item], seq_len - 1]
                            last_taken[b_prime][seq_len] = item

    best_val = mat[weight_bound, k]

    if best_val > target_value + tol:
        logger.debug('best val: {} target: {}'.format(best_val, target_value))
        subset = backtrack(last_taken, weight_bound, k, item_weights)
        assert isinstance(subset, list)
        return subset
    else:
        return None
