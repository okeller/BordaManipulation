from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import *

import numpy as np
cimport numpy as np
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






def k_multiset_knapsack_numpy(np.ndarray values, np.ndarray weights, int k, float target_value, int weight_bound, float tol=0.001):
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

    cdef int w,ell
    cdef np.ndarray ok_items
    cdef np.ndarray options_per_item
    cdef int item_taken


    cdef np.ndarray mat = np.empty((weight_bound + 1, k + 1), dtype=np.float32)
    mat.fill( -sys.maxsize)
    mat[:, 0] = 0

    cdef np.ndarray last_taken = np.empty((weight_bound + 1, k + 1), dtype=np.int)
    last_taken.fill(-1)

    # last_taken = [[-1] * (k + 1) for w in range(weight_bound + 1)]


    # mat[:, 0] = 0  # init table - we always don't want to take the dummy item
    for w in range(weight_bound + 1):
        for ell in range(1,k + 1):

            # for item in range(0, num_types):
            #     if w - weights[item] >= 0:  # if item is relevant, i.e., can be at all taken
            #         if values[item] + mat[w - weights[item], ell - 1] > mat[w, ell]:  # if it given better value
            #             mat[w, ell] = values[item] + mat[w - weights[item], ell - 1]
            #             last_taken[w][ell] = item

            ok_items = w-weights >= 0
            if np.any(ok_items):
                options_per_item = np.where(ok_items,  values + mat[w - weights, ell - 1] ,  -sys.maxsize  )
                item_taken = np.argmax(options_per_item)
                mat[w, ell] = options_per_item[item_taken]
                last_taken[w, ell] = item_taken



    cdef int best_val = mat[weight_bound, k]

    if best_val > target_value + tol:
        logger.debug('best val: {} target: {}'.format(best_val, target_value))
        subset = backtrack(last_taken, weight_bound, k, weights)
        # assert best_val - tol < sum(values[item] for item in subset) < best_val + tol
        assert isinstance(subset, list)
        return subset
    else:
        return None



