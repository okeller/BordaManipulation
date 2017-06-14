import numpy as np
import utils


def find_strategy(initial_sigmas, alpha, weights):
    if len(initial_sigmas) != len(alpha):
        raise ValueError(len(initial_sigmas) != len(alpha))

    m = len(initial_sigmas)
    k = len(weights)

    current_awarded = initial_sigmas.copy()

    config_mat = np.zeros((m, k), dtype=int)

    for ell in range(k):

        # now sort the candidates according to the current awarded score, high to low:
        candidates_sorted = sorted(utils.borda(m), key=lambda i: current_awarded[i], reverse=True)
        # now achieve inverse-sort: every candidate gets his rank (high-to-low) in sort
        ballot = np.zeros(m, dtype=int)
        for rank, cand in enumerate(candidates_sorted):
            ballot[cand] = rank

        config_mat[:, ell] = ballot
        current_awarded += alpha[ballot]

    return config_mat
