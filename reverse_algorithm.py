import numpy as np



def find_strategy(initial_sigmas, alpha, weights):
    if len(initial_sigmas) != len(alpha):
        raise ValueError(len(initial_sigmas) != len(alpha))

    m = len(initial_sigmas)
    k = len(weights)

    current_awarded = initial_sigmas.copy()

    config_mat = np.zeros((m, k), dtype=int)

    for ell in range(k):
        # now sort the score indexes according to the current awarded score, high to low:
        ballot = sorted(np.arange(m), key=lambda i: current_awarded[i], reverse=True)
        config_mat[:, ell] = ballot
        current_awarded += alpha[ballot]

    return config_mat
