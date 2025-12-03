import numpy as np
from numba import njit

def nested_spectral_partition(C, zero_negative=True):
    """
    Compute hierarchical integration/segregation measures from a symmetric
    connectivity matrix C using the Nested Spectral Partition (NSP) method.

    References
    ----------
    Wang, R. et al. (2021). Segregation, integration, and balance of large-scale resting brain networks 
    configure different cognitive abilities. PNAS, 118(23). https://doi.org/10.1073/pnas.2022288118

    Parameters
    ----------
    C : (N, N) array_like
        Connectivity matrix. Must be symmetric.
    zero_negative : bool, default True
        If True, set negative connectivity values to zero.
    return_assignments : bool, default False
        If True, additionally return module assignments for each level.

    Returns
    -------
    results : dict
        {
          'H_In'       : global integration component,
          'H_Se'       : hierarchical segregation component,
          'H_B'        : balance indicator (H_In - H_Se),
          'M_levels'   : array of module counts M_i for each level,
          'H_levels'   : array of H_i for each eigenmode/level,
          'p_levels'   : size-heterogeneity corrections p_i,
          'eigvals'    : eigenvalues (sorted descending),
          'eigvecs'    : eigenvectors,
          'assignments': module assignments at each level
        }
    """
    C = np.asarray(C, dtype=float).copy()
    N = C.shape[0]

    # Ensure symmetry
    if not np.allclose(C, C.T):
        raise ValueError("Error: Matrix is not symmetric")

    # Zero negative values
    if zero_negative:
        C = np.maximum(C, 0.0)

    # Eigendecomposition (NumPy, not numba)
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    vals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Squared non-negative eigenvalues
    eigvals = np.where(eigvals > 1e-10, eigvals, 0.0)
    eigvals_sq = eigvals ** 2

    # Call numba-compiled core
    H_levels, M_levels, p_levels, assignments_all = _nsp_core(eigvecs, eigvals_sq)

    # Integration, segregation, balance
    N_float = float(C.shape[0])
    H_In = H_levels[0] / N_float
    H_Se = H_levels[1:].sum() / N_float
    H_B = H_In - H_Se

    return {
        "H_In": H_In,
        "H_Se": H_Se,
        "H_B": H_B,
        "M_levels": M_levels,
        "H_levels": H_levels,
        "p_levels": p_levels,
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        "assignments": assignments_all
    }


@njit(fastmath=True, cache=True)
def _nsp_core(eigvecs, eigvals_sq):
    """
    numba-compiled core: builds NSP hierarchy and computes H_i, M_i, p_i.
    eigvecs: (N, N) eigenvectors (columns)
    eigvals_sq: (N,) squared eigenvalues, descending
    """
    N = eigvecs.shape[0]

    H_levels = np.zeros(N)
    M_levels = np.zeros(N, np.int64)
    p_levels = np.zeros(N)

    assignments_all = np.zeros((N, N), np.int64)

    # All nodes initially in module 0
    assignments = np.zeros(N, np.int64)
    M_prev = 1

    # Temp array for cluster node indices
    node_idx = np.empty(N, np.int64)

    for i in range(N):
        u = eigvecs[:, i]  # eigenvector i
        new_assignments = -1 * np.ones(N, np.int64)
        next_label = 0

        # For each module in previous level, split by sign of u
        for m in range(M_prev):
            # collect indices of nodes in module m
            count = 0
            for n in range(N):
                if assignments[n] == m:
                    node_idx[count] = n
                    count += 1

            if count == 0:
                continue

            # check if both signs appear
            pos_found = False
            neg_found = False
            for k in range(count):
                n = node_idx[k]
                if u[n] >= 0.0:
                    pos_found = True
                else:
                    neg_found = True

            if pos_found and neg_found:
                # split into two new modules
                pos_label = next_label
                neg_label = next_label + 1
                for k in range(count):
                    n = node_idx[k]
                    if u[n] >= 0.0:
                        new_assignments[n] = pos_label
                    else:
                        new_assignments[n] = neg_label
                next_label += 2
            else:
                # keep as single module
                label = next_label
                for k in range(count):
                    n = node_idx[k]
                    new_assignments[n] = label
                next_label += 1

        assignments = new_assignments
        M_prev = next_label

        # store assignments at this level
        for n in range(N):
            assignments_all[i, n] = assignments[n]

        # compute module sizes and heterogeneity p_i
        Mi = M_prev
        M_levels[i] = Mi

        if Mi > 0:
            counts = np.zeros(Mi, np.int64)
            for n in range(N):
                lbl = assignments[n]
                if lbl >= 0:
                    counts[lbl] += 1

            ideal_size = float(N) / float(Mi)
            s = 0.0
            for j in range(Mi):
                s += abs(counts[j] - ideal_size)
            p_levels[i] = s / float(N)
        else:
            p_levels[i] = 0.0

        # hierarchical component H_i
        H_levels[i] = eigvals_sq[i] * Mi * (1.0 - p_levels[i]) / float(N)

    return H_levels, M_levels, p_levels, assignments_all
