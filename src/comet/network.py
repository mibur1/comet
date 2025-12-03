import numpy as np

def nsp(C, zero_negative=True):
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

    # Eigendecomposition
    vals, vecs = np.linalg.eigh(C)
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]

    # Squared non-negative eigenvalues
    vals = np.where(vals > 1e-10, vals, 0.0)
    vals_sq = vals ** 2

    # NSP hierarchical partitioning
    H_levels = np.zeros(N)
    M_levels = np.zeros(N, dtype=int)
    p_levels = np.zeros(N)

    assignments = np.zeros(N, dtype=int)
    assignments_all = np.zeros((N, N), dtype=int)
    M_prev = 1
    modules = np.arange(N)

    for i in range(N):
        u = vecs[:, i]
        new_assignments = np.full(N, -1, dtype=int)
        modules = []
        next_label = 0

        for m in range(M_prev):
            nodes = np.where(assignments == m)[0]
            if nodes.size == 0:
                continue

            v_seg = u[nodes]
            pos_mask = v_seg >= 0
            neg_mask = v_seg < 0

            if pos_mask.any() and neg_mask.any():
                nodes_pos = nodes[pos_mask]
                nodes_neg = nodes[neg_mask]

                new_assignments[nodes_pos] = next_label
                modules.append(nodes_pos); next_label += 1

                new_assignments[nodes_neg] = next_label
                modules.append(nodes_neg); next_label += 1

            else:
                new_assignments[nodes] = next_label
                modules.append(nodes); next_label += 1

        assignments = new_assignments
        M_prev = next_label
        assignments_all[i] = assignments

        # Compute module sizes and heterogeneity
        Mi = len(modules)
        M_levels[i] = Mi
        sizes = np.array([len(nodes) for nodes in modules])

        if Mi > 0:
            ideal_size = N / Mi
            p_levels[i] = np.sum(np.abs(sizes - ideal_size)) / N

        # Compute hierarchical component H_i
        H_levels[i] = vals_sq[i] * Mi * (1.0 - p_levels[i]) / N

    # Compute integration, segregation, and balance
    H_In = H_levels[0] / N
    H_Se = np.sum(H_levels[1:]) / N
    H_B = H_In - H_Se

    return {
        "H_In": H_In,
        "H_Se": H_Se,
        "H_B": H_B,
        "M_levels": M_levels,
        "H_levels": H_levels,
        "p_levels": p_levels,
        "eigvals": vals,
        "eigvecs": vecs,
        "assignments": assignments_all
    }