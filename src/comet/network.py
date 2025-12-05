import numpy as np
from tqdm import tqdm
from numba import njit

class NestedSpectralPartition():
    """
    Implementation of the Nested Spectral Partition (NSP) method as implemented in Wang et al. (2021).

    References
    ----------
    Wang, R. et al. (2021). Segregation, integration, and balance of large-scale resting brain networks 
    configure different cognitive abilities. PNAS, 118(23). https://doi.org/10.1073/pnas.2022288118

    Parameters
    ----------
    C : (N, N), (S, N, N), or (S, T, N, N) array (or list of arrays)
        Symmetric connectivity matrices.
            N nodes, S subjects, T time points (dynamic)
    negative_values : str, default 'zero'
        How to handle negative connectivity values. Options:
            'zero': set negative values to zero
            'abs' : take absolute values
    type : str, default 'static'
        Type of connectivity data. Options:
            'static': first dim in 3D data is subjects
            'dynamic: first dim in 3D data or second in 4D data is time

    Attributes
    ---------
    C          : connectivity data,
    S          : number of subjects,
    N          : number of nodes,
    T          : number of time points,
    neg_val    : method for handling negative values,
    
    H_In       : global integration component,
    H_Seg      : hierarchical segregation component,
    H_B        : balance indicator (H_In - H_Seg),
    
    M_levels   : array of module counts M_i for each level,
    H_levels   : array of H_i for each eigenmode/level,
    p_levels   : size-heterogeneity corrections p_i,
    eigvals    : eigenvalues (sorted descending),
    eigvals_sq : squared eigenvalues,
    eigvecs    : eigenvectors,
    assignments: module assignments at each level
    
    H_In_cal   : calibrated global integration component,
    H_Seg_cal  : calibrated hierarchical segregation component,
    H_B_cal    : calibrated balance indicator (H_In_cal - H_Seg_cal)
    """
    def __init__(self, C=None, negative_values="zero", type="static"):
        # Figure out data dimensions and set inital attributes
        self.C:         np.ndarray = np.asarray(C, dtype=float).copy()
        self.neg_val:   str = negative_values
        self.type:      str = type
        
        self.S:         int = self.C.shape[0] if self.C.ndim >= 3 else 1
        self.N:         int = self.C.shape[-1]
        self.T:         int = 1 if self.type == "static" else self.C.shape[-3]

        # Perform inital checks
        if self.neg_val not in ["zero", "abs"]:
            raise ValueError("Error: negative_values must be 'zero' or 'abs'")
        if self.type not in ["static", "dynamic"]:       
            raise ValueError("Error: type must be 'static' or 'dynamic'")
        if self.C.ndim not in [2, 3, 4]:
            raise ValueError("Error: C must be 2D, 3D, or 4D array or list of 2D, 3D arrays")
        if self.C.shape[-1] != self.C.shape[-2]:
            raise ValueError("Error: connectivity matrices must be square")
        
        print(f"NSP initialized with {self.S} subject(s), {self.N} node(s), {self.T} time point(s).")
        
        # Empty attributes for results
        self.H_In:        np.ndarray = np.full(self.S, np.nan)
        self.H_Seg:       np.ndarray = np.full(self.S, np.nan)
        self.H_B:         np.ndarray  = np.full(self.S, np.nan)
        
        self.M_levels:    np.ndarray = np.full((self.S, self.N), -1, dtype=int)
        self.H_levels:    np.ndarray = np.full((self.S, self.N), np.nan)
        self.p_levels:    np.ndarray = np.full((self.S, self.N), np.nan)
        self.eigvals:     np.ndarray = np.full((self.S, self.N), np.nan)
        self.eigvals_sq:  np.ndarray = np.full((self.S, self.N), np.nan)
        self.eigvecs:     np.ndarray = np.full((self.S, self.N, self.N), np.nan)
        self.assignments: np.ndarray = np.full((self.S, self.N, self.N), -1, dtype=int)

        self.H_In_cal:    np.ndarray = np.full(self.S, np.nan)
        self.H_Seg_cal:   np.ndarray = np.full(self.S, np.nan)
        self.H_B_cal:     np.ndarray  = np.full(self.S, np.nan)

    def estimate(self):
        """
        Compute hierarchical integration/segregation measures from a symmetric
        connectivity matrix/matrices C using the Nested Spectral Partition (NSP) method.
        """
        for s in tqdm(range(self.S)):
            C = self.C[s, :, :] if self.C.ndim == 3 else self.C
            results = self._estimate_single(C)
            
            # Assign attributes
            self.H_In[s]        = results[0]
            self.H_Seg[s]       = results[1]
            self.H_B[s]         = results[2]
            self.M_levels[s, :] = results[3]
            self.H_levels[s, :] = results[4]
            self.p_levels[s, :] = results[5]
            self.eigvals[s, :]  = results[6]
            self.eigvals_sq[s, :] = results[7]
            self.eigvecs[s, :, :] = results[8]
            self.assignments[s, :, :] = results[9]

    def estimate_dynamic(self):
        pass

    def _estimate_single(self, C):
        """Estimation for a single connectivity matrix C."""
        # Ensure symmetry
        if not np.allclose(C, C.T):
            raise ValueError("Error: Matrix is not symmetric")

        # Handle negative values
        C = np.maximum(C, 0.0) if self.neg_val == "zero" else np.abs(C)

        # Eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(C)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Squared non-negative eigenvalues
        eigvals = np.where(eigvals > 1e-10, eigvals, 0.0)
        eigvals_sq = eigvals ** 2

        # Numba-compiled core
        M_levels, H_levels, p_levels, assignments = _nsp_core(eigvecs, eigvals_sq)

        # Integration, segregation, balance
        H_In = H_levels[0] / self.N
        H_Seg = H_levels[1:].sum() / self.N
        H_B = H_In - H_Seg

        return (H_In, H_Seg, H_B, M_levels, H_levels, p_levels, eigvals, eigvals_sq, eigvecs, assignments)

    def calibrate(self, C_master):
        """
        Calibration method

        Parameters
        ----------
        C_master : (N, N) array
            Master connectivity matrix for calibration. Must be symmetric.
        negative_values : str, default 'zero'
            How to handle negative connectivity values. Options:
                'zero': set negative values to zero
                'abs' : take absolute values
        
        Attributes
        ----------
        H_in_cal : calibrated global integration component,
        H_seg_cal: calibrated hierarchical segregation component,
        H_B_cal  : calibrated balance indicator (H_in_cal - H_seg_cal)
        """
        # Ensure .estimate() has been called
        if np.isnan(self.H_In).any():
            raise RuntimeError("Call .estimate() before .calibrate().")
        
        # Calculate stable integration component
        results = self._estimate_single(C_master)
        H_In_stable = results[0] 

        # Population means
        H_In_pop = self.H_In.mean()
        H_Seg_pop = self.H_Seg.mean()

        # Calibration
        self.H_In_cal = self.H_In * (H_In_stable / H_In_pop)
        self.H_Seg_cal = self.H_Seg * (H_In_stable / H_Seg_pop)
        self.H_B_cal  = self.H_In_cal - self.H_Seg_cal
        return


@njit(fastmath=True, cache=True)
def _nsp_core(eigvecs, eigvals_sq):
    """
    numba-compiled core: builds NSP hierarchy and computes M_i, H_i, p_i, and assignments.
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

    return M_levels, H_levels, p_levels, assignments_all
