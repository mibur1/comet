import numpy as np
from tqdm.auto import tqdm
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
    C : (N,N), (S,N,N), (N,N,T), or (S,N,N,T) array (or list of arrays)
        Symmetric connectivity matrices with N nodes, S subjects, T time points (dynamic FC)
    type : str, default 'static'
        Type of connectivity data. Options:
            'static': first dim in 3D data is subjects
            'dynamic: first dim in 3D data or second in 4D data is time
    negative_values : str, default 'zero'
        How to handle negative connectivity values. Options:
            'zero': set negative values to zero
            'abs' : take absolute values

    Attributes
    ---------
    C          : connectivity data,
    S          : number of subjects,
    N          : number of nodes,
    T          : number of time points,
    type       : type of connectivity data,
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
    assignments: module assignments at each level,
    
    H_In_cal   : calibrated global integration component,
    H_Seg_cal  : calibrated hierarchical segregation component,
    H_B_cal    : calibrated balance indicator (H_In_cal - H_Seg_cal)
    """
    def __init__(self, C=None, type="static", negative_values="zero"):
        # Figure out data dimensions and set inital attributes
        self.C:         np.ndarray = np.asarray(C, dtype=float).copy()
        self.neg_val:   str = negative_values
        self.type:      str = type
        
        # Data dimensions
        if self.type == "static":
            self.S:         int = self.C.shape[0] if self.C.ndim == 3 else 1
            self.T:         int = 1
            self.N:         int = self.C.shape[-1]
        elif self.type == "dynamic":
            self.S:         int = self.C.shape[0] if self.C.ndim >= 3 else 1
            self.T:         int = self.C.shape[-1]
            self.N:         int = self.C.shape[-2]
        else:
            raise ValueError("Error: type must be 'static' or 'dynamic'")

        # Perform inital checks
        if self.neg_val not in ["zero", "abs"]:
            raise ValueError("Error: negative_values must be 'zero' or 'abs'")
        if self.C.ndim not in [2, 3, 4]:
            raise ValueError("Error: C must be 2D, 3D, or 4D array or list of 2D, 3D arrays")
        if (self.T > 1 and (self.C.shape[-2] != self.C.shape[-3])) or (self.T == 1 and (self.C.shape[-1] != self.C.shape[-2])):
            print(self.C.shape)
            print(self.T, self.C.shape[-2], self.C.shape[-3], self.C.shape[-1], self.C.shape[-2])
            raise ValueError("Error: connectivity matrices must be square. Did you forget to set 'type'?")
        
        if type =="dynamic":
            print(f"NSP (dynamic) initialized with {self.S} subject(s), {self.T} time point(s), and {self.N} nodes.")
        elif type == "static":
            print(f"NSP (static) initialized with {self.S} subject(s) and {self.N} nodes.")

        # Empty attributes for results
        shape1 = self.S if self.S == 1 else (self.S, self.T)
        shape2 = (self.S, self.N) if self.T == 1 else (self.S, self.N, self.T)
        shape3 = (self.S, self.N, self.N) if self.T == 1 else (self.S, self.N, self.N, self.T)
        
        self.H_In:        np.ndarray = np.full(shape1, np.nan)
        self.H_Seg:       np.ndarray = np.full(shape1, np.nan)
        self.H_B:         np.ndarray = np.full(shape1, np.nan)
        
        self.H_In_cal:    np.ndarray = np.full(shape2, np.nan)
        self.H_Seg_cal:   np.ndarray = np.full(shape2, np.nan)
        self.H_B_cal:     np.ndarray = np.full(shape2, np.nan)
        
        self.M_levels:    np.ndarray = np.full(shape2, -1, dtype=int)
        self.H_levels:    np.ndarray = np.full(shape2, np.nan)
        self.p_levels:    np.ndarray = np.full(shape2, np.nan)
        self.eigvals:     np.ndarray = np.full(shape2, np.nan)
        self.eigvals_sq:  np.ndarray = np.full(shape2, np.nan)
        self.eigvecs:     np.ndarray = np.full(shape3, np.nan)
        self.assignments: np.ndarray = np.full(shape3, -1, dtype=int)

        self.A_In       = np.full((self.S), np.nan)
        self.A_Seg      = np.full((self.S), np.nan)
        self.Dwell_In   = np.full((self.S), np.nan)
        self.Dwell_Seg  = np.full((self.S), np.nan)
        self.Trans_freq = np.full((self.S), np.nan)
        self.H_B_thr    = np.full((self.S), np.nan)

    def estimate(self):
        """
        Compute hierarchical integration/segregation measures from connectivity matrix/matrices C
        using the Nested Spectral Partition (NSP) method.
        """
        if self.type == "dynamic":
            for s in tqdm(range(self.S), desc="Subjects"):
                for t in tqdm(range(self.T), desc="Time points", leave=False):
                    C = self.C[s, :, :, t]
                    (H_In, H_Seg, H_B,
                     M_levels, H_levels, p_levels,
                     eigvals, eigvals_sq, eigvecs, assignments) = self._estimate_single(C)

                    self.H_In[s, t]  = H_In
                    self.H_Seg[s, t] = H_Seg
                    self.H_B[s, t]   = H_B
                    self.M_levels[s, :, t]   = M_levels
                    self.H_levels[s, :, t]   = H_levels
                    self.p_levels[s, :, t]   = p_levels
                    self.eigvals[s, :, t]    = eigvals
                    self.eigvals_sq[s, :, t] = eigvals_sq
                    self.eigvecs[s, :, :, t]     = eigvecs
                    self.assignments[s, :, :, t] = assignments

        elif self.type == "static":
            for s in tqdm(range(self.S), desc="Subjects"):
                C = self.C[s, :, :] if self.C.ndim == 3 else self.C
                (H_In, H_Seg, H_B,
                 M_levels, H_levels, p_levels,
                 eigvals, eigvals_sq, eigvecs, assignments) = self._estimate_single(C)

                self.H_In[s]      = H_In
                self.H_Seg[s]     = H_Seg
                self.H_B[s]       = H_B
                self.M_levels[s]  = M_levels
                self.H_levels[s]  = H_levels
                self.p_levels[s]  = p_levels
                self.eigvals[s]   = eigvals
                self.eigvals_sq[s] = eigvals_sq
                self.eigvecs[s]   = eigvecs
                self.assignments[s] = assignments
        else:
            raise ValueError("Error: type must be 'static' or 'dynamic'")
        return

    def calibrate(self, C_master):
        """
        Calibration method

        Parameters
        ----------
        C_master : (N, N) array
            Master connectivity matrix for calibration. Must be symmetric.

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
        H_Seg_stable = results[1] 

        # Population means
        H_In_pop = self.H_In.mean()
        H_Seg_pop = self.H_Seg.mean()

        # Calibration
        self.H_In_cal = self.H_In * (H_In_stable / H_In_pop)
        self.H_Seg_cal = self.H_Seg * (H_Seg_stable / H_Seg_pop)
        self.H_B_cal  = self.H_In_cal - self.H_Seg_cal
        return

    def dynamic_measures(self, calibrated=False, mean="individual"):
        """
        Compute dynamic measures of integration/segregation states over time
        and store them as attributes.

        Parameters
        ----------
        calibrated : bool, default False
            If True, use H_B_cal instead of H_B.
        mean : {"individual", "population"}, default "individual"
            How to define the threshold:
            - "individual": per-subject mean over time (one threshold per subject)
            - "population": global mean over all subjects and time points
                            (one threshold for everyone). If only one subject is
                            present, this falls back to the individual mean.

        Sets
        ----
        self.A_In       : (S,) total integration strength above threshold
        self.A_Seg      : (S,) total segregation strength below threshold
        self.Dwell_In   : (S,) fractional occupancy of integration state
        self.Dwell_Seg  : (S,) fractional occupancy of segregation state
        self.Trans_freq : (S,) transition frequency per frame
        self.H_B_thr    : (S,) threshold(s) used per subject
        """
        if np.isnan(self.H_In).any():
            raise RuntimeError("Call .estimate() before .dynamic_measures().")
        if self.T <= 1:
            raise RuntimeError("Data must be dynamic (T > 1) to compute dynamic measures.")

        # H_B must be (S, T)
        H_B = self.H_B_cal if calibrated else self.H_B

        if H_B.ndim == 1:
            H_B = H_B[np.newaxis, :]

        if mean == "individual":
            # per subject mean over time
            H_B_thr_tmp = H_B.mean(axis=1, keepdims=True)   # (S, 1)

        elif mean == "population":
            if self.S == 1:
                print("Warning: Only one subject present, the population mean is the individual mean.")
                H_B_thr_tmp = H_B.mean(axis=1, keepdims=True)   # (1, 1)
            else:
                # global mean over all subjects and time points -> scalar
                global_mean = H_B.mean()
                H_B_thr_tmp = np.full((self.S, 1), global_mean)      # (S, 1)
        else:
            raise ValueError("mean must be 'individual' or 'population'.")

        # store thresholds as (S,) array
        self.H_B_thr = H_B_thr_tmp[:, 0]

        for s in range(self.S):
            HB_s  = H_B[s]
            thr_s = float(self.H_B_thr[s])

            # Total strengths
            mask_in  = HB_s > thr_s
            mask_seg = HB_s < thr_s

            if np.any(mask_in):
                self.A_In[s] = HB_s[mask_in].sum()
            else:
                self.A_In[s] = 0.0

            if np.any(mask_seg):
                self.A_Seg[s] = np.abs(HB_s[mask_seg]).sum()
            else:
                self.A_Seg[s] = 0.0

            # Fractional occupancy (dwell time)
            self.Dwell_In[s]  = np.count_nonzero(HB_s >= thr_s) / self.T
            self.Dwell_Seg[s] = np.count_nonzero(HB_s <  thr_s) / self.T

            # Transition frequency per frame:
            #   1 if state is "integration" (HB > thr), 0 if "segregation"
            HB_binary = np.where(HB_s > thr_s, 1, 0)     # (T,)
            # transitions = number of times the state changes between t and t+1
            self.Trans_freq[s] = np.sum(np.abs(np.diff(HB_binary))) / (self.T - 1) # each change contributes 1

        return

    def _estimate_single(self, C):
        """Internal method: Estimation for a single connectivity matrix C."""
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
