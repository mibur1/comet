import os
import re
import mat73
import pickle
import inspect
import numpy as np
import pandas as pd
import importlib_resources
from nilearn import signal
from scipy.io import loadmat
from sklearn.cluster import KMeans

def load_timeseries(path=None):
    """
    Load time series data from a file.
    Supported file formats are: .pkl, .txt, .npy, .mat, and .tsv

    Parameters
    ----------
    path : string
        path to the time series data file.

    Returns
    -------
    data : TxP np.ndarray
        time series data
    """

    if path is None:
        raise ValueError("Please provide a path to the time series data")

    if path.endswith(".pkl"):
        with open(path, 'rb') as file:
            data = pickle.load(file)
    elif path.endswith(".txt"):
        data = np.loadtxt(path)
    elif path.endswith(".npy"):
        data = np.load(path)
    elif path.endswith(".mat"):
        try:
            data = loadmat(path)
        except Exception as e:
            print("Error using scipy, using mat73 instead.", e)
            data = mat73.loadmat(path)
    elif path.endswith(".tsv"):
        data = pd.read_csv(path, sep='\t', header=None, na_values='n/a')

        if data.iloc[0].apply(lambda x: np.isscalar(x) and np.isreal(x)).all():
            rois = None  # No rois found, the first row is part of the data
        else:
            rois = data.iloc[0]  # The first row is rois
            data = data.iloc[1:]  # Remove the header row from the data

        # Convert all data to numeric, making sure 'n/a' and other non-numeric are treated as NaN
        data = data.apply(pd.to_numeric, errors='coerce')

        # Identify entirely empty columns
        empty_columns = data.columns[data.isna().all()]

        # Remove corresponding rois if rois exist
        if rois is not None:
            removed_rois = rois[empty_columns].to_list()
            print("The following regions were empty and thus removed:", removed_rois)
            rois = rois.drop(empty_columns)

        # Remove entirely empty columns and rows
        data = data.dropna(axis=1, how='all').dropna(axis=0, how='all')

        # Convert the cleaned data back to numpy array
        data = data.to_numpy()

        # Update header_list if rois exist
        rois = rois.to_list() if rois is not None else None
        return data, rois
    
    elif path.endswith(".nii") or path.endswith(".nii.gz"):
        data = None # For compatibility with the GUI

    else:
        raise ValueError("Unsupported file format")

    return data

def load_example(fname="time_series.txt"):
    """
    Load example data.

    Parameters
    ----------
    fname : str, optional
        File name for any of the included data
        - 'time_series.txt':            Parcellated BOLD time series data for one subject
        - 'time_series_multiple.npy':   Parcellated BOLD time series data for 5 subjects
        - 'simulation.mat':             Simulated time series data for the tutorials
        - 'hurricane.tsv':              Hurricane data for the hurricane multiverse tutorial
        Default is 'time_series.txt'.

    Returns
    -------
    data : np.ndarray
       TxP np.ndarray containing the time series data

    """
    with importlib_resources.path("comet.data", fname) as file_path:
        # Handle different data files
        if fname == "time_series.txt":
            data = np.loadtxt(file_path)
        elif fname == "time_series_multiple.npy":
            data = np.load(file_path)
        elif fname == "simulation.mat":
            data = mat73.loadmat(file_path)
        elif fname == "hurricane.tsv":
            data = pd.read_csv(file_path, sep="\t")
        else:
            print("Error: Unsupported file name")

    return data

def load_testdata(data=None):
    """
    Load test data for unit tests.
    """
    if data in ["graph", "connectivity"]:
        fname = f"{data}.mat"
    else:
        raise ValueError("Valid test names are: 'graph', 'connectivity', 'multiverse', or 'data'.")
    
    with importlib_resources.path("comet.data.tests", fname) as file_path:
        data = loadmat(file_path)
    return data

def save_universe_results(data):
    """
    This saves the results of a universe.

    If it is a single value, it will be saved in the summary .csv file.
    In any other case the results will be saved in a universe specific .pkl file.

    Parameters
    ----------
    data : any
        Data to save as .pkl file
    """

    if not isinstance(data, dict):
        raise ValueError("Data must be povided as a dictionary.")

    # Identify calling universe and paths
    caller_stack = inspect.stack()
    universe_fname = caller_stack[1].filename               # .../multiverse/scripts/universe_#.py
    scripts_dir     = os.path.dirname(universe_fname)       # .../multiverse/scripts
    multiverse_dir  = os.path.dirname(scripts_dir)          # .../multiverse

    m = re.search(r'universe_(\d+)\.py$', os.path.basename(universe_fname))
    if not m:
        raise RuntimeError("Could not parse universe number from filename.")
    universe_number = int(m.group(1))

    # Attach decisions from multiverse_summary.csv
    summary_path = os.path.join(multiverse_dir, "multiverse_summary.csv")
    df = pd.read_csv(summary_path)

    # Match row: Universe column uses 'Universe_#'
    key = f"Universe_{universe_number}".lower()
    row = df.loc[df["Universe"].str.lower() == key]
    if not row.empty:
        decisions = row.drop(columns=["Universe"]).iloc[0].to_dict()
        data = dict(data)
        data["decisions"] = decisions

    # Save the data as a .pkl file in scripts/temp
    savedir = os.path.join(scripts_dir, "temp")
    os.makedirs(savedir, exist_ok=True)

    file = os.path.join(savedir, f"universe_{universe_number}.pkl")
    with open(file, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def clean(time_series, detrend=False, confounds=None, standardize=False, standardize_confounds=True, \
          filter='butterworth', low_pass=None, high_pass=None, t_r=None, ensure_finite=False):
    """
    Wrapper function for nilearn.clean() for cleaning time series data

    Parameters
    ----------
    time_series : TxP np.ndarray
        time series data

    detrend : bool, optional
        Detrend the data. Default is False.

    confounds : np.ndarray, str, pathlib.Path, pandas.DataFrame, or list of confounds
        Confounds to be regressed out from the data. Default is None.

    standardize : bool, optional
        Z-score the data. Default is False.

    standardize_confounds : bool, optional
        Z-score the confounds. Default is True.

    filter : str {butterworth, cosine, False}
        Filtering method. Default is 'butterworth'.

    low_pass : float, optional
        Low cutoff frequency in Hertz. Default is None.

    high_pass : float, optional
        High cutoff frequency in Hertz. Default is None.

    t_r : float, optional
        Repetition time, in seconds (sampling period). Default is None

    ensure_finite : bool, optional
        Check if the data contains only finite numbers. Default is False.

    Returns
    -------
    data : TxP np.ndarray
        cleaned time series data
    """

    return signal.clean(time_series, detrend=detrend, confounds=confounds, standardize=standardize, standardize_confounds=standardize_confounds, \
                        filter=filter, low_pass=low_pass, high_pass=high_pass, t_r=t_r, ensure_finite=ensure_finite)

def notebookToScript(notebook):
    """
    Convert a Jupyter notebook JSON to a Python script.
    """
    scriptContent = ""
    try:
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                scriptContent += ''.join(cell['source']) + '\n\n'
    except KeyError as e:
        print("Error", f"Invalid notebook format: {str(e)}")

    return scriptContent

# State-analysis utilities
def kmeans_cluster(
    dfc,
    num_states: int = 5,
    strategy: str = "pooled",   # "pooled" or "two_level"
    subject_clusters: int = 5,  # only used for "two_level"
    standardise_features: bool = False,
    diag_value: float = 0.0,
    random_state: int | None = None,
    n_init: int = 50):
    """
    Cluster continuously varying dFC into K discrete states using k-means.

    Parameters
    ----------
    dfc : np.ndarray
        (P,P,T) single-subject or (S,P,P,T) multi-subject dynamic FC.
    num_states : int
        Number of group states (K).
    strategy : {"pooled", "two_level"}
        - "pooled": cluster all timepoints across subjects together.
        - "two_level": per-subject k-means to obtain subject-level centroids,
          then cluster all centroids to define group states; finally assign each
          time point to the nearest group state.
    subject_clusters : int
        First-level k for "two_level".
    standardise_features : bool
        Z-score each edge across all samples before k-means.
    diag_value : float
        Main diagonal values for state connectivity matrices.
    random_state : int or None
        Reproducibility for KMeans.

    Returns
    -------
    state_tc : (S,T) int array
        State label per time point per subject (S=1 for single-subject).
    states : (P,P,K) float array
        State connectivity matrices reconstructed from centroids (same units as input).
    inertia : float
        Inertia of the final (group) k-means.
    """
    X_dfc = np.asarray(dfc)
    if X_dfc.ndim == 3:
        # (P,P,T) -> (T,P,P)
        P, P2, T = X_dfc.shape
        if P != P2 or T == 0:
            raise ValueError("For (P,P,T), P must match and T>0.")
        S = 1
        iu = np.tril_indices(P, k=-1)
        # (T, M)
        X_all = X_dfc.transpose(2, 0, 1)[:, iu[0], iu[1]]
        # (S,T,M)
        X_all = X_all[None, ...]
    elif X_dfc.ndim == 4:
        # (S,P,P,T) -> (S,T,P,P)
        S, P, P2, T = X_dfc.shape
        if P != P2 or T == 0:
            raise ValueError("For (S,P,P,T), P must match and T>0.")
        iu = np.tril_indices(P, k=-1)
        # (S,T,M)
        X_all = X_dfc.transpose(0, 3, 1, 2)[:, :, iu[0], iu[1]]
    else:
        raise ValueError("dfc must be (P,P,T) or (S,P,P,T).")

    # Flatten subjects×time to samples: (S*T, M)
    X = X_all.reshape(S * T, -1)

    # Optional standardisation across all samples (global)
    if standardise_features:
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True) + 1e-8
        Xs = (X - mu) / sd
    else:
        Xs = X

    def rebuild_states(centres_vec):
        """Rebuild (P,P,K) from lower-tri vectors using the same iu ordering as above."""
        K = centres_vec.shape[0]
        states = np.zeros((P, P, K), dtype=centres_vec.dtype)
        for k in range(K):
            m = np.zeros((P, P), dtype=centres_vec.dtype)
            m[iu] = centres_vec[k]
            m = m + m.T

            np.fill_diagonal(m, diag_value)
            states[:, :, k] = m
        return states

    if strategy == "pooled":
        km = KMeans(n_clusters=num_states, n_init=n_init, random_state=random_state)
        labels_flat = km.fit_predict(Xs)                  # (S*T,)
        states = rebuild_states(km.cluster_centers_)      # (P,P,K)
        inertia = float(km.inertia_)
        state_tc = labels_flat.reshape(S, T)

    elif strategy == "two_level":
        # First-level per-subject clustering on (T,M) slices from Xs_all
        if Xs.shape[0] != S * T:
            raise ValueError("Internal shape error: samples != S*T.")
        centroids_list = []
        for s in range(S):
            Xs_s = Xs[s*T:(s+1)*T, :]                     # (T,M)
            if subject_clusters > T:
                raise ValueError(f"subject_clusters ({subject_clusters}) > T ({T}) for subject {s}.")
            km1 = KMeans(n_clusters=subject_clusters, n_init=n_init, random_state=random_state)
            km1.fit(Xs_s)
            centroids_list.append(km1.cluster_centers_)   # (k1,M)
        C = np.vstack(centroids_list)                     # (S*subject_clusters, M)

        # Group-level clustering on centroids
        km2 = KMeans(n_clusters=num_states, n_init=50, random_state=random_state)
        km2.fit(C)
        states = rebuild_states(km2.cluster_centers_)     # (P,P,K)
        inertia = float(km2.inertia_)

        # Assign each sample to nearest group centroid
        labels_flat = km2.predict(Xs)                     # (S*T,)
        state_tc = labels_flat.reshape(S, T)

    else:
        raise ValueError("strategy must be 'pooled' or 'two_level'.")

    return state_tc, states, inertia

def summarise_state_tc(state_tc: np.ndarray) -> dict:
    """
    Summarise dwell times, occupancies, transitions, and switching behaviour across subjects.

    Parameters
    ----------
    state_tc : (S, T) int array
        State time courses for S subjects, each of length T.
        Each entry represents the active state label at a given time point.

    Returns
    -------
    dict with:
        'dwell_times'        : (S, K)
            Mean contiguous dwell length per state and subject (in time points).
        'fractional_occupancy' : (S, K)
            Fraction of total time spent in each state per subject.
        'transitions'        : (S, K, K)
            Row-stochastic transition probability matrices per subject (P[j|i]).
        'transition_counts'  : (S, K, K)
            Raw transition count matrices per subject.
        'transitions_sum'    : (S,)
            Total number of state changes (switches) per subject.
        'switch_rate'        : (S,)
            Fraction of time points involving a switch, i.e. transitions_sum / (T-1).
    """
    state_tc = np.asarray(state_tc)
    if state_tc.ndim != 2:
        raise ValueError("state_tc must be a 2D array of shape (S, T).")

    S, T = state_tc.shape
    K = int(state_tc.max()) + 1 if state_tc.size else 0

    dwell = np.zeros((S, K), dtype=float)
    fo = np.zeros((S, K), dtype=float)
    trans_P = np.zeros((S, K, K), dtype=float)
    trans_C = np.zeros((S, K, K), dtype=int)
    ntrans = np.zeros(S, dtype=int)
    srate = np.zeros(S, dtype=float)

    for s in range(S):
        labels = state_tc[s]
        dwell[s] = dwell_times(labels, K)
        fo[s] = fractional_occupancy(labels, K)
        trans_C[s] = transition_counts(labels, K)
        trans_P[s] = transition_matrix(labels, K)
        ntrans[s] = num_transitions(labels)
        srate[s] = switch_rate(labels)

    summary = {
        "dwell_times": dwell,
        "fractional_occupancy": fo,
        "transitions": trans_P,
        "transition_counts": trans_C,
        "transitions_sum": ntrans,
        "switch_rate": srate,
    }

    return summary

def fractional_occupancy(labels: np.ndarray, K: int) -> np.ndarray:
    """
    Fraction of time spent in each state.
    """
    labels = np.asarray(labels, dtype=int)
    T = labels.size
    fo = np.bincount(labels, minlength=K) / max(T, 1)
    return fo.astype(float)

def dwell_times(labels: np.ndarray, K: int) -> np.ndarray:
    """
    Mean dwell time (average contiguous run length) per state.
    Returns 0 for states never visited.
    """
    labels = np.asarray(labels, dtype=int)
    runs = [[] for _ in range(K)]
    if labels.size == 0:
        return np.zeros(K, dtype=float)

    run_len = 1
    for t in range(1, labels.size):
        if labels[t] == labels[t - 1]:
            run_len += 1
        else:
            runs[labels[t - 1]].append(run_len)
            run_len = 1
    runs[labels[-1]].append(run_len)

    out = np.zeros(K, dtype=float)
    for k in range(K):
        out[k] = np.mean(runs[k]) if runs[k] else 0.0
    return out

def transition_counts(labels: np.ndarray, K: int) -> np.ndarray:
    """
    Raw transition counts C[i,j] = number of i→j transitions.
    """
    labels = np.asarray(labels, dtype=int)
    C = np.zeros((K, K), dtype=int)
    if labels.size < 2:
        return C
    a, b = labels[:-1], labels[1:]
    np.add.at(C, (a, b), 1)
    return C

def transition_matrix(labels: np.ndarray, K: int) -> np.ndarray:
    """
    Row-stochastic transition matrix P(j | i).
    """
    C = transition_counts(labels, K).astype(float)
    row_sums = C.sum(axis=1, keepdims=True)
    P = np.zeros_like(C, dtype=float)
    np.divide(C, row_sums, out=P, where=row_sums > 0)
    return P

def num_transitions(labels: np.ndarray) -> int:
    """
    Number of state changes (i.e., i_t != i_{t-1}).
    """
    labels = np.asarray(labels, dtype=int)
    if labels.size < 2:
        return 0
    return int(np.count_nonzero(np.diff(labels) != 0))

def switch_rate(labels: np.ndarray) -> float:
    """
    Proportion of steps that are transitions: num_transitions / (T-1).
    """
    labels = np.asarray(labels, dtype=int)
    T = labels.size
    return num_transitions(labels) / max(T - 1, 1)

def state_plots(states=None, state_tc=None, summary=None, sub_ids=None, figsize=None):
    from matplotlib import pyplot as plt

    if states is not None:
        # Plot states
        fig, ax = plt.subplots(1, states.shape[2], figsize=figsize)
        for i in range(states.shape[2]):
            ax[i].imshow(states[:,:,i], cmap="coolwarm")
            ax[i].set_title(f"State {i+1}")
            ax[i].axis("off")

    elif state_tc is not None:
        # Plot state time courses
        fig, ax = plt.subplots(state_tc.shape[0], 1, figsize=figsize)
        
        if state_tc.shape[0] == 1:
            ax = [ax]

        for i in range(state_tc.shape[0]):
            ax[i].plot(state_tc[i,:])
            
            if sub_ids is not None:
                ax[i].set(title=f"Time course for sub {sub_ids[i]}", xlabel="Time", ylabel="State")
            else:
                ax[i].set(title=f"Time course for sub {i+1}", xlabel="Time", ylabel="State")
            
            K = int(state_tc.max() + 1)
            ax[i].set_yticks(range(K))
            ax[i].set_yticklabels([str(k + 1) for k in range(K)])
    
    elif summary is not None:
        fo_mean = summary["fractional_occupancy"].mean(axis=0)
        fo_std = summary["fractional_occupancy"].std(axis=0)
        trans_mean = summary["transitions"].mean(axis=0)
        K = summary["fractional_occupancy"].shape[-1]

        fig, ax = plt.subplots(1,2, figsize=figsize)

        # Group fo (mean ± sd)
        has_std = fo_std is not None and np.any(fo_std != 0)
        if has_std:
            yerr_lower = np.minimum(fo_std, fo_mean)
            yerr_upper = fo_std
            yerr = [yerr_lower, yerr_upper]
            title_suffix = "(group mean ± sd)"
        else:
            yerr = None
            title_suffix = ""
                
        ax[0].bar(range(K), fo_mean, yerr=yerr, capsize=3)
        ax[0].set_xticks(range(K), [f"S{k}" for k in range(1,K+1)])
        ax[0].set_ylabel("%")
        ax[0].set_title(f"Fractional occupancy {title_suffix}")
        ax[0].set_ylim(bottom=0)

        # Mean transition matrix
        im = ax[1].imshow(trans_mean, interpolation='nearest', aspect='auto')
        fig.colorbar(im, ax=ax[1], label="P(next | current)")
        ax[1].set_xticks(range(K), [f"S{k}" for k in range(1,K+1)])
        ax[1].set_yticks(range(K), [f"S{k}" for k in range(1,K+1)])
        ax[1].set_title("Transition matrix (group mean)")

    plt.tight_layout()
    return fig, ax
