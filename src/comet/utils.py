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
    Load simulation time series with two randomly changing connectivity states.

    Parameters
    ----------
    fname : str, optional
        File name for any of the included data
        - 'time_series.txt':          Parcellated BOLD time series data for one subject
        - 'time_series_multiple.npy': Parcellated BOLD time series data for 5 subjects
        - 'simulation.txt':           Simulated time series data from the preprint
        - 'simulation.pkl':           Simulated time series data from the preprint + parameters
        Default is 'time_series.txt'.

    Returns
    -------
    data : np.ndarray
       TxP np.ndarray containing the time series data

    """
    with importlib_resources.path("comet.data", fname) as file_path:
        # Handle different file formats
        if fname.endswith(".pkl"):
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
        elif fname.endswith(".npy"):
            data = np.load(file_path)
        elif fname.endswith(".txt"):
            data = np.loadtxt(file_path)
        else:
            print("Error: Unsupported file format")

    return data

def load_testdata(data=None):
    """
    Load test data for unit tests.
    """
    if data == "data":
        pass
    elif data == "connectivity":
        fname = "testdata_connectivity.mat"
    elif data == "graph":
        fname = "testdata_graph.mat"
    elif data == "multiverse":
        pass
    else:
        raise ValueError("Valid test names are: 'graph', 'connectivity', 'multiverse', or 'data'.")
    
    with importlib_resources.path("comet.data", fname) as file_path:
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

    # Get the directory and universe name of the calling script
    caller_stack = inspect.stack()
    universe_fname = caller_stack[1].filename
    calling_script_dir = os.path.dirname(universe_fname)
    match = re.search(r'universe_(\d+).py', universe_fname) # get universe number
    universe_number = int(match.group(1))

    savedir = os.path.join(calling_script_dir, "temp")
    os.makedirs(savedir, exist_ok=True)

    # Save the data as a .pkl file
    file = savedir + f"/universe_{universe_number}.pkl"
    with open(file, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    return

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
def kmeans_cluster(self, dfc, num_states: int = 5):
    from sklearn.cluster import KMeans
    
    if not len(dfc.shape) in [3,4]:
        raise ValueError("Connectivity estimates must be a 3D array with shape (P, P, T) or a 4D array with shape (S,P,P,T).")
    
    # Extract lower triangle and reshape to (T, M)
    mask = np.tril(self.dfc[:, :, 0], k=-1) != 0
    dfc_triu = np.array([matrix_2d[mask] for matrix_2d in self.dfc.transpose(2, 0, 1)])

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_states, random_state=42)
    state_tc = kmeans.fit_predict(dfc_triu)
    centers = kmeans.cluster_centers_  # shape: (K, M)
    
    # Convert vectorized centroids back to full symmetric matrices
    def vec_to_sym(vec, P, mask):
        mat = np.zeros((P, P), dtype=vec.dtype)
        mat[mask] = vec
        mat = mat + mat.T 
        return mat
    
    states = np.array([vec_to_sym(c, self.P, mask) for c in centers])  # (K, P, P)
    
    # Get inertia for elbow method
    inertia = kmeans.inertia_

    # Set class attributes
    state_tc = state_tc[np.newaxis, :]
    states = np.transpose(states, (1, 2, 0))

    return state_tc, states, inertia
    
def summarise_state_tc(state_tc):
    """
    Summarise dwell times & transition matrices across subjects.

    Parameters
    ----------
    state_tc : (n_subjects, T) integer array of labels per subject..

    Returns
    -------
    dict with:
        'dwell_per_subj' : (S, K)
        'dwell_mean'     : (K,)
        'dwell_std'      : (K,)
        'trans_per_subj' : (S, K, K)
        'trans_mean'     : (K, K)
    """
    state_tc = np.asarray(state_tc)
    if state_tc.ndim != 2:
        raise ValueError("state_tc must be a 2D array of shape (n_subjects, T).")
    
    S, _ = state_tc.shape
    K = state_tc.max() + 1

    dwell = np.zeros((S, K), dtype=float)
    trans = np.zeros((S, K, K), dtype=float)
    
    for s in range(S):
        stc = state_tc[s]
        dwell[s] = dwell_times(stc, K)
        trans[s] = transition_matrix(stc, K)
    
    return {
        "dwell_per_subj": dwell,
        "dwell_mean": dwell.mean(axis=0),
        "dwell_std": dwell.std(axis=0),
        "trans_per_subj": trans,
        "trans_mean": trans.mean(axis=0),
    }

def dwell_times(labels: np.ndarray, K: int) -> np.ndarray:
    """
    Calculate the dwell times (fraction of time spent) in each state.

    Parameters
    ----------
    labels : (T,) integer array in [0..K-1].
    K : int
        Number of states.

    Returns
    -------
    (K,) float array
        Dwell times for each state.
    """
    return np.array([(labels == k).sum() / labels.size for k in range(K)], dtype=float)

def transition_matrix(labels: np.ndarray, K: int) -> np.ndarray:
    """
    Row-stochastic transition matrix P(next | current).

    Parameters
    ----------
    labels : (T,) integer array in [0..K-1]
    K : int

    Returns
    -------
    (K, K) float array
        Transition matrix.
    """
    # Count the transitions
    labels = labels.astype(int)
    a, b = labels[:-1], labels[1:]
    M = np.zeros((K, K), dtype=float)
    np.add.at(M, (a, b), 1)
    
    # Get row sums and normalize the probabilities
    row_sums = M.sum(axis=1, keepdims=True)
    P = np.zeros_like(M)
    np.divide(M, row_sums, out=P, where=row_sums > 0)
    return P

def state_plots(states=None, state_tc=None, summary=None, sub_ids=None, figsize=None):
    from matplotlib import pyplot as plt

    if states is not None:
        # Plot states
        fig, ax = plt.subplots(1, states.shape[2], figsize=figsize)
        for i in range(states.shape[2]):
            ax[i].imshow(states[:,:,i])
            ax[i].set_title(f"State {i+1}")
            ax[i].axis("off")

    elif state_tc is not None:
        # Plot state time courses
        fig, ax = plt.subplots(state_tc.shape[0], 1, figsize=figsize)
        for i in range(state_tc.shape[0]):
            ax[i].plot(state_tc[i,:])
            
            if sub_ids is not None:
                ax[i].set(title=f"Time course for sub {sub_ids[i]}", xlabel="Timepoint", ylabel="State")
            else:
                ax[i].set(title=f"Time course for sub {i+1}", xlabel="Timepoint", ylabel="State")
            
            K = int(state_tc.max() + 1)
            ax[i].set_yticks(range(K))
            ax[i].set_yticklabels([str(k + 1) for k in range(K)])
    
    elif summary is not None:
        dwell_mean = summary["dwell_mean"]
        dwell_std = summary["dwell_std"]
        trans_mean = summary["trans_mean"]
        K = summary["dwell_per_subj"].shape[-1]

        fig, ax = plt.subplots(1,2, figsize=figsize)

        # Group dwell times (mean ± sd)
        ax[0].bar(range(K), dwell_mean, yerr=dwell_std, capsize=3)
        ax[0].set_xticks(range(K), [f"S{k}" for k in range(1,K+1)])
        ax[0].set_ylabel("Dwell time (fraction)")
        ax[0].set_title("Dwell times (group mean ± sd)")

        # Mean transition matrix
        im = ax[1].imshow(trans_mean, interpolation='nearest', aspect='auto')
        fig.colorbar(im, ax=ax[1], label="P(next | current)")
        ax[1].set_xticks(range(K), [f"S{k}" for k in range(1,K+1)])
        ax[1].set_yticks(range(K), [f"S{k}" for k in range(1,K+1)])
        ax[1].set_title("Transition matrix (group mean)")

    plt.tight_layout()
    return fig, ax
