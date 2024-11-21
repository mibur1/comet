import os
import re
import mat73
import pickle
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
        except:
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

    else:
        raise ValueError("Unsupported file format")

    if rois is not None:
        return data, rois
    else:
        return data

def load_example(ftype=None):
    """
    Load simulation time series with two randomly changing connectivity states.

    Parameters
    ----------
    ftype : str, optional
        File type to load. If specified as "pkl", a .pkl file with additional
        information is loaded. Otherwise, only time series data is returned.
        Default is None.

    Returns
    -------
    data : np.ndarray or tuple
        If `ftype` is not specified or is None, the function returns a
        TxP np.ndarray containing the time series data

        If `ftype` is "pkl", the function returns a tuple containing:
         - data[0] : TxP np.ndarray
           Time series data.
         - data[1] : np.ndarray
           Time in seconds.
         - data[2] : np.ndarray
           Trial onsets in seconds.
         - data[3] : np.ndarray
           Trial labels indicating two changing connectivity states.
    """

    if ftype == "pkl":
        with importlib_resources.path("comet.data", "simulation.pkl") as file_path:
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
    else:
        with importlib_resources.path("comet.data", "simulation.txt") as file_path:
            data = np.loadtxt(file_path)

    return data

def load_single_state():
    """
    Load simulated time series data with a single connectivity state

    Returns
    -------
    data : TxP np.ndarray
        Single state time series data
    """

    with importlib_resources.path("comet.data", "single_state.txt") as file_path:
        data = np.loadtxt(file_path)

    return data

def save_universe_results(data, universe=os.path.abspath(__file__)):
    """
    This saves the results of a universe.

    If it is a single value, it will be saved in the summary .csv file.
    In any other case the results will be saved in a universe specific .pkl file.

    Parameters
    ----------
    data : any
        Data to save as .pkl file
    universe : str
        File name of the calling script (universe)
    """
    if type(data) is not dict:
        raise ValueError("Data must be povided as a dictionary.")

    calling_script_dir = os.path.dirname(universe)

    # A bit of regex to get the universe number from the filename
    match = re.search(r'universe_(\d+).py', universe)
    universe_number = int(match.group(1))

    savedir = calling_script_dir + "/results"
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # Save the data as a .pkl file
    file = savedir + f"/universe_{universe_number}.pkl"
    with open(file, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    return

def clean(time_series, runs=None, detrend=False, confounds=None, standardize=False, standardize_confounds=True, \
          filter='butterworth', low_pass=None, high_pass=None, t_r=None, ensure_finite=False):
    """
    Wrapper function for nilearn.clean() for cleaning time series data

    Parameters
    ----------
    time_series : TxP np.ndarray
        time series data

    runs : np.ndarray, optional
        Add a run level to the cleaning process. Each run will be cleaned independently.
        Must be a 1D array of n_samples elements.

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

