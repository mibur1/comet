import os
import re
import sys
import pickle
import numpy as np
import pandas as pd
from nilearn import signal
from scipy.io import loadmat
import importlib.resources as pkg_resources

from .methods import *
from .multiverse import in_notebook

def load_timeseries(path=None, rois=None):
    """
    Load time series data from a file
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
        data = loadmat(path)
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

def load_example(type=None):
    """
    Load simulated time series data with two randomly changing connectivity states
    """
    if type == "pkl":
        with pkg_resources.path("comet.resources", "simulation.pkl") as file_path:
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
    else:
        with pkg_resources.path("comet.resources", "simulation.txt") as file_path:
            data = np.loadtxt(file_path)
    
    return data

def load_single_state():
    """
    Load simulated time series data with a single connectivity state
    """
    with pkg_resources.path("comet.resources", "single_state.txt") as file_path:
        data = np.loadtxt(file_path)
    
    return data

def save_results(data=None, universe=None):
    """
    Save all kinds of results as .pkl file
    """
    calling_script_dir = os.getcwd() if in_notebook else os.path.dirname(sys.path[0])
    
    # A bit of regex to get the universe number from the filename
    match = re.search(r'universe_(\d+).py', universe)
    universe_number = int(match.group(1))

    savedir = calling_script_dir + "/universes/results"
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    # Save as pkl file
    filepath = savedir + f"/universe_{universe_number}.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def clean(time_series, runs=None, detrend=False, confounds=None, standardize=False, standardize_confounds=True, filter='butterworth', low_pass=None, high_pass=None, t_r=0.72, ensure_finite=False):
    """
    Standard nilearn cleaning of the time series
    """
    return signal.clean(time_series, detrend=detrend, confounds=confounds, standardize=standardize, standardize_confounds=standardize_confounds, filter=filter, low_pass=low_pass, high_pass=high_pass, t_r=t_r, ensure_finite=ensure_finite)
