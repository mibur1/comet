import os
import re
import sys
import shutil
import pickle
import numpy as np
from nilearn import signal
from .methods import *
from .multiverse import in_notebook
import importlib.resources as pkg_resources

def load_example(type=None):
    """
    Load simulated time series data
    """
    if type == "pkl":
        with pkg_resources.path("comet.resources", "simulation.pkl") as file_path:
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
    else:
        with pkg_resources.path("comet.resources", "simulation.txt") as file_path:
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

    # Copy multiverse summary to results folder for convenience
    source_file = os.path.join(calling_script_dir, "universes/multiverse_summary.csv")
    destination_file = os.path.join(savedir, "multiverse_summary.csv")
    shutil.copy(source_file, destination_file)


def clean(time_series, runs=None, detrend=False, confounds=None, standardize=False, standardize_confounds=True, filter='butterworth', low_pass=None, high_pass=None, t_r=0.72, ensure_finite=False):
    """
    Standard nilearn cleaning of the time series
    """
    return signal.clean(time_series, detrend=detrend, confounds=confounds, standardize=standardize, standardize_confounds=standardize_confounds, filter=filter, low_pass=low_pass, high_pass=high_pass, t_r=t_r, ensure_finite=ensure_finite) 
