import os
import re
import ast
import sys
import copy
import json
import pickle
import inspect
import tempfile
import numpy as np
import pandas as pd

import mat73
import importlib_resources
from scipy.io import loadmat, savemat
from dataclasses import dataclass, field
from importlib import util
from typing import Any, Dict, get_type_hints, get_origin, Literal, Optional

# fMRI data related imports
import nibabel as nib
from bids import BIDSLayout
from nilearn import datasets, maskers
from nilearn.interfaces.fmriprep import load_confounds

# Plotting imports
from matplotlib import pyplot as plt
from matplotlib.image import imread
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec

# Qt imports
import qdarkstyle
from PyQt6.QtCore import Qt, QPoint, QThread, pyqtSignal, QObject, QRegularExpression, QLocale
from PyQt6.QtGui import QEnterEvent, QFontMetrics, QSyntaxHighlighter, QTextCharFormat
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, \
     QSlider, QToolTip, QWidget, QLabel, QFileDialog, QComboBox, QLineEdit, QSizePolicy, QGridLayout, \
     QSpacerItem, QCheckBox, QTabWidget, QSpinBox, QDoubleSpinBox, QTextEdit, QMessageBox, QGroupBox

# Comet imports
from . import cifti, connectivity, graph, multiverse, utils

"""
Helper classes for the GUI
"""
class Worker(QObject):
    '''
    Worker class to execute functions in a separate thread to keep the GUI responsive.

    Parameters
    ----------
    func : function
        The function to be executed in the worker thread.

    params : dict
        A dictionary of parameters to pass to the function.

    Signals
    -------
    result : pyqtSignal(object)
        Emitted when the function successfully completes

    error : pyqtSignal(str)
        Emitted if the function raises an exception

    finished : pyqtSignal()
        Emitted when the function execution is complete

    Methods
    -------
    run()
        Executes the function with the provided parameters in a try-except block and emits the corresponding signals.
    '''
    result = pyqtSignal(object)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, func, params):
        super().__init__()
        self.func = func
        self.params = params

    def run(self):
        try:
            result = self.func(**self.params)  # Pass parameters as keyword arguments
            self.result.emit(result)  # Emit results
        except Exception as e:
            self.error.emit(str(e))  # Emit errors
        finally:
            self.finished.emit()  # Notify completion

class InfoButton(QPushButton):
    '''
    Simple class to create an info button that shows a tooltip when hovered over.
    '''
    def __init__(self, info_text, parent=None):
        super().__init__("i", parent)
        self.info_text = info_text
        self.setStyleSheet("QPushButton { border: 1px solid black;}")
        self.setFixedSize(20, 20)

    def enterEvent(self, event: QEnterEvent):
        tooltip_pos = self.mapToGlobal(QPoint(self.width(), 0)) # Tooltip position can be adjusted here
        QToolTip.showText(tooltip_pos, self.info_text)
        super().enterEvent(event)

class CustomSpinBox(QSpinBox):
    '''
    Subclass of QSpinBox to allow for a special value "all" to be selected.
    Used for the number of components in the CompCor method and for plotting.
    '''
    def __init__(self, special_value="all", min=0, max=100, parent=None):
        super(CustomSpinBox, self).__init__(parent)
        self.special_value = special_value
        self.special_selected = False
        self.setRange(min, max)
        self.setValue(min)
        self.setSpecialValueText(str(self.special_value))

    def get_value(self):
        return self.special_value if self.special_selected else super(CustomSpinBox, self).value()

class CustomDoubleSpinbox(QDoubleSpinBox):
    '''
    Subclass of QSpinBox to allow for a special value None to be selected.
    Used when cleaning nifti/cifti files and for plotting.
    '''
    def __init__(self, special_value=None, min=-0.1, max=999.0, parent=None):
        super(CustomDoubleSpinbox, self).__init__(parent)
        self.special_value = special_value
        self.special_selected = False
        self.setRange(min, max)
        self.setValue(min)
        self.setSpecialValueText(str(self.special_value))

    def get_value(self):
        return self.special_value if self.special_selected else super(CustomDoubleSpinbox, self).value()

class CustomListSpinbox(QDoubleSpinBox):
    '''
    Subclass of QDoubleSpinBox to allow for a special value (e.g., None) to be selected.
    Used when cleaning nifti/cifti files and for plotting.
    '''
    def __init__(self, special_value=None, min=-0.1, max=999.0, parent=None, values=[1]):
        super(CustomListSpinbox, self).__init__(parent)
        self._user_special_value = special_value
        self.special_value_numeric = min if special_value is None else special_value
        self.special_selected = False
        self.setRange(min, max)
        self.setValue(self.special_value_numeric)
        self.setSpecialValueText(str(special_value))
        self.values = [self.special_value_numeric] + values

    def stepBy(self, steps):
        current_value = self.value()
        try:
            index = self.values.index(current_value)
        except ValueError:
            index = min(range(len(self.values)),
                        key=lambda i: abs(self.values[i] - current_value))
        
        new_index = index + steps
        new_index = max(0, min(new_index, len(self.values) - 1))
        new_value = self.values[new_index]
        self.setValue(new_value)
        self.special_selected = (new_value == self.special_value_numeric)

    def get_value(self):
        return self._user_special_value if self.special_selected else super(CustomListSpinbox, self).value()
    
class ParameterOptions:
    '''
    Parameters for the GUI
    '''
    PARAM_NAMES = {
        "self":                 "self",
        "time_series":          "Time series",
        "windowsize":           "Window size",
        "stepsize":             "Step size",
        "shape":                "Window shape",
        "std":                  "Window sigma",
        "diagonal":             "Main diagonal",
        "fisher_z":             "Fisher z-transform",
        "num_cores":            "Number of CPU cores",
        "standardizeData":      "Z-score data",
        "mu":                   "Weighting parameter μ",
        "flip_eigenvectors":    "Flip eigenvectors",
        "crp":                  "Cosine of rel. phase",
        "pcoh":                 "Phase coherence",
        "teneto":               "Teneto implementation",
        "dist":                 "Distance function",
        "weighted":             "Weighted average",
        "TR":                   "Repetition Time",
        "fmin":                 "Minimum frequency",
        "fmax":                 "Maximum frequency",
        "n_scales":             "Number of scales",
        "drop_scales":          "Drop n scales",
        "drop_timepoints":      "Drop n timepoints",
        "standardize":          "Z-score connectivity",
        "tril":                 "Extract lower triangle",
        "method":               "Specific method",
        "params":               "Various parameters",
        "coi_correction":       "COI correction",
        "clstr_distance":       "Distance metric",
        "num_bins":             "Number of bins",
        "n_overlap":            "Window overlap",
        "tapered_window":       "Tapered window",
        "n_states":             "Number of states",
        "subject_clusters":     "Number subject-level clusters",
        "normalization":        "Normalization",
        "clstr_distance":       "Distance measure",
        "subject":              "Subject",
        "iterations":           "Iterations",
        "sw_method":            "Sliding window",
        "dhmm_obs_state_ratio": "State ratio",
        "state_ratio":          "State ratio",
        "vlim":                 "Color axis limit",
        "hmm_iter":             "HMM iterations",
        "parcellation":         "Parcellation"
    }

    CONNECTIVITY_METHODS = {
        'SlidingWindow':                'CONT Sliding Window',
        'Jackknife':                    'CONT Jackknife Correlation',
        'FlexibleLeastSquares':         'CONT Flexible Least Squares',
        'SpatialDistance':              'CONT Spatial Distance',
        'TemporalDerivatives':          'CONT Multiplication of Temporal Derivatives',
        'DCC':                          'CONT Dynamic Conditional Correlation',
        'PhaseSynchrony':               'CONT Phase Synchronization',
        'LeiDA':                        'CONT Leading Eigenvector Dynamics',
        'WaveletCoherence':             'CONT Wavelet Coherence',
        'EdgeConnectivity':             'CONT Edge-centric Connectivity',
        'KSVD':                         'STATE K-SVD',
        'CoactivationPatterns':         'STATE Co-activation Patterns',
        'SlidingWindowClustering':      'STATE Sliding Window Clustering',
        'DiscreteHMM':                  'STATE Discrete Hidden Markov Model',
        'ContinuousHMM':                'STATE Continuous Hidden Markov Model',
        'Static_Pearson':               'STATIC Pearson Correlation',
        'Static_Partial':               'STATIC Partial Correlation',
        'Static_Mutual_Info':           'STATIC Mutual Information'
    }

    GRAPH_OPTIONS = {
        "handle_negative_weights":      "PREP Negative weights",
        "threshold":                    "PREP Threshold",
        "binarise":                     "PREP Binarise",
        "normalise":                    "PREP Normalise",
        "invert":                       "PREP Invert",
        "logtransform":                 "PREP Log-transform",
        "symmetrise":                   "PREP Symmetrise",
        "randomise":                    "PREP Randomise",
        "postproc":                     "PREP Post-processing",
        "efficiency":                   "COMET Efficiency",
        "matching_ind_und":             "COMET Matching index",
        "small_world_propensity":       "COMET Small world propensity",
        "backbone_wu":                  "BCT Backbone (weighted)",
        "betweenness":                  "BCT Betweenness centrality",
        "clustering_coef":              "BCT Clustering coefficient",
        "degrees_und":                  "BCT Degrees",
        "density_und":                  "BCT Density",
        "eigenvector_centrality_und":   "BCT Eigenvector centrality",
        "gateway_coef_sign":            "BCT Gateway coefficient (sign)",
        "pagerank_centrality":          "BCT Pagerank centrality",
        "participation_coef":           "BCT Participation coef",
        "participation_coef_sign":      "BCT Participation coef (sign)",
    }

    # Reverse mappings
    REVERSE_PARAM_NAMES = {v: k for k, v in PARAM_NAMES.items()}
    REVERSE_CONNECTIVITY_METHODS = {v: k for k, v in CONNECTIVITY_METHODS.items()}
    REVERSE_GRAPH_OPTIONS = {v: k for k, v in GRAPH_OPTIONS.items()}

    ATLAS_OPTIONS = {
        "AAL template (3v2)":       ["166"],
        "BASC multiscale":          ["7", "12", "20", "36", "64", "122", "197", "325", "444"],
        "Destrieux et al. (2009)":  ["148"],
        "Pauli et al. (2017)":      ["deterministic"],
        "Schaefer et al. (2018)":   ["100", "200", "300", "400", "500", "600", "700", "800", "900", "1000"],
        "Talairach atlas":          ["hemisphere"],
        "Yeo (2011) networks":      ["thin_7", "thick_7", "thin_17", "thick_17"],
        "Dosenbach et al. (2010)":  ["160"],
        "Power et al. (2011)":      ["264"],
        "Seitzmann et al. (2018)":  ["300"]
    }

    ATLAS_OPTIONS_CIFTI = {
        "Glasser MMP":              ["379"],
        "Schaefer Kong":            ["100", "200", "300", "400", "500", "600", "700", "800", "900", "1000"],
        "Schaefer Tian":            ["154", "254", "354", "454", "554", "654", "754", "854", "954", "1054"]
    }

    ATLAS_MAP = {
                "Schaefer Kong 100": "schaefer_100_cortical",
                "Schaefer Kong 200": "schaefer_200_cortical",
                "Schaefer Kong 300": "schaefer_300_cortical",
                "Schaefer Kong 400": "schaefer_400_cortical",
                "Schaefer Kong 500": "schaefer_500_cortical",
                "Schaefer Kong 600": "schaefer_600_cortical",
                "Schaefer Kong 700": "schaefer_700_cortical",
                "Schaefer Kong 800": "schaefer_800_cortical",
                "Schaefer Kong 900": "schaefer_900_cortical",
                "Schaefer Kong 1000": "schaefer_1000_cortical",

                "Schaefer Tian 154": "schaefer_100_subcortical",
                "Schaefer Tian 254": "schaefer_200_subcortical",
                "Schaefer Tian 354": "schaefer_300_subcortical",
                "Schaefer Tian 454": "schaefer_400_subcortical",
                "Schaefer Tian 554": "schaefer_500_subcortical",
                "Schaefer Tian 654": "schaefer_600_subcortical",
                "Schaefer Tian 754": "schaefer_700_subcortical",
                "Schaefer Tian 854": "schaefer_800_subcortical",
                "Schaefer Tian 954": "schaefer_900_subcortical",
                "Schaefer Tian 1054": "schaefer_1000_subcortical",

                "Glasser MMP 379": "glasser_mmp_subcortical",
            }

    INFO_OPTIONS = {
        "windowsize":               "Size of the window used by the method. Should typically be an uneven number to have a center.",
        "shape":                    "Shape of the windowing function.",
        "std":                      "Width (sigma) of the window.",
        "diagonal":                 "Values for the main diagonal of the connectivity matrix.",
        "fisher_z":                 "Fisher z-transform the connectivity values.",
        "num_cores":                "Parallelize on multiple cores (highly recommended for DCC and FLS).",
        "standardizeData":          "z-standardize the time series data.",
        "mu":                       "Weighting parameter for FLS. Smaller values will produce more erratic changes in connectivity estimate.",
        "flip_eigenvectors":        "Flips the sign of the eigenvectors.",
        "dist":                     "Distance function",
        "TR":                       "Repetition time of the data (in seconds)",
        "fmin":                     "Minimum wavelet frequency",
        "fmax":                     "Maximum wavelet frequency",
        "n_scales":                 "Number of wavelet scales",
        "drop_scales":              "Drop the n largest and smallest scales to account for the cone of influence",
        "drop_timepoints":          "Drop n first and last time points from the time series to account for the cone of influence",
        "method":                   "Specific implementation of the method",
        "params":                   "Various parameters",
        "coi_correction":           "Cone of influence correction",
        "clstr_distance":           "Distance metric",
        "num_bins":                 "Number of bins for discretization",
        "method":                   "Specific type of method",
        "n_overlap":                "Window overlap",
        "tapered_window":           "Tapered window",
        "n_states":                 "Number of states",
        "n_subj_clusters":          "Number of subjects",
        "normalization":            "Normalization",
        "subject":                  "Subject",
        "Base measure":             "Base measure for the clustering",
        "Iterations":               "Number of iterations",
        "Sliding window":           "Sliding window method",
        "State ratio":              "Observation/state ratio for the DHMM",
        "vlim":                     "Limit for color axis (for eTS visualization only)"
    }

    CONFOUND_OPTIONS = {
            "Cleaning\nstrategy": ["motion", "wm_csf", "compcor", "global_signal", "scrub", "demean", "high_pass", "ica_aroma"],
            "motion": ["full", "basic", "power2", "derivatives"],
            "wm_csf": ["basic", "power2", "derivatives", "full"],
            "compcor": ["anat_combined", "anat_separated", "temporal", "temporal_anat_combined", "temporal_anat_separated"],
            "n_compcor": ["all"],
            "global_signal": ["basic", "power2", "derivatives", "full"],
            "ica_aroma": ["full", "basic"],
            "scrub": 5,
            "fd_threshold": 0.5,
            "std_dvars_threshold": 1.5,
        }

    CLEANING_INFO = {
        "motion": "Type of confounds extracted from head motion estimates\n\
            - basic: translation/rotation (6 parameters)\n\
            - power2: translation/rotation + quadratic terms (12 parameters)\n\
            - derivatives: translation/rotation + derivatives (12 parameters)\n\
            - full: translation/rotation + derivatives + quadratic terms + power2d derivatives (24 parameters)",
        "wm_csf": "Type of confounds extracted from masks of white matter and cerebrospinal fluids\n\
            - basic: the averages in each mask (2 parameters)\n\
            - power2: averages and quadratic terms (4 parameters)\n\
            - derivatives: averages and derivatives (4 parameters)\n\
            - full: averages + derivatives + quadratic terms + power2d derivatives (8 parameters)",
        "compcor": "Type of confounds extracted from a component based noise correction method\n\
            - anat_combined: noise components calculated using a white matter and CSF combined anatomical mask\n\
            - anat_separated: noise components calculated using white matter mask and CSF mask compcor separately; two sets of scores are concatenated\n\
            - temporal: noise components calculated using temporal compcor\n\
            - temporal_anat_combined: components of temporal and anat_combined\n\
            - temporal_anat_separated:  components of temporal and anat_separated",
        "n_compcor": "The number of noise components to be extracted.\n\
            - acompcor_combined=False, and/or compcor=full: the number of components per mask.\n\
            - all: all components (50% variance explained by fMRIPrep defaults)",
        "global_signal": "Type of confounds xtracted from the global signal\n\
            - basic: just the global signal (1 parameter)\n\
            - power2: global signal and quadratic term (2 parameters)\n\
            - derivatives: global signal and derivative (2 parameters)\n\
            - full: global signal + derivatives + quadratic terms + power2d derivatives (4 parameters)",
        "ica_aroma": "ICA-AROMA denoising\n\
            - full: use fMRIPrep output ~desc-smoothAROMAnonaggr_bold.nii.gz\n\
            - basic use noise independent components only.",
        "scrub": "Lenght of segment to remove around time frames with excessive motion.",
        "fd_threshold": "Framewise displacement threshold for scrub in mm.",
        "std_dvars_threshold": "Standardized DVARS threshold for scrub.\n\
            DVARS is the root mean squared intensity difference of volume N to volume N+1"
    }

    def __init__(self):
        self.param_names = self.PARAM_NAMES
        self.reverse_param_names = self.REVERSE_PARAM_NAMES
        self.connectivityMethods = self.CONNECTIVITY_METHODS
        self.reverse_connectivityMethods = self.REVERSE_CONNECTIVITY_METHODS
        self.graphOptions = self.GRAPH_OPTIONS
        self.reverse_graphOptions = self.REVERSE_GRAPH_OPTIONS
        self.atlas_options = self.ATLAS_OPTIONS
        self.atlas_options_cifti = self.ATLAS_OPTIONS_CIFTI
        self.atlas_map = self.ATLAS_MAP
        self.info_options = self.INFO_OPTIONS
        self.confound_options = self.CONFOUND_OPTIONS
        self.cleaning_info = self.CLEANING_INFO

class PythonHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super(PythonHighlighter, self).__init__(parent)

        # Define the formats for different types of syntax elements
        self.keywordFormat = QTextCharFormat()
        self.keywordFormat.setForeground(Qt.GlobalColor.darkMagenta)

        self.classFormat = QTextCharFormat()
        self.classFormat.setForeground(Qt.GlobalColor.darkMagenta)

        self.singleLineCommentFormat = QTextCharFormat()
        self.singleLineCommentFormat.setForeground(Qt.GlobalColor.darkGreen)
        self.singleLineCommentFormat.setFontItalic(True)

        self.stringFormat = QTextCharFormat()
        self.stringFormat.setForeground(Qt.GlobalColor.darkRed)

        self.functionFormat = QTextCharFormat()
        self.functionFormat.setForeground(Qt.GlobalColor.darkCyan)

        self.tripleQuoteFormat = QTextCharFormat()
        self.tripleQuoteFormat.setForeground(Qt.GlobalColor.darkGreen)

        # Create the rules for Python syntax
        self.rules = []

        keywords = [
            'and', 'as', 'assert', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except',
            'False', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'None',
            'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'True', 'try', 'while', 'with', 'yield'
        ]
        keywordPatterns = [f"\\b{keyword}\\b" for keyword in keywords]
        self.rules += [(QRegularExpression(pattern), self.keywordFormat) for pattern in keywordPatterns]

        classPattern = QRegularExpression("\\bQ[A-Za-z]+\\b")
        self.rules.append((classPattern, self.classFormat))

        stringPattern = QRegularExpression("\".*\"|'.*'")
        self.rules.append((stringPattern, self.stringFormat))

        functionPattern = QRegularExpression("\\b[A-Za-z0-9_]+(?=\\()")
        self.rules.append((functionPattern, self.functionFormat))

        singleLineCommentPattern = QRegularExpression("#[^\n]*")
        self.commentRule = (singleLineCommentPattern, self.singleLineCommentFormat)

    def highlightBlock(self, text):
        # Apply all regular expression rules first, except for comments
        if self.previousBlockState() != 1:
            for pattern, fmt in self.rules:
                matchIterator = pattern.globalMatch(text)
                while matchIterator.hasNext():
                    match = matchIterator.next()
                    start = match.capturedStart()
                    length = match.capturedLength()
                    self.setFormat(start, length, fmt)

        # Check for triple-quoted strings (''' or """)
        startTripleSingle = text.find("'''")
        startTripleDouble = text.find('"""')

        if self.previousBlockState() == 1:
            # We're continuing from a previous block with an open triple-quoted string
            self.setFormat(0, len(text), self.tripleQuoteFormat)
            endTripleSingle = text.find("'''")
            endTripleDouble = text.find('"""')
            if endTripleSingle != -1 or endTripleDouble != -1:
                self.setCurrentBlockState(0)  # End of the triple-quoted string
            else:
                self.setCurrentBlockState(1)  # Continue in the next block
        elif startTripleSingle != -1 or startTripleDouble != -1:
            # Starting a new triple-quoted string
            startTriple = startTripleSingle if startTripleSingle != -1 else startTripleDouble
            endTripleSingle = text.find("'''", startTriple + 3)
            endTripleDouble = text.find('"""', startTriple + 3)

            if endTripleSingle != -1 or endTripleDouble != -1:
                endTriple = endTripleSingle if endTripleSingle != -1 else endTripleDouble
                self.setFormat(startTriple, endTriple + 3 - startTriple, self.tripleQuoteFormat)
                self.setCurrentBlockState(0)  # Triple-quoted string ends in the same block
            else:
                self.setFormat(startTriple, len(text) - startTriple, self.tripleQuoteFormat)
                self.setCurrentBlockState(1)  # Triple-quoted string continues in the next block
        else:
            self.setCurrentBlockState(0)  # No triple-quoted string detected

        # Apply the single-line comment rule last to ensure it overrides other formats
        matchIterator = self.commentRule[0].globalMatch(text)
        while matchIterator.hasNext():
            match = matchIterator.next()
            start = match.capturedStart()
            length = match.capturedLength()
            self.setFormat(start, length, self.commentRule[1])

@dataclass
class Data:
    '''
    Data class which stores all relevant data for the GUI.
    '''
    # File variables
    file_path:     str        = field(default=None)         # data file path
    file_name:     str        = field(default=None)         # data file name
    file_data:     np.ndarray = field(default=None)         # data file data
    ts_data:       np.ndarray = field(default=None)         # (processed) time series data (dFC input)
    sample_mask:   np.ndarray = field(default=None)         # time series mask

    # DFC variables
    dfc_instance:  Any        = field(default=None)         # instance of the dFC class
    dfc_name:      str        = field(default=None)         # method class name
    dfc_params:    Dict       = field(default_factory=dict) # input parameters
    dfc_data:      np.ndarray = field(default=None)         # dfc data
    dfc_states:    np.ndarray = field(default=None)         # dfc states
    dfc_state_tc:  np.ndarray = field(default=None)         # dfc state time course
    dfc_edge_ts:   np.ndarray = field(default=None)         # dfc edge time series
    dfc_edge_rms:  np.ndarray = field(default=None)         # dfc edge time series RMS

    # Graph variables
    graph_file:    str        = field(default=None)         # graph file name
    graph_raw:     np.ndarray = field(default=None)         # raw input data for graph (dFC matrix)
    graph_data:    np.ndarray = field(default=None)         # working data for graph (while processing)
    graph_results: Dict       = field(default_factory=dict) # calculated graph measures

    # Multiverse variables
    multiverse_folder: str   = field(default=None)         # folder for multiverse analysis
    forking_paths: Dict      = field(default_factory=dict) # decision points for multiverse analysis
    invalid_paths: list      = field(default_factory=list) # invalid paths for multiverse analysis

    # Misc variables
    roi_names:     np.ndarray = field(default=None)         # input roi data (for .tsv files)

    def clear_dfc_data(self):
        self.dfc_params   = {}
        self.dfc_data     = None
        self.dfc_states   = None
        self.dfc_state_tc = None
        self.dfc_edge_ts  = None

class DataStorage:
    '''
    Database class for storing data objects

    Methods
    -------
    generate_hash(data_obj)
        Generate a hash based on method_name, file_name, and sorted params

    add_data(data_obj)
        Add data object to the storage

    delete_data(data_obj)
        Delete data object from the storage

    check_for_identical_data(data_obj)
        Check if identical data exists in the storage

    check_previous_data(methodName)
        Get data object of previous calculations
    '''

    def __init__(self):
        self.storage = {}

    def generate_hash(self, data_obj):
        # Generate a hash based on method_name, file_name, and sorted params
        # This hash will be used to check if identical data exists
        hashable_params = {k: v for k, v in data_obj.dfc_params.items() if not isinstance(v, np.ndarray)}
        params_tuple = tuple(sorted(hashable_params.items()))
        return hash((data_obj.file_name, data_obj.dfc_name, params_tuple))

    def add_data(self, data_obj):
        self.delete_data(data_obj) # Delete existing data for the same method

        data_hash = self.generate_hash(data_obj)
        if data_hash not in self.storage:
            self.storage[data_hash] = copy.deepcopy(data_obj) # IMPORTANT: deep copy for a completely new data object
            return True
        return False

    def delete_data(self, data_obj):
        # Identify and delete existing entries with the same dfc_name as data_obj
        keys_to_delete = [key for key, value in self.storage.items() if value.dfc_name == data_obj.dfc_name]
        for key in keys_to_delete:
            del self.storage[key]

    def check_for_identical_data(self, data_obj):
        # Get data and parameters for previously calculated identical data
        data_hash = self.generate_hash(data_obj)
        data = self.storage.get(data_hash, None)
        return copy.deepcopy(data) # IMPORTANT: deep copy

    def check_previous_data(self, methodName):
        # Get data for the last calculation with a given method
        for data_obj in reversed(list(self.storage.values())):
            if data_obj.dfc_name == methodName and data_obj.dfc_data is not None:
                return copy.deepcopy(data_obj) # IMPORTANT: deep copy
        return None


"""
Main class of the GUI
"""
class App(QMainWindow):
    # Initialize the GUI and tabs
    def __init__(self):
        super().__init__()
        self.title = 'Comet Toolbox'

        # Data and data storage
        self.data = Data()
        self.data_storage = DataStorage()

        # Parameter names
        # TODO: integrate in entire script?
        parameterNames = ParameterOptions()
        self.param_names = parameterNames.param_names
        self.reverse_param_names = parameterNames.reverse_param_names
        self.connectivityMethods = parameterNames.connectivityMethods
        self.reverse_connectivityMethods = parameterNames.reverse_connectivityMethods
        self.graphOptions = parameterNames.graphOptions
        self.reverse_graphOptions = parameterNames.reverse_graphOptions
        self.atlas_options = parameterNames.atlas_options
        self.atlas_options_cifti = parameterNames.atlas_options_cifti
        self.atlas_map = parameterNames.atlas_map
        self.info_options = parameterNames.info_options
        self.confound_options = parameterNames.confound_options
        self.cleaning_info = parameterNames.cleaning_info

        # Get all the dFC methods and names
        # TODO: Make better implementation
        self.class_info = {
            obj.name: name  # Map human-readable name to class name
            for name, obj in inspect.getmembers(connectivity)
            if inspect.isclass(obj) and obj.__module__ == connectivity.__name__ and name != "ConnectivityMethod"
        }

        # Init the top-level layout which contains connectivity, graph, and multiverse tabs
        self.setWindowTitle(self.title)
        topLayout = QVBoxLayout()
        self.topTabWidget = QTabWidget()
        topLayout.addWidget(self.topTabWidget)

        # Init the individual tabs
        self.dataTab()
        self.connectivityTab()
        self.graphTab()
        self.multiverseTab()
        self.currentTabIndex = 0
        self.currentGraphTabIndex = 0

        # Set main window layout to the top-level layout
        centralWidget = QWidget()
        centralWidget.setLayout(topLayout)
        self.setCentralWidget(centralWidget)

        return

    def dataTab(self):
        dataTab = QWidget()
        dataLayout = QHBoxLayout()
        dataTab.setLayout(dataLayout)

        # Left section
        leftLayout = QVBoxLayout()
        self.addDataLoadLayout(leftLayout)
        self.addFmriprepLayout(leftLayout)
        self.addMiscDataLayout(leftLayout)

        # Right section
        rightLayout = QVBoxLayout()
        self.addDataPlotLayout(rightLayout)

        # Combine sections
        dataLayout.addLayout(leftLayout, 3)
        dataLayout.addLayout(rightLayout, 4)
        self.topTabWidget.addTab(dataTab, "Data Preparation")

        return

    def connectivityTab(self):
        connectivityTab = QWidget()
        connectivityLayout = QHBoxLayout()
        connectivityTab.setLayout(connectivityLayout)

        # Left section
        leftLayout = QVBoxLayout()
        self.addConnectivityLayout(leftLayout)

        # Right section
        rightLayout = QVBoxLayout()
        self.addConnectivityPlotLayout(rightLayout)

        # Combine sections
        connectivityLayout.addLayout(leftLayout, 3)
        connectivityLayout.addLayout(rightLayout, 4)
        self.topTabWidget.addTab(connectivityTab, "Connectivity Analysis")

        return

    def graphTab(self):
        graphTab = QWidget()
        graphLayout = QVBoxLayout()
        graphTab.setLayout(graphLayout)

        # Left section
        leftLayout = QVBoxLayout()
        self.addGraphLayout(leftLayout)

        # Right section
        rightLayout = QVBoxLayout()
        self.addGraphPlotLayout(rightLayout)

        # Combine sections
        mainLayout = QHBoxLayout()
        mainLayout.addLayout(leftLayout, 3)
        mainLayout.addLayout(rightLayout, 4)
        graphLayout.addLayout(mainLayout)
        self.topTabWidget.addTab(graphTab, "Graph Analysis")

        return

    def multiverseTab(self):
        multiverseTab = QWidget()
        multiverseLayout = QVBoxLayout()
        multiverseTab.setLayout(multiverseLayout)

        # Left section
        leftLayout = QVBoxLayout()
        self.addMultiverseLayout(leftLayout)

        # Right section
        rightLayout = QVBoxLayout()
        self.addMultiversePlotLayout(rightLayout)

        # Combine sections
        mainLayout = QHBoxLayout()
        mainLayout.addLayout(leftLayout, 3)
        mainLayout.addLayout(rightLayout, 4)
        multiverseLayout.addLayout(mainLayout)
        self.topTabWidget.addTab(multiverseTab, "Multiverse Analysis")

        return

    # Data layouts
    def addDataLoadLayout(self, leftLayout):
        """
        Add the layout for loading single files to the left layout of the data tab.
        """
        # Butttons for file loading
        loadLayout = QVBoxLayout()
        buttonLayout = QHBoxLayout()

        fileButton = QPushButton('Load from single file')
        fmriprepButton = QPushButton('Load from fMRIprep outputs')
        self.fileNameLabel = QLabel('No data loaded yet.')

        buttonLayout.addWidget(fileButton)
        buttonLayout.addWidget(fmriprepButton)

        loadLayout.addLayout(buttonLayout)
        loadLayout.addWidget(self.fileNameLabel)

        # Subject dropdown for .npy files containing multiple time series
        self.subjectDropdownContainer = QWidget()
        self.subjectDropdownLayout = QHBoxLayout()
        self.subjectLabel = QLabel("Subject number:")
        self.subjectLabel.setFixedWidth(110)
        self.subjectDropdown = QComboBox()
        self.subjectDropdownLayout.addWidget(self.subjectLabel)
        self.subjectDropdownLayout.addWidget(self.subjectDropdown)
        self.subjectDropdownLayout.setContentsMargins(5, 5, 5, 0)
        self.subjectDropdownContainer.setLayout(self.subjectDropdownLayout)
        self.subjectDropdownContainer.hide()

        # Cleaning container options for nifti/cifti files
        self.cleaningContainer = QWidget()
        cleaningLayout = QVBoxLayout()
        cleaningLayout.setContentsMargins(5, 5, 5, 0)

        # Sphere Layout Container
        self.sphereContainer = QWidget()
        sphereLayout = QHBoxLayout(self.sphereContainer)
        sphereLayoutLabel = QLabel("Sphere radius (mm):")
        self.sphereRadiusSpinbox = CustomDoubleSpinbox(special_value="single voxel", min=0.0, max=20.0)
        self.sphereRadiusSpinbox.setValue(5.0)
        self.overlapCheckbox = QCheckBox("Allow overlap")
        sphereLayout.addWidget(sphereLayoutLabel)
        sphereLayout.addWidget(self.sphereRadiusSpinbox)
        sphereLayout.addWidget(self.overlapCheckbox)
        sphereLayout.addStretch(1)
        sphereLayout.setContentsMargins(0, 0, 0, 0)
        self.sphereContainer.setLayout(sphereLayout)

        # Misc Cleaning Layout Container
        self.miscCleaningContainer = QWidget()
        miscCleaningLayout = QVBoxLayout(self.miscCleaningContainer)
        miscCleaningLayout.setContentsMargins(0, 0, 0, 0)

        firstRowLayout = QHBoxLayout()
        self.detrendCheckbox = QCheckBox("Detrend")
        self.standardizeCheckbox = QCheckBox("Standardize")
        firstRowLayout.addItem(QSpacerItem(5, 0, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum))
        firstRowLayout.addWidget(self.detrendCheckbox)
        firstRowLayout.addItem(QSpacerItem(10, 0, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum))
        firstRowLayout.addWidget(self.standardizeCheckbox)
        firstRowLayout.addStretch(1)

        secondRowLayout = QHBoxLayout()
        self.highVarianceCheckbox = QCheckBox("Regress high variance confounds")
        self.gsrCheckbox = QCheckBox("Regress global signal")
        secondRowLayout.addItem(QSpacerItem(5, 0, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum))
        secondRowLayout.addWidget(self.highVarianceCheckbox)
        secondRowLayout.addItem(QSpacerItem(10, 0, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum))
        secondRowLayout.addWidget(self.gsrCheckbox)
        secondRowLayout.addStretch(1)

        miscCleaningLayout.addLayout(firstRowLayout)
        miscCleaningLayout.addLayout(secondRowLayout)

        self.standardizeCheckbox.stateChanged.connect(self.updateCleaningOptions)
        self.detrendCheckbox.stateChanged.connect(self.updateCleaningOptions)
        self.gsrCheckbox.stateChanged.connect(self.updateCleaningOptions)
        self.highVarianceCheckbox.stateChanged.connect(self.updateCleaningOptions)

        # Smoothing Layout Container
        self.smoothingContainer = QWidget()
        smoothingLayout = QHBoxLayout(self.smoothingContainer)
        smoothingLabel = QLabel("Smoothing fwhm (mm):")
        self.smoothingSpinbox = CustomDoubleSpinbox(special_value=None, min=0.0, max=20.0)
        self.smoothingSpinbox.setDecimals(2)
        self.smoothingSpinbox.setSingleStep(1.0)
        smoothingLayout.addWidget(smoothingLabel)
        smoothingLayout.addWidget(self.smoothingSpinbox)
        smoothingLayout.addStretch(1)
        smoothingLayout.setContentsMargins(0, 0, 0, 0)
        self.smoothingContainer.setLayout(smoothingLayout)

        # Filtering Layout Container
        self.filteringContainer = QWidget()
        filteringLayout = QHBoxLayout(self.filteringContainer)
        highPassLabel = QLabel("High Pass:")
        self.highPassCutoff = CustomDoubleSpinbox(special_value=None, min=0.0, max=1.0)
        self.highPassCutoff.setDecimals(3)
        self.highPassCutoff.setSingleStep(0.01)
        self.highPassCutoff.setSuffix(" Hz")

        lowPassLabel = QLabel("Low Pass:")
        self.lowPassCutoff = CustomDoubleSpinbox(special_value=None, min=0.0, max=1.0)
        self.lowPassCutoff.setDecimals(3)
        self.lowPassCutoff.setSingleStep(0.1)
        self.lowPassCutoff.setSuffix(" Hz")

        trLabel = QLabel("TR:")
        self.trValue = CustomListSpinbox(special_value=None, min=0.0, max=20.0, values=[0.72, 0.8, 1.5, 2.5])
        self.trValue.setSuffix(" s  ")

        filteringLayout.addWidget(highPassLabel)
        filteringLayout.addWidget(self.highPassCutoff)
        filteringLayout.addWidget(lowPassLabel)
        filteringLayout.addWidget(self.lowPassCutoff)
        filteringLayout.addWidget(trLabel)
        filteringLayout.addWidget(self.trValue)
        filteringLayout.addStretch(1)
        filteringLayout.setContentsMargins(0, 0, 0, 0)
        self.filteringContainer.setLayout(filteringLayout)

        # Add containers to the main cleaning layout
        cleaningLayout.addWidget(self.miscCleaningContainer)
        cleaningLayout.addWidget(self.sphereContainer)
        cleaningLayout.addWidget(self.smoothingContainer)
        cleaningLayout.addWidget(self.filteringContainer)

        self.cleaningContainer.setLayout(cleaningLayout)
        self.cleaningContainer.hide()

        # Parcellation dropdown for nifti files
        self.parcellationContainer = QWidget()
        self.parcellationLayout = QHBoxLayout()
        self.parcellationLayout.setContentsMargins(5, 5, 5, 0)

        self.parcellationLabel = QLabel("Parcellation:")
        self.parcellationLabel.setFixedWidth(100)

        self.parcellationOptionsLabel = QLabel("Type:")
        self.parcellationOptionsLabel.setFixedWidth(40)
        self.parcellationOptions = QComboBox()

        self.parcellationDropdown = QComboBox()
        self.parcellationDropdown.addItems(self.atlas_options.keys())
        self.parcellationLayout.addWidget(self.parcellationLabel, 1)
        self.parcellationLayout.addWidget(self.parcellationDropdown, 3)
        self.parcellationLayout.addWidget(self.parcellationOptionsLabel, 1)
        self.parcellationLayout.addWidget(self.parcellationOptions, 2)
        self.parcellationContainer.setLayout(self.parcellationLayout)

        self.parcellationDropdown.currentIndexChanged.connect(self.onAtlasChanged)
        self.parcellationContainer.hide()

        # Discard container
        self.discardContainer = QWidget()
        discardLayout = QHBoxLayout()
        discardLabel = QLabel("Discard initial frames:")
        self.discardSpinBox = CustomSpinBox(None)
        self.discardSpinBox.setSuffix(" frames")
        self.discardSpinBox.setSingleStep(5)
        discardLayout.addWidget(discardLabel)
        discardLayout.addWidget(self.discardSpinBox)
        discardLayout.addStretch(1)
        discardLayout.setContentsMargins(5, 5, 5, 0)
        self.discardContainer.setLayout(discardLayout)

        # Calculate button
        self.niftiExtractButtonContainer = QWidget()
        calculateLayout = QHBoxLayout()
        calculateLayout.setContentsMargins(5, 5, 5, 0)

        self.parcellationCalculateButton = QPushButton("Extract time series")
        self.parcellationCalculateButton.clicked.connect(self.calculateTimeSeries)
        calculateLayout.addStretch(2)
        calculateLayout.addWidget(self.parcellationCalculateButton, 1)
        self.niftiExtractButtonContainer.setLayout(calculateLayout)

        # Cleaning button
        self.niftiCleanButtonContainer = QWidget()
        calculateLayout2 = QHBoxLayout()
        calculateLayout2.setContentsMargins(5, 5, 5, 0)

        self.cleanButton = QPushButton("Clean time series")
        self.cleanButton.clicked.connect(self.cleanTimeSeries)
        self.resetButton = QPushButton("Reset time series")
        self.resetButton.clicked.connect(self.resetTimeSeries)
        self.resetButton.setEnabled(False)
        calculateLayout2.addStretch(2)
        calculateLayout2.addWidget(self.resetButton, 1)
        calculateLayout2.addWidget(self.cleanButton, 1)
        self.niftiCleanButtonContainer.setLayout(calculateLayout2)

        # Transpose checkpox
        self.transposeCheckbox = QCheckBox("Transpose data (time has to be the first dimension).")
        self.transposeCheckbox.hide()
        loadLayout.addWidget(self.transposeCheckbox)
        #loadLayout.addItem(QSpacerItem(0, 15, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))

        # Container for timne series extraction
        self.tsExtractionContainer = QGroupBox("Time series extraction")
        tsExtractionContainerLayout = QVBoxLayout()

        tsExtractionContainerLayout.addWidget(self.subjectDropdownContainer)
        tsExtractionContainerLayout.addWidget(self.cleaningContainer)
        tsExtractionContainerLayout.addWidget(self.parcellationContainer)
        tsExtractionContainerLayout.addWidget(self.discardContainer)
        tsExtractionContainerLayout.addWidget(self.niftiExtractButtonContainer)
        tsExtractionContainerLayout.addWidget(self.niftiCleanButtonContainer)

        # Connect widgets
        self.transposeCheckbox.stateChanged.connect(self.onTransposeChecked)
        fileButton.clicked.connect(self.loadFile)
        fmriprepButton.clicked.connect(self.loadFmriprep)

        # Add file loading layout to the left layout
        leftLayout.addLayout(loadLayout)
        leftLayout.addItem(QSpacerItem(0, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))

        # Add the parcellation container to the left layout
        self.tsExtractionContainer.setLayout(tsExtractionContainerLayout)
        self.tsExtractionContainer.hide()

        leftLayout.addWidget(self.tsExtractionContainer)
        return

    def addFmriprepLayout(self, leftLayout):
        ##########################
        # Main container widget
        self.fmriprepContainer = QWidget()
        self.fmriprepLayout = QVBoxLayout(self.fmriprepContainer)

        ##########################
        # File selection container
        self.fmriprep_fileContainer = QGroupBox("File selection")
        self.fmriprep_fileLayout = QVBoxLayout(self.fmriprep_fileContainer)

        # Subjects Dropdown with Label
        self.fmriprep_subjectDropdownLayout = QHBoxLayout()
        self.fmriprep_subjectLabel = QLabel("Subject:")
        self.fmriprep_subjectLabel.setFixedWidth(90)
        self.fmriprep_subjectDropdown = QComboBox()
        self.fmriprep_subjectDropdownLayout.addWidget(self.fmriprep_subjectLabel, 1)
        self.fmriprep_subjectDropdownLayout.addWidget(self.fmriprep_subjectDropdown, 4)
        self.fmriprep_fileLayout.addLayout(self.fmriprep_subjectDropdownLayout)

        # Task/run dropdowns with Label
        self.fmriprep_taskDropdownLayout = QHBoxLayout()
        self.fmriprep_taskLabel = QLabel("Task:")
        self.fmriprep_taskLabel.setFixedWidth(90)
        self.fmriprep_taskDropdown = QComboBox()
        self.fmriprep_sessionLabel = QLabel("Session:")
        self.fmriprep_sessionLabel.setFixedWidth(65)
        self.fmriprep_sessionDropdown = QComboBox()
        self.fmriprep_runLabel = QLabel("Run:")
        self.fmriprep_runLabel.setFixedWidth(40)
        self.fmriprep_runDropdown = QComboBox()

        self.fmriprep_taskDropdownLayout.addWidget(self.fmriprep_taskLabel, 1)
        self.fmriprep_taskDropdownLayout.addWidget(self.fmriprep_taskDropdown, 4)
        self.fmriprep_taskDropdownLayout.addWidget(self.fmriprep_sessionLabel, 1)
        self.fmriprep_taskDropdownLayout.addWidget(self.fmriprep_sessionDropdown, 1)
        self.fmriprep_taskDropdownLayout.addWidget(self.fmriprep_runLabel, 1)
        self.fmriprep_taskDropdownLayout.addWidget(self.fmriprep_runDropdown, 1)
        self.fmriprep_fileLayout.addLayout(self.fmriprep_taskDropdownLayout)

        # Parcellation Dropdown with Label
        self.fmriprep_parcellationLayout = QHBoxLayout()
        self.fmriprep_parcellationLabel = QLabel("Parcellation:")
        self.fmriprep_parcellationLabel.setFixedWidth(90)
        self.fmriprep_parcellationDropdown = QComboBox()
        self.fmriprep_parcellationDropdown.addItems(self.atlas_options.keys())
        self.fmriprep_parcellationOptionsLabel = QLabel("Type:")
        self.fmriprep_parcellationOptionsLabel.setFixedWidth(40)
        self.fmriprep_parcellationOptions = QComboBox()

        self.fmriprep_parcellationLayout.addWidget(self.fmriprep_parcellationLabel, 1)
        self.fmriprep_parcellationLayout.addWidget(self.fmriprep_parcellationDropdown, 4)
        self.fmriprep_parcellationLayout.addWidget(self.fmriprep_parcellationOptionsLabel, 1)
        self.fmriprep_parcellationLayout.addWidget(self.fmriprep_parcellationOptions, 3)
        self.fmriprep_fileLayout.addLayout(self.fmriprep_parcellationLayout)

        # Sphere Layout Container
        self.fmriprep_sphereContainer = QWidget()
        fmriprep_sphereLayout = QHBoxLayout(self.fmriprep_sphereContainer)
        fmriprep_sphereLayoutLabel = QLabel("Sphere radius (mm):")
        self.fmriprep_sphereRadiusSpinbox = CustomDoubleSpinbox(special_value="single voxel", min=0.0, max=20.0)
        self.fmriprep_sphereRadiusSpinbox.setValue(5.0)
        self.fmriprep_overlapCheckbox = QCheckBox("Allow overlap")
        fmriprep_sphereLayout.addWidget(fmriprep_sphereLayoutLabel)
        fmriprep_sphereLayout.addWidget(self.fmriprep_sphereRadiusSpinbox)
        fmriprep_sphereLayout.addWidget(self.fmriprep_overlapCheckbox)
        fmriprep_sphereLayout.addStretch(1)
        fmriprep_sphereLayout.setContentsMargins(0, 0, 0, 0)
        self.fmriprep_sphereContainer.setLayout(fmriprep_sphereLayout)
        self.fmriprep_fileLayout.addWidget(self.fmriprep_sphereContainer)

        # Connect dropdown changes to handler function
        self.fmriprep_subjectDropdown.currentIndexChanged.connect(self.onFmriprepLayoutChanged)
        self.fmriprep_taskDropdown.currentIndexChanged.connect(self.onFmriprepLayoutChanged)
        self.fmriprep_sessionDropdown.currentIndexChanged.connect(self.onFmriprepLayoutChanged)
        self.fmriprep_runDropdown.currentIndexChanged.connect(self.onFmriprepLayoutChanged)
        self.fmriprep_parcellationDropdown.currentIndexChanged.connect(self.onFmriprepAtlasChanged)
        self.fmriprep_parcellationOptions.currentIndexChanged.connect(self.onFmriprepLayoutChanged)

        ##############################
        # Confound selection container
        fmriprep_confoundsContainer = QGroupBox("Cleaning options")
        fmriprep_confoundsLayout = QVBoxLayout(fmriprep_confoundsContainer)

        # Checkbox container widget
        self.generalCleaningContainer = QWidget()
        generalCleaningLayout = QVBoxLayout(self.generalCleaningContainer)
        generalCleaningLayout.setContentsMargins(0, 10, 0, 0)

        cleaningLayout = QHBoxLayout()
        self.fmriprep_standardizeCheckbox = QCheckBox("Standardize")
        self.fmriprep_detrendCheckbox = QCheckBox("Detrend")
        cleaningLayout.addItem(QSpacerItem(5, 0, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum))
        cleaningLayout.addWidget(self.fmriprep_standardizeCheckbox)
        cleaningLayout.addItem(QSpacerItem(10, 0, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum))
        cleaningLayout.addWidget(self.fmriprep_detrendCheckbox)
        cleaningLayout.addItem(QSpacerItem(10, 0, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum))
        self.fmriprep_highVarianceCheckbox = QCheckBox("Regress high variance confounds")
        cleaningLayout.addWidget(self.fmriprep_highVarianceCheckbox)
        cleaningLayout.setContentsMargins(0, 0, 0, 0)
        cleaningLayout.addStretch(1)
        generalCleaningLayout.addLayout(cleaningLayout)
        fmriprep_confoundsLayout.addWidget(self.generalCleaningContainer)

        self.fmriprep_standardizeCheckbox.setChecked(True)
        self.fmriprep_detrendCheckbox.setChecked(True)

        # Smoothing and filtering container widget
        self.fmriprep_smoothingContainer = QWidget()
        fmriprep_smoothingLayout = QHBoxLayout(self.fmriprep_smoothingContainer)
        smoothingLabel = QLabel("Smoothing fwhm (mm):")
        self.fmriprep_smoothingSpinbox = CustomDoubleSpinbox(special_value=None, min=0.0, max=20.0)
        self.fmriprep_smoothingSpinbox.setDecimals(2)
        self.fmriprep_smoothingSpinbox.setSingleStep(1.0)
        fmriprep_smoothingLayout.addWidget(smoothingLabel)
        fmriprep_smoothingLayout.addWidget(self.fmriprep_smoothingSpinbox)
        fmriprep_smoothingLayout.addStretch(1)
        fmriprep_smoothingLayout.setContentsMargins(0, 0, 0, 0)
        self.fmriprep_smoothingContainer.setLayout(fmriprep_smoothingLayout)
        fmriprep_confoundsLayout.addWidget(self.fmriprep_smoothingContainer)

        # Filtering Layout Container
        self.fmriprep_filteringContainer = QWidget()
        fmriprep_filteringLayout = QHBoxLayout(self.fmriprep_filteringContainer)
        fmriprep_highPassLabel = QLabel("High Pass:")
        self.fmriprep_highPassCutoff = CustomDoubleSpinbox(special_value=None, min=0.0, max=1.0)
        self.fmriprep_highPassCutoff.setDecimals(3)
        self.fmriprep_highPassCutoff.setSingleStep(0.01)
        self.fmriprep_highPassCutoff.setSuffix(" Hz")

        fmriprep_lowPassLabel = QLabel("Low Pass:")
        self.fmriprep_lowPassCutoff = CustomDoubleSpinbox(special_value=None, min=0.0, max=1.0)
        self.fmriprep_lowPassCutoff.setDecimals(3)
        self.fmriprep_lowPassCutoff.setSingleStep(0.1)
        self.fmriprep_lowPassCutoff.setSuffix(" Hz")

        fmriprep_trLabel = QLabel("TR:")
        self.fmriprep_trValue = CustomListSpinbox(special_value=None, min=0.0, max=20.0, values=[0.72, 0.8, 1.5, 2.5])
        self.fmriprep_trValue.setDecimals(3)
        self.fmriprep_trValue.setSingleStep(0.5)
        self.fmriprep_trValue.setSuffix(" s  ")

        fmriprep_filteringLayout.addWidget(fmriprep_highPassLabel)
        fmriprep_filteringLayout.addWidget(self.fmriprep_highPassCutoff)
        fmriprep_filteringLayout.addWidget(fmriprep_lowPassLabel)
        fmriprep_filteringLayout.addWidget(self.fmriprep_lowPassCutoff)
        fmriprep_filteringLayout.addWidget(fmriprep_trLabel)
        fmriprep_filteringLayout.addWidget(self.fmriprep_trValue)
        fmriprep_filteringLayout.addStretch(1)
        fmriprep_filteringLayout.setContentsMargins(0, 0, 0, 0)
        self.fmriprep_filteringContainer.setLayout(fmriprep_filteringLayout)
        fmriprep_confoundsLayout.addWidget(self.fmriprep_filteringContainer)

        # Discard values container
        self.fmriprep_discardContainer = QWidget()
        fmriprep_discardLayout = QHBoxLayout()
        fmriprep_discardLayout.setContentsMargins(0, 0, 0, 0)
        fmriprep_discardLabel = QLabel("Discard initial frames:")
        self.fmriprep_discardSpinBox = CustomSpinBox(None)
        self.fmriprep_discardSpinBox.setSuffix(" frames")
        self.fmriprep_discardSpinBox.setSingleStep(5)
        fmriprep_discardLayout.addWidget(fmriprep_discardLabel)
        fmriprep_discardLayout.addWidget(self.fmriprep_discardSpinBox)
        fmriprep_discardLayout.addStretch(1)
        self.fmriprep_discardContainer.setLayout(fmriprep_discardLayout)
        fmriprep_confoundsLayout.addWidget(self.fmriprep_discardContainer)

        # Confound strategy container widget
        fmriprep_confoundsStrategyWidget = self.loadConfounds()
        fmriprep_confoundsLayout.addWidget(fmriprep_confoundsStrategyWidget)

        ####################
        # Combine containers
        self.fmriprepLayout.addWidget(self.fmriprep_fileContainer)
        self.fmriprepLayout.addItem(QSpacerItem(0, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))
        self.fmriprepLayout.addWidget(fmriprep_confoundsContainer)

        ####################
        # Calculation button
        self.fmriprep_calculateButton = QPushButton('Extract time series')
        self.fmriprep_calculateButton.clicked.connect(self.calculateTimeSeries)
        self.fmriprepLayout.addWidget(self.fmriprep_calculateButton)

        # Add the fmriprep layout to the main layout
        leftLayout.addWidget(self.fmriprepContainer)
        self.fmriprepContainer.hide()

        return

    def addMiscDataLayout(self, leftLayout):
        """
        Add the layout for the label and save button to the left layout of the data tab.
        """
        self.processingResultsLabel = QLabel()
        leftLayout.addWidget(self.processingResultsLabel)
        leftLayout.addStretch()

        self.saveTimeSeriesButton = QPushButton('Save time series')
        self.saveTimeSeriesButton.clicked.connect(self.saveTimeSeries)
        leftLayout.addWidget(self.saveTimeSeriesButton)
        self.saveTimeSeriesButton.hide()

        return

    def addDataPlotLayout(self, rightLayout):
        plotTabWidget = QTabWidget()

        plotTab = QWidget()
        plotTab.setLayout(QVBoxLayout())
        self.boldFigure = Figure()
        self.boldCanvas = FigureCanvas(self.boldFigure)
        self.boldFigure.patch.set_facecolor('#f3f1f5')
        plotTab.layout().addWidget(self.boldCanvas)
        plotTabWidget.addTab(plotTab, "Carpet Plot")

        # Draw default plot (logo) on the canvas
        self.plotLogo(self.boldFigure)
        self.boldCanvas.draw()

        rightLayout.addWidget(plotTabWidget)

    # Connectivity layouts
    def addConnectivityLayout(self, leftLayout):
        self.connectivityFileNameLabel = QLabel('No time series data available. Please use the "Data Preparation" tab first.')
        leftLayout.addWidget(self.connectivityFileNameLabel)

        # Subject dropdown in case there are multiple subjects
        """self.connectivity_subjectDropdownContainer = QWidget()
        connectivity_subjectDropdownLayout = QHBoxLayout()
        connectivity_subjectLabel = QLabel("Use time series from subject:")
        connectivity_subjectLabel.setFixedWidth(180)
        self.connectivity_subjectDropdown = QComboBox()
        connectivity_subjectDropdownLayout.addWidget(connectivity_subjectLabel)
        connectivity_subjectDropdownLayout.addWidget(self.connectivity_subjectDropdown)
        connectivity_subjectDropdownLayout.setContentsMargins(0, 0, 5, 0)
        self.connectivity_subjectDropdownContainer.setLayout(connectivity_subjectDropdownLayout)
        self.connectivity_subjectDropdownContainer.hide()
        leftLayout.addWidget(self.connectivity_subjectDropdownContainer)"""
       
        leftLayout.addItem(QSpacerItem(0, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))

        # Connectivity method container
        self.connectivityContainer = QGroupBox("Connectivity method")
        connectivityContainerLayout = QVBoxLayout()

        # Checkboxes for method types
        self.continuousCheckBox = QCheckBox("Continuous")
        self.stateBasedCheckBox = QCheckBox("State-based")
        self.staticCheckBox = QCheckBox("Static")

        checkboxLayout = QHBoxLayout()
        checkboxLayout.addWidget(self.continuousCheckBox)
        checkboxLayout.addWidget(self.stateBasedCheckBox)
        checkboxLayout.addWidget(self.staticCheckBox)
        checkboxLayout.setSpacing(10)
        checkboxLayout.addStretch()

        connectivityContainerLayout.addLayout(checkboxLayout)

        # Connectivity methods
        self.methodComboBox = QComboBox()
        connectivityContainerLayout.addWidget(self.methodComboBox)

        # Create a layout for dynamic textboxes
        self.parameterLayout = QVBoxLayout()

        # Create a container widget for the parameter layout
        self.parameterContainer = QWidget()  # Use an instance attribute to access it later
        self.parameterContainer.setLayout(self.parameterLayout)
        self.parameterContainer.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        # Add the container widget to the left layout directly below the combobox
        connectivityContainerLayout.addWidget(self.parameterContainer)

        # Initial population of the combobox, this does the entire initialization
        self.updateMethodComboBox()

        # Add parameter textbox for time_series
        self.time_series_textbox = QLineEdit()
        self.time_series_textbox.setReadOnly(True) # read only as based on the loaded file

        # Add the connectiity container to the layout
        self.connectivityContainer.setLayout(connectivityContainerLayout)
        leftLayout.addWidget(self.connectivityContainer)

        # Add a stretch after the parameter layout container
        leftLayout.addStretch()

        # Calculate connectivity and save button
        buttonsLayout = QHBoxLayout()

        # Calculate connectivity button
        self.calculateConnectivityButton = QPushButton('Estimate Connectivity')
        self.calculateConnectivityButton.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        buttonsLayout.addWidget(self.calculateConnectivityButton, 2)  # 2/3 of the space
        self.calculateConnectivityButton.clicked.connect(self.calculateConnectivity)

        # Create the "Save" button
        self.saveConnectivityButton = QPushButton('Save Connectivity')
        self.saveConnectivityButton.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        buttonsLayout.addWidget(self.saveConnectivityButton, 1)  # 1/3 of the space
        self.saveConnectivityButton.clicked.connect(self.saveConnectivity)

        # Add the buttons layout to the left layout
        leftLayout.addLayout(buttonsLayout)

        # Memory buttons
        self.keepInMemoryCheckbox = QCheckBox("Keep in memory")
        self.keepInMemoryCheckbox.stateChanged.connect(self.onKeepInMemoryChecked)
        self.clearMemoryButton = QPushButton("Clear Memory")
        self.clearMemoryButton.clicked.connect(self.onClearMemory)

        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(self.keepInMemoryCheckbox)
        buttonLayout.addWidget(self.clearMemoryButton)
        
        # Assuming you have a QVBoxLayout named 'leftLayout'
        leftLayout.addLayout(buttonLayout)

        # Calculation info textbox
        self.calculatingLabel = QLabel('No data calculated yet.')
        leftLayout.addWidget(self.calculatingLabel)

        # Disable buttons
        self.calculateConnectivityButton.setEnabled(False)
        self.saveConnectivityButton.setEnabled(False)
        self.keepInMemoryCheckbox.setEnabled(False)
        self.clearMemoryButton.setEnabled(False)

        return

    def addConnectivityPlotLayout(self, rightLayout):
        self.tabWidget = QTabWidget()

        # Tab 1: Imshow plot
        imshowTab = QWidget()
        imshowLayout = QVBoxLayout()
        imshowTab.setLayout(imshowLayout)

        self.connectivityFigure = Figure()
        self.connectivityCanvas = FigureCanvas(self.connectivityFigure)
        self.connectivityFigure.patch.set_facecolor('#f3f1f5')
        imshowLayout.addWidget(self.connectivityCanvas)
        self.tabWidget.addTab(imshowTab, "Connectivity")

        # Tab 2: Time series/course plot
        timeSeriesTab = QWidget()
        timeSeriesLayout = QVBoxLayout()
        timeSeriesTab.setLayout(timeSeriesLayout)

        self.timeSeriesFigure = Figure()
        self.timeSeriesCanvas = FigureCanvas(self.timeSeriesFigure)
        self.timeSeriesFigure.patch.set_facecolor('#f3f1f5')
        timeSeriesLayout.addWidget(self.timeSeriesCanvas)
        self.tabWidget.addTab(timeSeriesTab, "Time course")

        rightLayout.addWidget(self.tabWidget)

        # Tab 3: Distribution plot
        distributionTab = QWidget()
        distributionLayout = QVBoxLayout()
        distributionTab.setLayout(distributionLayout)

        self.distributionFigure = Figure()
        self.distributionCanvas = FigureCanvas(self.distributionFigure)
        self.distributionFigure.patch.set_facecolor('#f3f1f5')
        distributionLayout.addWidget(self.distributionCanvas)
        self.tabWidget.addTab(distributionTab, "Distribution")

        # Method for doing things if the tab is changed
        self.tabWidget.currentChanged.connect(self.onTabChanged)

        # Slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)  # Set the minimum value of the slider
        self.slider.setMaximum(0)
        self.slider.valueChanged.connect(self.onSliderValueChanged)
        rightLayout.addWidget(self.slider)

        # Navigation buttons layout
        self.navButtonContainer = QWidget()
        navButtonLayout = QHBoxLayout()
        navButtonLayout.addStretch(1)  # left stretch

        # Creating navigation buttons
        self.backLargeButton = QPushButton("<<")
        self.backButton = QPushButton("<")
        self.positionLabel = QLabel('no data available')
        self.forwardButton = QPushButton(">")
        self.forwardLargeButton = QPushButton(">>")

        # Buttons that interact with the slider
        navButtonLayout.addWidget(self.backLargeButton)
        navButtonLayout.addWidget(self.backButton)
        navButtonLayout.addWidget(self.positionLabel)
        navButtonLayout.addWidget(self.forwardButton)
        navButtonLayout.addWidget(self.forwardLargeButton)

        self.backLargeButton.clicked.connect(self.onSliderButtonClicked)
        self.backButton.clicked.connect(self.onSliderButtonClicked)
        self.forwardButton.clicked.connect(self.onSliderButtonClicked)
        self.forwardLargeButton.clicked.connect(self.onSliderButtonClicked)

        navButtonLayout.addStretch(1) # right stretch
        navButtonLayout.setContentsMargins(0, 0, 0, 0)
        self.navButtonContainer.setLayout(navButtonLayout)
        rightLayout.addWidget(self.navButtonContainer)

        # Roi selector widget for time series plot
        self.roiSelectorWidget = QWidget()
        
        self.rowSelector = QSpinBox()
        self.rowSelector.setMaximum(0)
        self.rowSelector.valueChanged.connect(self.plotTimeSeries)
        
        self.colSelector = QSpinBox()
        self.colSelector.setMaximum(0)
        self.colSelector.valueChanged.connect(self.plotTimeSeries)

        timeSeriesSelectorLayout = QHBoxLayout()
        timeSeriesSelectorLayout.addWidget(QLabel("Brain region 1 (row):"))
        timeSeriesSelectorLayout.addWidget(self.rowSelector)
        timeSeriesSelectorLayout.addWidget(QLabel("Brain region 2 (column):"))
        timeSeriesSelectorLayout.addWidget(self.colSelector)

        timeSeriesSelectorLayout.setContentsMargins(0, 0, 0, 0)
        self.roiSelectorWidget.setLayout(timeSeriesSelectorLayout)
        rightLayout.addWidget(self.roiSelectorWidget)

        # Hide all the elements at first
        self.slider.hide()
        self.navButtonContainer.hide()
        self.roiSelectorWidget.hide()

        # Connect the stateChanged signal of checkboxes to the slot
        self.continuousCheckBox.stateChanged.connect(self.updateMethodComboBox)
        self.stateBasedCheckBox.stateChanged.connect(self.updateMethodComboBox)
        self.staticCheckBox.stateChanged.connect(self.updateMethodComboBox)

        self.continuousCheckBox.setChecked(True)
        self.stateBasedCheckBox.setChecked(True)
        self.staticCheckBox.setChecked(True)

        return

    # Graph layouts
    def addGraphLayout(self, leftLayout):
        buttonsLayout = QHBoxLayout()

        self.loadGraphFileButton = QPushButton('Load from data')
        self.loadGraphFileButton.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        buttonsLayout.addWidget(self.loadGraphFileButton, 1)
        self.loadGraphFileButton.clicked.connect(self.loadGraphFile)

        self.graphFileNameLabel = QLabel('No data available.')
        self.graphProcessingStepCounter = 1
        self.graphMeasureStepCounter = 1

        leftLayout.addLayout(buttonsLayout)
        leftLayout.addWidget(self.graphFileNameLabel)
        leftLayout.addItem(QSpacerItem(0, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))

        self.selectionContainer = QGroupBox("Select connectivity estimates")
        selectionLayout = QHBoxLayout()

        self.graphUseAllCheckbox = QCheckBox("All")
        self.graphUseAllCheckbox.clicked.connect(self.handleGraphCheckboxes)
        self.graphUseSingleCheckbox = QCheckBox("Current (as plotted)")
        self.graphUseSingleCheckbox.clicked.connect(self.handleGraphCheckboxes)
        self.graphUseCustomCheckbox = QCheckBox("Custom:")
        self.graphUseCustomCheckbox.clicked.connect(self.handleGraphCheckboxes)
        self.graphUseSpecificTextbox = QLineEdit()
        self.graphUseSpecificTextbox.setPlaceholderText("e.g.: 10:50:2, 100, 200")
        self.graphUseSpecificTextbox.setEnabled(False)
        self.graphUsePushButton = QPushButton("  Apply  ")
        self.graphUsePushButton.clicked.connect(self.onGraphAmountSelected)

        selectionLayout.addWidget(self.graphUseAllCheckbox)
        selectionLayout.addItem(QSpacerItem(7, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))
        selectionLayout.addWidget(self.graphUseSingleCheckbox)
        selectionLayout.addItem(QSpacerItem(7, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))
        selectionLayout.addWidget(self.graphUseCustomCheckbox)
        selectionLayout.addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))
        selectionLayout.addWidget(self.graphUseSpecificTextbox)
        selectionLayout.addItem(QSpacerItem(7, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))
        selectionLayout.addWidget(self.graphUsePushButton)

        self.selectionContainer.setLayout(selectionLayout)
        leftLayout.addWidget(self.selectionContainer)
        leftLayout.addItem(QSpacerItem(0, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))
        
        self.graphUseAllCheckbox.setChecked(True)
        self.graphUsePushButton.setEnabled(False)

        # Graph analysis container
        self.graphContainer = QGroupBox("Graph analysis options")
        graphContainerLayout = QVBoxLayout()

        # Checkboxes for method types
        self.preprocessingCheckBox = QCheckBox("Preprocessing")
        self.graphCheckBox = QCheckBox("Comet")
        self.BCTCheckBox = QCheckBox("BCT")

        # Checkboxes for function types
        checkboxLayout = QHBoxLayout()
        checkboxLayout.addWidget(self.preprocessingCheckBox)
        checkboxLayout.addWidget(self.graphCheckBox)
        checkboxLayout.addWidget(self.BCTCheckBox)
        checkboxLayout.setSpacing(10)
        checkboxLayout.addStretch()

        # Init checkbox states
        self.preprocessingCheckBox.setChecked(True)
        self.graphCheckBox.setChecked(True)
        self.BCTCheckBox.setChecked(True)
        graphContainerLayout.addLayout(checkboxLayout)

        # Create the combo box for selecting the graph analysis type
        self.graphAnalysisComboBox = QComboBox()
        graphContainerLayout.addWidget(self.graphAnalysisComboBox)

        self.graphAnalysisComboBox.currentIndexChanged.connect(self.onGraphCombobox)
        self.graphParameterLayout = QVBoxLayout()

        # Connect the stateChanged signal of checkboxes to the slot
        self.preprocessingCheckBox.stateChanged.connect(self.updateGraphComboBox)
        self.graphCheckBox.stateChanged.connect(self.updateGraphComboBox)
        self.BCTCheckBox.stateChanged.connect(self.updateGraphComboBox)

        self.updateGraphComboBox()

        # Create a container widget for the parameter layout
        parameterContainer = QWidget()  # Use an instance attribute to access it later
        parameterContainer.setLayout(self.graphParameterLayout)
        parameterContainer.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        # Add the container widget to the left layout directly below the combobox
        graphContainerLayout.addWidget(parameterContainer)

        # Stretch empty space
        graphContainerLayout.addStretch()

        # Option add/clear buttons
        buttonsLayout = QHBoxLayout()
        self.addOptionButton = QPushButton('Add option')
        buttonsLayout.addWidget(self.addOptionButton, 2)
        self.addOptionButton.clicked.connect(self.calculateGraph)

        self.clearAllButton = QPushButton('Clear all')
        buttonsLayout.addWidget(self.clearAllButton, 1)
        self.clearAllButton.clicked.connect(self.onClearAllGraphOptions)

        self.addOptionButton.setEnabled(False)
        self.clearAllButton.setEnabled(False)

        graphContainerLayout.addLayout(buttonsLayout)
        self.graphContainer.setLayout(graphContainerLayout)
        leftLayout.addWidget(self.graphContainer, 3)

        # Step container
        leftLayout.addItem(QSpacerItem(0, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))

        self.graphStepContainer = QGroupBox("List of performed steps")
        graphContainerLayout = QHBoxLayout()

        graphContainerLayoutBox1 = QVBoxLayout()
        processingLabel = QLabel("Processing:")
        graphContainerLayoutBox1.addWidget(processingLabel)

        self.optionsTextbox = QTextEdit()
        self.optionsTextbox.setReadOnly(True)
        graphContainerLayoutBox1.addWidget(self.optionsTextbox)

        graphContainerLayoutBox2 = QVBoxLayout()
        measureLabel = QLabel("Graph measures:")
        graphContainerLayoutBox2.addWidget(measureLabel)

        self.measureTextbox = QTextEdit()
        self.measureTextbox.setReadOnly(True)
        graphContainerLayoutBox2.addWidget(self.measureTextbox)

        graphContainerLayout.addLayout(graphContainerLayoutBox1, 2)
        graphContainerLayout.addLayout(graphContainerLayoutBox2, 1)
        self.graphStepContainer.setLayout(graphContainerLayout)
        leftLayout.addWidget(self.graphStepContainer, 4)

        # Save button
        self.graphSaveButton = QPushButton('Save')
        leftLayout.addWidget(self.graphSaveButton)
        self.graphSaveButton.clicked.connect(self.saveGraphFile)
        self.graphSaveButton.setEnabled(False)

        return

    def addGraphPlotLayout(self, rightLayout):
        self.graphTabWidget = QTabWidget()

        # Tab 1: Adjacency matrix plot
        matrixTab = QWidget()
        matrixLayout = QVBoxLayout(matrixTab)

        self.matrixFigure = Figure()
        self.matrixCanvas = FigureCanvas(self.matrixFigure)
        self.matrixFigure.patch.set_facecolor('#f3f1f5')
        matrixLayout.addWidget(self.matrixCanvas)

        # Draw default plot (logo)
        self.plotLogo(self.matrixFigure)
        self.matrixCanvas.draw()

        self.graphTabWidget.addTab(matrixTab, "Processing")

        # Tab 2: Graph measures plot
        measureTab = QWidget()
        measureLayout = QVBoxLayout()
        measureTab.setLayout(measureLayout)

        dropdownWidget = QWidget()
        dropdownLayout = QHBoxLayout()
        dropdownLayout.addWidget(QLabel("Available graph measures:"))
        self.graphResultMeasureDropdown = QComboBox()
        self.graphResultMeasureDropdown.currentIndexChanged.connect(self.onGraphResultMeasureSelected)
        self.graphResultMeasureDropdown.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.graphResultMeasureDropdown.setPlaceholderText("No graph measure available")
        dropdownLayout.addWidget(self.graphResultMeasureDropdown)
        dropdownLayout.setContentsMargins(0, 0, 0, 0)
        dropdownWidget.setLayout(dropdownLayout)
        measureLayout.addWidget(dropdownWidget)

        # Widget for the plot
        plotWidget = QWidget()  # Create a widget to hold the plot
        plotLayout = QVBoxLayout()  # Use QVBoxLayout for the plot widget
        plotWidget.setLayout(plotLayout)

        # Add the graph canvas to the plot layout
        self.graphFigure = Figure()
        self.graphCanvas = FigureCanvas(self.graphFigure)
        self.graphFigure.patch.set_facecolor('#f3f1f5')
        plotLayout.addWidget(self.graphCanvas)

        # Widget for the textbox
        self.graphTextbox = QTextEdit()
        self.graphTextbox.setReadOnly(True)

        # Add the plot widget and the textbox to the measure layout
        measureLayout.addWidget(plotWidget, 4)
        measureLayout.addWidget(self.graphTextbox, 1)
        self.graphTabWidget.addTab(measureTab, "Graph Measures")

        # Draw default plot (e.g., logo)
        self.plotLogo(self.graphFigure)
        self.graphCanvas.draw()

        # Add widgets to the right layout
        rightLayout.addWidget(self.graphTabWidget)
        self.graphTabWidget.currentChanged.connect(self.onGraphTabChanged)

        # Slider and navigation buttons
        self.graphSlider = QSlider(Qt.Orientation.Horizontal)
        self.graphSlider.setMinimum(0)
        self.graphSlider.setMaximum(0)
        self.graphSlider.valueChanged.connect(self.onGraphSliderValueChanged)
        self.graphSlider.hide()
        rightLayout.addWidget(self.graphSlider)

        self.graphNavButtonContainer = QWidget()
        navButtonLayout = QHBoxLayout()
        navButtonLayout.addStretch(1) # left stretch

        self.graphBackLargeButton = QPushButton("<<")
        self.graphBackButton = QPushButton("<")
        self.graphPositionLabel = QLabel('no data available')
        self.graphForwardButton = QPushButton(">")
        self.graphForwardLargeButton = QPushButton(">>")

        navButtonLayout.addWidget(self.graphBackLargeButton)
        navButtonLayout.addWidget(self.graphBackButton)
        navButtonLayout.addWidget(self.graphPositionLabel)
        navButtonLayout.addWidget(self.graphForwardButton)
        navButtonLayout.addWidget(self.graphForwardLargeButton)
        
        navButtonLayout.addStretch(1) # right stretch
        navButtonLayout.setContentsMargins(0, 0, 0, 0)
        self.graphNavButtonContainer.setLayout(navButtonLayout)
        self.graphNavButtonContainer.hide()

        self.graphBackLargeButton.clicked.connect(self.onGraphSliderButtonClicked)
        self.graphBackButton.clicked.connect(self.onGraphSliderButtonClicked)
        self.graphForwardButton.clicked.connect(self.onGraphSliderButtonClicked)
        self.graphForwardLargeButton.clicked.connect(self.onGraphSliderButtonClicked)

        rightLayout.addWidget(self.graphNavButtonContainer)

        return

    # Multiverse layouts
    def addMultiverseLayout(self, leftLayout):
        # Top section
        self.mv_containers = []  # decision containers for multiverse analysis
        self.mv_from_file = False

        createMvContainer = QGroupBox("Create multiverse analysis template")
        self.createMvContainerLayout = QVBoxLayout()  # Make it an instance variable to access it globally
        self.createMvContainerLayout.addItem(QSpacerItem(0, 5, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))

        # Creating a first decision container
        decisionWidget = self.addDecisionContainer()
        self.createMvContainerLayout.addWidget(decisionWidget)

        # Horizontal layout for add/collapse buttons
        buttonLayout = QHBoxLayout()
        buttonLayout.setContentsMargins(0, 2, 0, 0)

        newDecisionButton = QPushButton("\u2795")
        newDecisionButton.clicked.connect(lambda: self.addNewDecision(self.createMvContainerLayout, buttonLayoutWidget))
        buttonLayout.addWidget(newDecisionButton, 5)
        buttonLayout.addStretch(21)

        # Wrap the buttonLayout in a QWidget
        buttonLayoutWidget = QWidget()
        buttonLayoutWidget.setLayout(buttonLayout)
        self.createMvContainerLayout.addWidget(buttonLayoutWidget)

        createMvContainer.setLayout(self.createMvContainerLayout)
        leftLayout.addWidget(createMvContainer)
        leftLayout.addStretch()

        # Bottom section
        performMvContainer = QGroupBox("Perform multiverse analysis")
        performMvContainerLayout = QVBoxLayout()
        performMvContainerLayout.addItem(QSpacerItem(0, 5, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))

        # Choose folder layout
        chooseFolderLayout = QHBoxLayout()
        chooseFolderButton = QPushButton('Select multiverse folder')
        chooseFolderLayout.addWidget(chooseFolderButton, 3)

        # Textbox to display the loaded script path
        self.chooseFolderEdit = QLineEdit()
        self.chooseFolderEdit.setPlaceholderText("No folder selected yet")
        self.chooseFolderEdit.setReadOnly(True)
        chooseFolderLayout.addWidget(self.chooseFolderEdit, 5)
        performMvContainerLayout.addLayout(chooseFolderLayout)

        # Load script layout
        loadLayout = QHBoxLayout()
        loadMultiverseScriptButton = QPushButton('Load multiverse script   ')
        loadLayout.addWidget(loadMultiverseScriptButton, 3)

        # Textbox to display the loaded script path
        self.loadedScriptDisplay = QLineEdit()
        self.loadedScriptDisplay.setPlaceholderText("No multiverse script loaded yet")
        self.loadedScriptDisplay.setReadOnly(True)
        loadLayout.addWidget(self.loadedScriptDisplay, 5)

        # Add the horizontal layout to the container layout
        performMvContainerLayout.addLayout(loadLayout)

        # Row with the three buttons
        buttonLayout = QHBoxLayout()

        self.createMultiverseButton = QPushButton('Create multiverse')
        buttonLayout.addWidget(self.createMultiverseButton)

        self.runMultiverseButton = QPushButton('Run multiverse')
        buttonLayout.addWidget(self.runMultiverseButton)

        # Label an spinbox for parallelisation
        parallelLabel = QLabel('Parallel:')
        parallelLabel.setFixedWidth(60)
        buttonLayout.addWidget(parallelLabel)

        self.paralleliseMultiverseSpinbox = QSpinBox()
        self.paralleliseMultiverseSpinbox.setRange(1, 64)
        self.paralleliseMultiverseSpinbox.setValue(1)
        self.paralleliseMultiverseSpinbox.setFixedWidth(45)
        buttonLayout.addWidget(self.paralleliseMultiverseSpinbox)

        # Add the buttons layout to the container layout
        performMvContainerLayout.addLayout(buttonLayout)

        # Set the layout for the container
        performMvContainer.setLayout(performMvContainerLayout)

        # Add the container to the main layout (leftLayout)
        leftLayout.addWidget(performMvContainer)

        # Disable and connect
        self.createMultiverseButton.setEnabled(False)
        self.runMultiverseButton.setEnabled(False)
        self.paralleliseMultiverseSpinbox.setEnabled(False)

        chooseFolderButton.clicked.connect(self.selectMultiverseFolder)
        loadMultiverseScriptButton.clicked.connect(self.loadMultiverseScript)
        self.createMultiverseButton.clicked.connect(self.createMultiverse)
        self.runMultiverseButton.clicked.connect(self.runMultiverseScript)

        return

    def addMultiversePlotLayout(self, rightLayout):
        # Creating a tab widget for different purposes
        multiverseTabWidget = QTabWidget()

        # Tab 1: Template textbox for script display
        templateTab = QWidget()
        templateLayout = QVBoxLayout()
        templateTab.setLayout(templateLayout)

        self.scriptDisplay = QTextEdit()
        self.scriptDisplay.setStyleSheet("""QTextEdit { background-color: #f3f1f5; color: #19232d;}""")
        self.scriptDisplay.setReadOnly(True)
        templateLayout.addWidget(self.scriptDisplay)

        # Add syntax highlighting to the script
        self.highlighter = PythonHighlighter(self.scriptDisplay.document())

        # Create the toggle button and add it inside the QTextEdit
        self.toggleButton = QPushButton(" 🔒 ", self.scriptDisplay)
        self.toggleButton.setFixedSize(30, 20)
        self.toggleButton.clicked.connect(self.toggleReadOnly)

        # Position the button at the top right corner inside the QTextEdit
        self.toggleButton.move(self.scriptDisplay.width() - self.toggleButton.width() - 25, 5)

        # Adjust button position when the QTextEdit is resized
        self.scriptDisplay.resizeEvent = self.updateToggleButtonPosition

        # Create a layout for the buttons (reset, save))
        scriptButtonLayout = QHBoxLayout()

        resetMultiverseScriptButton = QPushButton('Reset template script')
        resetMultiverseScriptButton.clicked.connect(self.resetMultiverseScript)
        scriptButtonLayout.addWidget(resetMultiverseScriptButton)

        saveMultiverseScriptButton = QPushButton('Save template script')
        saveMultiverseScriptButton.clicked.connect(self.saveMultiverseScript)
        scriptButtonLayout.addWidget(saveMultiverseScriptButton)

        # Add to the layout
        templateLayout.addLayout(scriptButtonLayout)
        multiverseTabWidget.addTab(templateTab, "Multiverse template")
        self.generateMultiverseScript(init_template=True)

        # Tab 2: Plot for visualization
        self.plotMvTab = QWidget()
        self.plotMvTab.setLayout(QVBoxLayout())

        # Create and add the canvas for the multiverse plot
        self.multiverseFigure = Figure()
        self.multiverseCanvas = FigureCanvas(self.multiverseFigure)
        self.multiverseFigure.patch.set_facecolor('#f3f1f5')
        self.plotMvTab.layout().addWidget(self.multiverseCanvas)
        multiverseTabWidget.addTab(self.plotMvTab, "Multiverse Overview")

        self.plotLogo(self.multiverseFigure)
        self.multiverseCanvas.draw()
        self.createSummaryWidgets()

        # Tab 3: Plot for specification curve
        self.specTab = QWidget()
        self.specTab.setLayout(QVBoxLayout())

        # Create and add the canvas for the specification curve
        self.specFigure = Figure()
        self.specCanvas = FigureCanvas(self.specFigure)
        self.specFigure.patch.set_facecolor('#f3f1f5')
        self.specTab.layout().addWidget(self.specCanvas)

        multiverseTabWidget.addTab(self.specTab, "Specification Curve")
        self.plotLogo(self.specFigure)
        self.specCanvas.draw()
        self.createSpecificationCurveWidgets()


        ########################################
        # Add the tab widget to the right layout
        rightLayout.addWidget(multiverseTabWidget)


    """
    Data tab
    """
    # Single time series functions
    def loadFile(self):
        """
        Load a single file and display the data in the GUI.
        """
        # Allowed file types
        fileFilter = "All supported files (*.mat *.txt *.npy *.tsv *.nii *.nii.gz *.dtseries.nii *.ptseries.nii);;\
                      MAT files (*.mat);;Text files (*.txt);;NumPy files (*.npy);;TSV files (*.tsv);;\
                      NIFTI files (*.nii, .nii.gz);;CIFTI files (*.dtseries.nii *.ptseries.nii)"
        file_path, _ = QFileDialog.getOpenFileName(self, "Load File", "", fileFilter)

        if not file_path:
            QMessageBox.warning(self, "Load Error", f"No valid file selected.")
            return

        # Initial setup
        self.data.ts_data = None
        self.data.file_data = None
        self.data.roi_names = None
        self.data.file_path = file_path
        self.data.file_name = file_path.split('/')[-1]
        
        self.plotLogo(self.boldFigure)
        self.boldCanvas.draw()

        self.fmriprep_layout = None
        self.data.sample_mask = None
        self.processingResultsLabel.setText("")
        self.detrendCheckbox.setChecked(True)
        self.standardizeCheckbox.setChecked(True)

        # Disable/enable containers
        self.tsExtractionContainer.show() # main container
        self.fmriprepContainer.hide() # fMRIprep container

        try:
            self.subjectDropdown.currentIndexChanged.disconnect(self.onSubjectChanged)
            self.subjectDropdown.clear()   
            #self.connectivity_subjectDropdown.currentIndexChanged.disconnect(self.onSubjectChanged)
            #self.connectivity_subjectDropdown.clear()

        except:
            pass

        if file_path.endswith('.mat'):
            try:
                data_dict = loadmat(file_path)
            except:
                data_dict = mat73.loadmat(file_path)
            print("Loaded mat file with keys:", data_dict.keys())
            print(f"Using data from key: {list(data_dict.keys())[-1]}")
            self.data.file_data = data_dict[list(data_dict.keys())[-1]] # always get data for the last key

        elif file_path.endswith('.txt'):
            self.data.file_data = np.loadtxt(file_path)

        elif file_path.endswith('.npy'):
            self.data.file_data = np.load(file_path)

            if not np.ndim(self.data.file_data) == 3:
                QMessageBox.warning(self, "Error", ".npy files must contain a single 3D array.")
                return

        elif file_path.endswith(".tsv"):
            data = pd.read_csv(file_path, sep='\t', header=None, na_values='n/a')

            if data.iloc[0].apply(lambda x: np.isscalar(x) and np.isreal(x)).all():
                rois = None  # No rois found, the first row is part of the data
            else:
                rois = data.iloc[0]  # The first row is rois
                data = data.iloc[1:]  # Remove the header row from the data

            # Identify empty columns and remove rois
            data = data.apply(pd.to_numeric, errors='coerce') # Convert all data to numeric so 'n/a' and other non-numerics are treated as NaN
            empty_columns = data.columns[data.isna().all()]

            if rois is not None:
                removed_rois = rois[empty_columns].to_list()
                print("The following regions were empty and thus removed:", removed_rois)
                rois = rois.drop(empty_columns)
            data = data.dropna(axis=1, how='all').dropna(axis=0, how='all')

            self.data.file_data = data.to_numpy()
            self.data.roi_names = np.array(rois, dtype=object)
        
        elif file_path.endswith(".nii") or file_path.endswith(".nii.gz"):
            self.parcellationDropdown.currentIndexChanged.disconnect(self.onAtlasChanged)
            self.parcellationDropdown.clear()

        else:
            self.time_series_textbox.setText("Unsupported file format")
            self.data.file_data = None

        # Copy file_data into ts_data for further processing
        if self.data.file_data is not None:
            self.data.ts_data = self.data.file_data.copy()

        # Enable/disable layouts corresponding to file types
        self.setFileLayout(file_path)

        # New data, reset slider and plot
        self.backupSliderValueTab02 = 0
        self.backupSliderValueTab1 = 0
        self.slider.setValue(0)
        self.graphSlider.setValue(0)
        self.connectivityFigure.clear()
        self.connectivityCanvas.draw()
        
        # Reset and enable the GUI elements
        self.methodComboBox.setEnabled(True)
        self.methodComboBox.setEnabled(True)
        self.calculateConnectivityButton.setEnabled(True)
        self.clearMemoryButton.setEnabled(True)
        self.keepInMemoryCheckbox.setEnabled(True)

        # Setting the labels
        self.time_series_textbox.setText(self.data.file_name)
        fname = self.data.file_name[:20] + self.data.file_name[-30:] if len(self.data.file_name) > 50 else self.data.file_name

        if file_path.endswith(".nii") or file_path.endswith(".nii.gz"):
            self.fileNameLabel.setText(f"Loaded {fname}.")
        else:
            fshape = self.data.ts_data.shape

            if file_path.endswith(".npy"):
                self.fileNameLabel.setText(f"Loaded {fname} with {fshape[0]} subjects. Shape: {fshape[1:]}.")
                self.fileNameLabel.setText(f"Loaded {fname} with {fshape[0]} subjects. Shape: {fshape[1:]}.")
                self.connectivityFileNameLabel.setText(f"Time series data with shape {fshape} is available for state-based analysis.")
                self.saveTimeSeriesButton.show()
            elif file_path.endswith(".mat"):
                self.fileNameLabel.setText(f"Loaded data from {fname} (key: {list(data_dict.keys())[-1]}) with shape {fshape}.")
                self.connectivityFileNameLabel.setText(f"Time series data with shape {fshape} is available for connectivity analysis.")
            else:
                self.fileNameLabel.setText(f"Loaded {fname} with shape {fshape}.")
                self.connectivityFileNameLabel.setText(f"Time series data with shape {fshape} is available for connectivity analysis.")
            
            self.processingResultsLabel.setText(f"Time series data with shape {fshape} is ready for connectivity analysis.")
            self.saveTimeSeriesButton.show()

        # Enable connectivity buttions
        self.calculateConnectivityButton.setEnabled(True)
        self.saveConnectivityButton.setEnabled(True)
        self.keepInMemoryCheckbox.setEnabled(True)
        self.clearMemoryButton.setEnabled(True)

        return

    def setFileLayout(self, file_path):

        self.cleaningContainer.show()
        self.miscCleaningContainer.hide()
        self.sphereContainer.hide()
        self.smoothingContainer.hide()
        self.filteringContainer.hide()

        self.transposeCheckbox.hide()
        self.subjectDropdownContainer.hide()
        self.parcellationContainer.hide()
        self.discardContainer.hide()
        self.saveTimeSeriesButton.hide()

        self.niftiExtractButtonContainer.hide()
        self.niftiCleanButtonContainer.hide()

        # Some kind of nifti/cifti file
        if self.data.file_data is None:      
            if file_path.endswith(".dtseries.nii") or file_path.endswith(".ptseries.nii"):
                self.parcellationDropdown.addItems(self.atlas_options_cifti.keys())      
            
            elif file_path.endswith(".nii") or file_path.endswith(".nii.gz"):
                self.parcellationDropdown.addItems(self.atlas_options.keys()) 
                self.sphereContainer.show()
                self.smoothingContainer.show()       
            
            else:
                QMessageBox.warning(self, "Error", "File type not supported.")
                return
            
            self.parcellationDropdown.currentIndexChanged.connect(self.onAtlasChanged)
            self.onAtlasChanged()
            self.plotLogo(self.boldFigure)

            # Containers
            self.miscCleaningContainer.show()
            self.parcellationContainer.show()
            self.filteringContainer.show()
            self.discardContainer.show()
            self.niftiExtractButtonContainer.show()

        # 2D time series data
        elif np.ndim(self.data.file_data) == 2:
            self.plotCarpet()

            # Containers
            self.transposeCheckbox.show()
            self.miscCleaningContainer.show()
            self.filteringContainer.show()
            self.discardContainer.show()
            self.niftiCleanButtonContainer.show()

            # Connectivity checkboxes
            self.stateBasedCheckBox.setEnabled(True)
            self.stateBasedCheckBox.setChecked(True)

            self.continuousCheckBox.setEnabled(True)
            self.continuousCheckBox.setChecked(True)

            self.staticCheckBox.setEnabled(True)
            self.staticCheckBox.setChecked(True)

        # 3D time series data
        elif np.ndim(self.data.file_data) == 3:
            self.subjectDropdown.addItems(np.arange(self.data.file_data.shape[0]).astype(str))
            self.subjectDropdown.currentIndexChanged.connect(self.onSubjectChanged)

            #self.connectivity_subjectDropdown.addItems(np.arange(self.data.file_data.shape[0]).astype(str))
            #self.connectivity_subjectDropdownContainer.show()
            #self.connectivity_subjectDropdown.currentIndexChanged.connect(self.onSubjectChanged)

            self.plotCarpet()

            # Containers
            self.subjectDropdownContainer.show()
            self.transposeCheckbox.show()
            self.miscCleaningContainer.show()
            self.filteringContainer.show()
            self.discardContainer.show()
            self.niftiCleanButtonContainer.show() 
            self.niftiCleanButtonContainer.show()

            # Connectivity checkboxes
            self.stateBasedCheckBox.setEnabled(True)
            self.stateBasedCheckBox.setChecked(True)

            self.continuousCheckBox.setEnabled(False)
            self.continuousCheckBox.setChecked(False)

            self.staticCheckBox.setEnabled(False)
            self.staticCheckBox.setChecked(False)    

        else:
            QMessageBox.warning(self, "Error", "File type not supported.")
            return

    def onTransposeChecked(self, state):
        """
        Transpose checkbox event
        """
        if self.data.ts_data is None:
            return  # No data loaded
        
        # Transpose the data
        self.data.ts_data = self.data.ts_data.transpose(0, 2, 1) if self.data.ts_data.ndim == 3 else self.data.ts_data.transpose()

        # Update the labels
        fshape = self.data.ts_data.shape
        fname = self.time_series_textbox.text()
  
        if self.data.file_name.endswith('.npy'):
            self.fileNameLabel.setText(f"Loaded {fname} with {fshape[0]} subjects. Shape: {fshape[1:]}")
        else:
            self.fileNameLabel.setText(f"Loaded {fname} with shape {fshape}")

        self.time_series_textbox.setText(self.data.file_name)

        # Update carpet plot
        self.plotCarpet()

    def onSubjectChanged(self):
        """
        .npy file subject dropdown event
        """
        self.plotCarpet()
        return

    def fetchAtlas(self, atlasname, option):
        """
        Fetch parcellation with nilearn
        """
        if atlasname in self.atlas_options.keys():
            if atlasname == "AAL template (3v2)":
                atlas = datasets.fetch_atlas_aal(version="3v2")
                return atlas["maps"], atlas["labels"]

            elif atlasname == "BASC multiscale":
                atlas = datasets.fetch_atlas_basc_multiscale_2015(resolution=int(option))
                return atlas["maps"], None

            elif atlasname == "Destrieux et al. (2009)":
                atlas = datasets.fetch_atlas_destrieux_2009()
                return atlas["maps"], atlas["labels"]

            elif atlasname == "Pauli et al. (2017)":
                atlas = datasets.fetch_atlas_pauli_2017(version="det")
                return atlas["maps"], atlas["labels"]

            elif atlasname == "Schaefer et al. (2018)":
                atlas = datasets.fetch_atlas_schaefer_2018(n_rois=int(option))
                return atlas["maps"], atlas["labels"]

            elif atlasname == "Talairach atlas":
                atlas = datasets.fetch_atlas_talairach(level_name="hemisphere")
                return atlas["maps"], atlas["labels"]

            elif atlasname == "Yeo (2011) networks":
                atlas = datasets.fetch_atlas_yeo_2011()
                return atlas[option], None

            elif atlasname == "Power et al. (2011)":
                atlas = datasets.fetch_coords_power_2011(legacy_format=False)
                coords = np.vstack((atlas.rois["x"], atlas.rois["y"], atlas.rois["z"])).T
                return coords

            elif atlasname == "Dosenbach et al. (2010)":
                atlas = datasets.fetch_coords_dosenbach_2010(legacy_format=False)
                coords = np.vstack((atlas.rois["x"], atlas.rois["y"], atlas.rois["z"])).T
                return coords, atlas["networks"], atlas["labels"]

            elif atlasname == "Seitzmann et al. (2018)":
                atlas = datasets.fetch_coords_seitzman_2018(legacy_format=False)
                coords = np.vstack((atlas.rois["x"], atlas.rois["y"], atlas.rois["z"])).T
                return coords, atlas["networks"], atlas["regions"]

        else:
            QMessageBox.warning(self, "Error", "Atlas not found")
            return

    def onAtlasChanged(self):
        """
        Atlas dropdown event
        """
        self.parcellationOptions.clear()

        if self.parcellationDropdown.currentText() in self.atlas_options.keys():
            self.parcellationOptions.addItems(self.atlas_options[self.parcellationDropdown.currentText()])

        elif self.parcellationDropdown.currentText() in self.atlas_options_cifti.keys():
            self.parcellationOptions.addItems(self.atlas_options_cifti[self.parcellationDropdown.currentText()])

        else:
            QMessageBox.warning(self, "Error when extracting time series", f"Atlas not found in options list")

        # Enable/disable cleaning options depending on the atlas
        current_atlas = self.parcellationDropdown.currentText()
        if current_atlas in ["Power et al. (2011)", "Seitzmann et al. (2018)", "Dosenbach et al. (2010)"]:
            self.sphereContainer.show()
        else:
            self.sphereContainer.hide()

        return

    # fmriprep dataset functions
    def loadFmriprep(self):
        # Clear plot and layout
        self.plotLogo(self.boldFigure)
        self.boldCanvas.draw()
        self.processingResultsLabel.setText("")

        # Reset data
        self.data.ts_data = None
        self.data.file_data = None
        self.data.file_name = None
        self.data.file_path = None
        self.data.sample_mask = None
        self.data.roi_names = None

        # Hide containers
        self.tsExtractionContainer.hide()
        self.transposeCheckbox.hide()
        self.saveTimeSeriesButton.hide()

        # Open a dialog to select the fmriprep outputs directory
        fmriprep_folder = QFileDialog.getExistingDirectory(self, "Select fMRIprep directory")

        # User canceled the selection
        if not fmriprep_folder:
            return

        # Initialize a fmriprep Layout
        try:
            # Reset GUI elements
            self.graphSlider.setValue(0)
            self.slider.setValue(0)
            self.plotLogo(self.connectivityFigure)
            self.connectivityCanvas.draw()

            # Initialize fmriprep layout
            self.fmriprepContainer.hide()
            self.fmriprep_subjectDropdown.clear()
            self.fmriprep_subjectDropdown.setEnabled(False)
            self.fileNameLabel.setText(f"Initializing fMRIprep layout, please wait...")

            QApplication.processEvents()

            # Load fmriprep layout in a separate thread
            self.workerThread = QThread()
            self.worker = Worker(self.loadFmriprepThread, {'fmriprep_folder': fmriprep_folder})
            self.worker.moveToThread(self.workerThread)

            self.worker.finished.connect(self.workerThread.quit)
            self.workerThread.started.connect(self.worker.run)
            self.worker.result.connect(lambda: self.onFmriprepResult(fmriprep_folder))
            self.worker.error.connect(self.onFmriprepError)
            self.workerThread.start()

        except Exception as e:
            QMessageBox.warning(self, "Load Error", f"Failed to load fMRIprep data: {str(e)}")

    def loadFmriprepThread(self, fmriprep_folder):
        # Get the layout
        self.fmriprep_layout = BIDSLayout(fmriprep_folder, is_derivative=True)

        # Get subjects and update the dropdown
        subjects = self.fmriprep_layout.get_subjects()
        sub_id = [f"sub-{subject}" for subject in subjects]
        self.fmriprep_subjectDropdown.addItems(sub_id)

        # Update the GUI
        self.onFmriprepLayoutChanged()
        self.onFmriprepAtlasChanged()

        return

    def onFmriprepResult(self, fmriprep_folder):
        # Layout loaded successfully
        self.fileNameLabel.setText(f"Loaded fMRIprep data from {fmriprep_folder}")
        self.fmriprep_subjectDropdown.setEnabled(True)
        self.fmriprep_subjectDropdown.show()
        self.fmriprepContainer.show()
        return

    def onFmriprepError(self, error):
        self.fileNameLabel.setText("Failed to load fMRIprep data, please try again.")
        QMessageBox.warning(self, "Failed to load fMRIprep data", f"Error when loading fMRIprep data: {error}")

        return

    def getNifti(self):
        selected_subject = self.fmriprep_subjectDropdown.currentText()
        selected_task = self.fmriprep_taskDropdown.currentText()
        selected_session = self.fmriprep_sessionDropdown.currentText()
        selected_run = self.fmriprep_runDropdown.currentText()

        # Nifti file
        img = self.fmriprep_layout.get(return_type='file', suffix='bold', extension='nii.gz',
                                   subject=selected_subject.split('-')[-1], task=selected_task, run=selected_run, session=selected_session, space='MNI152NLin2009cAsym')

        # result is a list of a single path, we get rid of the list
        if img:
            self.data.file_path = img[0]
            self.data.file_name = img[0].split('/')[-1]
        else:
            self.data.file_name = None

        # Mask file
        mask = self.fmriprep_layout.get(return_type='file', suffix='mask', extension='nii.gz',
                                   subject=selected_subject.split('-')[-1], task=selected_task, run=selected_run, session=selected_session, space='MNI152NLin2009cAsym')
        if mask:
            self.mask_name = mask[0]
        else:
            self.mask_name = None

        # AROMA file
        aroma = self.fmriprep_layout.get(subject=selected_subject.split('-')[-1], session=selected_session, task=selected_task, run=selected_run, suffix='bold', desc='smoothAROMAnonaggr', extension='nii.gz', return_type='file')
        if aroma:
            self.aroma_file = aroma[0]
        else:
            self.aroma_file = None
            self.strategy_checkboxes["ica_aroma"].setEnabled(False)

        return

    def onFmriprepLayoutChanged(self):
        # Disconnect the signal to avoid recursive calls
        dropdowns = [
            self.fmriprep_subjectDropdown,
            self.fmriprep_taskDropdown,
            self.fmriprep_sessionDropdown,
            self.fmriprep_runDropdown,
            self.fmriprep_parcellationOptions
        ]

        for dropdown in dropdowns:
            try:
                dropdown.currentIndexChanged.disconnect(self.onFmriprepLayoutChanged)
            except TypeError:
                pass

        # Disable inputs while loading
        self.fmriprep_taskDropdown.setEnabled(False)
        self.fmriprep_sessionDropdown.setEnabled(False)
        self.fmriprep_runDropdown.setEnabled(False)
        self.fmriprep_parcellationDropdown.setEnabled(False)
        self.fmriprep_parcellationOptions.setEnabled(False)

        QApplication.processEvents()

        """
        The following lines of code get the evailable scans in an hierarchical way
        A full subjects list was previously initialized and doesnt change. Depending on the chosen task, the available sessions and runs are updated.
        """
        try:
            # 1. Get selected subject and sessions
            selected_subject = self.fmriprep_subjectDropdown.currentText()
            subject_id = selected_subject.split('-')[-1]

            # 2. Available tasks for the selected subject
            tasks = self.fmriprep_layout.get_tasks(subject=subject_id)
            current_task = self.fmriprep_taskDropdown.currentText() if self.fmriprep_taskDropdown.count() > 0 else tasks[0]

            self.fmriprep_taskDropdown.clear()
            self.fmriprep_taskDropdown.addItems(tasks)
            if current_task in tasks:
                self.fmriprep_taskDropdown.setCurrentText(current_task)

            # 3. Available sessions for the selected subject and task
            sessions = self.fmriprep_layout.get_sessions(subject=subject_id, task=current_task)
            current_session = self.fmriprep_sessionDropdown.currentText() if (self.fmriprep_sessionDropdown.count() > 0 and self.fmriprep_sessionDropdown.currentText() in sessions) else str(sessions[0])
            session_ids = [f"{session}" for session in sessions]

            self.fmriprep_sessionDropdown.clear()
            self.fmriprep_sessionDropdown.addItems(session_ids)
            if current_session in session_ids:
                self.fmriprep_sessionDropdown.setCurrentText(current_session)

            # 4. Available runs for the selected subject, sessions, and task
            runs = self.fmriprep_layout.get_runs(subject=subject_id, session=current_session, task=current_task)
            current_run = self.fmriprep_runDropdown.currentText() if (self.fmriprep_runDropdown.count() > 0 and self.fmriprep_runDropdown.currentText() in runs) else str(runs[0])
            run_ids = [f"{run}" for run in runs]

            self.fmriprep_runDropdown.clear()
            self.fmriprep_runDropdown.addItems(run_ids)
            if current_run in run_ids:
                self.fmriprep_runDropdown.setCurrentText(current_run)

            # 5. ICA-AROMA files
            aroma_files = self.fmriprep_layout.get(subject=subject_id, session=current_session, task=current_task, suffix='bold', desc='smoothAROMAnonaggr', extension='nii.gz', return_type='file')
            if aroma_files:
                self.aroma_file = aroma_files[0]
            else:
                self.aroma_file = None

            # 6. Get the corresponding nifti file
            self.getNifti()  # get currently selected image

        except Exception as e:
            print(f"Error when updating fMRIprep layout: {str(e)}")
            print("TODO: This might not a problem, but it should be handled properly.")

        """
        End of hierarchical scan selection
        """
        # Enable GUI elements
        self.fmriprep_taskDropdown.setEnabled(True)
        self.fmriprep_sessionDropdown.setEnabled(True)
        self.fmriprep_runDropdown.setEnabled(True)
        self.fmriprep_parcellationDropdown.setEnabled(True)
        self.fmriprep_parcellationOptions.setEnabled(True)

        # Reconnect the signals
        self.fmriprep_subjectDropdown.currentIndexChanged.connect(self.onFmriprepLayoutChanged)
        self.fmriprep_taskDropdown.currentIndexChanged.connect(self.onFmriprepLayoutChanged)
        self.fmriprep_sessionDropdown.currentIndexChanged.connect(self.onFmriprepLayoutChanged)
        self.fmriprep_runDropdown.currentIndexChanged.connect(self.onFmriprepLayoutChanged)
        self.fmriprep_parcellationOptions.currentIndexChanged.connect(self.onFmriprepLayoutChanged)

        return

    def onFmriprepAtlasChanged(self):
        """
        Atlas dropdown event
        """
        self.fmriprep_parcellationOptions.clear()

        if self.fmriprep_parcellationDropdown.currentText() in self.atlas_options.keys():
            self.fmriprep_parcellationOptions.addItems(self.atlas_options[self.fmriprep_parcellationDropdown.currentText()])

        elif self.fmriprep_parcellationDropdown.currentText() in self.atlas_options_cifti.keys():
            self.fmriprep_parcellationOptions.addItems(self.atlas_options_cifti[self.fmriprep_parcellationDropdown.currentText()])

        else:
            QMessageBox.warning(self, "Error when extracting time series", f"Atlas not found in options list")

        # Enable/disable cleaning options depending on the atlas
        current_atlas = self.fmriprep_parcellationDropdown.currentText()
        if current_atlas in ["Power et al. (2011)", "Seitzmann et al. (2018)", "Dosenbach et al. (2010)"]:
            self.fmriprep_sphereContainer.show()
        else:
            self.fmriprep_sphereContainer.hide()

        return

    def loadConfounds(self):
        confoundsWidget = QWidget()
        layout = QVBoxLayout(confoundsWidget)
        self.strategy_layouts = {}  # Dict to hold the layouts for each strategy

        for key, param in self.confound_options.items():
            h_layout = QHBoxLayout()
            label = QLabel(f"{key}:")
            label.setFixedWidth(70)
            h_layout.addWidget(label)
            h_layout.setContentsMargins(0, 0, 0, 0)

            # Info button content
            if key != "Cleaning\nstrategy":
                label.setFixedWidth(145)
                info_text = self.cleaning_info[key]
                info_button = InfoButton(info_text)

            # Special cases
            if key == "n_compcor":
                input_widget = CustomSpinBox(special_value="all", min=0, max=100)
                input_widget.setObjectName(f"{key}_input")
                h_layout.addWidget(input_widget)
                h_layout.setObjectName("compcor")
                h_layout.addWidget(info_button)

            elif key in ["fd_threshold", "std_dvars_threshold"]:
                input_widget = QDoubleSpinBox()
                input_widget.setRange(0.0, 999.0)
                input_widget.setValue(param)
                input_widget.setSingleStep(0.1)
                input_widget.setObjectName(f"{key}_input")
                h_layout.addWidget(input_widget)
                h_layout.setObjectName("scrub")
                h_layout.addWidget(info_button)

            elif key == "Cleaning\nstrategy":
                strategy_group = QGroupBox()
                strategy_layout = QGridLayout(strategy_group)
                self.strategy_checkboxes = {}
                row, col = 0, 0
                for strategy in param:
                    checkbox = QCheckBox(strategy)
                    checkbox.setObjectName(strategy)
                    checkbox.stateChanged.connect(self.updateCleaningOptions)
                    self.strategy_checkboxes[strategy] = checkbox
                    strategy_layout.addWidget(checkbox, row, col)
                    col += 1
                    if col == 4:
                        col = 0
                        row += 1
                h_layout.addWidget(strategy_group)
                input_widget = None

            else:
                if isinstance(param, list):
                    input_widget = QComboBox()
                    input_widget.addItems(param)

                elif isinstance(param, int):
                    input_widget = QSpinBox()
                    input_widget.setRange(0, 999)
                    input_widget.setValue(param)
                elif isinstance(param, float):
                    input_widget = QDoubleSpinBox()
                    input_widget.setRange(0.0, 999.0)
                    input_widget.setValue(param)
                elif isinstance(param, str):
                    input_widget = QTextEdit()
                    input_widget.setText(param)
                    input_widget.setEnabled(False)
                else:
                    input_widget = QLineEdit("ERROR: Unsupported type")

                h_layout.setObjectName(key)
                if input_widget is not None:
                    h_layout.addWidget(input_widget)
                h_layout.addWidget(info_button)

            # Store the layout in the dictionary
            self.strategy_layouts[key] = h_layout
            if key != "Cleaning\nstrategy":
                self.hideCleaningLayout(h_layout)  # Initially hide all options except "strategy"
            layout.addLayout(h_layout)

        self.dynamic_container = QWidget()
        self.dynamic_layout = QVBoxLayout(self.dynamic_container)
        layout.addWidget(self.dynamic_container)

        return confoundsWidget

    def updateCleaningOptions(self):
        for strategy, checkbox in self.strategy_checkboxes.items():
            if strategy in self.strategy_layouts:
                if checkbox.isChecked():
                    self.showCleaningLayout(self.strategy_layouts[strategy])
                else:
                    self.hideCleaningLayout(self.strategy_layouts[strategy])

            # Compcor requires high pass
            if self.strategy_checkboxes["compcor"].isChecked():
                self.strategy_checkboxes["high_pass"].setChecked(True)

        # Handle special cases for scrub and compcor
        if "scrub" in self.strategy_checkboxes and self.strategy_checkboxes["scrub"].isChecked():
            self.showCleaningLayout(self.strategy_layouts["fd_threshold"])
            self.showCleaningLayout(self.strategy_layouts["std_dvars_threshold"])
        else:
            self.hideCleaningLayout(self.strategy_layouts["fd_threshold"])
            self.hideCleaningLayout(self.strategy_layouts["std_dvars_threshold"])

        if "compcor" in self.strategy_checkboxes and self.strategy_checkboxes["compcor"].isChecked():
            self.showCleaningLayout(self.strategy_layouts["n_compcor"])
        else:
            self.hideCleaningLayout(self.strategy_layouts["n_compcor"])

    def showCleaningLayout(self, layout):
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item.widget():
                item.widget().show()
            elif item.layout():
                self.showLayout(item.layout())

    def hideCleaningLayout(self, layout):
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item.widget():
                item.widget().hide()
            elif item.layout():
                self.hideLayout(item.layout())

    def collectCleaningArguments(self):
        strategy_list = []
        for strategy, checkbox in self.strategy_checkboxes.items():
            if checkbox.isChecked() and strategy != "demean":
                strategy_list.append(strategy)

        args = {
            "strategy": None,
            "motion": None,
            "wm_csf": None,
            "compcor": None,
            "n_compcor": None,
            "global_signal": None,
            "ica_aroma": None,
            "scrub": None,
            "fd_threshold": None,
            "std_dvars_threshold": None,
            "demean": None,
        }

        # Get the selected strategies
        args["strategy"] = strategy_list

        # Set specific options for each strategy
        for strategy in strategy_list:
            if strategy == "motion":
                args[strategy] = self.strategy_layouts[strategy].itemAt(1).widget().currentText()
            elif strategy == "wm_csf":
                args[strategy] = self.strategy_layouts[strategy].itemAt(1).widget().currentText()
            elif strategy == "compcor":
                args[strategy] = self.strategy_layouts[strategy].itemAt(1).widget().currentText()
                args["n_compcor"] = self.strategy_layouts["n_compcor"].itemAt(1).widget().get_value()
            elif strategy == "global_signal":
                args[strategy] = self.strategy_layouts[strategy].itemAt(1).widget().currentText()
            elif strategy == "ica_aroma":
                args[strategy] = self.strategy_layouts[strategy].itemAt(1).widget().currentText()
            elif strategy == "scrub":
                args[strategy] = self.strategy_layouts[strategy].itemAt(1).widget().value()
                args["fd_threshold"] = self.strategy_layouts["fd_threshold"].itemAt(1).widget().value()
                args["std_dvars_threshold"] = self.strategy_layouts["std_dvars_threshold"].itemAt(1).widget().value()

        # Demean is not a strategy, but a separate option
        args["demean"] = True if self.strategy_checkboxes["demean"].isChecked() else False

        return args

    # Parcellation and cleaning (nifi)
    def calculateTimeSeries(self):
        # Disable the buttons
        self.parcellationCalculateButton.setEnabled(False)
        self.fmriprep_calculateButton.setEnabled(False)

        QApplication.processEvents()

        # Load fmriprep layout in a separate thread
        self.workerThread = QThread()

        if self.fmriprep_layout is None:
            self.worker = Worker(self.calculateTimeSeriesThread, {"img_path": self.data.file_path,
                                                                  "atlas": self.parcellationDropdown.currentText(),
                                                                  "option": self.parcellationOptions.currentText()})
        else:
            self.worker = Worker(self.calculateTimeSeriesThread, {"img_path": self.data.file_path,
                                                                  "atlas": self.fmriprep_parcellationDropdown.currentText(),
                                                                  "option": self.fmriprep_parcellationOptions.currentText()})

        self.worker.moveToThread(self.workerThread)
        self.worker.finished.connect(self.workerThread.quit)
        self.workerThread.started.connect(self.worker.run)
        self.worker.result.connect(self.handleTimeSeriesResult)
        self.worker.error.connect(self.handleTimeSeriesError)
        self.workerThread.start()

        self.processingResultsLabel.setText(f'Calculating time series, please wait...')

        return

    def calculateTimeSeriesThread(self, **params):
        img_path = params["img_path"]
        atlas = params["atlas"]
        option = params["option"]
        mask = None
        confounds_df = None
        self.data.sample_mask

        # Collect cleaning arguments
        if self.fmriprep_layout is None:

            # Discard the initial n volumes
            discarded_values = self.discardSpinBox.value()
            if discarded_values > 0:
                if self.data.file_name.endswith(".dtseries.nii") or self.data.file_name.endswith(".ptseries.nii"):
                    img = nib.load(self.data.file_path)
                    img_data = img.get_fdata()
                    img_data = img_data[discarded_values:, :]
                    img = nib.Cifti2Image(img_data, header=img.header)  
                
                elif self.data.file_name.endswith(".nii") or self.data.file_name.endswith(".nii.gz"):
                    img = nib.load(self.data.file_path)
                    img_data = img.get_fdata()
                    img_data = img_data[:,:,:, discarded_values:]
                    img = nib.Nifti1Image(img_data, affine=img.affine, header=img.header)

                else:
                    QMessageBox.warning(self, "Error when extracting time series", f"File type not supported.")
                    return
            else:
                img = nib.load(self.data.file_path)
                
            radius = self.sphereRadiusSpinbox.value() if self.sphereRadiusSpinbox.value() > 0 else None # none is single voxel
            allow_ovelap = self.overlapCheckbox.isChecked()

            standardize = self.standardizeCheckbox.isChecked()
            standardize_confounds = True if standardize else False
            detrend = self.detrendCheckbox.isChecked()
            high_variance_confounds = self.highVarianceCheckbox.isChecked()
            smoothing_fwhm = self.smoothingSpinbox.value() if self.smoothingSpinbox.value() > 0 else None

            high_pass = self.highPassCutoff.value() if self.highPassCutoff.value() > 0 else None
            low_pass = self.lowPassCutoff.value() if self.lowPassCutoff.value() > 0 else None
            tr = self.trValue.value() if self.trValue.value() > 0 else None

            # GSR
            fdata = img.get_fdata()
            if self.gsrCheckbox.isChecked():
                confounds_df = pd.DataFrame()
                
                if np.ndim(fdata) == 4:
                    confounds_df["global_signal"] = np.mean(fdata, axis=(0, 1, 2))
                else:
                    confounds_df["global_signal"] = np.mean(fdata, axis=1)
            else:
                confounds_df = None

        else:
            img = nib.load(self.data.file_path)

            radius = self.fmriprep_sphereRadiusSpinbox.value() if self.fmriprep_sphereRadiusSpinbox.value() > 0 else None # none is single voxel
            allow_ovelap = self.fmriprep_overlapCheckbox.isChecked()

            standardize = self.fmriprep_standardizeCheckbox.isChecked()
            standardize_confounds = True if standardize else False
            detrend = self.fmriprep_detrendCheckbox.isChecked()
            high_variance_confounds = self.fmriprep_highVarianceCheckbox.isChecked()
            smoothing_fwhm = self.fmriprep_smoothingSpinbox.value() if self.fmriprep_smoothingSpinbox.value() > 0 else None

            high_pass = self.fmriprep_highPassCutoff.value() if self.fmriprep_highPassCutoff.value() > 0 else None
            low_pass = self.fmriprep_lowPassCutoff.value() if self.fmriprep_lowPassCutoff.value() > 0 else None
            tr = self.fmriprep_trValue.value() if self.fmriprep_trValue.value() > 0 else None

            args = self.collectCleaningArguments()
            confounds_df, self.data.sample_mask = load_confounds(img_path, **args)
            mask = self.mask_name

            # Workaround for nilearn bug, will be fixed in the next nilearn release
            if confounds_df is not None and confounds_df.empty:
                confounds_df = None

        if atlas in ["Power et al. (2011)", "Seitzmann et al. (2018)", "Dosenbach et al. (2010)"]:
            if atlas == "Power et al. (2011)":
                rois = self.fetchAtlas(atlas, option)
            else:
                rois, _, self.data.roi_names = self.fetchAtlas(atlas, option)

            masker = maskers.NiftiSpheresMasker(seeds=rois, mask_img=mask, radius=radius, allow_overlap=allow_ovelap,
                                                standardize=standardize, standardize_confounds=standardize_confounds, detrend=detrend, 
                                                smoothing_fwhm=smoothing_fwhm, high_variance_confounds=high_variance_confounds,
                                                low_pass=low_pass, high_pass=high_pass, t_r=tr)
            time_series = masker.fit_transform(img, confounds=confounds_df)

        # Select the correct atlas for Schaefer and Glasser
        elif (atlas.startswith("Schaefer") or atlas.startswith("Glasser")) and atlas != "Schaefer et al. (2018)":
            atlas_string = self.atlas_map.get(f"{atlas} {option}", None)
            time_series_raw = cifti.parcellate(img, atlas=atlas_string)
            time_series = utils.clean(time_series_raw, standardize=standardize, detrend=detrend, high_pass=high_pass, low_pass=low_pass, t_r=tr)
        else:
            atlas, _ = self.fetchAtlas(atlas, option)
            masker = maskers.NiftiLabelsMasker(labels_img=atlas, mask_img=mask, background_label=0, 
                                               standardize=standardize, standardize_confounds=standardize_confounds, 
                                               detrend=detrend, smoothing_fwhm=smoothing_fwhm, high_variance_confounds=high_variance_confounds,
                                               low_pass=low_pass, high_pass=high_pass, t_r=tr)
            time_series = masker.fit_transform(img, confounds=confounds_df)

        self.data.ts_data = time_series

        return

    def handleTimeSeriesResult(self):
        # Discard flagged volumes
        original_data_shape = self.data.ts_data.shape

        if self.fmriprep_layout is not None:
            self.plotCarpet()
            self.data.ts_data = self.data.ts_data[self.data.sample_mask,:]
        else:
            self.plotCarpet()

        # Set the labels
        n_removed_volumes = original_data_shape[0] - self.data.ts_data.shape[0]
        if n_removed_volumes > 0:
            self.processingResultsLabel.setText(f'Time series data with shape {self.data.ts_data.shape} is ready for connectivity analysis.\n{n_removed_volumes} volumes were removed.')
        else:
            self.processingResultsLabel.setText(f'Time series data with shape {self.data.ts_data.shape} is ready for connectivity analysis.')
        
        self.time_series_textbox.setText(self.data.file_name)
        self.connectivityFileNameLabel.setText(f"Time series data with shape {self.data.ts_data.shape} is available for connectivity analysis.")

        # Re-enable buttons
        self.parcellationCalculateButton.setEnabled(True)
        self.fmriprep_calculateButton.setEnabled(True)

        self.saveTimeSeriesButton.show()

        return

    def handleTimeSeriesError(self, error):
        # Handles errors in the worker thread
        QMessageBox.warning(self, "Error when extracting time series", f"Error when extracting time series: {error}")

        self.parcellationCalculateButton.setEnabled(True)
        self.fmriprep_calculateButton.setEnabled(True)
        return

    # Cleaning only
    def cleanTimeSeries(self):
        self.cleanButton.setEnabled(False)
        QApplication.processEvents()

        # Clean in a separate thread
        self.workerThread = QThread()
        self.worker = Worker(self.cleanTimeSeriesThread, {})
        self.worker.moveToThread(self.workerThread)
        self.worker.finished.connect(self.workerThread.quit)
        self.workerThread.started.connect(self.worker.run)
        self.worker.result.connect(self.handleCleaningResult)
        self.worker.error.connect(self.handleCleaningError)
        self.workerThread.start()

        return
    
    def cleanTimeSeriesThread(self):
        detrend = self.detrendCheckbox.isChecked()
        high_pass = self.highPassCutoff.value() if self.highPassCutoff.value() > 0 else None
        low_pass = self.lowPassCutoff.value() if self.lowPassCutoff.value() > 0 else None
        tr = self.trValue.value() if self.trValue.value() > 0 else None
        
        # Copy the data to keep the original
        self.data.ts_data = self.data.file_data.copy()

        # Discard initial n volumes
        discarded_values = self.discardSpinBox.value()

        if discarded_values:
            if np.ndim(self.data.ts_data) == 3:
                temp_data = np.zeros((self.data.ts_data.shape[0], self.data.ts_data.shape[1]
                                      - discarded_values, self.data.ts_data.shape[2]))
                for i in range(self.data.ts_data.shape[0]):
                    temp_data[i,:,:] = self.data.ts_data[i,discarded_values:,:]
                self.data.ts_data = temp_data
            else:
                self.data.ts_data = self.data.ts_data[discarded_values:,:]

        # Calculate global signal regressor
        if self.gsrCheckbox.isChecked():
            if np.ndim(self.data.ts_data) == 3:
                confounds_df_list = []
                for i in range(self.data.ts_data.shape[0]):
                    confounds_df = pd.DataFrame()
                    confounds_df["global_signal"] = np.mean(self.data.ts_data[i,:,:], axis=1)
                    confounds_df_list.append(confounds_df)
            else:
                confounds_df = pd.DataFrame()
                confounds_df["global_signal"] = np.mean(self.data.ts_data, axis=1)
        else:
            if np.ndim(self.data.ts_data) == 3:
                confounds_df_list = [None] * self.data.ts_data.shape[0]
            else:
                confounds_df = None

        # Standardize
        if self.standardizeCheckbox.isChecked():
            standardize = True
            standardize_confounds = True
        else:
            standardize = False
            standardize_confounds = False

        # Cleaning
        if np.ndim(self.data.ts_data) == 3:
            for i in range(self.data.ts_data.shape[0]):
                self.data.ts_data[i,:,:] = utils.clean(self.data.ts_data[i,:,:], standardize=standardize,
                                                         confounds=confounds_df_list[i], standardize_confounds=standardize_confounds,
                                                         detrend=detrend, high_pass=high_pass, low_pass=low_pass, t_r=tr)
        else:
            self.data.ts_data = utils.clean(self.data.ts_data, standardize=standardize,
                                               confounds=confounds_df, standardize_confounds=standardize_confounds,
                                               detrend=detrend, high_pass=high_pass, low_pass=low_pass, t_r=tr)
        
        return

    def handleCleaningResult(self):
        self.processingResultsLabel.setText(f'Time series data with shape {self.data.ts_data.shape} is ready for connectivity analysis')
        self.time_series_textbox.setText(self.data.file_name)
        self.plotCarpet()
        self.cleanButton.setEnabled(True)
        self.resetButton.setEnabled(True)
        self.saveTimeSeriesButton.show()

    def handleCleaningError(self, error):
        # Handles errors in the worker thread
        QMessageBox.warning(self, "Error when cleaning time series", f"Error when cleaning time series: {error}")

        self.cleanButton.setEnabled(True)
        return

    def resetTimeSeries(self):
        self.data.ts_data = self.data.file_data.copy()
        self.data.sample_mask = None
        self.resetButton.setEnabled(False)
        self.processingResultsLabel.setText(f'Time series data with shape {self.data.ts_data.shape} is ready for connectivity analysis.')

        if isinstance(self.data.file_data, np.ndarray):
            self.plotCarpet()
        else:
            self.plotLogo(self.boldFigure)
            self.boldCanvas.draw()
        
        return

    # Plotting
    def plotCarpet(self):
        # Clear the current plot
        self.boldFigure.clear()
        ax = self.boldFigure.add_subplot(111)
        
        # Custom colormap with nans being red
        cmap = plt.cm.gray
        cmap.set_bad(color='red')

        if np.ndim(self.data.ts_data) == 3:
            current_subject = int(self.subjectDropdown.currentText())
            ts = self.data.ts_data[current_subject].copy()
        else:
            ts = self.data.ts_data.copy()
        
        # Show scrubbed volumes
        if self.data.sample_mask is not None:
            # We have data with missing scans (non-steady states or scrubbing)
            # Create a mask of the same shape as ts and set the values to 0 where sample_mask is False
            mask = np.ones(ts.shape, dtype=bool)
            mask[self.data.sample_mask] = False
            ts[mask] = np.nan

            ts = np.ma.masked_where(mask, ts)

        # Prepend a red block for discarded volumes
        if self.fmriprep_layout is None:
            discarded_values = self.discardSpinBox.value()
        else:
            discarded_values = self.fmriprep_discardSpinBox.value()
        
        red_block = np.full((discarded_values, ts.shape[1]), np.nan)
        ts = np.vstack((red_block, ts))

        # Plot the data
        im = ax.imshow(ts, cmap=cmap, aspect='auto')
        ax.set_xlabel("ROIs")
        ax.set_ylabel("TRs")

        # Create the colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.15)
        cbar = self.boldFigure.colorbar(im, cax=cax)
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1f}'))

        # Adjust and draw
        self.boldFigure.set_facecolor('#f3f1f5')
        self.boldFigure.tight_layout()
        self.boldCanvas.draw()

    def plotLogo(self, figure=None):
        with importlib_resources.path("comet.data.img", "logo.png") as file_path:
            logo = imread(file_path)

        figure.clear()
        ax = figure.add_subplot(111)
        ax.set_axis_off()
        ax.imshow(logo)

        figure.set_facecolor('#f4f1f6')
        figure.tight_layout()

    # Saving
    def saveTimeSeries(self):
        """
        Save the time series data to a file
        """
        if self.data.ts_data is None:
            QMessageBox.warning(self, "Output Error", "No time series data available to save.")
            return

        base, ext = os.path.splitext(self.data.file_name)

        if ext == ".nii" or ext == ".gz":
            ext = ".txt"

        if self.data.ts_data.ndim == 3:
            default_file_name = f"{base}_processed.npy"
            fileFilter = ("All supported files (*.mat *.npy);;"
                        "MAT file (*.mat);;"
                        "NumPy file (*.npy)")
        else:
            default_file_name = f"{base}_processed{ext}"
            fileFilter = ("All supported files (*.txt *.mat *.npy *.tsv);;"
                        "Text file (*.txt);;"
                        "MAT file (*.mat);;"
                        "NumPy file (*.npy);;"
                        "TSV file (*.tsv)")

        filePath, _ = QFileDialog.getSaveFileName(self, "Save File", default_file_name, fileFilter)
        if not filePath:
            return

        if filePath.lower().endswith('.mat'):
            savemat(filePath, {"time_series": self.data.ts_data})
        elif filePath.lower().endswith(('.txt', '.tsv')):
            np.savetxt(filePath, self.data.ts_data, delimiter="\t")
        elif filePath.lower().endswith('.npy'):
            np.save(filePath, self.data.ts_data)
        else:
            np.save(filePath, self.data.ts_data)

    """
    Connectivity tab
    """
    def saveConnectivity(self):
        if self.data.dfc_data is None:
            QMessageBox.warning(self, "Output Error", "No dFC data available to save.")
            return

        # Open a file dialog to specify where to save the file
        filePath, _ = QFileDialog.getSaveFileName(self, "Save File", "", "MAT Files (*.mat)")

        if filePath:
            # Ensure the file has the correct extension
            if not filePath.endswith('.mat'):
                filePath += '.mat'

            # Save the the current data object to a .mat file
            try:
                data_dict = {}
                for field in self.data.__dataclass_fields__:
                    value = getattr(self.data, field)

                    if isinstance(value, np.ndarray):
                        data_dict[field] = value
                    elif isinstance(value, dict):
                        # Ensure all dict values are appropriately converted
                        converted_dict = {}
                        for k, v in value.items():
                            if isinstance(v, np.ndarray):
                                converted_dict[k] = v
                            elif v is None:
                                converted_dict[k] = np.array([])
                            else:
                                converted_dict[k] = v
                        data_dict[field] = converted_dict
                    elif value is None:
                        data_dict[field] = np.array([])
                    elif field == 'dfc_instance':
                        pass
                    else:
                        data_dict[field] = value

                savemat(filePath, data_dict)

            except Exception as e:
                QMessageBox.warning(self, "Output Error", f"Error saving data: {e}")

        return

    def onMethodCombobox(self, methodName=None):
        self.data.clear_dfc_data()
        
        # Clear old variables and data
        self.clearParameters(self.parameterLayout)

        # Hide widgets below the plot
        self.slider.hide()
        self.navButtonContainer.hide()
        self.roiSelectorWidget.hide()

        # Return if no methods are available
        if methodName == None or methodName == "Use checkboxes to get available methods":
            return

        # Get selected connectivity method
        self.data.dfc_instance = getattr(connectivity, self.class_info.get(methodName), None) # the actual class
        self.data.dfc_name = self.class_info.get(methodName) # class name

        # Create and get new parameter layout
        #self.data.dfc_data = None
        self.data.dfc_params = {}
        self.initParameters(self.data.dfc_instance)
        self.parameterLayout.addStretch(1) # Stretch to fill empty space
        self.getParameters()

        # See if some data has previously been calculated, we change the paramters to this
        previous_data = self.data_storage.check_previous_data(self.data.dfc_name)
        if previous_data is not None:
            self.data = previous_data
            self.setParameters()
            self.slider.show()
            self.calculatingLabel.setText(f"Loaded {self.data.dfc_name} with shape {self.data.dfc_data.shape}")

            # Plot the data
            self.plotConnectivity()
            self.plotDistribution()
            self.plotTimeSeries()

            # Update the slider
            total_length = self.data.dfc_data.shape[2] if len(self.data.dfc_data.shape) == 3 else 0
            position_text = f"t = {self.slider.value()} / {total_length-1}" if len(self.data.dfc_data.shape) == 3 else " static "
            self.positionLabel.setText(position_text)
            self.slider.setValue(self.slider.value())

            self.onTabChanged()

        # If connectivity data does not exist we reset the figure and slider to prepare for a new calculation
        # This also indicates to the user that this data was not yet calculated/saved
        else:
            self.connectivityFigure.clear()
            self.plotLogo(self.connectivityFigure)
            self.connectivityCanvas.draw()
            self.distributionFigure.clear()
            self.distributionCanvas.draw()
            self.timeSeriesFigure.clear()
            self.timeSeriesCanvas.draw()

            position_text = f"no data available"
            self.positionLabel.setText(position_text)
            self.slider.setValue(self.slider.value())
            self.slider.hide()

        self.update()

    def updateMethodComboBox(self):

        def shouldIncludeClass(className):
            if self.continuousCheckBox.isChecked() and className.startswith("CONT"):
                    return True
            if self.stateBasedCheckBox.isChecked() and className.startswith("STATE"):
                    return True
            if self.staticCheckBox.isChecked() and className.startswith("STATIC"):
                    return True
            return False

        # Disconnect existing connections to avoid multiple calls
        try:
            self.methodComboBox.currentTextChanged.disconnect(self.onMethodCombobox)
        except TypeError:
            pass

        # Update the combobox
        methods_to_include = [method for method in self.reverse_connectivityMethods.keys() if shouldIncludeClass(method)]

        self.methodComboBox.clear()
        self.methodComboBox.addItems(methods_to_include)

        # Adjust combobox width
        if self.methodComboBox.count() > 0:
            font_metrics = QFontMetrics(self.methodComboBox.font())
            longest_text_width = max(font_metrics.boundingRect(self.methodComboBox.itemText(i)).width() for i in range(self.methodComboBox.count()))
            minimum_width = longest_text_width + 30
            self.methodComboBox.setMinimumWidth(minimum_width)
        else:
            default_minimum_width = 300
            self.methodComboBox.setMinimumWidth(default_minimum_width)

        # Reconnect the signal
        self.methodComboBox.currentTextChanged.connect(self.onMethodCombobox)

        # Trigger the onMethodCombobox for the initial setup
        if methods_to_include:
            self.onMethodCombobox(methods_to_include[0])

    # Calculations
    def calculateConnectivity(self):
        # Check if time series data is available
        if self.data.ts_data is None:
            QMessageBox.warning(self, "Load Error", f"No time series data is currently available.")
            return

        # Get the current parameters from the UI for the upcoming calculation
        self.getParameters()

        # Process all pending events
        QApplication.processEvents()

        # Start worker thread for dFC calculations and submit for calculation
        self.fcThread = QThread()
        self.fcWorker = Worker(self.calculateConnectivityThread, self.data.dfc_params)
        self.fcWorker.moveToThread(self.fcThread)

        self.fcWorker.finished.connect(self.fcThread.quit)
        self.fcWorker.finished.connect(self.fcWorker.deleteLater)
        self.fcThread.finished.connect(self.fcThread.deleteLater)
        self.fcWorker.result.connect(self.handleConnectivityResult)
        self.fcWorker.error.connect(self.handleConnectivityError)

        self.fcThread.started.connect(self.fcWorker.run)
        self.fcThread.start()

        self.calculatingLabel.setText(f"Calculating {self.methodComboBox.currentText()}, please wait...")
        self.calculateConnectivityButton.setEnabled(False)

    def calculateConnectivityThread(self, **params):
        keep_in_memory = self.keepInMemoryCheckbox.isChecked()

        # Check if data already exists
        existing_data = self.data_storage.check_for_identical_data(self.data)
        if existing_data is not None:
            return existing_data

        # Remove keys not allowed for calculation
        clean_params = params.copy()
        clean_params.pop('parcellation', None)

        # Data does not exist, perform calculation
        connectivityObject = self.data.dfc_instance(**clean_params)
        result = connectivityObject.estimate()

        self.data.dfc_data = result
        self.data.dfc_params = params
        self.data.dfc_state_tc = None
        self.data.dfc_edge_ts = None
        self.data.dfc_states = None

        # Edge time series can be contain either eFC (3D) or eTS (2D)
        if self.data.dfc_instance == connectivity.EdgeConnectivity:
            self.data.dfc_edge_ts = self.data.dfc_data if clean_params["method"] == "eTS" else None

        # Result is state-based connectivity
        if self.data.dfc_instance in [connectivity.SlidingWindowClustering, 
                                      connectivity.KSVD, 
                                      connectivity.CoactivationPatterns, 
                                      connectivity.ContinuousHMM, 
                                      connectivity.DiscreteHMM]:
            self.data.dfc_state_tc = result[0]
            self.data.dfc_states = result[1]
            self.data.dfc_data = result[1]

        # Store in memory if checkbox is checked
        if keep_in_memory:
            # Update the dictionary entry for the selected_class_name with the new data and parameters
            self.data_storage.add_data(self.data)

        return self.data

    def handleConnectivityResult(self):
        # Update the sliders and text
        if self.data.dfc_data is not None:
            if self.data.dfc_states is not None:
                self.calculatingLabel.setText(f"Calculated {self.data.dfc_name} with for {self.data.dfc_state_tc.shape[1]} subjects and {self.data.dfc_states.shape[0]} states.")
            else:
                self.calculatingLabel.setText(f"Calculated {self.data.dfc_name} with shape {self.data.dfc_data.shape}.")

            if len(self.data.dfc_data.shape) == 3:
                self.slider.show()
                self.rowSelector.setMaximum(self.data.dfc_data.shape[0] - 1)
                self.colSelector.setMaximum(self.data.dfc_data.shape[1] - 1)
                self.rowSelector.setValue(1)

            # Update time label
            total_length = self.data.dfc_data.shape[2] if len(self.data.dfc_data.shape) == 3 else 0

            if self.currentTabIndex == 0 or self.currentTabIndex == 2:
                if self.data.dfc_states is None:
                    position_text = f"t = {self.slider.value()} / {total_length-1}" if len(self.data.dfc_data.shape) == 3 else " static "
                else:
                    position_text = f"State {self.slider.value()}"
            else:
                position_text = ""

            self.positionLabel.setText(position_text)
            self.backupSliderValueTab02 = 0
            self.backupSliderValueTab1 = 0
            self.slider.setValue(0)

        # Plot
        self.plotConnectivity()
        self.plotDistribution()
        self.plotTimeSeries()

        self.calculateConnectivityButton.setEnabled(True)
        self.onTabChanged()

        # Init the graph tab
        self.data.graph_data = self.data.dfc_data.copy()
        self.data.graph_raw  = self.data.dfc_data.copy()

        self.graphFileNameLabel.setText(f"Using connectivity data with shape {self.data.graph_data.shape}")

        total_length = self.data.graph_data.shape[2] if np.ndim(self.data.graph_data) == 3 else 0
        position_text = f"t = {self.graphSlider.value()} / {total_length-1}" if len(self.data.graph_data.shape) == 3 else " static "
        self.graphPositionLabel.setText(position_text)
        self.currentGraphOption = None
        
        self.plotGraphMatrix()
        self.onGraphCombobox()
        self.addOptionButton.setEnabled(True)
        self.graphUsePushButton.setEnabled(True)

        # Update the GUI
        self.update()

        return

    def handleConnectivityError(self, error):
        # Handles errors in the worker thread
        QMessageBox.warning(self, "Calculation Error", f"Error occurred during calculation: {error}")
        self.calculateConnectivityButton.setEnabled(True)
        self.data.clear_dfc_data()
        self.positionLabel.setText("no data available")
        self.plotLogo(self.connectivityFigure)
        self.connectivityCanvas.draw()

    # Parameters
    def initParameters(self, class_instance):
        # Set up the parameter labels and boxes
        labels = []

        # Calculate the maximum label width (just a visual thing)
        max_label_width = 0
        init_signature = inspect.signature(class_instance.__init__)
        type_hints = get_type_hints(class_instance.__init__)
        font_metrics = QFontMetrics(self.font())
        for param in init_signature.parameters.values():
            label_width = font_metrics.boundingRect(f"{self.param_names[param.name]}:").width()
            max_label_width = max(max_label_width, label_width)

        # Special case for 'time_series' parameter as this is created from the loaded file
        # Add label for time_series
        time_series_label = QLabel("Time series:")
        time_series_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        time_series_label.setMinimumSize(time_series_label.sizeHint())
        labels.append(time_series_label)

        self.time_series_textbox.setPlaceholderText("No data available")
        if self.data.file_name:
            self.time_series_textbox.setText(self.data.file_name)
        self.time_series_textbox.setEnabled(True)

        # Create info button for time_series
        time_series_info_text = "2D time series loaded from file (time has to be the first dimension)."
        time_series_info_button = InfoButton(time_series_info_text)

        time_series_layout = QHBoxLayout()
        time_series_layout.addWidget(time_series_label)
        time_series_layout.addWidget(self.time_series_textbox)
        time_series_layout.addWidget(time_series_info_button)
        self.parameterLayout.addLayout(time_series_layout)

        # Adjust max width for aesthetics
        max_label_width += 5
        time_series_label.setFixedWidth(max_label_width)

        for name, param in init_signature.parameters.items():
            if param.name not in ['self', 'time_series', 'tril', 'standardize', 'params']:
                param_type = type_hints.get(name)
                # Create label for parameter
                param_label = QLabel(f"{self.param_names[param.name]}:")
                param_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
                param_label.setMinimumSize(param_label.sizeHint())
                param_label.setFixedWidth(max_label_width)
                labels.append(param_label)

                # Determine the widget type based on the parameter
                # Dropdown for boolean parameters
                if param_type == bool:
                    param_input_widget = QComboBox()
                    param_input_widget.addItems(["True", "False"])

                    default_index = param_input_widget.findText(str(param.default))
                    param_input_widget.setCurrentIndex(default_index)
                    param_input_widget.setEnabled(True)

                # Dropdown for parameters with predefined options
                elif get_origin(type_hints.get(name)) is Literal:
                    options = type_hints.get(name).__args__
                    param_input_widget = QComboBox()
                    param_input_widget.addItems([str(option) for option in options])

                    default_index = param_input_widget.findText(param.default)
                    param_input_widget.setCurrentIndex(default_index)
                    param_input_widget.setEnabled(True)

                # Spinbox for integer parameterss
                elif param_type == int:
                    param_input_widget = QSpinBox()
                    param_input_widget.setMaximum(10000)
                    param_input_widget.setMinimum(-10000)
                    param_input_widget.setSingleStep(1)

                    param_input_widget.setValue(int(param.default) if param.default != inspect.Parameter.empty else 0)
                    param_input_widget.setEnabled(True)

                # Spinbox for float parameters
                elif param_type == float:
                    param_input_widget = QDoubleSpinBox()
                    param_input_widget.setMaximum(10000.0)
                    param_input_widget.setMinimum(-10000.0)
                    param_input_widget.setSingleStep(0.1)

                    param_input_widget.setValue(float(param.default) if param.default != inspect.Parameter.empty else 0.0)
                    param_input_widget.setEnabled(True)

                # Text field for other types of parameters
                else:
                    param_input_widget = QLineEdit(str(param.default) if param.default != inspect.Parameter.empty else "")
                    param_input_widget.setEnabled(True)

                # Create info button with tooltip
                info_text = self.info_options.get(param.name, "No information available")
                info_button = InfoButton(info_text)

                # Create layout for label, widget, and info button
                param_layout = QHBoxLayout()
                param_layout.addWidget(param_label)
                param_layout.addWidget(param_input_widget)
                param_layout.addWidget(info_button)

                # Add the layout to the main parameter layout
                self.parameterLayout.addLayout(param_layout)

    def getParameters(self):
        # Get the time series and parameters (from the UI) for the selected connectivity method and store them in a dictionary
        # TODO: Should I allow multiple subjects for continuous and static FC as well? This could be added here
        self.data.dfc_params['time_series'] = self.data.ts_data # Time series data

        # Converts string to boolean, float, or retains as string if conversion is not applicable
        def convert_value(value):
            if value.lower() in ['true', 'false']:
                return value.lower() == 'true'
            try:
                return float(value)
            except ValueError:
                return value

        # Gets the value from the widget based on its type
        def get_widget_value(widget):
            if isinstance(widget, QLineEdit):
                return widget.text()
            elif isinstance(widget, QComboBox):
                return widget.currentText()
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                return widget.value()
            return None  # Default return if widget type is not recognized

        for i in range(self.parameterLayout.count()):
            layout = self.parameterLayout.itemAt(i).layout()
            if layout:
                label = layout.itemAt(0).widget().text().rstrip(':')
                if label == 'Time series':
                    continue  # Skip 'time_series' as it's already added

                widget = layout.itemAt(1).widget()
                value = get_widget_value(widget)

                if value is not None:  # Ensure there is a value before attempting to convert and store
                    param_key = self.reverse_param_names.get(label)
                    if param_key:  # Ensure the key exists in the reverse_param_names dictionary
                        self.data.dfc_params[param_key] = convert_value(value) if isinstance(value, str) else value
                    else:
                        QMessageBox.warning(self, "Parameter Error", f"Unrecognized parameter '{label}'")
                else:
                    QMessageBox.warning(self, "Parameter Error", f"No value entered for parameter '{label}'")

        return

    def setParameters(self):
        # Converts value to string
        def convert_value_to_string(value):
            if isinstance(value, bool):
                return 'true' if value else 'false'
            elif isinstance(value, (int, float)):
                return str(value)
            else:
                return value

        # Sets the value of the widget based on its type
        def set_widget_value(widget, value):
            if isinstance(widget, QLineEdit):
                widget.setText(value)
            elif isinstance(widget, QComboBox):
                index = widget.findText(value)
                if index >= 0:
                    widget.setCurrentIndex(index)
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(value))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(value))

        # No parameters yet, return
        if not self.data.dfc_params:
            self.getParameters()
            return

        # Set the parameters in the UI based on the stored dictionary
        for i in range(self.parameterLayout.count()):
            layout = self.parameterLayout.itemAt(i).layout()
            if layout:
                label = layout.itemAt(0).widget().text().rstrip(':')
                if label == 'Time series':
                    continue  # Skip 'time_series' as it's already set

                param_key = self.reverse_param_names.get(label)
                if param_key:  # Ensure the key exists in the reverse_param_names dictionary
                    value = self.data.dfc_params.get(param_key)
                    if value is not None:  # Ensure there is a value before attempting to convert and set
                        widget = layout.itemAt(1).widget()
                        set_widget_value(widget, convert_value_to_string(value))

                    else:
                        # Value could not be retrieved from the dictionary
                        QMessageBox.warning(self, "Parameter Error", f"No value entered for parameter '{label}'")
                else:
                    QMessageBox.warning(self, "Parameter Error", f"Unrecognized parameter(s).")
        return

    def clearParameters(self, layout):
        while layout.count():
            item = layout.takeAt(0)  # Take the first item from the layout
            if item.widget():  # If the item is a widget
                widget = item.widget()
                if widget is not None and widget is not self.time_series_textbox: # do not clear time series textbox
                    widget.deleteLater()  # Schedule the widget for deletion
            elif item.layout():  # If the item is a layout
                self.clearParameters(item.layout())  # Recursively clear the layout
                item.layout().deleteLater()  # Delete the layout itself
            elif item.spacerItem():  # If the item is a spacer
                # No need to delete spacer items; they are automatically handled by Qt
                pass

    # Memory
    def onKeepInMemoryChecked(self, state):
        if state == 2 and self.data.dfc_data is not None:
            self.data_storage.add_data(self.data)

    def onClearMemory(self):
        self.data_storage = DataStorage()
        self.plotLogo(self.connectivityFigure)
        self.connectivityCanvas.draw()
        self.plotLogo(self.timeSeriesFigure)
        self.timeSeriesCanvas.draw()
        self.plotLogo(self.distributionFigure)
        self.distributionCanvas.draw()

        self.calculatingLabel.setText(f"Cleared memory")

        return

    # Plotting
    def plotConnectivity(self):
        current_data = self.data.dfc_data
        self.connectivityFigure.clear()
        
        if current_data is None and self.data.dfc_edge_ts is None:
            # No connectivity data
            QMessageBox.warning(self, "No calculated data available for plotting")
            return
        
        elif self.data.dfc_edge_ts is not None:
            # Edge time series
            gs = gridspec.GridSpec(3, 1, self.connectivityFigure, height_ratios=[2, 0.5, 1]) # GridSpec with 3 rows and 1 column

            # The first subplot occupies the 1st row
            ax1 = self.connectivityFigure.add_subplot(gs[:1, 0])
            ax1.imshow(self.data.dfc_edge_ts.T, cmap='coolwarm', aspect='auto', vmin=-1*self.data.dfc_params["vlim"], vmax=self.data.dfc_params["vlim"])
            ax1.set_title("Edge time series")
            ax1.set_xlabel("Time (TRs)")
            ax1.set_ylabel("Edges")

            # The second subplot occupies the 3rd row
            ax2 = self.connectivityFigure.add_subplot(gs[2, 0])
            self.data.dfc_edge_rms = np.sqrt(np.mean(self.data.dfc_edge_ts.T ** 2, axis=0))
            ax2.plot(self.data.dfc_edge_rms)
            ax2.set_xlim(0, len(self.data.dfc_edge_rms) - 1)
            ax2.set_title("RMS Time Series")
            ax2.set_xlabel("Time (TRs)")
            ax2.set_ylabel("Mean Edge Value")
            
            self.connectivityFigure.set_facecolor('#f3f1f5')
            self.connectivityFigure.tight_layout()
            self.connectivityCanvas.draw()
            return
        
        else:
            # Plot normal 3D connectivity matrix
            vmax = np.abs(current_data.flatten()).max()
            ax = self.connectivityFigure.add_subplot(111)

            try:
                current_slice = current_data[:, :, self.slider.value()] if len(current_data.shape) == 3 else current_data
                self.im = ax.imshow(current_slice, cmap='coolwarm', vmin=-vmax, vmax=vmax)
            except:
                current_slice = current_data[:, :, 0] if len(current_data.shape) == 3 else current_data
                self.im = ax.imshow(current_slice, cmap='coolwarm', vmin=-vmax, vmax=vmax)

            ax.set_xlabel("ROI")
            ax.set_ylabel("ROI")

            # If we have roi names and less than 100 ROIS, we can plot the names
            if self.data.roi_names is not None and len(self.data.roi_names) < 100:
                ax.set_xticks(np.arange(len(self.data.roi_names)))
                ax.set_yticks(np.arange(len(self.data.roi_names)))
                ax.set_xticklabels(self.data.roi_names, rotation=45, fontsize=120/len(self.data.roi_names) + 2)
                ax.set_yticklabels(self.data.roi_names,              fontsize=120/len(self.data.roi_names) + 2)

            # Create the colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.15)
            cbar = self.connectivityFigure.colorbar(self.im, cax=cax)
            cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1f}'))

            self.slider.setMaximum(current_data.shape[2] - 1 if len(current_data.shape) == 3 else 0)

            self.connectivityFigure.set_facecolor('#f3f1f5')
            self.connectivityFigure.tight_layout()
            self.connectivityCanvas.draw()

            return

    def plotTimeSeries(self):
        current_data = self.data.dfc_data

        # Get dimensions of the data
        if current_data is not None and current_data.ndim == 3:
            self.rowSelector.setMaximum(current_data.shape[0] - 1)
            self.colSelector.setMaximum(current_data.shape[1] - 1)

        row = self.rowSelector.value()
        col = self.colSelector.value()

        if current_data is not None and row < current_data.shape[0] and col < current_data.shape[1] and self.data.dfc_edge_ts is None and self.data.dfc_state_tc is None:
            self.timeSeriesFigure.clear()
            ax = self.timeSeriesFigure.add_subplot(111)
            time_series = current_data[row, col, :] if len(current_data.shape) == 3 else current_data[row, col]
            ax.set_xlabel("time (TRs)")
            ax.set_ylabel("dFC strength")
            ax.plot(time_series)

        elif self.data.dfc_state_tc is not None:
            self.timeSeriesFigure.clear()
            
            current_subject = self.slider.value() if self.slider.value() < self.data.dfc_state_tc.shape[0] else 0
            time_series = self.data.dfc_state_tc[current_subject,:]
            num_states = self.data.dfc_states.shape[2]

            # Setup the gridspec layout
            gs = gridspec.GridSpec(3, num_states, self.timeSeriesFigure, height_ratios=[1, 0.5, 1])

            # Plotting the state time course across all columns
            ax_time_series = self.timeSeriesFigure.add_subplot(gs[0, :])
            ax_time_series.plot(time_series)
            ax_time_series.set_ylabel("State")
            ax_time_series.set_title(f"State time course for subject {current_subject}")
            ax_time_series.set_xlabel("Time (TRs)")

            # Plot the individual states
            for col in range(self.data.dfc_states.shape[2]):
                matrix = self.data.dfc_states[:, :, col]
                vmax = np.abs(matrix).max()
                ax_state = self.timeSeriesFigure.add_subplot(gs[2, col])
                ax_state.imshow(matrix, cmap='coolwarm', aspect=1, vmin=-vmax, vmax=vmax)
                ax_state.set_title(f"State {col}")
                ax_state.set_xticks([])
                ax_state.set_yticks([])

        else:
            # Clear the plot if the data is not available
            self.timeSeriesFigure.clear()

        self.timeSeriesFigure.set_facecolor('#f3f1f5')
        self.timeSeriesFigure.tight_layout()
        self.timeSeriesCanvas.draw()

        return

    def plotDistribution(self):
        current_data = self.data.dfc_data

        if current_data is None or not hasattr(self, 'distributionFigure'):
            self.distributionFigure.clear()
            return

        # Clear the current distribution plot
        self.distributionFigure.clear()

        # Assuming you want to plot the distribution of values in the current slice
        current_slice = current_data[:, :, self.slider.value()] if len(current_data.shape) == 3 else current_data
        ax = self.distributionFigure.add_subplot(111)
        ax.hist(current_slice.flatten(), bins=50)  # number of bins
        ax.set_xlabel("dFC values")
        ax.set_ylabel("frequency")

        self.distributionFigure.set_facecolor('#f3f1f5')
        self.distributionFigure.tight_layout()
        self.distributionCanvas.draw()

        return

    def updateTimeSeriesPlot(self, center):
        if self.data.dfc_data is None:
            return

        max_index = self.data.dfc_data.shape[2] - 1 if len(self.data.dfc_data.shape) == 3 else 0
        width = 101

        # Determine if we should show the entire series or a window
        if center == 0 or center == max_index:
            start = 0
            end = max_index
        else:
            start = max(0, center - width // 2)
            end = min(max_index, center + width // 2)

        row = self.rowSelector.value()
        col = self.colSelector.value()
        time_series_slice = self.dfc_data['data'][row, col, start:end]

        self.timeSeriesFigure.clear()
        ax = self.timeSeriesFigure.add_subplot(111)
        ax.plot(range(start, end), time_series_slice)

        self.timeSeriesFigure.tight_layout()
        self.timeSeriesCanvas.draw()

        return

    def onTabChanged(self):        
        # index 0: Connectivity plot
        # index 1: Time series plot
        # index 2: Distribution plot
        self.currentTabIndex = self.tabWidget.currentIndex()

        # No dFC data: reset everything and return
        if self.data.dfc_data is None:
            self.plotLogo(self.connectivityFigure)
            self.connectivityCanvas.draw()
            self.distributionFigure.clear()
            self.distributionCanvas.draw()
            self.timeSeriesFigure.clear()
            self.timeSeriesCanvas.draw()
           
            self.slider.hide()
            self.roiSelectorWidget.hide()
            self.navButtonContainer.hide()
            
            return
        
        # Static or edge FC: hide slider/roi widgets and return
        elif np.ndim(self.data.dfc_data) == 2:
            self.slider.hide()
            self.navButtonContainer.hide()
            self.roiSelectorWidget.hide()

            self.timeSeriesFigure.clear()
            self.timeSeriesCanvas.draw()
            
            return
        
        # Normal 3D dFC data: get initial values
        else:
            # Get number of subjects and states (if applicable)
            if self.data.dfc_state_tc is not None:
                maxTab1 = self.data.dfc_state_tc.shape[0]
                maxTab02 = self.data.dfc_states.shape[2]
            else:
                maxTab1 = 1
                maxTab02 = self.data.dfc_data.shape[2]

        # Individual tab settings  
        # Connectivity and distribution tab (index 0 and 2)
        if self.currentTabIndex == 0 or self.currentTabIndex == 2:
            self.slider.setRange(0, max(0, maxTab02 - 1))
            self.slider.setValue(self.backupSliderValueTab02)
            self.slider.show()
            
            self.navButtonContainer.show()
            self.roiSelectorWidget.hide()

            total_length = self.data.dfc_data.shape[2] if np.ndim(self.data.dfc_data) == 3 else 0
            if self.data.dfc_states is None:
                text_label = f"t = {self.slider.value()} / {total_length - 1}"
            else:
                text_label = f"State {self.slider.value()}"

        # Time series tab (index 1)
        else:
            self.slider.setRange(0, max(0, maxTab1 - 1))
            self.slider.setValue(self.backupSliderValueTab1)
            # State based FC
            if self.data.dfc_state_tc is not None:
                # Only one subject
                if self.data.dfc_state_tc.shape[0] == 1:
                    text_label = "Single subject"
                # Multiple subjects
                else:
                    text_label = f"Subject {self.slider.value()}"

                self.slider.show()
                self.navButtonContainer.show()
                self.roiSelectorWidget.hide()
 
            # Continuous FC (3D data)
            else:
                self.slider.setMaximum(self.data.dfc_data.shape[2] - 1)
                text_label = f"t = {self.slider.value()} / {self.data.dfc_data.shape[2] - 1}"

                self.slider.hide()
                self.navButtonContainer.hide()
                self.roiSelectorWidget.show()


        self.positionLabel.setText(text_label)
        self.update()

        return

    def onSliderValueChanged(self, value):          
        # Ensure there is data to work with
        if self.data.dfc_data is None or self.im is None:
            return

        position_text = ""
        # Tab 0 or 2: Connectivity or distribution plot
        if self.currentTabIndex == 0 or self.currentTabIndex == 2:
            
            if np.ndim(self.data.dfc_data) == 3:
                self.im.set_data(self.data.dfc_data[:, :, value])
                vlim = np.max(np.abs(self.data.dfc_data[:, :, value]))
                self.im.set_clim(-vlim, vlim)

                total_length = self.data.dfc_data.shape[2]

                if self.data.dfc_states is None:
                    position_text = f"t = {value} / {total_length-1}"
                else:
                    position_text = f"State {value}"
                
            else:
                self.im.set_data(self.data.dfc_data)
                vlim = np.max(np.abs(self.data.dfc_data))
                self.im.set_clim(-vlim, vlim)

                total_length = 0

                if self.data.dfc_states is not None:
                    position_text = f"State {value}"

            # Redraw the canvas
            self.connectivityCanvas.draw()
            self.plotDistribution()

            # Store backup value
            self.backupSliderValueTab02 = self.slider.value()

        # Tab 1: Time series plot
        else:
            position_text = f"Subject {value}"
            self.plotTimeSeries()

            # Store backup value
            self.backupSliderValueTab1 = self.slider.value()

        self.positionLabel.setText(position_text)

        return

    def onSliderButtonClicked(self):
        # Clicking a button moves the slider by x steps
        button = self.sender()
        delta = 0

        if button == self.backButton:
            delta = -1

        if button == self.forwardButton:
            delta = 1

        if button == self.backLargeButton:
            delta = -10

        if button == self.forwardLargeButton:
            delta = 10

        old = self.slider.value()

        self.slider.setValue(max(0, min(self.slider.value() + delta, self.slider.maximum())))
        self.slider.update()
  
        return

    """
    Graph tab
    """
    def loadGraphFile(self):
        fileFilter = "All supported files (*.mat *.npy *.tsv *.dtseries.nii *.ptseries.nii);;\
                      MAT files (*.mat);;Text files (*.txt);;NumPy files (*.npy);;TSV files (*.tsv);;CIFTI files (*.dtseries.nii *.ptseries.nii)"
        file_path, _ = QFileDialog.getOpenFileName(self, "Load File", "", fileFilter)
        file_name = file_path.split('/')[-1]
        self.data.graph_file = file_name

        if not file_path:
            return  # Early exit if no file is selected

        if file_path.endswith('.mat'):
            try:
                data_dict = loadmat(file_path)
            except:
                data_dict = mat73.loadmat(file_path)

            try:
                self.data.graph_data = data_dict["dfc_data"] # Try to load dfc_data (saving connectivity estimates with Comet will create this field)
            except:
                self.data.graph_data = data_dict[list(data_dict.keys())[-1]] # Else get the last item in the file (which is the data if there is only one field)

        elif file_path.endswith('.txt'):
            self.data.graph_data = np.loadtxt(file_path)

        elif file_path.endswith('.npy'):
            self.data.graph_data = np.load(file_path)

        else:
            self.data.graph_data = None
            QMessageBox.warning(self, "Data Error", "Unsupported file format.")
            return

        if self.data.graph_data.shape[0] != self.data.graph_data.shape[1]:
            QMessageBox.warning(self, "Data Error", "Input data needs to be of shape PxP (square matrix) or PxPxN (multiple square matrices).")
            self.data.graph_data = None
            
            self.plotLogo(self.matrixFigure)
            self.matrixCanvas.draw()

            self.plotLogo(self.graphFigure)
            self.graphCanvas.draw()

            self.graphFileNameLabel.setText(f"No data available.")
            return

        self.data.graph_raw = self.data.graph_data.copy()
        self.graphFileNameLabel.setText(f"Loaded {self.data.graph_file} with shape {self.data.graph_data.shape}")

        total_length = self.data.graph_data.shape[2] if np.ndim(self.data.graph_data) == 3 else 0
        position_text = f"t = {self.graphSlider.value()} / {total_length-1}" if len(self.data.graph_data.shape) == 3 else " static "
        self.graphPositionLabel.setText(position_text)
        self.currentGraphOption = None
        
        self.plotGraphMatrix()
        self.onGraphCombobox()

        self.addOptionButton.setEnabled(True)
        self.graphUsePushButton.setEnabled(True)

        return

    def saveGraphFile(self):
        if self.data.graph_data is None:
            QMessageBox.warning(self, "Output Error", "No graph data available to save.")
            return

        # Open a file dialog to specify where to save the file
        filePath, _ = QFileDialog.getSaveFileName(self, "Save File", "", "MAT Files (*.mat)")

        if filePath:
            # Ensure the file has the correct extension
            if not filePath.endswith('.mat'):
                filePath += '.mat'

            # Save the the current data object to a .mat file
            try:
                data_dict = {}
                for field in [f for f in self.data.__dataclass_fields__ if f.startswith('graph_')]:
                    value = getattr(self.data, field)

                    if isinstance(value, np.ndarray):
                        data_dict[field] = value
                    elif isinstance(value, dict):
                        # Ensure all dict values are appropriately converted
                        converted_dict = {}
                        for k, v in value.items():
                            if isinstance(v, np.ndarray):
                                converted_dict[k] = v
                            elif v is None:
                                converted_dict[k] = np.array([])
                            else:
                                converted_dict[k] = v
                        data_dict[field] = converted_dict
                    elif value is None:
                        data_dict[field] = np.array([])
                    elif field == 'dfc_instance':
                        pass
                    else:
                        data_dict[field] = value

                savemat(filePath, data_dict)

            except Exception as e:
                QMessageBox.warning(self, "Output Error", f"Error saving data: {e}")

            return

    def onGraphCombobox(self):
        self.setGraphParameters()
        return

    def updateGraphComboBox(self):
        def shouldIncludeFunc(funcName):
            if self.preprocessingCheckBox.isChecked() and funcName.startswith("PREP"):
                return True
            if self.graphCheckBox.isChecked() and funcName.startswith("COMET"):
                return True
            if self.BCTCheckBox.isChecked() and funcName.startswith("BCT"):
                return True
            return False

        # Disconnect existing connections to avoid multiple calls
        try:
            self.graphAnalysisComboBox.currentTextChanged.disconnect(self.onGraphCombobox)
        except TypeError:
            pass

        # Clear the combobox
        self.graphAnalysisComboBox.clear()

        # Filter options based on the checkboxes
        filtered_options = {name: desc for name, desc in self.graphOptions.items() if shouldIncludeFunc(desc)}

        for analysis_function, pretty_name in filtered_options.items():
            self.graphAnalysisComboBox.addItem(pretty_name, analysis_function)

        # Reconnect the signal
        self.graphAnalysisComboBox.currentTextChanged.connect(self.onGraphCombobox)

        # Trigger the onGraphCombobox for the initial setup if there are any options
        if filtered_options:
            self.onGraphCombobox()

        return

    def handleGraphCheckboxes(self):
        # Make sure at most one checkbox is checked
        sender = self.sender()
        if sender.isChecked():
            for cb in (self.graphUseAllCheckbox, self.graphUseSingleCheckbox, self.graphUseCustomCheckbox):
                if cb is not sender:
                    cb.setChecked(False)

        if self.graphUseCustomCheckbox.isChecked():
            self.graphUseSpecificTextbox.setEnabled(True)
        else:
            self.graphUseSpecificTextbox.setEnabled(False)

    def onGraphAmountSelected(self):
        # Handle the graph slice selection
        if not isinstance(self.data.graph_raw, np.ndarray):
            return

        if self.graphUseAllCheckbox.isChecked():
            selectedIndices = slice(None)
 
        elif self.graphUseSingleCheckbox.isChecked():
            selectedIndices = self.graphSlider.value()

        else:
            inds = []
            for t in self.graphUseSpecificTextbox.text().split(','):
                t = t.strip()
                if t:
                    if '-' in t or ':' in t:
                        sep = '-' if '-' in t else ':'
                        parts = t.split(sep)
                        if len(parts) == 3:
                            a, b, c = map(int, parts)
                            # For three parts, include b by setting stop to b + (1 if c>0 else -1)
                            inds += list(np.arange(a, b + (1 if c > 0 else -1), c))
                        elif len(parts) == 2:
                            a, b = map(int, parts)
                            # Default step: 1 if a <= b else -1, and adjust stop accordingly
                            step = 1 if a <= b else -1
                            inds += list(np.arange(a, b + step, step))
                    else:
                        inds.append(int(t))
            selectedIndices = inds if inds else slice(None)
            
        # Subset the data
        self.data.graph_data = self.data.graph_raw[:, :, selectedIndices]
        
        # Set labels
        self.graphFileNameLabel.setText(f"Using dFC estimates with shape {self.data.graph_data.shape}")
        self.data.graph_file = f"dfC from {self.data.file_name}"
        
        total_length = self.data.graph_data.shape[2] if np.ndim(self.data.graph_data) == 3 else 0
        position_text = f"t = {self.graphSlider.value()} / {total_length-1}" if len(self.data.graph_data.shape) == 3 else " "
        self.graphPositionLabel.setText(position_text)
        
        # Perform all plotting and GUI operations
        self.plotGraphMatrix()
        self.onGraphCombobox()

        self.optionsTextbox.clear()
        self.measureTextbox.clear()
        self.graphTextbox.clear()
        self.graphProcessingStepCounter = 1
        self.graphMeasureStepCounter = 1
        self.clearAllButton.setEnabled(False)
        self.graphSaveButton.setEnabled(False)

        self.graphFigure.clear()
        self.graphCanvas.draw()

        self.addOptionButton.setEnabled(True)
        self.currentGraphOption = None

        if np.ndim(self.data.graph_data) == 3:
            self.graphSlider.setMaximum(self.data.graph_data.shape[2] - 1)
            self.graphSlider.setValue(min(self.graphSlider.value(), self.data.graph_data.shape[2] - 1))
        
        return

    # Calculations
    def calculateGraph(self):
        self.graphUsePushButton.setEnabled(False)
        self.addOptionButton.setEnabled(False)
        self.clearAllButton.setEnabled(False)
        self.graphSaveButton.setEnabled(False)
        
        # Start worker thread for graph calculations
        self.graphThread = QThread()
        self.graphWorker = Worker(self.calculateGraphThread, {})
        self.graphWorker.moveToThread(self.graphThread)

        self.graphWorker.finished.connect(self.graphThread.quit)
        self.graphWorker.finished.connect(self.graphWorker.deleteLater)
        self.graphThread.finished.connect(self.graphThread.deleteLater)
        self.graphWorker.result.connect(self.handleGraphResult)
        self.graphWorker.error.connect(self.handleGraphError)

        self.graphThread.started.connect(self.graphWorker.run)
        self.graphThread.start()

    def calculateGraphThread(self, **unused):
        option, params = self.getGraphOptions()

        # Get the function
        func_name = self.reverse_graphOptions[option]
        func = getattr(graph, func_name)

        option_name = re.sub(r'^\S+\s+', '', option) # regex to remove the PREP/GRAPH part

        if option.split()[0].lower() == 'prep':
            self.optionsTextbox.append(f"Calculating {option_name.lower()}, please wait...")
        else:
            self.measureTextbox.append(f"Calculating {option_name.lower()}, please wait...")

        first_param = next(iter(params))
        graph_params = {first_param: self.data.graph_data}
        graph_params.update({k: v for k, v in params.items() if k != first_param})

        # Calculate graph measure
        if np.ndim(self.data.graph_data) == 3:
            graph_data = []
            for estimate in range(self.data.graph_data.shape[2]):
                graph_params[first_param] = self.data.graph_data[:, :, estimate]
                graph_data.append(func(**graph_params))
        else:
            graph_data = func(**graph_params)

        return f'graph_{option.split()[0].lower()}', graph_data, option_name, graph_params

    def handleGraphResult(self, result):
        output = result[0]
        data   = result[1]
        option = result[2]
        params = result[3]

        # Update self.data.graph_data based on the result
        if output == 'graph_prep':
            self.data.graph_data = np.asarray(data).T
            self.plotGraphMatrix()

            # Output step and options to textbox, remove unused parameters
            if option == 'Threshold':
                if params.get('type') == 'absolute':
                    filtered_params = {k: v for k, v in params.items() if k != 'density'}
                elif params.get('type') == 'density':
                    filtered_params = {k: v for k, v in params.items() if k != 'threshold'}
            else:
                filtered_params = params

            filtered_params = {k: v for k, v in list(filtered_params.items())[1:]}

            for k, v in filtered_params.items():
                if isinstance(v, np.ndarray) or isinstance(v, float):
                    filtered_params[k] = f"{v:.2f}"

            # Update the textbox with the current step and options
            current_text = self.optionsTextbox.toPlainText()
            lines = current_text.split('\n')

            if len(filtered_params) > 0:
                output_text = f"{self.graphProcessingStepCounter}. {option} {filtered_params}"
            else:
                output_text = f"{self.graphProcessingStepCounter}. {option}"
            
            if len(lines) > 1:
                lines[-1] = output_text
            else:
                lines = [output_text]

            updated_text = '\n'.join(lines)
            self.optionsTextbox.setPlainText(updated_text)
            
            self.graphProcessingStepCounter += 1
            self.currentGraphOption = option

        # Graph measure
        else:
            self.data.graph_results[option] = data

            # Populate dropdown
            if self.graphResultMeasureDropdown.findText(option) == -1:
                self.graphResultMeasureDropdown.addItem(option)
                
            index = self.graphResultMeasureDropdown.findText(option)
            self.graphResultMeasureDropdown.setCurrentIndex(index)

            # Textbox stuff
            current_text = self.measureTextbox.toPlainText()
            lines = current_text.split('\n')
            output_text = f"{self.graphMeasureStepCounter}. {option}"   
            
            if len(lines) > 1:
                lines[-1] = output_text
            else:
                lines = [output_text]

            updated_text = '\n'.join(lines)
            self.measureTextbox.setPlainText(updated_text)
            self.currentGraphOption = option
            self.graphMeasureStepCounter += 1

            # Plot the results
            self.plotGraphMeasure(option)

        self.graphUsePushButton.setEnabled(True)
        self.addOptionButton.setEnabled(True)
        self.clearAllButton.setEnabled(True)
        self.graphSaveButton.setEnabled(True)

    def handleGraphError(self, error):
        # Handle errors in the worker thread
        self.optionsTextbox.clear()
        self.measureTextbox.clear()
        QMessageBox.warning(self, "Calculation Error", f"Error calculating graph measure: {error}.")

        self.graphUsePushButton.setEnabled(True)
        self.addOptionButton.setEnabled(True)
        self.clearAllButton.setEnabled(True)
        self.graphSaveButton.setEnabled(True)
        return

    # Parameters
    def setGraphParameters(self):
        # Clear parameters
        self.clearParameters(self.graphParameterLayout)

        # Retrieve the selected function from the graph module
        if self.graphAnalysisComboBox.currentData() == None:
            return

        func = getattr(graph, self.graphAnalysisComboBox.currentData())

        # Retrieve the signature of the function
        func_signature = inspect.signature(func)
        type_hints = get_type_hints(func)

        # Calculate the maximum label width
        max_label_width = 0
        font_metrics = QFontMetrics(self.font())
        for name, param in func_signature.parameters.items():
            if name not in ['self', 'args', 'kwargs']:  # Skip unwanted parameters
                label_width = font_metrics.boundingRect(f"{name}:").width()
                max_label_width = max(max_label_width, label_width)

        is_first_parameter = True  # Flag to identify the first parameter

        # Iterate over parameters in the function signature
        temp_widgets = {}
        for name, param in func_signature.parameters.items():

            if name not in ['self', 'copy', 'args', 'kwargs']:  # Skip unwanted parameters
                # Horizontal layout for each parameter
                param_layout = QHBoxLayout()
                param_type = type_hints.get(name)
                param_default = 1 if isinstance(param.default, inspect._empty) else param.default

                if param_default == None:
                    if param_type == bool or param_type == Optional[bool]:
                        param_default = False
                    elif param_type == int or param_type == Optional[int]:
                        param_default = 1
                    elif param_type == float or param_type == Optional[float]:
                        param_default = 1.0
                    else:
                        param_default = "empty"

                # Create a label for the parameter and set its fixed width
                param_label = QLabel(f"{name}:")
                param_label.setFixedWidth(max_label_width + 20)  # Add some padding
                param_layout.addWidget(param_label)

                # For the first parameter, set its value based on the data source and lock it
                if is_first_parameter:
                    param_widget = QLineEdit()
                    param_widget.setPlaceholderText("No data available")
                    if self.data.graph_file or self.data.graph_data is not None:
                        param_widget = QLineEdit("As shown in plot")
                    param_widget.setReadOnly(True)  # Make the widget read-only
                    is_first_parameter = False  # Update the flag so this block runs only for the first parameter
                else:
                    # Bool
                    if param_type == bool or param_type == Optional[bool]:
                        param_widget = QComboBox()
                        param_widget.addItems(["False", "True"])
                        param_widget.setCurrentIndex(int(param_default))
                    # Int
                    elif param_type == int or param_type == Optional[int]:
                        param_widget = QSpinBox()
                        param_widget.setValue(param_default)
                        param_widget.setMaximum(10000)
                        param_widget.setMinimum(-10000)
                        param_widget.setSingleStep(1)
                    # Float
                    elif param_type == float or param_type == Optional[float]:
                        param_widget = QDoubleSpinBox()
                        if name == "threshold":
                            param_widget.setValue(0.0)
                        else:
                            param_widget.setValue(param_default)
                        param_widget.setMaximum(1.0)
                        param_widget.setMinimum(0.0)
                        param_widget.setSingleStep(0.1)
                        param_widget.setDecimals(2)
                    # String
                    elif get_origin(type_hints.get(name)) is Literal:
                        options = type_hints.get(name).__args__
                        param_widget = QComboBox()
                        param_widget.addItems([str(option) for option in options])
                    # Fallback
                    else:
                        param_widget = QLineEdit(str(param.default) if param.default != inspect.Parameter.empty else "")

                temp_widgets[name] = (param_label, param_widget)
                param_layout.addWidget(param_widget)
                self.graphParameterLayout.addLayout(param_layout)

        # Adjust visibility based on 'type' parameter
        type_widget = None
        if 'type' in temp_widgets:
            _, type_widget = temp_widgets['type']

        if type_widget:
            # Function to update parameter visibility
            def updateVisibility():
                selected_type = type_widget.currentText()
                if selected_type == 'absolute':
                    if 'threshold' in temp_widgets:
                        temp_widgets['threshold'][0].show()
                        temp_widgets['threshold'][1].show()
                    if 'density' in temp_widgets:
                        temp_widgets['density'][0].hide()
                        temp_widgets['density'][1].hide()
                elif selected_type == 'density':
                    if 'threshold' in temp_widgets:
                        temp_widgets['threshold'][0].hide()
                        temp_widgets['threshold'][1].hide()
                    if 'density' in temp_widgets:
                        temp_widgets['density'][0].show()
                        temp_widgets['density'][1].show()

            # Connect the signal from the type_widget to the updateVisibility function
            type_widget.currentIndexChanged.connect(updateVisibility)
            updateVisibility()

        self.graphParameterLayout.addStretch()

    def getGraphOptions(self):
        # Initialize a dictionary to hold parameter names and their values
        params_dict = {}

        # Iterate over all layout items in the graphParameterLayout
        for i in range(self.graphParameterLayout.count()):
            layout_item = self.graphParameterLayout.itemAt(i)

            # Check if the layout item is a QHBoxLayout (as each parameter is in its own QHBoxLayout)
            if isinstance(layout_item, QHBoxLayout):
                param_layout = layout_item.layout()

                # The parameter name is in the QLabel, and the value is in the second widget (QLineEdit, QComboBox, etc.)
                if param_layout.count() >= 2:
                    # Extract the parameter name from the QLabel
                    param_name_label = param_layout.itemAt(0).widget()
                    if isinstance(param_name_label, QLabel):
                        param_name = param_name_label.text().rstrip(':')  # Remove the colon at the end

                        # Extract the parameter value from the appropriate widget type
                        param_widget = param_layout.itemAt(1).widget()
                        if isinstance(param_widget, QLineEdit):
                            param_value = param_widget.text()
                        elif isinstance(param_widget, QComboBox):
                            param_value = param_widget.currentText()
                        elif isinstance(param_widget, QSpinBox) or isinstance(param_widget, QDoubleSpinBox):
                            param_value = param_widget.value()

                        # Convert to appropriate boolean type (LineEdit ad ComboBox return strings))
                        if param_value == "True":
                            param_value = True
                        elif param_value == "False":
                            param_value = False

                        # Add the parameter name and value to the dictionary
                        params_dict[param_name] = param_value

        # Retrieve the currently selected option in the graphAnalysisComboBox
        current_option = self.graphAnalysisComboBox.currentText()

        # Return the current option and its parameters
        return current_option, params_dict

    def onClearAllGraphOptions(self):
        self.data.graph_data = self.data.graph_raw
        self.data.graph_results = {}

        self.plotGraphMatrix()
        self.optionsTextbox.clear()
        self.measureTextbox.clear()
        self.graphTextbox.clear()

        self.graphResultMeasureDropdown.clear()
        self.graphResultMeasureDropdown.setCurrentIndex(-1)
        self.graphResultMeasureDropdown.setPlaceholderText("No graph measure calculated yet")

        self.graphAnalysisComboBox.setCurrentIndex(0)


        self.plotLogo(self.graphFigure)
        self.graphCanvas.draw()

        self.graphProcessingStepCounter = 1
        self.graphMeasureStepCounter = 1
        self.clearAllButton.setEnabled(False)
        self.graphSaveButton.setEnabled(False)

    # Plotting
    def plotGraphMatrix(self):
        current_data = self.data.graph_data

        if current_data is None:
            self.plotLogo(self.matrixFigure)
            self.matrixCanvas.draw()
            self.graphSlider.hide()
            self.graphNavButtonContainer.hide()

            QMessageBox.warning(self, "No calculated data available for plotting")
            return

        # Enable/disable slider
        if np.ndim(current_data) == 3:
            current_data = current_data[:, :, min(self.data.graph_data.shape[2] - 1, self.graphSlider.value())]
            self.graphSlider.setRange(0, self.data.graph_data.shape[2] - 1)
            self.graphSlider.setValue(self.graphSlider.value())
            self.graphSlider.show()
            self.graphNavButtonContainer.show()
        else:
            self.graphSlider.setValue(0)
            self.graphSlider.hide()
            self.graphNavButtonContainer.hide()
        
        self.matrixFigure.clear()
        ax = self.matrixFigure.add_subplot(111)

        vmax = np.max(np.abs(current_data))
        self.graph_im = ax.imshow(current_data, cmap='coolwarm', vmin=-vmax, vmax=vmax)
        ax.set_xlabel("ROI")
        ax.set_ylabel("ROI")

        # Create the colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.15)
        cbar = self.matrixFigure.colorbar(self.graph_im, cax=cax)
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1f}'))

        self.matrixFigure.set_facecolor('#f3f1f5')
        self.matrixFigure.tight_layout()
        self.matrixCanvas.draw()
        
    def plotGraphMeasure(self, measure):
        self.graphFigure.clear()
        ax = self.graphFigure.add_subplot(111)
        plot_data = self.data.graph_results[measure]

        # Check type of the graph output data
        if isinstance(plot_data, (np.ndarray, np.float64)):
            if np.ndim(plot_data) == 0:
                # If graph_out is a single value (0D array)
                self.graphTextbox.setText(f"{measure}: {plot_data.item()}")
            elif np.ndim(plot_data) == 1:
                # For a 1D array, plot a vertical lollipop plot
                ax.stem(plot_data, linefmt="#19232d", markerfmt='o', basefmt=" ")
                ax.set_xlabel("ROI")
                ax.set_ylabel(measure)

                # Calculate mean and std, and update the textbox
                mean_val = np.mean(plot_data)
                std_val = np.std(plot_data)
                self.graphTextbox.setText(f"{measure} (mean: {mean_val:.2f}, std: {std_val:.2f})")

            elif np.ndim(plot_data) == 2:
                # For a 2D array, use imshow
                vmax = np.max(np.abs(plot_data))
                im = ax.imshow(plot_data, cmap='coolwarm', vmin=-vmax, vmax=vmax)

                # Create the colorbar
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.15)
                self.graphFigure.colorbar(im, cax=cax).ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1f}'))
            else:
                self.graphTextbox.setText("3D graph data not currently supported for plotting.")

        elif isinstance(plot_data, tuple):
            # Setup data for output
            output_string = f""
            output_arrays = []

            data = plot_data[0]
            label = self.graphAnalysisComboBox.currentText()

            if isinstance(data, (int, float)):
                output_string += f"{label}: {data:.2f}, "
                self.plotLogo(self.graphFigure)

            elif isinstance(data, np.ndarray):
                output_arrays.append((label, data))

            elif isinstance(data, tuple):
                for i, dat in enumerate(data):
                    if isinstance(dat, (int, float)):
                        output_string += f"{label[i]}: {dat:.2f}, "
                        self.plotLogo(self.graphFigure)

                    elif isinstance(dat, np.ndarray):
                        output_arrays.append((label[i], dat))

            else:
                self.graphTextbox.setText("Graph output data is not in expected format.")
                return

            # Print the output string
            self.graphTextbox.setText(output_string.strip(', '))  # Remove the trailing comma

            # Plot the output arrays
            if output_arrays:
                self.graphFigure.clear()
                n_subplots = len(output_arrays)

                for i, (key, value) in enumerate(output_arrays):
                    ax = self.graphFigure.add_subplot(1, n_subplots, i + 1)
                    vmax = np.max(np.abs(value))
                    if value.ndim == 1:
                        # For a 1D vector, plot a vertical lollipop plot
                        ax.stem(value, linefmt="#19232d", markerfmt='o', basefmt=" ")
                        ax.set_xlabel("ROI")
                        ax.set_ylabel(measure)

                        # Calculate mean and std, and update the textbox
                        mean_val = np.mean(value)
                        std_val = np.std(value)
                        self.graphTextbox.setText(f"{measure} (mean: {mean_val:.2f}, variance: {std_val:.2f})")

                    elif value.ndim == 2:
                        # For a 2D array, use imshow
                        im = ax.imshow(value, cmap='coolwarm', vmin=-vmax, vmax=vmax)

                        # Create the colorbar
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="5%", pad=0.15)
                        cbar = self.graphFigure.colorbar(im, cax=cax)
                        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1f}'))

                    else:
                        self.graphTextbox.setText("Graph output data is not in expected format.")

                    ax.set_title(label)

        elif isinstance(plot_data, list):
            if np.ndim(plot_data[0]) == 0:
                # If graph_out is a single value (0D array)
                im = ax.plot(plot_data)
                ax.set_xlabel("t")
                ax.set_ylabel(measure)
                self.graphTextbox.setText(f"{measure}: {plot_data[self.graphSlider.value()]}")
            elif np.ndim(plot_data[0]) == 1:
                # For a 1D array, plot a vertical lollipop plot
                ax.stem(plot_data[self.graphSlider.value()], linefmt="#19232d", markerfmt='o', basefmt=" ")
                ax.set_xlabel("ROI")
                ax.set_ylabel(measure)

                # Calculate mean and std, and update the textbox
                mean_val = np.mean(plot_data[self.graphSlider.value()])
                std_val = np.std(plot_data[self.graphSlider.value()])
                self.graphTextbox.setText(f"{measure} (mean: {mean_val:.2f}, std: {std_val:.2f})")

            elif np.ndim(plot_data[0]) == 2:
                # For a 2D array, use imshow
                vmax = np.max(np.abs(plot_data[self.graphSlider.value()]))
                im = ax.imshow(plot_data[self.graphSlider.value()], cmap='coolwarm', vmin=-vmax, vmax=vmax)

                # Create the colorbar
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.15)
                self.graphFigure.colorbar(im, cax=cax).ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1f}'))
            else:
                self.graphTextbox.setText("3D graph data not currently supported for plotting.")  
        
        elif plot_data == None:
            pass
        
        else:
            self.graphTextbox.append("Graph output data is not in expected format.")

        # Draw the plot
        self.graphFigure.set_facecolor('#f3f1f5')
        self.graphFigure.tight_layout()
        self.graphCanvas.draw()

        return

    def onGraphResultMeasureSelected(self):
        # Update the graph measure plot based on the selected measure in the dropdown
        self.currentGraphOption = self.graphResultMeasureDropdown.currentText()
        
        if self.currentGraphOption in self.data.graph_results:
            self.plotGraphMeasure(self.currentGraphOption)
        
        return

    def onGraphTabChanged(self):
        self.currentGraphTabIndex = self.graphTabWidget.currentIndex()
        # index 0: Graph matrix plot
        # index 1: Graph measure plot

        if self.data.graph_data is None:
            self.plotLogo(self.graphFigure)
            self.matrixCanvas.draw()
            self.graphSlider.hide()
            self.graphNavButtonContainer.hide()
            self.graphPositionLabel.setText("")
            return

        if self.data.graph_data is not None:
            if np.ndim(self.data.graph_data) == 3:
                self.graphSlider.show()
                self.graphNavButtonContainer.show()

            else:
                self.graphSlider.hide()
                self.graphNavButtonContainer.hide()

        return

    def onGraphSliderValueChanged(self):
        # Ensure there is data to work with
        if self.data.graph_data is None:
            return

        if self.currentGraphTabIndex == 0:
            # Matrix plot
            self.graph_im.set_data(self.data.graph_data[:, :, self.graphSlider.value()]) if np.ndim(self.data.graph_data) == 3 else self.graph_im.set_data(self.data.graph_data)
            vlim = np.max(np.abs(self.data.graph_data[:, :, self.graphSlider.value()])) if np.ndim(self.data.graph_data) == 3 else np.max(np.abs(self.data.graph_data))
            self.graph_im.set_clim(-vlim, vlim)
            self.matrixCanvas.draw()

        # Measure plot
        if self.currentGraphOption:
            self.plotGraphMeasure(self.currentGraphOption)

        total_length = self.data.graph_data.shape[2] if len(self.data.graph_data.shape) == 3 else 0

        if self.data.dfc_states is None:
            position_text = f"t = {self.graphSlider.value()} / {total_length-1}" if np.ndim(self.data.graph_data) == 3 else " static "
        else:
            position_text = f"State {self.graphSlider.value()}"
        
        self.graphPositionLabel.setText(position_text)
        
        return

    def onGraphSliderButtonClicked(self):
        button = self.sender()
        delta = 0

        if button == self.graphBackButton:
            delta = -1

        if button == self.graphForwardButton:
            delta = 1

        if button == self.graphBackLargeButton:
            delta = -10

        if button == self.graphForwardLargeButton:
            delta = 10

        self.graphSlider.setValue(max(0, min(self.graphSlider.value() + delta, self.graphSlider.maximum())))
        self.graphSlider.update()

        return


    """
    Multiverse tab
    """
    def addDecisionContainer(self):
        """
        Add a combined container widget for all things related to creating decisions and options
        """
        decisionWidget = QWidget()  # Widget that holds the entire container
        mainLayout = QVBoxLayout(decisionWidget)  # Vertical layout: contains decision layout and function layout
        mainLayout.setContentsMargins(0, 0, 0, 0) # Remove space to remove excessive space between widgets

        functionLayout = QHBoxLayout()  # Controls layout for functions and properties

        # Create the dropdown menu
        categoryComboBox = QComboBox()
        categoryComboBox.addItems(["General", "FC", "Graph", "Other"])
        categoryComboBox.setObjectName("categoryComboBox")

        # Decision name input field
        decisionNameInput = QLineEdit()
        decisionNameInput.setPlaceholderText("Decision name")
        decisionNameInput.setObjectName("decisionNameInput")

        # Decision options input field
        decisionOptionsInput = QLineEdit()
        decisionOptionsInput.setPlaceholderText("Options (comma-separated)")
        decisionOptionsInput.setObjectName("decisionOptionsInput")

        # Collapse button to hide/show the function and parameter widgets
        collapseButton = QPushButton(" \u25B2 ")
        collapseButton.setObjectName("collapseButton")
        collapseButton.hide()

        # Add option button to add a new option to the decision
        addOptionButton = QPushButton(' \u25B6 ')
        addOptionButton.setObjectName("addOptionButton")
        addOptionButton.hide()

        # Include button to confirm the decision
        includeButton = QPushButton(' \u2714 ')

        # Remove button to delete the decision
        removeButton = QPushButton(' \u2718 ')

        # Add widgets to the layout with appropriate stretch factors
        functionLayout.addWidget(categoryComboBox, 6)
        functionLayout.addWidget(decisionNameInput, 7)
        functionLayout.addWidget(decisionOptionsInput, 12)
        functionLayout.addWidget(collapseButton, 1)
        functionLayout.addWidget(addOptionButton, 1)
        functionLayout.addWidget(includeButton, 1)
        functionLayout.addWidget(removeButton, 1)

        # Combo box for selecting specific functions or methods (will be inside the parameterContainer widget)
        functionComboBox = QComboBox()
        functionComboBox.currentIndexChanged.connect(lambda _: self.updateFunctionParameters(functionComboBox, parameterContainer))
        functionComboBox.setObjectName("functionComboBox")

        # Parameter container widget
        parameterContainer = QWidget()
        parameterContainer.setObjectName("parameterContainer")
        parameterLayout = QVBoxLayout()
        parameterContainer.setLayout(parameterLayout)
        parameterContainer.hide()

        # Connect category combo box change
        categoryComboBox.currentIndexChanged.connect(lambda _: self.onCategoryComboBoxChanged(categoryComboBox, functionComboBox, parameterContainer, addOptionButton, collapseButton, decisionNameInput, decisionOptionsInput))

        # Connect the signals for the buttons, done here so all widgets are available
        includeButton.clicked.connect(lambda: self.includeDecision(categoryComboBox, decisionNameInput, decisionOptionsInput))
        removeButton.clicked.connect(lambda: self.removeDecision(decisionNameInput, decisionWidget, decisionOptionsInput))
        collapseButton.clicked.connect(lambda: self.collapseOption(collapseButton, parameterContainer))
        addOptionButton.clicked.connect(lambda: self.addOption(functionComboBox, parameterContainer, decisionNameInput, decisionOptionsInput))

        # Adding the controls layout to the main layout
        mainLayout.addLayout(functionLayout)
        mainLayout.addWidget(parameterContainer)

        # Add the widget to a list so we can keep track of individual items
        self.mv_containers.append(decisionWidget)

        return decisionWidget

    def populateMultiverseContainers(self, forking_paths):
        """
        Populate multiverse containers with decisions and options from the forking paths dictionary.
        """
        # Clear any previous containers
        for container in self.mv_containers:
            container.deleteLater()
        self.mv_containers.clear()

        # Loop through the forking_paths dictionary and create a container for each entry
        for decision_name, options in forking_paths.items():
            # Create a new decision container
            decisionWidget = self.addDecisionContainer()

            # Get the widgets from the container to update them
            categoryComboBox = decisionWidget.findChild(QComboBox, "categoryComboBox")
            decisionNameInput = decisionWidget.findChild(QLineEdit, "decisionNameInput")
            decisionOptionsInput = decisionWidget.findChild(QLineEdit, "decisionOptionsInput")

            # Check if the options list contains dictionaries
            if isinstance(options, list) and len(options) > 0 and isinstance(options[0], dict):
                first_dict = options[0]

                if 'func' in first_dict:
                    # Check if the func key corresponds to comet.connectivity
                    func_value = first_dict['func']
                    if func_value.startswith("comet.connectivity"):
                        # Set the category to FC
                        categoryComboBox.setCurrentText("FC")
                        self.populateFunctionParameters(decisionWidget, decision_name, options, type="FC")

                    elif func_value.startswith("comet.graph") or func_value.startswith("bct."):
                        # Set the category to FC
                        categoryComboBox.setCurrentText("Graph")
                        self.populateFunctionParameters(decisionWidget, decision_name, options, type="Graph")
                    else:
                        # Handle other categories or leave it empty
                        categoryComboBox.setCurrentText("Other")
                else:
                    QMessageBox.warning(self, "Warning", "Something went wrong when trying to populate the multiverse containers, please check the forking paths.")
            else:
                # Non-dict options, treat as General category
                categoryComboBox.setCurrentText("General")
                decisionNameInput.setText(decision_name)
                decisionOptionsInput.setText(", ".join([str(option) for option in options]))  # Convert options list to strings

            # Add the decision widget to the main layout
            self.createMvContainerLayout.insertWidget(self.createMvContainerLayout.count() - 1, decisionWidget)  # Insert before the button layout widget
            self.includeDecision(categoryComboBox, decisionNameInput, decisionOptionsInput)

    def populateFunctionParameters(self, decisionWidget, decisionName, options, type=None):
        """
        Populate the function parameter container with parameters from the given dictionary.
        """
        for option in options:
            # Get the widgets
            decisionNameInput = decisionWidget.findChild(QLineEdit, "decisionNameInput")
            functionComboBox = decisionWidget.findChild(QComboBox, "functionComboBox")
            addOptionButton = decisionWidget.findChild(QPushButton, "addOptionButton")
            collapseButton = decisionWidget.findChild(QPushButton, "collapseButton")

            # Get the func name and update the functionComboBox
            func_name = option['func'].split('.')[-1]

            if type == "FC":
                comboboxItem = self.connectivityMethods.get(func_name, None)
            elif type == "Graph":
                comboboxItem = self.graphOptions.get(func_name, None)

            if comboboxItem is None:
                print(f"Warning: comboboxItem for function '{func_name}' not found in the provided type '{type}'.")
                continue  # Skip if combobox item is not found

            functionComboBox.setCurrentText(comboboxItem)
            decisionNameInput.setText(decisionName)

            # Get the parameter container and update its widgets with the dict contents
            parameterContainer = decisionWidget.findChild(QWidget, comboboxItem)

            if parameterContainer is None:
                print(f"Warning: parameterContainer '{comboboxItem}' not found.")
                continue  # Skip if parameter container is not found

            # Set the name
            name_edit = parameterContainer.findChild(QLineEdit, f"name_edit_{comboboxItem}")
            if name_edit:
                name_edit.setText(option['name'])
            else:
                print(f"Warning: name_edit '{f'name_edit_{comboboxItem}'}' not found in parameterContainer.")

            # Set the args
            args = option.get('args', {})
            for arg_name, arg_value in args.items():
                # Ensure that when you created the widgets, the object name was set as arg_name
                widget = parameterContainer.findChild(QWidget, f"{arg_name}_{comboboxItem}")
                if widget:
                    if isinstance(widget, QLineEdit):
                        widget.setText(str(arg_value))
                    elif isinstance(widget, QComboBox):
                        widget.setCurrentText(str(arg_value))
                    elif isinstance(widget, QSpinBox):
                        widget.setValue(arg_value)
                    elif isinstance(widget, QDoubleSpinBox):
                        widget.setValue(arg_value)
                else:
                    print(f"Warning: Widget '{arg_name}_{comboboxItem}' not found in parameterContainer.")

            # Add options and collapse the container
            addOptionButton.click()

        collapseButton.click()

    def addNewDecision(self, layout, buttonLayoutWidget):
        """
        Add a new decision widget to the layout
        """
        # Collapse all existing parameter containers before adding a new one
        for container in self.mv_containers:
            parameterContainer = container.findChild(QWidget, "parameterContainer")
            collapseButton = container.findChild(QPushButton, "collapseButton")
            if parameterContainer and collapseButton and parameterContainer.isVisible():
                self.collapseOption(collapseButton, parameterContainer)

        # Add new decision container
        newDecisionWidget = self.addDecisionContainer()

        # Insert the new decision widget before the buttonLayoutWidget
        layout.insertWidget(layout.indexOf(buttonLayoutWidget), newDecisionWidget)

        return

    def onCategoryComboBoxChanged(self, categoryComboBox, functionComboBox, parameterContainer, addOptionButton, collapseButton, decisionNameInput, decisionOptionsInput):
        """
        Handle if the type of the decision is changed
        """
        selected_category = categoryComboBox.currentText()
        self.clearLayout(parameterContainer.layout())

        functionComboBox.clear()
        decisionNameInput.clear()
        decisionOptionsInput.clear()

        functionComboBox.hide()
        parameterContainer.hide()
        addOptionButton.hide()
        collapseButton.hide()

        # Sets up the layout and input fields for the new category/options
        decisionOptionsInput.setPlaceholderText("Options (comma-separated)" if selected_category == "General" else "Define options below")
        decisionOptionsInput.setReadOnly(selected_category != "General")

        if selected_category in ["FC", "Graph"]:
            methods = self.graphOptions if selected_category == "Graph" else self.connectivityMethods
            for name, description in methods.items():
                functionComboBox.addItem(description, name)

            functionComboBox.show()
            parameterContainer.show()
            addOptionButton.show()
            collapseButton.show()

        elif selected_category == "Other":
            parameterContainer.show()
            addOptionButton.show()
            collapseButton.show()
            self.otherOptionCategory(parameterContainer)

        self.update()

        return

    def getFunctionParameters(self, parameterContainer):
        """
        Get a dict with the current function parameters
        """
        params_dict = {}
        paramLayout = parameterContainer.layout()

        # Iterate over all layout items in the parameter layout
        for i in range(paramLayout.count()):
            layout_item = paramLayout.itemAt(i)

            # Ensure the layout item is a QHBoxLayout (each parameter is in its own QHBoxLayout)
            if isinstance(layout_item.layout(), QHBoxLayout):
                param_layout = layout_item.layout()

                # The parameter name is in the QLabel, and the value is in the second widget (QLineEdit, QComboBox, etc.)
                if param_layout.count() >= 2:
                    # Extract the parameter name from the QLabel
                    param_name_label = param_layout.itemAt(0).widget()
                    if isinstance(param_name_label, QLabel):
                        param_name = param_name_label.text().rstrip(':')  # Remove the colon at the end

                        # Extract the parameter value from the appropriate widget type
                        param_widget = param_layout.itemAt(1).widget()
                        if isinstance(param_widget, QLineEdit):
                            param_value = param_widget.text()
                        elif isinstance(param_widget, QComboBox):
                            param_value = param_widget.currentText()
                        elif isinstance(param_widget, QSpinBox) or isinstance(param_widget, QDoubleSpinBox):
                            param_value = param_widget.value()

                        # Convert to appropriate boolean type if necessary (LineEdit and QComboBox return strings)
                        if param_value == "True":
                            param_value = True
                        elif param_value == "False":
                            param_value = False

                        # Add the parameter name and value to the dictionary
                        params_dict[param_name] = param_value

        return params_dict

    def updateFunctionParameters(self, functionComboBox, parameterContainer):
        """
        Create and update all the parameter widgets based on the selected function
        """
        if functionComboBox.currentData() is None:
            return

        func_key = functionComboBox.currentData()
        try:
            method_name = self.connectivityMethods[func_key]
        except:
            method_name = self.graphOptions[func_key]
        prefix = method_name.strip().split(' ')[0]

        if prefix == "COMET" or prefix == "PREP" or prefix == "BCT":
            func = getattr(graph, functionComboBox.currentData())
        elif prefix == "CONT" or prefix == "STATE" or prefix == "STATIC":
            dfc_class_ = getattr(connectivity, functionComboBox.currentData())
            func = dfc_class_.__init__
        else:
            QMessageBox.warning(self, "Error", "Function is not recognized")

        # Retrieve the signature of the function
        func_signature = inspect.signature(func)
        type_hints = get_type_hints(func)

        # Clear previous parameters
        self.clearLayout(parameterContainer.layout())

        # Calculate the maximum label width
        max_label_width = 0
        font_metrics = QFontMetrics(self.font())
        for name, param in func_signature.parameters.items():
            if name not in ['self', 'args', 'kwargs']:  # Skip unwanted parameters
                label_width = font_metrics.boundingRect(f"{name}:").width()
                max_label_width = max(max_label_width, label_width)

        # Add the function combobox
        func_layout = QHBoxLayout()
        func_label = QLabel("Option:")
        func_label.setFixedWidth(max_label_width + 20)
        func_layout.addWidget(func_label)
        func_layout.addWidget(functionComboBox)
        parameterContainer.layout().addLayout(func_layout)

        # Add the 'Name' QLineEdit before other parameters
        name_layout = QHBoxLayout()
        name_label = QLabel("Name:")
        name_label.setFixedWidth(max_label_width + 20)
        name_edit = QLineEdit()
        name_edit.setObjectName(f"name_edit_{functionComboBox.currentText()}")
        name_edit.setPlaceholderText("Option name")
        name_layout.addWidget(name_label)
        name_layout.addWidget(name_edit)
        parameterContainer.layout().addLayout(name_layout)

        is_first_parameter = True  # Flag to identify the first parameter

        # Iterate over parameters in the function signature
        temp_widgets = {}
        for name, param in func_signature.parameters.items():
            if name not in ['self', 'copy', 'args', 'kwargs']:  # Skip unwanted parameters
                # Horizontal layout for each parameter
                param_layout = QHBoxLayout()
                param_type = type_hints.get(name)
                param_default = 1 if isinstance(param.default, inspect._empty) else param.default

                if param_default == None:
                    if param_type == bool:
                        param_default = False
                    elif param_type == int or param_type == float:
                        param_default = 1
                    else:
                        param_default = "empty"

                # Create a label for the parameter and set its fixed width
                param_label = QLabel(f"{name}:")
                param_label.setFixedWidth(max_label_width + 20)  # Add some padding
                param_layout.addWidget(param_label)

                # For the first parameter, set its value based on the data source and lock it
                if is_first_parameter:
                    param_widget = QLineEdit()
                    param_widget.setPlaceholderText("Data (name of the variable in the script)")
                    is_first_parameter = False  # Update the flag so this block runs only for the first parameter
                else:
                    # Bool
                    if param_type == bool:
                        param_widget = QComboBox()
                        param_widget.addItems(["False", "True"])
                        param_widget.setCurrentIndex(int(param_default))
                    # Int
                    elif param_type == int:
                        param_widget = QSpinBox()
                        param_widget.setValue(param_default)
                        param_widget.setMaximum(10000)
                        param_widget.setMinimum(-10000)
                        param_widget.setSingleStep(1)
                    # Float
                    elif param_type == float:
                        param_widget = QDoubleSpinBox()
                        if name == "threshold":
                            param_widget.setValue(0.0)
                        else:
                            param_widget.setValue(param_default)

                        if prefix == "COMET" or prefix == "PREP" or prefix == "BCT":
                            param_widget.setMaximum(1.0)
                            param_widget.setMinimum(0.0)
                        else:
                            param_widget.setMaximum(10000.0)
                            param_widget.setMinimum(-10000.0)

                        param_widget.setSingleStep(0.01)

                    # String
                    elif get_origin(type_hints.get(name)) is Literal:
                        options = type_hints.get(name).__args__
                        param_widget = QComboBox()
                        param_widget.addItems([str(option) for option in options])
                    # Fallback
                    else:
                        param_widget = QLineEdit(str(param.default) if param.default != inspect.Parameter.empty else "")

                param_widget.setObjectName(f"{name}_{functionComboBox.currentText()}") # Set object name to allow us to find the widgets later when e.g. populating it from a template script
                temp_widgets[name] = (param_label, param_widget)
                param_layout.addWidget(param_widget)
                parameterContainer.layout().addLayout(param_layout)
                parameterContainer.setObjectName(functionComboBox.currentText())

        # Adjust visibility based on 'type' parameter
        type_widget = None
        if 'type' in temp_widgets:
            _, type_widget = temp_widgets['type']

        if type_widget:
            # Function to update parameter visibility
            def updateVisibility():
                selected_type = type_widget.currentText()
                if selected_type == 'absolute':
                    if 'threshold' in temp_widgets:
                        temp_widgets['threshold'][0].show()
                        temp_widgets['threshold'][1].show()
                    if 'density' in temp_widgets:
                        temp_widgets['density'][0].hide()
                        temp_widgets['density'][1].hide()
                elif selected_type == 'density':
                    if 'threshold' in temp_widgets:
                        temp_widgets['threshold'][0].hide()
                        temp_widgets['threshold'][1].hide()
                    if 'density' in temp_widgets:
                        temp_widgets['density'][0].show()
                        temp_widgets['density'][1].show()

            # Connect the signal from the type_widget to the updateVisibility function
            type_widget.currentIndexChanged.connect(updateVisibility)
            updateVisibility()

        return

    def otherOptionCategory(self, parameterContainer):
        """
        "Other" category for custom functions
        """
        # Clear the parameter container
        self.clearLayout(parameterContainer.layout())

        # Add a single QLineEdit for the user to input the option
        font_metrics = QFontMetrics(self.font())
        label_width = font_metrics.boundingRect(f"Parameters:").width()

        option_layout = QHBoxLayout()
        option_label = QLabel("Function:")
        option_label.setFixedWidth(label_width + 20)
        option_edit = QLineEdit()
        option_edit.setObjectName("option_edit")

        option_edit.setPlaceholderText("Name of the function (e.g. np.mean)")
        option_layout.addWidget(option_label)
        option_layout.addWidget(option_edit)
        parameterContainer.layout().addLayout(option_layout)

        param_layout = QHBoxLayout()
        param_label = QLabel("Parameters:")
        param_label.setFixedWidth(label_width + 20)
        param_edit = QLineEdit()
        param_edit.setObjectName("param_edit")
        param_edit.setPlaceholderText("Function parameters as dict (e.g. {'axis': 0})")
        param_layout.addWidget(param_label)
        param_layout.addWidget(param_edit)
        parameterContainer.layout().addLayout(param_layout)

        return

    def addOption(self, functionComboBox, parameterContainer, nameInputField, optionsInputField):
        """
        Add option to a decision
        """
        # Name for the decision must be provided
        name = nameInputField.text().strip()
        if not name:
            QMessageBox.warning(self, "Input Error", "Please ensure a name is provided for the decision.")
            return


        # Retrieve the selected function key and determine its module prefix
        func_key = functionComboBox.currentData()

        try:
            method_name = self.connectivityMethods[func_key]
        except:
            method_name = self.graphOptions[func_key]
        prefix = method_name.strip().split(' ')[0]

        if prefix == "COMET" or prefix == "PREP":
            module_prefix = "comet.graph"
        elif prefix == "BCT":
            module_prefix = "bct"
        elif prefix == "CONT" or prefix == "STATE" or prefix == "STATIC":
            module_prefix = "comet.connectivity"
        else:
            QMessageBox.warning(self, "Error", "Function is not recognized")

        # Construct the full function path
        func = f"{module_prefix}.{func_key}"

        params = self.getFunctionParameters(parameterContainer)
        option_name = params.get('Name', '').strip()

        if not option_name:
            QMessageBox.warning(self, "Error", "Please provide a name for the option")
            return

        # Get current values from name and options input fields
        currentName = nameInputField.text().strip()
        currentOptions = optionsInputField.text().strip()

        # Prepare the new options string by appending the new option name
        newOptions = f"{currentOptions}, {option_name}" if currentOptions else option_name
        optionsInputField.setText(newOptions)

        # Construct the dict for the new option
        option_dict = {
            "name": option_name,
            "func": func,
            "args": {k: v for k, v in params.items() if k != 'Name'}
        }

        # Append the new option to the existing list in the data dictionary
        if currentName not in self.data.forking_paths:
            self.data.forking_paths[currentName] = []

        # Add to forking paths
        self.data.forking_paths[currentName].append(option_dict)
        return

    def collapseOption(self, collapseButton, parameterContainer):
        """
        Collapse the option layout
        """
        if collapseButton.text() == " \u25B2 ":
            parameterContainer.hide()
            collapseButton.setText(" \u25BC ")
            return

        if collapseButton.text() == " \u25BC ":
            parameterContainer.show()
            collapseButton.setText(" \u25B2 ")
            return

        return

    def includeDecision(self, categoryComboBox, nameInput, optionsInput):
        """
        Adds decision to the script
        """
        category = categoryComboBox.currentText()
        name = nameInput.text().strip()

        if not name:
            QMessageBox.warning(self, "Input Error", "Please ensure a name is provided for the decision.")
            return

        if category == "General":
            options = [self.setDtypeForOption(option.strip()) for option in optionsInput.text().split(',') if option.strip()]
            self.data.forking_paths[name] = options
        else:
            options = self.data.forking_paths[name]

        if name and options:
            self.generateMultiverseScript()
        else:
            QMessageBox.warning(self, "Input Error", "Please ensure a name and at least one option are provided.")
        return

    def setDtypeForOption(self, option):
        """
        Handle data conversion based on the input
        """
        # Try to convert to integer
        try:
            return int(option)
        except ValueError:
            pass

        # Try to convert to float
        try:
            return float(option)
        except ValueError:
            pass

        # Check for boolean values
        if option.lower() in ['true', 'false']:
            return option.lower() == 'true'

        return option

    def removeDecision(self, decisionNameInput, decisionWidget, optionsInputField):
        """
        Remove one option with each click and finally the entire decision
        """
        key = decisionNameInput.text().strip()

        # No key means the decision widget is empty, so clear and delete everything
        if key == "":
            if decisionWidget.layout():
                self.clearLayout(decisionWidget.layout())
                decisionWidget.deleteLater()
                optionsInputField.clear()
                self.mv_containers.remove(decisionWidget)
                self.generateMultiverseScript()
                return

        if key in self.data.forking_paths:
            options = self.data.forking_paths[key]

            # Remove the last option and update the input field
            if options:
                options.pop()  # Remove the last option
                options_str = ', '.join(opt['name'] if isinstance(opt, dict) else str(opt) for opt in options)
                optionsInputField.setText(options_str)  # Update the input field

            # If all options are removed or there are no options, clear and delete everything
            if not options:
                del self.data.forking_paths[key]
                if decisionWidget.layout():
                    self.clearLayout(decisionWidget.layout())  # Clear all child widgets and sub-layouts
                decisionWidget.deleteLater()  # Delete the decision widget
                optionsInputField.clear()  # Clear the options input field
                self.mv_containers.remove(decisionWidget)

        # If all containers have been deleted, we reset everything
        if len(self.mv_containers) == 0:
            self.resetMultiverseAnalysis()
        else:
            self.generateMultiverseScript()

        return

    def clearLayout(self, layout):
        """
        Clear the entire decisionWidget layout
        """
        # Recursively delete all items in a layout. This method handles both widgets and sub-layouts.
        while layout.count():
            item = layout.takeAt(0)  # Take the first item in the layout
            if item.widget():
                if item.widget().objectName() != "functionComboBox":  # Ensure it's not the functionComboBox
                    item.widget().deleteLater()  # Delete the widget if the item is a widget
            elif item.layout():
                self.clearLayout(item.layout())  # Recursively clear if the item is a layout
                item.layout().deleteLater()  # Delete the sub-layout after clearing it

    def resetMultiverseAnalysis(self):
            self.mv_from_file = False
            self.multiverseName = None
            self.multiverseName = None
            self.loadedScriptDisplay.clear()

            self.createMultiverseButton.setEnabled(False)
            self.runMultiverseButton.setEnabled(False)

            self.generateMultiverseScript(init_template=True)

    def _listAllWidgets(self, container):
        """
        List all widgets and their memory addresses within a given container.
        """
        widgets = container.findChildren(QWidget)
        for widget in widgets:
            print(f"Widget: {widget.objectName()}, Type: {type(widget).__name__}, Address: {hex(id(widget))}")

    # Template script functions
    def toggleReadOnly(self):
        if self.scriptDisplay.isReadOnly():
            self.scriptDisplay.setReadOnly(False)
            self.toggleButton.setText(" 🔓 ")  # editable
        else:
            self.scriptDisplay.setReadOnly(True)
            self.toggleButton.setText(" 🔒 ")  # read-only

    def updateToggleButtonPosition(self, event):
        # Update the position of the toggle button when the QTextEdit is resized
        self.toggleButton.move(self.scriptDisplay.width() - self.toggleButton.width() - 25, 5)
        QTextEdit.resizeEvent(self.scriptDisplay, event)

    def generateMultiverseScript(self, init_template=False):
        """
        Generates the multiverse script on the right side of the tab
        """
        # If we loaded a template from file we will not overwrite the analysis_template() function
        if self.mv_from_file:
            script_content = (
                "from comet.multiverse import Multiverse\n"
                "\n"
                "forking_paths = {\n"
            )
            for name, options in self.data.forking_paths.items():
                if isinstance(options, list) and all(isinstance(item, dict) for item in options):
                    formatted_options = json.dumps(options, indent=4)
                    formatted_options = formatted_options.replace('true', 'True').replace('false', 'False').replace('null', 'None')
                    script_content += f'    "{name}": {formatted_options},\n'
                else:
                    script_content += f'    "{name}": {options},\n'

            script_content += (
                "}\n\n"
                f"{self.analysis_template}"
            )

        # Create temnplate from scratch, analysis_template() function is empty
        else:
            if init_template:
                script_content = (
                    "\"\"\"\n"
                    "Running Multiverse analysis\n"
                    "\n"
                    "Multiverse analysis requires a Python script to be created by the user.\n"
                    "An initial template for this can be created through the GUI, with forking paths being stored in a dict and later used through double curly braces in the template function.\n\n"
                    "This example shows how one would create and run a multiverse analysis which will generate 3 Python scripts (universes) printing the numbers 1, 2, and 3, respectively.\n"
                    "\"\"\"\n"
                    "\n"
                    "from comet.multiverse import Multiverse\n"
                    "\n"
                    "forking_paths = {\n"
                    "    \"numbers\": [1, 2, 3]\n"
                    "}\n"
                    "\n"
                    "def analysis_template():\n"
                    "    print({{numbers}})\n"
                    "\n"
                )

            else:
                script_content = (
                    "from comet.multiverse import Multiverse\n"
                    "\n"
                    "forking_paths = {\n"
                )
                for name, options in self.data.forking_paths.items():
                    if isinstance(options, list) and all(isinstance(item, dict) for item in options):
                        formatted_options = json.dumps(options, indent=4)
                        formatted_options = formatted_options.replace('true', 'True').replace('false', 'False').replace('null', 'None')
                        script_content += f'    "{name}": {formatted_options},\n'
                    else:
                        script_content += f'    "{name}": {options},\n'

                script_content += (
                    "}\n\n"
                    "def analysis_template():\n"
                    "    # The following forking paths are available for multiverse analysis:\n"
                )

                for name in self.data.forking_paths:
                    script_content += f"    {{{{{name}}}}}\n"

        self.scriptDisplay.setText(script_content)
        self.scriptDisplay.setReadOnly(True)
        self.toggleButton.setText("🔒")

    def selectMultiverseFolder(self):
        self.data.multiverse_folder = QFileDialog.getExistingDirectory(self, "Select folder for multiverse analysis")
        self.chooseFolderEdit.setText(self.data.multiverse_folder)
        # User canceled the selection
        if not self.data.multiverse_folder:
            return

    def loadMultiverseScript(self):
        """
        Load a multiverse script and extract specific components.
        """
        fileFilter = "All supported files (*.py *.ipynb);;MAT files (*.mat);;Python files (*.py);;Jupyter notebooks (*.ipynb)"
        self.multiverseFileName, _ = QFileDialog.getOpenFileName(self, "Load multiverse template file", "", fileFilter)
        self.multiverseName = self.multiverseFileName.split('/')[-1].split('.')[0]
        self.mv_from_file = True
        self.createMultiverseButton.setEnabled(True)
        self.runMultiverseButton.setEnabled(False)
        self.paralleliseMultiverseSpinbox.setEnabled(False)
        self.plotButton.setEnabled(False)

        self.data.forking_paths = {}

        if self.multiverseFileName:
            try:
                if self.multiverseFileName.endswith('.ipynb'):
                    # Convert .ipynb to Python script
                    with open(self.multiverseFileName, 'r', encoding='utf-8') as file:
                        notebook = json.load(file)
                        self.multiverseScriptContent = self.convertNotebookToScript(notebook)
                else:
                    # Load as a normal Python script
                    with open(self.multiverseFileName, 'r', encoding='utf-8') as file:
                        self.multiverseScriptContent = file.read()

                # Parse the script and extract components
                extracted_content = self.extractScriptComponents(self.multiverseScriptContent)
                self.scriptDisplay.setText(extracted_content)
                self.scriptDisplay.setReadOnly(False)

                # Update the text box
                self.loadedScriptDisplay.setText(self.multiverseFileName.split('/')[-1])

                # Load the forking paths and populate the decision containers
                tree = ast.parse(extracted_content)

                # Traverse the AST to find the forking_paths assignment
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and target.id == 'forking_paths':
                                # Safely evaluate the forking_paths dictionary (node.value)
                                forking_paths = ast.literal_eval(node.value)

                self.populateMultiverseContainers(forking_paths)

                self.specFigure.clf()
                self.specCanvas.draw()


            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to read the file: {str(e)}")

    def extractScriptComponents(self, script_content):
        """
        Extract the forking paths dict and analysis_template function.
        """
        tree = ast.parse(script_content)
        extracted_content = ["from comet.multiverse import Multiverse"]

        # Initialize placeholders for the extracted components
        forking_paths = None
        analysis_template = None

        for node in tree.body:
            # Extract the forking_paths dictionary
            if isinstance(node, ast.Assign):
                if isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'forking_paths':
                    forking_paths = ast.get_source_segment(script_content, node)
                    extracted_content.append(forking_paths)

            # Extract the analysis_template function
            if isinstance(node, ast.FunctionDef) and node.name == 'analysis_template':
                analysis_template = ast.get_source_segment(script_content, node)
                self.analysis_template = analysis_template
                extracted_content.append(analysis_template)

        # Add empty forking_paths dictionary if missing
        if not forking_paths:
            extracted_content.append("forking_paths = {}")

        # Add empty analysis_template function if missing
        if not analysis_template:
            extracted_content.append("def analysis_template():\n    pass")

        if not forking_paths or not analysis_template:
            QMessageBox.warning(self, "Error", "Failed to extract the forking_paths dictionary and/or analysis_template function. Empty placeholders were added.")

        return "\n\n".join(extracted_content)

    def convertNotebookToScript(self, notebook):
        """
        Convert a Jupyter notebook JSON to a Python script.
        """
        scriptContent = "error"
        try:
            scriptContent = multiverse.notebookToScript(notebook)
        except KeyError as e:
            QMessageBox.critical(self, "Error", f"Invalid notebook format: {str(e)}")
        return scriptContent

    def createMultiverse(self):
        """
        Create the multiverse from the template file
        """
        if hasattr(self, 'multiverseFileName'):
            multiverse_save_path = self.data.multiverse_folder
            os.makedirs(multiverse_save_path, exist_ok=True)

            # Create a template script in the multiverse folder (the script content is taken from the GUI)
            template_file = os.path.join(multiverse_save_path, "template.py")
            with open(template_file, "w") as file:
                file.write("# Template script containing the required data for multiverse analysis.\n")
                file.write("# This file is used/overwritten by the GUI, users should usually directly interact with their own multiverse script.\n\n")
                file.write(self.scriptDisplay.toPlainText())

            # Load the forking paths and the analysis template function
            module_name = os.path.splitext(os.path.basename(template_file))[0]
            spec = util.spec_from_file_location(module_name, template_file)
            module = util.module_from_spec(spec)
            spec.loader.exec_module(module)

            forking_paths = getattr(module, 'forking_paths', None)
            analysis_template = getattr(module, 'analysis_template', None)

            self.mverse = multiverse.Multiverse(name=self.multiverseFileName)
            self.mverse.create(analysis_template, forking_paths)

            # If we already have results, we can populate the measure input
            results_file = os.path.join(self.mverse.results_dir, 'universe_1.pkl')
            if os.path.exists(results_file):
                with open(results_file, 'rb') as file:
                    universe_data = pickle.load(file)

                variable_names = list(universe_data.keys())
                self.measureInput.clear()
                self.measureInput.addItems(variable_names)

            # Create the summary plot
            self.plotMultiverseSummary()

            # Check size of the multiverse. If we have an identical amount of results we take this as a heuristic that the multiverse was already run
            summary_df = self.mverse.summary()
            all_files = os.listdir(self.mverse.results_dir)
            num_results = len([f for f in all_files if f.startswith('universe_') and f.endswith('.pkl')])

            if num_results == len(summary_df):
                self.plotSpecificationCurve()

            # Enable the buttons
            self.runMultiverseButton.setEnabled(True)
            self.paralleliseMultiverseSpinbox.setEnabled(True)

        else:
            QMessageBox.warning(self, "Multiverse Analysis", "No multiverse template script was provided.")

    def resetMultiverseScript(self):
        self.mv_from_file = False
        self.generateMultiverseScript(init_template=True)

    def saveMultiverseScript(self):
        """
        Save the template script
        """
        script_text = self.scriptDisplay.toPlainText()
        fileName, _ = QFileDialog.getSaveFileName(self, "Save Script", "", "Python Files (*.py);;All Files (*)")

        if fileName:
            if not fileName.endswith('.py'):
                fileName += '.py'

            with open(fileName, 'w') as file:
                file.write(script_text)

    def runMultiverseScript(self):
        """
        Run the multiverse analysis from the generated/loaded script
        """
        self.createMultiverseButton.setEnabled(False)
        self.runMultiverseButton.setEnabled(False)
        self.paralleliseMultiverseSpinbox.setEnabled(False)
        self.plotButton.setEnabled(False)

        self.mvThread = QThread()
        self.mvWorker = Worker(self.mverse.run, {"parallel": self.paralleliseMultiverseSpinbox.value(), "folder": self.data.multiverse_folder})
        self.mvWorker.moveToThread(self.mvThread)

        self.mvThread.started.connect(self.mvWorker.run)
        self.mvWorker.finished.connect(self.mvThread.quit)
        self.mvWorker.finished.connect(self.mvWorker.deleteLater)
        self.mvThread.finished.connect(self.mvThread.deleteLater)

        self.mvWorker.result.connect(self.handleMultiverseResult)
        self.mvWorker.error.connect(self.handleMultiverseError)

        self.mvThread.start()

    def handleMultiverseResult(self, result):
        QMessageBox.information(self, "Multiverse Analysis", "Multiverse analysis completed successfully.")
        self.createMultiverseButton.setEnabled(True)
        self.runMultiverseButton.setEnabled(True)
        self.paralleliseMultiverseSpinbox.setEnabled(True)
        self.plotButton.setEnabled(True)

        # Populate the measure input
        results_file = os.path.join(self.mverse.results_dir, 'universe_1.pkl')
        if os.path.exists(results_file):
            with open(results_file, 'rb') as file:
                universe_data = pickle.load(file)

            variable_names = list(universe_data.keys())
            self.measureInput.clear()
            self.measureInput.addItems(variable_names)

    def handleMultiverseError(self, error):
        QMessageBox.warning(self, "Multiverse Analysis", f"An error occurred during multiverse analysis: {error}")
        self.createMultiverseButton.setEnabled(True)
        self.runMultiverseButton.setEnabled(True)
        self.paralleliseMultiverseSpinbox.setEnabled(True)

    def plotMultiverse(self):
        # Remove the old canvas from the layout if it exists
        if hasattr(self, 'multiverseCanvas'):
            self.plotMvTab.layout().removeWidget(self.multiverseCanvas)
            self.multiverseCanvas.setParent(None)

        fig = self.mverse.visualize()
        self.multiverseCanvas = FigureCanvas(fig)

        self.plotMvTab.layout().addWidget(self.multiverseCanvas)
        self.multiverseCanvas.draw()

    def createSpecificationCurveWidgets(self):
        # Create a layout for the parameter inputs
        paramLayout = QVBoxLayout()

        # First Row: Measure, Title
        firstRowLayout = QHBoxLayout()

        # Measure
        firstRowLayout.addWidget(QLabel('Measure:'))
        self.measureInput = QComboBox(self)
        firstRowLayout.addWidget(self.measureInput)

        # Title (string)
        firstRowLayout.addWidget(QLabel('Title:'))
        self.titleInput = QLineEdit(self)
        self.titleInput.setText("Specification Curve")
        firstRowLayout.addWidget(self.titleInput)

        firstRowLayout.setStretch(1, 1)
        firstRowLayout.setStretch(3, 1)

        paramLayout.addLayout(firstRowLayout)

        # Second Row: Baseline, P-value, CI, Smooth CI, Figure Size, Plot Button
        secondRowLayout = QHBoxLayout()

        # Baseline (float spin box)
        secondRowLayout.addWidget(QLabel('Baseline:'))
        self.baselineInput = CustomDoubleSpinbox(special_value=None, min=-0.1, max=999.0)
        self.baselineInput.setSingleStep(0.1)
        secondRowLayout.addWidget(self.baselineInput)

        # P-value (float spin box)
        secondRowLayout.addWidget(QLabel('P-value:'))
        self.pValueInput = CustomDoubleSpinbox(special_value=None, min=0.0, max=1.0)
        self.pValueInput.setSingleStep(0.01)
        secondRowLayout.addWidget(self.pValueInput)

        # CI (int spin box)
        secondRowLayout.addWidget(QLabel('CI:'))
        self.ciInput = CustomSpinBox(special_value=None, min=89, max=100)
        self.ciInput.setSingleStep(1)
        secondRowLayout.addWidget(self.ciInput)

        # Smooth CI (checkbox)
        self.smoothCiCheckbox = QCheckBox('Smooth', self)
        self.smoothCiCheckbox.setChecked(True)  # Default is True
        secondRowLayout.addWidget(self.smoothCiCheckbox)

        # Figure Size (two numbers for width and height)
        secondRowLayout.addWidget(QLabel('Figsize:'))
        self.figsizeInput = QLineEdit(self)
        self.figsizeInput.setText("5,6")
        secondRowLayout.addWidget(self.figsizeInput)

        # Plot Button
        self.plotButton = QPushButton(' Create Plot ', self)
        self.plotButton.clicked.connect(self.plotSpecificationCurve)
        secondRowLayout.addWidget(self.plotButton)
        self.plotButton.setEnabled(False)

        # Add layouts
        paramLayout.addLayout(secondRowLayout)
        self.specTab.layout().addLayout(paramLayout)

    def createSummaryWidgets(self):
        # Create a layout for the parameter inputs
        paramLayout = QVBoxLayout()

        # First Row: Universe, Node Size
        firstRowLayout = QHBoxLayout()

        # Universe (int spin box)
        firstRowLayout.addWidget(QLabel('Universe:'))
        self.universeInput = CustomSpinBox(special_value=None, min=0, max=999)
        self.universeInput.setSingleStep(1)
        firstRowLayout.addWidget(self.universeInput)

        # Node Size (int spin box)
        firstRowLayout.addWidget(QLabel('Node Size:'))
        self.nodeSizeInput = QSpinBox()
        self.nodeSizeInput.setMinimum(200)
        self.nodeSizeInput.setMaximum(5000)
        self.nodeSizeInput.setSingleStep(200)
        self.nodeSizeInput.setValue(1000)
        firstRowLayout.addWidget(self.nodeSizeInput)

        # Figure Size (two numbers for width and height)
        firstRowLayout.addWidget(QLabel('Figsize:'))
        self.figsizeInput = QLineEdit(self)
        self.figsizeInput.setText("8,6")
        firstRowLayout.addWidget(self.figsizeInput)

        # Label Offset (float spin box)
        firstRowLayout.addWidget(QLabel('Label Offset:'))
        self.labelOffsetInput = QDoubleSpinBox()
        self.labelOffsetInput.setMinimum(0.0)
        self.labelOffsetInput.setMaximum(1.0)
        self.labelOffsetInput.setSingleStep(0.01)
        self.labelOffsetInput.setValue(0.04)
        firstRowLayout.addWidget(self.labelOffsetInput)

        # Plot Button
        self.plotMvButton = QPushButton(' Update Plot ', self)
        self.plotMvButton.clicked.connect(self.plotMultiverseSummary)
        firstRowLayout.addWidget(self.plotMvButton)

        # Add layouts
        paramLayout.addLayout(firstRowLayout)
        self.plotMvTab.layout().addLayout(paramLayout)

    def plotMultiverseSummary(self):
        # Get input values
        universe = None if self.universeInput.value() == 0 else self.universeInput.value()
        node_size = self.nodeSizeInput.value()
        figsize = tuple(map(int, self.figsizeInput.text().split(',')))
        label_offset = self.labelOffsetInput.value()

        # Plot the multiverse summary
        fig = self.mverse.visualize(universe=universe, node_size=node_size, figsize=figsize, label_offset=label_offset)

        if hasattr(self, 'multiverseCanvas'):
            self.plotMvTab.layout().removeWidget(self.multiverseCanvas)
            self.multiverseCanvas.setParent(None)
            plt.close(self.multiverseCanvas.figure)  # Close the old figure

        with tempfile.NamedTemporaryFile(suffix='.png', delete=True) as tmpfile:
            fig.savefig(tmpfile.name, bbox_inches='tight')

            fig = plt.figure()
            img = plt.imread(tmpfile.name)
            plt.imshow(img)
            plt.tight_layout()
            plt.axis('off')

            self.multiverseCanvas = FigureCanvas(fig)
            self.plotMvTab.layout().insertWidget(0, self.multiverseCanvas)  # Insert the canvas at the top
            self.multiverseCanvas.draw()

    def plotSpecificationCurve(self):
        """
        Run the multiverse analysis from the generated/loaded script
        """
        measure = self.measureInput.currentText()
        title = self.titleInput.text()
        baseline = None if self.baselineInput.value() == self.baselineInput.minimum() else self.baselineInput.value()
        p_value = None if self.pValueInput.value() == self.pValueInput.minimum() else self.pValueInput.value()
        ci = None if self.ciInput.value() == self.ciInput.minimum() else self.ciInput.value()
        smooth_ci = self.smoothCiCheckbox.isChecked()
        figsize = tuple(map(int, self.figsizeInput.text().split(',')))  # make string a tuple

        self.plotButton.setEnabled(False)

        self.plotSpecificationCurveThread = QThread()
        self.plotSpecificationCurveWorker = Worker(self.mverse.specification_curve, {"measure": measure, "title": title, "baseline": baseline, \
                                                                                     "p_value": p_value, "ci": ci, "smooth_ci": smooth_ci, "figsize": figsize})
        self.plotSpecificationCurveWorker.moveToThread(self.plotSpecificationCurveThread)

        self.plotSpecificationCurveThread.started.connect(self.plotSpecificationCurveWorker.run)
        self.plotSpecificationCurveWorker.finished.connect(self.plotSpecificationCurveThread.quit)
        self.plotSpecificationCurveWorker.finished.connect(self.plotSpecificationCurveWorker.deleteLater)
        self.plotSpecificationCurveThread.finished.connect(self.plotSpecificationCurveWorker.deleteLater)

        self.plotSpecificationCurveWorker.result.connect(self.handleSpecificationCurveResult)
        self.plotSpecificationCurveWorker.error.connect(self.handleSpecificationCurveError)

        self.plotSpecificationCurveThread.start()

    def handleSpecificationCurveResult(self, result):
        # Remove the old canvas from the layout if it exists
        if hasattr(self, 'specCanvas'):
            self.specTab.layout().removeWidget(self.specCanvas)
            self.specCanvas.setParent(None)
            plt.close(self.specCanvas.figure)  # Close the old figure

        with tempfile.NamedTemporaryFile(suffix='.png', delete=True) as tmpfile:
            result.savefig(tmpfile.name, bbox_inches='tight')

            fig = plt.figure()
            img = plt.imread(tmpfile.name)
            plt.imshow(img)
            plt.tight_layout()
            plt.axis('off')

            self.specCanvas = FigureCanvas(fig)
            self.specTab.layout().insertWidget(0, self.specCanvas)  # Insert the canvas at the top
            self.specCanvas.draw()
            self.plotButton.setEnabled(True)

    def handleSpecificationCurveError(self, error):
        QMessageBox.warning(self, "Specification Curve", f"An error occurred while plotting the specification curve: {error}")
        self.plotButton.setEnabled(True)

"""
Run the application
"""
def run():
    app = QApplication(sys.argv)
    QLocale.setDefault(QLocale(QLocale.Language.English, QLocale.Country.UnitedStates))

    # Set global stylesheet for tooltips
    QApplication.instance().setStyleSheet("""
        QToolTip {
            background-color: #f3f1f5;
            border: 1px solid black;
        }
    """)
    ex = App()
    ex.setStyleSheet(qdarkstyle.load_stylesheet_pyqt6())

    default_width = ex.width()
    default_height = ex.height()
    new_width = int(default_width * 2.0)
    new_height = int(default_height * 1.6)
    ex.resize(new_width, new_height)

    ex.show()

    try:
        sys.exit(app.exec())
    except SystemExit as e:
        print(f"GUI closed with status {e}")

if __name__ == '__main__':
    run()
