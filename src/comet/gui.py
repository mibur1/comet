import re
import sys
import copy
import json
import pickle
import inspect
import subprocess
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.io import loadmat, savemat
from dataclasses import dataclass, field
from importlib import resources as pkg_resources
from typing import Any, Dict, get_type_hints, get_origin, Literal

# Plotting imports
from matplotlib.image import imread
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec

# Qt imports
import qdarkstyle
from PyQt6.QtCore import Qt, QPoint, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QEnterEvent, QFontMetrics
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, \
    QSlider, QToolTip, QWidget, QLabel, QFileDialog, QComboBox, QLineEdit, QSizePolicy, \
    QSpacerItem, QCheckBox, QTabWidget, QSpinBox, QDoubleSpinBox, QTextEdit, QMessageBox

# Comet imports and state-based dFC methods from pydfc
from . import cifti, methods, graph
import pydfc

class Worker(QObject):
    # Worker class for calculations (runs in a separate thread)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    result = pyqtSignal(object)

    def __init__(self, calculationFunc, parameters):
        super().__init__()
        self.calculationFunc = calculationFunc
        self.parameters = parameters

    def run(self):
        try:
            result = self.calculationFunc(self.parameters)
            self.result.emit(result)  # Emit results
        except Exception as e:
            self.error.emit(str(e))  # Emit errors
        finally:
            self.finished.emit()  # Notify completion

class InfoButton(QPushButton):
    # Info button class
    def __init__(self, info_text, parent=None):
        super().__init__("i", parent)
        self.info_text = info_text
        self.setStyleSheet("QPushButton { border: 1px solid black;}")
        self.setFixedSize(20, 20)

    def enterEvent(self, event: QEnterEvent):
        tooltip_pos = self.mapToGlobal(QPoint(self.width(), 0)) # Tooltip position can be adjusted here
        QToolTip.showText(tooltip_pos, self.info_text)
        super().enterEvent(event)

@dataclass
class Data:
    # File variables
    file_name:     str        = field(default=None)         # data file name
    file_data:     np.ndarray = field(default=None)         # input time series data  

    # DFC variables
    dfc_instance:  Any        = field(default=None)         # instance of the dFC class
    dfc_name:      str        = field(default=None)         # method class name
    dfc_params:    Dict       = field(default_factory=dict) # input parameters
    dfc_data:      np.ndarray = field(default=None)         # dfc data
    dfc_states:    Dict       = field(default_factory=dict) # dfc states
    dfc_state_tc:  np.ndarray = field(default=None)         # dfc state time course
    dfc_edge_ts:   np.ndarray = field(default=None)         # dfc edge time series

    # Graph variables
    graph_file:    str        = field(default=None)         # graph file name
    graph_raw:     np.ndarray = field(default=None)         # raw input data for graph (dFC matrix)
    graph_data:    np.ndarray = field(default=None)         # working data for graph (while processing)
    graph_out:     Any = field(default=None)                # output graph measure data

    # Multiverse variables
    forking_paths: Dict      = field(default_factory=dict) # Decision points for multiverse analysis
    invalid_paths: list      = field(default_factory=list) # Invalid paths for multiverse analysis

    # Misc variables
    cifti_data:    np.ndarray = field(default=None)         # input cifti data (for .dtseries files)
    roi_names:     np.ndarray = field(default=None)         # input roi data (for .tsv files)
    mv_containers: list       = field(default_factory=list) # decision containers for multiverse analysis

    def clear_dfc_data(self):
        self.dfc_params   = {}
        self.dfc_data     = None
        self.dfc_states   = {}
        self.dfc_state_tc = None
        self.dfc_edge_ts  = None

class DataStorage:
    # Database class for storing calculated data
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

class App(QMainWindow):
    """
    Initialization and GUI setup functions
    """
    def __init__(self, init_dfc_data=None, init_dfc_instance=None):
        super().__init__()
        self.title = 'Comet Toolbox'
        self.init_flag = True

        self.data = Data()
        self.data_storage = DataStorage()

        self.currentSliderValue = 0
        self.currentTabIndex = 0
        self.graphStepCounter = 1

        # Parameter names for the GUI
        self.param_names = {
            "self":                 "self", 
            "time_series":          "Time series",
            "windowsize":           "Window size",
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
            "fmax":                 "Maximum freqency",
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
            "n_subj_clusters":      "Number of subjects",
            "normalization":        "Normalization",
            "clstr_distance":       "Distance measure",
            "subject":              "Subject",
            "clstr_base_measure":   "Base measure",
            "iterations":            "Iterations",
            "sw_method":            "Sliding window",
            "dhmm_obs_state_ratio": "State ratio",
            "vlim":                 "Color axis limit",
            "parcellation":         "Parcellation"

        }
        self.reverse_param_names = {v: k for k, v in self.param_names.items()}
        
        # All availble connectivity methods
        self.connectivityMethods = {
            'SlidingWindow':                'CONT Sliding Window',
            'Jackknife':                    'CONT Jackknife Correlation',
            'FlexibleLeastSquares':         'CONT Flexible Least Squares',
            'SpatialDistance':              'CONT Spatial Distance', 
            'TemporalDerivatives':          'CONT Multiplication of Temporal Derivatives',
            'DCC':                          'CONT Dynamic Conditional Correlation',
            'PhaseSynchrony':               'CONT Phase Synchronization',
            'LeiDA':                        'CONT Leading Eigenvector Dynamics',
            'WaveletCoherence':             'CONT Wavelet Coherence',
            'Edge_centric_connectivity':    'CONT Edge-centric Connectivity',
            
            'Sliding_Window_Clustr':        'STATE Sliding Window Clustering',
            'Cap':                          'STATE Co-activation patterns',
            'HMM_Disc':                     'STATE Discrete Hidden Markov Model',
            'HM_Cont':                      'STATE Continuous Hidden Markov Model',
            'Windowless':                   'STATE Windowless',

            'Static_Pearson':               'STATIC Pearson Correlation',
            'Static_Partial':               'STATIC Partial Correlation',
            'Static_Mutual_Info':           'STATIC Mutual Information'
        }

        self.reverse_connectivityMethods = {v: k for k, v in self.connectivityMethods.items()}
        
        
        # All availble graph analysis functions
        self.graphOptions = {
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
            "transitivity":                 "BCT Transitivity",
        }
        self.reverse_graphOptions = {v: k for k, v in self.graphOptions.items()}

        # Init the UI
        self.initUI()

        if init_dfc_data is not None:
            self.initFromData(init_dfc_data, init_dfc_instance)

    def initUI(self):
        self.setWindowTitle(self.title)
        
        # Top-level layout which contains connectivity, graph, and multiverse tabs
        topLayout = QVBoxLayout()
        self.topTabWidget = QTabWidget()
        topLayout.addWidget(self.topTabWidget)

        # Setup the individual tabs
        self.connectivityTab()
        self.graphTab()
        self.multiverseTab()

        # Set main window layout to the top-level layout
        centralWidget = QWidget()
        centralWidget.setLayout(topLayout)
        self.setCentralWidget(centralWidget)

    def initFromData(self, init_dfc_data=None, init_dfc_instance=None):
        # Make sure both the dFC data and the method object are provided
        assert self.data.dfc_instance is not None, "Please provide the method object corresponding to your dFC data as the second argument to the GUI."
        
        # Init the data structures
        self.data.dfc_data = init_dfc_data
        self.data.dfc_instance = init_dfc_instance
        self.data.file_name = "loaded from script"
        self.data.dfc_name = init_dfc_instance.name
        self.data.time_series = np.array([]) # for potential saving to .mat

        # In case the method returns multiple values. The first one is always the NxNxT dfc matrix
        if isinstance(init_dfc_data, tuple):
            self.data.dfc_data = init_dfc_data[0]
            self.data.dfc_state_tc = None
            self.data.dfc_edge_ts = init_dfc_data[1][0] if isinstance(init_dfc_data[1], tuple) else None

        # Result is DFC object (pydfc methods)
        elif isinstance(init_dfc_data, pydfc.dfc.DFC):
            self.data.dfc_data = np.transpose(init_dfc_data.get_dFC_mat(), (1, 2, 0))
            self.data.dfc_states = init_dfc_data.FCSs_
            self.data.dfc_state_tc = init_dfc_data.state_TC()
            self.data.dfc_edge_ts = None
        
        # Only a single matrix is returned (most cases)
        else:
            self.data.dfc_data = init_dfc_data
            self.data.dfc_state_tc = None
            self.data.dfc_edge_ts = None
        
        # Add to data storage
        self.data_storage.add_data(self.data)
        
        # Disable the GUI elements
        self.methodComboBox.setEnabled(False)
        self.calculateButton.setEnabled(False)
        self.clearMemoryButton.setEnabled(False)
        self.keepInMemoryCheckbox.setEnabled(False)

        # Set labels
        self.fileNameLabel.setText(f"Loaded dFC from script")
        self.calculatingLabel.setText(f"Loaded dFC from script")
        self.methodComboBox.setCurrentText(self.data.dfc_name)

        # Disable checkboxes
        self.continuousCheckBox.setChecked(True)
        self.stateBasedCheckBox.setChecked(True)
        self.staticCheckBox.setChecked(True)
        self.continuousCheckBox.setEnabled(False)
        self.stateBasedCheckBox.setEnabled(False)
        self.staticCheckBox.setEnabled(False)
        
        # Set plots
        self.plotConnectivity()
        self.plotDistribution()
        self.plotTimeSeries()

        # Set the slider elements
        total_length = self.data.dfc_data.shape[2] if len(self.data.dfc_data.shape) == 3 else 0
        position_text = f"t = {self.currentSliderValue} / {total_length-1}" if len(self.data.dfc_data.shape) == 3 else " static "
        self.positionLabel.setText(position_text)
        self.slider.setValue(self.slider.value())
        self.slider.show()

        # This first gets the parameters from the signature of the method and then fill them with the curent values
        init_signature = inspect.signature(init_dfc_instance.__init__)
        self.data.dfc_params = {}

        for param_name in init_signature.parameters:
            self.data.dfc_params[param_name] = getattr(init_dfc_instance, param_name, None)
        
        self.setParameters(disable=True)

    def connectivityTab(self):
        connectivityTab = QWidget()
        connectivityLayout = QHBoxLayout()
        connectivityTab.setLayout(connectivityLayout)

        ###############################
        #  Left section for settings  #
        ###############################
        self.leftLayout = QVBoxLayout()

        # Create button and label for file loading
        loadLayout = QHBoxLayout()
        self.fileButton = QPushButton('Load time series')
        self.bidsButton = QPushButton('Load BIDS dataset')
        self.fileNameLabel = QLabel('No file loaded yet')
        
        loadLayout.addWidget(self.fileButton)
        loadLayout.addWidget(self.bidsButton)
        self.leftLayout.addLayout(loadLayout)
        self.leftLayout.addWidget(self.fileNameLabel)
        self.fileButton.clicked.connect(self.loadConnectivityFile)

        # Create a checkbox for reshaping the data
        self.transposeCheckbox = QCheckBox("Transpose data (time has to be the first dimension)")
        self.leftLayout.addWidget(self.transposeCheckbox)
        self.transposeCheckbox.setEnabled(False)

        # Connect the checkbox to a method
        self.transposeCheckbox.stateChanged.connect(self.onTransposeChecked)

        # Add spacer for an empty line
        self.leftLayout.addItem(QSpacerItem(0, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))

        # Method label and combobox
        self.methodLabel = QLabel("Dynamic functional connectivity method:")
        
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

        # Connect the stateChanged signal of checkboxes to the slot
        self.continuousCheckBox.stateChanged.connect(self.updateMethodComboBox)
        self.stateBasedCheckBox.stateChanged.connect(self.updateMethodComboBox)
        self.staticCheckBox.stateChanged.connect(self.updateMethodComboBox)
        
        self.methodComboBox = QComboBox()
        self.leftLayout.addWidget(self.methodLabel)
        self.leftLayout.addLayout(checkboxLayout)
        self.leftLayout.addWidget(self.methodComboBox)

        # Get all the dFC methods and names
        self.class_info = {
            obj.name: name  # Map human-readable name to class name
            for name, obj in inspect.getmembers(methods)
            if inspect.isclass(obj) and obj.__module__ == methods.__name__ and name != "ConnectivityMethod"
        }

        # Create a layout for dynamic textboxes
        self.parameterLayout = QVBoxLayout()

        # Create a container widget for the parameter layout
        self.parameterContainer = QWidget()  # Use an instance attribute to access it later
        self.parameterContainer.setLayout(self.parameterLayout)
        self.parameterContainer.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        # Add the container widget to the left layout directly below the combobox
        self.leftLayout.addWidget(self.parameterContainer)
        
        # Initial population of the combobox, this does the entire initialization
        self.updateMethodComboBox()

        # Add parameter textbox for time_series
        self.time_series_textbox = QLineEdit()
        self.time_series_textbox.setReadOnly(True) # read only as based on the loaded file

        # Set up the atlas combobox
        self.atlasComboBox = QComboBox()
        self.atlasComboBox.addItems(["Glasser MMP", "Schaefer Kong 200", "Schaefer Tian 254"])
        self.atlasComboBox.currentIndexChanged.connect(self.onAtlasSelected)

        # Add a stretch after the parameter layout container
        self.leftLayout.addStretch()

        # Calculate connectivity and save button
        buttonsLayout = QHBoxLayout()

        # Calculate connectivity button
        self.calculateButton = QPushButton('Calculate Connectivity')
        self.calculateButton.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        buttonsLayout.addWidget(self.calculateButton, 2)  # 2/3 of the space
        self.calculateButton.clicked.connect(self.onCalculateButton)

        # Create the "Save" button
        self.saveButton = QPushButton('Save')
        self.saveButton.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        buttonsLayout.addWidget(self.saveButton, 1)  # 1/3 of the space
        self.saveButton.clicked.connect(self.saveConnectivityFile)

        # Add the buttons layout to the left layout
        self.leftLayout.addLayout(buttonsLayout)

        # Memory buttons
        self.keepInMemoryCheckbox = QCheckBox("Keep in memory")
        self.keepInMemoryCheckbox.stateChanged.connect(self.onKeepInMemoryChecked)
        self.clearMemoryButton = QPushButton("Clear Memory")
        self.clearMemoryButton.clicked.connect(self.onClearMemory)

        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(self.keepInMemoryCheckbox)
        buttonLayout.addWidget(self.clearMemoryButton)

        # Assuming you have a QVBoxLayout named 'leftLayout'
        self.leftLayout.addLayout(buttonLayout)

        # Calculation info textbox
        self.calculatingLabel = QLabel('No data calculated yet')
        self.leftLayout.addWidget(self.calculatingLabel)
        
        ################################
        #  Right section for plotting  #
        ################################
        rightLayout = QVBoxLayout()
        self.tabWidget = QTabWidget()
        
        # Tab 1: Imshow plot
        imshowTab = QWidget()
        imshowLayout = QVBoxLayout()
        imshowTab.setLayout(imshowLayout)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.figure.patch.set_facecolor('#E0E0E0')
        imshowLayout.addWidget(self.canvas)
        self.tabWidget.addTab(imshowTab, "Connectivity")

        # Tab 2: Time series/course plot
        timeSeriesTab = QWidget()
        timeSeriesLayout = QVBoxLayout()
        timeSeriesTab.setLayout(timeSeriesLayout)

        self.timeSeriesFigure = Figure()
        self.timeSeriesCanvas = FigureCanvas(self.timeSeriesFigure)
        self.timeSeriesFigure.patch.set_facecolor('#E0E0E0')
        timeSeriesLayout.addWidget(self.timeSeriesCanvas)
        self.tabWidget.addTab(timeSeriesTab, "Time course")

        rightLayout.addWidget(self.tabWidget)

        # Tab 3: Distribution plot
        distributionTab = QWidget()
        distributionLayout = QVBoxLayout()
        distributionTab.setLayout(distributionLayout)

        self.distributionFigure = Figure()
        self.distributionCanvas = FigureCanvas(self.distributionFigure)
        self.distributionFigure.patch.set_facecolor('#E0E0E0')
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
        navButtonLayout = QHBoxLayout()
        navButtonLayout.addStretch(1)  # Spacer to the left of the buttons

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

        navButtonLayout.addStretch(1) # Spacer to the right of the buttons
        rightLayout.addLayout(navButtonLayout)

        # UI elements for dFC time series plotting
        self.rowSelector = QSpinBox()
        self.rowSelector.setMaximum(0)
        self.rowSelector.valueChanged.connect(self.plotTimeSeries)

        self.colSelector = QSpinBox()
        self.colSelector.setMaximum(0)
        self.colSelector.valueChanged.connect(self.plotTimeSeries)

        self.timeSeriesSelectorLayout = QHBoxLayout()
        self.timeSeriesSelectorLayout.addWidget(QLabel("Brain region 1 (row):"))
        self.timeSeriesSelectorLayout.addWidget(self.rowSelector)
        self.timeSeriesSelectorLayout.addWidget(QLabel("Brain region 2 (column):"))
        self.timeSeriesSelectorLayout.addWidget(self.colSelector)

        timeSeriesLayout.addLayout(self.timeSeriesSelectorLayout)

        # Set checkboxes to default values
        self.continuousCheckBox.setChecked(True)
        self.stateBasedCheckBox.setChecked(True)
        self.staticCheckBox.setChecked(True)

        #####################
        #  Combine layouts  #
        #####################
        connectivityLayout.addLayout(self.leftLayout, 1)
        connectivityLayout.addLayout(rightLayout, 2)
        self.topTabWidget.addTab(connectivityTab, "Connectivity Analysis")

    def graphTab(self):
        graphTab = QWidget()
        graphLayout = QVBoxLayout()  # Main layout for the tab
        graphTab.setLayout(graphLayout)
        
        ###############################
        #  Left section for settings  #
        ###############################
        leftLayout = QVBoxLayout()

        # Calculate connectivity and save button
        buttonsLayout = QHBoxLayout()

        self.loadGraphFileButton = QPushButton('Load File')
        self.loadGraphFileButton.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        buttonsLayout.addWidget(self.loadGraphFileButton, 1)
        self.loadGraphFileButton.clicked.connect(self.loadGraphFile)

        self.takeCurrentButton = QPushButton('From current dFC')
        self.takeCurrentButton.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        buttonsLayout.addWidget(self.takeCurrentButton, 1)
        self.takeCurrentButton.clicked.connect(self.takeCurrentData)
        
        self.graphFileNameLabel = QLabel('No file loaded yet')

        leftLayout.addLayout(buttonsLayout)
        leftLayout.addWidget(self.graphFileNameLabel)
        leftLayout.addItem(QSpacerItem(0, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))

        # Graph analysis options
        graphTabLabel = QLabel("Graph analysis options:")
        leftLayout.addWidget(graphTabLabel)

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
        self.preprocessingCheckBox.setChecked(False)
        self.graphCheckBox.setChecked(False)
        self.BCTCheckBox.setChecked(True)
        leftLayout.addLayout(checkboxLayout)
            
        # Create the combo box for selecting the graph analysis type
        self.graphAnalysisComboBox = QComboBox()
        leftLayout.addWidget(self.graphAnalysisComboBox)

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
        leftLayout.addWidget(parameterContainer)

        # Stretch empty space
        leftLayout.addStretch()

        # Create a layout for the buttons
        buttonsLayout = QHBoxLayout()

        # Create the "Add option" button
        addOptionButton = QPushButton('Add current option')
        buttonsLayout.addWidget(addOptionButton, 1) # The second parameter is the stretch factor
        addOptionButton.clicked.connect(self.onAddGraphOption)

        # Create the "Save" button
        saveButton = QPushButton('Clear options')
        buttonsLayout.addWidget(saveButton, 1)
        saveButton.clicked.connect(self.onClearGraphOptions)

        # Add the buttons layout to the left layout
        leftLayout.addLayout(buttonsLayout)

        self.optionsTextbox = QTextEdit()
        self.optionsTextbox.setReadOnly(True) # Make the textbox read-only
        leftLayout.addWidget(self.optionsTextbox)

        # "Save" button
        saveButton = QPushButton('Save')
        leftLayout.addWidget(saveButton)
        saveButton.clicked.connect(self.saveGraphFile)

        ################################
        #  Right section for plotting  #
        ################################
        rightLayout = QVBoxLayout()
        
        # Different plotting tabs
        graphTabWidget = QTabWidget()

        # Tab 1: Adjacency matrix plot
        matrixTab = QWidget()
        matrixLayout = QVBoxLayout()
        matrixTab.setLayout(matrixLayout)

        self.matrixFigure = Figure()
        self.matrixCanvas = FigureCanvas(self.matrixFigure)
        self.matrixFigure.patch.set_facecolor('#E0E0E0')
        matrixLayout.addWidget(self.matrixCanvas)
        graphTabWidget.addTab(matrixTab, "Adjacency Matrix")
       
        # Draw default plot (logo)
        self.plotLogo(self.matrixFigure)
        self.matrixCanvas.draw()

        # Tab 2: Graph measures plot
        measureTab = QWidget()
        measureLayout = QVBoxLayout()
        measureTab.setLayout(measureLayout)

        # Widget for the plot
        plotWidget = QWidget()  # Create a widget to hold the plot
        plotLayout = QVBoxLayout()  # Use QVBoxLayout for the plot widget
        plotWidget.setLayout(plotLayout)

        # Add the graph canvas to the plot layout
        self.graphFigure = Figure()
        self.graphCanvas = FigureCanvas(self.graphFigure)
        self.graphFigure.patch.set_facecolor('#E0E0E0')
        plotLayout.addWidget(self.graphCanvas)

        # Widget for the textbox
        self.graphTextbox = QTextEdit()
        self.graphTextbox.setReadOnly(True)

        # Add the plot widget and the textbox to the measure layout
        measureLayout.addWidget(plotWidget, 2)
        measureLayout.addWidget(self.graphTextbox, 1)
        graphTabWidget.addTab(measureTab, "Graph Measure")

        # Draw default plot (e.g., logo)
        self.plotLogo(self.graphFigure)
        self.graphCanvas.draw()

        # Add widgets to the right layout
        rightLayout.addWidget(graphTabWidget)

        #####################
        #  Combine layouts  #
        #####################
        mainLayout = QHBoxLayout()
        mainLayout.addLayout(leftLayout, 1)
        mainLayout.addLayout(rightLayout, 2)
        graphLayout.addLayout(mainLayout)

        # Add the tab to the top level tab widget
        self.topTabWidget.addTab(graphTab, "Graph Analysis")

    def multiverseTab(self):
        multiverseTab = QWidget()
        multiverseLayout = QVBoxLayout()  # Main layout for the tab
        multiverseTab.setLayout(multiverseLayout)
        
        ###############################
        #  Left section for settings  #
        ###############################
        leftLayout = QVBoxLayout()

        loadLabel = QLabel('Create multiverse analysis template script:')
        leftLayout.addWidget(loadLabel)

        # Creating a first decision container
        decisionWidget = self.addDecisionContainer()
        leftLayout.addWidget(decisionWidget)

        # Horizontal layout for add/collapse buttons
        buttonLayout = QHBoxLayout()

        newDecisionButton = QPushButton("\u2795")
        newDecisionButton.clicked.connect(lambda: self.addNewDecision(leftLayout, buttonLayout))
        buttonLayout.addWidget(newDecisionButton, 5)
        buttonLayout.addStretch(21)

        leftLayout.addLayout(buttonLayout)

        leftLayout.addStretch()


        #########################
        #  Bottom left section  #
        #########################

        # Perform multiverse analysis
        loadLabel = QLabel('Perform multiverse analysis:')
        leftLayout.addWidget(loadLabel)

        loadLayout = QHBoxLayout()
        loadScriptButton = QPushButton('Load Script')
        loadLayout.addWidget(loadScriptButton, 1)

        # Textbox to display the loaded script path
        self.loadedScriptDisplay = QLineEdit()
        self.loadedScriptDisplay.setPlaceholderText("No script loaded yet")
        self.loadedScriptDisplay.setReadOnly(True)
        loadLayout.addWidget(self.loadedScriptDisplay, 3)

        # Add the horizontal layout to the main left layout
        leftLayout.addLayout(loadLayout)

        # Connect the button to the method that handles file loading
        loadScriptButton.clicked.connect(self.loadScript)

        runButton = QPushButton('Run multiverse analysis')
        leftLayout.addWidget(runButton)
        runButton.clicked.connect(self.runScript)

        ################################
        #  Right section for plotting  #
        ################################
        rightLayout = QVBoxLayout()

        # Creating a tab widget for different purposes
        multiverseTabWidget = QTabWidget()

        # Tab 1: Template textbox for script display
        templateTab = QWidget()
        templateLayout = QVBoxLayout()
        templateTab.setLayout(templateLayout)

        self.scriptDisplay = QTextEdit()
        self.scriptDisplay.setReadOnly(True)
        templateLayout.addWidget(self.scriptDisplay)

        # Create and add the save button
        saveScriptButton = QPushButton('Save template script')
        saveScriptButton.clicked.connect(self.saveScript)
        templateLayout.addWidget(saveScriptButton)

        # Add the complete tab to the multiverseTabWidget
        multiverseTabWidget.addTab(templateTab, "Template")
        self.generateScript(init_template=True) # Generate the template script

        # Tab 2: Plot for visualizations
        plotTab = QWidget()
        plotTab.setLayout(QVBoxLayout())
        self.multiverseFigure = Figure()
        self.multiverseCanvas = FigureCanvas(self.multiverseFigure)
        self.multiverseFigure.patch.set_facecolor('#E0E0E0')
        plotTab.layout().addWidget(self.multiverseCanvas)
        multiverseTabWidget.addTab(plotTab, "Multiverse Plot")

        # Draw default plot (logo) on the canvas
        self.plotLogo(self.multiverseFigure)
        self.multiverseCanvas.draw()

        # Add the tab widget to the right layout
        rightLayout.addWidget(multiverseTabWidget)

        #####################
        #  Combine layouts  #
        #####################
        mainLayout = QHBoxLayout()
        mainLayout.addLayout(leftLayout, 1)
        mainLayout.addLayout(rightLayout, 1)
        multiverseLayout.addLayout(mainLayout)

        self.topTabWidget.addTab(multiverseTab, "Multiverse Analysis")

    """
    Multiverse functions
    """
    # Adds a combined container widget for all things related to creating decisions and options
    def addDecisionContainer(self):
        decisionWidget = QWidget()  # Widget that holds the entire container
        mainLayout = QVBoxLayout(decisionWidget)  # Vertical layout: contains decision layout and function layout
        mainLayout.setContentsMargins(0, 0, 0, 0) # Remove space to remove excessive space between widgets

        functionLayout = QHBoxLayout()  # Controls layout for functions and properties

        # Create the dropdown menu
        categoryComboBox = QComboBox()
        categoryComboBox.addItems(["General", "FC", "Graph", "Other"])

        # Decision name input field
        decisionNameInput = QLineEdit()
        decisionNameInput.setPlaceholderText("Decision name")

        # Decision options input field
        decisionOptionsInput = QLineEdit()
        decisionOptionsInput.setPlaceholderText("Options (comma-separated)")

        # Collapse button to hide/show the function and parameter widgets
        collapseButton = QPushButton(" \u25B2 ")
        collapseButton.setObjectName("collapseButton")
        collapseButton.hide()

        # Add option button to add a new option to the decision
        addOptionButton = QPushButton(' \u25B6 ')
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
        self.data.mv_containers.append(decisionWidget)

        return decisionWidget

    # Handles if the type of the decision is changed
    def onCategoryComboBoxChanged(self, categoryComboBox, functionComboBox, parameterContainer, addOptionButton, collapseButton, decisionNameInput, decisionOptionsInput):
        selected_category = categoryComboBox.currentText()
        self.clearLayout(parameterContainer.layout())

        functionComboBox.clear()
        decisionNameInput.clear()
        decisionOptionsInput.clear()

        functionComboBox.hide()
        parameterContainer.hide()
        addOptionButton.hide()
        collapseButton.hide()

        """
        # Re-populate fields based on the last entry in forking_paths for the selected category
        last_key, last_entry = None, None
        if selected_category in ['FC', 'Graph']:
            # Filter entries by prefix in "Option" within args dict
            prefix = 'PREP' if selected_category == 'Graph' else 'CONT'
            for key, entries in reversed(list(self.data.forking_paths.items())):
                if isinstance(entries, list) and any(isinstance(d, dict) and 'args' in d and 'Option' in d['args'] and prefix in d['args']['Option'] for d in entries):
                    last_key, last_entry = key, entries
                    break
        elif selected_category == 'General':
            # General is simply the last key in the dictionary where value is a list of ints
            for key, value in reversed(list(self.data.forking_paths.items())):
                if isinstance(value, list) and all(isinstance(x, int) for x in value):
                    last_key, last_entry = key, value
                    break

        # Set the inputs if there was an entry found
        if last_key:
            decisionNameInput.setText(last_key)
            if isinstance(last_entry, list) and all(isinstance(x, dict) for x in last_entry):
                decisionOptionsInput.setText(', '.join(d['name'] for d in last_entry if 'name' in d))
            elif isinstance(last_entry, list):
                decisionOptionsInput.setText(', '.join(map(str, last_entry)))"""

        # Sets up the layout and input fields for the new category/options
        decisionOptionsInput.setPlaceholderText("Enter options, comma-separated" if selected_category == "General" else "Define options below")
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
    
    # Gets a dict with the current function parameters
    def getFunctionParameters(self, parameterContainer):
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

    # Creates and updates all the parameter widgets based on the selected function
    def updateFunctionParameters(self, functionComboBox, parameterContainer):
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
            dfc_class_ = getattr(methods, functionComboBox.currentData())
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
                print(name, param_type)
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
                    param_widget.setPlaceholderText("Input data (name of the variable in the script)")
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

                temp_widgets[name] = (param_label, param_widget)
                param_layout.addWidget(param_widget)
                parameterContainer.layout().addLayout(param_layout)

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

    # "Other" category for custom functions
    def otherOptionCategory(self, parameterContainer):
        # Clear the parameter container
        self.clearLayout(parameterContainer.layout())

        # Add a single QLineEdit for the user to input the option
        font_metrics = QFontMetrics(self.font())
        label_width = font_metrics.boundingRect(f"Parameters:").width()

        option_layout = QHBoxLayout()
        option_label = QLabel("Function:")
        option_label.setFixedWidth(label_width + 20)
        option_edit = QLineEdit()
        option_edit.setPlaceholderText("Name of the function (e.g. np.mean)")
        option_layout.addWidget(option_label)
        option_layout.addWidget(option_edit)
        parameterContainer.layout().addLayout(option_layout)

        param_layout = QHBoxLayout()
        param_label = QLabel("Parameters:")
        param_label.setFixedWidth(label_width + 20)
        param_edit = QLineEdit()
        param_edit.setPlaceholderText("Function parameters as dict (e.g. {'axis': 0})")
        param_layout.addWidget(param_label)
        param_layout.addWidget(param_edit)
        parameterContainer.layout().addLayout(param_layout)

        return  

    # Adds a new decision widget to the layout
    def addNewDecision(self, layout, buttonLayout):
        # Collapse all existing parameter containers before adding a new one
        for container in self.data.mv_containers:
            parameterContainer = container.findChild(QWidget, "parameterContainer")
            collapseButton = container.findChild(QPushButton, "collapseButton")
            if parameterContainer and collapseButton and parameterContainer.isVisible():
                self.collapseOption(collapseButton, parameterContainer)

        # Add new decision container
        newDecisionWidget = self.addDecisionContainer()
        buttonLayoutIndex = layout.indexOf(buttonLayout)
        layout.insertWidget(buttonLayoutIndex, newDecisionWidget)

        return

    # Add option to a decision
    def addOption(self, functionComboBox, parameterContainer, nameInputField, optionsInputField):
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
            module_prefix = "comet.methods"
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
    
    # Collapse the option layout
    def collapseOption(self, collapseButton, parameterContainer):
        if collapseButton.text() == " \u25B2 ":
            parameterContainer.hide()
            collapseButton.setText(" \u25BC ")
            return
            
        if collapseButton.text() == " \u25BC ":
            parameterContainer.show()
            collapseButton.setText(" \u25B2 ")
            return
        
        return

    # Adds decision to the script
    def includeDecision(self, categoryComboBox, nameInput, optionsInput):
        category = categoryComboBox.currentText()
        name = nameInput.text().strip()

        if category == "General":
            options = [self.setDtypeForOption(option.strip()) for option in optionsInput.text().split(',') if option.strip()]
            self.data.forking_paths[name] = options
        else: 
            options = self.data.forking_paths[name]

        if name and options:
            self.generateScript()
        else:
            QMessageBox.warning(self, "Input Error", "Please ensure a name and at least one option are provided.")
        return
    
    # Handles data conversion based on the input
    def setDtypeForOption(self, option):
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

    # Removes one option with each click and finally the entire decision
    def removeDecision(self, decisionNameInput, decisionWidget, optionsInputField):
        key = decisionNameInput.text().strip()

        # No key means the decision widget is empty, so clear and delete everything
        if key == "":
            if decisionWidget.layout():
                self.clearLayout(decisionWidget.layout())
                decisionWidget.deleteLater()
                optionsInputField.clear()
                self.data.mv_containers.remove(decisionWidget)
                self.generateScript()
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
                self.data.mv_containers.remove(decisionWidget)

        self.generateScript()
        return

    # Clears the entire decisionWidget layout
    def clearLayout(self, layout):
        # Recursively delete all items in a layout. This method handles both widgets and sub-layouts.
        while layout.count():
            item = layout.takeAt(0)  # Take the first item in the layout
            if item.widget():
                if item.widget().objectName() != "functionComboBox":  # Ensure it's not the functionComboBox
                    item.widget().deleteLater()  # Delete the widget if the item is a widget
            elif item.layout():
                self.clearLayout(item.layout())  # Recursively clear if the item is a layout
                item.layout().deleteLater()  # Delete the sub-layout after clearing it
            elif item.spacerItem():
                # No need to delete spacer items as Qt does it automatically
                pass
    
    # Generates the template script
    def generateScript(self, init_template=False):
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
                "multiverse = Multiverse(name=\"multiverse_example\")\n"
                "multiverse.create(analysis_template, forking_paths)\n"
                "multiverse.summary()\n"
                "#multiverse.run()\n"
            )
        
        else:
            script_content = (
                "from comet.multiverse import Multiverse\n"
                "\n"
                "forking_paths = {\n"
            )
            for name, options in self.data.forking_paths.items():
                if isinstance(options, list) and all(isinstance(item, dict) for item in options):
                    # Format as JSON only if it's a list of dictionaries
                    formatted_options = json.dumps(options, indent=4)
                    formatted_options = formatted_options.replace('true', 'True').replace('false', 'False').replace('null', 'None')
                    script_content += f'    "{name}": {formatted_options},\n'
                else:
                    # Simple list of primitives or a single dictionary
                    script_content += f'    "{name}": {options},\n'

            script_content += (
                "}\n\n"
                "def analysis_template():\n"
                "    # The following forking paths are available for multiverse analysis:\n"
            )

            for name in self.data.forking_paths:
                script_content += f"    {{{{{name}}}}}\n"  # placeholder variables are in double curly braces {{variable}}

            script_content += (
                "\nmultiverse = Multiverse(name=\"example_multiverse\")\n"
                "multiverse.create(analysis_template, forking_paths)\n"
                "multiverse.summary()\n"
                "#multiverse.run()\n"
            )

        self.scriptDisplay.setText(script_content)
        self.scriptDisplay.setReadOnly(True)

    # Loads a multiverse script
    def loadScript(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Load Script", "", "Python Files (*.py);;All Files (*)")
        if fileName:
            self.loadedScriptDisplay.setText(f"Loaded: {fileName}")
            self.loadedScriptPath = fileName
            
            # Read the content of the file
            try:
                with open(fileName, 'r', encoding='utf-8') as file:
                    scriptContent = file.read()
                    self.scriptDisplay.setText(scriptContent)
                    self.scriptDisplay.setReadOnly(False)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to read the file: {str(e)}")

    # Saves the current template script
    def saveScript(self):
        script_text = self.scriptDisplay.toPlainText()
        fileName, _ = QFileDialog.getSaveFileName(self, "Save Script", "", "Python Files (*.py);;All Files (*)")

        if fileName:
            # Ensure the file has the correct extension
            if not fileName.endswith('.py'):
                fileName += '.py'
            
            with open(fileName, 'w') as file:
                file.write(script_text)

    # Runs the script/multiverse analysis
    def runScript(self):
        if hasattr(self, 'loadedScriptPath') and self.loadedScriptPath:
            try:
                # Run the script
                subprocess.run(['python', self.loadedScriptPath], check=True)
                QMessageBox.information(self, "Multiverse Analysis", "Multiverse ran successfully!")
            except subprocess.CalledProcessError:
                QMessageBox.warning(self, "Multiverse Analysis", "Multiverse failed!")
            except Exception as e:
                QMessageBox.warning(self, "Multiverse Analysis", f"Multiverse failed!\n{str(e)}")
        else:
            QMessageBox.warning(self, "Multiverse Analysis", "No script was loaded to run.")

    """
    I/O and data related functions
    """
    # dFC tab
    def loadConnectivityFile(self):
        fileFilter = "All Supported Files (*.mat *.txt *.npy *.pkl *.tsv *.dtseries.nii *.ptseries.nii);;MAT files (*.mat);;Text files (*.txt);;NumPy files (*.npy);;Pickle files (*.pkl);;TSV files (*.tsv);;CIFTI files (*.dtseries.nii *.ptseries.nii)"
        file_path, _ = QFileDialog.getOpenFileName(self, "Load File", "", fileFilter)
        file_name = file_path.split('/')[-1]
        self.data.file_name = file_name
        self.getParameters() # Get current UI parameters

        if not file_path:
            return  # Early exit if no file is selected

        if file_path.endswith('.mat'):
            data_dict = loadmat(file_path)
            self.data.file_data = data_dict[list(data_dict.keys())[-1]] # always get data for the last key
        
        elif file_path.endswith('.txt'):
            self.data.file_data = np.loadtxt(file_path)
        
        elif file_path.endswith('.npy'):
            self.data.file_data = np.load(file_path)
        
        elif file_path.endswith('.pkl'):
            with open(file_path, 'rb') as f:
                self.data.file_data = pickle.load(f)
        
        elif file_path.endswith(".tsv"):
            data = pd.read_csv(file_path, sep='\t', header=None, na_values='n/a')

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
            self.data.file_data = data.to_numpy()

            # Update header_list if rois exist
            self.data.roi_names = np.array(rois, dtype=object)
        
        elif file_path.endswith(".dtseries.nii"):
            self.data.cifti_data = nib.load(file_path)
            self.data.file_data = cifti.parcellate(self.data.cifti_data, atlas="glasser")

        elif file_path.endswith(".ptseries.nii"):
            data = nib.load(file_path)
            self.data.file_data = data.get_fdata()

        else:
            self.data.file_data = None
            self.time_series_textbox.setText("Unsupported file format")

        # New data, reset slider and plot
        self.currentSliderValue = 0
        self.slider.setValue(0)
        self.figure.clear()
        self.canvas.draw()

        # Set filenames depending on file type
        if file_path.endswith('.pkl'):
            self.fileNameLabel.setText(f"Loaded TIME_SERIES object")
            self.time_series_textbox.setText(file_name)

            self.continuousCheckBox.setEnabled(False)
            self.continuousCheckBox.setChecked(False)

            self.stateBasedCheckBox.setEnabled(True)
            self.stateBasedCheckBox.setChecked(True)

            self.staticCheckBox.setEnabled(False)
            self.staticCheckBox.setChecked(False)

            self.transposeCheckbox.setEnabled(False)
        
        else:
            self.time_series_textbox.setText(file_name)

            self.continuousCheckBox.setEnabled(True)
            self.continuousCheckBox.setChecked(True)

            self.stateBasedCheckBox.setEnabled(False)
            self.stateBasedCheckBox.setChecked(False)

            self.staticCheckBox.setEnabled(True)
            self.staticCheckBox.setChecked(True)

            if file_path.endswith('.nii'):
                self.fileNameLabel.setText(f"Loaded and parcellated {self.data.file_name} with shape {self.data.file_data.shape}")
                self.transposeCheckbox.setEnabled(False)
            else:
                self.fileNameLabel.setText(f"Loaded {self.data.file_name} with shape {self.data.file_data.shape}")
                self.transposeCheckbox.setEnabled(True)
        
        # Reset and enable the GUI elements
        self.methodComboBox.setEnabled(True)
        self.methodComboBox.setEnabled(True)
        self.calculateButton.setEnabled(True)
        self.clearMemoryButton.setEnabled(True)
        self.keepInMemoryCheckbox.setEnabled(True)

    def saveConnectivityFile(self):
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
                                print(f"Converted None to empty array for dict key: {k}")
                            else:
                                converted_dict[k] = v
                        data_dict[field] = converted_dict
                    elif value is None:
                        data_dict[field] = np.array([])
                        print(f"Converted None to empty array for field: {field}")
                    elif field == 'dfc_instance':
                        pass
                    else:
                        data_dict[field] = value

                savemat(filePath, data_dict)

            except Exception as e:
                QMessageBox.warning(self, "Output Error", f"Error saving data: {e}")

        return

    def onTransposeChecked(self, state):
        if self.data.file_data is None:
            return  # No data loaded, so do nothing

        if state == Qt.CheckState.Checked:
            # Transpose the data
            self.data.file_data = self.data.file_data.transpose()
        else:
            # Transpose it back to original
            self.data.file_data = self.data.file_data.transpose()

        # Update the labels
        self.fileNameLabel.setText(f"Loaded {self.time_series_textbox.text()} with shape: {self.data.file_data.shape}")
        self.time_series_textbox.setText(self.data.file_name)

    # Graph tab
    def loadGraphFile(self):
        fileFilter = "All Supported Files (*.mat *.txt *.npy *.pkl *.tsv *.dtseries.nii *.ptseries.nii);;MAT files (*.mat);;Text files (*.txt);;NumPy files (*.npy);;Pickle files (*.pkl);;TSV files (*.tsv);;CIFTI files (*.dtseries.nii *.ptseries.nii)"
        file_path, _ = QFileDialog.getOpenFileName(self, "Load File", "", fileFilter)
        file_name = file_path.split('/')[-1]
        self.data.graph_file = file_name

        if not file_path:
            return  # Early exit if no file is selected

        if file_path.endswith('.mat'):
            data_dict = loadmat(file_path)
            try:
                self.data.graph_data = data_dict["graph_data"] # Try to load graph_data (saving files with comet will create this field)
            except:
                self.data.graph_data = data_dict[list(data_dict.keys())[-1]] # Else get the last item in the file (which is the data if there is only one field)

        elif file_path.endswith('.txt'):
            self.data.graph_data = np.loadtxt(file_path)
        
        elif file_path.endswith('.npy'):
            self.data.graph_data = np.load(file_path)
      
        else:
            self.data.graph_data = None
            self.time_series_textbox.setText("Unsupported file format")

        # Check if data is square
        if self.data.graph_data.ndim != 2 or self.data.graph_data.shape[0] != self.data.graph_data.shape[1]:
            QMessageBox.warning(self, "Data Error", "The loaded data is not a square matrix.")
            self.data.graph_data = None
            return

        self.data.graph_raw = self.data.graph_data
        self.graphFileNameLabel.setText(f"Loaded {self.data.graph_file} with shape {self.data.graph_data.shape}")
        
        self.plotGraph()
        self.onGraphCombobox()

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
                                print(f"Converted None to empty array for dict key: {k}")
                            else:
                                converted_dict[k] = v
                        data_dict[field] = converted_dict
                    elif value is None:
                        data_dict[field] = np.array([])
                        print(f"Converted None to empty array for field: {field}")
                    elif field == 'dfc_instance':
                        pass
                    else:
                        data_dict[field] = value

                savemat(filePath, data_dict)
            
            except Exception as e:
                QMessageBox.warning(self, "Output Error", f"Error saving data: {e}")
            
            return

    def takeCurrentData(self):
        if self.data.dfc_data is None:
            QMessageBox.warning(self, "Output Error", "No current dFC data available.")
            return
        
        if len(self.data.dfc_data.shape) == 3:
            self.data.graph_data = self.data.dfc_data[:,:,self.currentSliderValue]
        elif len(self.data.dfc_data.shape) == 2:
            self.data.graph_data = self.data.dfc_data
        else:
            QMessageBox.warning(self, "Output Error", "FC data seems to have the wrong shape.")
            return
        
        self.data.graph_raw = self.data.graph_data
        
        print(f"Used current dFC data with shape {self.data.graph_data.shape}")
        self.graphFileNameLabel.setText(f"Used current dFC data with shape {self.data.graph_data.shape}")
        self.data.graph_file = f"dfC from {self.data.file_name}" #with {self.data.dfc_name} at t={self.currentSliderValue}"
        self.plotGraph()
        self.onGraphCombobox()

    """
    dFC/graph functions
    """
    # dFC tab
    def onMethodCombobox(self, methodName=None):
        # Clear old variables and data
        self.clearParameters(self.parameterLayout)

        # Return if no methods are available
        if methodName == None or methodName == "Use checkboxes to get available methods":
            return
        
        # Get selected connectivity method
        self.data.dfc_instance = getattr(methods, self.class_info.get(methodName), None) # the actual class
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
            print(f"Loaded {self.data.dfc_name} from memory")

            # Plot the data
            self.plotConnectivity()
            self.plotDistribution()
            self.plotTimeSeries()

            # Update the slider
            total_length = self.data.dfc_data.shape[2] if len(self.data.dfc_data.shape) == 3 else 0
            position_text = f"t = {self.currentSliderValue} / {total_length-1}" if len(self.data.dfc_data.shape) == 3 else " static "
            self.positionLabel.setText(position_text)
            self.slider.setValue(self.slider.value())
        
        # If connectivity data does not exist we reset the figure and slider to prepare for a new calculation
        # This also indicates to the user that this data was not yet calculated/saved
        else:
            self.figure.clear()
            self.plotLogo(self.figure)
            self.canvas.draw()
            self.distributionFigure.clear()
            self.distributionCanvas.draw()
            self.timeSeriesFigure.clear()
            self.timeSeriesCanvas.draw()

            position_text = f"no data available"
            self.positionLabel.setText(position_text)
            self.slider.setValue(self.slider.value())
            self.slider.hide()

        self.update() # Update UI

    def updateMethodComboBox(self):

        def shouldIncludeClass(className):
            if self.continuousCheckBox.isChecked() and className.startswith("CONT"):
                    return True
            if self.stateBasedCheckBox.isChecked() and className.startswith("STATE"):
                    return True 
            if self.staticCheckBox.isChecked() and className.startswith("STATIC"):
                    return True
            return False

        class_mappings = {
            'CONT': [
                'Sliding Window', 'Jackknife Correlation', 'Flexible Least Squares', 'Spatial Distance', 
                'Multiplication of Temporal Derivatives', 'Dynamic Conditional Correlation', 
                'Phase Synchronization', 'Leading Eigenvector Dynamics', 'Wavelet Coherence', 'Edge-centric Connectivity'
            ],
            'STATE': [
                'Sliding Window Clustering', 'Co-activation patterns', 'Discrete Hidden Markov Model', 
                'Continuous Hidden Markov Model', 'Windowless'
            ],
            'STATIC': [
                'Pearson Correlation', 'Partial Correlation', 'Mutual Information'
            ]
        }

        # Dynamically generate the ordered classes based on available mappings
        ordered_classes = [
            f"{prefix} {name}" for prefix, names in class_mappings.items() for name in names
        ]

        # Generic filtering function
        filtered_and_ordered_classes = [
            class_name for class_name in ordered_classes
            if shouldIncludeClass(class_name) and class_name in self.class_info
        ]


        # Disconnect existing connections to avoid multiple calls
        try:
            self.methodComboBox.currentTextChanged.disconnect(self.onMethodCombobox)
        except TypeError:
            pass

        # Update the combobox
        self.methodComboBox.clear()
        self.methodComboBox.addItems(filtered_and_ordered_classes)

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
        if filtered_and_ordered_classes:
            self.onMethodCombobox(filtered_and_ordered_classes[0])

    def getInfoText(self, param, dfc_method):
        if param == "windowsize":
            text = "Size of the window used by the method. Should typically be an uneven number to have a center."
        elif param == "shape":
            text = "Shape of the windowing function."
        elif param == "std":
            text = "Width (sigma) of the window."
        elif param == "diagonal":
            text = "Values for the main diagonal of the connectivity matrix."
        elif param == "fisher_z":
            text = "Fisher z-transform the connectivity values."
        elif param == "num_cores":
            text = "Parallelize on multiple cores (highly recommended for DCC and FLS)."
        elif param == "standardizeData":
            text = "z-standardize the time series data."
        elif param == "mu":
            text = "Weighting parameter for FLS. Smaller values will produce more erratic changes in connectivity estimate."
        elif param == "flip_eigenvectors":
            text = "Flips the sign of the eigenvectors."
        elif param == "dist":
            text = "Distance function"
        elif param == "TR":
            text = "Repetition time of the data (in seconds)"
        elif param == "fmin":
            text = "Minimum wavelet frequency"
        elif param == "fmax":
            text = "Maximum wavelet frequency"
        elif param == "n_scales":
            text = "Number of wavelet scales"
        elif param == "drop_scales":
            text = "Drop the n largest and smalles scales to account for the cone of influence"
        elif param == "drop_timepoints":
            text = "Drop n first and last time points from the time series to account for the cone of influence"
        elif param == "method" and dfc_method == "WaveletCoherence":
            text = "Specific implementation of the method"
        elif param == "method" and dfc_method == "PhaseSynchrony":
            text = "Specific implementation of the method"
        elif param == "params":
            text = "Various parameters"
        elif param == "coi_correction":
            text = "Cone of influence correction"
        elif param == "clstr_distance":
            text = "Distance metric"
        elif param == "num_bins":
            text = "Number of bins for discretization"
        elif param == "method":
            text = "Specific type of method"
        elif param == "n_overlap":
            text = "Window overlap"
        elif param == "tapered_window":
            text = "Tapered window"
        elif param == "n_states":
            text = "Number of states"
        elif param == "n_subj_clusters":
            text = "Number of subjects"
        elif param == "normalization":
            text = "Normalization"
        elif param == "clstr_distance":
            text = "Distance measure"
        elif param == "subject":
            text = "Subject"
        elif param == "Base measure":
            text = "Base measure for the clustering"
        elif param == "Iterations":
            text = "Number of iterations"
        elif param == "Sliding window":
            text = "Sliding window method"
        elif param == "State ratio":
            text = "Observation/state ratio for the DHMM"
        elif param == "vlim":
            text = "Limit for color axis (edge time series)"
        else:
            text = f"TODO"
        return text

    def onAtlasSelected(self):
        atlas_name = self.atlasComboBox.currentText()
        atlas_map = {
            "Glasser MMP": "glasser",
            "Schaefer Kong 200": "schaefer_kong",
            "Schaefer Tian 254": "schaefer_tian"
        }
        atlas_name = atlas_map.get(atlas_name, None)

        self.data.file_data = cifti.parcellate(self.data.cifti_data, atlas=atlas_name)
        self.fileNameLabel.setText(f"Loaded and parcellated {self.data.file_name} with shape {self.data.file_data.shape}")

    # Graph tab
    def onGraphCombobox(self):
        self.setGraphParameters()

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

    """
    Parameters
    """
    # dFC tab
    def initParameters(self, class_instance):
        # Now the parameter labels and boxes are set up    
        labels = []

        # Calculate the maximum label width (just a visual thing)
        max_label_width = 0
        init_signature = inspect.signature(class_instance.__init__)
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
        
        self.time_series_textbox.setPlaceholderText("No data loaded yet")
        if self.data.file_name:
            self.time_series_textbox.setText(self.data.file_name)
        self.time_series_textbox.setEnabled(True)

        # Create info button for time_series
        time_series_info_text = "2D time series loaded from file. Time has to be the first dimension."
        time_series_info_button = InfoButton(time_series_info_text)

        time_series_layout = QHBoxLayout()
        time_series_layout.addWidget(time_series_label)
        time_series_layout.addWidget(self.time_series_textbox)
        time_series_layout.addWidget(time_series_info_button)
        self.parameterLayout.addLayout(time_series_layout)

        # Adjust max width for aesthetics
        max_label_width += 5
        time_series_label.setFixedWidth(max_label_width)

        # If we have a .dtseries.nii file, we need to add an atlas dropdown. This defaults to the glasser atlas
        if self.data.file_name is not None and self.data.file_name.endswith('.dtseries.nii'):
            atlas_label = QLabel("Parcellation:")
            atlas_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
            atlas_label.setMinimumSize(atlas_label.sizeHint())
            atlas_label.setFixedWidth(max_label_width)
            labels.append(atlas_label)

            # Create the info button for parcellation
            atlas_info_button_text = "Atlas to parcellate the .dtseries.nii file."
            atlas_info_button = InfoButton(atlas_info_button_text)

            # Create layout for the atlas dropdown
            atlas_layout = QHBoxLayout()
            atlas_layout.addWidget(atlas_label)
            atlas_layout.addWidget(self.atlasComboBox)
            atlas_layout.addWidget(atlas_info_button)

            # Add the atlas layout to the main parameter layout
            self.parameterLayout.addLayout(atlas_layout)

        for param in init_signature.parameters.values():
            if param.name not in ['self', 'time_series', 'tril', 'standardize', 'params']:
                # Create label for parameter
                param_label = QLabel(f"{self.param_names[param.name]}:")
                param_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
                param_label.setMinimumSize(param_label.sizeHint())
                param_label.setFixedWidth(max_label_width)
                labels.append(param_label)

                # Determine the widget type based on the parameter
                # Dropdown for boolean parameters
                if type(param.default) == bool:
                    param_input_widget = QComboBox()
                    param_input_widget.addItems(["True", "False"])
                    
                    default_index = param_input_widget.findText(str(param.default))
                    param_input_widget.setCurrentIndex(default_index)
                    param_input_widget.setEnabled(True)

                # Dropdown for parameters with predefined options
                elif param.name in class_instance.options:
                    param_input_widget = QComboBox()
                    param_input_widget.addItems(class_instance.options[param.name])
                    
                    if param.default in class_instance.options[param.name]:
                        default_index = param_input_widget.findText(param.default)
                        param_input_widget.setCurrentIndex(default_index)
                        param_input_widget.setEnabled(True)

                # Spinbox for integer parameterss
                elif type(param.default) == int:
                    param_input_widget = QSpinBox()
                    param_input_widget.setMaximum(10000)
                    param_input_widget.setMinimum(-10000)
                    param_input_widget.setSingleStep(1)

                    param_input_widget.setValue(int(param.default) if param.default != inspect.Parameter.empty else 0)
                    param_input_widget.setEnabled(True)

                # Spinbox for float parameters
                elif type(param.default) == float:
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
                info_text = self.getInfoText(param.name, self.data.dfc_name)
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
        
        self.data.dfc_params['time_series'] = self.data.file_data # Time series data
        
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
                        self.calculatingLabel.setText(f"Error: Unrecognized parameter '{label}'")
                else:
                    # Value could not be retrieved from the widget
                    self.calculatingLabel.setText(f"Error: No value entered for parameter '{label}'")

    def setParameters(self, disable=False):
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
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.setValue(int(value))

        # No parameters yet, return
        if not self.data.dfc_params:
            self.getParameters()
            return

        # Time series data has to be in the params as we run the dFC method with just these params
        self.data.dfc_params['time_series'] = self.data.file_data

        if disable:
            self.time_series_textbox.setText(self.data.file_name)
            self.time_series_textbox.setEnabled(False)

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
                        if disable:
                            widget.setEnabled(False)
                    else:
                        # Value could not be retrieved from the dictionary
                        self.calculatingLabel.setText(f"Error: No value entered for parameter '{label}'")
                else:
                    self.calculatingLabel.setText(f"Error: Unrecognized parameter '{label}'")

    def clearParameters(self, layout):
        while layout.count():
            item = layout.takeAt(0)  # Take the first item from the layout
            if item.widget():  # If the item is a widget
                widget = item.widget()
                if widget is not None and widget is not self.time_series_textbox and widget is not self.atlasComboBox: # do not clear time series textbox and atlas combobox
                    widget.deleteLater()  # Schedule the widget for deletion
            elif item.layout():  # If the item is a layout
                self.clearParameters(item.layout())  # Recursively clear the layout
                item.layout().deleteLater()  # Delete the layout itself
            elif item.spacerItem():  # If the item is a spacer
                # No need to delete spacer items; they are automatically handled by Qt
                pass
    
    # Graph tab
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
                    param_widget.setPlaceholderText("No data loaded yet")
                    if self.data.graph_file:
                        param_widget = QLineEdit("as shown in plot")
                    param_widget.setReadOnly(True)  # Make the widget read-only
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
                        param_widget.setSingleStep(param_default)
                    # Float 
                    elif param_type == float:    
                        param_widget = QDoubleSpinBox()
                        if name == "threshold":
                            param_widget.setValue(0.0)
                        else:
                            param_widget.setValue(param_default)
                        param_widget.setMaximum(1.0)
                        param_widget.setMinimum(0.0)
                        param_widget.setSingleStep(0.01)
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

    def onClearGraphOptions(self):
        self.data.graph_data = self.data.graph_raw
        self.plotGraph()
        self.optionsTextbox.clear()
        self.graphTextbox.clear()
        self.graphStepCounter = 1

    """
    Calculation
    """
    # dFC tab
    def onCalculateButton(self):
        # Check if ts_data is available
        if self.data.file_data is None:
            self.calculatingLabel.setText(f"Error. No time series data has been loaded.")
            return
        
        # Get the current parameters from the UI for the upcoming calculation
        self.getParameters()
    
        # Process all pending events
        QApplication.processEvents() 
        
        # Start worker thread for dFC calculations and submit for calculation
        self.workerThread = QThread()
        self.worker = Worker(self.calculateConnectivity, self.data.dfc_params)
        self.worker.moveToThread(self.workerThread)
        
        self.worker.finished.connect(self.workerThread.quit)
        self.worker.result.connect(self.handleResult)
        self.worker.error.connect(self.handleError)

        self.workerThread.started.connect(self.worker.run)
        self.workerThread.start()
        self.calculatingLabel.setText(f"Calculating {self.methodComboBox.currentText()}, please wait...")
        self.calculateButton.setEnabled(False)
    
    def calculateConnectivity(self, parameters):
        keep_in_memory = self.keepInMemoryCheckbox.isChecked()
        
        # Check if data already exists
        existing_data = self.data_storage.check_for_identical_data(self.data)
        if existing_data is not None:
            return existing_data
        
        # Remove keys not allowed for calculation
        clean_parameters = parameters.copy()
        clean_parameters.pop('parcellation', None)

        # Data does not exist, perform calculation
        connectivity_calculator = self.data.dfc_instance(**clean_parameters)
        result = connectivity_calculator.connectivity()
        self.init_flag = False

        # In case the method returns multiple values. The first one is always the NxNxT dfc matrix
        if isinstance(result, tuple):
            self.data.dfc_data = result[0]
            self.data.dfc_params = parameters
            self.data.dfc_state_tc = None
            self.data.dfc_edge_ts = result[1][0] if isinstance(result[1], tuple) else None

        # Result is DFC object (pydfc methods)
        elif isinstance(result, pydfc.dfc.DFC):
            self.data.dfc_data = np.transpose(result.get_dFC_mat(), (1, 2, 0))
            self.data.dfc_params = parameters
            self.data.dfc_states = result.FCSs_
            self.data.dfc_state_tc = result.state_TC()
            self.data.dfc_edge_ts = None
        
        # Only a single matrix is returned (most cases)
        else:
            self.data.dfc_data = result
            self.data.dfc_params = parameters
            self.data.dfc_state_tc = None
            self.data.dfc_edge_ts = None

        # Store in memory if checkbox is checked
        if keep_in_memory:
            # Update the dictionary entry for the selected_class_name with the new data and parameters
            self.data_storage.add_data(self.data)

        print("Finished calculation.")
        return self.data

    def handleResult(self):
        # Update the sliders and text
        if self.data.dfc_data is not None:
            self.calculatingLabel.setText(f"Calculated {self.data.dfc_name} with shape {self.data.dfc_data.shape}")
            
            if len(self.data.dfc_data.shape) == 3:
                self.slider.show()
                self.rowSelector.setMaximum(self.data.dfc_data.shape[0] - 1)
                self.colSelector.setMaximum(self.data.dfc_data.shape[1] - 1)
                self.rowSelector.setValue(1)

            # Update time label
            total_length = self.data.dfc_data.shape[2] if len(self.data.dfc_data.shape) == 3 else 0
            
            if self.currentTabIndex == 0 or self.currentTabIndex == 2:
                position_text = f"t = {self.currentSliderValue} / {total_length-1}" if len(self.data.dfc_data.shape) == 3 else " static "
            else:
                position_text = ""

            self.positionLabel.setText(position_text)
            self.slider.setValue(self.slider.value())
            
        # Plot
        self.plotConnectivity()
        self.plotDistribution()
        self.plotTimeSeries()

        self.calculateButton.setEnabled(True)
        self.onTabChanged()
        self.update()

    def handleError(self, error):
        # Handles errors in the worker thread
        QMessageBox.warning(self, "Calculation Error", f"Error occurred furing calculation: {error}")
        self.calculateButton.setEnabled(True)
        self.data.clear_dfc_data()
        self.positionLabel.setText("no data available")
        self.plotLogo(self.figure)
        self.canvas.draw()

    # Graph tab
    def onAddGraphOption(self):

        # Start worker thread for graph calculations
        self.workerThread = QThread()
        self.worker = Worker(self.calculateGraph, None)
        self.worker.moveToThread(self.workerThread)

        self.worker.finished.connect(self.workerThread.quit)
        self.worker.result.connect(self.handleGraphResult)  # Ensure you have a slot to handle results
        self.worker.error.connect(self.handleGraphError)  # And error handling

        self.workerThread.started.connect(self.worker.run)
        self.workerThread.start()

    def calculateGraph(self, unused):
        option, params = self.getGraphOptions()

        # Get the function
        func_name = self.reverse_graphOptions[option]
        func = getattr(graph, func_name)

        option_name = re.sub(r'^\S+\s+', '', option) # regex to remove the PREP/GRAPH part
        self.optionsTextbox.append(f"{self.graphStepCounter}. {option_name}: calculating, please wait...")

        first_param_name = next(iter(params))
        graph_params = {first_param_name: self.data.graph_data}
        graph_params.update({k: v for k, v in params.items() if k != first_param_name})

        graph_data = func(**graph_params)
        return f'graph_{option.split()[0].lower()}', graph_data, option_name, graph_params

    def handleGraphResult(self, result):
        output = result[0]
        data   = result[1]
        option = result[2]
        params = result[3]

        print(f"Finished calculation for {option}, output data: {type(data)}.")

        # Update self.data.graph_data or self.data.graph_out based on the result
        if output == 'graph_prep':
            self.data.graph_data = data
            self.plotGraph()
        else:
            self.data.graph_out = data
            self.plotMeasure(option)

        # Output step and options to textbox, remove unused parameters
        if option == 'Threshold':
            if params.get('type') == 'absolute':
                filtered_params = {k: v for k, v in params.items() if k != 'density'}
            elif params.get('type') == 'density':
                filtered_params = {k: v for k, v in params.items() if k != 'threshold'}
        else:
            filtered_params = params

        filtered_params = {k: v for k, v in list(filtered_params.items())[1:]}

        # Update the textbox with the current step and options
        current_text = self.optionsTextbox.toPlainText()
        lines = current_text.split('\n')

        if len(lines) > 1:
            lines[-1] = f"{self.graphStepCounter}. {option}: {filtered_params}"
        else:
            lines = [f"{self.graphStepCounter}. {option}: {filtered_params}"]

        updated_text = '\n'.join(lines)
        self.optionsTextbox.setPlainText(updated_text)

        self.graphStepCounter += 1

    def handleGraphError(self, error):
        # Handles errors in the worker thread
        QMessageBox.warning(self, "Calculation Error", f"Error occurred furing calculation: {error}")

    """
    Memory functions
    """
    # dFC tab
    def onKeepInMemoryChecked(self, state):
        if state == 2 and self.data.dfc_data is not None:
            self.data_storage.add_data(self.data)
                
    def onClearMemory(self):
        self.data_storage = DataStorage()
        
        self.figure.clear()
        self.canvas.draw()
        self.distributionFigure.clear()
        self.distributionCanvas.draw()

        self.calculatingLabel.setText(f"Cleared memory")
        print("Cleared memory")
        return

    """
    Plotting functions
    """
    # dFC tab
    def plotConnectivity(self):
        current_data = self.data.dfc_data
        
        if current_data is None:
            QMessageBox.warning(self, "No calculated data available for plotting")
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        try:
            current_slice = current_data[:, :, self.currentSliderValue] if len(current_data.shape) == 3 else current_data
            vmax = np.max(np.abs(current_slice))
            self.im = ax.imshow(current_slice, cmap='coolwarm', vmin=-vmax, vmax=vmax)
        except:
            current_slice = current_data[:, :, 0] if len(current_data.shape) == 3 else current_data
            vmax = np.max(np.abs(current_slice))
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
        cbar = self.figure.colorbar(self.im, cax=cax)
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1f}'))

        self.slider.setMaximum(current_data.shape[2] - 1 if len(current_data.shape) == 3 else 0)
    
        self.figure.set_facecolor('#E0E0E0')
        self.figure.tight_layout()
        self.canvas.draw()

    def plotTimeSeries(self):
        current_data = self.data.dfc_data

        # Get dimensions of the data
        if current_data is not None and current_data.ndim == 3:
            self.rowSelector.setMaximum(current_data.shape[0] - 1)
            self.colSelector.setMaximum(current_data.shape[1] - 1)

        row = self.rowSelector.value()
        col = self.colSelector.value()
        self.rowSelector.show()
        self.colSelector.show()

        if current_data is not None and row < current_data.shape[0] and col < current_data.shape[1] and self.data.dfc_edge_ts is None and self.data.dfc_state_tc is None:    
            self.timeSeriesFigure.clear()
            ax = self.timeSeriesFigure.add_subplot(111)
            time_series = current_data[row, col, :] if len(current_data.shape) == 3 else current_data[row, col]
            ax.set_title(f"dFC time course between region {row} and {col}.")
            ax.set_xlabel("time (TRs)")
            ax.set_ylabel("dFC strength")
            ax.plot(time_series)

        elif self.data.dfc_state_tc is not None:
            self.timeSeriesFigure.clear()

            time_series = self.data.dfc_state_tc
            num_states = len(self.data.dfc_states)

            # Setup the gridspec layout
            gs = gridspec.GridSpec(3, num_states, self.timeSeriesFigure, height_ratios=[1, 0.5, 1])

            # Hite selectors
            self.rowSelector.hide()
            self.colSelector.hide()

            # Plotting the state time course across all columns
            ax_time_series = self.timeSeriesFigure.add_subplot(gs[0, :])
            ax_time_series.plot(time_series)
            ax_time_series.set_ylabel("State")
            ax_time_series.set_title("State time course")
            ax_time_series.set_xlabel("Time (TRs)")

            # Plot the individual states
            for col, (state, matrix) in enumerate(self.data.dfc_states.items()):
                ax_state = self.timeSeriesFigure.add_subplot(gs[2, col])
                ax_state.imshow(matrix, cmap='coolwarm', aspect=1)
                ax_state.set_title(f"State {col+1}")
                ax_state.set_xticks([])
                ax_state.set_yticks([]) 

        elif self.data.dfc_edge_ts is not None:
            self.timeSeriesFigure.clear()
            gs = gridspec.GridSpec(3, 1, self.timeSeriesFigure, height_ratios=[2, 0.5, 1]) # GridSpec with 3 rows and 1 column

            # The first subplot occupies the 1st row
            ax1 = self.timeSeriesFigure.add_subplot(gs[:1, 0])
            ax1.imshow(self.data.dfc_edge_ts.T, cmap='coolwarm', aspect='auto', vmin=-1*self.data.dfc_params["vlim"], vmax=self.data.dfc_params["vlim"])
            ax1.set_title("Edge time series")
            ax1.set_xlabel("Time (TRs)")
            ax1.set_ylabel("Edges")

            # The second subplot occupies the 3rd row
            ax2 = self.timeSeriesFigure.add_subplot(gs[2, 0])
            mean_edge_values = np.mean(self.data.dfc_edge_ts.T, axis=0)
            ax2.plot(mean_edge_values)
            ax2.set_xlim(0, len(mean_edge_values) - 1)
            ax2.set_title("Mean time series")
            ax2.set_xlabel("Time (TRs)")
            ax2.set_ylabel("Mean Edge Value")
        
        else:
            # Clear the plot if the data is not available
            self.timeSeriesFigure.clear()

        self.timeSeriesFigure.set_facecolor('#E0E0E0')
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

        self.distributionFigure.set_facecolor('#E0E0E0')
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
        self.currentTabIndex = self.tabWidget.currentIndex()
        # index 0: Connectivity plot
        # index 1: Time series plot
        # index 2: Distribution plot
        # index 3: Graph analysis

        if self.data.dfc_data is None:
            self.plotLogo(self.figure)
            self.canvas.draw()
            self.distributionFigure.clear()
            self.distributionCanvas.draw()
            self.timeSeriesFigure.clear()
            self.timeSeriesCanvas.draw()
            self.backLargeButton.hide()
            self.backButton.hide()
            self.forwardButton.hide()
            self.forwardLargeButton.hide()
            self.slider.hide()
            position_text = ""
            return
        
        if self.currentTabIndex == 0 or self.currentTabIndex == 2:
            self.slider.show()
            self.slider.setValue(self.currentSliderValue)
            self.backLargeButton.show()
            self.backButton.show()
            self.forwardButton.show()
            self.forwardLargeButton.show()

            if self.data.dfc_data is not None:
                total_length = self.data.dfc_data.shape[2] if len(self.data.dfc_data.shape) == 3 else 0
                position_text = f"t = {self.currentSliderValue} / {total_length-1}" if len(self.data.dfc_data.shape) == 3 else " static "
            else:
                position_text = "no data available"

            self.positionLabel.setText(position_text)

        elif self.currentTabIndex == 1:
            self.backLargeButton.hide()
            self.backButton.hide()
            self.forwardButton.hide()
            self.forwardLargeButton.hide()
            
            # If we have nothing to scroll though, hide some GUI elements
            if len(self.data.dfc_data.shape) == 2 or self.data.dfc_edge_ts is not None or self.data.dfc_state_tc is not None:
                position_text = ""
                self.slider.hide()
                
                # Disable brain area selector widgets
                for i in range(self.timeSeriesSelectorLayout.count()):    
                    widget = self.timeSeriesSelectorLayout.itemAt(i).widget()
                    if widget is not None:
                        widget.setVisible(False)

            else:
                self.slider.hide()
                position_text = ""
                #position_text = f"Use the slider to zoom in and scroll through the time series"
                
                # Enable brain area selector widgets
                for i in range(self.timeSeriesSelectorLayout.count()):    
                    widget = self.timeSeriesSelectorLayout.itemAt(i).widget()
                    if widget is not None:
                        widget.setVisible(True)

            # We have a static measure
            if len(self.data.dfc_data.shape) == 2 and self.data.dfc_edge_ts is None and self.data.dfc_state_tc is None:
                self.timeSeriesFigure.clear()
                self.timeSeriesCanvas.draw()

                # Disable brain area selector widgets
                for i in range(self.timeSeriesSelectorLayout.count()):    
                    widget = self.timeSeriesSelectorLayout.itemAt(i).widget()
                    if widget is not None:
                        widget.setVisible(False)
                        
        if self.currentTabIndex == 3:
            self.backLargeButton.hide()
            self.backButton.hide()
            self.forwardButton.hide()
            self.forwardLargeButton.hide()
            self.slider.hide()
            position_text = ""
            
        self.positionLabel.setText(position_text)
        self.update()

    def onSliderValueChanged(self, value):
        # Ensure there is data to work with
        if self.data.dfc_data is None or self.im is None:
            return
        
        if self.currentTabIndex == 0 or self.currentTabIndex == 2:
            # Get and update the data of the imshow object
            self.currentSliderValue = value
            data = self.data.dfc_data
            self.im.set_data(data[:, :, value]) if len(data.shape) == 3 else self.im.set_data(data)

            vlim = np.max(np.abs(data[:, :, value])) if len(data.shape) == 3 else np.max(np.abs(data))
            self.im.set_clim(-vlim, vlim)

            # Redraw the canvas
            self.canvas.draw()
            self.plotDistribution()

            total_length = self.data.dfc_data.shape[2] if len(self.data.dfc_data.shape) == 3 else 0
            position_text = f"t = {self.currentSliderValue} / {total_length-1}" if len(self.data.dfc_data.shape) == 3 else " static "
            self.positionLabel.setText(position_text)

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

        self.currentSliderValue = max(0, min(self.slider.value() + delta, self.slider.maximum()))
        self.slider.setValue(self.currentSliderValue)
        self.slider.update()
        
        self.plotConnectivity()
        self.plotDistribution()

    # Graph tab
    def plotGraph(self):
        current_data = self.data.graph_data
        
        if current_data is None:
            QMessageBox.warning(self, "No calculated data available for plotting")
            return

        self.matrixFigure.clear()
        ax = self.matrixFigure.add_subplot(111)

        vmax = np.max(np.abs(current_data))
        self.im = ax.imshow(current_data, cmap='coolwarm', vmin=-vmax, vmax=vmax)
        ax.set_xlabel("ROI")
        ax.set_ylabel("ROI")

        # Create the colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.15)
        cbar = self.matrixFigure.colorbar(self.im, cax=cax)
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1f}'))
    
        self.matrixFigure.set_facecolor('#E0E0E0')
        self.matrixFigure.tight_layout()
        self.matrixCanvas.draw()

    def plotMeasure(self, measure):
        self.graphFigure.clear()
        ax = self.graphFigure.add_subplot(111)
        
        # Check type of the graph output data
        if isinstance(self.data.graph_out, (np.ndarray, np.float64)):
            if self.data.graph_out.ndim == 0:
                # If graph_out is a single value (0D array)
                self.graphTextbox.append(f"{measure}: {self.data.graph_out.item()}")
            elif self.data.graph_out.ndim == 1:
                # For a 1D array, plot a vertical lollipop plot
                ax.stem(self.data.graph_out, linefmt="#19232d", markerfmt='o', basefmt=" ")
                ax.set_xlabel("ROI")
                ax.set_ylabel(measure)

                # Calculate mean and variance, and update the textbox
                mean_val = np.mean(self.data.graph_out)
                var_val = np.var(self.data.graph_out)
                self.graphTextbox.append(f"{measure} (mean: {mean_val:.2f}, variance: {var_val:.2f})")
            
            elif self.data.graph_out.ndim == 2:
                # For a 2D array, use imshow
                vmax = np.max(np.abs(self.data.graph_out))
                im = ax.imshow(self.data.graph_out, cmap='coolwarm', vmin=-vmax, vmax=vmax)

                # Create the colorbar
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.15)
                self.graphFigure.colorbar(im, cax=cax).ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1f}'))
            else:
                self.graphTextbox.append("3D graph data not currently supported for plotting.")
        
        elif isinstance(self.data.graph_out, dict):
            # Setup data for output
            output_string = f"{measure}: "
            output_arrays = []
            for key, value in self.data.graph_out.items():
                if isinstance(value, (int, float)):
                    output_string += f"{key}: {value:.2f}, "
                    self.plotLogo(self.graphFigure)

                elif isinstance(value, np.ndarray):
                    output_arrays.append((key, value))

            # Print the output string
            self.graphTextbox.append(output_string.strip(', '))  # Remove the trailing comma

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

                        # Calculate mean and variance, and update the textbox
                        mean_val = np.mean(value)
                        var_val = np.var(value)
                        self.graphTextbox.append(f"{measure} (mean: {mean_val:.2f}, variance: {var_val:.2f})")
                        
                    elif value.ndim == 2:
                        # For a 2D array, use imshow
                        im = ax.imshow(value, cmap='coolwarm', vmin=-vmax, vmax=vmax)
                        ax.set_title(key)
    
                        # Create the colorbar
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="5%", pad=0.15)
                        cbar = self.graphFigure.colorbar(im, cax=cax)
                        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1f}'))
  
                    else:
                        self.graphTextbox.append("Graph output data is not in expected format.")

                    ax.set_title(key)
        
        else:
            self.graphTextbox.append("Graph output data is not in expected format.")

        # Draw the plot
        self.graphFigure.set_facecolor('#E0E0E0')
        self.graphFigure.tight_layout()
        self.graphCanvas.draw()
        
        return
    
    # Shared tab
    def plotLogo(self, figure=None):
        with pkg_resources.path("comet.resources.img", "logo.png") as file_path:
            logo = imread(file_path)

        figure.clear()
        ax = figure.add_subplot(111)
        ax.set_axis_off()
        ax.imshow(logo)

        figure.set_facecolor('#f4f1f6')
        figure.tight_layout()

"""
Run the application
"""
def run(dfc_data=None, method=None):
    app = QApplication(sys.argv)

    # Set global stylesheet for tooltips
    QApplication.instance().setStyleSheet("""
        QToolTip {
            background-color: #E0E0E0;
            border: 1px solid black;
        }
    """)
    ex = App(init_dfc_data=dfc_data, init_dfc_instance=method)
    ex.setStyleSheet(qdarkstyle.load_stylesheet_pyqt6())

    default_width = ex.width()
    default_height = ex.height()
    new_width = int(default_width * 1.8)
    new_height = int(default_height * 1.5)
    ex.resize(new_width, new_height)

    ex.show()

    try:
        sys.exit(app.exec())
    except SystemExit as e:
        print(f"GUI closed with status {e}")

if __name__ == '__main__':
    run()
