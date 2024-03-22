import sys
import pickle
import inspect
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from importlib import resources as pkg_resources

import qdarkstyle
from PyQt6.QtCore import Qt, QPoint, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QEnterEvent, QFontMetrics
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, \
    QSlider, QToolTip, QWidget, QLabel, QFileDialog, QComboBox, QLineEdit, QSizePolicy, \
    QSpacerItem, QCheckBox, QTabWidget, QMessageBox, QSpinBox, QDoubleSpinBox

from matplotlib.image import imread
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec

from . import methods
import pydfc

# Worker class for dFC calculations, which run in a separate thread
class Worker(QObject):
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

# Info button class
class InfoButton(QPushButton):
    def __init__(self, info_text, parent=None):
        super().__init__("i", parent)
        self.info_text = info_text
        self.setStyleSheet("QPushButton { border: 1px solid black;}")
        self.setFixedSize(20, 20)

    def enterEvent(self, event: QEnterEvent):
        tooltip_pos = self.mapToGlobal(QPoint(self.width(), 0)) # Tooltip position can be adjusted here
        QToolTip.showText(tooltip_pos, self.info_text)
        super().enterEvent(event)

class App(QMainWindow):
    def __init__(self, init_data=None, init_method=None):
        super().__init__()
        self.title = 'Comet Dynamic Functional Connectivity Toolbox'
        self.ts_data = None
        self.roi_data = None
        self.dfc_data = {}
        self.state_tc = None
        self.dfc_states = None
        self.edge_ts = None
        self.abortFlag = False
        self.init_method = init_method
        self.dfc_data_dict = {}
        self.selected_class_name = None
        self.currentSliderValue = 0
        self.currentTabIndex = 0
        self.file_name = ""
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
            "vlim":                 "Color axis limit"

        }
        self.reverse_param_names = {v: k for k, v in self.param_names.items()}

        self.initUI()
        
        self.dfc_data['data'] = init_data
        self.dfc_data['parameters'] = None # TODO pass parameters

        if self.dfc_data['data'] is not None:
            self.init_from_calculated_data()

    def initUI(self):
        self.setWindowTitle(self.title)
        mainLayout = QHBoxLayout()

        ###############################
        #  Left section for settings  #
        ###############################
        self.leftLayout = QVBoxLayout()

        # Create button and label for file loading
        self.fileButton = QPushButton('Load File')
        self.fileNameLabel = QLabel('No file loaded')
        self.leftLayout.addWidget(self.fileButton)
        self.leftLayout.addWidget(self.fileNameLabel)
        self.fileButton.clicked.connect(self.loadFile)

        # Create a checkbox for reshaping the data
        self.reshapeCheckbox = QCheckBox("Transpose")
        self.leftLayout.addWidget(self.reshapeCheckbox)
        self.reshapeCheckbox.hide()

        # Connect the checkbox to a method
        self.reshapeCheckbox.stateChanged.connect(self.onReshapeCheckboxChanged)

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

        # Retrieve class names and their human-readable names
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
        
        # Initial population of the combobox
        self.updateMethodComboBox()

        # Add parameter textbox for time_series
        self.time_series_textbox = QLineEdit()
        self.time_series_textbox.setReadOnly(True) # read only as based on the loaded file

        # Add a stretch after the parameter layout container
        self.leftLayout.addStretch()

        # Calculate connectivity and save button
        buttonsLayout = QHBoxLayout()

        # Calculate connectivity button
        self.calculateButton = QPushButton('Calculate Connectivity')
        self.calculateButton.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        buttonsLayout.addWidget(self.calculateButton, 2)  # 2/3 of the space
        self.calculateButton.clicked.connect(self.onCalculateConnectivity)

        # Create the "Save" button
        self.saveButton = QPushButton('Save')
        self.saveButton.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        buttonsLayout.addWidget(self.saveButton, 1)  # 1/3 of the space
        self.saveButton.clicked.connect(self.saveFile)

        # Add the buttons layout to the left layout
        self.leftLayout.addLayout(buttonsLayout)

        # Memory buttons
        self.keepInMemoryCheckbox = QCheckBox("Keep in memory")
        self.keepInMemoryCheckbox.stateChanged.connect(self.onKeepInMemoryChanged)
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

        # Tab 4: Graph analysis
        #graphTab = QWidget()
        #graphLayout = QVBoxLayout()
        #graphTab.setLayout(graphLayout)
        #self.tabWidget.addTab(graphTab, "Graph analysis")

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

        # Adding buttons to the layout
        navButtonLayout.addWidget(self.backLargeButton)
        navButtonLayout.addWidget(self.backButton)
        navButtonLayout.addWidget(self.positionLabel)
        navButtonLayout.addWidget(self.forwardButton)
        navButtonLayout.addWidget(self.forwardLargeButton)

        # Connect buttons to their respective slots
        self.backLargeButton.clicked.connect(self.moveBackLarge)
        self.backButton.clicked.connect(self.moveBack)
        self.forwardButton.clicked.connect(self.moveForward)
        self.forwardLargeButton.clicked.connect(self.moveForwardLarge)

        navButtonLayout.addStretch(1)  # Spacer to the right of the buttons
        rightLayout.addLayout(navButtonLayout)

        # Initialize parameters for the default method (from left layout but has to be done after figure creation)
        self.onMethodChanged()

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

        #####################
        #  Combine layouts  #
        #####################
        mainLayout.addLayout(self.leftLayout, 1)
        mainLayout.addLayout(rightLayout, 2)

        # Set main window layout
        centralWidget = QWidget()
        centralWidget.setLayout(mainLayout)
        self.setCentralWidget(centralWidget)

        # Set checkboxes to default values
        self.continuousCheckBox.setChecked(True)
        self.stateBasedCheckBox.setChecked(True)
        self.staticCheckBox.setChecked(True)

    def init_from_calculated_data(self):
        # Make sure both the dFC data and the method object are provided
        assert self.init_method is not None, "Please provide the method object corresponding to your dFC data as the second argument to the GUI."

        # Get parameters
        self.selected_class_name = self.class_info.get(self.init_method.name)
        self.getParameters() # TODO: Something goes wrong here for SW (maybe because of strings in the parameters)
        self.dfc_data['parameters'] = self.parameters
        self.dfc_data_dict[self.selected_class_name] = {'data': self.dfc_data['data'], 'parameters': self.dfc_data['parameters']}

        # Set the slider elements
        total_length = self.dfc_data['data'].shape[2] if len(self.dfc_data['data'].shape) == 3 else 0
        position_text = f"t = {self.currentSliderValue} / {total_length-1}" if len(self.dfc_data['data'].shape) == 3 else " static "
        self.positionLabel.setText(position_text)
        self.slider.setValue(self.slider.value())

        # Disable the GUI elements
        self.methodComboBox.setEnabled(False)
        self.calculateButton.setEnabled(False)
        self.clearMemoryButton.setEnabled(False)
        self.keepInMemoryCheckbox.setEnabled(False)

        # Set labels
        self.fileNameLabel.setText(f"Loaded dFC from script")
        self.calculatingLabel.setText(f"Loaded dFC from script")
        self.methodComboBox.setCurrentText(self.init_method.name)
        
        # Set plots
        self.plot_dfc()
        self.updateDistribution()

        self.rowSelector.setValue(1)
        self.plotTimeSeries()

    def onMethodChanged(self, methodName=None):
        # Clear old variables and data
        self.clearLayout(self.parameterLayout)
        self.dfc_data['data'] = None
        self.dfc_data['parameters'] = None

        if methodName == None or methodName == "Use checkboxes to get available methods":
            return
        
        # Get selected connectivity method
        self.selected_class_name = self.class_info.get(methodName)
        selected_class = getattr(methods, self.selected_class_name, None)
        if self.init_method is not None:
            selected_class = self.init_method

        # If connectivity for this method already exists we load and plot it
        if self.selected_class_name in self.dfc_data_dict:
            self.dfc_data = self.dfc_data_dict[self.selected_class_name]
            self.plot_dfc()
            self.updateDistribution()
            self.plotTimeSeries()
            self.slider.show()
            self.calculatingLabel.setText(f"Loaded {self.selected_class_name} with shape {self.dfc_data['data'].shape}")
            print(f"Loaded {self.selected_class_name} from memory")

            # Update the slider
            total_length = self.dfc_data['data'].shape[2] if len(self.dfc_data['data'].shape) == 3 else 0
            position_text = f"t = {self.currentSliderValue} / {total_length-1}" if len(self.dfc_data['data'].shape) == 3 else " static "
            self.positionLabel.setText(position_text)
            self.slider.setValue(self.slider.value())
        
        # If connectivity data does not exist we reset the figure and slider to prepare for a new calculation
        # This also indicates to the user that this data was not yet calculated/saved
        else:
            self.figure.clear()
            self.plot_logo()
            self.canvas.draw()
            self.distributionFigure.clear()
            self.distributionCanvas.draw()
            self.timeSeriesFigure.clear()
            self.timeSeriesCanvas.draw()

            position_text = f"no data available"
            self.positionLabel.setText(position_text)
            self.slider.setValue(self.slider.value())
            self.slider.hide()

        # This dynamically creates the parameter labels and input boxes
        self.setup_class_parameters(selected_class)

        self.parameterLayout.addStretch(1) # Stretch to fill empty space
        self.update() # Update UI

    def setup_class_parameters(self, selected_class):
        # Now the parameter labels and boxes are set up    
        labels = []

        # Calculate the maximum label width (just a visual thing)
        max_label_width = 0
        init_signature = inspect.signature(selected_class.__init__)
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
        
        if self.init_method is None:
            self.time_series_textbox.setText(self.file_name)
            self.time_series_textbox.setEnabled(True)
        else:
            self.time_series_textbox.setText("from script")
            self.time_series_textbox.setEnabled(False)

        # Create info button for time_series
        time_series_info_text = "2D time series loaded from file. Time has to be the first dimension."
        time_series_info_button = InfoButton(time_series_info_text)

        time_series_layout = QHBoxLayout()
        time_series_layout.addWidget(time_series_label)
        time_series_layout.addWidget(self.time_series_textbox)
        time_series_layout.addWidget(time_series_info_button)
        self.parameterLayout.addLayout(time_series_layout)

        # Adjust max width for aesthetics
        max_label_width += 10
        time_series_label.setFixedWidth(max_label_width)

        existing_params = vars(selected_class)

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
                    
                    if self.init_method is None:
                        default_index = param_input_widget.findText(str(param.default))
                        param_input_widget.setCurrentIndex(default_index)
                        param_input_widget.setEnabled(True)
                    else:
                        default_index = param_input_widget.findText(str(existing_params[param.name]))
                        param_input_widget.setCurrentIndex(default_index)
                        param_input_widget.setEnabled(False)
                # Dropdown for parameters with predefined options
                elif param.name in selected_class.options:
                    param_input_widget = QComboBox()
                    param_input_widget.addItems(selected_class.options[param.name])
                    if param.default in selected_class.options[param.name]:
                        if self.init_method is None:
                            default_index = param_input_widget.findText(param.default)
                            param_input_widget.setCurrentIndex(default_index)
                            param_input_widget.setEnabled(True)
                        else:
                            param_input_widget.setCurrentIndex(str(existing_params[param.name]))
                            param_input_widget.setEnabled(False)    
                # Spinbox for integer parameterss
                elif type(param.default) == int:
                    param_input_widget = QSpinBox()
                    param_input_widget.setMaximum(10000)
                    param_input_widget.setMinimum(-10000)
                    param_input_widget.setSingleStep(1)
                    if self.init_method is None:
                        param_input_widget.setValue(int(param.default) if param.default != inspect.Parameter.empty else 0)
                        param_input_widget.setEnabled(True)
                    else:
                        param_input_widget.setValue(int(existing_params[param.name]))
                        param_input_widget.setEnabled(False)
                # Spinbox for float parameters
                elif type(param.default) == float:
                    param_input_widget = QDoubleSpinBox()
                    param_input_widget.setMaximum(10000.0)
                    param_input_widget.setMinimum(-10000.0)
                    param_input_widget.setSingleStep(0.1)
                    if self.init_method is None:
                        param_input_widget.setValue(float(param.default) if param.default != inspect.Parameter.empty else 0.0)
                        param_input_widget.setEnabled(True)
                    else: 
                        param_input_widget.setValue(float(existing_params[param.name]))
                        param_input_widget.setEnabled(False)
                # Text field for other types of parameters
                else:
                    if self.init_method is None:
                        param_input_widget = QLineEdit(str(param.default) if param.default != inspect.Parameter.empty else "")
                        param_input_widget.setEnabled(True)
                    else:
                        param_input_widget.setValue(str(existing_params[param.name]))
                        param_input_widget.setEnabled(False)

                # Create info button with tooltip
                info_text = self.getInfoText(param.name, self.selected_class_name)
                info_button = InfoButton(info_text)

                # Create layout for label, widget, and info button
                param_layout = QHBoxLayout()
                param_layout.addWidget(param_label)
                param_layout.addWidget(param_input_widget)
                param_layout.addWidget(info_button) 

                # Add the layout to the main parameter layout
                self.parameterLayout.addLayout(param_layout)

    def updateMethodComboBox(self):
        # Ordered lists as per 'shouldIncludeClass'
        continuous_classes = [
            'CONT Sliding Window', 'CONT Jackknife Correlation', 'CONT Dynamic Conditional Correlation', 
            'CONT Flexible Least Squares', 'CONT Spatial Distance', 'CONT Multiplication of Temporal Derivatives', 
            'CONT Phase Synchronization', 'CONT Leading Eigenvector Dynamics', 'CONT Wavelet Coherence', 'CONT Edge-centric Connectivity'
        ]

        state_based_classes = [
            'STATE Sliding Window Clustering', 'STATE Co-activation patterns', 'STATE Discrete Hidden Markov Model', 
            'STATE Continuous Hidden Markov Model', 'STATE Windowless'
        ]

        static_classes = [
            'STATIC Pearson Correlation', 'STATIC Partial Correlation', 'STATIC Mutual Information'
        ]

        # Concatenate the lists in the desired order
        ordered_classes = continuous_classes + state_based_classes + static_classes

        # Filter and order the class names based on the checkboxes
        filtered_and_ordered_classes = [
            class_name for class_name in ordered_classes if self.shouldIncludeClass(class_name) and class_name in self.class_info
        ]

        # Disconnect existing connections to avoid multiple calls
        try:
            self.methodComboBox.currentTextChanged.disconnect(self.onMethodChanged)
        except TypeError:
            pass

        # Update the combobox
        self.methodComboBox.clear()
        self.methodComboBox.addItems(filtered_and_ordered_classes)

        # Adjust combobox width to fit the longest option
        self.adjustComboBoxWidth()

        # Reconnect the signal
        self.methodComboBox.currentTextChanged.connect(self.onMethodChanged)

        # Optionally, trigger the onMethodChanged for the initial setup
        if filtered_and_ordered_classes:
            self.onMethodChanged(filtered_and_ordered_classes[0])

    def adjustComboBoxWidth(self):
        if self.methodComboBox.count() > 0:  # Check if the combobox has at least one item
            #font_metrics = QFontMetrics(self.methodComboBox.font())
            #longest_text_width = max(font_metrics.boundingRect(self.methodComboBox.itemText(i)).width() for i in range(self.methodComboBox.count()))
            #width = longest_text_width + 50
            width = 320
            self.methodComboBox.setFixedWidth(width)
        else:
            default_width = 320
            self.methodComboBox.setFixedWidth(default_width)

    def shouldIncludeClass(self, className):
        if self.continuousCheckBox.isChecked() and className.startswith("CONT"):
                return True
        if self.stateBasedCheckBox.isChecked() and className.startswith("STATE"):
                return True 
        if self.staticCheckBox.isChecked() and className.startswith("STATIC"):
                return True
        return False

    def onTabChanged(self):
        self.currentTabIndex = self.tabWidget.currentIndex()
        # index 0: Connectivity plot
        # index 1: Time series plot
        # index 2: Distribution plot
        # index 3: Graph analysis

        if self.dfc_data['data'] is None:
            self.plot_logo()
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

            if self.dfc_data['data'] is not None:
                total_length = self.dfc_data['data'].shape[2] if len(self.dfc_data['data'].shape) == 3 else 0
                position_text = f"t = {self.currentSliderValue} / {total_length-1}" if len(self.dfc_data['data'].shape) == 3 else " static "
            else:
                position_text = "no data available"

            self.positionLabel.setText(position_text)

        elif self.currentTabIndex == 1:
            self.backLargeButton.hide()
            self.backButton.hide()
            self.forwardButton.hide()
            self.forwardLargeButton.hide()
            
            # If we have nothing to scroll though, hide some GUI elements
            if len(self.dfc_data['data'].shape) == 2 or self.edge_ts is not None or self.state_tc is not None:
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
            if len(self.dfc_data['data'].shape) == 2 and self.edge_ts is None and self.state_tc is None:
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

    def showInfoPopup(self, param_name):
        info_text = f"Information about {param_name}"  # Replace with actual information
        QMessageBox.information(self, "Parameter Information", info_text)

    def clearLayout(self, layout):
        while layout.count():
            item = layout.takeAt(0)  # Take the first item from the layout
            if item.widget():  # If the item is a widget
                widget = item.widget()
                if widget is not None and widget is not self.time_series_textbox:
                    widget.deleteLater()  # Schedule the widget for deletion
            elif item.layout():  # If the item is a layout
                self.clearLayout(item.layout())  # Recursively clear the layout
                item.layout().deleteLater()  # Delete the layout itself
            elif item.spacerItem():  # If the item is a spacer
                # No need to delete spacer items; they are automatically handled by Qt
                pass
    
    def loadFile(self):
        fileFilter = "All Supported Files (*.mat *.txt *.npy *pkl *tsv);;MAT Files (*.mat);;Text Files (*.txt);;NumPy Files (*.npy);;Pickle Files (*.pkl);;TSV Files (*.tsv))"
        file_path, _ = QFileDialog.getOpenFileName(self, "Load File", "", fileFilter)
        self.file_name = file_path.split('/')[-1]

        if not file_path:
            return  # Early exit if no file is selected

        try:
            if file_path.endswith('.mat'):
                self.ts_data = loadmat(file_path)  # Assuming you'll adjust how to extract the array
            elif file_path.endswith('.txt'):
                self.ts_data = np.loadtxt(file_path)
            elif file_path.endswith('.npy'):
                self.ts_data = np.load(file_path)
            elif file_path.endswith('.pkl'):
                with open(file_path, 'rb') as f:
                    self.ts_data = pickle.load(f)
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
                self.ts_data = data.to_numpy()

                # Update header_list if rois exist
                self.roi_data = np.array(rois, dtype=object)

            else:
                self.ts_data = None
                self.time_series_textbox.setText("Unsupported file format")

            # New data, reset slider and plot
            self.currentSliderValue = 0
            self.slider.setValue(0)
            self.figure.clear()
            self.canvas.draw()

            # Set filenames depending on file type
            if file_path.endswith('.pkl'):
                self.fileNameLabel.setText(f"Loaded TIME_SERIES object")
                self.time_series_textbox.setText(self.file_name)

                self.continuousCheckBox.setEnabled(False)
                self.continuousCheckBox.setChecked(False)

                self.stateBasedCheckBox.setEnabled(True)
                self.stateBasedCheckBox.setChecked(True)

                self.staticCheckBox.setEnabled(False)
                self.staticCheckBox.setChecked(False)

                self.reshapeCheckbox.setEnabled(False)
            else: 
                self.fileNameLabel.setText(f"Loaded {self.file_name} with shape {self.ts_data.shape}")
                self.time_series_textbox.setText(self.file_name)

                self.continuousCheckBox.setEnabled(True)
                self.continuousCheckBox.setChecked(True)

                self.stateBasedCheckBox.setEnabled(False)
                self.stateBasedCheckBox.setChecked(False)

                self.staticCheckBox.setEnabled(True)
                self.staticCheckBox.setChecked(True)
      
                self.reshapeCheckbox.setEnabled(True)

            # Show transpose textbox
            self.reshapeCheckbox.show()

            # Reset and enable the GUI elements
            self.methodComboBox.setEnabled(True)
            #self.methodComboBox.setCurrentText("Sliding Window")
            self.init_method = None
            #self.onMethodChanged()

            self.methodComboBox.setEnabled(True)
            self.calculateButton.setEnabled(True)
            self.clearMemoryButton.setEnabled(True)
            self.keepInMemoryCheckbox.setEnabled(True)

        except Exception as e:
            print(f"Error loading data: {e}")
            self.fileNameLabel.setText(f"Error. No time series data has been loaded.")

    def saveFile(self):
        if not hasattr(self, 'dfc_data'):
            print("No dFC data available to save.")
            return

        # Open a file dialog to specify where to save the file
        filePath, _ = QFileDialog.getSaveFileName(self, "Save File", "", "MAT Files (*.mat)")
        
        if filePath:
            # Ensure the file has the correct extension
            if not filePath.endswith('.mat'):
                filePath += '.mat'
            
            # Save the data
            try:
                savemat(filePath, {'dfc_data': self.dfc_data['data'], 'roi_data': self.roi_data})
            except Exception as e:
                print(f"Error saving data: {e}")

    def onReshapeCheckboxChanged(self, state):
        if self.ts_data is None:
            return  # No data loaded, so do nothing

        if state == Qt.CheckState.Checked:
            # Transpose the data
            self.ts_data = self.ts_data.transpose()
        else:
            # Transpose it back to original
            self.ts_data = self.ts_data.transpose()

        # Update the labels
        self.fileNameLabel.setText(f"Loaded {self.time_series_textbox.text()} with shape: {self.ts_data.shape}")
        self.time_series_textbox.setText(self.file_name)

    def handleResult(self, result):
        # Handles the result of the worker thread
        self.dfc_data = result

        #if self.abortFlag:
        #    print("Calculation was aborted, no results will be shown.")
        #    self.abortFlag = False
        #    return

        # Update the sliders and text
        if self.dfc_data['data'] is not None:
            self.calculatingLabel.setText(f"Calculated {self.selected_class_name} with shape {self.dfc_data['data'].shape}")
            
            if len(self.dfc_data['data'].shape) == 3:
                self.slider.show()
                self.rowSelector.setMaximum(self.dfc_data['data'].shape[0] - 1)
                self.colSelector.setMaximum(self.dfc_data['data'].shape[1] - 1)
                self.rowSelector.setValue(1)

            # Update time label
            total_length = self.dfc_data['data'].shape[2] if len(self.dfc_data['data'].shape) == 3 else 0
            
            if self.currentTabIndex == 0 or self.currentTabIndex == 2:
                position_text = f"t = {self.currentSliderValue} / {total_length-1}" if len(self.dfc_data['data'].shape) == 3 else " static "
            else:
                position_text = ""

            self.positionLabel.setText(position_text)
            self.slider.setValue(self.slider.value())
            
        # Plot
        self.plot_dfc()
        self.updateDistribution()
        self.plotTimeSeries()
        self.calculateButton.setEnabled(True)
        self.onTabChanged()

    def handleError(self, error):
        # Handles errors in the worker thread
        print(f"Error occurred: {error}")

    def getParameters(self):
        # Get the time series and parameters (from the UI) for the selected connectivity method and store them in a dictionary
        self.parameters = {}
        self.parameters['time_series'] = self.ts_data

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
                        self.parameters[param_key] = convert_value(value) if isinstance(value, str) else value
                    else:
                        self.calculatingLabel.setText(f"Error: Unrecognized parameter '{label}'")
                else:
                    # Value could not be retrieved from the widget
                    self.calculatingLabel.setText(f"Error: No value entered for parameter '{label}'")

    def onCalculateConnectivity(self):
        # Check if ts_data is available
        if self.ts_data is None:
            self.calculatingLabel.setText(f"Error. No time series data has been loaded.")
            return
    
        # Check if method is available
        selected_class = getattr(methods, self.selected_class_name, None)
        if not selected_class:
            print("Selected class not found in connectivity module")
            return
        
        # Process all pending events
        QApplication.processEvents() 

        # Gets the parameters and stores them in self.parameters
        self.getParameters()
        
        # Start worker thread for dFC calculations and submit for calculation
        self.workerThread = QThread()
        self.worker = Worker(self.calculateDFC, self.parameters)
        self.worker.moveToThread(self.workerThread)
        
        self.worker.finished.connect(self.workerThread.quit)
        self.worker.result.connect(self.handleResult)
        self.worker.error.connect(self.handleError)

        self.workerThread.started.connect(self.worker.run)
        self.workerThread.start()
        self.calculatingLabel.setText(f"Calculating {self.methodComboBox.currentText()}, please wait...")
        self.calculateButton.setEnabled(False)
    
    def check_data_dict_equality(self, dict1, dict2):
        if dict1.keys() != dict2.keys():
            return False  # The dictionaries have different sets of keys
        
        for key in dict1:
            val1, val2 = dict1[key], dict2[key]
            
            # If both values are numpy arrays, use numpy.array_equal
            if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
                if not np.array_equal(val1, val2):
                    return False
            # If values are dictionaries, recursively compare
            elif isinstance(val1, dict) and isinstance(val2, dict):
                if not self.check_data_dict_equality(val1, val2):
                    return False
            # For other types, use standard equality check
            else:
                if val1 != val2:
                    return False

        return True  # All keys and values are equal

    def calculateDFC(self, parameters):
        keep_in_memory = self.keepInMemoryCheckbox.isChecked()
        
        # Try to calculate dFC, throw an exception if  it fails
        try:
            # Check if data for the selected class name exists with the same parameters
            existing_data = self.dfc_data_dict.get(self.selected_class_name) # returns None if key does not exist (avoid KeyError)
            #if existing_data is not None and existing_data.get('data') is not None and existing_data.get('data').size > 0 and self.check_data_dict_equality(existing_data.get('parameters'), parameters):
            if existing_data is not None and existing_data.get('data') is not None and self.check_data_dict_equality(existing_data.get('parameters'), parameters):
                print(f"Using stored data for {self.selected_class_name} with given parameters")
                return {"data": existing_data['data'], "parameters": parameters}

            # If parameters have changed or data doesn't exist, proceed to calculate new data
            selected_class = getattr(methods, self.selected_class_name, None)
            if not selected_class:
                print("Selected class not found in connectivity module")
                return None

            connectivity_calculator = selected_class(**parameters)
            result = connectivity_calculator.connectivity()
            
            # In case the method returns multiple values. The first one is always the NxNxT dfc matrix
            if isinstance(result, tuple):
                self.dfc_data['data'], _ = result
                self.dfc_data['parameters'] = parameters
                self.state_tc = None
                self.edge_ts = result[1][0] if isinstance(result[1], tuple) else None

            # Result is DFC object (pydfc methods)
            elif isinstance(result, pydfc.dfc.DFC):
                self.dfc_data['data'] = np.transpose(result.get_dFC_mat(), (1, 2, 0))
                self.dfc_data['parameters'] = parameters
                self.dfc_states = result.FCSs
                self.state_tc = result.state_TC()
                self.edge_ts = None
            
            # Only a single matrix is returned (most cases)
            else:
                self.dfc_data['data'] = result
                self.dfc_data['parameters'] = parameters
                self.state_tc = None
                self.edge_ts = None

            # Store in memory if checkbox is checked
            if keep_in_memory:
                # Update the dictionary entry for the selected_class_name with the new data and parameters
                self.dfc_data_dict[self.selected_class_name] = {'data': self.dfc_data['data'], 'parameters': parameters}
                print(f"Updated {self.selected_class_name} data with new parameters in memory")

            return self.dfc_data

        except Exception as e:
            print(f"Exception when calculating dFC: {e}")
            self.calculatingLabel.setText(f"Error calculating connectivity, check parameters.")
            return None

    def saveDataToDict(self, parameters):
        self.dfc_data_dict[self.selected_class_name] = {'data': self.dfc_data['data'], 'parameters': parameters}
        return

    def onKeepInMemoryChanged(self, state):
        if state == 2 and self.dfc_data['data'] is not None:
            self.saveDataToDict(self.parameters)
            print(f"Saved {self.selected_class_name} data to memory")
                
    def onClearMemory(self):
        self.dfc_data_dict = {}
        
        self.figure.clear()
        self.canvas.draw()
        self.distributionFigure.clear()
        self.distributionCanvas.draw()

        self.calculatingLabel.setText(f"Cleared memory")
        print("Cleared memory")

        return

    def plot_dfc(self):
        if not hasattr(self, 'dfc_data'):
            print("No calculated data available for plotting")
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if self.dfc_data['data'] is not None:
            current_data = self.dfc_data['data']
            try:
                current_slice = current_data[:, :, self.currentSliderValue] if len(current_data.shape) == 3 else current_data
                vmax = np.max(np.abs(current_slice))
                self.im = ax.imshow(current_slice, cmap='coolwarm', vmin=-vmax, vmax=vmax)
            except:
                current_slice = current_data[:, :, 0] if len(current_data.shape) == 3 else current_data
                vmax = np.max(np.abs(current_slice))
                self.im = ax.imshow(current_slice, cmap='coolwarm', vmin=-vmax, vmax=vmax)

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
        current_data = self.dfc_data['data']
        # Get dimensions of the data
        if current_data is not None and current_data.ndim == 3:
            self.rowSelector.setMaximum(current_data.shape[0] - 1)
            self.colSelector.setMaximum(current_data.shape[1] - 1)

        row = self.rowSelector.value()
        col = self.colSelector.value()
        self.rowSelector.show()
        self.colSelector.show()

        if current_data is not None and row < current_data.shape[0] and col < current_data.shape[1] and self.edge_ts is None and self.state_tc is None:    
            self.timeSeriesFigure.clear()
            ax = self.timeSeriesFigure.add_subplot(111)
            time_series = current_data[row, col, :] if len(current_data.shape) == 3 else current_data[row, col]
            ax.set_title(f"dFC time course between region {row} and {col}.")
            
            ax.plot(time_series)
            self.timeSeriesCanvas.draw()

        elif self.state_tc is not None:
            self.timeSeriesFigure.clear()

            time_series = self.state_tc
            num_states = len(self.dfc_states)
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
            for col, (state, matrix) in enumerate(self.dfc_states.items()):
                ax_state = self.timeSeriesFigure.add_subplot(gs[2, col])
                ax_state.imshow(matrix, cmap='coolwarm', aspect=1)
                ax_state.set_title(f"State {col+1}")
                ax_state.set_xticks([])
                ax_state.set_yticks([]) 

            self.timeSeriesFigure.canvas.draw()

        elif self.edge_ts is not None:
            self.timeSeriesFigure.clear()
            gs = gridspec.GridSpec(3, 1, self.timeSeriesFigure, height_ratios=[2, 0.5, 1]) # GridSpec with 3 rows and 1 column

            # The first subplot occupies the 1st row
            ax1 = self.timeSeriesFigure.add_subplot(gs[:1, 0])
            ax1.imshow(self.edge_ts.T, cmap='coolwarm', aspect='auto', vmin=-1*self.parameters["vlim"], vmax=self.parameters["vlim"])
            ax1.set_title("Edge time series")
            ax1.set_xlabel("Time (TRs)")
            ax1.set_ylabel("Edges")

            # The second subplot occupies the 3rd row
            ax2 = self.timeSeriesFigure.add_subplot(gs[2, 0])
            mean_edge_values = np.mean(self.edge_ts.T, axis=0)
            ax2.plot(mean_edge_values)
            ax2.set_xlim(0, len(mean_edge_values) - 1)
            ax2.set_title("Mean time series")
            ax2.set_xlabel("Time (TRs)")
            ax2.set_ylabel("Mean Edge Value")
            
            self.timeSeriesFigure.canvas.draw()
        
        else:
            # Clear the plot if the data is not available
            self.timeSeriesFigure.clear()
            self.timeSeriesCanvas.draw()

    def plot_logo(self):
        with pkg_resources.path("comet.resources.img", "logo.png") as file_path:
            logo = imread(file_path)

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_axis_off()
        self.im = ax.imshow(logo)

        self.figure.set_facecolor('#f4f1f6')
        self.figure.tight_layout()
        self.canvas.draw()

    def updateDistribution(self):
        current_data = self.dfc_data['data']

        if current_data is None or not hasattr(self, 'distributionFigure'):
            self.distributionFigure.clear()
            return

        # Clear the current distribution plot
        self.distributionFigure.clear()

        # Assuming you want to plot the distribution of values in the current slice
        current_slice = current_data[:, :, self.slider.value()] if len(current_data.shape) == 3 else current_data
        ax = self.distributionFigure.add_subplot(111)
        ax.hist(current_slice.flatten(), bins=50)  # number of bins

        self.distributionCanvas.draw()

    def onSliderValueChanged(self, value):
        # Ensure there is data to work with
        if self.dfc_data['data'] is None or self.im is None:
            return
        
        if self.currentTabIndex == 0 or self.currentTabIndex == 2:
            # Get and update the data of the imshow object
            self.currentSliderValue = value
            data = self.dfc_data['data']
            self.im.set_data(data[:, :, value]) if len(data.shape) == 3 else self.im.set_data(data)

            vlim = np.max(np.abs(data[:, :, value])) if len(data.shape) == 3 else np.max(np.abs(data))
            self.im.set_clim(-vlim, vlim)

            # Redraw the canvas
            self.canvas.draw()
            self.updateDistribution()

            total_length = self.dfc_data['data'].shape[2] if len(self.dfc_data['data'].shape) == 3 else 0
            position_text = f"t = {self.currentSliderValue} / {total_length-1}" if len(self.dfc_data['data'].shape) == 3 else " static "
            self.positionLabel.setText(position_text)

    
        elif self.currentTabIndex == 1:
            self.updateTimeSeriesPlot(value)

    def moveBack(self):
        # Move slider back by 1
        self.updateSliderValue(-1)

    def moveForward(self):
        # Move slider forward by 1
        self.updateSliderValue(1)

    def moveBackLarge(self):
        # Move slider back by 10
        self.updateSliderValue(-10)

    def moveForwardLarge(self):
        # Move slider forward by 10
        self.updateSliderValue(10)

    def updateSliderValue(self, delta):
        # Update the slider value considering its range
        self.currentSliderValue = max(0, min(self.slider.value() + delta, self.slider.maximum()))
        self.slider.setValue(self.currentSliderValue)

        self.plot_dfc()
        self.updateDistribution()

    def updateTimeSeriesPlot(self, center):
        if self.dfc_data['data'] is None:
            return

        max_index = self.dfc_data['data'].shape[2] - 1 if len(self.dfc_data['data'].shape) == 3 else 0
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
        #ax.set_title(f"dFC time series ({row}, {col})")
        self.timeSeriesCanvas.draw()

def run(dfc_data=None, method=None):
    app = QApplication(sys.argv)

    # Set global stylesheet for tooltips
    QApplication.instance().setStyleSheet("""
        QToolTip {
            background-color: #E0E0E0;
            border: 1px solid black;
        }
    """)
    ex = App(init_data=dfc_data, init_method=method)
    ex.setStyleSheet(qdarkstyle.load_stylesheet_pyqt6())
    ex.show()

    try:
        sys.exit(app.exec())
    except SystemExit as e:
        print(f"GUI closed with status {e}")

if __name__ == '__main__':
    run()
