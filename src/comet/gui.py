import sys
import inspect
import numpy as np
from scipy.io import loadmat, savemat
import importlib.resources as pkg_resources
from matplotlib.image import imread

from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QEnterEvent, QFontMetrics
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QSlider, QToolTip, QWidget, \
     QLabel, QFileDialog, QComboBox, QLineEdit, QSizePolicy, QSpacerItem, QCheckBox, QTabWidget, QMessageBox, QSpinBox, QDoubleSpinBox
import qdarkstyle

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from . import methods

class InfoButton(QPushButton):
    def __init__(self, info_text, parent=None):
        super().__init__("i", parent)
        self.info_text = info_text
        self.setStyleSheet("QPushButton { border: 1px solid black;}")
        self.setFixedSize(20, 20)

    def enterEvent(self, event: QEnterEvent):
        # Adjust the position to show the tooltip right next to the button
        tooltip_pos = self.mapToGlobal(QPoint(self.width(), 0))
        QToolTip.showText(tooltip_pos, self.info_text)
        super().enterEvent(event)

class App(QMainWindow):
    def __init__(self, init_data=None, init_method=None):
        super().__init__()
        self.title = 'Comet Dynamic Functional Connectivity Toolbox'
        self.ts_data = None
        self.dfc_data = None
        self.init_method = init_method
        self.dfc_data_dict = {}
        self.selected_class_name = None
        self.currentSliderValue = 0
        self.file_name = ""
        self.param_names = {
            "self":              "self", 
            "time_series":       "Time series",
            "windowsize":        "Window size",
            "shape":             "Window shape",
            "std":               "Window sigma",
            "diagonal":          "Main diagonal",
            "fisher_z":          "Fisher z-transform",
            "num_cores":         "Number of CPU cores",
            "standardizeData":   "Z-score data",
            "mu":                "Weighting parameter μ",
            "flip_eigenvectors": "Flip eigenvectors",
            "crp":               "Cosine of rel. phase",
            "pcoh":              "Phase coherence",
            "teneto":            "Teneto implementation",
            "dist":              "Distance function",
            "weighted":          "Weighted average",
            "TR":                "Repetition Time",
            "fmin":              "Minimum frequency",
            "fmax":              "Maximum freqency",
            "n_scales":          "Number of scales",
            "drop_scales":       "Drop n scales",
            "drop_timepoints":   "Drop n timepoints",
            "standardize":       "Z-score connectivity",
            "tril":              "Extract triangle",
            "method":            "Method"
        }
        self.reverse_param_names = {v: k for k, v in self.param_names.items()}

        self.initUI()
        
        self.dfc_data = init_data
        if self.dfc_data is not None:
            self.init_from_calculated_data()

    def initUI(self):
        self.setWindowTitle(self.title)
        mainLayout = QHBoxLayout()

        ###############################
        #  Left section for settings  #
        ###############################
        leftLayout = QVBoxLayout()

        # Create button and label for file loading
        self.fileButton = QPushButton('Load File')
        self.fileNameLabel = QLabel('No file loaded')
        leftLayout.addWidget(self.fileButton)
        leftLayout.addWidget(self.fileNameLabel)
        self.fileButton.clicked.connect(self.loadFile)

        # Create a checkbox for reshaping the data
        self.reshapeCheckbox = QCheckBox("Transpose")
        leftLayout.addWidget(self.reshapeCheckbox)
        self.reshapeCheckbox.hide()

        # Connect the checkbox to a method
        self.reshapeCheckbox.stateChanged.connect(self.onReshapeCheckboxChanged)

        # Add spacer for an empty line
        leftLayout.addItem(QSpacerItem(0, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))

        # Method label and combobox
        self.methodLabel = QLabel("Dynamic functional connectivity method:")
        self.methodComboBox = QComboBox()
        leftLayout.addWidget(self.methodLabel)
        leftLayout.addWidget(self.methodComboBox)

        # Retrieve class names and their human-readable names
        self.class_info = {
            obj.name: name  # Map human-readable name to class name
            for name, obj in inspect.getmembers(methods)
            if inspect.isclass(obj) and obj.__module__ == methods.__name__ and name != "ConnectivityMethod"
        }

        self.methodComboBox.addItems(self.class_info.keys())
        self.methodComboBox.setCurrentText("Sliding Window")
        self.methodComboBox.currentTextChanged.connect(self.onMethodChanged)

        # Create a layout for dynamic textboxes
        self.parameterLayout = QVBoxLayout()

        # Create a container widget for the parameter layout
        parameterContainer = QWidget()
        parameterContainer.setLayout(self.parameterLayout)
        parameterContainer.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        # Add the container widget to the left layout directly below the combobox
        leftLayout.addWidget(parameterContainer)

        # Add a stretch after the parameter layout container
        leftLayout.addStretch()

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
        leftLayout.addLayout(buttonsLayout)

        # Memory buttons
        self.keepInMemoryCheckbox = QCheckBox("Keep in memory")
        self.keepInMemoryCheckbox.stateChanged.connect(self.onKeepInMemoryChanged)
        self.clearMemoryButton = QPushButton("Clear Memory")
        self.clearMemoryButton.clicked.connect(self.onClearMemory)

        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(self.keepInMemoryCheckbox)
        buttonLayout.addWidget(self.clearMemoryButton)

        # Assuming you have a QVBoxLayout named 'leftLayout'
        leftLayout.addLayout(buttonLayout)

        # Calculation info textbox
        self.calculatingLabel = QLabel('No data calculated yet')
        leftLayout.addWidget(self.calculatingLabel)

        
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

        # Tab 2: Distribution plot
        distributionTab = QWidget()
        distributionLayout = QVBoxLayout()
        distributionTab.setLayout(distributionLayout)

        self.distributionFigure = Figure()
        self.distributionCanvas = FigureCanvas(self.distributionFigure)
        self.distributionFigure.patch.set_facecolor('#E0E0E0')
        distributionLayout.addWidget(self.distributionCanvas)
        self.tabWidget.addTab(distributionTab, "Distribution")

        # Tab 3: Time Series plot
        timeSeriesTab = QWidget()
        timeSeriesLayout = QVBoxLayout()
        timeSeriesTab.setLayout(timeSeriesLayout)

        self.timeSeriesFigure = Figure()
        self.timeSeriesCanvas = FigureCanvas(self.timeSeriesFigure)
        self.timeSeriesFigure.patch.set_facecolor('#E0E0E0')
        timeSeriesLayout.addWidget(self.timeSeriesCanvas)
        self.tabWidget.addTab(timeSeriesTab, "Time series")

        rightLayout.addWidget(self.tabWidget)

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
        self.positionLabel = QLabel('t = 0 / 0')
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
        self.onMethodChanged("Sliding Window")

        # UI elements for dFC time series plotting
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

        timeSeriesLayout.addLayout(timeSeriesSelectorLayout)

        #####################
        #  Combine layouts  #
        #####################
        mainLayout.addLayout(leftLayout, 1)
        mainLayout.addLayout(rightLayout, 2)

        # Set main window layout
        centralWidget = QWidget()
        centralWidget.setLayout(mainLayout)
        self.setCentralWidget(centralWidget)

    def init_from_calculated_data(self):
        # Make sure both the dFC data and the method object are provided
        assert self.init_method is not None, "Please provide the method object corresponding to your dFC data as the second argument to the GUI."

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

        self.rowSelector.setMaximum(1)
        self.rowSelector.setValue(1)
        self.plotTimeSeries()

    def onMethodChanged(self, methodName):
        # Clear old variables and data
        self.clearLayout(self.parameterLayout)
        #self.dfc_data = None
        
        # Get selected connectivity method
        self.selected_class_name = self.class_info.get(methodName)
        selected_class = getattr(methods, self.selected_class_name, None)
        
        if self.init_method is not None:
            selected_class = self.init_method
        
        # If connectivity for this method already exists we load and plot ot
        if self.selected_class_name in self.dfc_data_dict:
            self.dfc_data = self.dfc_data_dict[self.selected_class_name]
            self.plot_dfc()
            self.updateDistribution()
            self.calculatingLabel.setText(f"Loaded {self.selected_class_name} with shape {self.dfc_data.shape}")
            print(f"Loaded {self.selected_class_name} from memory")

            # Update the slider
            total_length = self.dfc_data.shape[2]
            position_text = f"t = {self.currentSliderValue} / {total_length-1}"
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

            position_text = f"t = 0 / 0"
            self.positionLabel.setText(position_text)
            self.slider.setValue(self.slider.value())

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

        # Add parameter textbox for time_series
        self.time_series_textbox = QLineEdit()
        self.time_series_textbox.setReadOnly(True) # read only as based on the loaded file
        
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
        max_label_width += 20
        time_series_label.setFixedWidth(max_label_width)

        existing_params = vars(selected_class)

        for param in init_signature.parameters.values():
            if param.name not in ['self', 'time_series', 'tril', 'standardize']:
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

    def onTabChanged(self, index):
        if index == 0 or index == 1:
            self.slider.setValue(self.currentSliderValue)
            self.backLargeButton.show()
            self.backButton.show()
            self.forwardButton.show()
            self.forwardLargeButton.show()

            if self.dfc_data is not None:
                total_length = self.dfc_data.shape[2]  # Assuming dfc_data is 3D
                position_text = f"t = {self.currentSliderValue} / {total_length-1}"
            else:
                position_text = "t = 0 / 0"
            self.positionLabel.setText(position_text)

        elif index == 2:
            self.backLargeButton.hide()
            self.backButton.hide()
            self.forwardButton.hide()
            self.forwardLargeButton.hide()
            self.slider.setValue(0)

            position_text = f"Use the slider to zoom in and scroll through the time series"
            self.positionLabel.setText(position_text)

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
        
        return text

    def showInfoPopup(self, param_name):
        info_text = f"Information about {param_name}"  # Replace with actual information
        QMessageBox.information(self, "Parameter Information", info_text)

    def clearLayout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                widget = item.widget()
                widget.setParent(None)  # Detach the widget from its parent
                widget.deleteLater()    # Schedule the widget for deletion
            elif item.layout():       # If the item is a layout, clear it recursively
                self.clearLayout(item.layout())
    
    def loadFile(self):
        fileFilter = "All Supported Files (*.mat *.txt *.npy);;MAT Files (*.mat);;Text Files (*.txt);;NumPy Files (*.npy)"
        file_path, _ = QFileDialog.getOpenFileName(self, "Load File", "", fileFilter)
        self.file_name = file_path.split('/')[-1]

        try:
            if file_path.endswith('.mat'):
                self.ts_data = loadmat(file_path)  # Assuming you'll adjust how to extract the array
            elif file_path.endswith('.txt'):
                self.ts_data = np.loadtxt(file_path)
            elif file_path.endswith('.npy'):
                self.ts_data = np.load(file_path)
            else:
                self.ts_data = None
                self.time_series_textbox.setText("Unsupported file format")

            # New data, reset slider and plot
            self.currentSliderValue = 0
            self.slider.setValue(0)
            self.figure.clear()
            self.canvas.draw()

            # Set the filename in the text box
            self.fileNameLabel.setText(f"Loaded {self.file_name} with shape {self.ts_data.shape}")
            self.time_series_textbox.setText(self.file_name)

            # Show transpose textbox
            self.reshapeCheckbox.show()

            # Reset and enable the GUI elements
            self.methodComboBox.setEnabled(True)
            self.methodComboBox.setCurrentText("Sliding Window")
            self.init_method = None
            self.onMethodChanged("Sliding Window")

            self.methodComboBox.setEnabled(True)
            self.calculateButton.setEnabled(True)
            self.clearMemoryButton.setEnabled(True)
            self.keepInMemoryCheckbox.setEnabled(True)

        except Exception as e:
            print(f"Error loading data: {e}")
            self.fileNameLabel.setText(f"Error. No time series data has been loaded.")

    def saveFile(self):
        if not hasattr(self, 'dfc_data'):
            print("No DFC data available to save.")
            return

        # Open a file dialog to specify where to save the file
        filePath, _ = QFileDialog.getSaveFileName(self, "Save File", "", "MAT Files (*.mat)")
        
        if filePath:
            # Ensure the file has the correct extension
            if not filePath.endswith('.mat'):
                filePath += '.mat'
            
            # Save the data
            try:
                savemat(filePath, {'dfc_data': self.dfc_data})
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

        # Update the shape label
        self.fileNameLabel.setText(f"Loaded {self.time_series_textbox.text()} with shape: {self.ts_data.shape}")

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
        
        # Perform main calculations
        self.calculatingLabel.setText(f"Calculating {self.methodComboBox.currentText()}, please wait...")
        QApplication.processEvents()

        # Get the time series and parameters (from the UI) for the selected connectivity method
        parameters = {}
        parameters['time_series'] = self.ts_data

        for i in range(self.parameterLayout.count()):
            layout = self.parameterLayout.itemAt(i).layout()
            if layout:
                label = layout.itemAt(0).widget().text()[:-1]  # Remove the colon at the end
                if label == 'Time series':  # Skip 'time_series' from UI collection as this contains te data
                    continue

                # This might be the worst piece of code I've ever written, need do rework this
                # Convert the parameter strings from the UI into their intended data types
                widget = layout.itemAt(1).widget()
                if isinstance(widget, QLineEdit):
                    value = widget.text()
                elif isinstance(widget, QComboBox):
                    value = widget.currentText()
                elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    value = widget.value()

                if isinstance(value, str):
                    if value.lower() in ['true', 'false']:
                        parameters[self.reverse_param_names[label]] = value.lower() == 'true'
                    else:
                        try:
                            parameters[self.reverse_param_names[label]] = float(value)
                        except ValueError:
                            parameters[self.reverse_param_names[label]] = value
                else:
                    parameters[self.reverse_param_names[label]] = value
                    self.calculatingLabel.setText(f"Error: No value entered for parameter '{self.reverse_param_names[label]}'")

        self.dfc_data = self.manageMemory(parameters)

        if self.dfc_data is not None:
            self.calculatingLabel.setText(f"Calculated {self.selected_class_name} with shape {self.dfc_data.shape}")
            
            # Update time label
            total_length = self.dfc_data.shape[2]  # Assuming dfc_data is 3D
            position_text = f"t = {self.currentSliderValue} / {total_length-1}"
            self.positionLabel.setText(position_text)
            self.slider.setValue(self.slider.value())
            
        # Plot
        self.plot_dfc()
        self.updateDistribution()
        self.plotTimeSeries()

    def manageMemory(self, parameters):
        keep_in_memory = self.keepInMemoryCheckbox.isChecked()
        
        try:
            if self.selected_class_name in self.dfc_data_dict:
                # Use stored data
                return self.dfc_data_dict[self.selected_class_name]
            else:
                # Calculate new data
                selected_class = getattr(methods, self.selected_class_name, None)
                if not selected_class:
                    print("Selected class not found in connectivity module")
                    return None

                connectivity_calculator = selected_class(**parameters)
                result = connectivity_calculator.connectivity() 
                
                # In case the method returns multiple values. The first one is always the NxNxT dfc matrix
                if isinstance(result, tuple):
                    self.dfc_data, _ = result
                else:
                    self.dfc_data = result

                
                # Store in memory if checkbox is checked
                if keep_in_memory:
                    self.dfc_data_dict[self.selected_class_name] = self.dfc_data
                    print(f"Saved {self.selected_class_name} to memory")

                return self.dfc_data
        except Exception as e:
            print(f"Hi, Exception: {e}")
            self.calculatingLabel.setText(f"Error calculating connectivity, check parameters.")
            return None
            
    def onKeepInMemoryChanged(self, state):
        if state == 2 and self.dfc_data is not None:
            print(f"Saved {self.selected_class_name} data to memory")
            self.dfc_data_dict[self.selected_class_name] = self.dfc_data
                
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

        if self.dfc_data is not None:
            try:
                current_slice = self.dfc_data[:, :, self.currentSliderValue]
                vmax = np.max(np.abs(current_slice))
                self.im = ax.imshow(current_slice, cmap='coolwarm', vmin=-vmax, vmax=vmax)
            except:
                current_slice = self.dfc_data[:, :, 0]
                vmax = np.max(np.abs(current_slice))
                self.im = ax.imshow(current_slice, cmap='coolwarm', vmin=-vmax, vmax=vmax)

            # Create the colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.15)
            cbar = self.figure.colorbar(self.im, cax=cax)
            cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1f}'))

            self.slider.setMaximum(self.dfc_data.shape[2] - 1)
        
        self.figure.set_facecolor('#E0E0E0')
        self.figure.tight_layout()
        self.canvas.draw()

    def plotTimeSeries(self):
        # Get dimensions of the data
        if self.dfc_data is not None and self.dfc_data.ndim == 3:
            self.rowSelector.setMaximum(self.dfc_data.shape[0] - 1)
            self.colSelector.setMaximum(self.dfc_data.shape[1] - 1)

        row = self.rowSelector.value()
        col = self.colSelector.value()

        if self.dfc_data is not None and row < self.dfc_data.shape[0] and col < self.dfc_data.shape[1]:
            time_series = self.dfc_data[row, col, :]

            self.timeSeriesFigure.clear()
            ax = self.timeSeriesFigure.add_subplot(111)
            ax.plot(time_series)
            ax.set_title(f"dFC time course between region {row} and {col}.")
            self.timeSeriesCanvas.draw()
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
        if self.dfc_data is None or not hasattr(self, 'distributionFigure'):
            return

        # Clear the current distribution plot
        self.distributionFigure.clear()

        # Assuming you want to plot the distribution of values in the current slice
        current_slice = self.dfc_data[:, :, self.slider.value()]
        ax = self.distributionFigure.add_subplot(111)
        ax.hist(current_slice.flatten(), bins=60)  # Adjust the number of bins as needed

        self.distributionCanvas.draw()

    def onSliderValueChanged(self, value):
        # Ensure there is data to work with
        if self.dfc_data is None or self.im is None:
            return
        
        currentTabIndex = self.tabWidget.currentIndex()
        
        if currentTabIndex == 0 or currentTabIndex == 1:
            # Get and update the data of the imshow object
            self.currentSliderValue = value
            data = self.dfc_data
            self.im.set_data(data[:, :, value])

            vlim = np.max(np.abs(data[:, :, value]))
            self.im.set_clim(-vlim, vlim)

            # Redraw the canvas
            self.canvas.draw()
            self.updateDistribution()

            total_length = self.dfc_data.shape[2]
            position_text = f"t = {value} / {total_length-1}"
            self.positionLabel.setText(position_text)

            currentTabIndex = self.tabWidget.currentIndex()
    
        elif currentTabIndex == 2:
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
        if self.dfc_data is None:
            return

        max_index = self.dfc_data.shape[2] - 1
        width = 101

        # Determine if we should show the entire series or a window
        if center == 0 or center == max_index:
            start = 0
            end = max_index
        else:
            start = max(0, center - width // 2)
            end = min(self.dfc_data.shape[2], center + width // 2)

        row = self.rowSelector.value()
        col = self.colSelector.value()
        time_series_slice = self.dfc_data[row, col, start:end]

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