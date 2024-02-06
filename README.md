## Comet - A dynamic functional connectivity toolbox for multiverse analysis
[![DOI](src/comet/resources/img/badge.svg)](https://doi.org/10.1101/2024.01.21.576546)

**Please note**: This package is at a very early stage of development, with frequent changes being made. If you intend to use this package at this stage, I kindly ask that you contact me via the email address in the [pyproject.toml](https://github.com/mibur1/dfc-multiverse/blob/main/pyproject.toml) file.

### Installation

Installation is possible through the Python Package Index (PyPI) with the pip or pip3 command, depending on your system:

```
pip install comet-toolbox
```
We further recommend using a dedicated [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://conda.io/projects/conda/en/latest/index.html) environment to mitigate the risk of potential version conflicts.


Installation from the source code of this repository is also possible:

1. Download/clone the repository
2. Open a terminal in the folder which contains the pyproject.toml file
3. Install the package via pip (or pip3, depending on your environment)
4. If you intend to implement your own modification, installing in editable mode (-e) is a helpful approach

```
pip install -e .
```

### Usage

**GUI** 

After installation, you can use the graphical user interface through the terminal by typing:

```
comet-gui
```

If you want to explore the toolbox with example data, you can load the ```src/comet/resources/simulation.txt``` file which should result in two changing connectivity patterns.

**Scripting**

If you intend to use the toolbox in a standard python script, [demo scripts](https://github.com/mibur1/dfc-multiverse/tree/main/tutorials) are provided as a starting point:
* Demo script for calculating dFC: [click here](tutorials/example_dfc.ipynb)
* Demo script for performing multiverse analysis: [click here](tutorials/example_multiverse.ipynb)
* Demo script for the multiverse analysis as presented preprint (+ additional visualizations): [click here](tutorials/example_analysis.ipynb)


### Outlook
The toolbox is under active development, with various features being underway, e.g.:

* Addition of state-based dFC methods
* Multiverse analysis within the GUI
* Visualization of results

If you have any wishes, suggestions, feedback, or encounter any bugs, please don't hesitate to contact me via email or create an issue [here](https://github.com/mibur1/dfc-multiverse/issues).