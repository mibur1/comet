Installation
------------

It is recommended to use a dedicated `Anaconda <https://www.anaconda.com/download>`_ or `Miniconda <https://conda.io/projects/conda/en/latest/index.html>`_ environment to mitigate the risk of potential version conflicts:

.. code-block:: shell

    conda create -n comet python==3.11
    conda activate comet

Installation is then possible through the Python Package Index (PyPI) with the pip or pip3 command, depending on your system:

.. code-block:: shell

    pip install comet-toolbox

Installation from the source code of this repository is also possible:

1. Download/clone the repository
2. Open a terminal in the folder which contains the pyproject.toml file
3. Install the package via pip (installing in editable mode (-e) is a helpful approach if you intend to modify the source code):

.. code-block:: shell

    pip install -e .
