Installation
------------

As Comet contains a fair amount of dependencies, it should be installed in a dedicated Python environment (e.g. `Conda <https://conda-forge.org/download>`_ or another environment manager of your choice) 
to avoid version conflicts.

.. note::

    Supported Python versions are 3.11 to 3.13. Earlier versions might still work, but will require users to change the version constraints in the pyproject.toml.

.. note::

    Comet runs on all major operating systems (Linux, Windows, macOS), although development and testing are primarily conducted on Linux.
    If you encounter any issues, please let us know via the `issue tracker <https://github.com/mibur1/comet/issues>`_.


.. code-block:: shell

    conda create -n comet python==3.13
    conda activate comet

Installation is then possible through the Python Package Index (PyPI) via the pip command:

.. code-block:: shell

    pip install comet-toolbox

If you require the most recent updates, which may not yet be available on PyPI, you can also install the package directly from the source code of this repository:

1. Download/clone the repository
2. Open a terminal in the main folder (the folder which contains the pyproject.toml file)
3. Install the package via pip (installing in editable mode (-e) is a helpful approach if you intend to modify the source code):

.. code-block:: shell

    pip install -e .


.. tip::

    Online usage (e.g. through Google Colab) is also possible if the toolbox is installed in the runtime environment: `!pip install comet-toolbox`


Troubleshooting
...............

If you encounter errors regarding the matplotlib backend, you may need to install the following packages:

.. code-block:: shell

    sudo apt-get update
    sudo apt-get install -y libegl1-mesa libxkbcommon-x11-0


Optional dependencies
.....................

Dependecies for testing and building are not installed by default and can be installed via:

.. code-block:: shell

    pip install "comet-toolbox[doc]"
    pip install "comet-toolbox[test]"
    pip install "comet-toolbox[build]"