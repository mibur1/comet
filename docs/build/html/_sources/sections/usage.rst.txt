Usage
=====

Quickstart
----------

The toolbox is designed in a modular way, which means you can use the individual parts in combination with others, but also by themselves.

* continuous and static dFC measures require 2D time series data (n_timepoints x n_regions) as input
* state-based dFC methods require a TIME_SERIES object (as used in the `pydfc toolbox <https://github.com/neurodatascience/dFC>`_) containing data for multiple subjects as input
* Graph measures need 2D adjacency/connectivity matrices as input
* Multiverse analysis needs decision/option pairs of any kind to create forking paths in the analysis as well as a template script for the analysis

GUI
---

After installation, you can use the graphical user interface through the terminal by typing:

.. code-block:: bash

    comet-gui

If you want to explore the toolbox with example data, you can load data included in the ``tutorials/example_data/`` folder:

* ``simulation.txt`` contains simulated BOLD data for 10 brain regions with 2 changing brain states (usable for continuous and static dFC measures)
* ``abide_50088.txt`` contains parcellated BOLD data for a single subject from the ABIDE data set (usable for continuous and static dFC measures)
* ``aomic_multi.pkl`` contains parcellated BOLD data for 5 subjects from the AOMIC data set (usable for state-based dFC measures)

Scripting
---------

Dynamic functional connectivity can be estimated through the methods module. An example for sliding window correlation:

.. code-block:: python

    from comet import data, methods

    ts = data.load_example_data()
    dFC = methods.SlidingWindow(ts, windowsize=30, shape="gaussian").connectivity()


Graph measures can be calculated through the graph module. An example for global efficiency:

.. code-block:: python

    from comet import graph

    adj = dFC[:,:,0]
    dFC = graph.efficiency(adj, local=False)


Multiverse analysis can be conducted through the multiverse module. An example for a simple decision point:

.. code-block:: python

    from comet import multiverse
