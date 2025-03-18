Usage
=====

The toolbox is designed in a modular way, which means the individual modules can be used in combination with others, but also by themselves.
Users also have the option to choose between a normal Python scripting API and a graphical user interface (GUI). 

.. tip::

  The GUI offers many features for data loading and processing, (dynamic) functional connectivity estimation, graph analysis, and multiverse analysis. The scripting API is recommended if more flexibility is needed.


GUI
---

After installation, graphical user interface can be accessed through the terminal by typing:

.. code-block:: bash

    comet-gui

For exploration with example data, data included in the ``src/comet/data/`` folder can be loaded:

* ``simulation.txt``           Simulated BOLD data for 10 brain regions with 2 changing brain states (as used in the preprint).
* ``time_series.txt``          Parcellated BOLD data for a single subject.
* ``time_series_multiple.npy`` Parcellated BOLD data for 5 subjects.


Scripting
---------

Dynamic functional connectivity can be estimated through the ``connectivity`` module. An example for sliding window correlation:

.. code-block:: python

    from comet import connectivity, utils

    ts = utils.load_example()
    dFC = connectivity.SlidingWindow(ts, windowsize=30, shape="gaussian").estimate()


Graph measures can be calculated through the graph module. An example for global efficiency (using the dFC data calculated in the previous example):

.. code-block:: python

    from comet import connectivity, graph, utils

    ts = utils.load_example()

    sw = connectivity.SlidingWindow(ts, windowsize=30, shape="gaussian")
    dFC = sw.estimate()

    adj = dFC[:,:,0]
    global_eff = graph.efficiency(adj, local=False)

Multiverse analysis can be conducted through the multiverse module.
This exaple will create and run a multiverse analysis with two decisions (6 possible combinations):

.. code-block:: python

    from comet.multiverse import Multiverse

    forking_paths = {
        "decision1": [1, 2, 3],
        "decision2": ["Hello", "World"]
        }

    def analysis_template():
        print(f"Decision1: {{decision1}}")
        print(f"Decision2: {{decision2}}")

    mverse = Multiverse(name="example_multiverse")
    mverse.create(analysis_template, forking_paths)
    mverse.summary()
    mverse.run()
