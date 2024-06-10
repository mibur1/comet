Usage
=====

The toolbox is designed in a modular way, which means the individual methods can be used in combination with others, but also by themselves.
For full functionality the scripting API is recommended, however the graphical user interface (GUI) offers many of the same features.

GUI
---

After installation, graphical user interface can be accessed through the terminal by typing:

.. code-block:: bash

    comet-gui

For exploration with example data, data included in the ``tutorials/example_data/`` folder can be loaded:

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


Graph measures can be calculated through the graph module. An example for global efficiency (using the dFC data calculated in the previous example):

.. code-block:: python

    from comet import data, methods, graph

    ts = data.load_example_data()
    dFC = methods.SlidingWindow(ts, windowsize=30, shape="gaussian").connectivity()

    adj = dFC[:,:,0]
    dFC = graph.efficiency(adj, local=False)


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
