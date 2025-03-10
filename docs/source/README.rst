.. raw:: html

    <div style="display: flex; align-items: center;">
        <img src="_static/logo.svg" alt="logo" style="width: 100px; margin-right: 15px;">
        <h1 style="margin: 0;">Comet - A toolbox for dynamic functional connectivity and multiverse analysis</h1>
    </div>

.. raw:: html

    <br>

.. image:: https://img.shields.io/badge/DOI-10.1101%2F2024.01.21.576546-blue?logo=arxiv
   :target: https://doi.org/10.1101/2024.01.21.576546
   :alt: DOI Badge

.. image:: https://img.shields.io/badge/PyPI-comet--toolbox-orange?logo=PyPI
   :target: https://pypi.org/project/comet-toolbox/
   :alt: PyPI Badge

.. image:: https://app.codacy.com/project/badge/Grade/2e766745c5c04d4786ea28f7135c193e
   :target: https://app.codacy.com/gh/mibur1/dfc-multiverse/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade
   :alt: Codacy Badge

.. image:: https://readthedocs.org/projects/comet-toolbox/badge/?version=latest
   :target: https://comet-toolbox.readthedocs.io/en/latest
   :alt: Documentation Status

.. image:: https://coveralls.io/repos/github/mibur1/dfc-multiverse/badge.svg?branch=main
   :target: https://coveralls.io/github/mibur1/dfc-multiverse?branch=main
   :alt: Coverage Status

.. raw:: html

    <br><br>

Yet another toolbox?
--------------------

You might ask yourself, "Why do I need yet another toolbox for dynamic functional connectivity (dFC) analysis and network neuroscience?"
The answer is simple: the reproducibility crisis, driven in part by the maze of arbitrary yet defensible decisions in our analyses.

To address this, we introduce the Comet toolbox — a tool that helps ensure your findings don't get lost in the Bermuda Triangle of irreproducible results.
Whether you want to take full advantage of multiverse analysis or stick with traditional analyses, Comet lets you explore a broad range of methodological decisions, making your research more robust and transparent.

From dFC estimation to graph analyses, Comet offers a wide range of methods. Plus, with an easy-to-use graphical user interface and comprehensive demo scripts, even the most Python-phobic among us can dive in and
start exploring brain dynamics like never before. So, buckle up, and let's bring some cosmic clarity to your research universe!

.. note::

 The Comet toolbox is under active development. If you have any questions, suggestions, or want to contribute, please don't hesitate to reach out on GitHub or via email.


Current features
----------------

.. raw:: html

    <table style="width: 100%; border-collapse: collapse;">
        <tr>
            <th style="text-align: left; padding: 8px; border: 1px solid #ddd;">Functional Connectivity</th>
            <th style="text-align: left; padding: 8px; border: 1px solid #ddd;">Graph Analysis</th>
            <th style="text-align: left; padding: 8px; border: 1px solid #ddd;">Multiverse Analysis</th>
        </tr>
        <tr>
            <td style="vertical-align: top; padding: 8px; border: 1px solid #ddd;">
                <strong>Continuous</strong>
                <ul>
                    <li>Sliding Window</li>
                    <li>Jackknife Correlation</li>
                    <li>Flexible Least Squares</li>
                    <li>Spatial Distance</li>
                    <li>Temporal Derivatives</li>
                    <li>Dynamic Conditional Correlation</li>
                    <li>Phase Synchronization</li>
                    <li>Leading Eigenvector Dynamics</li>
                    <li>Wavelet Coherence</li>
                    <li>Edge-centric connectivity</li>
                </ul>
                <strong>State-Based</strong>
                <ul>
                    <li>SW Clustering</li>
                    <li>Co-activation Patterns</li>
                    <li>Discrete HMM</li>
                    <li>Continuous HMM</li>
                    <li>k-SVD</li>
                </ul>
                <strong>Static</strong>
                <ul>
                    <li>Pearson Correlation</li>
                    <li>Partial Correlation</li>
                    <li>Mutual Information</li>
                </ul>
            </td>
            <td style="vertical-align: top; padding: 8px; border: 1px solid #ddd;">
                <strong>Optimized implementations</strong>
                <ul>
                    <li>Average Path Length</li>
                    <li>Global Efficiency</li>
                    <li>Nodal Efficiency</li>
                    <li>Small-World Sigma</li>
                    <li>Small-World Propensity</li>
                    <li>Matching Index</li>
                </ul>
                <strong>Standard Graph Functions</strong>
                <ul>
                    <li>Threshold</li>
                    <li>Binarise</li>
                    <li>Symmetrise</li>
                    <li>Handle negative weights</li>
                    <li>...</li>
                </ul>
                <strong>BCT Integration</strong>
                <ul>
                    <li>All BCT functions can be used seamlessly for multiverse analysis</li>
                    <li>Many BCT functions are available in the GUI</li>
                </ul>
            </td>
            <td style="vertical-align: top; padding: 8px; border: 1px solid #ddd;">
                <strong>Simple Definition</strong>
                <ul>
                    <li>Forking paths as dictionary</li>
                    <li>Analysis pipeline template with decision points</li>
                </ul>
                <strong>Generation</strong>
                <ul>
                    <li>Universes are created as individual scripts</li>
                    <li>Modular approach</li>
                    <li>Complex multiverse structures</li>
                </ul>
                <strong>Analysis</strong>
                <ul>
                    <li>Individual universes</li>
                    <li>Entire multiverse (parallel)</li>
                </ul>
                <strong>Visualization</strong>
                <ul>
                    <li>Multiverse summary</li>
                    <li>Multiverse as a network</li>
                    <li>Specification Curve analysis</li>
                </ul>
            </td>
        </tr>
    </table>

.. raw:: html

    <br>