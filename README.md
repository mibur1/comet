# Comet - A dynamic functional connectivity toolbox for multiverse analysis


[![DOI](https://img.shields.io/badge/DOI-10.1101%2F2024.01.21.576546-blue?logo=arxiv)](https://doi.org/10.1101/2024.01.21.576546) [![PyPI](https://img.shields.io/badge/PyPI-comet--toolbox-orange?logo=PyPI)](https://pypi.org/project/comet-toolbox/) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/2e766745c5c04d4786ea28f7135c193e)](https://app.codacy.com/gh/mibur1/dfc-multiverse/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade) [![Documentation Status](https://readthedocs.org/projects/comet-toolbox/badge/?version=latest)](https://comet-toolbox.readthedocs.io/en/latest/?badge=latest)

## Important notes


- This package is at an early stage of development, with frequent changes being made. If you intend to use this package at this stage, I kindly ask that you contact me via the email address in the [`pyproject.toml`](https://github.com/mibur1/dfc-multiverse/blob/main/pyproject.toml) file.
- Many features are not yet tested, so there will be bugs (the question is just how many). A comprehensive testing suite and documentation will be added in the near future.

## Installation and Usage

It is recommended to use a dedicated Python environment to mitigate the risk of potential version conflicts.

Please refer to the **[documentation](https://comet-toolbox.readthedocs.io/en/latest/)** for comprehensive installation and usage instructions. Briefly, installation is possible through the Python Package Index (PyPI) or from the source code in this repository:

```
pip install comet-toolbox
```

Usage of the toolbox is then possible through either the GUI:

```
comet-gui
```

or (for more versatile usage) through the scripting API. For this, **[demo scripts](https://github.com/mibur1/dfc-multiverse/tree/main/tutorials)** are provided as starting points:

* Demo script for **[calculating dFC](tutorials/example_dfc.ipynb)**
* Demo script for **[performing multiverse analysis](tutorials/example_multiverse.ipynb)**
* Demo script for **[multiverse analysis example from the preprint](tutorials/example_analysis.ipynb)** (+ additional visualizations)
* Demo script for **[graph analysis](tutorials/example_graph.ipynb)**

## Current Features

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
                <li>Sliding Window Correlation</li>
                <li>Jackknife Correlation</li>
                <li>Flexible Least Squares</li>
                <li>Spatial Distance</li>
                <li>Temporal Derivatives</li>
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
                <li>Windowless</li>
            </ul>
            <strong>Static</strong>
            <ul>
                <li>Pearson Correlation</li>
                <li>Partial Correlation</li>
                <li>Mutual Information</li>
            </ul>
        </td>
        <td style="vertical-align: top; padding: 8px; border: 1px solid #ddd;">
            <strong>Optimized implementation</strong>
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
                <li>Negative weights</li>
                 <li>...</li>
            </ul>
            <strong>BCT Integration</strong>
            <ul>
                <li>All BCT functions can be<br>used seamlessly fory<br>multiverse analysis</li>
                <li>Many BCT functions are available in the GUI</li>
            </ul>
        </td>
        <td style="vertical-align: top; padding: 8px; border: 1px solid #ddd;">
            <strong>Simple Definition</strong>
            <ul>
                <li>Forking paths as<br>python dictionary</li>
                <li>Analysis pipeline template with decision points</li>
            </ul>
            <strong>Generation</strong>
            <ul>
                <li>Universes are created<br>as individual scripts</li>
                <li>Modular approach</li>
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
