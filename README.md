<div style="padding-top:1em; padding-bottom: 0.5em;">
<img src="src/comet/data/img/logo.svg" width =130 align="right" />
</div>

# Comet - A toolbox for dynamic functional connectivity and multiverse analysis

[![DOI](https://img.shields.io/badge/DOI-10.1101%2F2024.01.21.576546-blue?logo=arxiv)](https://doi.org/10.1101/2024.01.21.576546) [![PyPI](https://img.shields.io/badge/PyPI-comet--toolbox-orange?logo=PyPI)](https://pypi.org/project/comet-toolbox/)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/2e766745c5c04d4786ea28f7135c193e)](https://app.codacy.com/gh/mibur1/comet/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade) [![Documentation Status](https://readthedocs.org/projects/comet-toolbox/badge/?version=latest)](https://comet-toolbox.readthedocs.io/en/latest/) [![Coverage Status](https://coveralls.io/repos/github/mibur1/dfc-multiverse/badge.svg?branch=main)](https://coveralls.io/github/mibur1/dfc-multiverse?branch=main)


## About the toolbox

Please refer to the **[documentation](https://comet-toolbox.readthedocs.io/en/latest/)** for detailed information about the toolbox. The following README will only provide a very brief overview.

Please also note that the package is under active development. If you have any questions, suggestions, or want to contribute, please don't hesitate to reach out on GitHub or via the email address in the [`pyproject.toml`](https://github.com/mibur1/dfc-multiverse/blob/main/pyproject.toml) file. Some features are also not yet tested, so there will be bugs (the question is just how many).


### Installation and usage

It is recommended to use a dedicated Python environment (e.g. through conda) to mitigate the risk of potential version conflicts. Installation is possible through the Python Package Index (PyPI) or from the source code in this repository:

```
conda create -n comet python==3.11
conda activate comet
pip install comet-toolbox
```

Usage of the toolbox is then possible through either the GUI:

```
comet-gui
```

or through the scripting API:

```{code}python
from comet import connectivity, graph, multiverse
```

For this, **[demo scripts](https://github.com/mibur1/dfc-multiverse/tree/main/tutorials)** are provided as starting points.

### Current features

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
                <li>Specification curve analysis</li>
            </ul>
        </td>
    </tr>
</table>


### Code structure

```{code}   
/
├─ src/comet/         ← Parent directory
│  ├─ connectivity.py ← Functional connectivity module
│  ├─ graph.py        ← Graph analysis module
│  ├─ multiverse.py   ← Multiverse analysis module
│  ├─ gui.py          ← Graphical user interface
│  ├─ utils.py        ← Miscellaneous helper functions
│  ├─ cifti.py        ← CIFTI related functions
│  └─ bids.py         ← BIDS related functions (placeholder)
├─ docs/              ← Documentation
├─ tutorials/         ← Example jupyter notebooks
├─ tests/             ← Unit tests
├─ pyproject.toml     ← Packaging & dependencies
└─ README.md          ← Project overview
```