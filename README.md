<div style="padding-top:1em; padding-bottom: 0.5em;">
<img src="src/comet/data/img/logo.svg" width =130 align="right" />
</div>

# Comet - A toolbox for dynamic functional connectivity and multiverse analysis

[![DOI](https://img.shields.io/badge/DOI-10.1101%2F2024.01.21.576546-blue?logo=arxiv)](https://doi.org/10.1101/2024.01.21.576546) [![PyPI](https://img.shields.io/badge/PyPI-comet--toolbox-orange?logo=PyPI)](https://pypi.org/project/comet-toolbox/)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/2e766745c5c04d4786ea28f7135c193e)](https://app.codacy.com/gh/mibur1/comet/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade) [![Documentation Status](https://readthedocs.org/projects/comet-toolbox/badge/?version=latest)](https://comet-toolbox.readthedocs.io/en/latest/) [![Coverage Status](https://coveralls.io/repos/github/mibur1/comet/badge.svg)](https://coveralls.io/github/mibur1/comet)

## About the toolbox

Please refer to the **[documentation](https://comet-toolbox.readthedocs.io/en/latest/)** for detailed information about the toolbox and the current features. The following README will only provide a very brief overview.

> [!NOTE]
> This package is under active development. If you have any questions, suggestions, or want to contribute, please do not hesitate to reach out on GitHub or via the email address in the [`pyproject.toml`](https://github.com/mibur1/dfc-multiverse/blob/main/pyproject.toml) file. Some features are also not yet tested, so there will be bugs (the question is just how many).


### Installation and usage

As Comet contains a fair amount of dependencies, it should be installed in a dedicated Python environment (e.g. [conda](https://conda-forge.org/download) or another environment manager of your choice) to avoid version conflicts. Comet runs on all major operating systems (Linux, Windows, macOS), although development and testing are primarily conducted on Linux. If you encounter any issues, please let us know via the [issue tracker](https://github.com/mibur1/comet/issues).

```
conda create -n comet python==3.11
conda activate comet
pip install comet-toolbox
```

Usage of the toolbox is then possible through either the GUI (might take 1-2 minutes to open on the *first* start):

```
comet-gui
```

or through the scripting API:

```{code}python
from comet import connectivity, graph, multiverse
```

A comprehensive set of usage examples are provided in the **[documentation](https://github.com/mibur1/dfc-multiverse/tree/main/tutorials)**.

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

### Contributing

We warmly welcome contributions and suggestions for new features! Comet is an open and collaborative project, and your input helps make it better for the entire community.
More detailed contribution guidelines will follow soon. For now, before submitting a pull request, please open an [issue](https://github.com/mibur1/comet/issues) on GitHub to start a discussion or share ideas.
