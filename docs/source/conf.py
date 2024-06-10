import os
import sys
import sphinx_rtd_theme

# Add the project's root directory to sys.path
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
project = 'Comet Toolbox'
copyright = '2024, Micha Burkhardt'
author = 'Micha Burkhardt'
release = '2024'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme',
    'nbsphinx',
]
templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_static_path = ['_static']

html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': True,
    'sticky_navigation': True,
    'titles_only': False
}

#print(f"Available files: {os.listdir()}")