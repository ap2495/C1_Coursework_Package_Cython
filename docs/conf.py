# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

#import sphinx_rtd_theme
import os
import sys
# Add the path to the src directory to import dual_autodiff_x
sys.path.insert(0, os.path.abspath('../src'))

project = 'Dual Autodifferentiation Package for Cython'
copyright = '2024, Alexandr Prucha'
author = 'Alexandr Prucha'
release = '0.1.dev0+d20241205'
autoclass_content = 'both'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'nbsphinx',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode'
]

mathjax3_config = {
    'tex2jax': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'displayMath': [['$$', '$$'], ['\\[', '\\]']]
    }
}

templates_path = ['_templates']

exclude_patterns = [
    '_build', 
    'Thumbs.db', 
    '.DS_Store', 
    'dual_autodiff_x/version.py',  # Updated to dual_autodiff_x
    '**/version.py'
]



autosummary_generate = True
add_module_names = False  # Remove module paths like dual_autodiff_x.dual
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'special-members': '__add__,__sub__,__mul__,__pow__',
    'show-inheritance': True
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
# Add Cython `.pyx` files to the list of source file suffixes
source_suffix = ['.rst', '.pyx']


cython_directives = {
    'binding': True,  # Enable Python function annotations for Cython files
    'language_level': 3,  # Set the language level to Python 3
}