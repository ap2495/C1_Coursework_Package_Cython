# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sphinx_rtd_theme
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'Cython Dual Autodifferentiaton Package'
copyright = '2024, Alexandr Prucha'
author = 'Alexandr Prucha'
release = '0.1.dev0+d20241205'
autoclass_content = 'both'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinx.ext.mathjax',
              'nbsphinx',]

mathjax3_config = {
    'tex2jax': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'displayMath': [['$$', '$$'], ['\\[', '\\]']]
    }
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'dual_autodiff_x/version.py']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


source_suffix = {
    '.rst': 'restructuredtext',
    '.pyx': 'cython',  # Treat `.pyx` files as Cython
}

cython_directives = {
    'binding': True,  # Enable Python function annotations for Cython files
    'language_level': 3,  # Set the language level to Python 3
}