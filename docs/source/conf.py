# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'vitabel'
copyright = '2025, Benjamin Hackl, Wolfgang Kern, Simon Orlob'
author = 'Benjamin Hackl, Wolfgang Kern, Simon Orlob'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "numpydoc",
    "autoapi.extension",
]
numpydoc_show_class_members = False
autoapi_dirs = ["../../src"]
autoapi_template_dir = "_autoapi_templates"
autoapi_options = [
    'members', 'undoc-members', 'private-members', 'show-inheritance', 'show-module-summary', 'special-members'
]

templates_path = ['_templates', '_autoapi_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
