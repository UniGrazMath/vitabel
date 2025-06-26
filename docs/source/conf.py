# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "vitabel"
copyright = "2025, Benjamin Hackl, Wolfgang Kern, Simon Orlob"
author = "Benjamin Hackl, Wolfgang Kern, Simon Orlob"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
    "numpydoc",
    "autoapi.extension",
    "sphinxcontrib.bibtex",
]
numpydoc_show_class_members = False
autoapi_dirs = ["../../src"]
autoapi_template_dir = "_autoapi_templates"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
nb_execution_mode = "off"

templates_path = ["_templates", "_autoapi_templates"]
exclude_patterns = []

bibtex_bibfiles = ["bibliography.bib"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_logo = "../../assets/logo/Vitabel_Logo.svg"
html_css_files = [
    "css/extra.css",
]
