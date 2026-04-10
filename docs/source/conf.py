# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import re
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version as package_version


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


def get_release() -> str:
    try:
        return package_version("vitabel")
    except PackageNotFoundError:
        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        match = re.search(r'^version = "([^"]+)"', pyproject.read_text(), re.MULTILINE)
        return match.group(1) if match else "unknown"


project = "vitabel"
release = get_release()
version = ".".join(release.split(".")[:2])
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
html_title = f"{project} v{release} documentation"
html_static_path = ["_static"]
html_logo = "../../assets/logo/Vitabel_Logo.svg"
html_css_files = [
    "css/extra.css",
]
