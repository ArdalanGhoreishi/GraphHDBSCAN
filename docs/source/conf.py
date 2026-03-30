import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../../src"))

project = "GraphHDBSCAN-star"
author = "S. A. Ghoreishi"
copyright = f"{datetime.now():%Y}, {author}"
release = "0.2.9"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_nb",
]

templates_path = ["../_templates"]
exclude_patterns = []

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
    ".ipynb": "myst-nb",
}

html_theme = "furo"
html_title = f"{project} documentation"
html_static_path = ["../_static"]

html_logo = "../_static/graphhdbscan_logo.png"

html_theme_options = {
    "sidebar_hide_name": True,
}

autosummary_generate = True
autosummary_imported_members = False

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# Render stored notebook outputs without re-executing at docs build time.
nb_execution_mode = "off"

# Enable HTML embedding support for ipywidgets in rendered notebook pages.
nb_ipywidgets_js = {
    "https://cdn.jsdelivr.net/npm/@jupyter-widgets/html-manager@*/dist/embed-amd.js": {
        "data-jupyter-widgets-cdn": "https://cdn.jsdelivr.net/npm/",
        "crossorigin": "anonymous",
    }
}

# Mock heavy scientific imports for docs-only builds if needed.
autodoc_mock_imports = [
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "networkx",
    "sklearn",
    "hdbscan",
    "scanpy",
    "phenograph",
    "umap",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
