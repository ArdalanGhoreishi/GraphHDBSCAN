Installation and quick start
============================

This page explains how to install ``coresg-graphhdbscan`` and run a first
clustering example.

Installation
------------

Local installation
^^^^^^^^^^^^^^^^^^

Install the package from a local checkout in editable mode:

.. code-block:: bash

   git clone https://github.com/Campello-Lab/GraphHDBSCAN.git
   cd GraphHDBSCAN
   pip install -e .

This is the most convenient option during development because changes in the
source tree are picked up without reinstalling the package each time.

GitHub installation
^^^^^^^^^^^^^^^^^^^

If you want to install directly from the repository:

.. code-block:: bash

   pip install git+https://github.com/Campello-Lab/GraphHDBSCAN.git

.. Documentation dependencies
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^

.. If you are also building the documentation locally, install the docs
.. requirements:

.. .. code-block:: bash

..    pip install -r docs/requirements.txt

Typical dependencies
--------------------

Depending on the selected graph backend, the following libraries may be
required:

- ``numpy``
- ``scipy``
- ``scikit-learn``
- ``networkx``
- ``hdbscan``
- ``scanpy`` for ``sc_gauss`` and ``sc_umap``
- ``scanpy.external`` for ``jaccard_phenograph``

These dependencies are normally handled through the package installation, but
they are useful to know when setting up a fresh environment or troubleshooting
imports.

Recommended environment setup
-----------------------------

A clean Python environment is recommended, especially when working with
scientific Python packages.

For example:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -e .

If you use Conda, an equivalent workflow is:

.. code-block:: bash

   conda create -n graphhdbscan python=3.11
   conda activate graphhdbscan
   pip install -e .

Minimal import test
-------------------

After installation, verify that the package imports correctly:

.. code-block:: python

   from coresg_graphhdbscan import GraphCoreSGHDBSCAN

If this import works, the package is installed and the public entry point is
available.

Quick start
-----------

Simple example
^^^^^^^^^^^^^^

A minimal clustering workflow looks like this:

.. code-block:: python

   from coresg_graphhdbscan import GraphCoreSGHDBSCAN

   model = GraphCoreSGHDBSCAN(
       min_samples=10,
       sim_graph_method="sc_umap",
       metric="euclidean",
       n_neighbors=15,
       no_noise=True,
       heuristic_connect=False,
   )

   model.fit(X)
   labels = model.fit_predict(X)

This example uses:

- ``min_samples=10`` as a default clustering setting
- ``sim_graph_method="sc_umap"`` as the default graph builder
- ``metric="euclidean"`` as the default metric strategy
- ``n_neighbors=15`` as the default local graph size
- ``no_noise=True`` to reassign noise points after clustering

Single ``min_samples`` value
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want one clustering solution only:

.. code-block:: python

   model = GraphCoreSGHDBSCAN(min_samples=10)
   labels = model.fit_predict(X)

Multiple ``min_samples`` values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One strength of the package is that you can fit several ``min_samples`` values
in a single run and inspect them later.

.. code-block:: python

   model = GraphCoreSGHDBSCAN(min_samples=[5, 10, 15])
   model.fit(X)

   labels_5 = model.labels_for(5)
   labels_10 = model.labels_for(10)
   labels_15 = model.labels_for(15)

This is useful when you want to compare clustering solutions across several
density settings without repeating the full workflow from scratch.

Precomputed graph input
^^^^^^^^^^^^^^^^^^^^^^^

If you already have a graph or adjacency representation, use
``sim_graph_method="precomputed"``:

.. code-block:: python

   model = GraphCoreSGHDBSCAN(
       min_samples=10,
       sim_graph_method="precomputed",
       no_noise=True,
   )
   model.fit(my_graph)

In precomputed mode, the input to ``fit(...)`` may be:

- a ``networkx.Graph``
- a SciPy sparse adjacency matrix
- a square dense adjacency matrix

Inspecting the hierarchy
^^^^^^^^^^^^^^^^^^^^^^^^

After fitting, you can inspect the hierarchical clustering structure through the
condensed tree:

.. code-block:: python

   model.fit(X)
   model.plot_condensed_tree(10)

If you fit multiple ``min_samples`` values, inspect a specific one by passing
the selected value.

Typical workflow
----------------

A common workflow is:

1. install the package in a clean environment
2. import ``GraphCoreSGHDBSCAN``
3. choose a graph builder and metric
4. fit the model on feature data or a precomputed graph
5. inspect the condensed tree
6. retrieve labels for the selected ``min_samples`` value

A good exploratory run looks like:

.. code-block:: python

   from coresg_graphhdbscan import GraphCoreSGHDBSCAN

   g = GraphCoreSGHDBSCAN(
       min_samples=range(2, 20),
       sim_graph_method="sc_gauss",
       n_neighbors=16,
       no_noise=True,
       metric="euclidean",
       heuristic_connect=True,
   )

   g.fit(X)
   g.plot_condensed_tree(4)
   labels_18 = g.labels_for(18)

Troubleshooting installation
----------------------------

If imports fail, the cause is usually one of these:

- the environment is missing optional scientific dependencies
- binary packages such as NumPy, SciPy, or scikit-learn are mismatched
- optional graph-construction dependencies are not installed

Helpful check:

.. code-block:: bash

   python -c "from coresg_graphhdbscan import GraphCoreSGHDBSCAN; print('ok')"

.. If you are building documentation and imports fail during Sphinx builds, it may
.. also help to mock heavy imports in ``docs/source/conf.py``.

Related pages
-------------

For more detail, continue with:

- :doc:`overview`
- :doc:`parameters`
- :doc:`examples`
- :doc:`api`
