Usage
=====

This page shows how to work with ``GraphCoreSGHDBSCAN`` in practice after the
package is installed.

The package supports two main workflows:

- build a similarity graph from feature data and cluster from that graph
- provide a graph or adjacency representation directly in ``precomputed`` mode

For most users, ``GraphCoreSGHDBSCAN`` is the main entry point.

Basic workflow
--------------

A typical workflow is:

1. prepare a feature matrix ``X`` or a precomputed graph
2. create a ``GraphCoreSGHDBSCAN`` model
3. call ``fit(...)`` or ``fit_predict(...)``
4. inspect labels
5. inspect the condensed tree
6. compare several ``min_samples`` values if needed

Minimal example
---------------

The simplest usage pattern is:

.. code-block:: python

   from coresg_graphhdbscan import GraphCoreSGHDBSCAN

   model = GraphCoreSGHDBSCAN()
   labels = model.fit_predict(X)

This uses the default configuration:

- ``min_samples=10``
- ``sim_graph_method="sc_umap"``
- ``metric="euclidean"``
- ``add_neighbor=True``
- ``no_noise=True``
- ``n_neighbors=15``
- ``heuristic_connect=False``
- ``min_cluster_size=None``

fit vs. fit_predict
-------------------

Use ``fit_predict(X)`` when you want labels immediately for a single requested
configuration.

.. code-block:: python

   model = GraphCoreSGHDBSCAN(min_samples=10)
   labels = model.fit_predict(X)

Use ``fit(X)`` when you want to inspect the fitted object, view the hierarchy,
or retrieve results for several ``min_samples`` values later.

.. code-block:: python

   model = GraphCoreSGHDBSCAN(min_samples=10)
   model.fit(X)

Single ``min_samples`` value
----------------------------

If you want one clustering solution only, pass a single integer:

.. code-block:: python

   model = GraphCoreSGHDBSCAN(min_samples=10)
   labels = model.fit_predict(X)

This is the simplest and most common starting point.

Multiple ``min_samples`` values in one run
------------------------------------------

One of the package's main strengths is the ability to fit several
``min_samples`` values in one model run.

.. code-block:: python

   from coresg_graphhdbscan import GraphCoreSGHDBSCAN

   model = GraphCoreSGHDBSCAN(min_samples=[5, 10, 15])
   model.fit(X)

   labels_5 = model.labels_for(5)
   labels_10 = model.labels_for(10)
   labels_15 = model.labels_for(15)

This is useful when you want to compare clustering behavior across several
density settings without rebuilding the full workflow from scratch.

You can also use ranges:

.. code-block:: python

   model = GraphCoreSGHDBSCAN(min_samples=range(2, 10))
   model.fit(X)

Inspecting the hierarchy
------------------------

After fitting, you can inspect the hierarchical structure through the condensed
tree.

Static condensed tree
^^^^^^^^^^^^^^^^^^^^^

The most reliable visualization method is the static condensed tree:

.. code-block:: python

   model.fit(X)
   model.plot_condensed_tree(10)

If you fit several ``min_samples`` values, pass the specific value you want to
inspect.

Interactive condensed tree
^^^^^^^^^^^^^^^^^^^^^^^^^^

For live notebook work, the package also provides:

.. code-block:: python

   widget = model.interactive_condensed_tree()
   widget

This is most useful in a live Jupyter environment. In rendered documentation,
the static condensed tree is usually more reliable than widget-based output.

Choosing a graph-construction backend
-------------------------------------

The package supports several graph-construction backends through
``sim_graph_method``.

Default UMAP-style graph
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   model = GraphCoreSGHDBSCAN(
       sim_graph_method="sc_umap",
       n_neighbors=15,
   )
   model.fit(X)

This is the default and a good starting point for many datasets.

Gaussian connectivity graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   model = GraphCoreSGHDBSCAN(
       sim_graph_method="sc_gauss",
       n_neighbors=15,
   )
   model.fit(X)

This uses Scanpy's Gaussian connectivity routine.

PhenoGraph-style graph
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   model = GraphCoreSGHDBSCAN(
       sim_graph_method="jaccard_phenograph",
       n_neighbors=15,
   )
   model.fit(X)

This uses a PhenoGraph-style graph construction.

Using different metric modes
----------------------------

The package supports three metric modes.

Euclidean mode
^^^^^^^^^^^^^^

.. code-block:: python

   model = GraphCoreSGHDBSCAN(
       sim_graph_method="sc_umap",
       metric="euclidean",
   )
   model.fit(X)

This is the default mode.

Cosine mode
^^^^^^^^^^^

.. code-block:: python

   model = GraphCoreSGHDBSCAN(
       sim_graph_method="sc_gauss",
       metric="cosine",
   )
   model.fit(X)

Use this when angular similarity is more meaningful than Euclidean distance.

Hybrid Euclidean-cosine mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   model = GraphCoreSGHDBSCAN(
       sim_graph_method="sc_umap",
       metric="hybrid_euclidean_cosine",
   )
   model.fit(X)

In this mode, full distances remain Euclidean while neighborhood graph
construction uses cosine geometry.

Using precomputed graphs
------------------------

If you already have a graph or adjacency representation, use
``sim_graph_method="precomputed"``.

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

This mode is useful when the graph is already part of the experimental design
or has been built by another method.

Connectivity handling
---------------------

The final graph used by clustering must be connected.

Default behavior
^^^^^^^^^^^^^^^^

With ``heuristic_connect=False``, disconnected components are connected using a
simple fallback that adds synthetic bridge edges.

.. code-block:: python

   model = GraphCoreSGHDBSCAN(
       heuristic_connect=False,
   )

Heuristic connectivity
^^^^^^^^^^^^^^^^^^^^^^

With ``heuristic_connect=True``, the package increases ``n_neighbors`` until
the graph becomes connected.

.. code-block:: python

   model = GraphCoreSGHDBSCAN(
       heuristic_connect=True,
       n_neighbors=15,
   )

During fitting, the package may report messages such as:

.. code-block:: text

   Trying n_neighbors = 16
   Trying n_neighbors = 17

Noise reassignment
------------------

With ``no_noise=True``, points initially labeled ``-1`` are reassigned by an
MST-based propagation step after clustering.

.. code-block:: python

   model = GraphCoreSGHDBSCAN(
       no_noise=True,
   )

If you want to preserve the original HDBSCAN-style noise labels, disable it:

.. code-block:: python

   model = GraphCoreSGHDBSCAN(
       no_noise=False,
   )

Useful outputs after fitting
----------------------------

After fitting, users commonly inspect:

- ``model.coresg_`` for the internal CoreSG object
- ``model.similarity_graph_`` for the initial similarity graph
- ``model.similarity_graph_WSS`` for the weighted structural similarity graph
- ``model.dissimilarity_graph_`` for the graph after similarity-to-dissimilarity conversion
- ``model.connected_graph_`` for the final connected graph
- ``model.dist_matrix_`` for the dense matrix used by CoreSG-HDBSCAN

If you fit multiple ``min_samples`` values, retrieve a selected solution with:

.. code-block:: python

   labels = model.labels_for(10)

For lower-level access, labels are also stored inside the internal model
objects associated with each fitted ``min_samples`` value.

Common usage patterns
---------------------

Default configuration
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from coresg_graphhdbscan import GraphCoreSGHDBSCAN

   model = GraphCoreSGHDBSCAN()
   labels = model.fit_predict(X)

Explore several density settings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   model = GraphCoreSGHDBSCAN(
       min_samples=range(2, 20),
       sim_graph_method="sc_gauss",
       n_neighbors=16,
       no_noise=True,
       metric="euclidean",
       heuristic_connect=True,
   )

   model.fit(X)
   model.plot_condensed_tree(4)
   labels_18 = model.labels_for(18)

Cosine-based graph construction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   model = GraphCoreSGHDBSCAN(
       min_samples=[5, 10],
       sim_graph_method="sc_gauss",
       metric="cosine",
       n_neighbors=20,
   )
   model.fit(X)
   labels_10 = model.labels_for(10)

Precomputed graph workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   model = GraphCoreSGHDBSCAN(
       min_samples=10,
       sim_graph_method="precomputed",
       no_noise=True,
   )
   model.fit(my_graph)
   model.plot_condensed_tree(10)

Practical notes
---------------

- ``min_samples`` is the main clustering hyperparameter and is often the first
  thing to tune.
- ``min_cluster_size=None`` means that the package follows the selected
  ``min_samples`` value for each run.
- ``plot_condensed_tree(...)`` is the most reliable visualization for static
  documentation and reports.
- ``interactive_condensed_tree()`` is best suited for live notebooks.
- Some graph builders depend on optional packages and will raise an import
  error if those packages are not installed.

Related pages
-------------

For more detail, see:

- :doc:`overview`
- :doc:`installation`
- :doc:`parameters`
- :doc:`examples`
- :doc:`api`
