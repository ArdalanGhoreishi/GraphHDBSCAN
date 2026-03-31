Parameter selection
===================

This page explains how to choose the main parameters of
``GraphCoreSGHDBSCAN`` in practice.

For most users, the most important decisions are:

- ``min_samples``
- ``sim_graph_method``
- ``metric``
- ``n_neighbors``
- ``heuristic_connect``
- ``no_noise``
- ``min_cluster_size``

Start here
----------

A good default starting point for many datasets is:

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

A simple way to think about the main settings is:

- use ``min_samples`` to control how conservative clustering should be
- use ``sim_graph_method`` to choose how the similarity graph is built
- use ``metric`` to choose the geometry used during graph construction
- use ``n_neighbors`` to control local graph density
- use ``heuristic_connect`` to decide how disconnected graphs are handled
- use ``no_noise`` to decide whether noise points should be reassigned

Constructor
-----------

The public constructor is:

.. code-block:: python

   GraphCoreSGHDBSCAN(
       min_samples=10,
       sim_graph_method="sc_umap",
       metric="euclidean",
       add_neighbor=True,
       no_noise=True,
       n_neighbors=15,
       heuristic_connect=False,
       min_cluster_size=None,
       **kwargs
   )

At-a-glance reference
---------------------

.. list-table::
   :header-rows: 1
   :widths: 18 14 68

   * - Parameter
     - Default
     - Practical meaning
   * - ``min_samples``
     - ``10``
     - Main clustering hyperparameter. Controls how conservative the density requirement is.
   * - ``sim_graph_method``
     - ``"sc_umap"``
     - Chooses how the similarity graph is built.
   * - ``metric``
     - ``"euclidean"``
     - Chooses the geometry used during graph construction.
   * - ``add_neighbor``
     - ``True``
     - Controls how weighted structural similarity is expanded into graph edges.
   * - ``no_noise``
     - ``True``
     - Reassigns points initially labeled ``-1`` after clustering.
   * - ``n_neighbors``
     - ``15``
     - Controls local graph density.
   * - ``heuristic_connect``
     - ``False``
     - Chooses how disconnected graphs are made connected.
   * - ``min_cluster_size``
     - ``None``
     - Minimum cluster size in the clustering stage.

How to choose each parameter
----------------------------

``min_samples``
^^^^^^^^^^^^^^^

Default: ``10``

This is the main clustering hyperparameter. It may be:

- a single integer, such as ``10``
- an iterable of integers, such as ``[5, 10, 15]`` or ``range(2, 10)``

Internally, the package converts it into an internal list of values used by
CoreSGHDBSCAN.

Examples:

- ``min_samples=10`` gives ``[10]``
- ``min_samples=7`` gives ``[7]``
- ``min_samples=[5, 10, 15]`` gives ``[5, 10, 15]``
- ``min_samples=range(2, 10)`` gives ``[2, 3, 4, 5, 6, 7, 8, 9]``

Practical interpretation:

- smaller values usually produce finer, more local cluster structure
- larger values usually produce more conservative and more stable clusters
- multiple values are useful when you want to compare density settings in one run

Recommended workflow:

1. Start with ``10``.
2. If clusters seem too coarse, try smaller values.
3. If clusters seem unstable or fragmented, try larger values.
4. When in doubt, fit several values and compare the condensed trees.

Example:

.. code-block:: python

   model = GraphCoreSGHDBSCAN(min_samples=[5, 10, 15])
   model.fit(X)

   labels_5 = model.labels_for(5)
   labels_10 = model.labels_for(10)
   labels_15 = model.labels_for(15)

``sim_graph_method``
^^^^^^^^^^^^^^^^^^^^

Default: ``"sc_umap"``

This parameter chooses the graph-construction backend.

Supported values are:

- ``"sc_gauss"``
- ``"sc_umap"``
- ``"jaccard_phenograph"``
- ``"precomputed"``

Choosing a method:

``sc_umap``
   Good default choice. Uses Scanpy's UMAP-style connectivity routine.

``sc_gauss``
   Useful when you want Scanpy's Gaussian connectivity construction.

``jaccard_phenograph``
   Useful when you want a PhenoGraph-style graph construction.

``precomputed``
   Use this when you already have a graph or adjacency representation and do
   not want the package to build a graph from raw features.

Supported inputs in ``precomputed`` mode:

- a ``networkx.Graph``
- a SciPy sparse adjacency matrix
- a square dense adjacency matrix

Practical recommendation:

- start with ``"sc_umap"``
- try ``"sc_gauss"`` if you prefer Gaussian connectivity
- use ``"jaccard_phenograph"`` for PhenoGraph-style neighborhood structure
- use ``"precomputed"`` when your graph is part of the experimental design

``metric``
^^^^^^^^^^

Default: ``"euclidean"``

This controls the metric strategy used during graph construction.

Supported values are:

- ``"euclidean"``
- ``"cosine"``
- ``"hybrid_euclidean_cosine"``

Choosing a metric:

``euclidean``
   Default mode. Euclidean distances are used for distance computation and
   graph construction.

``cosine``
   Cosine distances are used for graph construction. This is often useful
   when angular similarity is more meaningful than raw Euclidean distance.

``hybrid_euclidean_cosine``
   Full pairwise distances remain Euclidean, but the neighborhood graph is
   constructed using cosine geometry. This is useful when you want global
   geometry to remain Euclidean while local neighborhood structure follows
   angular similarity.

Practical recommendation:

- use ``"euclidean"`` for standard geometric tabular data
- use ``"cosine"`` when direction matters more than magnitude
- use ``"hybrid_euclidean_cosine"`` when you want Euclidean global distances
  but cosine-based local neighborhoods

Example:

.. code-block:: python

   model = GraphCoreSGHDBSCAN(
       min_samples=range(2, 10),
       sim_graph_method="sc_umap",
       metric="hybrid_euclidean_cosine",
       n_neighbors=16,
   )

``n_neighbors``
^^^^^^^^^^^^^^^

Default: ``15``

This is the number of neighbors used during similarity graph construction.

Practical interpretation:

- smaller values make the graph more local and sparse
- larger values make the graph denser and may improve connectivity
- increasing this value is often the first thing to try when the graph is too fragmented

Practical recommendation:

- start with ``15``
- increase it if connectivity is poor
- decrease it if the graph becomes overly broad or too smoothed

Example:

.. code-block:: python

   GraphCoreSGHDBSCAN(
       sim_graph_method="sc_gauss",
       n_neighbors=20,
   )

``add_neighbor``
^^^^^^^^^^^^^^^^

Default: ``True``

This controls how weighted structural similarity is computed.

When enabled, an edge may still be added even when two nodes do not already
share a direct edge, as long as their weighted structural similarity is greater
than zero.

Practical recommendation:

- keep the default unless you are specifically studying graph-construction behavior
- change it only when you want to examine the effect of this edge-expansion step

``heuristic_connect``
^^^^^^^^^^^^^^^^^^^^^

Default: ``False``

The final graph used for clustering must be connected. This parameter
controls how disconnected graphs are handled.

``heuristic_connect=False``
   Default behavior. If the graph has multiple connected components, the
   package connects consecutive components by adding edges with maximum
   distance, equivalent to weight ``1`` in the dissimilarity graph.

``heuristic_connect=True``
   The package repeatedly increases ``n_neighbors`` until the graph becomes
   connected.

Example fitting log:

.. code-block:: text

   Trying n_neighbors = 16
   Trying n_neighbors = 17

Practical recommendation:

- use ``False`` when you want a simple and predictable fallback
- use ``True`` when you prefer connectivity to come from a denser neighborhood graph
  rather than from synthetic bridge edges

``no_noise``
^^^^^^^^^^^^

Default: ``True``

If enabled, points initially labeled as ``-1`` are reassigned by an
MST-based label propagation step after clustering.

Conceptually, this post-processing step:

1. builds a mutual-reachability view from graph-derived distances and core distances
2. computes an MST
3. propagates labels from labeled points to unlabeled points in increasing edge-weight order
4. resolves competition using a top-``c`` path comparison rule

Practical recommendation:

- use ``True`` if you prefer a full assignment with no final noise labels
- use ``False`` if you want to preserve the original HDBSCAN*-style noise behavior

``min_cluster_size``
^^^^^^^^^^^^^^^^^^^^

Default: ``None``

This is the minimum cluster size used in the clustering stage.

When left as ``None``, the package follows the selected ``min_samples``
value for each run.

Practical recommendation:

- leave it as ``None`` if you want cluster size to track ``min_samples``
- set it explicitly if you want a fixed minimum cluster size independent of
  the selected ``min_samples`` values

Practical selection workflow
----------------------------

A useful tuning order is:

1. choose ``sim_graph_method`` based on how you want the graph to be built
2. choose ``metric`` based on the geometry that makes sense for your data
3. start with ``n_neighbors=15``
4. tune ``min_samples``
5. decide whether you want ``no_noise=True``
6. only then adjust ``heuristic_connect`` and ``add_neighbor`` if needed

A good exploratory run looks like:

.. code-block:: python

   g = GraphCoreSGHDBSCAN(
       min_samples=range(2, 20),
       sim_graph_method="sc_gauss",
       n_neighbors=16,
       no_noise=True,
       metric="euclidean",
       heuristic_connect=True,
   )
   g.fit(X)

Then inspect the hierarchy and choose a specific solution:

.. code-block:: python

   g.plot_condensed_tree(4)
   labels_18 = g.labels_for(18)

Ready-to-use presets
--------------------

Default baseline
^^^^^^^^^^^^^^^^

.. code-block:: python

   model = GraphCoreSGHDBSCAN()
   model.fit(X)
   labels = model.fit_predict(X)

More conservative clustering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   model = GraphCoreSGHDBSCAN(
       min_samples=20,
       sim_graph_method="sc_umap",
       metric="euclidean",
       n_neighbors=20,
   )

Finer local structure
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   model = GraphCoreSGHDBSCAN(
       min_samples=5,
       sim_graph_method="sc_umap",
       metric="euclidean",
       n_neighbors=12,
   )

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

Hybrid Euclidean-cosine mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   model = GraphCoreSGHDBSCAN(
       min_samples=range(2, 10),
       sim_graph_method="sc_umap",
       metric="hybrid_euclidean_cosine",
       n_neighbors=16,
   )
   model.fit(X)

Precomputed graph input
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   model = GraphCoreSGHDBSCAN(
       min_samples=10,
       sim_graph_method="precomputed",
       no_noise=True,
   )
   model.fit(my_graph)

Troubleshooting by symptom
--------------------------

Too many tiny clusters
^^^^^^^^^^^^^^^^^^^^^^

Try:

- increasing ``min_samples``
- increasing ``n_neighbors``
- using ``metric="euclidean"`` if cosine-based neighborhoods are too fine

Clusters are too coarse
^^^^^^^^^^^^^^^^^^^^^^^

Try:

- decreasing ``min_samples``
- decreasing ``n_neighbors``
- checking whether ``no_noise=True`` is absorbing points you would rather keep as noise

Graph is disconnected
^^^^^^^^^^^^^^^^^^^^^

Try:

- increasing ``n_neighbors``
- setting ``heuristic_connect=True``
- checking whether the selected metric is making neighborhoods too sparse

Too many noise points
^^^^^^^^^^^^^^^^^^^^^

Try:

- lowering ``min_samples``
- increasing ``n_neighbors``
- using ``no_noise=True`` if a full assignment is desired

Practical notes
---------------

- If the graph is disconnected and ``heuristic_connect=False``, the package
  connects components with synthetic edges of weight ``1``. This is simple and
  effective, but it is a design choice worth reporting in experiments.
- ``min_cluster_size=None`` means that the package matches cluster size to each
  selected ``min_samples`` value.
- When several ``min_samples`` values are passed, fit once and retrieve labels
  later for the requested value.
- Some graph builders depend on optional packages and will raise a clear import
  error if those packages are not installed.
