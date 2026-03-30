Overview
========

``coresg-graphhdbscan`` is a Python package for density-based clustering on
graphs derived from tabular data or provided directly by the user.

The package combines two main ideas:

1. construction of a weighted graph that reflects local similarity structure
2. application of a CoreSG-HDBSCAN pipeline on a graph-derived distance representation

This package is designed for settings where Euclidean geometry alone is not the
desired representation, and where clustering should instead be guided by a
learned or hand-crafted similarity graph.

Main features
-------------

The package currently supports the following capabilities:

- graph-based clustering through ``GraphCoreSGHDBSCAN``
- multiple ``min_samples`` values in a single model run
- three graph-construction backends plus a precomputed graph mode
- three metric modes: ``euclidean``, ``cosine``, and ``hybrid_euclidean_cosine``
- optional relabeling of noise points by MST-based propagation
- compatibility with NetworkX graphs, dense adjacency matrices, and sparse adjacency matrices in precomputed mode
- access to HDBSCAN-style outputs such as labels, probabilities, and condensed trees

Package structure
-----------------

The package is centered around two public classes:

``CoreSGHDBSCAN``
   The lower-level CoreSG implementation operating on feature vectors or
   distance matrices.

``GraphCoreSGHDBSCAN``
   The graph-oriented wrapper that constructs a graph, converts it to a
   graph-derived distance representation, and then runs CoreSG-HDBSCAN.

For most users, ``GraphCoreSGHDBSCAN`` is the main entry point.

When to use this package
------------------------

This package is especially useful when:

- Euclidean distance alone is not the best description of local structure
- a similarity graph is more meaningful than a raw feature-space view
- you want to compare several ``min_samples`` values efficiently in one run
- you want HDBSCAN-style hierarchical clustering behavior on top of a graph-based representation
- you already have a graph or adjacency matrix and want to cluster directly from it

Typical workflow
----------------

A typical graph-based clustering workflow in this package is:

1. construct or provide a similarity graph
2. convert it into a weighted structural similarity graph
3. convert similarity to dissimilarity
4. ensure graph connectivity
5. run CoreSG-HDBSCAN for one or more ``min_samples`` values
6. inspect the condensed tree and choose a solution
7. optionally reassign noise points if ``no_noise=True``

Typical entry point
-------------------

For most users, the main entry point is:

.. code-block:: python

   from coresg_graphhdbscan import GraphCoreSGHDBSCAN

A simple starting example is:

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

Related pages
-------------

For more detail, see:

- :doc:`installation`
- :doc:`usage`
- :doc:`parameters`
- :doc:`api`
