Third-party notices
===================

``coresg-graphhdbscan`` uses third-party open-source software for numerical
computation, graph processing, clustering, and graph construction.

Third-party packages
--------------------

This package relies on the following major libraries:

- ``numpy``
- ``scipy``
- ``scikit-learn``
- ``pandas``
- ``matplotlib``
- ``networkx``
- ``hdbscan``
- ``scanpy``
- ``PhenoGraph``
- ``umap-learn``

How they are used
-----------------

``numpy``
   Numerical array operations and matrix-based computation.

``scipy``
   Scientific computing utilities, sparse matrices, and distance-related tools.

``scikit-learn``
   Neighbor search, clustering-related helpers, and machine-learning utilities.

``pandas``
   Tabular data handling in workflows and examples.

``matplotlib``
   Static plotting for visual outputs such as condensed trees.

``networkx``
   Graph representation and graph-based processing, including precomputed graph workflows.

``hdbscan``
   HDBSCAN-style clustering outputs and related hierarchical clustering behavior.

``scanpy``
   Graph construction backends such as ``sc_gauss`` and ``sc_umap``.

``PhenoGraph``
   Graph construction through the ``jaccard_phenograph`` backend.

``umap-learn``
   UMAP-related neighbor graph utilities used through the graph-construction stack.

License notice
--------------

These dependencies remain the property of their respective authors and are
distributed under their own license terms.

Users who redistribute this package should ensure that they comply with the
license terms of all included and required third-party software.

Method attribution
------------------

If your workflow uses features built on top of ``hdbscan``, ``scanpy``, or
``PhenoGraph``, those software packages and related methods should be cited or
acknowledged where appropriate.

Summary
-------

This page is an informational summary of third-party software used by
``coresg-graphhdbscan``. The authoritative license terms for each dependency are
the ones provided by the original upstream projects.