# coresg-graphhdbscan

Installable package version of the CoreSG + GraphHDBSCAN* implementation.

## Documentation

Full documentation is available here:

[GraphHDBSCAN* Documentation](https://graphhdbscan.readthedocs.io/en/latest/)

## Installation

### Install from GitHub

```bash
pip install git+https://github.com/Campello-Lab/GraphHDBSCAN.git
```

### Install locally

```bash
pip install -e .
```

## Minimal usage

```python
from coresg_graphhdbscan import GraphCoreSGHDBSCAN

model = GraphCoreSGHDBSCAN(
    min_samples=[5, 10, 15],
    sim_graph_method="sc_gauss",
    metric="euclidean",
    n_neighbors=10,
    no_noise=True,
    heuristic_connect=False,
    # min_cluster_size defaults to match each selected min_samples value when omitted
)

model.fit(X)

labels_5 = model.labels_for(5)
labels_10 = model.labels_for(10)
labels_15 = model.labels_for(15)

model.plot_condensed_tree(10)
```

## Graph options

The package provides the following built-in similarity graph construction methods:

- `sc_gauss`
- `sc_umap`
- `jaccard_phenograph`

Users can also pass their own similarity graph by setting:

```python
sim_graph_method="precomputed"
```

## Distance metrics

The `metric` parameter controls the distance measure used during similarity graph construction.

Supported metrics are:

- `cityblock`
- `cosine`
- `euclidean`
- `l1`
- `l2`
- `manhattan`
- `braycurtis`
- `canberra`
- `chebyshev`
- `correlation`
- `dice`
- `hamming`
- `jaccard`
- `mahalanobis`
- `minkowski`
- `rogerstanimoto`
- `russellrao`
- `seuclidean`
- `sokalmichener`
- `sokalsneath`
- `sqeuclidean`
- `yule`
- `hybrid_euclidean_cosine`

Metric behavior:

- `metric="euclidean"` is the default.
- `metric="cosine"` uses cosine distances during graph construction.
- `metric="correlation"` uses correlation distances during graph construction.
- `metric="manhattan"` and `metric="l1"` use Manhattan/L1 distances.
- `metric="minkowski"` supports additional options through `metric_kwds`, for example `metric_kwds={"p": 1.5}`.
- `metric="mahalanobis"` requires an inverse covariance matrix through `metric_kwds`, for example `metric_kwds={"VI": VI}`.
- `metric="seuclidean"` requires a variance vector through `metric_kwds`, for example `metric_kwds={"V": V}`.
- `metric="hybrid_euclidean_cosine"` is package-specific: full distances are Euclidean, while the similarity-graph neighbor search uses cosine distances.

The metric `kulsinski` is not supported because it is not available in current versions of `scikit-learn`'s `pairwise_distances`.

The following combination is intentionally not supported:

```python
GraphCoreSGHDBSCAN(
    sim_graph_method="sc_gauss",
    metric="yule",
)
```

because it can produce non-finite graph weights. Use `metric="yule"` with `sim_graph_method="sc_umap"` or `sim_graph_method="jaccard_phenograph"` instead.

## Examples with metric_kwds

### Minkowski distance

```python
model = GraphCoreSGHDBSCAN(
    min_samples=10,
    sim_graph_method="sc_umap",
    metric="minkowski",
    metric_kwds={"p": 1.5},
    n_neighbors=15,
)

model.fit(X)
```

### Mahalanobis distance

```python
import numpy as np

VI = np.linalg.pinv(np.cov(X, rowvar=False))

model = GraphCoreSGHDBSCAN(
    min_samples=10,
    sim_graph_method="sc_umap",
    metric="mahalanobis",
    metric_kwds={"VI": VI},
    n_neighbors=15,
)

model.fit(X)
```

### Standardized Euclidean distance

```python
import numpy as np

V = np.var(X, axis=0, ddof=1)

model = GraphCoreSGHDBSCAN(
    min_samples=10,
    sim_graph_method="sc_umap",
    metric="seuclidean",
    metric_kwds={"V": V},
    n_neighbors=15,
)

model.fit(X)
```

## Min-samples behavior

- `min_samples=10` by default, so the internal `m_list` becomes `[10]`.
- `min_samples=7` makes `m_list=[7]`.
- `min_samples=[5, 10, 15]` makes `m_list=[5, 10, 15]`.
- The internal `m_list` is derived from `min_samples`.

Example:

```python
model = GraphCoreSGHDBSCAN(
    min_samples=[5, 10, 15],
    sim_graph_method="sc_umap",
    metric="euclidean",
)

model.fit(X)

labels_5 = model.labels_for(5)
labels_10 = model.labels_for(10)
labels_15 = model.labels_for(15)
```

## Precomputed graph input

You can also pass an already-built similarity graph with:

```python
sim_graph_method="precomputed"
```

The input to `fit(...)` may be:

- a NetworkX graph
- a scipy sparse adjacency matrix
- a square dense adjacency matrix

Example:

```python
model = GraphCoreSGHDBSCAN(
    min_samples=10,
    sim_graph_method="precomputed",
)

model.fit(G)
labels = model.labels_for(10)
```

When `sim_graph_method="precomputed"`, the package starts from the provided similarity graph and uses it to construct the internal CoreSG / WSS representation used by GraphHDBSCAN*.

## Current public constructor parameters

```python
GraphCoreSGHDBSCAN(
    min_samples=10,
    sim_graph_method="sc_umap",
    metric="euclidean",
    metric_kwds=None,
    add_neighbor=True,
    no_noise=True,
    n_neighbors=15,
    heuristic_connect=False,
    min_cluster_size=None,
    save_models=False,
    similarity_backend="auto",
)
```

Parameter overview:

- `min_samples`: controls the density level or list of density levels used by GraphHDBSCAN*.
- `sim_graph_method`: chooses the similarity graph construction method.
- `metric`: chooses the distance metric used during graph construction.
- `metric_kwds`: optional keyword arguments passed to the selected distance metric.
- `add_neighbor`: controls whether additional neighbor information is added internally.
- `no_noise`: controls whether noise points are reassigned.
- `n_neighbors`: controls the neighborhood size used during similarity graph construction.
- `heuristic_connect`: controls whether a heuristic graph-connection step is used.
- `min_cluster_size`: minimum cluster size. If omitted, it defaults to the corresponding `min_samples` value.
- `similarity_backend`: backend used for similarity graph construction.

## Notes

Several graph construction modes require optional scientific Python dependencies, such as:

- `scanpy`
- `umap-learn`
- `phenograph`

## Related paper

The associated paper is available on bioRxiv:

[https://www.biorxiv.org/content/10.64898/2026.03.24.713924v1](https://www.biorxiv.org/content/10.64898/2026.03.24.713924v1)

To reproduce the results, first generate the input data by following the preprocessing instructions provided in the [Clustering-benchmarking-for-scRNAseq](https://github.com/Campello-Lab/Clustering-benchmarking-for-scRNAseq) repository.

## Third-party software

This package uses third-party open-source dependencies, including `hdbscan`, `scanpy`, and `PhenoGraph`.

See `THIRD_PARTY_NOTICES.md` for details.

## Contact

For questions or support, please open an issue or contact:

`ghoreishi@imada.sdu.dk`
