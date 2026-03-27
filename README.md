# coresg-graphhdbscan

Installable package version of the CoreSG + GraphHDBSCAN.

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
    min_samples=10,
    sim_graph_method="sc_umap",
    n_neighbors=10,
    no_noise=True,
    heuristic_connect=False,
    # min_cluster_size defaults to match each selected min_samples value when omitted
    # metric defaults to "euclidean" and can also be "cosine" or "hybrid_euclidean_cosine"
)

model.fit(X)
labels = model.labels_for(5)
model.plot_condensed_tree(10)
```

## Notes

- Several graph construction modes require optional scientific Python dependencies such as `scanpy`, `umap-learn`, and `phenograph`. 



## Graph options

Only these graph builders are provided in this version but users can pass their own similarity graph when they set sim_graph_method to 'precomputed':
- `sc_gauss`
- `sc_umap`
- `jaccard_phenograph`

Metric behavior:
- `metric="euclidean"` (default): full distances and graph neighbors use Euclidean distances
- `metric="cosine"`: full distances and graph neighbors use cosine distances
- `metric="hybrid_euclidean_cosine"`: full distances stay Euclidean, while the similarity-graph neighbor search uses cosine distances

Min-samples behavior:
- `min_samples=10` by default, so the internal `m_list` becomes `[10]`
- `min_samples=7` makes `m_list=[7]`
- `min_samples=[5, 10, 15]` makes `m_list=[5, 10, 15]`
- the internal `m_list` is derived from `min_samples` 


Current public constructor parameters:
- `min_samples`
- `sim_graph_method`
- `metric`
- `add_neighbor`
- `no_noise`
- `n_neighbors`
- `heuristic_connect`
- `min_cluster_size`


## Precomputed graph input

You can alsp pass an already-built similarity graph with `sim_graph_method="precomputed"`.
The input to `fit(...)` may be a NetworkX graph, a scipy sparse adjacency matrix, or a square dense adjacency matrix.

