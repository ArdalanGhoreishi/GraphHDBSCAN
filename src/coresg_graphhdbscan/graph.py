"""Graph-based wrapper around CoreSG-HDBSCAN."""

from .core import CoreSGHDBSCAN

import importlib

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hdbscan
from scipy.spatial.distance import cdist
from scipy.spatial import distance
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.cluster import HDBSCAN
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors as NN, kneighbors_graph
import heapq
from collections.abc import Iterable


def _optional_import(module_name, package_name=None):
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        pkg = package_name or module_name
        raise ImportError(f"Optional dependency '{pkg}' is required for this functionality. Please install it.") from e


def _get_scanpy_modules():
    sc = _optional_import("scanpy")
    sce = _optional_import("scanpy.external")
    sc_neighbors_connectivity = _optional_import("scanpy.neighbors._connectivity")
    sc_neighbors_common = _optional_import("scanpy.neighbors._common")
    return (
        sc,
        sce,
        sc_neighbors_connectivity.gauss,
        sc_neighbors_connectivity.umap,
        sc_neighbors_common._get_indices_distances_from_dense_matrix,
    )


class GraphCoreSGHDBSCAN(CoreSGHDBSCAN):
    def __init__(self,
                 min_samples=10,
                 sim_graph_method='sc_umap',
                 metric='euclidean',
                 add_neighbor=True,
                 no_noise=True,
                 n_neighbors=15,
                 heuristic_connect=False,
                 min_cluster_size=None,
                 **kwargs):

        # store graph params
        valid_graph_methods = {'sc_gauss', 'sc_umap', 'jaccard_phenograph', 'precomputed'}
        if sim_graph_method not in valid_graph_methods:
            raise ValueError(
                f"Unsupported sim_graph_method '{sim_graph_method}'. "
                f"Use one of {sorted(valid_graph_methods)}."
            )
        valid_metrics = {'euclidean', 'cosine', 'hybrid_euclidean_cosine'}
        if metric not in valid_metrics:
            raise ValueError(
                f"Unsupported metric '{metric}'. "
                f"Use one of {sorted(valid_metrics)}."
            )

        self.sim_graph_method = sim_graph_method
        self.metric = metric
        self.add_neighbor = add_neighbor
        self.no_noise = no_noise
        self.n_neighbors = n_neighbors
        if 'mst_approx' in kwargs:
            heuristic_connect = kwargs.pop('mst_approx')
        self.heuristic_connect = bool(heuristic_connect)

        # Backward-compatible handling of removed parameters.
        kwargs.pop('force_connected', None)
        kwargs.pop('gamma', None)
        kwargs.pop('min_dist', None)

        # ``m_list`` is now internal rather than a public hyperparameter.
        # Keep a legacy escape hatch through kwargs only.
        legacy_m_list = kwargs.pop('m_list', None)
        if legacy_m_list is not None:
            resolved_m_list = list(legacy_m_list)
        elif isinstance(min_samples, Iterable) and not isinstance(min_samples, (str, bytes, np.str_)):
            resolved_m_list = list(min_samples)
        else:
            resolved_m_list = [int(min_samples)]

        if len(resolved_m_list) == 0:
            raise ValueError("min_samples must define at least one value.")

        self.m_list = [int(m) for m in resolved_m_list]
        self.min_samples = list(self.m_list) if len(self.m_list) > 1 else int(self.m_list[0])

        resolved_min_cluster_size = None if min_cluster_size is None else int(min_cluster_size)

        core_metric = 'euclidean' if metric == 'hybrid_euclidean_cosine' else metric
        super().__init__(
            min_samples_list=self.m_list,
            metric=core_metric,
            min_cluster_size=resolved_min_cluster_size,
            **kwargs
        )
        self.min_cluster_size = resolved_min_cluster_size

    def _min_cluster_size_for(self, m):
        m = int(m)
        return m if self.min_cluster_size is None else int(self.min_cluster_size)


    @staticmethod
    def compute_tsne_affinities(data, perplexity, tol=1e-5, max_iter=50):
        """
        Compute the conditional probability matrix (affinities) as used in t-SNE.

        Parameters:
          data: numpy array (n_samples x n_features)
          perplexity: target perplexity (a positive float)
          tol: tolerance for the binary search (default: 1e-5)
          max_iter: maximum iterations of binary search per point

        Returns:
          P: numpy array (n x n) with conditional probabilities (with zeros on diagonal)
        """
        n = data.shape[0]
        # Compute pairwise squared Euclidean distances.
        distances = pairwise_distances(data, squared=True)
        P = np.zeros((n, n))
        logU = np.log(perplexity)

        # Loop over all data points.
        for i in range(n):
            # Exclude the i-th element.
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            Di = distances[i, mask]

            # Initialize binary search for beta = 1/(2*sigma^2)
            beta = 1.0
            betamin = -np.inf
            betamax = np.inf

            # Binary search to get the correct entropy.
            for _ in range(max_iter):
                # Compute the conditional probabilities for point i.
                P_i = np.exp(-Di * beta)
                sumP_i = np.sum(P_i)
                if sumP_i == 0:
                    sumP_i = 1e-10
                P_i = P_i / sumP_i
                # Shannon entropy in natural log.
                H = -np.sum(P_i * np.log(P_i + 1e-10))
                Hdiff = H - logU
                # Check for convergence.
                if np.abs(Hdiff) < tol:
                    break
                # Adjust beta based on whether the entropy is too high or too low.
                if Hdiff > 0:  # entropy too high, increase beta (reduce sigma)
                    betamin = beta
                    beta = beta * 2.0 if betamax == np.inf else (beta + betamax) / 2.0
                else:          # entropy too low, decrease beta (increase sigma)
                    betamax = beta
                    beta = beta / 2.0 if betamin == -np.inf else (beta + betamin) / 2.0

            # Fill the conditional probabilities for the i-th row.
            # Note: Insert zeros for the self-affinity.
            inds = np.arange(n)[mask]
            P[i, inds] = P_i

        return P

    def compute_similarity(self, graph):
        if self.add_neighbor == True:
            new_graph = nx.Graph()
            nodes = list(graph.nodes())
            for i, u in enumerate(nodes):
                for v in nodes[i+1:]:
                    neighbors_u = set(graph.neighbors(u)) | {u}
                    neighbors_v = set(graph.neighbors(v)) | {v}
                    common_neighbors = neighbors_u & neighbors_v
                    numerator = 0
                    for x in common_neighbors:
                        weight_ux = graph[u][x]['weight'] if graph.has_edge(u, x) else (1 if u == x else 0)
                        weight_vx = graph[v][x]['weight'] if graph.has_edge(v, x) else (1 if v == x else 0)
                        numerator += weight_ux * weight_vx
                    sum_u = sum((graph[u][x]['weight'] if graph.has_edge(u, x) else (1 if u == x else 0)) ** 2 for x in neighbors_u)
                    sum_v = sum((graph[v][x]['weight'] if graph.has_edge(v, x) else (1 if v == x else 0)) ** 2 for x in neighbors_v)
                    denominator = np.sqrt(sum_u) * np.sqrt(sum_v)
                    similarity = numerator / denominator if denominator != 0 else 0
                    if similarity > 0:
                        new_graph.add_edge(u, v, weight=similarity)
            return new_graph
        else:
            new_graph = nx.Graph()
            for u, v in graph.edges():
                neighbors_u = set(graph.neighbors(u)) | {u}
                neighbors_v = set(graph.neighbors(v)) | {v}
                common_neighbors = neighbors_u & neighbors_v
                numerator = 0
                for x in common_neighbors:
                    weight_ux = graph[u][x]['weight'] if graph.has_edge(u, x) else (1 if u == x else 0)
                    weight_vx = graph[v][x]['weight'] if graph.has_edge(v, x) else (1 if v == x else 0)
                    numerator += weight_ux * weight_vx
                sum_u = sum((graph[u][x]['weight'] if graph.has_edge(u, x) else (1 if u == x else 0)) ** 2 for x in neighbors_u)
                sum_v = sum((graph[v][x]['weight'] if graph.has_edge(v, x) else (1 if v == x else 0)) ** 2 for x in neighbors_v)
                denominator = np.sqrt(sum_u) * np.sqrt(sum_v)
                similarity = numerator / denominator if denominator != 0 else 0
                new_graph.add_edge(u, v, weight=similarity)
            return new_graph

    @staticmethod
    def similarity_to_dissimilarity(similarity_graph):
        dissimilarity_graph = nx.Graph()
        for u, v, data in similarity_graph.edges(data=True):
            dissimilarity_graph.add_edge(u, v, weight=1 - data['weight'])
        return dissimilarity_graph

    @staticmethod
    def is_graph_connected(graph):
        return nx.is_connected(graph)

    @staticmethod
    def _coerce_precomputed_graph(graph_like):
        """Convert a supported precomputed graph representation into a NetworkX graph."""
        if isinstance(graph_like, nx.Graph):
            graph = graph_like.copy()
        elif hasattr(graph_like, 'tocoo'):
            graph = nx.from_scipy_sparse_array(graph_like, edge_attribute='weight')
        else:
            arr = np.asarray(graph_like)
            if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
                raise ValueError(
                    "For sim_graph_method='precomputed', input must be a NetworkX graph, "
                    "a scipy sparse adjacency matrix, or a square dense adjacency matrix."
                )
            graph = nx.from_numpy_array(arr)

        graph.remove_edges_from([(u, v) for u, v, d in graph.edges(data=True) if d.get('weight', 0) == 0])
        return graph

    def create_similarity_graph(self, data):
        if self.sim_graph_method == 'precomputed':
            return self._coerce_precomputed_graph(data)

        sc, sce, sc_gauss, sc_umap, _get_indices_distances_from_dense_matrix = _get_scanpy_modules()

        X = data.toarray() if hasattr(data, "toarray") else np.asarray(data)
        if X.ndim != 2:
            raise ValueError("Input data must be a 2D array-like object.")

        if self.metric == 'hybrid_euclidean_cosine':
            distances_full = pairwise_distances(X, metric='euclidean')
            knn_metric = 'cosine'
            use_precomputed_knn = False
        else:
            distances_full = pairwise_distances(X, metric=self.metric)
            knn_metric = 'precomputed'
            use_precomputed_knn = True

        self.distances_full_ = distances_full

        if self.sim_graph_method == 'jaccard_phenograph':
            if use_precomputed_knn:
                knn_dist = kneighbors_graph(
                    distances_full,
                    n_neighbors=self.n_neighbors - 1,
                    mode='distance',
                    metric='precomputed',
                    include_self=False,
                )
            else:
                knn_dist = kneighbors_graph(
                    X,
                    n_neighbors=self.n_neighbors - 1,
                    mode='distance',
                    metric=knn_metric,
                    include_self=False,
                )

            _, conn, _ = sce.tl.phenograph(
                knn_dist,
                directed=False,
                clustering_algo=None,
            )
            return nx.from_scipy_sparse_array(conn.tocsr(), edge_attribute='weight')

        if self.sim_graph_method == 'sc_gauss':
            if use_precomputed_knn:
                knn_dist = kneighbors_graph(
                    distances_full,
                    n_neighbors=self.n_neighbors - 1,
                    mode='distance',
                    metric='precomputed',
                    include_self=False,
                )
            else:
                knn_dist = kneighbors_graph(
                    X,
                    n_neighbors=self.n_neighbors - 1,
                    mode='distance',
                    metric=knn_metric,
                    include_self=False,
                )
            conn = sc_gauss(knn_dist, n_neighbors=self.n_neighbors, knn=True)
            return nx.from_scipy_sparse_array(conn, edge_attribute='weight')

        if self.sim_graph_method == 'sc_umap':
            if use_precomputed_knn:
                idx, dists = _get_indices_distances_from_dense_matrix(
                    distances_full, self.n_neighbors
                )
            else:
                nn = NN(n_neighbors=self.n_neighbors, metric=knn_metric)
                nn.fit(X)
                dists, idx = nn.kneighbors(X, return_distance=True)
            conn = sc_umap(
                idx,
                dists,
                n_obs=distances_full.shape[0],
                n_neighbors=self.n_neighbors,
            )
            return nx.from_scipy_sparse_array(conn, edge_attribute='weight')

        raise ValueError(
            "Unsupported sim_graph_method. Use one of 'sc_gauss', 'sc_umap', 'jaccard_phenograph', or 'precomputed'."
        )

    def connect_graph_heuristically(self, similarity_graph, data):
        """Connect disconnected graph components using one of two simple strategies.

        - heuristic_connect=True: increase n_neighbors until the graph becomes connected.
        - heuristic_connect=False: connect consecutive components with bridging edges of weight 1.
        """
        new_graph = similarity_graph.copy()
        new_graph.add_nodes_from(range(len(data)))
        if nx.is_connected(new_graph):
            return new_graph

        if self.heuristic_connect:
            original_n_neighbors = self.n_neighbors
            new_n_neighbors = self.n_neighbors
            while True:
                new_n_neighbors += 1
                print("Trying n_neighbors =", new_n_neighbors)
                self.n_neighbors = new_n_neighbors
                new_graph = self.create_similarity_graph(data)
                new_graph.add_nodes_from(range(len(data)))
                if nx.is_connected(new_graph):
                    break
            return new_graph

        components = list(nx.connected_components(new_graph))
        for i in range(len(components) - 1):
            u = next(iter(components[i]))
            v = next(iter(components[i + 1]))
            new_graph.add_edge(u, v, weight=1)
        return new_graph

    @staticmethod
    def compute_full_distance_matrix(graph):
        """
        Compute the full dense matrix of shortest path distances using Floyd–Warshall.
        """
        return np.array(nx.floyd_warshall_numpy(graph, weight='weight'))

    @staticmethod
    def compute_sparse_distance_dict(graph):
        """
        Compute a dictionary-of-dictionaries of shortest path distances.
        For each node, run single_source_dijkstra_path_length and store the results.
        """
        distance_dict = {}
        for node in graph.nodes():
            # Compute shortest path lengths from 'node' to all others.
            distance_dict[node] = nx.single_source_dijkstra_path_length(graph, node, weight='weight')
        return distance_dict

    def graph_metric(self, u, v):
        """
        Custom distance metric that uses the precomputed sparse distance dictionary.
        The data points are mapped to node indices using self._point_to_index.
        """
        idx_u = self._point_to_index.get(tuple(u))
        idx_v = self._point_to_index.get(tuple(v))
        try:
            return self.distance_dict_[idx_u][idx_v]
        except KeyError:
            # Since the graph is connected, this should rarely happen.
            # Check the reverse ordering as a fallback.
            return self.distance_dict_[idx_v][idx_u]

    @staticmethod
    def compute_custom_distance_matrix(graph):
        """
        Compute a dense distance matrix where for each pair of nodes:
          - The distance is the edge weight if an edge exists.
          - Otherwise, the distance is set to infinity.
        The diagonal is set to 0.
        """
        n = graph.number_of_nodes()
        dist = np.full((n, n), 1, dtype=np.float64)
        np.fill_diagonal(dist, 0)
        for u, v, data in graph.edges(data=True):
            weight = data['weight']
            dist[u, v] = weight
            dist[v, u] = weight
        return dist


    @staticmethod
    def reassign_noise_via_mst(mst_graph, labels0, c=5):
        """
        Reassign noise labels by propagating labels over a precomputed MST.

        Parameters
        ----------
        mst_graph : networkx.Graph
            Minimum spanning tree of the final connected WSS graph.
        labels0 : ndarray
            Initial labels with noise marked as -1.
        c : int, default=5
            Number of largest edge weights to keep in the lexicographic path
            signature during propagation.
        """
        if not isinstance(mst_graph, nx.Graph):
            raise TypeError("mst_graph must be a networkx.Graph.")

        n = len(labels0)
        labels = np.asarray(labels0).copy()
        if mst_graph.number_of_nodes() != n:
            raise ValueError("mst_graph and labels0 must have the same number of nodes.")

        # adjacency of the tree
        adj = [[] for _ in range(n)]
        for u, v, data in mst_graph.edges(data=True):
            w = float(data.get('weight', 1.0))
            adj[int(u)].append((int(v), w))
            adj[int(v)].append((int(u), w))

        # Multi-source propagation from labeled vertices.
        pq = []
        paths = [None] * n
        for u in range(n):
            if labels[u] != -1:
                paths[u] = [0.0] * c
                for v, w in adj[u]:
                    if labels[v] == -1:
                        heapq.heappush(pq, (w, u, v))

        while pq:
            w, u, v = heapq.heappop(pq)
            if labels[v] != -1:
                continue

            same = [(u, v)]
            while pq and pq[0][0] == w and pq[0][2] == v:
                _, u2, _ = heapq.heappop(pq)
                same.append((u2, v))

            def top_c_path(u_idx):
                vec = list(paths[u_idx]) + [w]
                return sorted(vec, reverse=True)[:c]

            candidates = [(top_c_path(u_idx), labels[u_idx]) for u_idx, _ in same]
            best_path, best_label = min(candidates, key=lambda x: tuple(x[0]))

            labels[v] = best_label
            paths[v] = best_path

            for nbr, w2 in adj[v]:
                if labels[nbr] == -1:
                    heapq.heappush(pq, (w2, v, nbr))

        return labels




    # ------------------------------------------------------------------
    # ------------------------ GRAPH PREPROCESSING ---------------------
    # ------------------------------------------------------------------

    def _build_graph_distance(self, X):
        """
        Build similarity graph → refined graph → connected graph → dense distance matrix.
        """
        self.data_ = X if self.sim_graph_method == 'precomputed' else (np.array(X) if isinstance(X, pd.DataFrame) else X)

        # similarity graph (your method)
        self.similarity_graph_ = self.create_similarity_graph(self.data_)

        # WSS graph
        self.similarity_graph_WSS = self.compute_similarity(self.similarity_graph_)

        # dissimilarity = 1 - similarity
        self.dissimilarity_graph_ = self.similarity_to_dissimilarity(self.similarity_graph_WSS)

        # ensure connectivity
        self.connected_graph_ = self.connect_graph_heuristically(
            self.dissimilarity_graph_, self.data_
        )

        # final dense matrix (for CoreSG)
        self.dist_matrix_ = self.compute_custom_distance_matrix(self.connected_graph_)
        self.mst_graph_ = nx.minimum_spanning_tree(self.connected_graph_, weight='weight')

    # ------------------------------------------------------------------
    # ------------------------- FIT ------------------------------------
    # ------------------------------------------------------------------

    def fit(self, X, y=None):
        """
        Build graph → dense distance matrix → run CoreSG on that matrix.
        """
        # 1) Build graph + D
        self.data_ = X if self.sim_graph_method == 'precomputed' else np.array(X)
        self.similarity_graph_ = self.create_similarity_graph(self.data_)
        self.similarity_graph_WSS = self.compute_similarity(self.similarity_graph_)
        self.dissimilarity_graph_ = self.similarity_to_dissimilarity(self.similarity_graph_WSS)

        self.connected_graph_ = self.connect_graph_heuristically(
            self.dissimilarity_graph_, self.data_
        )

        # edge-based dense matrix used by CoreSG/HDBSCAN
        self.dist_matrix_ = self.compute_custom_distance_matrix(self.connected_graph_)
        self.mst_graph_ = nx.minimum_spanning_tree(self.connected_graph_, weight='weight')

        # 2) Create CoreSG wrapper
        self.coresg_ = CoreSGHDBSCAN(
            min_samples_list=self.m_list,   # stored in __init__
            metric="precomputed",
            min_cluster_size=self.min_cluster_size,
        )

        # 3) Fit Core-SG **using the graph distances**
        self.coresg_.fit_from_distance_matrix(self.dist_matrix_)

        # 4) Run HDBSCAN pipeline for all m
        self.coresg_.run()

        return self




    def fit_predict(self, X, y=None, m=None, c=5, **fit_params):
        self.fit(X, y, **fit_params)

        if m is None:
            if len(self.m_list) != 1:
                raise ValueError(
                    "fit_predict requires `m` when m_list contains multiple values. "
                    "Use labels_for(m) or pass m=... explicitly."
                )
            m = self.m_list[0]

        labels = self.coresg_.models_[int(m)].labels_

        if self.no_noise:
            return self.reassign_noise_via_mst(
                self.mst_graph_,
                labels,
                c=c,
            )
        return labels

    def fit_coresg(self, X, m_list, coresg_kwargs=None):
        """
        1) Build graph and dense distance matrix self.dist_matrix_
        2) Run CoreSGHDBSCAN on that distance matrix
        """
        # --- 1) existing graph building ---
        self.data_ = np.array(X) if isinstance(X, pd.DataFrame) else X
        self.similarity_graph_ = self.create_similarity_graph(self.data_)
        self.similarity_graph_WSS = self.compute_similarity(self.similarity_graph_)
        self.dissimilarity_graph_ = self.similarity_to_dissimilarity(self.similarity_graph_WSS)

        self.connected_graph_ = self.connect_graph_heuristically(
            self.dissimilarity_graph_, self.data_
        )

        self.dist_matrix_ = self.compute_custom_distance_matrix(
            self.connected_graph_
        )

        # --- 2) CoreSG on the graph distance matrix ---
        if coresg_kwargs is None:
            coresg_kwargs = {}

        self.coresg_ = CoreSGHDBSCAN(
            min_samples_list=list(m_list),
            min_cluster_size=self.min_cluster_size,
            **coresg_kwargs
        ).fit_from_distance_matrix(self.dist_matrix_)

        # run the full HDBSCAN pipeline for all m
        self.coresg_.run()

        return self


   # ------------------------------------------------------------------
    # -------------------------- ACCESSORS ------------------------------
    # ------------------------------------------------------------------

    def labels_for(self, m, no_noise=None, c=5):
        labels = self.coresg_.models_[m].labels_

        if no_noise is None:
            no_noise = self.no_noise

        if no_noise:
            labels = self.reassign_noise_via_mst(
                self.mst_graph_,
                labels,
                c=c,
            )
        return labels


    def plot_condensed_tree(self, m=None, ax=None, figsize=(10, 6), **kwargs):
        """Plot the condensed tree for a previously fitted ``m`` value."""
        if not hasattr(self, "coresg_"):
            raise RuntimeError("Call fit(...) first.")

        if m is None:
            m = getattr(self, "min_samples", None)
        if m is None:
            raise ValueError("No m was provided and self.min_samples is not set.")

        m = int(m)
        core_model = self.coresg_.models_[m]

        if not hasattr(core_model, "condensed_tree_") or not hasattr(core_model.condensed_tree_, "plot"):
            raise AttributeError(f"No condensed tree available for m={m}.")

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        else:
            ax.clear()
            plt.sca(ax)

        # ``hdbscan`` CondensedTree.plot() often draws on the current axis and

        plt.sca(ax)
        core_model.condensed_tree_.plot(**kwargs)
        ax.set_title(f"Condensed Tree for m = {m}")
        return ax

    def interactive_condensed_tree(self, X=None, figsize=(10, 6)):
        """Interactive condensed-tree viewer that reuses the fitted CoreSG results.

        Parameters
        ----------
        X : ignored, optional
            Kept only for backward compatibility. The viewer uses the models
            already computed by ``fit`` and does not refit.
        figsize : tuple, default=(10, 6)
            Figure size for the plot.
        """
        try:
            import ipywidgets as widgets
            from IPython.display import display, clear_output
        except ImportError as e:
            raise ImportError(
                "ipywidgets is required for interactive plotting. "
                "Install it with `pip install ipywidgets`."
            ) from e

        if not hasattr(self, "coresg_"):
            raise RuntimeError("Call fit(...) before interactive_condensed_tree().")

        if not hasattr(self, "m_list") or self.m_list is None or len(self.m_list) == 0:
            raise ValueError("`self.m_list` must exist and must not be empty.")

        m_list = list(self.m_list)
        output = widgets.Output()

        slider = widgets.SelectionSlider(
            options=m_list,
            value=m_list[0],
            description="m",
            continuous_update=False,
            style={"description_width": "initial"},
            layout=widgets.Layout(width="500px"),
        )

        def redraw(m):
            with output:
                clear_output(wait=True)
                fig, ax = plt.subplots(figsize=figsize)
                self.plot_condensed_tree(m=int(m), ax=ax)
                plt.show()

        def on_change(change):
            if change["name"] == "value":
                redraw(change["new"])

        slider.observe(on_change, names="value")
        display(widgets.VBox([slider, output]))
        redraw(m_list[0])

        return slider