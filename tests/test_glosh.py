import numpy as np
import pytest
import hdbscan
from hdbscan._hdbscan_tree import outlier_scores as hdbscan_glosh
from sklearn.metrics import adjusted_rand_score
from coresg_graphhdbscan import GraphCoreSGHDBSCAN


def _toy_data(seed=0):
    rng = np.random.default_rng(seed)
    return np.vstack([
        rng.normal(0, 1, (80, 5)),
        rng.normal(8, 1, (80, 5)),
        rng.normal(4, 6, (5, 5)),   # a few diffuse outliers
    ])


def test_glosh_is_exact_reference_on_same_tree():
    """Definitive integration test: the accessor returns exactly the official
    hdbscan GLOSH on GraphHDBSCAN*'s own condensed tree (atol=0). Guards the
    wiring -- that the correct stored tree is fed to the reference function."""
    X = _toy_data()
    model = GraphCoreSGHDBSCAN(min_samples=10).fit(X)

    expected = hdbscan_glosh(model.condensed_trees_[10].to_numpy())

    np.testing.assert_allclose(
        model.outlier_scores_for(10), expected, rtol=0.0, atol=0.0, equal_nan=True
    )


def test_glosh_updates_after_refit():
    """Scores reflect the current fit, not a cached earlier dataset. run()
    clears and rebuilds models_/condensed_trees_, so per-model caching is safe."""
    model = GraphCoreSGHDBSCAN(min_samples=10)

    model.fit(_toy_data(seed=0))
    _ = model.outlier_scores_for(10)              # populate the per-model cache

    model.fit(_toy_data(seed=1))
    expected = hdbscan_glosh(model.condensed_trees_[10].to_numpy())

    np.testing.assert_allclose(
        model.outlier_scores_for(10), expected, rtol=0.0, atol=0.0, equal_nan=True
    )


def test_dense_coresg_glosh_matches_full_hdbscan():
    """Dense CORE-SG end-to-end regression: dense CORE-SG (use_sparse_fit=False)
    vs full HDBSCAN on the same dist_matrix_ (already symmetric -- no manual
    symmetrisation)."""
    X = _toy_data()
    m, min_cluster_size = 10, 10

    model = GraphCoreSGHDBSCAN(
        min_samples=m, min_cluster_size=min_cluster_size, use_sparse_fit=False,
    ).fit(X)

    D = np.asarray(model.dist_matrix_, dtype=np.float64)
    reference = hdbscan.HDBSCAN(
        metric="precomputed", min_samples=m,
        min_cluster_size=min_cluster_size, algorithm="generic",
    ).fit(D)

    np.testing.assert_allclose(
        model.outlier_scores_for(m), reference.outlier_scores_,
        rtol=1e-6, atol=1e-6, equal_nan=True,
    )


def test_sparse_vs_dense_labels():
    """Documented sparse-path guarantee: identical CORE-SG flat clustering
    (before optional MST noise reassignment). Compared with no_noise=False so
    the post-hoc reassignment layer is not conflated with the guarantee."""
    X = _toy_data()
    m = 10
    common = dict(min_samples=m, min_cluster_size=10)

    sparse = GraphCoreSGHDBSCAN(**common, use_sparse_fit=True).fit(X)
    dense = GraphCoreSGHDBSCAN(**common, use_sparse_fit=False).fit(X)

    assert adjusted_rand_score(
        sparse.labels_for(m, no_noise=False),
        dense.labels_for(m, no_noise=False),
    ) == 1.0


@pytest.mark.xfail(
    strict=False,
    reason=("Sparse and dense CORE-SG guarantee equivalent flat clustering, but "
            "identical condensed hierarchies/GLOSH are not guaranteed."),
)
def test_sparse_vs_dense_glosh_characterization():
    """Non-blocking: XFAIL if sparse/dense GLOSH diverge, XPASS if they agree."""
    X = _toy_data()
    m = 10
    common = dict(min_samples=m, min_cluster_size=10)

    sparse = GraphCoreSGHDBSCAN(**common, use_sparse_fit=True).fit(X)
    dense = GraphCoreSGHDBSCAN(**common, use_sparse_fit=False).fit(X)

    np.testing.assert_allclose(
        sparse.outlier_scores_for(m), dense.outlier_scores_for(m),
        rtol=1e-6, atol=1e-6, equal_nan=True,
    )
