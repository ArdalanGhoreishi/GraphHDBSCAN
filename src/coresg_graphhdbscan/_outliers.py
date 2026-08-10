from __future__ import annotations

import numpy as np
from hdbscan._hdbscan_tree import outlier_scores as _hdbscan_outlier_scores


def glosh_from_condensed_tree(condensed_tree) -> np.ndarray:
    """Compute GLOSH outlier scores from an HDBSCAN-format condensed tree.

    Uses hdbscan's official GLOSH implementation
    (``hdbscan._hdbscan_tree.outlier_scores``), evaluated on the supplied
    hierarchy. The result is therefore the reference HDBSCAN GLOSH computed on
    *this* condensed tree -- exact conditional on the tree.

    Parameters
    ----------
    condensed_tree : hdbscan.plots.CondensedTree or numpy structured ndarray
        Either an hdbscan ``CondensedTree`` wrapper, or the raw structured
        array with fields ('parent', 'child', 'lambda_val', 'child_size') in
        the canonical order produced by hdbscan's ``condense_tree``. The row
        ordering matters: ``outlier_scores`` walks the tree in reverse and does
        not re-sort.

    Returns
    -------
    scores : ndarray, shape (n_samples,)
        GLOSH scores indexed by condensed-tree point ids (0 .. n_samples-1).
        Larger means more outlying. Finite scores lie in [0, 1]. Hierarchies
        containing infinite lambda values, e.g. from zero-distance/duplicate-
        point configurations, may produce NaN scores.
    """
    if hasattr(condensed_tree, "to_numpy"):
        raw = condensed_tree.to_numpy()          # public hdbscan API
    else:
        raw = condensed_tree                     # already a raw structured array
    return _hdbscan_outlier_scores(raw)
