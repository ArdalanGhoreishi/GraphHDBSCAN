"""Clustering evaluation helpers."""

from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score


def evaluate_clustering(true_labels, predicted_labels):
    """Compute AMI and ARI between true and predicted labels.

    Returns
    -------
    tuple
        ``(ami, ari)``
    """
    ami = adjusted_mutual_info_score(true_labels, predicted_labels)
    ari = adjusted_rand_score(true_labels, predicted_labels)
    return ami, ari
