import numpy as np

from coresg_graphhdbscan.core import CoreSGHDBSCAN


def test_core_fit_runs_on_tiny_dataset():
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [5.0, 5.0],
        [5.1, 5.0],
        [10.0, 10.0],
    ])
    model = CoreSGHDBSCAN(min_samples_list=[2])
    model.fit(X).run()
    assert 2 in model.models_
    assert len(model.models_[2].labels_) == len(X)
