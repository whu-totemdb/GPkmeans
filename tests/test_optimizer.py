import numpy as np

from gpkmeans import GPKmeans, build_ml_components


def test_optimizer_runs_on_toy_data():
    X = np.array([
        [-2.0, 0.0], [-2.1, 0.1], [-1.9, -0.1],
        [2.0, 0.0], [2.1, 0.1], [1.9, -0.1],
    ])
    ml = build_ml_components(len(X), [(0, 1), (3, 4)])
    cl = [[0, 3], [1, 4]]
    result = GPKmeans(n_clusters=2, lambda_ml=0.1, lambda_cl=0.2, random_state=0).fit(X, ml, cl)
    assert result.labels.shape == (6,)
    assert result.n_iter >= 1
    assert len(set(result.labels.tolist())) == 2
