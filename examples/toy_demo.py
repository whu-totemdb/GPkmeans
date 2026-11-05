#!/usr/bin/env python3
"""Small deterministic demo for GPKmeans."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from gpkmeans import GPKmeans


def main():
    rng = np.random.default_rng(7)
    left = rng.normal(loc=(-2.0, 0.0), scale=0.25, size=(20, 2))
    right = rng.normal(loc=(2.0, 0.0), scale=0.25, size=(20, 2))
    X = np.vstack([left, right])

    ml_groups = [[0, 1, 2], [21, 22, 23]]
    cl_groups = [[0, 25], [4, 30]]
    model = GPKmeans(n_clusters=2, lambda_ml=0.05, lambda_cl=0.2, random_state=3)
    result = model.fit(X, ml_groups=ml_groups, cl_groups=cl_groups)

    print("labels:", result.labels.tolist())
    print("iterations:", result.n_iter)
    print("moves:", result.n_moves)
    print("objective_trace:", [round(v, 4) for v in result.objective_trace])


if __name__ == "__main__":
    main()
