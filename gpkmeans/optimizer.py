"""Cached coordinate-descent optimizer for group-penalty k-means."""

from collections import namedtuple
from typing import Iterable, List, Sequence, Tuple

import numpy as np

WeightedGroup = Tuple[List[int], float]


GPKmeansResult = namedtuple("GPKmeansResult", ["labels", "objective_trace", "n_iter", "n_moves"])


def _kmeans_plus_plus(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]
    centers = np.empty((k, X.shape[1]), dtype=float)
    first = int(rng.integers(n))
    centers[0] = X[first]
    dist_sq = np.sum((X - centers[0]) ** 2, axis=1)
    for c in range(1, k):
        total = float(dist_sq.sum())
        if total <= 0:
            centers[c] = X[int(rng.integers(n))]
        else:
            idx = int(rng.choice(n, p=dist_sq / total))
            centers[c] = X[idx]
        dist_sq = np.minimum(dist_sq, np.sum((X - centers[c]) ** 2, axis=1))
    return centers


def _normalize_groups(groups: Iterable[Sequence[int]], n_samples: int) -> List[List[int]]:
    clean = []
    for group in groups:
        seen = []
        for item in group:
            idx = int(item)
            if 0 <= idx < n_samples and idx not in seen:
                seen.append(idx)
        if len(seen) >= 2:
            clean.append(seen)
    return clean


def _groups_to_adjacency(groups: List[WeightedGroup], n_samples: int) -> List[List[int]]:
    adjacency = [[] for _ in range(n_samples)]
    for group_idx, (members, weight) in enumerate(groups):
        if weight == 0:
            continue
        for idx in members:
            adjacency[idx].append(group_idx)
    return adjacency


def _init_group_counts(groups: List[WeightedGroup], labels: np.ndarray, k: int):
    counts = np.zeros((len(groups), k), dtype=float)
    sizes = np.zeros(len(groups), dtype=float)
    weights = np.zeros(len(groups), dtype=float)
    for gidx, (members, weight) in enumerate(groups):
        sizes[gidx] = len(members)
        weights[gidx] = float(weight)
        for idx in members:
            counts[gidx, labels[idx]] += 1.0
    return counts, sizes, weights


def _ml_cost(counts: np.ndarray, size: float) -> float:
    if size <= 1:
        return 0.0
    return float(size - np.max(counts)) / float(size - 1.0)


def _cl_cost(counts: np.ndarray, size: float) -> float:
    if size <= 1:
        return 0.0
    denom = size * (size - 1.0) / 2.0
    return float(np.sum(counts * (counts - 1.0) / 2.0)) / denom


class GPKmeans:
    """Noise-aware group-penalty k-means.

    Parameters
    ----------
    n_clusters:
        Target number of clusters.
    lambda_ml, lambda_cl:
        Penalty weights for ML and CL groups.
    max_iter:
        Maximum coordinate-descent sweeps.
    tol:
        Objective tolerance for early stopping.
    random_state:
        Seed used by k-means++ initialization.
    coordinate_order:
        ``"constrained_first"`` visits constrained points first; ``"natural"``
        uses index order.
    """

    def __init__(
        self,
        n_clusters: int,
        lambda_ml: float = 0.01,
        lambda_cl: float = 0.2,
        max_iter: int = 80,
        tol: float = 1e-8,
        random_state=None,
        coordinate_order: str = "constrained_first",
    ):
        self.n_clusters = int(n_clusters)
        self.lambda_ml = float(lambda_ml)
        self.lambda_cl = float(lambda_cl)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state
        self.coordinate_order = coordinate_order

    def fit(
        self,
        X: np.ndarray,
        ml_groups=None,
        cl_groups=None,
    ) -> GPKmeansResult:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        n, d = X.shape
        k = self.n_clusters
        if not 1 <= k <= n:
            raise ValueError("n_clusters must be between 1 and n_samples")

        ml_groups_clean = _normalize_groups(ml_groups or [], n)
        cl_groups_clean = _normalize_groups(cl_groups or [], n)
        weighted_ml = [(group, self.lambda_ml) for group in ml_groups_clean]
        weighted_cl = [(group, self.lambda_cl) for group in cl_groups_clean]
        ml_adj = _groups_to_adjacency(weighted_ml, n)
        cl_adj = _groups_to_adjacency(weighted_cl, n)
        order = self._coordinate_order(n, weighted_ml, weighted_cl)

        rng = np.random.default_rng(self.random_state)
        centers = _kmeans_plus_plus(X, k, rng)
        labels = np.argmin(((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), axis=1)

        cluster_weight = np.bincount(labels, minlength=k).astype(float)
        cluster_sum = np.zeros((k, d), dtype=float)
        for idx, label in enumerate(labels):
            cluster_sum[label] += X[idx]
        cluster_norm = np.einsum("ij,ij->i", cluster_sum, cluster_sum)

        ml_counts, ml_sizes, ml_weights = _init_group_counts(weighted_ml, labels, k)
        cl_counts, cl_sizes, cl_weights = _init_group_counts(weighted_cl, labels, k)

        objective = [self._objective(X, labels, weighted_ml, weighted_cl)]
        n_moves = 0

        for iteration in range(self.max_iter):
            sweep_moves = 0
            for idx in order:
                source = int(labels[idx])
                if cluster_weight[source] <= 1:
                    continue

                x = X[idx]
                x_norm = float(x @ x)
                source_old = cluster_norm[source] / cluster_weight[source]
                source_new_vec = cluster_sum[source] - x
                source_new = float(source_new_vec @ source_new_vec) / (cluster_weight[source] - 1.0)

                dest_new_norm = cluster_norm + 2.0 * (cluster_sum @ x) + x_norm
                dest_old = np.zeros(k)
                non_empty = cluster_weight > 0
                dest_old[non_empty] = cluster_norm[non_empty] / cluster_weight[non_empty]
                dest_new = dest_new_norm / (cluster_weight + 1.0)

                sse_delta = source_old + dest_old - source_new - dest_new
                sse_delta[source] = 0.0
                penalty_delta = np.zeros(k)
                self._add_ml_delta(penalty_delta, idx, source, ml_adj, ml_counts, ml_sizes, ml_weights)
                self._add_cl_delta(penalty_delta, idx, source, cl_adj, cl_counts, cl_sizes, cl_weights)
                total_delta = sse_delta + penalty_delta
                total_delta[source] = 0.0
                target = int(np.argmin(total_delta))

                if target != source and total_delta[target] < -1e-12:
                    cluster_sum[source] -= x
                    cluster_sum[target] += x
                    cluster_weight[source] -= 1.0
                    cluster_weight[target] += 1.0
                    cluster_norm[source] = float(cluster_sum[source] @ cluster_sum[source])
                    cluster_norm[target] = float(cluster_sum[target] @ cluster_sum[target])
                    labels[idx] = target
                    self._update_counts(ml_adj, ml_counts, idx, source, target)
                    self._update_counts(cl_adj, cl_counts, idx, source, target)
                    sweep_moves += 1
                    n_moves += 1

            objective.append(self._objective(X, labels, weighted_ml, weighted_cl))
            if sweep_moves == 0 or abs(objective[-1] - objective[-2]) < self.tol:
                return GPKmeansResult(labels.copy(), objective, iteration + 1, n_moves)

        return GPKmeansResult(labels.copy(), objective, self.max_iter, n_moves)

    def _coordinate_order(self, n: int, ml_groups: List[WeightedGroup], cl_groups: List[WeightedGroup]):
        if self.coordinate_order in {"natural", "index"}:
            return np.arange(n, dtype=int)
        if self.coordinate_order not in {"constrained_first", "constraint_first"}:
            raise ValueError("coordinate_order must be 'constrained_first' or 'natural'")
        degree = np.zeros(n)
        for groups in (ml_groups, cl_groups):
            for members, _ in groups:
                for idx in members:
                    degree[idx] += len(members)
        return np.asarray(sorted(range(n), key=lambda i: (degree[i] == 0, -degree[i], i)), dtype=int)

    def _objective(self, X, labels, ml_groups, cl_groups):
        k = self.n_clusters
        cluster_sum = np.zeros((k, X.shape[1]), dtype=float)
        cluster_weight = np.bincount(labels, minlength=k).astype(float)
        for idx, label in enumerate(labels):
            cluster_sum[label] += X[idx]
        reward = 0.0
        for c in range(k):
            if cluster_weight[c] > 0:
                reward += float(cluster_sum[c] @ cluster_sum[c]) / cluster_weight[c]
        sse = float(np.sum(X * X)) - reward
        penalty = 0.0
        for members, weight in ml_groups:
            counts = np.bincount(labels[members], minlength=k)
            penalty += weight * _ml_cost(counts, len(members))
        for members, weight in cl_groups:
            counts = np.bincount(labels[members], minlength=k)
            penalty += weight * _cl_cost(counts, len(members))
        return sse + penalty

    @staticmethod
    def _add_ml_delta(delta, idx, source, adjacency, counts, sizes, weights):
        for gidx in adjacency[idx]:
            size = sizes[gidx]
            if size <= 1:
                continue
            group_counts = counts[gidx]
            old_cost = (size - float(np.max(group_counts))) / (size - 1.0)
            base = group_counts.copy()
            base[source] -= 1.0
            base_max = float(np.max(base))
            new_max = np.maximum(base_max, group_counts + 1.0)
            contribution = weights[gidx] * ((size - new_max) / (size - 1.0) - old_cost)
            contribution[source] = 0.0
            delta += contribution

    @staticmethod
    def _add_cl_delta(delta, idx, source, adjacency, counts, sizes, weights):
        for gidx in adjacency[idx]:
            size = sizes[gidx]
            if size <= 1:
                continue
            denom = size * (size - 1.0) / 2.0
            group_counts = counts[gidx]
            contribution = weights[gidx] * (group_counts - (group_counts[source] - 1.0)) / denom
            contribution[source] = 0.0
            delta += contribution

    @staticmethod
    def _update_counts(adjacency, counts, idx, source, target):
        for gidx in adjacency[idx]:
            counts[gidx, source] -= 1.0
            counts[gidx, target] += 1.0
