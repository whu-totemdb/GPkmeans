"""Utilities for constructing group-level ML/CL constraints.

The optimizer accepts constraints as groups of integer sample indices:

* ML groups: samples in a group are softly encouraged to share one cluster.
* CL groups: samples in a group are softly encouraged not to collapse into one cluster.

For the method described in the paper, generated pairwise ML edges are collapsed
into connected components, while CL constraints are usually kept as local
anchor-candidate pairs.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

ConstraintGroup = List[int]
Pair = Tuple[int, int]


class DSU:
    """Disjoint-set union for connected-component construction."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def build_ml_components(n_samples: int, ml_pairs: Iterable[Pair]) -> List[ConstraintGroup]:
    """Collapse pairwise must-link edges into connected components."""

    dsu = DSU(n_samples)
    for a, b in ml_pairs:
        if 0 <= int(a) < n_samples and 0 <= int(b) < n_samples:
            dsu.union(int(a), int(b))

    components = defaultdict(list)
    for idx in range(n_samples):
        components[dsu.find(idx)].append(idx)
    return [group for group in components.values() if len(group) >= 2]


def normalize_groups(groups: Iterable[Sequence[int]], n_samples: int) -> List[ConstraintGroup]:
    """Remove duplicates, out-of-range indices, and singleton groups."""

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


def load_constraint_json(path) -> List[ConstraintGroup]:
    """Load a JSON file containing either a list of groups or a dict of groups."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [list(map(int, group)) for group in data]
    if isinstance(data, dict):
        groups = []
        for value in data.values():
            if isinstance(value, dict):
                groups.extend(list(map(int, group)) for group in value.values())
            else:
                groups.append(list(map(int, value)))
        return groups
    raise ValueError(f"Unsupported constraint JSON format in {path}")


def save_constraint_json(path, groups: Iterable[Sequence[int]]) -> None:
    """Write constraints as a simple JSON list of index groups."""

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    clean = [list(map(int, group)) for group in groups]
    Path(path).write_text(json.dumps(clean, indent=2) + "\n", encoding="utf-8")
