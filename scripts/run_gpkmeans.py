#!/usr/bin/env python3
"""Run GPKmeans on NumPy embeddings and JSON constraints."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import numpy as np

from gpkmeans.constraints import load_constraint_json
from gpkmeans.optimizer import GPKmeans


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True, help="Path to .npy matrix with shape [n, d]")
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--ml", default=None, help="JSON list of ML groups")
    parser.add_argument("--cl", default=None, help="JSON list of CL groups")
    parser.add_argument("--lambda-ml", type=float, default=0.01)
    parser.add_argument("--lambda-cl", type=float, default=0.2)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", required=True, help="Output .npy file for predicted labels")
    args = parser.parse_args()

    X = np.load(args.embeddings)
    ml_groups = load_constraint_json(args.ml) if args.ml else []
    cl_groups = load_constraint_json(args.cl) if args.cl else []
    model = GPKmeans(
        n_clusters=args.k,
        lambda_ml=args.lambda_ml,
        lambda_cl=args.lambda_cl,
        max_iter=args.max_iter,
        random_state=args.seed,
    )
    result = model.fit(X, ml_groups=ml_groups, cl_groups=cl_groups)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, result.labels)
    print(f"iterations={result.n_iter} moves={result.n_moves} objective={result.objective_trace[-1]:.6f}")
    print(f"labels saved to {args.output}")


if __name__ == "__main__":
    main()
