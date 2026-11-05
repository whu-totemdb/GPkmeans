#!/usr/bin/env python3
"""Generate ML/CL pair constraints with an OpenAI-compatible LLM endpoint."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

from gpkmeans.constraints import build_ml_components, save_constraint_json
from gpkmeans.llm_query import LLMConfig, generate_pairwise_constraints, save_pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--texts", required=True, help="JSONL file with a 'text' or 'input' field")
    parser.add_argument("--pairs", required=True, help="JSON file containing candidate index pairs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-description", default="text clustering")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.texts).read_text(encoding="utf-8").splitlines() if line.strip()]
    texts = [row.get("text", row.get("input", "")) for row in rows]
    pairs = [tuple(map(int, pair)) for pair in json.loads(Path(args.pairs).read_text(encoding="utf-8"))]
    config = LLMConfig(model=args.model, base_url=args.base_url)
    ml_pairs, cl_pairs = generate_pairwise_constraints(texts, pairs, config, args.task_description)

    out = Path(args.output_dir)
    save_pairs(out / "ml_pairs.json", ml_pairs)
    save_pairs(out / "cl_pairs.json", cl_pairs)
    save_constraint_json(out / "ml_groups.json", build_ml_components(len(texts), ml_pairs))
    save_constraint_json(out / "cl_groups.json", cl_pairs)
    print(f"wrote constraints to {out}")


if __name__ == "__main__":
    main()
