# GPKmeans

GPKmeans is a noise-aware group-penalty k-means implementation for text clustering with LLM-generated must-link (ML) and cannot-link (CL) constraints.

The code is intentionally small and anonymized for double-blind review. It contains the core optimizer, constraint utilities, prompt template, and command-line entry points. It does not include paper source files, private experiment logs, API keys, generated datasets, or large embedding files.

## Method Overview

GPKmeans uses LLM-generated pairwise constraints as soft semantic preferences rather than hard feasibility rules.

1. Generate sparse local pairwise constraints with an LLM.
2. Collapse pairwise ML edges into connected components.
3. Keep CL constraints as local pair groups.
4. Optimize a k-means SSE objective with normalized ML/CL group penalties.
5. Use cached local delta updates during coordinate descent.

## Repository Layout

```text
gpkmeans/
  optimizer.py          # cached coordinate-descent GPKmeans optimizer
  constraints.py        # ML component construction and JSON utilities
  llm_query.py          # OpenAI-compatible LLM querying helper
prompts/
  pairwise_constraint_prompt.txt
scripts/
  run_gpkmeans.py       # run optimizer on .npy embeddings and JSON constraints
  generate_constraints.py
examples/
  toy_demo.py
tests/
  test_optimizer.py
MAINTAINERS.md          # notes for updating this clean release
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development tests:

```bash
pip install pytest
pytest -q
```

## Quick Demo

```bash
python examples/toy_demo.py
```

## Running GPKmeans on Embeddings

Inputs:

- `embeddings.npy`: NumPy array with shape `[n_samples, embedding_dim]`.
- `ml_groups.json`: JSON list of ML groups, e.g. `[[0, 1, 2], [8, 9]]`.
- `cl_groups.json`: JSON list of CL groups, e.g. `[[0, 8], [2, 9]]`.

Example:

```bash
python scripts/run_gpkmeans.py \
  --embeddings embeddings.npy \
  --k 20 \
  --ml ml_groups.json \
  --cl cl_groups.json \
  --lambda-ml 0.01 \
  --lambda-cl 0.2 \
  --max-iter 80 \
  --seed 1 \
  --output outputs/labels.npy
```

## LLM Constraint Generation

The released prompt template is in `prompts/pairwise_constraint_prompt.txt`.
The default LLM model used by the experiments is `deepseek-v4-flash` through an OpenAI-compatible API.

No API key is stored in this repository. To generate constraints, set an environment variable locally:

```bash
export DEEPSEEK_API_KEY=your_api_key_here
```

Then run:

```bash
python scripts/generate_constraints.py \
  --texts data/texts.jsonl \
  --pairs data/candidate_pairs.json \
  --output-dir constraints/example \
  --task-description "intent clustering" \
  --model deepseek-v4-flash
```

The script writes:

```text
ml_pairs.json
cl_pairs.json
ml_groups.json
cl_groups.json
```

## Data Format

`texts.jsonl` should contain one record per line with either a `text` or `input` field.

```json
{"text": "I need to change my address."}
```

`candidate_pairs.json` should be a JSON list of integer pairs:

```json
[[0, 1], [0, 5], [2, 7]]
```

## Notes

- The code release is designed to show and run the core method.
- Large embeddings, cached experiment outputs, and private LLM logs are intentionally excluded.
- The optimizer is independent of any specific embedding model; provide your own embeddings as `.npy` files.
- The LLM endpoint is configurable as long as it follows the OpenAI chat-completions format.
- The repository is intentionally anonymous; avoid adding author names, private paths, or paper-review materials before the double-blind review period ends.
