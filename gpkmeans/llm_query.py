"""OpenAI-compatible LLM querying for pairwise ML/CL constraints.

No API key is stored in this repository. Set ``DEEPSEEK_API_KEY`` or pass an
API key explicitly when running the script in ``scripts/generate_constraints.py``.
"""

import json
import os
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import requests

Pair = Tuple[int, int]


class LLMConfig:
    def __init__(self, model="deepseek-v4-flash", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", api_key=None, temperature=0.0, timeout=60):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout


def build_pair_prompt(text_a: str, text_b: str, task_description: str = "text clustering") -> str:
    template = Path(__file__).resolve().parents[1] / "prompts" / "pairwise_constraint_prompt.txt"
    return template.read_text(encoding="utf-8").format(
        task_description=task_description,
        text_a=text_a.replace("\n", " ").strip(),
        text_b=text_b.replace("\n", " ").strip(),
    )


def parse_relation(response: str) -> str:
    """Parse a model response into ``ML`` or ``CL``."""

    text = response.strip().upper()
    match = re.search(r"\b(ML|CL)\b", text)
    if not match:
        raise ValueError(f"Could not parse ML/CL relation from response: {response!r}")
    return match.group(1)


def query_relation(prompt: str, config: LLMConfig) -> str:
    api_key = config.api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Missing API key. Set DEEPSEEK_API_KEY or pass LLMConfig(api_key=...).")
    response = requests.post(
        f"{config.base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": config.temperature,
        },
        timeout=config.timeout,
    )
    response.raise_for_status()
    raw = response.json()["choices"][0]["message"]["content"]
    return parse_relation(raw)


def generate_pairwise_constraints(
    texts: List[str],
    pairs: Iterable[Pair],
    config: LLMConfig,
    task_description: str = "text clustering",
):
    """Query pairwise relations and return ML/CL edge lists."""

    ml_pairs, cl_pairs = [], []
    for a, b in pairs:
        prompt = build_pair_prompt(texts[a], texts[b], task_description)
        relation = query_relation(prompt, config)
        if relation == "ML":
            ml_pairs.append((int(a), int(b)))
        else:
            cl_pairs.append((int(a), int(b)))
    return ml_pairs, cl_pairs


def save_pairs(path, pairs: Iterable[Pair]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([[int(a), int(b)] for a, b in pairs], indent=2) + "\n", encoding="utf-8")
