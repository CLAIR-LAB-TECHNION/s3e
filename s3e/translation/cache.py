"""NL translation cache for storing predicate-to-query mappings.

This module provides simple JSON file I/O for caching the results of
LLM-driven PDDL-to-natural-language translation. Caches are keyed by
model ID, problem name, and inference kwargs.
"""

import json
import os


def make_cache_key(model_id: str, problem_name: str, **kwargs) -> str:
    """Build a deterministic cache filename from model ID, problem name, and kwargs.

    The filename format is: ``model_id--(problem_name;k1=v1;k2=v2).json``
    where slashes in model_id are replaced with double underscores.

    Args:
        model_id: The model identifier (e.g. ``"meta-llama/Llama-3"``).
        problem_name: The PDDL problem name.
        **kwargs: Additional keyword arguments that were used for inference.

    Returns:
        A filename string safe for use as a file name.
    """
    safe_model_id = model_id.replace("/", "__")
    params = f"pddl_problem={problem_name}"
    if kwargs:
        params += ";" + ";".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
    return f"{safe_model_id}--({params}).json"


def load_cache(cache_dir: str, cache_key: str) -> dict[str, str]:
    """Load a cached queries dictionary from disk.

    Args:
        cache_dir: Directory containing cache files.
        cache_key: The cache filename (from :func:`make_cache_key`).

    Returns:
        Dictionary mapping predicate strings to query strings.

    Raises:
        FileNotFoundError: If the cache file does not exist.
    """
    path = os.path.join(cache_dir, cache_key)
    with open(path, "r") as f:
        return json.load(f)


def save_cache(cache_dir: str, cache_key: str, queries: dict[str, str]) -> None:
    """Save a queries dictionary to disk, merging with any existing cache.

    If the cache file already exists, the new queries are merged into it
    (new keys are added, existing keys are updated).

    Args:
        cache_dir: Directory to store cache files (created if needed).
        cache_key: The cache filename (from :func:`make_cache_key`).
        queries: Dictionary mapping predicate strings to query strings.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, cache_key)

    existing: dict[str, str] = {}
    if os.path.exists(path):
        with open(path, "r") as f:
            existing = json.load(f)

    existing.update(queries)

    with open(path, "w") as f:
        json.dump(existing, f, indent=4)
