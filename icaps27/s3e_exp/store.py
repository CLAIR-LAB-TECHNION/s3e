# SPDX-License-Identifier: MIT
"""Append-only JSONL result files, so long runs resume after preemption."""

import json
import math
from pathlib import Path


def _default(value):
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if hasattr(value, "item"):  # numpy scalars
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _clean(value):
    """NaN and inf are not JSON; write them as null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def dumps(obj, **kwargs) -> str:
    return json.dumps(_clean(obj), default=_default, **kwargs)


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(dumps(obj, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class JsonlStore:
    """One JSON record per line; a truncated last line (a crash) is ignored."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        records = []
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def keys(self, *fields: str) -> set[tuple]:
        return {tuple(r.get(f) for f in fields) for r in self.read()}

    def append(self, record: dict) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(dumps(record) + "\n")
            handle.flush()
