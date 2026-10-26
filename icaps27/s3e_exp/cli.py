# SPDX-License-Identifier: MIT
"""Argument helpers shared by the experiment scripts."""

import argparse
import json
from pathlib import Path

from .scenes import load_scenes


def scene_args(parser: argparse.ArgumentParser, required: bool = True) -> None:
    parser.add_argument("--instances", type=Path, required=required,
                        help="directory of scenes written by experiments/instances.py")
    parser.add_argument("--sizes", type=int, nargs="*", default=None,
                        help="only scenes with these object counts")
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="only scenes generated with these seeds")
    parser.add_argument("--limit", type=int, default=None,
                        help="at most this many scenes per (size, seed)")


def select_scenes(args) -> list:
    scenes = load_scenes(args.instances)
    if args.sizes:
        scenes = [s for s in scenes if s.size in args.sizes]
    if args.seeds:
        scenes = [s for s in scenes if s.meta.get("seed") in args.seeds]
    if args.limit is not None:
        kept, counts = [], {}
        for scene in scenes:
            key = (scene.size, scene.meta.get("seed"))
            if counts.get(key, 0) < args.limit:
                kept.append(scene)
                counts[key] = counts.get(key, 0) + 1
        scenes = kept
    if not scenes:
        raise SystemExit(f"no scenes selected under {args.instances}")
    return scenes


def model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True,
                        help="a name from configs/models.json, a Hugging Face id, "
                             "OpenAI/<model>, or 'fake' (an oracle with noise, no download)")
    parser.add_argument("--backend", choices=["hf", "vllm", "openai", "fake"], default=None,
                        help="override the registry's backend for this model")
    parser.add_argument("--vlm-kwargs", type=json.loads, default={},
                        help="JSON constructor kwargs for the backend")


def domain_of(scenes) -> str:
    names = {s.domain for s in scenes}
    if len(names) != 1:
        raise SystemExit(f"scenes from several domains: {sorted(names)}")
    return names.pop()


def run_dir(root: Path, *parts: str) -> Path:
    path = Path(root).joinpath(*[p.replace("/", "--") for p in parts])
    path.mkdir(parents=True, exist_ok=True)
    return path
