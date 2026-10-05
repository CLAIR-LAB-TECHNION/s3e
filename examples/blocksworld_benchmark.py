# SPDX-FileCopyrightText: CLAIR Lab Technion
# SPDX-License-Identifier: MIT

"""Measure a VLM on synthetic Blocksworld scenes with known ground truth.

The script renders random Blocksworld states as images (colored blocks
stacked on a table), asks the model about every grounded predicate, and
reports for each scoring mode (``logprobs`` reads the answer-token
probabilities; ``text_match`` generates a reply and matches its start):

* accuracy of the thresholded state, Brier score, and expected calibration
  error (ECE, 10 equal-width bins) of P(true);
* the share of predictions with no answer (``Prediction.has_answer`` False);
* wall-clock time per query and peak memory (on a GPU, the peak allocated
  on the first device by this process during that mode, so not vLLM's
  engine process or other shards; on CPU, the process's peak resident
  memory so far, which includes the model);
* optionally, metrics on held-out scenes before and after Platt scaling
  fitted offline on the other scenes, without querying the model again.

Every per-predicate result (with the raw answer masses) is written to a
JSON file together with the provenance needed to report and reproduce the
run: library versions, the model id and resolved Hugging Face revision,
the full prompts, answer tokens, decoding settings, the fitted calibrator,
dates, and hardware.

This is a template for your own evaluation, not a published benchmark: the
scenes are deliberately simple, and small models still do poorly on them.

Usage::

    pip install "s3e[pddl,hf,calibration]"
    python examples/blocksworld_benchmark.py \\
        --model HuggingFaceTB/SmolVLM-256M-Instruct --num-scenes 10 \\
        --output blocksworld-results.json

A GPU is strongly recommended: on CPU, even a 256M-parameter VLM can take
tens of seconds per query.
"""

import argparse
import json
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from PIL import Image, ImageDraw

import s3e
from s3e import (
    CalibrationSample,
    CalibrationSet,
    PlattCalibrator,
    SemanticStateEstimator,
    TemplateTranslator,
    resolve_backend,
)

COLORS = ("red", "green", "blue", "yellow", "purple", "orange")

TEMPLATES = {
    "on": "Is the {0} block directly on top of the {1} block?",
    "ontable": "Is the {0} block standing directly on the table?",
    "clear": "Is the top of the {0} block clear, with no block on it?",
}
PROMPT_TEMPLATE = "{query} Answer with exactly one word: yes or no."

DOMAIN = """
(define (domain blocksworld)
  (:requirements :typing)
  (:types block)
  (:predicates (on ?x - block ?y - block) (ontable ?x - block) (clear ?x - block)))
"""

# A cap on the generated reply for text_match scoring; the kwarg's name
# depends on the backend. Other backends (e.g. OpenAI) get no cap.
DEFAULT_GENERATION_KWARGS = {
    "HuggingFaceVLM": {"max_new_tokens": 8},
    "VLLMBackend": {"max_tokens": 8},
}

REPORTED_PACKAGES = (
    "s3e", "torch", "transformers", "accelerate", "vllm", "openai",
    "unified-planning", "scikit-learn", "numpy", "Pillow",
)


def make_problem(blocks: list[str]) -> str:
    """A PDDL problem over ``blocks``; only its objects matter for grounding."""
    return (
        "(define (problem synthetic) (:domain blocksworld)"
        f" (:objects {' '.join(blocks)} - block)"
        f" (:init {' '.join(f'(ontable {b})' for b in blocks)})"
        f" (:goal (clear {blocks[0]})))"
    )


def random_towers(blocks: list[str], rng: random.Random) -> list[list[str]]:
    """Stack the blocks into random towers, listed bottom to top."""
    order = list(blocks)
    rng.shuffle(order)
    towers, tower = [], []
    for block in order:
        tower.append(block)
        if rng.random() < 0.5:
            towers.append(tower)
            tower = []
    if tower:
        towers.append(tower)
    return towers


def true_state(towers: list[list[str]], blocks: list[str]) -> dict[str, bool]:
    """Every grounded predicate's truth value in the given arrangement."""
    state = {f"on({x},{y})": False for x in blocks for y in blocks}
    state.update({f"{name}({x})": False for name in ("ontable", "clear") for x in blocks})
    for tower in towers:
        state[f"ontable({tower[0]})"] = True
        state[f"clear({tower[-1]})"] = True
        for below, above in zip(tower, tower[1:]):
            state[f"on({above},{below})"] = True
    return state


def render(towers: list[list[str]], num_blocks: int) -> Image.Image:
    """Draw the towers left to right on a gray table, 60x45 px per block."""
    width, height = 90 * num_blocks + 40, 45 * num_blocks + 90
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    table_top = height - 40
    draw.rectangle((0, table_top, width, height), fill="gray")
    for column, tower in enumerate(towers):
        left = 30 + 90 * column
        for level, block in enumerate(tower):
            bottom = table_top - 45 * level
            draw.rectangle(
                (left, bottom - 45, left + 60, bottom), fill=block, outline="black", width=3
            )
    return image


def expected_calibration_error(probabilities, labels, bins: int = 10) -> float:
    """Mean |accuracy - confidence| over equal-width bins of P(true)."""
    total, error = len(labels), 0.0
    for b in range(bins):
        low, high = b / bins, (b + 1) / bins
        members = [
            (p, y) for p, y in zip(probabilities, labels)
            if low <= p < high or (b == bins - 1 and p == 1.0)
        ]
        if members:
            mean_p = sum(p for p, _ in members) / len(members)
            frequency = sum(y for _, y in members) / len(members)
            error += len(members) / total * abs(frequency - mean_p)
    return error


def metrics(rows: list[dict]) -> dict:
    """Accuracy, Brier score, ECE, and no-answer rate over per-predicate rows."""
    probabilities = [row["probability"] for row in rows]
    labels = [row["truth"] for row in rows]
    return {
        "predictions": len(rows),
        "accuracy": sum(row["state"] == row["truth"] for row in rows) / len(rows),
        "brier": sum((p - y) ** 2 for p, y in zip(probabilities, labels)) / len(rows),
        "ece": expected_calibration_error(probabilities, labels),
        "no_answer_rate": sum(not row["has_answer"] for row in rows) / len(rows),
    }


def peak_memory_bytes() -> "tuple[str, int | None]":
    """Peak CUDA memory since the last reset if a GPU is in use, else the
    process's peak resident set size since it started."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda_max_memory_allocated", torch.cuda.max_memory_allocated()
    except ImportError:
        pass
    try:
        import resource
    except ImportError:  # Windows
        return "unavailable", None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is in bytes on macOS and kilobytes elsewhere.
    return "process_peak_rss", peak if sys.platform == "darwin" else peak * 1024


def reset_peak_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def prediction_rows(scene: dict, results) -> list[dict]:
    """One JSON-ready row per predicate, with the stored masses."""
    state = results.to_state()
    return [
        {
            "scene": scene["id"],
            "predicate": predicate,
            "truth": scene["state"][predicate],
            "probability": prediction.probability,
            "state": state[predicate],
            "has_answer": prediction.has_answer,
            "masses": prediction.masses,
            "null_mass": prediction.null_mass,
            "unassigned_mass": prediction.unassigned_mass,
            "text": prediction.text,
        }
        for predicate, prediction in results.items()
    ]


def evaluate(backend, scenes, problem, scoring, inference_kwargs) -> dict:
    """Estimate every scene under one scoring mode; return rows and timing."""
    estimator = SemanticStateEstimator.from_pddl(
        DOMAIN,
        problem,
        vlm=backend,
        translator=TemplateTranslator(TEMPLATES),
        prompt_template=PROMPT_TEMPLATE,
        scoring=scoring,
        inference_kwargs=inference_kwargs,
    )
    # One untimed query warms up kernels and caches.
    estimator.estimate(scenes[0]["images"], predicates=estimator.predicates[:1])
    reset_peak_memory()

    rows, results_per_scene, elapsed = [], [], 0.0
    for scene in scenes:
        start = time.perf_counter()
        results = estimator.estimate(scene["images"])
        elapsed += time.perf_counter() - start
        results_per_scene.append(results)
        rows.extend(prediction_rows(scene, results))
    memory_kind, memory = peak_memory_bytes()
    return {
        "estimator": estimator,
        "rows": rows,
        "results": results_per_scene,
        "seconds_per_query": elapsed / len(rows),
        "memory": {"kind": memory_kind, "bytes": memory},
    }


def calibration_split(num_scenes: int, fraction: float) -> int:
    """Number of scenes to fit on; at least one scene on each side."""
    split = int(num_scenes * fraction)
    if not 1 <= split < num_scenes:
        raise ValueError(
            f"calibrate_fraction={fraction} of {num_scenes} scenes leaves no "
            "scene to fit on or none to evaluate on"
        )
    return split


def calibrate(run: dict, scenes: list[dict], split: int) -> dict:
    """Fit Platt scaling on the first ``split`` scenes; evaluate the rest
    before and after calibration.

    Uses the predictions already computed: calibration never queries the model.
    """
    samples = [
        CalibrationSample(predicate, prediction.score, scene["state"][predicate])
        for scene, results in zip(scenes[:split], run["results"][:split])
        for predicate, prediction in results.items()
        if prediction.has_answer
    ]
    if not samples:
        return {"skipped": "no prediction in the calibration scenes has an answer"}
    calibrator = PlattCalibrator.fit(
        CalibrationSet(samples=samples, meta=run["estimator"].calibration_meta()),
        scope="lifted",
        pass_through_single_class=True,
    )
    held_out = list(zip(scenes[split:], run["results"][split:]))
    return {
        "uncalibrated": [r for s, res in held_out for r in prediction_rows(s, res)],
        "calibrated": [
            r for s, res in held_out for r in prediction_rows(s, calibrator.apply(res))
        ],
        "fit": {
            "scenes": [scene["id"] for scene in scenes[:split]],
            "samples": len(samples),
            "scope": calibrator.scope,
            "groups": {
                key: vars(params) for key, params in calibrator.groups.items()
            },
        },
    }


def package_versions() -> dict:
    versions = {}
    for name in REPORTED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return versions


def git_commit() -> "dict | None":
    """The s3e source commit and whether the checkout has local changes,
    when s3e runs from its own git checkout (not, e.g., a virtualenv inside
    another repository)."""
    package_dir = Path(s3e.__file__).resolve().parent

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=package_dir, capture_output=True, text=True, check=True,
        ).stdout.strip()

    try:
        toplevel = Path(git("rev-parse", "--show-toplevel"))
        if toplevel.resolve() != package_dir.parent:
            return None
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return None


def model_revision(backend, vlm_kwargs: dict) -> "str | None":
    """The Hugging Face commit the backend's model was loaded from, if known.

    Older transformers record it as ``config._commit_hash``; otherwise it is
    the name of the cached snapshot folder that a (pinned or default)
    revision resolves to. ``None`` for API models and local paths.
    """
    config = getattr(getattr(backend, "model", None), "config", None)
    revision = getattr(config, "_commit_hash", None)
    if revision:
        return revision
    model_id = getattr(backend, "model_id", None)
    if not model_id:
        return None
    try:
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(
            model_id,
            revision=vlm_kwargs.get("revision"),
            cache_dir=vlm_kwargs.get("cache_dir"),
            local_files_only=True,
        )
    except Exception:  # not a cached Hub model: provenance stays unknown
        return None
    return Path(snapshot).name


def device_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return platform.processor() or platform.machine()


def run_benchmark(
    vlm,
    *,
    num_scenes: int = 10,
    num_blocks: int = 3,
    seed: int = 0,
    scoring_modes=("logprobs", "text_match"),
    generation_kwargs: "dict | None" = None,
    calibrate_fraction: "float | None" = None,
    vlm_kwargs: "dict | None" = None,
) -> dict:
    """Run the benchmark and return a JSON-serializable report.

    Args:
        vlm: A model id (resolved with :func:`s3e.resolve_backend`) or a
            :class:`s3e.VLMBackend` instance, shared by every scoring mode.
        num_scenes: Number of random scenes to render and estimate.
        num_blocks: Blocks per scene (2 to 6).
        seed: Seed for the random arrangements.
        scoring_modes: Any of ``"logprobs"`` and ``"text_match"``.
        generation_kwargs: ``inference_kwargs`` for ``text_match``; their
            names are backend specific. Defaults to a short reply cap for
            HuggingFace (``max_new_tokens``) and vLLM (``max_tokens``), and
            to nothing for other backends.
        calibrate_fraction: If set, fit Platt scaling (``logprobs`` only) on
            this share of the scenes and report metrics on the remaining
            scenes before (``logprobs held-out``) and after
            (``logprobs held-out+platt``) calibration.
        vlm_kwargs: Backend constructor kwargs when ``vlm`` is a model id.
    """
    if not 2 <= num_blocks <= len(COLORS):
        raise ValueError(f"num_blocks must be between 2 and {len(COLORS)}")
    if num_scenes < 1:
        raise ValueError("num_scenes must be at least 1")
    if calibrate_fraction:
        if "logprobs" not in scoring_modes:
            raise ValueError("calibrate_fraction needs the logprobs scoring mode")
        split = calibration_split(num_scenes, calibrate_fraction)
    started = datetime.now(timezone.utc)
    rng = random.Random(seed)
    blocks = list(COLORS[:num_blocks])
    problem = make_problem(blocks)
    scenes = []
    for i in range(num_scenes):
        towers = random_towers(blocks, rng)
        scenes.append({
            "id": i,
            "towers": towers,
            "state": true_state(towers, blocks),
            "images": [render(towers, num_blocks)],
        })

    backend = resolve_backend(vlm, **(vlm_kwargs or {}))
    report = {"summary": {}, "instances": {}}
    estimator = None
    inference = {}
    if generation_kwargs is None:
        generation_kwargs = DEFAULT_GENERATION_KWARGS.get(type(backend).__name__, {})
    for scoring in scoring_modes:
        kwargs = dict(generation_kwargs) if scoring == "text_match" else {}
        inference[scoring] = kwargs
        run = evaluate(backend, scenes, problem, scoring, kwargs)
        estimator = run["estimator"]
        report["summary"][scoring] = {
            **metrics(run["rows"]),
            "seconds_per_query": run["seconds_per_query"],
            "peak_memory": run["memory"],
        }
        report["instances"][scoring] = run["rows"]
        if calibrate_fraction and scoring == "logprobs":
            calibration = calibrate(run, scenes, split)
            if "skipped" in calibration:
                report["calibration"] = calibration
                continue
            for name, key in (("held-out", "uncalibrated"), ("held-out+platt", "calibrated")):
                report["summary"][f"logprobs {name}"] = metrics(calibration[key])
                report["instances"][f"logprobs {name}"] = calibration[key]
            report["calibration"] = calibration["fit"]

    report["provenance"] = {
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "s3e_git_commit": git_commit(),
        "packages": package_versions(),
        "python": sys.version,
        "platform": platform.platform(),
        "device": device_name(),
        "backend": type(backend).__name__,
        "model_id": vlm if isinstance(vlm, str) else getattr(backend, "model_id", None),
        "model_revision": model_revision(backend, vlm_kwargs or {}),
        "vlm_kwargs": vlm_kwargs or {},
        "seed": seed,
        "num_scenes": num_scenes,
        "blocks": blocks,
        "towers": [scene["towers"] for scene in scenes],
        "domain": DOMAIN,
        "problem": problem,
        "system_prompt": estimator.engine.system_prompt if estimator else None,
        "prompt_template": PROMPT_TEMPLATE,
        "queries": estimator.queries if estimator else None,
        "answer_space": estimator.engine.answers.to_dict() if estimator else None,
        "inference_kwargs": inference,
    }
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct",
                        help="model id, e.g. a Hugging Face id or OpenAI/<model>")
    parser.add_argument("--num-scenes", type=int, default=10)
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scoring", choices=["logprobs", "text_match", "both"], default="both")
    parser.add_argument("--generation-kwargs", type=json.loads, default=None,
                        help="JSON inference kwargs for text_match (default: a short reply "
                             "cap for HuggingFace and vLLM, none otherwise)")
    parser.add_argument("--vlm-kwargs", type=json.loads, default={},
                        help='JSON backend kwargs, e.g. \'{"revision": "<commit>"}\'')
    parser.add_argument("--calibrate-fraction", type=float, default=None,
                        help="fit Platt scaling on this share of scenes (logprobs only)")
    parser.add_argument("--output", type=Path, default=Path("blocksworld-results.json"))
    args = parser.parse_args(argv)

    modes = ("logprobs", "text_match") if args.scoring == "both" else (args.scoring,)
    report = run_benchmark(
        args.model,
        num_scenes=args.num_scenes,
        num_blocks=args.num_blocks,
        seed=args.seed,
        scoring_modes=modes,
        generation_kwargs=args.generation_kwargs,
        calibrate_fraction=args.calibrate_fraction,
        vlm_kwargs=args.vlm_kwargs,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"{'mode':24}{'accuracy':>10}{'Brier':>8}{'ECE':>8}{'no answer':>11}{'s/query':>9}")
    for mode, summary in report["summary"].items():
        seconds = summary.get("seconds_per_query")
        print(
            f"{mode:24}{summary['accuracy']:>10.3f}{summary['brier']:>8.3f}"
            f"{summary['ece']:>8.3f}{summary['no_answer_rate']:>11.3f}"
            + (f"{seconds:>9.3f}" if seconds is not None else f"{'-':>9}")
        )
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
