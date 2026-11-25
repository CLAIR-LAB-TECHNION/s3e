# SPDX-License-Identifier: MIT
"""E2: how S3E scales with the number of objects.

``count`` (no model): grounded atoms per scene with each mask (``none``,
``dynamic``, ``relevant``), and the time S3E spends grounding and
translating, for every scene of an instance set.

``time`` (model): wall clock per scene and per query, the share spent
inside the model, and peak memory, against object count, for each batch
size and mask; run once per backend (``--backend hf`` / ``--backend
vllm``) on the same model to compare single-GPU HuggingFace with vLLM
(tensor parallel over all visible GPUs). Each (batch size, mask) starts
with one untimed warm-up estimate.

``summary``: per (backend, batch size, mask, size): mean and standard
deviation of seconds per scene, queries per second, and atoms.

Examples::

    python experiments/e2_scale.py count --instances instances/blocksworld \\
        --out results/raw/e2/count-blocksworld.jsonl
    python experiments/e2_scale.py time --instances instances/blocksworld --limit 10 \\
        --model qwen3-vl-8b --backend hf --batch-sizes 1 8 32 64 \\
        --masks none dynamic --out results/raw/e2/time.jsonl
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s3e_exp import cli, domains, metrics, provenance  # noqa: E402
from s3e_exp.estimation import (  # noqa: E402
    EstimationConfig,
    atoms_for,
    build_estimator,
    estimate,
    run_provenance,
)
from s3e_exp.masking import MASKS  # noqa: E402
from s3e_exp.models import OracleVLM, TimedBackend, build_backend, model_spec  # noqa: E402
from s3e_exp.planning import parse  # noqa: E402
from s3e_exp.store import JsonlStore, write_json  # noqa: E402


def count_main(args) -> None:
    scenes = cli.select_scenes(args)
    domain = domains.get(cli.domain_of(scenes))
    store = JsonlStore(args.out)
    config = EstimationConfig()
    for scene in scenes:
        start = time.perf_counter()
        estimator = build_estimator(scene, domain, OracleVLM(), config)
        grounding = time.perf_counter() - start
        problem = parse(scene.domain_pddl, scene.problem_pddl)
        row = {"scene": scene.id, "domain": scene.domain, "size": scene.size,
               "seed": scene.meta.get("seed"), "objects": len(list(problem.all_objects)),
               "ground_and_translate_seconds": grounding,
               "distinct_queries": len(set(estimator.queries.values()))}
        for mask in MASKS:
            atoms, fixed = atoms_for(scene, domain, mask, problem)
            row[f"atoms_{mask}"] = len(atoms)
            row[f"true_{mask}"] = sum(scene.state.get(a, False) for a in atoms)
        store.append(row)
    print(f"wrote {args.out} ({len(scenes)} scenes)")


def backend_label(kind: str, kwargs: dict) -> str:
    """The backend and the setting that changes its speed: HF precision, or
    vLLM's top-k versus full-vocabulary logprobs."""
    if kind == "hf":
        return f"hf-{kwargs.get('torch_dtype', 'default')}"
    if kind == "vllm":
        k = kwargs.get("num_logprobs")
        return f"vllm-top{k}" if k else "vllm-fullvocab"
    return kind


def time_main(args) -> None:
    scenes = cli.select_scenes(args)
    domain = domains.get(cli.domain_of(scenes))
    spec = model_spec(args.model)
    kind = args.backend or spec.get("backend", "hf")
    backend = TimedBackend(build_backend(spec, args.backend, **args.vlm_kwargs))
    backend_kwargs = {**spec.get(f"{kind}_kwargs", {}), **args.vlm_kwargs}
    backend_kind = backend_label(kind, backend_kwargs)
    store = JsonlStore(args.out)
    done = store.keys("model", "backend", "batch_size", "mask", "scene")
    provenance_written = False
    for batch_size in args.batch_sizes:
        for mask in args.masks:
            config = EstimationConfig(scoring=args.scoring, mask=mask, batch_size=batch_size)
            warmed = False
            provenance.reset_peak_memory()
            for scene in scenes:
                key = (spec["name"], backend_kind, batch_size, mask, scene.id)
                if key in done:
                    continue
                estimator = build_estimator(scene, domain, backend, config)
                atoms, _ = atoms_for(scene, domain, mask, estimator.up_problem)
                images = scene.images(args.view)
                if not provenance_written:
                    inner = getattr(backend, "inner", backend)
                    resolved_dtype = None
                    model_obj = getattr(inner, "model", None)
                    if model_obj is not None:
                        resolved_dtype = str(getattr(model_obj, "dtype", None))
                    elif hasattr(inner, "llm"):
                        try:
                            resolved_dtype = str(inner.llm.llm_engine.model_config.dtype)
                        except Exception:
                            resolved_dtype = None
                    write_json(Path(args.out).with_suffix(".run.json"), run_provenance(
                        spec, backend, estimator, config,
                        {"batch_sizes": args.batch_sizes, "masks": args.masks,
                         "backend_label": backend_kind, "backend_kwargs": backend_kwargs,
                         "vlm_kwargs": args.vlm_kwargs, "resolved_dtype": resolved_dtype}))
                    provenance_written = True
                if not warmed:
                    estimate(estimator, backend, scene, atoms[:batch_size], images, scene.state)
                    warmed = True
                _, timing = estimate(estimator, backend, scene, atoms, images, scene.state)
                store.append({
                    "model": spec["name"], "backend": backend_kind, "batch_size": batch_size,
                    "backend_kwargs": backend_kwargs,
                    "mask": mask, "scene": scene.id, "size": scene.size,
                    "seed": scene.meta.get("seed"), "atoms": len(atoms), **timing,
                    "peak_memory": provenance.peak_memory(),
                })
            print(f"{backend_kind} batch {batch_size} mask {mask}: done", flush=True)
    print(f"wrote {args.out}")


def summary_main(args) -> None:
    rows = [r for path in args.files for r in JsonlStore(path).read()]
    fallback = [r for r in rows if r.get("effective_batch_size", r["batch_size"]) < r["batch_size"]]
    if fallback:
        print(f"leaving out {len(fallback)} scenes that ran out of memory at their batch size")
    rows = [r for r in rows if r not in fallback]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["backend"], r["batch_size"], r["mask"], r["size"])].append(r)
    table = []
    print(f"{'model/backend/batch/mask/size':48}{'atoms':>7}{'s/scene':>16}{'queries/s':>11}"
          f"{'in-model':>10}{'peak GB':>9}")
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        g = groups[key]
        seconds = metrics.summarize([r["seconds"] for r in g])
        qps = sum(r["queries"] for r in g) / sum(r["seconds"] for r in g)
        share = sum(r["inference_seconds"] for r in g) / sum(r["seconds"] for r in g)
        peak = max((r["peak_memory"]["bytes"] or 0) for r in g) / 1e9
        atoms = sum(r["atoms"] for r in g) / len(g)
        table.append({"group": list(key), "atoms": atoms, "seconds_per_scene": seconds,
                      "queries_per_second": qps, "in_model_share": share, "peak_gb": peak})
        label = "/".join(str(x) for x in key)
        print(f"{label:48}{atoms:>7.1f}{seconds['mean']:>9.3f}±{seconds['std']:<6.3f}"
              f"{qps:>11.1f}{share:>10.3f}{peak:>9.2f}")
    if args.json:
        write_json(args.json, table)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    count = sub.add_parser("count")
    cli.scene_args(count)
    count.add_argument("--out", type=Path, required=True)
    count.set_defaults(func=count_main)

    timing = sub.add_parser("time")
    cli.scene_args(timing)
    cli.model_args(timing)
    timing.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 16, 32, 64])
    timing.add_argument("--masks", nargs="+", default=["none", "dynamic"], choices=MASKS)
    timing.add_argument("--scoring", default="logprobs", choices=["logprobs", "text_match"])
    timing.add_argument("--view", default="0")
    timing.add_argument("--out", type=Path, required=True)
    timing.set_defaults(func=time_main)

    summary = sub.add_parser("summary")
    summary.add_argument("files", type=Path, nargs="+")
    summary.add_argument("--json", type=Path, default=None)
    summary.set_defaults(func=summary_main)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
