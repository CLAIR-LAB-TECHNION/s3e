# SPDX-License-Identifier: MIT
"""E5: reasoning (thinking) models in logprobs mode.

Conditions, each a fresh backend on the same model and scenes:

* ``as-is``: no thinking handling (``thinking_delimiters=None``), which is
  what S3E 0.5.0 does; on templates that open a thinking block by default
  (e.g. Qwen3.5-4B) the next token is the start of a thought, not an answer;
* ``as-is-on``: as ``as-is`` with the template's switch forced on, which
  shows the same failure on models whose templates think only on request;
* ``switch-off``: ``thinking=False``: the chat template's own switch turns
  thinking off (e.g. ``enable_thinking=False``);
* ``closed``: the template opens the block (its switch forced on through
  ``chat_template_kwargs``) and S3E closes it at once, so the answer follows
  an empty block; for templates without a switch this equals ``switch-off``;
* ``think-<B>``: ``thinking=True`` with ``max_thinking_tokens=B``: greedy
  reasoning first, then the answer token is scored.

Reports accuracy, F1, Brier, ECE, answer-token mass coverage (the mass on
yes/no, near 0 when the model is not answering), and seconds per query.

The thinking options are not in the S3E 0.5.0 release; run this script
against an S3E build that has them (see the README, "S3E overlays")::

    make overlay-s3e REF=2a09b7e
    PYTHONPATH=envs/s3e-2a09b7e python experiments/e5_reasoning.py predict \\
        --instances instances/blocksworld --sizes 3 4 5 --limit 10 \\
        --model qwen3.5-4b --conditions as-is switch-off closed think-256 think-1024
    python experiments/e5_reasoning.py summary results/raw/e5/blocksworld/qwen3.5-4b
"""

import argparse
import inspect
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s3e_exp import analysis, cli, domains, metrics, provenance  # noqa: E402
from s3e_exp.estimation import (  # noqa: E402
    EstimationConfig,
    atoms_for,
    build_estimator,
    load_predictions,
    run_provenance,
)
from s3e_exp.models import TimedBackend, build_backend, model_spec  # noqa: E402
from s3e_exp.store import JsonlStore, write_json  # noqa: E402

# Known template switches, in the order S3E probes them.
SWITCHES = ("enable_thinking", "thinking")


def thinking_supported() -> bool:
    from s3e import HuggingFaceVLM

    return "thinking" in inspect.signature(HuggingFaceVLM.__init__).parameters


def condition_kwargs(condition: str, switch: "str | None") -> dict:
    if condition == "as-is":
        return {"thinking_delimiters": None}
    if condition == "as-is-on":
        return {"thinking_delimiters": None,
                **({"chat_template_kwargs": {switch: True}} if switch else {})}
    if condition == "switch-off":
        return {"thinking": False}
    if condition == "closed":
        return {"thinking": False, **({"chat_template_kwargs": {switch: True}} if switch else {})}
    if condition.startswith("think-"):
        return {"thinking": True, "max_thinking_tokens": int(condition.removeprefix("think-"))}
    raise ValueError(f"Unknown condition {condition!r}")


def template_switch(spec: dict) -> "str | None":
    """The thinking switch the model's chat template reads, if any (no weights loaded)."""
    if spec.get("backend") in ("openai", "fake"):
        return None
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(spec["id"], revision=spec.get("revision"))
    template = getattr(processor, "chat_template", None) or getattr(
        getattr(processor, "tokenizer", None), "chat_template", None) or ""
    return next((s for s in SWITCHES if s in str(template)), None)


def reasoning_tokens(backend, text: "str | None") -> int:
    if not text:
        return 0
    tokenizer = getattr(getattr(backend, "processor", None), "tokenizer", None)
    if tokenizer is None:
        return len(text.split())
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def predict_main(args) -> None:
    if not thinking_supported():
        raise SystemExit(
            "This S3E build has no thinking options (S3E 0.5.0 does not). Run E5 against "
            "a build that has them, e.g. `make overlay-s3e REF=2a09b7e` and "
            "PYTHONPATH=envs/s3e-2a09b7e (see README).")
    scenes = cli.select_scenes(args)
    domain = domains.get(cli.domain_of(scenes))
    spec = model_spec(args.model)
    switch = template_switch(spec)
    for condition in args.conditions:
        out = cli.run_dir(args.out, scenes[0].domain, spec["name"], condition)
        store = JsonlStore(out / "predictions.jsonl")
        done = store.keys("scene")
        if all((s.id,) in done for s in scenes):
            continue
        kwargs = condition_kwargs(condition, switch)
        backend = TimedBackend(build_backend(spec, args.backend, **kwargs, **args.vlm_kwargs))
        config = EstimationConfig(scoring="logprobs", mask=args.mask, batch_size=args.batch_size)
        run_written = (out / "run.json").exists()
        for scene in scenes:
            if (scene.id,) in done:
                continue
            estimator = build_estimator(scene, domain, backend, config)
            atoms, _ = atoms_for(scene, domain, args.mask, estimator.up_problem)
            if not run_written:
                write_json(out / "run.json", run_provenance(
                    spec, backend, estimator, config,
                    {"condition": condition, "backend_kwargs": kwargs, "template_switch": switch}))
                run_written = True
            backend.reset()
            start = time.perf_counter()
            results = estimator.estimate(scene.images(args.view), predicates=atoms, keep_raw=True)
            seconds = time.perf_counter() - start
            lengths = [reasoning_tokens(backend.inner, getattr(p.raw, "reasoning", None))
                       for p in results.values()]
            store.append({
                "scene": scene.id, "view": args.view, "size": scene.size,
                "seed": scene.meta.get("seed"), "condition": condition, "seconds": seconds,
                "inference_seconds": backend.seconds, "queries": len(atoms),
                "reasoning_tokens": lengths,
                "reasoning_sample": [getattr(p.raw, "reasoning", None)
                                     for p in list(results.values())[:args.keep_reasoning]],
                "predictions": results.to_dict(),
            })
            print(f"{condition} {scene.id}: {len(atoms)} queries in {seconds:.1f}s", flush=True)
        del backend
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass
    print("done")


def summary_main(args) -> None:
    truth = analysis.load_truth(args.instances)
    report = {}
    print(f"{'condition':14}{'acc':>7}{'F1':>7}{'Brier':>7}{'ECE':>7}{'mass':>7}"
          f"{'argmax':>8}{'think tok':>10}{'s/query':>9}")
    for run_dir in sorted(p for p in Path(args.run_root).iterdir() if p.is_dir()):
        records = [r for r in load_predictions(run_dir) if r["scene"] in truth]
        if not records:
            continue
        rows = [row for r in records for row in analysis.rows(r, truth[r["scene"]])]
        masses = [sum(p.masses.values()) for r in records for p in r["results"].values()]
        argmax = [p.argmax_in_interest for r in records for p in r["results"].values()
                  if p.argmax_in_interest is not None]
        lengths = [n for r in records for n in r["reasoning_tokens"]]
        by_seed = defaultdict(list)
        for row in rows:
            by_seed[row["seed"]].append(row)
        queries = sum(r["queries"] for r in records)
        result = {
            "pooled": {**metrics.atom_metrics(rows), **metrics.scene_metrics(rows)},
            "across_seeds": metrics.across(by_seed, metrics.atom_metrics),
            "answer_mass": metrics.summarize(masses),
            "argmax_in_interest_rate": sum(argmax) / len(argmax) if argmax else None,
            "reasoning_tokens": metrics.summarize(lengths),
            "seconds_per_query": sum(r["seconds"] for r in records) / queries,
        }
        report[run_dir.name] = result
        m = result["pooled"]
        print(f"{run_dir.name:14}{m['accuracy']:>7.3f}{m['f1']:>7.3f}{m['brier']:>7.3f}"
              f"{m['ece']:>7.3f}{result['answer_mass']['mean']:>7.3f}"
              f"{(result['argmax_in_interest_rate'] or 0):>8.3f}"
              f"{result['reasoning_tokens']['mean']:>10.1f}{result['seconds_per_query']:>9.3f}")
    if args.json:
        write_json(args.json, {"conditions": report, "provenance": provenance.base()})


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    predict = sub.add_parser("predict")
    cli.scene_args(predict)
    cli.model_args(predict)
    predict.add_argument("--conditions", nargs="+",
                         default=["as-is", "switch-off", "closed", "think-256", "think-1024"])
    predict.add_argument("--mask", default="dynamic")
    predict.add_argument("--view", default="0")
    predict.add_argument("--batch-size", type=int, default=16)
    predict.add_argument("--keep-reasoning", type=int, default=3,
                         help="reasoning texts kept per scene, for inspection")
    predict.add_argument("--out", type=Path, default=Path("results/raw/e5"))
    predict.set_defaults(func=predict_main)
    summary = sub.add_parser("summary")
    summary.add_argument("run_root", type=Path, help="results/raw/e5/<domain>/<model>")
    summary.add_argument("--instances", type=Path, required=True)
    summary.add_argument("--json", type=Path, default=None)
    summary.set_defaults(func=summary_main)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
