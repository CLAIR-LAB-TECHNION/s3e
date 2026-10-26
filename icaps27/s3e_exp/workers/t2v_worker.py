# SPDX-License-Identifier: MIT
"""Score exported queries with t2v_metrics' VQAScore (run in its own environment).

t2v_metrics pins its own torch/transformers, so it gets a separate
environment (``make env-t2v``). This script imports neither S3E nor the
harness: it reads ``queries.jsonl`` from ``e6_tools.py export`` and writes,
per query, VQAScore's P("Yes") and P("No") (each a full-vocabulary softmax of
the first generated token, as t2v_metrics computes it), the normalized
P(yes) = P(Yes) / (P(Yes) + P(No)), and wall-clock time.

    envs/t2v/bin/python s3e_exp/workers/t2v_worker.py queries.jsonl out.jsonl \\
        --model qwen3-vl-8b
"""

import argparse
import json
import time
from importlib import metadata
from pathlib import Path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("queries", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--model", default="qwen3-vl-8b", help="a t2v_metrics VQAScore model name")
    parser.add_argument("--yes", default="Yes")
    parser.add_argument("--no", default="No")
    args = parser.parse_args(argv)

    import t2v_metrics

    started = time.perf_counter()
    score = t2v_metrics.VQAScore(model=args.model)
    load_seconds = time.perf_counter() - started
    done = set()
    if args.out.exists():
        done = {json.loads(line)["key"] for line in args.out.open(encoding="utf-8") if line.strip()}
    rows = [json.loads(line) for line in args.queries.open(encoding="utf-8") if line.strip()]
    with args.out.open("a", encoding="utf-8") as out:
        for row in rows:
            if row["key"] in done:
                continue
            start = time.perf_counter()
            # question_template="{}" passes S3E's full prompt (question plus answer
            # instruction) unchanged, so both tools see the same text.
            p_yes = float(score(images=[row["image"]], texts=[row["prompt"]],
                                question_template="{}", answer_template=args.yes)[0][0])
            p_no = float(score(images=[row["image"]], texts=[row["prompt"]],
                               question_template="{}", answer_template=args.no)[0][0])
            seconds = time.perf_counter() - start
            total = p_yes + p_no
            out.write(json.dumps({
                "key": row["key"], "p_yes_token": p_yes, "p_no_token": p_no,
                "probability": p_yes / total if total > 0 else 0.5, "has_answer": total > 0,
                "seconds": seconds, "forward_passes": 2,
            }) + "\n")
            out.flush()
    meta = args.out.with_suffix(".run.json")
    meta.write_text(json.dumps({
        "tool": "t2v_metrics", "version": _version("t2v-metrics"), "model": args.model,
        "torch": _version("torch"), "transformers": _version("transformers"),
        "load_seconds": load_seconds, "answer_templates": [args.yes, args.no],
    }, indent=1) + "\n", encoding="utf-8")


def _version(name: str):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


if __name__ == "__main__":
    main()
