# SPDX-License-Identifier: MIT
"""E3: token probabilities versus parsing generated text.

Compares, on the same model and scenes:

* ``logprobs``: S3E's answer-token masses (P(true) from yes/no mass);
* ``text_match``: S3E's text mode (the reply must start with an answer
  token variant; anything else is "no answer");
* ``regex``: the ad-hoc parser common in research code (the first yes/no
  word anywhere in the same generated reply), applied offline to the
  text_match predictions, so it costs the same.

Metrics: accuracy, F1 on true atoms, Brier, ECE, NLL, no-answer rate, and
time per query, each as mean and standard deviation across scene seeds.
Inputs are run directories from ``collect.py``::

    python experiments/collect.py ... --scoring logprobs
    python experiments/collect.py ... --scoring text_match --max-new-tokens 16
    python experiments/e3_scoring.py --instances instances/blocksworld \\
        --runs results/raw/predictions/blocksworld/qwen3-vl-8b/logprobs-none \\
               results/raw/predictions/blocksworld/qwen3-vl-8b/text_match-none \\
        --out results/raw/e3/qwen3-vl-8b.json
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s3e_exp import analysis, metrics, provenance  # noqa: E402
from s3e_exp.estimation import load_predictions  # noqa: E402
from s3e_exp.store import read_json, write_json  # noqa: E402


def method_rows(records, truth, method: str) -> list[dict]:
    out = []
    for record in records:
        results = record["results"]
        if method == "regex":
            results = analysis.regex_results(results)
        out.extend(analysis.rows(record, truth[record["scene"]], results))
    return out


def evaluate(rows: list[dict], records: list[dict]) -> dict:
    by_seed = defaultdict(list)
    for r in rows:
        by_seed[r["seed"]].append(r)
    seconds = sum(r["seconds"] for r in records)
    queries = sum(r["queries"] for r in records)
    return {
        "pooled": {**metrics.atom_metrics(rows), **metrics.scene_metrics(rows)},
        "across_seeds": metrics.across(
            by_seed, lambda rs: {**metrics.atom_metrics(rs), **metrics.scene_metrics(rs)}),
        "seconds_per_query": seconds / queries if queries else None,
        "inference_seconds_per_query": (
            sum(r["inference_seconds"] or 0 for r in records) / queries if queries else None),
        "reliability": metrics.reliability([r["probability"] for r in rows],
                                           [r["truth"] for r in rows]),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--view", default="0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    truth = analysis.load_truth(args.instances)
    report = {"methods": {}, "runs": {}}
    for run_dir in args.runs:
        run = read_json(run_dir / "run.json")
        scoring = run["config"]["scoring"]
        records = [r for r in load_predictions(run_dir)
                   if r["view"] == args.view and r["scene"] in truth]
        methods = ["logprobs"] if scoring == "logprobs" else ["text_match", "regex"]
        for method in methods:
            report["methods"][method] = evaluate(method_rows(records, truth, method), records)
        report["runs"][scoring] = {"dir": str(run_dir), "model": run["model"],
                                   "inference_kwargs": run["inference_kwargs"],
                                   "prompt_template": run["prompt_template"]}
    report["provenance"] = provenance.base()
    write_json(args.out, report)
    print(f"{'method':12}{'acc':>8}{'F1':>8}{'Brier':>8}{'ECE':>8}{'no-ans':>8}{'s/query':>9}")
    for method, result in report["methods"].items():
        m = result["pooled"]
        print(f"{method:12}{m['accuracy']:>8.3f}{m['f1']:>8.3f}{m['brier']:>8.3f}"
              f"{m['ece']:>8.3f}{m['no_answer_rate']:>8.3f}{result['seconds_per_query']:>9.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
