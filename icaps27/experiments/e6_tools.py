# SPDX-License-Identifier: MIT
"""E6: S3E versus other tools on the same images, questions, and model.

``export``: write every (scene, view, atom) query of an instance set, with
S3E's exact prompt, system prompt, image path, and truth, to
``queries.jsonl``. Other tools read this file, so they see the same text.

``adhoc``: run the hand-written single-token P("Yes") scorer that research
code typically uses (``s3e_exp/workers/adhoc_worker.py``) in this
environment.

``t2v``: run t2v_metrics' VQAScore (``s3e_exp/workers/t2v_worker.py``) in
its own environment (``make env-t2v``), on a model both tools support.

``compare``: join the tools' outputs with S3E's stored predictions (a
``collect.py`` logprobs run on the same instances and model) and report
accuracy, F1, Brier, ECE, no-answer rate, seconds per query, and agreement
with S3E's P(true).

``features``: print the feature matrix of related tools
(``configs/feature_matrix.json``), the table that stands in for tools that
cannot do this job at all.

Example::

    python experiments/e6_tools.py export --instances instances/blocksworld \\
        --limit 5 --out results/raw/e6/queries.jsonl
    python experiments/e6_tools.py adhoc --queries results/raw/e6/queries.jsonl \\
        --model qwen3-vl-8b --out results/raw/e6/adhoc-qwen3-vl-8b.jsonl
    python experiments/e6_tools.py t2v --queries results/raw/e6/queries.jsonl \\
        --t2v-model qwen3-vl-8b --out results/raw/e6/t2v-qwen3-vl-8b.jsonl
    python experiments/e6_tools.py compare --queries results/raw/e6/queries.jsonl \\
        --s3e results/raw/predictions/blocksworld/qwen3-vl-8b/logprobs-none \\
        --tool adhoc=results/raw/e6/adhoc-qwen3-vl-8b.jsonl \\
        --tool t2v_metrics=results/raw/e6/t2v-qwen3-vl-8b.jsonl --out results/raw/e6/compare.json
"""

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from s3e_exp import cli, domains, metrics, provenance  # noqa: E402
from s3e_exp.estimation import EstimationConfig, atoms_for, build_estimator, load_predictions  # noqa: E402
from s3e_exp.models import OracleVLM, model_spec  # noqa: E402
from s3e_exp.store import JsonlStore, read_json, write_json  # noqa: E402


def export_main(args) -> None:
    scenes = cli.select_scenes(args)
    domain = domains.get(cli.domain_of(scenes))
    store = JsonlStore(args.out)
    if store.path.exists():
        store.path.unlink()
    count = 0
    for scene in scenes:
        config = EstimationConfig(mask=args.mask, system_prompt=args.system_prompt)
        estimator = build_estimator(scene, domain, OracleVLM(), config)
        atoms, _ = atoms_for(scene, domain, args.mask, estimator.up_problem)
        image = scene.views[int(args.view)]
        if not isinstance(image, (str, Path)):
            raise SystemExit("export needs scenes stored on disk (experiments/instances.py)")
        for atom in atoms:
            query = estimator.queries[atom]
            store.append({
                "key": f"{scene.id}|{args.view}|{atom}", "scene": scene.id, "view": args.view,
                "atom": atom, "size": scene.size, "seed": scene.meta.get("seed"),
                "image": str(Path(image).resolve()), "question": query,
                "prompt": estimator.engine.prompt_template.format(query=query),
                "system_prompt": estimator.engine.system_prompt, "truth": bool(scene.state[atom]),
            })
            count += 1
    print(f"wrote {count} queries to {args.out}")


def adhoc_main(args) -> None:
    spec = model_spec(args.model)
    command = [sys.executable, str(ROOT / "s3e_exp/workers/adhoc_worker.py"), str(args.queries),
               str(args.out), "--model", spec["id"]]
    if spec.get("revision"):
        command += ["--revision", spec["revision"]]
    command += ["--dtype", args.dtype or spec.get("hf_kwargs", {}).get("torch_dtype", "bfloat16")]
    subprocess.run(command, check=True)
    write_json(Path(args.out).with_suffix(".run.json"),
               {**provenance.base(), "tool": "adhoc single-token P(Yes)", "model": spec,
                "dtype": command[-1], "queries_file": str(args.queries)})


def t2v_main(args) -> None:
    python = args.python or os.environ.get("T2V_PYTHON", str(ROOT / "envs/t2v/bin/python"))
    if not Path(python).exists():
        raise SystemExit(f"no t2v_metrics environment at {python}; run `make env-t2v`")
    env = dict(os.environ)
    # The environment's bin holds the ffmpeg t2v_metrics checks for at import.
    env["PATH"] = os.pathsep.join([str(Path(python).parent), env.get("PATH", "")])
    subprocess.run([python, str(ROOT / "s3e_exp/workers/t2v_worker.py"), str(args.queries),
                    str(args.out), "--model", args.t2v_model], check=True, env=env)


def _pearson(xs, ys) -> float:
    n = len(xs)
    if n < 2:
        return math.nan
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if not sx or not sy:
        return math.nan
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def compare_main(args) -> None:
    queries = {r["key"]: r for r in JsonlStore(args.queries).read()}
    s3e_prob, s3e_time = {}, {}
    for record in load_predictions(args.s3e):
        for atom, prediction in record["results"].items():
            key = f"{record['scene']}|{record['view']}|{atom}"
            if key in queries:
                s3e_prob[key] = (prediction.probability, prediction.has_answer)
                s3e_time[key] = record["seconds"] / max(1, record["queries"])
    tools = {"s3e": {k: {"probability": p, "has_answer": h, "seconds": s3e_time[k]}
                     for k, (p, h) in s3e_prob.items()}}
    for item in args.tool:
        name, _, path = item.partition("=")
        tools[name] = {r["key"]: r for r in JsonlStore(Path(path)).read()}
    for name in [n for n, t in tools.items() if not t]:
        print(f"no outputs for {name}; leaving it out")
        del tools[name]
    common = sorted(set(queries).intersection(*[set(t) for t in tools.values()]))
    if not common:
        raise SystemExit("no query has an output from every tool")
    report = {"queries": len(common), "tools": {}}
    for name, outputs in tools.items():
        rows = [{"scene": queries[k]["scene"], "truth": queries[k]["truth"],
                 "probability": outputs[k]["probability"],
                 "state": outputs[k]["probability"] >= 0.5,
                 "has_answer": outputs[k].get("has_answer", True)} for k in common]
        result = {**metrics.atom_metrics(rows), **metrics.scene_metrics(rows),
                  "seconds_per_query": sum(outputs[k]["seconds"] for k in common) / len(common)}
        if "p_yes_token" in next(iter(outputs.values())):
            result["mean_answer_mass"] = sum(
                outputs[k]["p_yes_token"] + outputs[k]["p_no_token"] for k in common) / len(common)
        if name != "s3e" and common:
            ours = [s3e_prob[k][0] for k in common]
            theirs = [outputs[k]["probability"] for k in common]
            result["agreement_with_s3e"] = {
                "pearson": _pearson(ours, theirs),
                "mean_abs_difference": sum(abs(a - b) for a, b in zip(ours, theirs)) / len(common),
                "same_decision": sum((a >= 0.5) == (b >= 0.5) for a, b in zip(ours, theirs))
                / len(common),
            }
        report["tools"][name] = result
    report["s3e_run"] = str(args.s3e)
    report["provenance"] = provenance.base()
    write_json(args.out, report)
    print(f"{len(common)} shared queries")
    print(f"{'tool':14}{'acc':>7}{'F1':>7}{'Brier':>7}{'ECE':>7}{'s/query':>9}{'agree':>7}")
    for name, m in report["tools"].items():
        agree = m.get("agreement_with_s3e", {}).get("same_decision", 1.0)
        print(f"{name:14}{m['accuracy']:>7.3f}{m['f1']:>7.3f}{m['brier']:>7.3f}"
              f"{m['ece']:>7.3f}{m['seconds_per_query']:>9.4f}{agree:>7.3f}")


def features_main(args) -> None:
    matrix = read_json(args.matrix)
    features = matrix["features"]
    print(f"{'tool':28}" + "".join(f"{f['short']:>10}" for f in features))
    for tool in matrix["tools"]:
        cells = "".join(f"{str(tool['features'].get(f['key'], '?')):>10}" for f in features)
        print(f"{tool['name']:28}{cells}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    cli.scene_args(export)
    export.add_argument("--mask", default="none")
    export.add_argument("--view", default="0")
    export.add_argument("--system-prompt", default=None,
                        help="override S3E's system prompt ('' sends none, as t2v_metrics does)")
    export.add_argument("--out", type=Path, required=True)
    export.set_defaults(func=export_main)

    adhoc = sub.add_parser("adhoc")
    adhoc.add_argument("--queries", type=Path, required=True)
    adhoc.add_argument("--model", required=True)
    adhoc.add_argument("--dtype", default=None, help="default: the registry's torch_dtype")
    adhoc.add_argument("--out", type=Path, required=True)
    adhoc.set_defaults(func=adhoc_main)

    t2v = sub.add_parser("t2v")
    t2v.add_argument("--queries", type=Path, required=True)
    t2v.add_argument("--t2v-model", default="qwen3-vl-8b")
    t2v.add_argument("--python", default=None, help="the t2v_metrics environment's python")
    t2v.add_argument("--out", type=Path, required=True)
    t2v.set_defaults(func=t2v_main)

    compare = sub.add_parser("compare")
    compare.add_argument("--queries", type=Path, required=True)
    compare.add_argument("--s3e", type=Path, required=True, help="collect.py logprobs run")
    compare.add_argument("--tool", nargs="*", default=[], help="name=outputs.jsonl")
    compare.add_argument("--out", type=Path, required=True)
    compare.set_defaults(func=compare_main)

    features = sub.add_parser("features")
    features.add_argument("--matrix", type=Path, default=ROOT / "configs/feature_matrix.json")
    features.set_defaults(func=features_main)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
