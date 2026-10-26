# SPDX-License-Identifier: MIT
"""E7: earlier versions of S3E.

``run``: the same scenes, model, templates, prompt, and batch size under each
S3E version, each in its own process with that version's overlay on
``PYTHONPATH`` (``make overlay-s3e REF=0.2.0``; ``installed`` is the version
in this environment). torch and transformers are shared, so differences
come from the library. Records the API surface and source size, the
features present, seconds per scene, and P(true) per atom.

``summary``: a table of versions (features, public names, source lines,
seconds per query, accuracy, Brier) and pairwise agreement of P(true).

``workshop``: the metrics of the 2025 workshop paper (accuracy at
theta = 0.3 / 0.5 / 0.7 and micro / macro average precision) for runs of
the current version, printed next to the paper's own numbers
(``results/external/llm4plan2025.json``), e.g. for LLaVA-OneVision on
recorded GroceriesWorld frames (the setting of the paper's experiment 1).

Examples::

    make overlay-s3e REF=0.2.0
    python experiments/e7_versions.py run --instances instances/blocksworld \\
        --sizes 3 5 7 --limit 5 --model qwen3-vl-8b --versions 0.2.0 installed
    python experiments/e7_versions.py summary results/raw/e7/qwen3-vl-8b
    python experiments/e7_versions.py workshop --instances instances/viplan-gw \\
        --run results/raw/predictions/viplan-gw/llava-ov-7b/logprobs-none --row 7B
"""

import argparse
import itertools
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from s3e_exp import analysis, cli, domains, metrics, provenance  # noqa: E402
from s3e_exp.estimation import load_predictions  # noqa: E402
from s3e_exp.models import model_spec  # noqa: E402
from s3e_exp.store import read_json, write_json  # noqa: E402

WORKER = ROOT / "s3e_exp/workers/version_worker.py"
WORKSHOP = ROOT / "results/external/llm4plan2025.json"


def overlay(version: str) -> "Path | None":
    if version == "installed":
        return None
    path = ROOT / "envs" / f"s3e-{version}"
    if not path.exists():
        raise SystemExit(f"no overlay for {version}; run `make overlay-s3e REF={version}`")
    return path


def run_main(args) -> None:
    scenes = cli.select_scenes(args)
    domain = domains.get(cli.domain_of(scenes))
    spec = model_spec(args.model)
    out = cli.run_dir(args.out, spec["name"])
    for version in args.versions:
        target = out / f"{version}.json"
        if target.exists() and not args.force:
            print(f"{target} exists; skipping (use --force)")
            continue
        job = {
            "scenes": [str(Path(s.views[0]).parent) for s in scenes],
            "templates": domain.TEMPLATES, "prompt_template": domain.PROMPT_TEMPLATE,
            "batch_size": args.batch_size, "model": spec["id"], "revision": spec.get("revision"),
            "dtype": args.dtype or spec.get("hf_kwargs", {}).get("torch_dtype", "bfloat16"),
        }
        job_path = out / f"{version}.job.json"
        write_json(job_path, job)
        env = dict(os.environ)
        path = overlay(version)
        if path is not None:
            env["PYTHONPATH"] = os.pathsep.join([str(path), env.get("PYTHONPATH", "")]).rstrip(
                os.pathsep)
        subprocess.run([sys.executable, str(WORKER), str(job_path), str(target)], env=env,
                       check=True)
        report = read_json(target)
        report["provenance"] = provenance.base()
        report["requested_version"] = version
        write_json(target, report)
    print(f"wrote {out}")


def api_main(args) -> None:
    out = cli.run_dir(args.out, "api")
    for version in args.versions:
        job_path = out / f"{version}.job.json"
        write_json(job_path, {"model": None})
        env = dict(os.environ)
        path = overlay(version)
        if path is not None:
            env["PYTHONPATH"] = str(path)
        subprocess.run([sys.executable, str(WORKER), str(job_path), str(out / f"{version}.json")],
                       env=env, check=True)
    print(f"wrote {out}")


def summary_main(args) -> None:
    reports = {p.stem: read_json(p) for p in sorted(Path(args.run).glob("*.json"))
               if not p.name.endswith(".job.json")}
    table = {}
    for name, report in reports.items():
        rows = [{"scene": s["scene"], "truth": s["truth"][a], "probability": p,
                 "state": p >= 0.5, "has_answer": True}
                for s in report["scenes"] for a, p in s["probabilities"].items()]
        atoms = sum(len(s["probabilities"]) for s in report["scenes"])
        seconds = sum(s["seconds"] for s in report["scenes"])
        surface = report["surface"]
        table[name] = {
            "version": surface["version"], "direct_url": surface.get("direct_url"),
            "public_names": len(surface["public_names"]), "modules": surface["modules"],
            "source_lines": surface["source_lines"], "features": surface["features"],
            "load_seconds": report.get("load_seconds"),
            "seconds_per_query": seconds / atoms if atoms else None,
            "queries_per_second": atoms / seconds if seconds else None,
            **({k: v for k, v in metrics.atom_metrics(rows).items()
                if k in ("accuracy", "f1", "brier", "ece")} if rows else {}),
        }
    agreement = {}
    for a, b in itertools.combinations(reports, 2):
        pa = {(s["scene"], k): v for s in reports[a]["scenes"] for k, v in s["probabilities"].items()}
        pb = {(s["scene"], k): v for s in reports[b]["scenes"] for k, v in s["probabilities"].items()}
        shared = set(pa) & set(pb)
        if shared:
            agreement[f"{a} vs {b}"] = {
                "atoms": len(shared),
                "mean_abs_difference": sum(abs(pa[k] - pb[k]) for k in shared) / len(shared),
                "same_decision": sum((pa[k] >= 0.5) == (pb[k] >= 0.5) for k in shared) / len(shared),
            }
    result = {"versions": table, "agreement": agreement}
    if args.json:
        write_json(args.json, result)
    features = sorted({f for t in table.values() for f in t["features"]})
    print(f"{'':28}" + "".join(f"{n:>14}" for n in table))
    for key in ("version", "public_names", "source_lines", "seconds_per_query", "accuracy", "brier"):
        print(f"{key:28}" + "".join(
            f"{t.get(key):>14.4f}" if isinstance(t.get(key), float) else f"{str(t.get(key)):>14}"
            for t in table.values()))
    for feature in features:
        print(f"{feature:28}" + "".join(f"{str(t['features'].get(feature)):>14}"
                                        for t in table.values()))
    for pair, values in agreement.items():
        print(f"{pair}: mean |dP| {values['mean_abs_difference']:.4f}, "
              f"same decision {values['same_decision']:.3f} over {values['atoms']} atoms")


def workshop_main(args) -> None:
    truth = analysis.load_truth(args.instances)
    records = [r for r in load_predictions(args.run) if r["scene"] in truth
               and r["view"] == args.view]
    rows = [row for r in records for row in analysis.rows(r, truth[r["scene"]])]
    ours = metrics.workshop_metrics(rows)
    paper = read_json(WORKSHOP)
    reference = paper["tables"][args.table].get(args.row) if args.row else None
    if isinstance(reference, dict):
        reference = reference.get(str(args.max_objects))
    columns = paper["columns"]
    print(f"{'':16}" + "".join(f"{c:>10}" for c in columns))
    print(f"{'S3E (this run)':16}" + "".join(f"{ours[c]:>10.3f}" for c in columns))
    if reference:
        print(f"{'2025 paper':16}" + "".join(f"{v:>10.2f}" for v in reference))
    if args.out:
        write_json(args.out, {"run": str(args.run), "ours": ours, "paper_row": args.row,
                              "paper_table": args.table, "paper": reference,
                              "atoms": len(rows), "scenes": len({r["scene"] for r in records}),
                              "provenance": provenance.base()})


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    cli.scene_args(run)
    run.add_argument("--model", required=True)
    run.add_argument("--versions", nargs="+", default=["0.2.0", "installed"])
    run.add_argument("--batch-size", type=int, default=16)
    run.add_argument("--dtype", default=None, help="default: the registry's torch_dtype")
    run.add_argument("--force", action="store_true")
    run.add_argument("--out", type=Path, default=Path("results/raw/e7"))
    run.set_defaults(func=run_main)

    api = sub.add_parser("api", help="API surface and features only (no model)")
    api.add_argument("--versions", nargs="+", default=["0.2.0", "installed"])
    api.add_argument("--out", type=Path, default=Path("results/raw/e7"))
    api.set_defaults(func=api_main)

    summary = sub.add_parser("summary")
    summary.add_argument("run", type=Path, help="results/raw/e7/<model> (or .../api)")
    summary.add_argument("--json", type=Path, default=None)
    summary.set_defaults(func=summary_main)

    workshop = sub.add_parser("workshop")
    workshop.add_argument("--instances", type=Path, required=True)
    workshop.add_argument("--run", type=Path, required=True, help="a collect.py logprobs run")
    workshop.add_argument("--view", default="0")
    workshop.add_argument("--table", default="table1_exp1_simulated")
    workshop.add_argument("--row", default=None, help="the paper's row to compare, e.g. 7B")
    workshop.add_argument("--max-objects", type=int, default=None, help="table5: object limit")
    workshop.add_argument("--out", type=Path, default=None)
    workshop.set_defaults(func=workshop_main)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
