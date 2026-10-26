# SPDX-License-Identifier: MIT
"""E4: offline calibration and multi-view averaging.

All from stored logprobs predictions (``collect.py --views 0 1 2 3 all``),
without querying the model again:

``scopes``: ECE, Brier, NLL, and accuracy before and after Platt scaling at
``global``, ``lifted``, and ``grounded`` scope, cross-fitted over scenes
(k folds; each scene is scored by a calibrator that never saw it), repeated
for every fold seed.

``transfer``: fit on some problem instances and test on others: on small
problems and test on larger ones (``--fit-sizes`` / ``--test-sizes``), and
across scene seeds (fit on one seed, test on the rest).

``views``: averaging 1 to 4 single-view estimates (every subset of a given
size, so view identity does not decide the result), versus all views in
one prompt; uncalibrated, and calibrated before averaging.

``fit``: fit and save one calibrator on a set of scenes (e.g. for the
closed-loop runs of E1).

Example::

    python experiments/e4_calibration.py scopes --instances instances/blocksworld \\
        --run results/raw/predictions/blocksworld/qwen3-vl-8b/logprobs-none \\
        --out results/raw/e4/scopes-qwen3-vl-8b.json
"""

import argparse
import itertools
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s3e_exp import analysis, metrics, provenance  # noqa: E402
from s3e_exp.estimation import load_predictions  # noqa: E402
from s3e_exp.store import write_json  # noqa: E402

KEYS = ("accuracy", "f1", "brier", "ece", "nll", "no_answer_rate")
LABELS = ("acc", "F1", "Brier", "ECE", "NLL", "no-ans")


def scored(records, results_by_key, truth) -> dict:
    rows = [row for r in records
            for row in analysis.rows(r, truth[r["scene"]],
                                     results_by_key.get((r["scene"], r["view"]), r["results"]))]
    return {**{k: v for k, v in metrics.atom_metrics(rows).items()}, **metrics.scene_metrics(rows)}


def load(args):
    truth = analysis.load_truth(args.instances)
    records = [r for r in load_predictions(args.run) if r["scene"] in truth]
    return truth, records


def scopes_main(args) -> None:
    truth, records = load(args)
    single = [r for r in records if r["view"] == args.view]
    report = {"uncalibrated": scored(single, {}, truth), "scopes": {}}
    for scope in analysis.SCOPES:
        runs = []
        for seed in range(args.repeats):
            calibrated = analysis.cross_fit(single, truth, scope, args.folds, seed=seed)
            runs.append(scored(single, calibrated, truth))
        report["scopes"][scope] = {
            "runs": runs,
            "summary": {k: metrics.summarize([r[k] for r in runs]) for k in KEYS},
        }
    rows = [row for r in single for row in analysis.rows(r, truth[r["scene"]])]
    report["reliability"] = {"uncalibrated": metrics.reliability(
        [r["probability"] for r in rows], [r["truth"] for r in rows])}
    calibrated = analysis.cross_fit(single, truth, "lifted", args.folds)
    rows = [row for r in single for row in
            analysis.rows(r, truth[r["scene"]], calibrated[(r["scene"], r["view"])])]
    report["reliability"]["lifted"] = metrics.reliability(
        [r["probability"] for r in rows], [r["truth"] for r in rows])
    finish(args, report)
    print(f"{'condition':14}" + "".join(f"{k:>10}" for k in LABELS))
    print(f"{'uncalibrated':14}" + "".join(f"{report['uncalibrated'][k]:>10.3f}" for k in KEYS))
    for scope, result in report["scopes"].items():
        print(f"{scope:14}" + "".join(f"{result['summary'][k]['mean']:>10.3f}" for k in KEYS))


def transfer_main(args) -> None:
    truth, records = load(args)
    single = [r for r in records if r["view"] == args.view]
    splits = {}
    if args.fit_sizes and args.test_sizes:
        splits[f"sizes {args.fit_sizes} -> {args.test_sizes}"] = (
            [r for r in single if r["size"] in args.fit_sizes],
            [r for r in single if r["size"] in args.test_sizes])
    seeds = sorted({r["seed"] for r in single})
    for seed in seeds:
        if len(seeds) > 1:
            splits[f"seed {seed} -> others"] = (
                [r for r in single if r["seed"] == seed],
                [r for r in single if r["seed"] != seed])
    report = {"splits": {}}
    for name, (fit, test) in splits.items():
        result = {"fit_scenes": len({r["scene"] for r in fit}),
                  "test_scenes": len({r["scene"] for r in test}),
                  "uncalibrated": scored(test, {}, truth)}
        for scope in analysis.SCOPES:
            calibrator = analysis.fit_calibrator([(r["results"], truth[r["scene"]]) for r in fit],
                                                 scope)
            applied = {(r["scene"], r["view"]): calibrator.apply(r["results"])
                       for r in test} if calibrator else {}
            result[scope] = scored(test, applied, truth)
        report["splits"][name] = result
    finish(args, report)
    for name, result in report["splits"].items():
        print(name)
        for condition in ("uncalibrated", *analysis.SCOPES):
            print(f"  {condition:14}" + "".join(f"{result[condition][k]:>10.3f}" for k in KEYS))


def views_main(args) -> None:
    truth, records = load(args)
    table = analysis.by_scene_view(records)
    singles = sorted({v for views in table.values() for v in views if v.isdigit()})
    if not singles:
        raise SystemExit("no single-view predictions (collect with --views 0 1 2 3)")
    # Compare view counts on the same scenes: those with every single view.
    table = {scene: views for scene, views in table.items() if all(v in views for v in singles)}
    records = [r for r in records if r["scene"] in table]
    calibrated = {}
    if args.calibrate:
        for view in singles:
            one = [r for r in records if r["view"] == view]
            calibrated.update(analysis.cross_fit(one, truth, args.scope, args.folds))
    report = {"views": singles, "scenes": len(table), "by_count": {}}
    for count in range(1, len(singles) + 1):
        subsets = {}
        for subset in itertools.combinations(singles, count):
            for mode in ("raw", "calibrated") if args.calibrate else ("raw",):
                rows = []
                for scene, per_view in table.items():
                    if not all(v in per_view for v in subset):
                        continue
                    if mode == "calibrated":
                        per_view = {v: calibrated[(scene, v)] for v in subset}
                    averaged = analysis.average_views(per_view, list(subset))
                    record = {"scene": scene, "view": "avg", "size": None, "seed": None}
                    rows.extend(analysis.rows(record, truth[scene], averaged))
                subsets[f"{mode}:{','.join(subset)}"] = {
                    **metrics.atom_metrics(rows), **metrics.scene_metrics(rows)}
        report["by_count"][count] = {
            "subsets": subsets,
            "summary": {mode: {k: metrics.summarize([m[k] for key, m in subsets.items()
                                                     if key.startswith(mode + ":")])
                               for k in (*KEYS, "exact_match")}
                        for mode in (("raw", "calibrated") if args.calibrate else ("raw",))},
        }
    together = [r for r in records if r["view"] == "all"]
    if together:
        report["all_in_one_prompt"] = scored(together, {}, truth)
        seconds = defaultdict(float)
        for r in records:
            seconds[r["view"]] += r["seconds"]
        report["seconds_by_view"] = dict(seconds)
    finish(args, report)
    for count, result in report["by_count"].items():
        for mode, summary in result["summary"].items():
            print(f"{count} view(s) {mode:11}" + "".join(
                f"{k}={summary[k]['mean']:.3f} " for k in ("accuracy", "f1", "ece",
                                                          "exact_match")))
    if together:
        m = report["all_in_one_prompt"]
        print("all in one prompt   " + "".join(
            f"{k}={m[k]:.3f} " for k in ("accuracy", "f1", "ece", "exact_match")))


def fit_main(args) -> None:
    truth, records = load(args)
    single = [r for r in records if r["view"] == args.view]
    if args.sizes:
        single = [r for r in single if r["size"] in args.sizes]
    if args.seeds:
        single = [r for r in single if r["seed"] in args.seeds]
    calibrator = analysis.fit_calibrator([(r["results"], truth[r["scene"]]) for r in single],
                                         args.scope)
    if calibrator is None:
        raise SystemExit("no answered predictions to fit on")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    calibrator.save(args.out)
    print(f"fitted {args.scope} Platt scaling on {len(single)} scenes -> {args.out}")


def finish(args, report) -> None:
    report["run"] = str(args.run)
    report["provenance"] = provenance.base()
    write_json(args.out, report)
    print(f"wrote {args.out}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func in (("scopes", scopes_main), ("transfer", transfer_main),
                       ("views", views_main), ("fit", fit_main)):
        p = sub.add_parser(name)
        p.add_argument("--instances", type=Path, required=True)
        p.add_argument("--run", type=Path, required=True,
                       help="a logprobs run directory written by collect.py")
        p.add_argument("--view", default="0")
        p.add_argument("--folds", type=int, default=5)
        p.add_argument("--scope", default="lifted", choices=analysis.SCOPES)
        p.add_argument("--out", type=Path, required=True)
        p.set_defaults(func=func)
        if name == "scopes":
            p.add_argument("--repeats", type=int, default=3, help="fold assignments to average")
        if name == "transfer":
            p.add_argument("--fit-sizes", type=int, nargs="*", default=[3, 4, 5])
            p.add_argument("--test-sizes", type=int, nargs="*", default=[6, 7])
        if name == "views":
            p.add_argument("--calibrate", action="store_true",
                           help="also calibrate each view (cross-fitted) before averaging")
        if name == "fit":
            p.add_argument("--sizes", type=int, nargs="*", default=None)
            p.add_argument("--seeds", type=int, nargs="*", default=None)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
