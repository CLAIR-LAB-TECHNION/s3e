# SPDX-License-Identifier: MIT
"""E1 (headline): planning success from estimated states.

Two parts:

``plan``: offline, from predictions stored by ``collect.py``. For every
scene, decode the estimate into a state, plan from it with Fast Downward
through Unified Planning, and replay the plan in the true initial state.
Decoders: ``threshold`` (P(true) >= 0.5), ``calibrated`` (cross-fitted Platt
scaling, then threshold), ``map`` (most probable physically consistent
state; Blocksworld), ``calibrated+map``, and ``views-avg`` (S3E's averaging
over every stored single view, next to ``views-single``, the first view alone on
the same scenes). Reference rows: ``oracle`` (the true state)
and ``noise-<p>`` (each estimated atom of the true state flipped with
probability p), which map predicate accuracy to planning success.

``online``: closed loop with the model in the loop on a simulated world
(rendered Blocksworld), with the ``open-loop``, ``replan``, and ``monitor``
policies (see ``s3e_exp/closed_loop.py``).

``summary``: aggregate either output by (model, size, decoder/policy) with
mean and standard deviation across seeds.

Examples::

    python experiments/e1_closed_loop.py plan --instances instances/blocksworld \\
        --predictions results/raw/predictions/blocksworld/qwen3-vl-8b/logprobs-none \\
        --out results/raw/e1/plan-qwen3-vl-8b.jsonl
    python experiments/e1_closed_loop.py online --instances instances/blocksworld \\
        --model qwen3-vl-8b --policy monitor --limit 20 --out results/raw/e1/online
    python experiments/e1_closed_loop.py summary results/raw/e1/plan-*.jsonl
"""

import argparse
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s3e_exp import analysis, cli, domains, metrics, provenance  # noqa: E402
from s3e_exp.estimation import EstimationConfig, atoms_for, build_estimator, load_predictions  # noqa: E402
from s3e_exp.models import TimedBackend, build_backend, model_spec  # noqa: E402
from s3e_exp.planning import parse, plan_and_validate  # noqa: E402
from s3e_exp.store import JsonlStore, read_json, write_json  # noqa: E402

DECODERS = ("threshold", "calibrated", "map", "calibrated+map", "hard-map", "views-single", "views-avg")


def _worker_init():
    # Fast Downward writes its intermediate files to the working directory, so
    # parallel planners need one directory each.
    import os
    import tempfile

    os.chdir(tempfile.mkdtemp(prefix="e1-planner-"))


def _plan_job(job):
    domain_pddl, problem_pddl, state, planner, timeout, key = job
    return key, plan_and_validate(domain_pddl, problem_pddl, state, planner, timeout)


def plan_main(args) -> None:
    scenes = {s.id: s for s in cli.select_scenes(args)}
    domain = domains.get(cli.domain_of(list(scenes.values())))
    store = JsonlStore(args.out)
    settings = {"planner": args.planner, "timeout": args.timeout, "scope": args.scope,
                "folds": args.folds, "view": args.view}
    meta_path = Path(args.out).with_suffix(".run.json")
    if store.path.exists() and meta_path.exists():
        previous = read_json(meta_path).get("settings")
        if previous is not None and previous != settings:
            raise SystemExit(f"{args.out} was written with {previous}; use a new --out for "
                             f"{settings}")
    done = store.keys("model", "scene", "view", "decoder")
    jobs, info = [], {}

    def add(model, scene, view, decoder, estimated, atoms, fixed):
        key = (model, scene.id, view, decoder)
        if key in done:
            return
        errors = sum(estimated[a] != scene.state[a] for a in atoms)
        info[key] = {"model": model, "domain": scene.domain, "scene": scene.id, "view": view,
                     "decoder": decoder, "size": scene.size, "seed": scene.meta.get("seed"), "atoms": len(atoms),
                     "atom_errors": errors, "exact": errors == 0}
        jobs.append((scene.domain_pddl, scene.problem_pddl, {**fixed, **estimated},
                     args.planner, args.timeout, key))

    masks = {}

    def mask_of(scene, mask):
        if (scene.id, mask) not in masks:
            masks[(scene.id, mask)] = atoms_for(scene, domain, mask)
        return masks[(scene.id, mask)]

    for path in args.predictions:
        run = read_json(Path(path) / "run.json")
        model = run["model"]["name"] + ("" if run["config"]["mask"] == "none"
                                        else f"[{run['config']['mask']}]")
        tag = Path(path).name.removeprefix(f"{run['config']['scoring']}-{run['config']['mask']}")
        model += f"[{tag.lstrip('-')}]" if tag else ""
        records = [r for r in load_predictions(path) if r["scene"] in scenes]
        calibrated = {}
        if any(d.startswith("calibrated") for d in args.decoders):
            single = [r for r in records if r["view"] == args.view]
            calibrated = analysis.cross_fit(single, {s: scenes[s].state for s in scenes},
                                            scope=args.scope, folds=args.folds)
        per_scene = analysis.by_scene_view(records)
        for scene_id, views in per_scene.items():
            scene = scenes[scene_id]
            atoms, fixed = mask_of(scene, run["config"]["mask"])
            objects = scene.meta.get("blocks")
            for decoder in args.decoders:
                if decoder in ("views-avg", "views-single"):
                    # Both rows cover only scenes with every single view, so the
                    # averaged and single-view numbers compare the same scenes.
                    singles = sorted(v for v in views if v.isdigit())
                    if len(singles) < 2:
                        continue
                    if decoder == "views-single":
                        estimated = analysis.decode(views[singles[0]])
                        add(model, scene, singles[0], decoder, estimated, list(views[singles[0]]),
                            fixed)
                        continue
                    results = analysis.average_views(views, singles)
                    view = "avg:" + ",".join(singles)
                    estimated = analysis.decode(results)
                else:
                    if args.view not in views:
                        continue
                    view = args.view
                    results = views[view]
                    if decoder.startswith("calibrated"):
                        results = calibrated.get((scene_id, view), results)
                    kind = decoder if decoder == "hard-map" else "map" if decoder.endswith("map") else "threshold"
                    if kind == "map" and (not hasattr(domain, "map_decode")
                                          or len(objects or ()) > 8):
                        continue
                    estimated = analysis.decode(results, kind, domain=domain, objects=objects)
                add(model, scene, view, decoder, estimated, list(results), fixed)

    for scene in scenes.values():
        atoms, fixed = mask_of(scene, "none")
        if "oracle" not in args.no_reference:
            add("oracle", scene, "-", "oracle", {a: scene.state[a] for a in atoms}, atoms, {})
        for rate in args.noise:
            noisy = analysis.noisy_state(scene.state, atoms, rate, f"{scene.id}:{rate}")
            add("oracle", scene, "-", f"noise-{rate}", noisy, atoms, {})

    print(f"planning {len(jobs)} estimated states with {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as pool:
        for i, (key, outcome) in enumerate(pool.map(_plan_job, jobs, chunksize=4)):
            store.append({**info[key], **outcome})
            if i % 200 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    write_json(meta_path, {
        **provenance.base(), "predictions": [str(p) for p in args.predictions],
        "settings": settings, "decoders": args.decoders, "noise": args.noise,
        "finished_utc": provenance.now(),
    })
    print(f"wrote {args.out}")


def online_main(args) -> None:
    from s3e_exp.closed_loop import run_episode

    scenes = cli.select_scenes(args)
    domain = domains.get(cli.domain_of(scenes))
    if not hasattr(domain, "World"):
        raise SystemExit(f"domain {scenes[0].domain} has no simulated world for closed loop")
    spec = model_spec(args.model)
    backend = TimedBackend(build_backend(spec, args.backend, **args.vlm_kwargs))
    calibrator = None
    if args.calibrator:
        from s3e import PlattCalibrator

        calibrator = PlattCalibrator.load(args.calibrator)
    config = EstimationConfig(scoring="logprobs", mask=args.mask, batch_size=args.batch_size)
    name = f"{spec['name']}-{args.policy}-{args.check}-{args.decoder}-{args.mask}"
    if calibrator:
        name += "-calibrated"
    out = cli.run_dir(args.out, scenes[0].domain, name)
    store = JsonlStore(out / "episodes.jsonl")
    done = store.keys("scene")
    run_path = out / "run.json"
    for scene in scenes:
        if (scene.id,) in done:
            continue
        estimator = build_estimator(scene, domain, backend, config)
        problem = parse(scene.domain_pddl, scene.problem_pddl)
        atoms, fixed = atoms_for(scene, domain, args.mask, problem)
        if not run_path.exists():
            from s3e_exp.estimation import run_provenance

            write_json(run_path, run_provenance(spec, backend, estimator, config, {
                "policy": args.policy, "check": args.check, "decoder": args.decoder,
                "calibrator": args.calibrator, "views": args.views,
                "max_steps_factor": args.max_steps_factor, "planner": args.planner}))
        reference = plan_and_validate(scene.domain_pddl, scene.problem_pddl, scene.state,
                                      args.planner, args.timeout)
        optimal = reference["plan_length"] or 1
        world = domain.World(scene)
        outcome = run_episode(
            world, problem, estimator, backend, atoms, fixed, policy=args.policy,
            check=args.check, decoder=args.decoder, calibrator=calibrator, domain=domain,
            objects=scene.meta.get("blocks"), views=args.views,
            max_steps=int(args.max_steps_factor * optimal) + 5, planner=args.planner,
            timeout=args.timeout)
        store.append({"scene": scene.id, "domain": scene.domain, "size": scene.size,
                      "seed": scene.meta.get("seed"), "model": spec["name"],
                      "policy": args.policy, "check": args.check, "decoder": args.decoder,
                      "mask": args.mask, "calibrated": bool(calibrator), "views": args.views,
                      "reference_plan_length": reference["plan_length"], **outcome})
        print(f"{scene.id}: {outcome['status']} success={outcome['success']} "
              f"steps={outcome['steps']} replans={outcome['replans']} "
              f"queries={outcome['queries']}", flush=True)
    print(f"wrote {out}")


def summary_main(args) -> None:
    records = [r for path in args.files for r in JsonlStore(path).read()]
    online = bool(records) and "policy" in records[0]
    group_keys = ("domain", "model", "policy", "check", "decoder", "mask", "calibrated") \
        if online else ("domain", "model", "decoder")
    groups = defaultdict(lambda: defaultdict(list))
    for r in records:
        if args.by_size:
            groups[tuple(r.get(k) for k in group_keys) + (r["size"],)][r.get("seed")].append(r)
        else:
            groups[tuple(r.get(k) for k in group_keys)][r.get("seed")].append(r)

    def rates(rows):
        n = len(rows)
        if online:
            return {"n": n, "success": sum(r["success"] for r in rows) / n,
                    "steps": sum(r["steps"] for r in rows) / n,
                    "replans": sum(r["replans"] for r in rows) / n,
                    "queries": sum(r["queries"] for r in rows) / n,
                    "failed_actions": sum(r["failed_actions"] for r in rows) / n}
        return {"n": n,
                "valid": sum(r["valid"] for r in rows) / n,
                "executable": sum(r["executable"] for r in rows) / n,
                "solved": sum(r["status"] == "solved" for r in rows) / n,
                "exact": sum(r["exact"] for r in rows) / n,
                "atom_accuracy": 1 - sum(r["atom_errors"] for r in rows)
                / max(1, sum(r["atoms"] for r in rows))}

    table = []
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        summary = metrics.across(groups[key], rates)
        table.append({"group": list(key), **{k: v for k, v in summary.items()}})
    columns = ["success", "steps", "replans", "queries"] if online else \
        ["valid", "executable", "exact", "atom_accuracy"]
    header = " / ".join(group_keys + (("size",) if args.by_size else ()))
    print(f"{header:60}" + "".join(f"{c:>18}" for c in columns))
    for row in table:
        label = " / ".join(str(x) for x in row["group"])
        cells = "".join(f"{row[c]['mean']:>11.3f}±{row[c]['std']:<6.3f}" for c in columns)
        print(f"{label:60}{cells}")
    if args.json:
        write_json(args.json, table)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="offline planning from stored predictions")
    cli.scene_args(plan)
    plan.add_argument("--predictions", type=Path, nargs="*", default=[],
                      help="run directories written by collect.py (logprobs)")
    plan.add_argument("--decoders", nargs="+", default=list(DECODERS), choices=DECODERS)
    plan.add_argument("--view", default="0", help="view spec decoded by single-view decoders")
    plan.add_argument("--scope", default="lifted", choices=analysis.SCOPES)
    plan.add_argument("--folds", type=int, default=5)
    plan.add_argument("--noise", type=float, nargs="*", default=[0.01, 0.02, 0.05, 0.1])
    plan.add_argument("--no-reference", nargs="*", default=[], choices=["oracle"])
    plan.add_argument("--planner", default="fast-downward")
    plan.add_argument("--timeout", type=float, default=60.0)
    plan.add_argument("--workers", type=int, default=4)
    plan.add_argument("--out", type=Path, required=True, help="JSONL of planning outcomes")
    plan.set_defaults(func=plan_main)

    online = sub.add_parser("online", help="closed loop with the model in the loop")
    cli.scene_args(online)
    cli.model_args(online)
    online.add_argument("--policy", choices=["open-loop", "replan", "monitor"], default="monitor")
    online.add_argument("--check", choices=["full", "action"], default="full",
                        help="monitor: re-estimate everything, or only the atoms of the "
                             "last and next action")
    online.add_argument("--decoder", choices=["threshold", "map"], default="threshold")
    online.add_argument("--calibrator", type=Path, default=None,
                        help="a PlattCalibrator JSON (e.g. from e4_calibration.py fit)")
    online.add_argument("--mask", default="none", choices=["none", "dynamic", "relevant"])
    online.add_argument("--views", type=int, nargs="*", default=[0])
    online.add_argument("--batch-size", type=int, default=16)
    online.add_argument("--max-steps-factor", type=float, default=3.0,
                        help="step budget as a multiple of the optimal-ish plan length")
    online.add_argument("--planner", default="fast-downward")
    online.add_argument("--timeout", type=float, default=60.0)
    online.add_argument("--out", type=Path, default=Path("results/raw/e1/online"))
    online.set_defaults(func=online_main)

    summary = sub.add_parser("summary", help="aggregate plan or online outcomes")
    summary.add_argument("files", type=Path, nargs="+")
    summary.add_argument("--by-size", action="store_true")
    summary.add_argument("--json", type=Path, default=None)
    summary.set_defaults(func=summary_main)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
