# SPDX-License-Identifier: MIT
"""Generate (or export) an instance set: scenes with views, PDDL, and truth.

Rendered Blocksworld runs anywhere::

    python experiments/instances.py blocksworld --sizes 3 4 5 6 7 \\
        --seeds 0 1 2 --count 100 --out instances/blocksworld

Live ViPlan++ environments (PRB, GW, ALF) must run under a Python that can
import ``viplan`` (see README, "ViPlan++ domains")::

    python experiments/instances.py viplan-prb --sizes 4 5 6 --seeds 0 1 2 \\
        --count 50 --out instances/viplan-prb

Recorded ViPlan++ episodes are exported frame by frame without ``viplan``::

    python experiments/instances.py viplan-gw --export $VIPLAN_DATA/episodes/gw \\
        --frames all --out instances/viplan-gw

The set is deterministic in its arguments; ``index.json`` records them.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s3e_exp import domains, provenance  # noqa: E402
from s3e_exp.scenes import save_scene  # noqa: E402
from s3e_exp.store import write_json  # noqa: E402


def complete_state(scene) -> int:
    """Give every atom S3E grounds a truth value (closed world); returns how
    many were missing from the source's state."""
    from s3e.pddl import ground_predicates, parse_domain_problem

    atoms = ground_predicates(parse_domain_problem(scene.domain_pddl, scene.problem_pddl))
    missing = [a for a in atoms if a not in scene.state]
    for atom in missing:
        scene.state[atom] = False
    return len(missing)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("domain", choices=sorted(domains.ADAPTERS))
    parser.add_argument("--sizes", type=int, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--count", type=int, default=10, help="scenes per (size, seed)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--options", default="{}",
                        help="JSON generator options for the domain adapter")
    parser.add_argument("--export", type=Path, default=None,
                        help="ViPlan++ only: export recorded episodes under this directory")
    parser.add_argument("--cells", nargs="*", default=None, help="--export: cells to include")
    parser.add_argument("--frames", choices=["initial", "all"], default="all")
    parser.add_argument("--episodes", type=int, default=None, help="--export: per cell")
    parser.add_argument("--every", type=int, default=1, help="--export: keep every k-th step")
    args = parser.parse_args(argv)
    if args.export is None and not args.sizes:
        parser.error("--sizes is required unless --export is given")

    import json

    options = json.loads(args.options)
    adapter = domains.get(args.domain)
    index = []

    def keep(scene):
        completed = complete_state(scene)
        save_scene(scene, args.out)
        index.append({"id": scene.id, "size": scene.size, "seed": scene.meta.get("seed"),
                      "atoms": len(scene.state), "views": scene.num_views,
                      "completed_atoms": completed})

    if args.export is not None:
        for scene in adapter.export(args.export, cells=args.cells, frames=args.frames,
                                    episodes=args.episodes, every=args.every):
            keep(scene)
        print(f"{args.domain}: exported {len(index)} frames", flush=True)
    else:
        for size in args.sizes:
            for seed in args.seeds:
                for scene in adapter.generate(size, args.count, seed, **options):
                    keep(scene)
                print(f"{args.domain}: size {size} seed {seed}: {args.count} scenes", flush=True)
    write_json(args.out / "index.json", {
        "domain": args.domain, "sizes": args.sizes, "seeds": args.seeds, "count": args.count,
        "options": options, "export": str(args.export) if args.export else None,
        "cells": args.cells, "frames": args.frames, "scenes": index,
        "provenance": provenance.base(),
    })
    print(f"wrote {len(index)} scenes to {args.out}")


if __name__ == "__main__":
    main()
