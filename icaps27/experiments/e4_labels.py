# SPDX-License-Identifier: MIT
"""E4 (labels): how many labeled scenes lifted Platt calibration needs.

Offline, from stored predictions: fit a lifted Platt calibrator on N scenes
drawn at random from seeds 0 and 1 (front view), apply it to every seed-2
scene, and report atom accuracy, ECE, and the exact-state rate after
calibrated MAP decoding. In E1, a MAP-decoded estimate gave a valid plan
exactly when it was the true state, so the exact-state rate after MAP is the
valid-plan rate without calling the planner.

    python experiments/e4_labels.py --instances instances/blocksworld \\
        --run results/raw/predictions/blocksworld/qwen3-vl-4b/logprobs-none \\
        --out results/raw/e4/labels-qwen3-vl-4b.json
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s3e_exp import analysis, domains, metrics, provenance  # noqa: E402
from s3e_exp.estimation import load_predictions  # noqa: E402
from s3e_exp.store import write_json  # noqa: E402


def evaluate(records, truth, meta, calibrator, domain):
    probs, labels, exact = [], [], []
    for r in records:
        results = calibrator.apply(r["results"]) if calibrator else r["results"]
        state = truth[r["scene"]]
        for atom, prediction in results.items():
            probs.append(prediction.probability)
            labels.append(bool(state[atom]))
        decoded = analysis.decode(results, "map", domain=domain, objects=meta[r["scene"]]["blocks"])
        exact.append(all(decoded[a] == bool(state[a]) for a in decoded))
    accuracy = sum((p >= 0.5) == y for p, y in zip(probs, labels)) / len(labels)
    return {"accuracy": accuracy, "ece": metrics.expected_calibration_error(probs, labels),
            "exact_map": sum(exact) / len(exact)}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 50, 200, 1000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--view", default="0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    truth, meta = analysis.load_truth(args.instances), analysis.load_meta(args.instances)
    domain = domains.get("blocksworld")
    records = [r for r in load_predictions(args.run) if r["view"] == args.view]
    pool = [r for r in records if meta[r["scene"]]["seed"] in (0, 1)]
    test = [r for r in records if meta[r["scene"]]["seed"] == 2]
    rows = [{"labeled_scenes": 0, "repeat": 0, **evaluate(test, truth, meta, None, domain)}]
    for n in args.sizes:
        for repeat in range(args.repeats if n < len(pool) else 1):
            train = random.Random(f"{n}:{repeat}").sample(pool, min(n, len(pool)))
            calibrator = analysis.fit_calibrator([(r["results"], truth[r["scene"]]) for r in train],
                                                 "lifted")
            rows.append({"labeled_scenes": n, "repeat": repeat,
                         **evaluate(test, truth, meta, calibrator, domain)})
            print(rows[-1], flush=True)
    write_json(args.out, {**provenance.base(), "run": str(args.run), "test_scenes": len(test),
                          "rows": rows, "finished_utc": provenance.now()})


if __name__ == "__main__":
    main()
