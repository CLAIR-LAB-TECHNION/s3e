# SPDX-License-Identifier: MIT
"""Offline views of stored predictions: rows, calibration, decoding, baselines.

Nothing here queries a model. Calibration is cross-fitted by scene (each
scene is calibrated by a calibrator that never saw it), so held-out metrics
over all scenes are honest and use every scene once.
"""

import random
import re
from collections import defaultdict
from pathlib import Path

from s3e import (
    CalibrationSample,
    CalibrationSet,
    Prediction,
    PlattCalibrator,
    PredictionSet,
)

from .masking import fluent_of
from .store import read_json

SCOPES = ("global", "lifted", "grounded")


def load_truth(instances: Path) -> dict[str, dict[str, bool]]:
    """``{scene id: true state}`` for every scene under ``instances``."""
    truth = {}
    for meta in Path(instances).rglob("meta.json"):
        scene_id = read_json(meta)["id"]
        truth[scene_id] = read_json(meta.parent / "state.json")
    return truth


def load_meta(instances: Path) -> dict[str, dict]:
    return {m["id"]: m for m in (read_json(p) for p in Path(instances).rglob("meta.json"))}


def to_state(results: PredictionSet, confidence: float = 0.5) -> dict[str, bool]:
    """Threshold P(true), with unanswered atoms False (the PDDL closed world).

    S3E gives an unanswered prediction P(true) = 0.5, which a 0.5 threshold
    would turn into True; every method here (logprobs, text, regex, other
    tools) instead treats "no answer" as False, the same way.
    """
    return {atom: bool(p.has_answer and p.probability >= confidence)
            for atom, p in results.items()}


def rows(record: dict, truth: dict[str, bool], results: "PredictionSet | None" = None,
         state: "dict[str, bool] | None" = None, confidence: float = 0.5) -> list[dict]:
    """One row per estimated atom (see :mod:`s3e_exp.metrics`)."""
    results = results if results is not None else record["results"]
    state = state if state is not None else to_state(results, confidence)
    return [
        {
            "scene": record["scene"], "view": record.get("view"), "size": record.get("size"),
            "seed": record.get("seed"), "atom": atom, "lifted": fluent_of(atom),
            "truth": bool(truth[atom]), "probability": prediction.probability,
            "state": bool(state[atom]), "has_answer": prediction.has_answer,
            "argmax_in_interest": prediction.argmax_in_interest,
            "unassigned_mass": prediction.unassigned_mass,
        }
        for atom, prediction in results.items()
    ]


# --- calibration ---


def calibration_set(pairs, meta: "dict | None" = None) -> CalibrationSet:
    """Samples from ``(PredictionSet, true state)`` pairs; unanswered skipped."""
    samples = [
        CalibrationSample(atom, prediction.score, bool(state[atom]))
        for results, state in pairs
        for atom, prediction in results.items()
        if prediction.has_answer
    ]
    first = next(iter(pairs), None)
    answers = next(iter(first[0].values())).answers.to_dict() if first and first[0] else None
    return CalibrationSet(samples=samples,
                          meta={"scoring": "logprobs", "answers": answers, **(meta or {})})


def fit_calibrator(pairs, scope: str = "lifted") -> "PlattCalibrator | None":
    data = calibration_set(pairs)
    if not data.samples:
        return None
    try:
        return PlattCalibrator.fit(data, scope=scope, pass_through_single_class=True)
    except ValueError:  # e.g. one class overall
        return None


def folds_by(keys: list, folds: int, seed: int = 0) -> dict:
    """Assign each distinct key to one of ``folds`` folds at random."""
    distinct = sorted(set(keys))
    random.Random(seed).shuffle(distinct)
    return {key: i % folds for i, key in enumerate(distinct)}


_STEP = re.compile(r"-t\d+$")


def scene_group(record: dict) -> str:
    """The unit calibration folds split on: the scene, or for frames of a
    recorded episode (ids ending in ``-t<step>``) the whole episode, whose
    frames are near duplicates."""
    return _STEP.sub("", record["scene"])


def cross_fit(records: list[dict], truth: dict, scope: str = "lifted", folds: int = 5,
              group=scene_group, seed: int = 0) -> dict[tuple, PredictionSet]:
    """Calibrated results per ``(scene, view)``, each from a calibrator fitted
    on the other folds' groups (scenes or episodes)."""
    assignment = folds_by([group(r) for r in records], folds, seed)
    calibrated = {}
    for fold in range(folds):
        train = [(r["results"], truth[r["scene"]]) for r in records
                 if assignment[group(r)] != fold]
        test = [r for r in records if assignment[group(r)] == fold]
        if not test:
            continue
        calibrator = fit_calibrator(train, scope)
        for r in test:
            key = (r["scene"], r.get("view"))
            calibrated[key] = calibrator.apply(r["results"]) if calibrator else r["results"]
    return calibrated


# --- decoding into a planner-ready state ---


def decode(results: PredictionSet, decoder: str = "threshold", *, confidence: float = 0.5,
           domain=None, objects: "list[str] | None" = None) -> dict[str, bool]:
    """Boolean state over the estimated atoms.

    ``threshold``: P(true) >= confidence. ``map``: the most probable
    physically consistent state (domains with ``map_decode`` only).
    """
    if decoder == "threshold":
        return to_state(results, confidence)
    if decoder == "map":
        if domain is None or not hasattr(domain, "map_decode"):
            raise ValueError("the map decoder needs a domain with map_decode")
        full = domain.map_decode(results.probabilities(), objects)
        return {atom: full[atom] for atom in results}
    raise ValueError(f"Unknown decoder {decoder!r}")


def noisy_state(truth: dict[str, bool], atoms, rate: float, seed: str) -> dict[str, bool]:
    """The true values of ``atoms`` with each flipped with probability ``rate``."""
    rng = random.Random(seed)
    return {a: (not truth[a]) if rng.random() < rate else bool(truth[a]) for a in atoms}


# --- baselines and aggregation ---

_WORD = re.compile(r"\b(yes|no|true|false)\b", re.IGNORECASE)


def regex_results(results: PredictionSet) -> PredictionSet:
    """The ad-hoc parser most research code uses: the first yes/no word
    *anywhere* in the generated reply decides; no match means no answer.

    Applied to text-mode predictions, so it costs nothing extra.
    """
    parsed = {}
    for atom, prediction in results.items():
        answers = prediction.answers
        match = _WORD.search(prediction.text or "")
        masses = {label: 0.0 for label in prediction.masses}
        if match:
            word = match.group(1).lower()
            label = answers.true_label if word in ("yes", "true") else answers.false_label
            masses[label] = 1.0
        parsed[atom] = Prediction(prediction.query, masses, 0.0,
                                  0.0 if match else 1.0, answers, text=prediction.text)
    return PredictionSet(parsed)


def by_scene_view(records: list[dict]) -> dict[str, dict[str, PredictionSet]]:
    table: dict = defaultdict(dict)
    for r in records:
        table[r["scene"]][r["view"]] = r["results"]
    return table


def average_views(per_view: dict[str, PredictionSet], views: list[str]) -> PredictionSet:
    """S3E's multi-view averaging of stored masses over the given views."""
    return PredictionSet.average([per_view[v] for v in views])
