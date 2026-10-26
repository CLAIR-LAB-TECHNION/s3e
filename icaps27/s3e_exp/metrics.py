# SPDX-License-Identifier: MIT
"""Prediction and planning metrics, with dispersion across runs.

Per-atom rows are dicts with ``truth`` (bool), ``probability`` (P(true)),
``state`` (thresholded bool), ``has_answer`` (bool), and ``scene``.
Accuracy is reported next to F1 on true atoms, because most grounded atoms
are false (e.g. 49 of the 64 ``on`` atoms with 8 blocks in one tower), which
makes accuracy flattering.
"""

import math
import random
from collections import defaultdict
from collections.abc import Callable, Sequence

EPS = 1e-6


def expected_calibration_error(probabilities, labels, bins: int = 10) -> float:
    """Mean |frequency - confidence| over equal-width bins of P(true)."""
    total, error = len(labels), 0.0
    if not total:
        return math.nan
    for b in range(bins):
        low, high = b / bins, (b + 1) / bins
        members = [
            (p, y) for p, y in zip(probabilities, labels)
            if low <= p < high or (b == bins - 1 and p == 1.0)
        ]
        if members:
            mean_p = sum(p for p, _ in members) / len(members)
            frequency = sum(y for _, y in members) / len(members)
            error += len(members) / total * abs(frequency - mean_p)
    return error


def reliability(probabilities, labels, bins: int = 10) -> list[dict]:
    """Per-bin mean confidence, frequency, and count (for reliability diagrams)."""
    table = []
    for b in range(bins):
        low, high = b / bins, (b + 1) / bins
        members = [
            (p, y) for p, y in zip(probabilities, labels)
            if low <= p < high or (b == bins - 1 and p == 1.0)
        ]
        table.append({
            "low": low, "high": high, "count": len(members),
            "confidence": sum(p for p, _ in members) / len(members) if members else None,
            "frequency": sum(y for _, y in members) / len(members) if members else None,
        })
    return table


def atom_metrics(rows: Sequence[dict]) -> dict:
    """Accuracy, F1/precision/recall on true atoms, Brier, ECE, NLL, no-answer rate."""
    n = len(rows)
    if not n:
        return {"atoms": 0}
    probabilities = [r["probability"] for r in rows]
    labels = [bool(r["truth"]) for r in rows]
    predicted = [bool(r["state"]) for r in rows]
    tp = sum(p and y for p, y in zip(predicted, labels))
    fp = sum(p and not y for p, y in zip(predicted, labels))
    fn = sum(not p and y for p, y in zip(predicted, labels))
    tn = n - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else math.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else math.nan
    specificity = tn / (tn + fp) if tn + fp else math.nan
    nll = -sum(
        math.log(min(max(p if y else 1 - p, EPS), 1.0)) for p, y in zip(probabilities, labels)
    ) / n
    return {
        "atoms": n,
        "positives": sum(labels),
        "accuracy": (tp + tn) / n,
        "balanced_accuracy": (recall + specificity) / 2 if tp + fn and tn + fp else math.nan,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "brier": sum((p - y) ** 2 for p, y in zip(probabilities, labels)) / n,
        "ece": expected_calibration_error(probabilities, labels),
        "nll": nll,
        "no_answer_rate": sum(not r["has_answer"] for r in rows) / n,
        "answered_accuracy": _answered_accuracy(rows),
    }


def _answered_accuracy(rows: Sequence[dict]) -> float:
    answered = [r for r in rows if r["has_answer"]]
    if not answered:
        return math.nan
    return sum(bool(r["state"]) == bool(r["truth"]) for r in answered) / len(answered)


def scene_metrics(rows: Sequence[dict]) -> dict:
    """Share of scenes whose estimated atoms are all correct (exact match)."""
    by_scene: dict = defaultdict(list)
    for r in rows:
        by_scene[r["scene"]].append(bool(r["state"]) == bool(r["truth"]))
    if not by_scene:
        return {"scenes": 0}
    errors = [len(v) - sum(v) for v in by_scene.values()]
    return {
        "scenes": len(by_scene),
        "exact_match": sum(e == 0 for e in errors) / len(errors),
        "mean_errors_per_scene": sum(errors) / len(errors),
    }


def summarize(values: Sequence[float]) -> dict:
    """Mean, standard deviation (n-1), min, max, and n of finite values."""
    finite = [v for v in values if v is not None and not math.isnan(v)]
    if not finite:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan, "n": 0}
    mean = sum(finite) / len(finite)
    std = (
        math.sqrt(sum((v - mean) ** 2 for v in finite) / (len(finite) - 1))
        if len(finite) > 1 else 0.0
    )
    return {"mean": mean, "std": std, "min": min(finite), "max": max(finite), "n": len(finite)}


def across(groups: dict, metric: Callable[[list], dict]) -> dict:
    """Apply ``metric`` to each group (e.g. one per seed) and summarize each
    resulting number across groups, as ``{name: summarize([...])}``."""
    per_group = {key: metric(rows) for key, rows in groups.items()}
    names = {k for m in per_group.values() for k, v in m.items() if isinstance(v, (int, float))}
    return {name: summarize([m.get(name, math.nan) for m in per_group.values()])
            for name in sorted(names)}


def bootstrap_ci(rows: Sequence, statistic: Callable[[Sequence], float], *,
                 cluster: "Callable[[object], object] | None" = None,
                 resamples: int = 1000, level: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap interval, resampling clusters (e.g. scenes) when given."""
    rng = random.Random(seed)
    if cluster is not None:
        groups: dict = defaultdict(list)
        for r in rows:
            groups[cluster(r)].append(r)
        units = list(groups.values())
    else:
        units = [[r] for r in rows]
    estimates = []
    for _ in range(resamples):
        sample = [r for unit in rng.choices(units, k=len(units)) for r in unit]
        estimates.append(statistic(sample))
    estimates.sort()
    low = estimates[int((1 - level) / 2 * resamples)]
    high = estimates[min(resamples - 1, int((1 + level) / 2 * resamples))]
    return low, high


def average_precision(rows: Sequence[dict], by: "Callable[[dict], object] | None" = None) -> float:
    """Average precision of P(true) for true atoms; with ``by``, the mean AP
    over groups (e.g. grounded or lifted predicates) that have both classes."""
    from sklearn.metrics import average_precision_score

    if by is None:
        labels = [bool(r["truth"]) for r in rows]
        if len(set(labels)) < 2:
            return math.nan
        return float(average_precision_score(labels, [r["probability"] for r in rows]))
    groups: dict = defaultdict(list)
    for r in rows:
        groups[by(r)].append(r)
    scores = [average_precision(g) for g in groups.values()]
    scores = [s for s in scores if not math.isnan(s)]
    return sum(scores) / len(scores) if scores else math.nan


def workshop_metrics(rows: Sequence[dict]) -> dict:
    """The 2025 workshop paper's metrics: accuracy at three thresholds and AP."""
    def accuracy(theta):
        return sum((r["probability"] >= theta) == bool(r["truth"]) for r in rows) / len(rows)

    return {
        "acc@0.3": accuracy(0.3), "acc@0.5": accuracy(0.5), "acc@0.7": accuracy(0.7),
        "AP_micro": average_precision(rows),
        "AP_macro": average_precision(rows, by=lambda r: r["atom"]),
        "AP_macro_lifted": average_precision(rows, by=lambda r: r["lifted"]),
    }
