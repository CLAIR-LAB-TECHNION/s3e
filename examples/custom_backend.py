# SPDX-FileCopyrightText: CLAIR Lab Technion
# SPDX-License-Identifier: MIT

"""Plug a custom model into s3e and run the whole pipeline, with no downloads.

``SimulatedVLM`` stands in for a vision-language model. Each scene image
carries its true Blocksworld state (in ``Image.info``), and the simulated
model answers each predicate correctly 80% of the time, but always with 95%
confidence. It is overconfident, which is what calibration corrects. The
script:

1. grounds a Blocksworld problem into predicates (``from_pddl``),
2. collects calibration data on labeled training scenes and fits a
   Platt-scaling calibrator,
3. compares accuracy and Brier score on held-out scenes before and after
   calibration, and
4. converts one estimated state into a Unified Planning state.

Run it with ``python examples/custom_backend.py`` after
``pip install "s3e[pddl,calibration]"``. To use a real model instead, pass a
model id (or a ``HuggingFaceVLM`` / ``OpenAIVLM`` / ``VLLMBackend``) as
``vlm=``; everything else stays the same.
"""

import random

from PIL import Image

from s3e import (
    CalibrationExample,
    CalibrationSet,
    PlattCalibrator,
    SemanticStateEstimator,
    VLMBackend,
    VLMOutput,
)

BLOCKS = ("a", "b", "c")

DOMAIN = """
(define (domain blocksworld)
  (:requirements :typing)
  (:types block)
  (:predicates (on ?x - block ?y - block) (ontable ?x - block) (clear ?x - block)))
"""

PROBLEM = """
(define (problem three-blocks)
  (:domain blocksworld)
  (:objects a b c - block)
  (:init (ontable a) (ontable b) (ontable c) (clear a) (clear b) (clear c))
  (:goal (and (on a b) (on b c))))
"""


def random_state(rng: random.Random) -> dict[str, bool]:
    """Stack the blocks into random towers; return every predicate's truth value."""
    order = list(BLOCKS)
    rng.shuffle(order)
    towers, tower = [], []
    for block in order:
        tower.append(block)
        if rng.random() < 0.5:
            towers.append(tower)
            tower = []
    if tower:
        towers.append(tower)

    state = {f"on({x},{y})": False for x in BLOCKS for y in BLOCKS}
    state.update({f"{name}({x})": False for name in ("ontable", "clear") for x in BLOCKS})
    for tower in towers:  # tower[0] stands on the table; tower[-1] is clear
        state[f"ontable({tower[0]})"] = True
        state[f"clear({tower[-1]})"] = True
        for below, above in zip(tower, tower[1:]):
            state[f"on({above},{below})"] = True
    return state


def make_scene(state: dict[str, bool], scene_id: int) -> list[Image.Image]:
    """A one-image scene whose image carries the true state for SimulatedVLM."""
    image = Image.new("RGB", (64, 64), "white")
    image.info["state"] = state
    image.info["scene_id"] = scene_id
    return [image]


class SimulatedVLM(VLMBackend):
    """A stand-in model that is right with probability ``accuracy`` and always
    ``confidence`` sure of its answer.

    With the default (identity) translator, each prompt is the grounded
    predicate itself, e.g. ``on(a,b)``, and the answer tokens are ``true`` and
    ``false``. Only :meth:`query` is required; :meth:`query_batch` falls back
    to calling it once per prompt.
    """

    def __init__(self, accuracy: float = 0.8, confidence: float = 0.95, seed: int = 0):
        self.accuracy = accuracy
        self.confidence = confidence
        self.seed = seed

    def query(
        self,
        images,
        prompt,
        system_prompt=None,
        generate=False,
        interest_tokens=None,
        **inference_kwargs,
    ):
        scene = images[0].info
        rng = random.Random(f"{self.seed}:{scene['scene_id']}:{prompt}")
        truth = scene["state"][prompt]
        says_true = truth if rng.random() < self.accuracy else not truth
        if generate:
            return VLMOutput(token_probs=None, text="true" if says_true else "false")

        p_true = self.confidence if says_true else 1 - self.confidence
        probs = {"true": p_true, "false": 1 - p_true}
        if interest_tokens is None:
            return VLMOutput(token_probs=probs)
        # The interest-token contract: report exactly the requested tokens.
        return VLMOutput(
            token_probs={token: probs.get(token, 0.0) for token in interest_tokens},
            argmax_in_interest=True,
        )


def score(results, state: dict[str, bool]) -> tuple[int, float]:
    """Correct thresholded predicates and summed squared error of P(true)."""
    predicted = results.to_state()
    correct = sum(predicted[p] == state[p] for p in state)
    squared_error = sum((results[p].probability - state[p]) ** 2 for p in state)
    return correct, squared_error


def main() -> None:
    rng = random.Random(0)
    states = [random_state(rng) for _ in range(60)]
    scenes = [make_scene(state, i) for i, state in enumerate(states)]
    train, test = range(0, 40), range(40, 60)

    estimator = SemanticStateEstimator.from_pddl(DOMAIN, PROBLEM, vlm=SimulatedVLM())
    print(f"{len(estimator.predicates)} grounded predicates, e.g. {estimator.predicates[:3]}")

    # With a real model this is the expensive step: one query per predicate
    # per labeled scene. Fitting and applying the calibrator never query it.
    data = CalibrationSet.collect(
        estimator,
        [CalibrationExample(images=scenes[i], state_dict=states[i]) for i in train],
    )
    calibrator = PlattCalibrator.fit(data, scope="global")
    print(f"fitted Platt scaling on {len(data.samples)} samples")

    print(f"\nheld-out scenes: {len(test)} x {len(estimator.predicates)} predicates")
    print(f"{'':14}{'accuracy':>10}{'Brier':>10}")
    for name, cal in (("uncalibrated", None), ("calibrated", calibrator)):
        correct = squared_error = 0.0
        for i in test:
            results = estimator.estimate(scenes[i], calibrator=cal)
            scene_correct, scene_error = score(results, states[i])
            correct += scene_correct
            squared_error += scene_error
        n = len(test) * len(estimator.predicates)
        print(f"{name:14}{correct / n:>10.3f}{squared_error / n:>10.3f}")

    state = estimator(scenes[test[0]])
    up_state = estimator.to_up_state(state)
    print(f"\nestimated state of scene {test[0]}: {sorted(p for p in state if state[p])}")
    print(f"as a Unified Planning state: {type(up_state).__name__}")


if __name__ == "__main__":
    main()
