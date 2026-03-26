"""Shared test fixtures for s3e tests."""

import pytest
from PIL import Image

from s3e.vlm.backend import VLMBackend, VLMOutput


# Minimal blocksworld PDDL
BLOCKSWORLD_DOMAIN = """
(define (domain blocksworld)
  (:requirements :typing)
  (:types block)
  (:predicates
    (on ?x - block ?y - block)
    (clear ?x - block)
  )
  (:action move
    :parameters (?b - block ?from - block ?to - block)
    :precondition (and (on ?b ?from) (clear ?b) (clear ?to))
    :effect (and (on ?b ?to) (clear ?from) (not (on ?b ?from)) (not (clear ?to)))
  )
)
"""

BLOCKSWORLD_PROBLEM = """
(define (problem bw-2)
  (:domain blocksworld)
  (:objects a b - block)
  (:init (on a b) (clear a))
  (:goal (on b a))
)
"""


class FakeVLM(VLMBackend):
    """A fake VLM that returns configurable token probabilities."""

    def __init__(
        self,
        token_probs: dict[str, float] | None = None,
        text: str | None = None,
        per_prompt_probs: dict[str, dict[str, float]] | None = None,
    ):
        self.token_probs = token_probs or {"yes": 0.8, "no": 0.2}
        self.text = text
        self.per_prompt_probs = per_prompt_probs or {}
        self.call_count = 0
        self.received_prompts: list[str] = []
        self.received_system_prompts: list[str | None] = []

    def query(self, images, prompt, system_prompt=None):
        self.call_count += 1
        self.received_prompts.append(prompt)
        self.received_system_prompts.append(system_prompt)

        # Check for per-prompt overrides
        for substring, probs in self.per_prompt_probs.items():
            if substring in prompt:
                return VLMOutput(token_probs=dict(probs), text=self.text)

        return VLMOutput(token_probs=dict(self.token_probs), text=self.text)


@pytest.fixture
def fake_vlm():
    """A FakeVLM with default 80% yes / 20% no probabilities."""
    return FakeVLM()


@pytest.fixture
def fake_images():
    """Two small blank RGB images."""
    return [Image.new("RGB", (64, 64)), Image.new("RGB", (64, 64))]


@pytest.fixture
def single_image():
    """A single small blank RGB image."""
    return [Image.new("RGB", (64, 64))]


@pytest.fixture
def blocksworld_domain():
    return BLOCKSWORLD_DOMAIN


@pytest.fixture
def blocksworld_problem():
    return BLOCKSWORLD_PROBLEM
