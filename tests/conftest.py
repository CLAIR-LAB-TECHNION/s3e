"""Shared test fixtures for s3e tests.

The shared fake backend lives in ``tests/fakes.py`` (``FakeVLM``); this
module holds sample PDDL and image helpers only.
"""

from PIL import Image


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


def make_blank_image(size=(8, 8)):
    """Tiny RGB image for backend tests."""
    return Image.new("RGB", size, color=(127, 127, 127))
