# SPDX-License-Identifier: MIT
"""Rendered Blocksworld with exact ground truth.

Random arrangements of colored blocks are drawn on a table, with a robot
gripper above them that is empty or holds one block. The drawing style of
the front view follows ``examples/blocksworld_benchmark.py`` in the S3E
repository (60x45 px blocks, 90 px per tower slot); the generator adds the
gripper, goals, a full four-operator planning domain, up to nine blocks,
and four camera views:

* ``front``: the reference style;
* ``back``: the scene seen from behind (mirrored), on a different backdrop;
* ``wide``: larger blocks, wider spacing, a darker table;
* ``degraded``: the front view with sensor noise, blur, and lower exposure.

Every grounded atom's truth value is known exactly, any state can be
rendered (so closed-loop execution can re-observe after each action), and
the set of physically consistent states can be enumerated, which gives an
exact most-probable-consistent-state decoder for S3E's probabilities.
"""

import random
import zlib
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from ..scenes import Scene

NAME = "blocksworld"

COLORS = ("red", "green", "blue", "yellow", "purple", "orange", "pink", "brown", "cyan")
MAX_BLOCKS = len(COLORS)

DOMAIN_PDDL = """(define (domain blocksworld)
  (:requirements :strips :typing)
  (:types block)
  (:predicates (on ?x - block ?y - block) (ontable ?x - block) (clear ?x - block)
               (holding ?x - block) (handempty))
  (:action pick-up
    :parameters (?x - block)
    :precondition (and (clear ?x) (ontable ?x) (handempty))
    :effect (and (not (ontable ?x)) (not (clear ?x)) (not (handempty)) (holding ?x)))
  (:action put-down
    :parameters (?x - block)
    :precondition (holding ?x)
    :effect (and (not (holding ?x)) (clear ?x) (handempty) (ontable ?x)))
  (:action stack
    :parameters (?x - block ?y - block)
    :precondition (and (holding ?x) (clear ?y))
    :effect (and (not (holding ?x)) (not (clear ?y)) (clear ?x) (handempty) (on ?x ?y)))
  (:action unstack
    :parameters (?x - block ?y - block)
    :precondition (and (on ?x ?y) (clear ?x) (handempty))
    :effect (and (holding ?x) (clear ?y) (not (clear ?x)) (not (handempty)) (not (on ?x ?y)))))
"""

TEMPLATES = {
    "on": "Is the {0} block directly on top of the {1} block?",
    "ontable": "Is the {0} block standing directly on the table?",
    "clear": "Is the top of the {0} block clear, with no block on it?",
    "holding": "Is the robot gripper holding the {0} block?",
    "handempty": "Is the robot gripper empty, holding no block?",
}
PROMPT_TEMPLATE = "{query} Answer with exactly one word: yes or no."


@dataclass(frozen=True)
class BlocksState:
    """Towers listed left to right as displayed, each bottom to top."""

    towers: tuple[tuple[str, ...], ...]
    held: str | None = None

    def key(self) -> tuple:
        """Display-independent identity of the logical state."""
        return (tuple(sorted(self.towers)), self.held)

    def atoms(self, blocks: list[str]) -> dict[str, bool]:
        """Truth value of every grounded atom, in s3e's predicate format."""
        state = {f"on({x},{y})": False for x in blocks for y in blocks}
        for name in ("ontable", "clear", "holding"):
            state.update({f"{name}({x})": False for x in blocks})
        state["handempty()"] = self.held is None
        if self.held is not None:
            state[f"holding({self.held})"] = True
        for tower in self.towers:
            state[f"ontable({tower[0]})"] = True
            state[f"clear({tower[-1]})"] = True
            for below, above in zip(tower, tower[1:]):
                state[f"on({above},{below})"] = True
        return state

    def apply(self, action: str, args: list[str]) -> "BlocksState | None":
        """The successor under the PDDL semantics, or None if inapplicable."""
        towers = [list(t) for t in self.towers]
        tops = {t[-1]: i for i, t in enumerate(towers)}
        if action == "pick-up":
            (x,) = args
            if self.held is None and x in tops and len(towers[tops[x]]) == 1:
                del towers[tops[x]]
                return BlocksState(tuple(map(tuple, towers)), x)
        elif action == "put-down":
            (x,) = args
            if self.held == x:
                return BlocksState(tuple(map(tuple, towers)) + ((x,),), None)
        elif action == "stack":
            x, y = args
            if self.held == x and y in tops:
                towers[tops[y]].append(x)
                return BlocksState(tuple(map(tuple, towers)), None)
        elif action == "unstack":
            x, y = args
            if self.held is None and x in tops:
                tower = towers[tops[x]]
                if len(tower) >= 2 and tower[-2] == y:
                    tower.pop()
                    return BlocksState(tuple(map(tuple, towers)), x)
        return None


def random_towers(blocks: list[str], rng: random.Random) -> tuple[tuple[str, ...], ...]:
    """Stack the blocks into random towers (as in the S3E example generator)."""
    order = list(blocks)
    rng.shuffle(order)
    towers, tower = [], []
    for block in order:
        tower.append(block)
        if rng.random() < 0.5:
            towers.append(tuple(tower))
            tower = []
    if tower:
        towers.append(tuple(tower))
    return tuple(towers)


def goal_atoms(goal_towers) -> list[str]:
    """Atoms that fully specify a goal arrangement (on and ontable)."""
    atoms = []
    for tower in goal_towers:
        atoms.append(f"ontable({tower[0]})")
        atoms.extend(f"on({above},{below})" for below, above in zip(tower, tower[1:]))
    return atoms


def _pddl_atom(atom: str) -> str:
    name, _, rest = atom.partition("(")
    args = [a for a in rest.rstrip(")").split(",") if a]
    return f"({' '.join([name, *args])})"


def problem_pddl(name: str, blocks: list[str], state: BlocksState, goal_towers) -> str:
    """A PDDL problem with ``state`` as its initial state."""
    init = [_pddl_atom(a) for a, value in state.atoms(blocks).items() if value]
    goal = [_pddl_atom(a) for a in goal_atoms(goal_towers)]
    return (
        f"(define (problem {name}) (:domain blocksworld)\n"
        f"  (:objects {' '.join(blocks)} - block)\n"
        f"  (:init {' '.join(init)})\n"
        f"  (:goal (and {' '.join(goal)})))\n"
    )


# --- rendering ---


@dataclass(frozen=True)
class ViewStyle:
    name: str
    block: tuple[int, int] = (60, 45)
    slot: int = 90
    margin: int = 30
    background: str = "white"
    table: str = "gray"
    outline: int = 3
    mirror: bool = False
    noise: float = 0.0
    blur: float = 0.0
    brightness: float = 1.0


VIEWS = (
    ViewStyle("front"),
    ViewStyle("back", mirror=True, background="#f2ede4", table="#787878"),
    ViewStyle("wide", block=(72, 52), slot=110, margin=50, table="#5c5c5c", outline=2),
    ViewStyle("degraded", noise=14.0, blur=1.2, brightness=0.8),
)
VIEW_NAMES = tuple(style.name for style in VIEWS)

GRIPPER = "#333333"


def render(state: BlocksState, num_blocks: int, style: ViewStyle = VIEWS[0],
           noise_seed: int = 0) -> Image.Image:
    """Draw ``state``; the canvas size depends only on ``num_blocks``."""
    bw, bh = style.block
    width = style.margin + style.slot * num_blocks + 10
    gripper_zone = bh + 40
    height = bh * num_blocks + 90 + gripper_zone
    image = Image.new("RGB", (width, height), style.background)
    draw = ImageDraw.Draw(image)
    table_top = height - 40
    draw.rectangle((0, table_top, width, height), fill=style.table)
    for column, tower in enumerate(state.towers):
        left = style.margin + style.slot * column
        for level, block in enumerate(tower):
            bottom = table_top - bh * level
            draw.rectangle((left, bottom - bh, left + bw, bottom), fill=block,
                           outline="black", width=style.outline)
    # The gripper hangs from the top center; a held block sits between its fingers.
    cx = width // 2
    draw.rectangle((cx - 4, 0, cx + 4, 12), fill=GRIPPER)
    draw.rectangle((cx - bw // 2 - 14, 12, cx + bw // 2 + 14, 20), fill=GRIPPER)
    for x in (cx - bw // 2 - 14, cx + bw // 2 + 6):
        draw.rectangle((x, 20, x + 8, 20 + bh + 10), fill=GRIPPER)
    if state.held is not None:
        draw.rectangle((cx - bw // 2, 26, cx + bw // 2, 26 + bh), fill=state.held,
                       outline="black", width=style.outline)
    if style.mirror:
        image = ImageOps.mirror(image)
    if style.brightness != 1.0:
        image = ImageEnhance.Brightness(image).enhance(style.brightness)
    if style.blur:
        image = image.filter(ImageFilter.GaussianBlur(style.blur))
    if style.noise:
        rng = np.random.default_rng(noise_seed)
        pixels = np.asarray(image, dtype=np.float32)
        pixels = pixels + rng.normal(0.0, style.noise, pixels.shape)
        image = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    return image


def render_views(state: BlocksState, num_blocks: int, noise_seed: int = 0) -> list[Image.Image]:
    return [render(state, num_blocks, style, noise_seed) for style in VIEWS]


# --- instance generation ---


def scene_from_state(scene_id: str, blocks: list[str], state: BlocksState, goal_towers,
                     meta: dict, noise_seed: int = 0) -> Scene:
    return Scene(
        id=scene_id,
        domain=NAME,
        domain_pddl=DOMAIN_PDDL,
        problem_pddl=problem_pddl(scene_id, blocks, state, goal_towers),
        state=state.atoms(blocks),
        meta={
            **meta,
            "blocks": blocks,
            "towers": [list(t) for t in state.towers],
            "held": state.held,
            "goal_towers": [list(t) for t in goal_towers],
            "goal": goal_atoms(goal_towers),
            "view_names": list(VIEW_NAMES),
        },
        views=render_views(state, len(blocks), noise_seed),
    )


def generate(size: int, count: int, seed: int) -> list[Scene]:
    """``count`` random initial scenes with ``size`` blocks and a random goal.

    Deterministic in (size, count, seed); the blocks are a random subset of
    the colors, so object names recur across sizes. The gripper is empty in
    initial scenes, as in the usual Blocksworld problems.
    """
    if not 2 <= size <= MAX_BLOCKS:
        raise ValueError(f"size must be between 2 and {MAX_BLOCKS}; got {size}")
    rng = random.Random(f"{NAME}:{size}:{seed}")
    scenes = []
    for index in range(count):
        chosen = set(rng.sample(COLORS, size))
        blocks = [c for c in COLORS if c in chosen]
        state = BlocksState(random_towers(blocks, rng))
        goal = random_towers(blocks, rng)
        while BlocksState(goal).key() == state.key():
            goal = random_towers(blocks, rng)
        scene_id = f"bw-n{size}-s{seed}-{index:03d}"
        scenes.append(scene_from_state(
            scene_id, blocks, state, goal,
            {"size": size, "seed": seed, "index": index},
            noise_seed=rng.randrange(2**31),
        ))
    return scenes


def state_from_meta(meta: dict) -> BlocksState:
    return BlocksState(tuple(tuple(t) for t in meta["towers"]), meta.get("held"))


# --- domain knowledge used by masking and decoding ---


def impossible(atom: str) -> bool:
    """Atoms false in every state: a block is never on itself."""
    if atom.startswith("on("):
        x, y = atom[3:-1].split(",")
        return x == y
    return False


@lru_cache(maxsize=64)
def consistent_states(blocks: tuple[str, ...]) -> tuple[BlocksState, ...]:
    """Every physically consistent state over ``blocks``.

    Built by inserting the blocks one at a time into any position of any
    tower or as a new tower, which yields each arrangement exactly once
    (sum of Lah numbers: 37,633 arrangements of 7 blocks), for an empty
    hand and for each held block.
    """
    def arrangements(items):
        if not items:
            return [()]
        result = []
        last = items[-1]
        for towers in arrangements(items[:-1]):
            result.append(towers + ((last,),))
            for i, tower in enumerate(towers):
                for position in range(len(tower) + 1):
                    grown = tower[:position] + (last,) + tower[position:]
                    result.append(towers[:i] + (grown,) + towers[i + 1:])
        return result

    states = [BlocksState(t) for t in arrangements(blocks)]
    for held in blocks:
        rest = tuple(b for b in blocks if b != held)
        states.extend(BlocksState(t, held) for t in arrangements(rest))
    return tuple(states)


@lru_cache(maxsize=64)
def _atom_matrix(blocks: tuple[str, ...], atoms: tuple[str, ...]) -> np.ndarray:
    states = consistent_states(blocks)
    matrix = np.zeros((len(states), len(atoms)), dtype=np.int8)
    for row, state in enumerate(states):
        values = state.atoms(list(blocks))
        matrix[row] = [values[a] for a in atoms]
    return matrix


def map_decode(probabilities: dict[str, float], blocks: list[str],
               eps: float = 1e-6) -> dict[str, bool]:
    """The most probable consistent state under independent per-atom P(true).

    Maximizes sum_a log p_a if a is true else log(1 - p_a) over the atoms in
    ``probabilities`` (masked atoms do not vote), among all consistent
    states. Exhaustive, so only for up to 8 blocks.
    """
    if len(blocks) > 8:
        raise ValueError("map_decode enumerates states; use at most 8 blocks")
    atoms = tuple(probabilities)
    matrix = _atom_matrix(tuple(blocks), atoms)
    p = np.clip(np.array([probabilities[a] for a in atoms]), eps, 1 - eps)
    scores = matrix @ np.log(p) + (1 - matrix) @ np.log(1 - p)
    best = consistent_states(tuple(blocks))[int(np.argmax(scores))]
    return best.atoms(list(blocks))


class BlocksWorld:
    """A simulated tabletop for closed-loop runs: apply actions, re-render."""

    def __init__(self, scene: Scene):
        self.scene = scene
        self.blocks = list(scene.meta["blocks"])
        self.goal = BlocksState(tuple(tuple(t) for t in scene.meta["goal_towers"]))
        self.current = state_from_meta(scene.meta)
        self.steps = 0

    def observe(self, views: "list[int] | None" = None) -> list[Image.Image]:
        styles = VIEWS if views is None else [VIEWS[i] for i in views]
        seed = zlib.crc32(f"{self.scene.id}:{self.steps}".encode())
        return [render(self.current, len(self.blocks), s, seed) for s in styles]

    def state(self) -> dict[str, bool]:
        return self.current.atoms(self.blocks)

    def apply(self, action: str, args: list[str]) -> bool:
        """Execute an action; an inapplicable one fails and changes nothing."""
        self.steps += 1
        successor = self.current.apply(action, args)
        if successor is None:
            return False
        self.current = successor
        return True

    def goal_reached(self) -> bool:
        values = self.state()
        return all(values[a] for a in goal_atoms(self.goal.towers))

    def snapshot(self) -> Scene:
        """The current state as a scene (same goal), e.g. for logging."""
        return scene_from_state(
            f"{self.scene.id}-t{self.steps}", self.blocks, self.current, self.goal.towers,
            {**{k: self.scene.meta[k] for k in ("size", "seed", "index") if k in self.scene.meta}},
        )

World = BlocksWorld
