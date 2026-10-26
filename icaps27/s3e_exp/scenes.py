# SPDX-License-Identifier: MIT
"""Scenes: a PDDL problem, its true state, and one or more camera views.

A scene is what every experiment consumes. It is stored on disk as a
directory (``domain.pddl``, ``problem.pddl``, ``state.json``, ``meta.json``,
``view0.png``, ``view1.png``, ...), so instance sets can be generated once,
shipped with the artifact, and re-read without the environment that made
them.

Atoms are s3e's grounded-predicate strings, e.g. ``on(red,blue)`` or
``handempty()``; ``state`` gives the truth value of every grounded atom of
the problem, not only the true ones.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

VIEW_ALL = "all"


@dataclass
class Scene:
    """One estimation instance.

    Attributes:
        id: Unique id within its instance set (used as a directory name).
        domain: Registry name of the domain adapter that made it.
        domain_pddl: Planning domain (with actions).
        problem_pddl: Problem with the *true* initial state and the goal.
            Estimation only grounds its objects; planning experiments use it
            to validate plans against the truth.
        state: Truth value of every grounded atom in the initial state.
        meta: Generator details (size, seed, layout, goal, ...).
        views: Camera views, as PIL images or paths to image files.
    """

    id: str
    domain: str
    domain_pddl: str
    problem_pddl: str
    state: dict[str, bool]
    meta: dict = field(default_factory=dict)
    views: list = field(default_factory=list)

    def view(self, index: int) -> Image.Image:
        """The ``index``-th camera view as an RGB image (loaded lazily)."""
        view = self.views[index]
        if not isinstance(view, Image.Image):
            with Image.open(view) as opened:
                view = opened.convert("RGB")
            self.views[index] = view
        return view

    def images(self, view_spec: str) -> list[Image.Image]:
        """The images shown together for one estimate.

        ``view_spec`` is a view index as a string (``"0"``), ``"all"`` for
        every view in one prompt, or a comma-separated list (``"0,2"``).
        """
        if view_spec == VIEW_ALL:
            return [self.view(i) for i in range(len(self.views))]
        return [self.view(int(i)) for i in view_spec.split(",")]

    @property
    def num_views(self) -> int:
        return len(self.views)

    @property
    def size(self) -> int | None:
        """Number of objects the generator was asked for, if recorded."""
        return self.meta.get("size")


def save_scene(scene: Scene, root: Path) -> Path:
    """Write ``scene`` to ``root/<scene.id>/`` and return that directory."""
    directory = Path(root) / scene.id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "domain.pddl").write_text(scene.domain_pddl, encoding="utf-8")
    (directory / "problem.pddl").write_text(scene.problem_pddl, encoding="utf-8")
    (directory / "state.json").write_text(
        json.dumps(scene.state, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    (directory / "meta.json").write_text(
        json.dumps({"id": scene.id, "domain": scene.domain, **scene.meta}, indent=1) + "\n",
        encoding="utf-8",
    )
    for index in range(scene.num_views):
        scene.view(index).save(directory / f"view{index}.png")
    return directory


def load_scene(directory: Path) -> Scene:
    """Read a scene written by :func:`save_scene`; views load lazily."""
    directory = Path(directory)
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    views = sorted(
        directory.glob("view*.png"), key=lambda path: int(path.stem.removeprefix("view"))
    )
    return Scene(
        id=meta.pop("id"),
        domain=meta.pop("domain"),
        domain_pddl=(directory / "domain.pddl").read_text(encoding="utf-8"),
        problem_pddl=(directory / "problem.pddl").read_text(encoding="utf-8"),
        state=json.loads((directory / "state.json").read_text(encoding="utf-8")),
        meta=meta,
        views=list(views),
    )


def load_scenes(root: Path) -> list[Scene]:
    """Every scene under ``root`` (any depth), sorted by id."""
    directories = sorted(path.parent for path in Path(root).rglob("meta.json"))
    scenes = [load_scene(directory) for directory in directories]
    return sorted(scenes, key=lambda scene: scene.id)
