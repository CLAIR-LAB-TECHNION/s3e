# SPDX-License-Identifier: MIT
"""ViPlan++ environments as S3E scenes: GW, PRB, and ALF.

* ``viplan-gw``: GroceriesWorld (MuJoCo): six grocery items on three tables
  and a robot arm; 25 grounded atoms. The same setting as experiment 1 of
  the 2025 S3E workshop paper. Six cameras; views 0-3 are ``frontview``,
  ``leftangleview``, ``rightangleview``, and ``birdview``.
* ``viplan-prb``: photorealistic Blocksworld (Blender): 3+ shapes (cubes,
  cylinders, spheres) with colors, sizes, and materials; one camera.
* ``viplan-alf``: ALFWorld household tasks (AI2-THOR) with labeled object
  outlines; 1,000-2,600 grounded atoms, mostly static type facts.

Scenes come from two sources:

``export(source)``: recorded ViPlan++ episodes on disk (``episode.json``,
``domain.pddl``, ``problem.pddl``, ``observations/*.npz``), one scene per
frame. Needs only numpy; ``viplan`` is not imported.

``generate(size, count, seed)`` and ``World(scene)``: the live environments,
which need ``viplan`` importable (ViPlan++'s own environment, with this
package and S3E 0.5.0 on ``PYTHONPATH``; see the README). ``size`` is the
number of shapes for PRB, the goal horizon for GW, and ignored for ALF
(tasks are taken in order from ``valid_seen``). Noise injection is off
and GW uses its deterministic closed-form executor.

Question templates and the per-domain instructions are ViPlan++'s
(``prompts/viplan/``, copied from ViPlan++ commit 06b3a8a).
"""

import json
import os
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image

from ..scenes import Scene

PROMPTS = Path(__file__).resolve().parents[2] / "prompts" / "viplan"
ENVS = {"viplan-gw": "gw", "viplan-prb": "prb", "viplan-alf": "alf"}
GW_CAMERAS = ("frontview", "leftangleview", "rightangleview", "birdview")
PROMPT_TEMPLATE = "{query} Answer with exactly one word: yes or no."


def _pddl_atom(atom: str) -> str:
    name, _, rest = atom.partition("(")
    args = [a for a in rest.rstrip(")").split(",") if a]
    return f"({' '.join([name, *args])})"


def replace_init(problem_pddl: str, state: dict[str, bool]) -> str:
    """``problem_pddl`` with its ``:init`` section set to the true atoms of ``state``."""
    match = re.search(r"\(\s*:init", problem_pddl, re.IGNORECASE)
    if match is None:
        raise ValueError("problem has no :init section")
    depth, end = 0, None
    for index in range(match.start(), len(problem_pddl)):
        char = problem_pddl[index]
        depth += char == "("
        depth -= char == ")"
        if depth == 0:
            end = index + 1
            break
    if end is None:
        raise ValueError("unbalanced :init section")
    init = " ".join(_pddl_atom(a) for a, value in state.items() if value)
    return f"{problem_pddl[:match.start()]}(:init {init}){problem_pddl[end:]}"


def _objects(problem_pddl: str) -> int:
    """Number of objects declared in a problem's ``:objects`` section."""
    match = re.search(r"\(\s*:objects(.*?)\)", problem_pddl, re.IGNORECASE | re.DOTALL)
    if not match:
        return 0
    tokens, count, skip = match.group(1).split(), 0, False
    for token in tokens:
        if skip:
            skip = False
        elif token == "-":
            skip = True
        else:
            count += 1
    return count


def _image(array) -> Image.Image:
    return Image.fromarray(np.asarray(array, dtype=np.uint8)).convert("RGB")


class ViPlanAdapter:
    """Adapter for one ViPlan++ environment (see the module docstring)."""

    PROMPT_TEMPLATE = PROMPT_TEMPLATE

    def __init__(self, name: str):
        self.NAME = name
        self.env_name = ENVS[name]
        self.TEMPLATES = json.loads(
            (PROMPTS / f"{self.env_name}_template.json").read_text(encoding="utf-8"))
        self.ADDITIONAL_INSTRUCTIONS = (PROMPTS / f"{self.env_name}.md").read_text(
            encoding="utf-8").strip()

    def impossible(self, atom: str) -> bool:
        """PRB: a shape is never on itself."""
        if self.env_name == "prb" and atom.startswith("shape-on-top-of-block("):
            x, y = atom[len("shape-on-top-of-block("):-1].split(",")
            return x == y
        return False

    # --- recorded episodes ---

    def export(self, source: Path, *, cells: "list[str] | None" = None,
               frames: str = "all", episodes: "int | None" = None, every: int = 1):
        """Scenes from recorded episodes under ``source`` (``<cell>/episodes/episode_*``).

        ``frames``: ``initial`` (each episode's first frame) or ``all``
        (every frame, keeping every ``every``-th step). The scene's
        ``seed`` is ``episode index mod 3``, three disjoint replicate groups
        for dispersion. Frames of failed episodes before the failure are kept.
        """
        source = Path(source)
        cell_dirs = sorted(p for p in source.iterdir() if (p / "episodes").is_dir())
        if cells:
            cell_dirs = [p for p in cell_dirs if p.name in cells]
        for cell_dir in cell_dirs:
            episode_dirs = sorted((cell_dir / "episodes").glob("episode_*"))[:episodes]
            for episode_dir in episode_dirs:
                record = json.loads((episode_dir / "episode.json").read_text(encoding="utf-8"))
                domain = (episode_dir / "domain.pddl").read_text(encoding="utf-8")
                problem = (episode_dir / "problem.pddl").read_text(encoding="utf-8")
                index = int(record.get("episode_index", episode_dir.name.split("_")[-1]))
                timeline = [(-1, None, record["initial_gt_state"], record["initial_observation"])]
                if frames == "all":
                    timeline += [(s["step_index"], s["action"], s["gt_state"], s["observation"])
                                 for s in record["steps"][::every] if s.get("observation")]
                for step, action, state, observation in timeline:
                    path = cell_dir / observation["path"]
                    if not path.exists():
                        continue
                    with np.load(path) as data:
                        view = _image(data["obs"])
                    yield Scene(
                        id=f"{self.env_name}-{cell_dir.name}-{index:04d}-t{step + 1:03d}",
                        domain=self.NAME, domain_pddl=domain,
                        problem_pddl=replace_init(problem, state), state=dict(state),
                        meta={"size": _objects(problem), "seed": index % 3, "episode": index,
                              "episode_seed": record.get("episode_seed"), "step": step + 1,
                              "cell": cell_dir.name, "after_action": action,
                              "source": "recorded"},
                        views=[view],
                    )

    # --- live environments ---

    def make_env(self, options: dict):
        from viplan.injection import InjectionConfig

        kwargs = dict(options.get("env_kwargs", {}))
        if self.env_name == "gw":
            from viplan.gw.env import GroceriesWorldEnv

            kwargs.setdefault("executor_mode", "closed_form")
            kwargs.setdefault("grasp_mode", "requested")
            return GroceriesWorldEnv(init_exposure="gt", injection=InjectionConfig(), **kwargs)
        if self.env_name == "prb":
            from viplan.prb.env import PRBEnv

            kwargs.setdefault("wiggle", False)
            low = options.get("num_objects", 3)
            return PRBEnv(num_objects_range=(low, low), init_exposure="gt",
                          injection=InjectionConfig(), **kwargs)
        from viplan.alf.env import AlfWorldEnv

        kwargs.setdefault("draw_bboxes", True)
        kwargs.setdefault("data_split", "valid_seen")
        return AlfWorldEnv(init_exposure="gt", injection=InjectionConfig(), **kwargs)

    def reset_options(self, size: int, options: dict) -> dict:
        if self.env_name == "gw":
            return {"task_horizon": size}
        if self.env_name == "prb":
            return {"num_objects": size, "task_horizon": options.get("task_horizon", 3)}
        return {}

    def render_views(self, env, views: "list[int] | None" = None) -> list[Image.Image]:
        if self.env_name != "gw":
            return [_image(env.render())]
        import mujoco

        renderer = getattr(env, "_s3e_exp_renderer", None)
        height, width = env.observation_space.shape[:2]
        if renderer is None:
            renderer = mujoco.Renderer(env._mj_model, height, width)
            env._s3e_exp_renderer = renderer
        images = []
        for index in views if views is not None else range(len(GW_CAMERAS)):
            renderer.update_scene(env._mj_data, camera=GW_CAMERAS[index])
            images.append(_image(renderer.render()))
        return images

    def _random_walk(self, env, info, steps: int, rng: random.Random):
        """Take ``steps`` random applicable actions (for non-initial states)."""
        from viplan.up_utils import state_dict_to_up_state

        taken = []
        for _ in range(steps):
            state = state_dict_to_up_state(env.problem, info["gt"]["state"])
            applicable = sorted(_applicable(env, state))
            if not applicable:
                break
            action = rng.choice(applicable)
            _, _, done, _, info = env.step(_action(env.problem, action))
            taken.append(action)
            if done:
                break
        return info, taken

    def generate(self, size: int, count: int, seed: int, **options) -> list[Scene]:
        env = self.make_env({**options, "num_objects": size})
        seeds = np.random.SeedSequence([seed, size]).generate_state(count) & 0x7FFFFFFF
        rng = random.Random(f"{self.NAME}:{size}:{seed}")
        scenes = []
        try:
            for index, episode_seed in enumerate(seeds):
                if self.env_name == "alf":
                    episode_seed = seed * count + index
                reset = self.reset_options(size, options)
                _, info = env.reset(seed=int(episode_seed), options=reset)
                walk = []
                if options.get("random_steps"):
                    info, walk = self._random_walk(env, info, rng.randint(0, options["random_steps"]),
                                                   rng)
                state = dict(info["gt"]["state"])
                domain, problem = env.get_problem_pddl("nominal")
                scenes.append(Scene(
                    id=f"{self.env_name}-n{size}-s{seed}-{index:03d}", domain=self.NAME,
                    domain_pddl=domain, problem_pddl=replace_init(problem, state), state=state,
                    meta={"size": size, "seed": seed, "index": index,
                          "episode_seed": int(episode_seed), "reset_options": reset, "walk": walk,
                          "env_options": options, "source": "live",
                          "viplan_commit": os.environ.get("VIPLAN_COMMIT")},
                    views=self.render_views(env, options.get("views")),
                ))
        finally:
            close = getattr(env, "close", None)
            if close:
                close()
        return scenes

    def World(self, scene: Scene) -> "ViPlanWorld":
        return ViPlanWorld(self, scene)


def _applicable(env, state) -> list[str]:
    """Applicable grounded actions in ``state``, as action strings."""
    return [f"{action.name}({','.join(str(p) for p in params)})"
            for action, params in env.problem_sim.get_applicable_actions(state)]


def _action(problem, action: str):
    from viplan.up_utils import action_instance_from_str

    name, _, rest = action.partition("(")
    args = [a.strip() for a in rest.rstrip(")").split(",") if a.strip()]
    return action_instance_from_str(problem, f"{name}({', '.join(args)})")


class ViPlanWorld:
    """A live ViPlan++ episode for closed-loop runs, reset to a scene's state."""

    def __init__(self, adapter: ViPlanAdapter, scene: Scene):
        if scene.meta.get("source") != "live":
            raise ValueError("closed loop needs scenes made by generate(), not exported frames")
        self.adapter = adapter
        options = dict(scene.meta.get("env_options", {}))
        self.env = adapter.make_env({**options, "num_objects": scene.size})
        _, info = self.env.reset(seed=scene.meta["episode_seed"],
                                 options=scene.meta["reset_options"])
        for action in scene.meta.get("walk", []):
            _, _, _, _, info = self.env.step(_action(self.env.problem, action))
        self._state = dict(info["gt"]["state"])
        if self._state != scene.state:
            raise RuntimeError(f"replaying {scene.id} did not reproduce its state")
        self._done = False

    def observe(self, views=None):
        return self.adapter.render_views(self.env, views)

    def state(self) -> dict[str, bool]:
        return dict(self._state)

    def apply(self, action: str, args: list[str]) -> bool:
        instance = _action(self.env.problem, f"{action}({','.join(args)})")
        _, _, done, _, info = self.env.step(instance)
        self._state = dict(info["gt"]["state"])
        self._done = bool(done)
        return bool(info.get("execution_success", False))

    def goal_reached(self) -> bool:
        from viplan.up_utils import state_dict_to_up_state

        return bool(self.env.problem_sim.is_goal(
            state_dict_to_up_state(self.env.problem, self._state)))


def adapter(name: str) -> ViPlanAdapter:
    return ViPlanAdapter(name)
