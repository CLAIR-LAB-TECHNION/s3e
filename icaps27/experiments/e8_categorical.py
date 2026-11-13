# SPDX-License-Identifier: MIT
"""E8: categorical answers on the benchmark (Blocksworld as finite-domain variables).

Each block is one variable ``pos(b)`` with the values ``table``, ``robot``
(held by the gripper) and every other block; ``clear`` and ``handempty`` are
derived. One categorical query per block replaces the yes/no query per atom,
and an argmax per variable cannot put a block in two places, which is part of
what E1's MAP decoder supplies. It can still decode two blocks on one block,
two held blocks, or a cycle, so the ``plan`` stage reports how often the
decoded state is consistent. Nothing is repaired.

``predict`` stores each query's raw option masses; ``plan`` decodes them,
plans as E1 does, and writes E1's record format, so ``e1_closed_loop.py
summary`` aggregates the output next to Table 2's rows.

Formats (``--format``):

* ``word``: the answer is the option's word, for instruction models;
* ``choice``: the options are lettered and the answer is a letter, which is
  the protocol of Vev decision models.

The option is called ``robot``, not ``gripper``, because ``gripper`` has no
single-token form on Qwen tokenizers, so logprob scoring cannot read it.

Examples::

    python experiments/e8_categorical.py predict --instances instances/blocksworld \\
        --model qwen3-vl-8b --out results/raw/e8
    PYTHONPATH=envs/s3e-2a09b7e python experiments/e8_categorical.py predict \\
        --instances instances/blocksworld --model vev-4b --format choice \\
        --vlm-kwargs '{"thinking": false}' --system-prompt "$VEV_SYSTEM_PROMPT" --out results/raw/e8
    python experiments/e8_categorical.py plan --instances instances/blocksworld \\
        --predictions results/raw/e8/blocksworld/qwen3-vl-8b/word --out results/raw/e8/plan-word.jsonl
    python experiments/e1_closed_loop.py summary results/raw/e8/plan-word.jsonl
"""

import argparse
import string
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s3e import AnswerOption, CategoricalAnswers, QueryEngine  # noqa: E402

from s3e_exp import cli, domains  # noqa: E402
from s3e_exp.estimation import EstimationConfig, atoms_for, run_provenance  # noqa: E402
from s3e_exp.models import TimedBackend, build_backend, model_spec  # noqa: E402
from s3e_exp.planning import plan_and_validate  # noqa: E402
from s3e_exp.store import JsonlStore, write_json  # noqa: E402

TABLE = "table"
HELD = "robot"
QUESTION = "What is the {0} block directly on?"
# S3E's default system prompts ask for true/false, so a categorical query needs its own.
SYSTEM_PROMPT = "Answer the question about the image with one of the given options and nothing else."
FORMATS = ("word", "choice")


def values_of(block: str, blocks: list[str]) -> list[str]:
    """The values of ``pos(block)``, in the order the prompt lists them."""
    return [TABLE, HELD] + [b for b in blocks if b != block]


def _describe(value: str) -> str:
    return f"{HELD} (held by the robot gripper)" if value == HELD else value


def word_query(block: str, values: list[str]) -> str:
    listed = [_describe(v) for v in values]
    return (f"{QUESTION.format(block)} Answer with exactly one word: "
            f"{', '.join(listed[:-1])}, or {listed[-1]}.")


def choice_query(block: str, values: list[str]) -> str:
    """The Vev choice layout (spec section 10), which S3E wraps with the image first."""
    lines = [f"{letter}. {_describe(v)}" for letter, v in zip(string.ascii_uppercase, values)]
    return (f"State:\n(image above)\n\nQuestion: {QUESTION.format(block)}\n\nOptions:\n"
            + "\n".join(lines) + "\n\nAnswer with the letter of the single best option.")


def build_query(block: str, blocks: list[str], fmt: str) -> tuple[str, CategoricalAnswers, dict]:
    """The prompt, answer space, and option-to-value map of one block's variable."""
    values = values_of(block, blocks)
    if fmt == "word":
        return word_query(block, values), CategoricalAnswers(values), {v: v for v in values}
    if fmt == "choice":
        letters = string.ascii_uppercase[:len(values)]
        # Letters only in their exact form: Vev reads "A", and "a" is an English word.
        answers = CategoricalAnswers([AnswerOption(c, (c,)) for c in letters])
        return choice_query(block, values), answers, dict(zip(letters, values))
    raise ValueError(f"Unknown format {fmt!r}; expected one of {FORMATS}")


def decode_positions(positions: dict[str, str], blocks: list[str]) -> dict[str, bool]:
    """Every grounded atom's value under ``positions``, in S3E's predicate format.

    Works for inconsistent assignments too, which is why it does not go
    through ``BlocksState``: ``clear(b)`` holds when nothing is on ``b`` and
    ``b`` is not held, and ``handempty`` when no block is held.
    """
    state = {f"on({x},{y})": positions[x] == y for x in blocks for y in blocks}
    for x in blocks:
        state[f"ontable({x})"] = positions[x] == TABLE
        state[f"holding({x})"] = positions[x] == HELD
        state[f"clear({x})"] = positions[x] != HELD and not any(
            positions[y] == x for y in blocks)
    state["handempty()"] = not any(p == HELD for p in positions.values())
    return state


def consistent(positions: dict[str, str]) -> bool:
    """One block held at most, one block on each block, and every tower on the table."""
    below = [p for p in positions.values() if p != TABLE]
    if len(below) != len(set(below)):  # also catches two held blocks
        return False
    for block in positions:
        seen, current = {block}, positions[block]
        while current not in (TABLE, HELD):
            if current in seen:
                return False
            seen.add(current)
            current = positions[current]
    return True


def predict_main(args) -> None:
    scenes = cli.select_scenes(args)
    spec = model_spec(args.model)
    backend = TimedBackend(build_backend(spec, args.backend, **args.vlm_kwargs))
    system_prompt = args.system_prompt if args.system_prompt is not None else SYSTEM_PROMPT
    out = cli.run_dir(args.out, scenes[0].domain, spec["name"], args.format)
    store = JsonlStore(out / "predictions.jsonl")
    done = store.keys("scene")
    engine = QueryEngine(backend, scoring="logprobs", system_prompt=system_prompt or None,
                         batch_size=1)
    if not (out / "run.json").exists():
        config = EstimationConfig(scoring="logprobs", mask="none", views=(args.view,),
                                  batch_size=1, system_prompt=system_prompt)
        write_json(out / "run.json", run_provenance(spec, backend, None, config, {
            "experiment": "e8", "format": args.format, "question": QUESTION,
            "instances": str(args.instances), "vlm_kwargs": args.vlm_kwargs}))
    for count, scene in enumerate(scenes):
        if (scene.id,) in done:
            continue
        blocks = scene.meta["blocks"]
        images = scene.images(args.view)
        backend.reset()
        start = time.perf_counter()
        answers = {block: ask(engine, images, block, blocks, args.format) for block in blocks}
        seconds = time.perf_counter() - start
        store.append({"scene": scene.id, "view": args.view, "size": scene.size,
                      "seed": scene.meta.get("seed"), "format": args.format,
                      "seconds": seconds, "inference_seconds": backend.seconds,
                      "queries": len(blocks), "answers": answers})
        if count % 25 == 0 or count == len(scenes) - 1:
            print(f"[{args.format}] {count + 1}/{len(scenes)} {scene.id}: "
                  f"{len(blocks)} queries in {seconds:.2f}s", flush=True)
    print("done")


def ask(engine, images, block: str, blocks: list[str], fmt: str) -> dict:
    """One variable's raw masses, keyed by value (letters are mapped back)."""
    query, answers, value_of = build_query(block, blocks, fmt)
    prediction = engine.ask(images, [query], answers=answers)[query]
    return {"masses": {value_of[k]: v for k, v in prediction.masses.items()},
            "unassigned_mass": prediction.unassigned_mass,
            "argmax_in_interest": prediction.argmax_in_interest}


def _plan_job(job):
    domain_pddl, problem_pddl, state, planner, timeout, key = job
    return key, plan_and_validate(domain_pddl, problem_pddl, state, planner, timeout)


def plan_main(args) -> None:
    scenes = {s.id: s for s in cli.select_scenes(args)}
    domain = domains.get(cli.domain_of(list(scenes.values())))
    store = JsonlStore(args.out)
    done = store.keys("model", "scene", "view", "decoder")
    jobs, info = [], {}
    for path in args.predictions:
        model, fmt = Path(path).parent.name, Path(path).name
        for record in JsonlStore(Path(path) / "predictions.jsonl").read():
            scene = scenes.get(record["scene"])
            key = (model, record["scene"], record["view"], f"categorical-{fmt}")
            if scene is None or key in done:
                continue
            positions = {b: max(a["masses"], key=a["masses"].get)
                         for b, a in record["answers"].items()}
            atoms, fixed = atoms_for(scene, domain, "none")
            estimated = decode_positions(positions, scene.meta["blocks"])
            errors = sum(estimated[a] != scene.state[a] for a in atoms)
            info[key] = {"model": model, "domain": scene.domain, "scene": scene.id,
                         "view": record["view"], "decoder": key[3], "size": scene.size,
                         "seed": scene.meta.get("seed"), "atoms": len(atoms),
                         "atom_errors": errors, "exact": errors == 0,
                         "queries": record["queries"], "consistent": consistent(positions)}
            jobs.append((scene.domain_pddl, scene.problem_pddl, {**fixed, **estimated},
                         args.planner, args.timeout, key))
    print(f"planning {len(jobs)} decoded states with {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for key, outcome in pool.map(_plan_job, jobs, chunksize=4):
            store.append({**info[key], **outcome})
    print(f"wrote {args.out}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    predict = sub.add_parser("predict")
    cli.scene_args(predict)
    cli.model_args(predict)
    predict.add_argument("--format", choices=FORMATS, default="word")
    predict.add_argument("--view", default="0")
    predict.add_argument("--system-prompt", default=None,
                         help="override the default instruction ('' sends none)")
    predict.add_argument("--out", type=Path, default=Path("results/raw/e8"))
    predict.set_defaults(func=predict_main)
    plan = sub.add_parser("plan")
    cli.scene_args(plan)
    plan.add_argument("--predictions", type=Path, nargs="+", required=True,
                      help="run directories written by predict")
    plan.add_argument("--planner", default="fast-downward")
    plan.add_argument("--timeout", type=float, default=60.0)
    plan.add_argument("--workers", type=int, default=4)
    plan.add_argument("--out", type=Path, required=True)
    plan.set_defaults(func=plan_main)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
