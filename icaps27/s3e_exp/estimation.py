# SPDX-License-Identifier: MIT
"""Run S3E over scenes and store raw predictions (the expensive stage).

Every record holds the scene's :class:`s3e.PredictionSet` serialized with
its raw answer masses, so thresholds, calibration, view averaging, and
planning are all applied offline afterwards without querying the model
again. Records are appended one scene-view at a time; a rerun skips what is
already stored.
"""

import re
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

from s3e import (
    PredictionSet,
    QueryTranslator,
    SemanticStateEstimator,
    TemplateTranslator,
    UnmatchedAnswerWarning,
)

from . import provenance
from .masking import select_atoms
from .models import TimedBackend, default_generation_kwargs
from .provenance import model_revision
from .planning import parse
from .store import JsonlStore, read_json, write_json


@dataclass
class EstimationConfig:
    scoring: str = "logprobs"
    mask: str = "none"
    views: tuple[str, ...] = ("0",)
    batch_size: int = 16
    # None: no kwargs for logprobs; a reply cap of ``max_new_tokens`` for text.
    inference_kwargs: "dict | None" = None
    max_new_tokens: int = 8
    prompt_template: "str | None" = None
    system_prompt: "str | None" = None
    true_tokens: "list[str] | None" = None
    false_tokens: "list[str] | None" = None
    null_tokens: "list[str] | None" = None
    extra: dict = field(default_factory=dict)

    def tag(self) -> str:
        return f"{self.scoring}-{self.mask}"


class FallbackTranslator(QueryTranslator):
    """Templates where given; a generic question for any other predicate.

    S3E's ``TemplateTranslator`` raises for predicates without a template;
    domains whose static predicates have none (they are masked away anyway)
    get ``Is it true that <name>(<args>)?`` for them instead.
    """

    def __init__(self, templates: dict[str, str]):
        self.templates = templates
        self.inner = TemplateTranslator(templates)

    def translate(self, predicates, domain=None, problem=None):
        known = [p for p in predicates if p.split("(", 1)[0] in self.templates]
        result = self.inner.translate(known, domain, problem) if known else {}
        for p in predicates:
            if p not in result:
                name, _, rest = p.partition("(")
                args = ", ".join(a for a in rest.rstrip(")").split(",") if a)
                words = re.sub(r"[-_]", " ", name)
                result[p] = f"Is it true that {words}({args})?"
        return {p: result[p] for p in predicates}


def inference_kwargs(config: EstimationConfig, backend) -> dict:
    if config.inference_kwargs is not None:
        return dict(config.inference_kwargs)
    if config.scoring == "text_match":
        return default_generation_kwargs(backend, config.max_new_tokens)
    return {}


def build_estimator(scene, domain, backend, config: EstimationConfig) -> SemanticStateEstimator:
    kwargs = dict(
        vlm=backend,
        translator=FallbackTranslator(domain.TEMPLATES),
        prompt_template=config.prompt_template or domain.PROMPT_TEMPLATE,
        scoring=config.scoring,
        batch_size=config.batch_size,
        inference_kwargs=inference_kwargs(config, backend),
        true_tokens=config.true_tokens,
        false_tokens=config.false_tokens,
        null_tokens=config.null_tokens,
    )
    if config.system_prompt:
        kwargs["system_prompt"] = config.system_prompt
    if getattr(domain, "ADDITIONAL_INSTRUCTIONS", None) and config.system_prompt != "":
        kwargs["additional_instructions"] = domain.ADDITIONAL_INSTRUCTIONS
    estimator = SemanticStateEstimator.from_pddl(scene.domain_pddl, scene.problem_pddl, **kwargs)
    if config.system_prompt == "":
        # No system message at all (the estimator would otherwise pick its default).
        estimator.engine.system_prompt = None
    return estimator


def atoms_for(scene, domain, mask: str, problem=None) -> tuple[list[str], dict[str, bool]]:
    """Atoms to estimate and the fixed values of the rest, for ``mask``."""
    problem = problem or parse(scene.domain_pddl, scene.problem_pddl)
    return select_atoms(problem, mask, impossible=getattr(domain, "impossible", None))


def prompts_for(estimator, atoms) -> dict[str, str]:
    template = estimator.engine.prompt_template
    return {atom: template.format(query=estimator.queries[atom]) for atom in atoms}


def estimate(estimator, backend, scene, atoms, images, truth=None) -> tuple[PredictionSet, dict]:
    """One estimate with timing; ``truth`` primes fake backends only."""
    if truth is not None and hasattr(backend, "prepare"):
        prompts = prompts_for(estimator, atoms)
        backend.prepare({prompts[a]: bool(truth[a]) for a in atoms})
    while True:
        if isinstance(backend, TimedBackend):
            backend.reset()
        start = time.perf_counter()
        try:
            with warnings.catch_warnings():
                # Unanswered predictions are counted in the metrics (no-answer rate).
                warnings.simplefilter("ignore", UnmatchedAnswerWarning)
                results = estimator.estimate(images, predicates=atoms)
            break
        except RuntimeError as error:
            # Deliberate fallback: halve the batch on CUDA OOM (recorded below).
            if "out of memory" not in str(error).lower() or estimator.engine.batch_size == 1:
                raise
            _empty_cuda_cache()
            estimator.engine.batch_size = max(1, estimator.engine.batch_size // 2)
            print(f"CUDA out of memory: retrying with batch size {estimator.engine.batch_size}",
                  flush=True)
    seconds = time.perf_counter() - start
    timing = {
        "effective_batch_size": estimator.engine.batch_size,
        "seconds": seconds,
        "inference_seconds": getattr(backend, "seconds", None),
        "backend_calls": getattr(backend, "calls", None),
        "queries": len(set(estimator.queries[a] for a in atoms)),
    }
    return results, timing


def _empty_cuda_cache() -> None:
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass


def run_provenance(spec: dict, backend, estimator, config: EstimationConfig, extra=None) -> dict:
    return {
        **provenance.base(),
        "model": {**spec, "resolved_revision": model_revision(backend, spec),
                  "backend_class": type(getattr(backend, "inner", backend)).__name__},
        "config": asdict(config),
        "system_prompt": estimator.engine.system_prompt if estimator else None,
        "prompt_template": estimator.engine.prompt_template if estimator else None,
        "answer_space": estimator.engine.answers.to_dict() if estimator else None,
        "inference_kwargs": estimator.engine.inference_kwargs if estimator else None,
        **(extra or {}),
    }


def run_signature(spec: dict, backend, config: EstimationConfig, extra: "dict | None") -> dict:
    """Everything that changes predictions; a run directory holds one signature."""
    settings = asdict(config)
    settings.pop("views")
    return {"model": spec, "backend_class": type(getattr(backend, "inner", backend)).__name__,
            "config": settings, "vlm_kwargs": (extra or {}).get("vlm_kwargs", {})}


def collect(scenes, domain, spec: dict, backend, config: EstimationConfig, out_dir: Path,
            *, extra_provenance: "dict | None" = None, progress: bool = True) -> Path:
    """Estimate every scene under every view spec; append to ``predictions.jsonl``.

    ``out_dir/run.json`` records the provenance (rewritten at the end with
    the finish time and the queries seen).
    """
    out_dir = Path(out_dir)
    store = JsonlStore(out_dir / "predictions.jsonl")
    done = store.keys("scene", "view")
    backend = backend if isinstance(backend, TimedBackend) else TimedBackend(backend)
    pending = [(s, v) for s in scenes for v in config.views if (s.id, v) not in done]
    run_path = out_dir / "run.json"
    run = read_json(run_path) if run_path.exists() else None
    signature = run_signature(spec, backend, config, extra_provenance)
    if run is not None and run.get("signature", signature) != signature:
        raise SystemExit(
            f"{out_dir} holds predictions made with another configuration "
            f"({run['signature']}); pass --tag (or a new --out) for {signature}")
    if run is not None and pending:
        # A later invocation into the same run (e.g. more views or seeds).
        run["config"]["views"] = sorted(set(run["config"]["views"]) | set(config.views))
        run.setdefault("invocations", []).append({"argv": sys.argv, "utc": provenance.now()})
    estimator = None
    reduced: dict[str, int] = {}  # view spec -> batch size after an out-of-memory error
    queries: dict[str, str] = dict(run.get("queries", {})) if run else {}
    provenance.reset_peak_memory()
    for count, (scene, view) in enumerate(pending):
        if estimator is None or estimator.problem_pddl != scene.problem_pddl:
            estimator = build_estimator(scene, domain, backend, config)
            atoms, _ = atoms_for(scene, domain, config.mask, estimator.up_problem)
        if run is None:
            run = run_provenance(spec, backend, estimator, config, extra_provenance)
            run["signature"] = signature
            run["queried_utc"] = []
            write_json(run_path, run)
        estimator.engine.batch_size = reduced.get(view, config.batch_size)
        results, timing = estimate(estimator, backend, scene, atoms, scene.images(view),
                                   truth=scene.state)
        if timing["effective_batch_size"] < config.batch_size:
            reduced[view] = timing["effective_batch_size"]
        queries.update({a: estimator.queries[a] for a in atoms})
        store.append({
            "scene": scene.id, "view": view, "size": scene.size, "seed": scene.meta.get("seed"),
            "mask": config.mask, "scoring": config.scoring, **timing,
            "predictions": results.to_dict(),
        })
        if progress and (count % 25 == 0 or count == len(pending) - 1):
            print(f"[{config.tag()}] {count + 1}/{len(pending)} {scene.id} view {view}: "
                  f"{timing['queries']} queries in {timing['seconds']:.2f}s", flush=True)
    if run is not None:
        run["finished_utc"] = provenance.now()
        run["queried_utc"] = run.get("queried_utc", []) + ([provenance.now()] if pending else [])
        run["peak_memory"] = provenance.peak_memory()
        run["queries"] = queries
        write_json(run_path, run)
    return out_dir


def load_predictions(path: Path) -> list[dict]:
    """Prediction records with their PredictionSet rebuilt (key ``results``)."""
    path = Path(path)
    if path.is_dir():
        path = path / "predictions.jsonl"
    records = JsonlStore(path).read()
    for record in records:
        record["results"] = PredictionSet.from_dict(record["predictions"])
    return records
