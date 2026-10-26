# SPDX-License-Identifier: MIT
"""Run one S3E version on a job and report its API surface, timing, and P(true).

Imports only the S3E found on ``sys.path`` (an overlay directory on
``PYTHONPATH`` selects the version; see ``make overlay-s3e``), so the same
torch and transformers serve every version. Handles both API generations:
0.1-0.3 (``SemanticStateEstimator(domain, problem, ...)`` with
``estimate_prediction_details``) and 0.4+ (``from_pddl`` and
``estimate`` returning a ``PredictionSet``).

    PYTHONPATH=envs/s3e-0.2.0 python s3e_exp/workers/version_worker.py job.json out.json
"""

import inspect
import json
import sys
import time
from importlib import metadata
from pathlib import Path

from PIL import Image


def load_scene(directory: Path) -> dict:
    return {
        "id": json.loads((directory / "meta.json").read_text(encoding="utf-8"))["id"],
        "domain": (directory / "domain.pddl").read_text(encoding="utf-8"),
        "problem": (directory / "problem.pddl").read_text(encoding="utf-8"),
        "state": json.loads((directory / "state.json").read_text(encoding="utf-8")),
        "image": directory / "view0.png",
    }


def _params(obj, name="__init__") -> set:
    try:
        return set(inspect.signature(getattr(obj, name)).parameters)
    except (TypeError, ValueError, AttributeError):
        return set()


def surface(s3e) -> dict:
    """Public names, features present, and source size of the imported S3E."""
    package = Path(s3e.__file__).resolve().parent
    names = sorted(getattr(s3e, "__all__", [n for n in dir(s3e) if not n.startswith("_")]))
    estimator = s3e.SemanticStateEstimator
    try:
        hf = s3e.HuggingFaceVLM
        hf_params = _params(hf)
    except Exception:  # optional dependency missing
        hf_params = set()
    backend_query = _params(s3e.VLMBackend, "query")
    files = sorted(package.rglob("*.py"))
    try:
        dist = metadata.distribution("s3e")
        direct_url = dist.read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        direct_url = None
    return {
        "version": getattr(s3e, "__version__", None),
        "direct_url": json.loads(direct_url) if direct_url else None,
        "path": str(package),
        "public_names": names,
        "modules": len(files),
        "source_lines": sum(len(f.read_text(encoding="utf-8").splitlines()) for f in files),
        "features": {
            "answer_spaces": hasattr(s3e, "BinaryAnswers"),
            "categorical_answers": hasattr(s3e, "CategoricalAnswers"),
            "null_option": "null_tokens" in _params(estimator) or hasattr(s3e, "BinaryAnswers"),
            "pddl_free_engine": hasattr(s3e, "QueryEngine"),
            "serializable_predictions": hasattr(s3e, "PredictionSet"),
            "calibrator_objects": hasattr(s3e, "PlattCalibrator"),
            "calibration": hasattr(s3e, "PlattCalibrator") or hasattr(estimator, "fit_platt_scaling"),
            "vllm_backend": "VLLMBackend" in names,
            "openai_backend": "OpenAIVLM" in names,
            "interest_token_scoring": "interest_tokens" in backend_query,
            "multi_view_averaging": hasattr(estimator, "estimate_averaged")
            or "multi_image_strategy" in _params(estimator),
            "predicate_subsets": "predicates" in _params(estimator, "estimate")
            or "predicates" in _params(estimator, "estimate_prediction_details"),
            "thinking_models": "thinking" in hf_params,
            "llm_translation": hasattr(s3e, "LLMTranslator"),
        },
    }


def build(s3e, job, scene, backend):
    translator = s3e.TemplateTranslator(job["templates"])
    if hasattr(s3e.SemanticStateEstimator, "from_pddl"):
        return s3e.SemanticStateEstimator.from_pddl(
            scene["domain"], scene["problem"], vlm=backend, translator=translator,
            prompt_template=job["prompt_template"], scoring="logprobs",
            batch_size=job["batch_size"])
    return s3e.SemanticStateEstimator(
        scene["domain"], scene["problem"], vlm=backend, query_translator=translator,
        user_prompt_template=job["prompt_template"], probability_method="logprobs",
        batch_size=job["batch_size"])


def estimate(estimator, images, predicates=None) -> dict:
    if hasattr(estimator, "estimate") and hasattr(estimator, "engine"):
        return estimator.estimate(images, predicates=predicates).probabilities()
    details = estimator.estimate_prediction_details(images, predicates=predicates)
    return {p: d.raw_probability for p, d in details.items()}


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    job = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    out = Path(argv[1])
    import s3e
    import torch

    report = {"surface": surface(s3e), "job": job, "scenes": []}
    if job.get("model"):
        started = time.perf_counter()
        backend = s3e.HuggingFaceVLM(job["model"], torch_dtype=getattr(torch, job["dtype"]),
                                     **({"revision": job["revision"]} if job.get("revision") else {}))
        report["load_seconds"] = time.perf_counter() - started
        scenes = [load_scene(Path(d)) for d in job["scenes"]]
        warmed = False
        for scene in scenes:
            with Image.open(scene["image"]) as image:
                images = [image.convert("RGB")]
            estimator = build(s3e, job, scene, backend)
            if not warmed:
                first = next(iter(scene["state"]))
                estimate(estimator, images, [first])
                warmed = True
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            probabilities = estimate(estimator, images)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            report["scenes"].append({"scene": scene["id"], "seconds": time.perf_counter() - start,
                                     "probabilities": probabilities,
                                     "truth": {p: scene["state"][p] for p in probabilities}})
            print(f"{report['surface']['version']} {scene['id']}: {len(probabilities)} atoms "
                  f"in {report['scenes'][-1]['seconds']:.2f}s", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
