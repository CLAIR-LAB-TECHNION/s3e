# S3E: Semantic Symbolic State Estimation

[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Overview

`s3e` estimates grounded PDDL state predicates from images using vision-language models (VLMs). It is built as concentric, independently usable layers:

1. **Backends** (`s3e.backends`) — a uniform `VLMBackend` interface over HuggingFace, OpenAI, and vLLM models.
2. **Engine** (`s3e.engine`) — `QueryEngine`: images + free-form queries + an answer space → `Prediction`s. No PDDL involved.
3. **Calibration** (`s3e.calibration`) — fit a Platt-scaling calibrator on labeled examples, offline and VLM-free after data collection.
4. **PDDL facade** (`s3e.estimator`) — `SemanticStateEstimator`: grounds a PDDL domain/problem into predicates, translates them into queries, and drives a `QueryEngine`.

Each layer works standalone: you can use `QueryEngine` to answer arbitrary visual questions without PDDL, or use `SemanticStateEstimator.from_pddl` for the full predicate-grounding workflow.

For a longer tutorial, see the [tutorial notebook](docs/s3e_walkthrough.ipynb) (currently being rewritten for this API — see the notebook's leading cell).

## Features

- Answer free-form visual queries against a pluggable answer space (`BinaryAnswers`, `CategoricalAnswers`) with logprob or text-match scoring.
- Parse PDDL domains and problems from strings or `.pddl` files, and ground predicates over the problem's objects.
- Translate predicates with pluggable strategies: `IdentityTranslator`, `TemplateTranslator`, `PrewrittenTranslator`, and `LLMTranslator`.
- Use HuggingFace VLMs, OpenAI VLMs, vLLM-backed local models, or custom `VLMBackend` implementations.
- Query one scene at a time, or average predictions across several scenes of the same state (`estimate_averaged` / `PredictionSet.average`).
- Lazy, cached derivations on results: probability, argmax answer, null-domination, confidence — computed on demand, never re-running inference.
- Offline calibration: collect VLM scores once, then fit/refit/apply a calibrator without querying the model again.
- Convert estimated states back into Unified Planning-compatible state objects.

## Installation

### Prerequisites

- Python `>=3.10`
- `pip`
- `git` if installing from source
- For larger HuggingFace VLMs, a GPU-capable PyTorch environment is recommended

### Install from source

```bash
git clone https://github.com/CLAIR-LAB-TECHNION/s3e.git
cd s3e
pip install -e ".[pddl,hf]"
```

You can also install directly from the GitHub repository without cloning:

```bash
pip install "git+https://github.com/CLAIR-LAB-TECHNION/s3e.git#egg=s3e[pddl,hf]"
```

A bare `pip install s3e` installs only the core (`Pillow`, `numpy`, `tqdm`): the engine, result objects, answer spaces, and non-LLM translators, with no heavy dependencies. Add extras for the pieces you need:

```bash
pip install -e ".[pddl]"          # PDDL grounding (SemanticStateEstimator.from_pddl)
pip install -e ".[hf]"            # HuggingFace VLM backend
pip install -e ".[openai]"        # OpenAI VLM backend
pip install -e ".[vllm]"          # local multi-GPU inference via vLLM
pip install -e ".[calibration]"   # Platt-scaling calibration (scikit-learn)
pip install -e ".[all]"           # everything except vllm (platform-constrained)
pip install -e ".[dev]"           # pytest + s3e[all], for contributing
pip install -e ".[dev-gpu]"       # dev + vllm, for contributing on CUDA hosts
```

Optional acceleration for supported HuggingFace models:

FlashAttention installation is platform- and hardware-dependent. If your chosen model and environment support it, follow the [installation guide](https://github.com/dao-ailab/flash-attention?tab=readme-ov-file#installation-and-features) to set it up.

## Quick Start

### Engine-only: answer a visual question, no PDDL

`QueryEngine` is the PDDL-free heart of `s3e`: images + queries + an answer space → predictions.

```python
from PIL import Image

from s3e import QueryEngine

engine = QueryEngine("HuggingFaceTB/SmolVLM-256M-Instruct")

scene = [Image.open("kitchen.png")]
predictions = engine.ask(scene, ["Is the stove on?", "Is the fridge door open?"])

print(predictions["Is the stove on?"].probability)   # P(true), e.g. 0.83
print(predictions.to_state())                        # {'Is the stove on?': True, ...}
```

### Categorical answers

Answer spaces are not limited to yes/no. `CategoricalAnswers` scores an arbitrary set of labeled options:

```python
from s3e import CategoricalAnswers, QueryEngine

engine = QueryEngine(
    "HuggingFaceTB/SmolVLM-256M-Instruct",
    answers=CategoricalAnswers(["red", "green", "blue"]),
)
predictions = engine.ask(scene, ["What color is the mug?"])

print(predictions["What color is the mug?"].answer)         # e.g. "red"
print(predictions["What color is the mug?"].distribution())  # {"red": 0.7, "green": 0.2, "blue": 0.1}
```

### Full workflow: `SemanticStateEstimator.from_pddl`

`SemanticStateEstimator` grounds a PDDL domain/problem into predicates, translates them into queries with a pluggable `QueryTranslator`, and drives a `QueryEngine`.

```python
from PIL import Image

from s3e import SemanticStateEstimator, TemplateTranslator

domain_pddl = """
(define (domain blocksworld)
  (:requirements :typing)
  (:types block)
  (:predicates
    (on ?x - block ?y - block)
    (clear ?x - block)
  )
)
"""

problem_pddl = """
(define (problem bw-2)
  (:domain blocksworld)
  (:objects a b - block)
  (:init (on a b) (clear a))
  (:goal (on b a))
)
"""

translator = TemplateTranslator(
    {
        "on": "Is the {0} block on top of the {1} block?",
        "clear": "Is the {0} block clear?",
    }
)

estimator = SemanticStateEstimator.from_pddl(
    domain_pddl,
    problem_pddl,
    vlm="HuggingFaceTB/SmolVLM-256M-Instruct",
    translator=translator,
)

images = [Image.open("scene.png")]

state = estimator(images)                # dict[str, bool | None]
results = estimator.estimate(images)     # PredictionSet: full detail per predicate
probabilities = results.probabilities()  # dict[str, float]

print(state)
print(probabilities)
```

Query only a subset of predicates (relevant-atom masking), or average across several scenes depicting the same state:

```python
subset_state = estimator.estimate(images, predicates=["on(a,b)", "clear(a)"]).to_state()

scenes = [[Image.open("scene-1.png")], [Image.open("scene-2.png")]]
averaged = estimator.estimate_averaged(scenes)
```

Inspect the normalized backend output behind a prediction with `keep_raw=True`:

```python
results = estimator.estimate(images, keep_raw=True)
print(results["on(a,b)"].raw)   # VLMOutput: token_probs, text, argmax_in_interest
```

Convert the boolean state back into a Unified Planning state object:

```python
up_state = estimator.to_up_state(state)
```

For OpenAI-backed models, install the optional dependency (`pip install -e ".[openai]"`) and use an `OpenAI/`-prefixed model ID, e.g. `vlm="OpenAI/gpt-4o"`. For local multi-GPU inference, construct `VLLMBackend(...)` explicitly and pass the instance as `vlm=`:

```python
from s3e import SemanticStateEstimator, VLLMBackend

vlm = VLLMBackend("Qwen/Qwen2-VL-7B-Instruct", tensor_parallel_size=2)
estimator = SemanticStateEstimator.from_pddl(
    domain_pddl, problem_pddl, vlm=vlm, translator=translator,
)
```

### Calibration

Calibration is a self-contained pipeline over prediction data, in `s3e.calibration`. It never touches estimator internals, and the expensive step (querying the VLM on labeled examples) is separate from fitting, which is cheap and offline:

```python
from PIL import Image

from s3e import CalibrationExample, CalibrationSet, PlattCalibrator

examples = [
    CalibrationExample(
        images=[Image.open("calibration-scene-1.png")],
        state_dict={
            "on(a,b)": True,
            "clear(a)": True,
            "clear(b)": False,
        },
    ),
]

# Expensive: queries the VLM once per example.
data = CalibrationSet.collect(estimator, examples)
data.save("calibration-data.json")

# Cheap and VLM-free from here on, any time later:
data = CalibrationSet.load("calibration-data.json")
calibrator = PlattCalibrator.fit(data, scope="lifted")
calibrator.save("platt-profile.json")

calibrator = PlattCalibrator.load("platt-profile.json")
calibrated_results = calibrator.apply(results)                       # new PredictionSet
calibrated_state = estimator.estimate(images, calibrator=calibrator).to_state()
```

`scope` groups samples for fitting: `"global"` (one calibrator for everything), `"lifted"` (one per predicate name, e.g. all `on(...)` instances share a fit), or `"grounded"` (one per fully-grounded predicate). When examples span multiple problem instances, set `CalibrationExample.problem` on each — `CalibrationSet.collect` re-grounds the estimator against that problem before querying it, and the saved sample carries the problem string alongside its score and label.

## API Reference / Configuration

### `SemanticStateEstimator`

`SemanticStateEstimator(predicates, vlm=..., translator=...)` builds from an explicit predicate list; `SemanticStateEstimator.from_pddl(domain, problem, vlm=..., translator=...)` grounds predicates from PDDL. Key arguments:

- `predicates` (constructor only): grounded predicate strings to estimate.
- `domain`, `problem` (`from_pddl` only): PDDL domain and problem, as strings or `.pddl` file paths.
- `vlm`: a `VLMBackend` instance or a model-id string (see `resolve_backend`). Strings prefixed with `OpenAI/` select `OpenAIVLM`; any other string selects `HuggingFaceVLM`. For vLLM, construct `VLLMBackend(...)` explicitly and pass the instance.
- `translator`: predicate-to-query strategy (default: `IdentityTranslator`).
- `answers`: the answer space (default: `BinaryAnswers()`; identity translation defaults to `BinaryAnswers("true", "false")`).
- `confidence`: default acceptance threshold used by `__call__`/`to_state`. A predicate is accepted as `True` when `P(true) >= confidence`; otherwise it is `False` when `P(false) >= confidence`, and `None` when neither side reaches the threshold (or the prediction is null-dominated). The `True` check runs first, so any value works: below `0.5` a predicate meeting both checks resolves to `True`; above `0.5` undecided predicates become `None`.
- `scoring`: `"logprobs"` (default) or `"text_match"`.
- `system_prompt`, `prompt_template`, `additional_instructions`: prompt construction; `prompt_template` must contain `{query}`.
- `true_tokens`, `false_tokens`, `null_tokens`: convenience overrides for the default binary answer space; ignored when `answers` is passed explicitly.
- `batch_size`, `vlm_kwargs`, `inference_kwargs`: forwarded to the underlying `QueryEngine` (see below).

Common methods:

- `estimator(images) -> dict[str, bool | None]`: estimate and threshold into a boolean state.
- `estimator.estimate(images, *, predicates=None, calibrator=None, keep_raw=False, inference_kwargs=None) -> PredictionSet`: full per-predicate detail.
- `estimator.estimate_averaged(scenes, **estimate_kwargs) -> PredictionSet`: estimate each scene separately and average the stored masses.
- `estimator.set_problem(domain, problem)`: re-ground a new PDDL problem; the engine/backend is untouched.
- `estimator.to_up_state(state)`: convert a boolean state dict into a Unified Planning state object (PDDL-built estimators only).

### `QueryEngine`

`QueryEngine(vlm, *, answers=None, scoring="logprobs", system_prompt=None, prompt_template="{query}", batch_size=8, inference_kwargs=None, vlm_kwargs=None)` is the PDDL-free engine `SemanticStateEstimator` is built on. `resolve_backend(vlm, **vlm_kwargs)` is the public model-string-to-backend factory it uses internally.

- `engine.ask(images, queries, *, answers=None, scoring=None, inference_kwargs=None, keep_raw=False) -> PredictionSet`: answer each query about one scene (a list of images shown together).
- `engine.ask_each(scenes, queries, **ask_kwargs) -> list[PredictionSet]`: run `ask` once per scene; combine with `PredictionSet.average(sets)`.

`vlm_kwargs` and `inference_kwargs` are intentionally different:

- `vlm_kwargs` configure backend/client construction, used only when `vlm` is a model string.
  - OpenAI backend: forwarded to `openai.OpenAI(...)` (e.g. `api_key`, `base_url`, `timeout`).
  - HuggingFace backend: forwarded to model construction (e.g. `device_map`, `torch_dtype`, `attn_implementation`).
  - vLLM backend: pass these directly to `VLLMBackend(...)` (e.g. `tensor_parallel_size`, `gpu_memory_utilization`).
- `inference_kwargs` configure runtime inference and are forwarded on every query.
  - OpenAI: request arguments for `chat.completions.create` (e.g. `temperature`, `max_completion_tokens`).
  - HuggingFace: forwarded to `model(...)` in logprobs mode and `model.generate(...)` in generation (`text_match`) mode.
  - vLLM: forwarded to `vllm.SamplingParams` (e.g. `temperature`, `max_tokens`).

### Answer spaces

- `BinaryAnswers(true_label="yes", false_label="no", *, true_tokens=None, false_tokens=None, null_label="unknown", null_tokens=None)`: two options with boolean semantics.
- `CategoricalAnswers(options, *, null_label="unknown", null_tokens=None)`: N options, given as labels or explicit `AnswerOption`s.
- `AnswerOption(label, tokens)` / `AnswerOption.make(label, tokens=None)`: a label plus the token strings that express it; labels auto-expand into case/leading-whitespace variants when `tokens` is omitted.

### Results

`Prediction` (one query's outcome) and `PredictionSet` (an ordered mapping of query/predicate → `Prediction`) are both immutable, with lazily cached derivations:

- `prediction.masses`, `.null_mass`: the stored raw data.
- `prediction.probability`, `.answer`, `.null_dominated`, `.confident(threshold)`, `.distribution()`, `.score`: derived on demand.
- `prediction_set.probabilities()`, `.to_state(confidence=0.5)`, `.where(predicate)`.
- `prediction_set.to_dict()` / `PredictionSet.from_dict(d)`: backend-free round trip (e.g. for offline recalibration).

### Translators

- `IdentityTranslator`: use grounded predicates as-is.
- `TemplateTranslator`: format grounded predicates with per-predicate templates.
- `PrewrittenTranslator`: provide explicit prompts for each grounded predicate.
- `LLMTranslator`: generate natural-language prompts with an LLM and optionally cache them (`cache_dir=...`).

### Environment variables and optional configuration

- `OPENAI_API_KEY`: required for `OpenAIVLM` and OpenAI-backed `LLMTranslator` usage.
- `cache_dir` on `LLMTranslator`: enables on-disk caching of generated predicate translations.

## Contributing

Install development dependencies:

```bash
pip install -e ".[dev]"       # CPU: full fast suite; vLLM-dependent tests skip
pip install -e ".[dev-gpu]"   # CUDA hosts: adds vllm for the vLLM test coverage
```

Run the fast test loop:

```bash
pytest -m "not slow"
```

Run the full test suite:

```bash
pytest
```

To contribute:

1. Fork the repository and create a feature branch.
2. Add or update tests for behavioral changes.
3. Run the relevant test commands before submitting.
4. Open a pull request with a concise description of the change and its motivation.

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.

## Citation

```bibtex
@inproceedings{azranS3ESemanticSymbolic2025,
  title = {{{S3E}}: {{Semantic Symbolic State Estimation With Vision-Language Foundation Models}}},
  shorttitle = {{{S3E}}},
  booktitle = {{{AAAI}} 2025 {{Workshop LM4Plan}}},
  author = {Azran, Guy and Goshen, Yuval and Yuan, Kai and Keren, Sarah},
  year = 2025,
}
```
