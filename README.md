# S3E: Semantic Symbolic State Estimation

[![Tests](https://github.com/CLAIR-LAB-TECHNION/s3e/actions/workflows/tests.yml/badge.svg)](https://github.com/CLAIR-LAB-TECHNION/s3e/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/s3e)](https://pypi.org/project/s3e/)
[![Python versions](https://img.shields.io/pypi/pyversions/s3e)](https://pypi.org/project/s3e/)
[![Documentation](https://readthedocs.org/projects/s3e/badge/?version=latest)](https://s3e.readthedocs.io)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/CLAIR-LAB-TECHNION/s3e/blob/main/LICENSE)

## Overview

`s3e` estimates grounded PDDL state predicates from images using vision-language models (VLMs). Given a planning domain, a problem, and images of a scene, it asks a VLM one question per grounded predicate (e.g. `on(a,b)` → "Is block a on top of block b?"), reads the probability the model assigns to the answer tokens instead of parsing its text, and returns a probability per predicate (optionally calibrated against labeled examples) plus a symbolic state you can hand back to a planner.

### Statement of need

Research that grounds symbolic planners in perception — task planning, task-and-motion planning, embodied AI, neuro-symbolic reasoning — keeps re-implementing the same pipeline: enumerate grounded predicates, phrase a question for each, prompt a particular model API with images, and turn the output into truth values. Parsing generated text throws away the model's uncertainty, and hard-wiring one provider makes cross-model comparisons and calibration studies laborious. `s3e` packages this pipeline as a backend-agnostic library with probabilities as a first-class output: answer-token probability scoring, optional abstention, multi-view averaging, and offline calibration, over HuggingFace, vLLM, and OpenAI models. Its query engine also works without PDDL, for anyone who needs probabilistic yes/no or multiple-choice answers from a VLM.

`s3e` is built as concentric, independently usable layers:

1. **Backends** (`s3e.backends`) — a uniform `VLMBackend` interface over HuggingFace, OpenAI, and vLLM models.
2. **Engine** (`s3e.engine`) — `QueryEngine`: images + free-form queries + an answer space → `Prediction`s. No PDDL involved.
3. **Calibration** (`s3e.calibration`) — fit a Platt-scaling calibrator on labeled examples, offline and VLM-free after data collection.
4. **PDDL facade** (`s3e.estimator`) — `SemanticStateEstimator`: grounds a PDDL domain/problem into predicates, translates them into queries, and drives a `QueryEngine`.

Each layer works standalone: you can use `QueryEngine` to answer arbitrary visual questions without PDDL, or use `SemanticStateEstimator.from_pddl` for the full predicate-grounding workflow.

For a longer tutorial, see the [tutorial notebook](https://github.com/CLAIR-LAB-TECHNION/s3e/blob/main/docs/s3e_walkthrough.ipynb); the full API reference is at [s3e.readthedocs.io](https://s3e.readthedocs.io).

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
- For larger HuggingFace VLMs, a GPU-capable PyTorch environment is recommended

### Install from PyPI

```bash
pip install "s3e[pddl,hf]"
```

A bare `pip install s3e` installs only the core (`Pillow`, `numpy`, `tqdm`): the engine, result objects, answer spaces, and non-LLM translators, with no heavy dependencies. Add extras for the pieces you need:

```bash
pip install "s3e[pddl]"          # PDDL grounding (SemanticStateEstimator.from_pddl)
pip install "s3e[hf]"            # HuggingFace VLM backend
pip install "s3e[openai]"        # OpenAI VLM backend
pip install "s3e[vllm]"          # local multi-GPU inference via vLLM (CUDA only)
pip install "s3e[calibration]"   # Platt-scaling calibration (scikit-learn)
pip install "s3e[all]"           # everything except vllm (platform-constrained)
```

Using a feature whose extra is missing raises an `ImportError` that names the extra to install.

### Install from source

```bash
git clone https://github.com/CLAIR-LAB-TECHNION/s3e.git
cd s3e
pip install -e ".[pddl,hf]"     # or ".[dev]" / ".[dev-gpu]" for contributing
```

You can also install the latest development version without cloning:

```bash
pip install "s3e[pddl,hf] @ git+https://github.com/CLAIR-LAB-TECHNION/s3e.git"
```

Optional acceleration for supported HuggingFace models:

FlashAttention installation is platform- and hardware-dependent. If your chosen model and environment support it, follow the [installation guide](https://github.com/dao-ailab/flash-attention?tab=readme-ov-file#installation-and-features) to set it up.

## Quick Start

The snippets below run as-is, in order, on CPU or GPU, with `pip install "s3e[pddl,hf,calibration]"`. They use a tiny model (`HuggingFaceTB/SmolVLM-256M-Instruct`, ~500 MB, downloaded on first use) and a synthetic scene so nothing else needs to be on disk. A model this small is useful for trying the API, not for accurate estimates — expect poorly calibrated probabilities, which is what the calibration layer is for. On a GPU each snippet takes seconds; on a laptop-class CPU, expect up to about a minute per query (SmolVLM splits each image into many tiles).

```python
from PIL import Image, ImageDraw

# A synthetic scene: a blue block stacked on an orange block.
# For real data, use Image.open("photo.png"). A scene is a list of images shown together.
image = Image.new("RGB", (256, 256), "white")
draw = ImageDraw.Draw(image)
draw.rectangle((78, 60, 178, 130), fill="deepskyblue", outline="black", width=3)
draw.rectangle((78, 130, 178, 200), fill="orange", outline="black", width=3)
scene = [image]
```

### Engine-only: answer a visual question, no PDDL

`QueryEngine` is the PDDL-free heart of `s3e`: images + queries + an answer space → predictions.

```python
from s3e import QueryEngine

engine = QueryEngine("HuggingFaceTB/SmolVLM-256M-Instruct")
predictions = engine.ask(scene, ["Is there a blue block?", "Is there a green block?"])

print(predictions["Is there a blue block?"].probability)  # P(true), a float in [0, 1]
print(predictions.to_state())                             # {'Is there a blue block?': True, ...}
```

### Categorical answers

Answer spaces are not limited to yes/no. `CategoricalAnswers` scores an arbitrary set of labeled options:

```python
from s3e import CategoricalAnswers

color_engine = QueryEngine(
    "HuggingFaceTB/SmolVLM-256M-Instruct",
    answers=CategoricalAnswers(["blue", "orange", "green"]),
)
question = "What color is the top block? Answer with one word."
predictions = color_engine.ask(scene, [question])

print(predictions[question].answer)          # the most likely label, e.g. "blue"
print(predictions[question].distribution())  # {"blue": ..., "orange": ..., "green": ...}
```

### Full workflow: `SemanticStateEstimator.from_pddl`

`SemanticStateEstimator` grounds a PDDL domain/problem into predicates, translates them into queries with a pluggable `QueryTranslator`, and drives a `QueryEngine`.

```python
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
(define (problem two-blocks)
  (:domain blocksworld)
  (:objects blue orange - block)
  (:init (on blue orange) (clear blue))
  (:goal (on orange blue))
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
    prompt_template="Answer with exactly one word, yes or no: {query}",
)

state = estimator(scene)                 # dict[str, bool]
results = estimator.estimate(scene)      # PredictionSet: full detail per predicate
probabilities = results.probabilities()  # dict[str, float]

print(state)
print(probabilities)
```

Query only a subset of predicates (relevant-atom masking), or average across several scenes depicting the same state (e.g. different camera views):

```python
from PIL import ImageOps

subset_state = estimator.estimate(scene, predicates=["on(blue,orange)", "clear(blue)"]).to_state()

views = [scene, [ImageOps.mirror(image)]]
averaged = estimator.estimate_averaged(views)
```

Inspect the normalized backend output behind a prediction with `keep_raw=True`:

```python
results = estimator.estimate(scene, keep_raw=True)
print(results["on(blue,orange)"].raw)   # VLMOutput: token_probs, text, argmax_in_interest
```

Convert the boolean state back into a Unified Planning state object (note that undecided `None` entries are currently converted to `False`; drop or resolve them first if that matters):

```python
up_state = estimator.to_up_state(state)
```

For OpenAI-backed models, install the optional dependency (`pip install "s3e[openai]"`), set `OPENAI_API_KEY`, and use an `OpenAI/`-prefixed model ID, e.g. `vlm="OpenAI/gpt-4o"`. For local multi-GPU inference on CUDA hosts, install `s3e[vllm]`, construct `VLLMBackend(...)` explicitly, and pass the instance as `vlm=`:

```python
from s3e import SemanticStateEstimator, VLLMBackend

vlm = VLLMBackend("Qwen/Qwen2-VL-7B-Instruct", tensor_parallel_size=2)
estimator = SemanticStateEstimator.from_pddl(
    domain_pddl, problem_pddl, vlm=vlm, translator=translator,
)
```

### Calibration

Calibration is a self-contained pipeline over prediction data, in `s3e.calibration`. It never touches estimator internals, and the expensive step (querying the VLM on labeled examples) is separate from fitting, which is cheap and offline. In practice, collect many labeled scenes that are held out from evaluation; one scene keeps this example short:

```python
from s3e import CalibrationExample, CalibrationSet, PlattCalibrator

examples = [
    CalibrationExample(
        images=scene,
        state_dict={
            "on(blue,orange)": True,
            "on(orange,blue)": False,
            "clear(blue)": True,
            "clear(orange)": False,
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
calibrated_state = estimator.estimate(scene, calibrator=calibrator).to_state()
```

`CalibrationSet.collect` skips predictions with no answer: their probability stays `0.5` whatever the calibrator says, so they are not training points.

`scope` groups samples for fitting: `"global"` (one calibrator for everything), `"lifted"` (one per predicate name, e.g. all `on(...)` instances share a fit), or `"grounded"` (one per fully-grounded predicate). Every group needs both true and false labels. When examples span multiple problem instances, set `CalibrationExample.problem` on each — `CalibrationSet.collect` re-grounds the estimator against that problem before querying it, and the saved sample carries the problem string alongside its score and label.

## API Reference / Configuration

### `SemanticStateEstimator`

`SemanticStateEstimator(predicates, vlm=..., translator=...)` builds from an explicit predicate list; `SemanticStateEstimator.from_pddl(domain, problem, vlm=..., translator=...)` grounds predicates from PDDL. Key arguments:

- `predicates` (constructor only): grounded predicate strings to estimate.
- `domain`, `problem` (`from_pddl` only): PDDL domain and problem, as strings or `.pddl` file paths.
- `vlm`: a `VLMBackend` instance or a model-id string (see `resolve_backend`). Strings prefixed with `OpenAI/` select `OpenAIVLM`; any other string selects `HuggingFaceVLM`. For vLLM, construct `VLLMBackend(...)` explicitly and pass the instance.
- `translator`: predicate-to-query strategy (default: `IdentityTranslator`).
- `answers`: the answer space (default: `BinaryAnswers()`; identity translation defaults to `BinaryAnswers("true", "false")`).
- `confidence`: default acceptance threshold used by `__call__`/`to_state`, for binary answer spaces only. A predicate is `True` when `P(true) >= confidence` and `False` otherwise, so every predicate gets a value. A predicate with no answer (see [Results](#results)) has `P(true) = 0.5`: it is `True` at the default `0.5` and `False` for any higher threshold.
- `scoring`: `"logprobs"` (default) scores the masses of the answer tokens; `"text_match"` lets the model generate and matches the reply's *start* against the answer tokens (whole words, case-sensitive, surrounding whitespace ignored, longest token first). A reply such as `"Answer: yes"`, `"**Yes**"`, or one that reasons before answering (e.g. a `<think>` block) matches nothing; prompt for a bare answer.
- `system_prompt`, `prompt_template`, `additional_instructions`: prompt construction; `prompt_template` must contain `{query}`.
- `true_tokens`, `false_tokens`, `null_tokens`: convenience overrides for the default binary answer space; ignored when `answers` is passed explicitly.
- `batch_size`, `vlm_kwargs`, `inference_kwargs`: forwarded to the underlying `QueryEngine` (see below).

Common methods:

- `estimator(images, confidence=None) -> dict[str, bool]`: estimate and threshold into a boolean state.
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

- `prediction.masses`, `.null_mass`, `.unassigned_mass`, `.probability_override` (a calibrator's output): the stored raw data, never altered by the decision rule.
- `prediction.probability`, `.answer`, `.has_answer`, `.null_dominated`, `.matched`, `.confident(threshold)`, `.distribution()`, `.score`: derived on demand.
- `prediction_set.probabilities()`, `.to_state(confidence=0.5)`, `.where(predicate)`.
- `prediction_set.to_dict()` / `PredictionSet.from_dict(d)`: backend-free round trip (e.g. for offline recalibration).

`probability` is `T / (T + F)` over the true and false masses, or the calibrated value when a calibrator was applied. A prediction has **no answer** (`has_answer` is `False`) when the null option is the model's top answer (`null_dominated`) or no answer token got any mass (`matched` is `False`, e.g. a `text_match` reply that starts with none of the tokens). Then `probability` is `0.5` even after calibration, `distribution()` is uniform, and `answer` falls to the first option on the resulting tie (`True` for binary). The engine emits one `UnmatchedAnswerWarning` per call that had unmatched queries; replies that match the null option do not warn.

Because the stored data is untouched, other rules can be derived from it. For example, to count null mass as half true and half false instead: `(T + n/2) / (T + F + n)` from `masses` and `null_mass`, or `w * probability_override + (1 - w) * 0.5` with `w = (T + F) / (T + F + n)` for calibrated predictions (use `0.5` when `T + F + n == 0`). Apply such rules per scene, before `PredictionSet.average`: averaging already applies the built-in rule to which overrides it keeps.

### Translators

- `IdentityTranslator`: use grounded predicates as-is.
- `TemplateTranslator`: format grounded predicates with per-predicate templates.
- `PrewrittenTranslator`: provide explicit prompts for each grounded predicate.
- `LLMTranslator`: generate natural-language prompts with an LLM and optionally cache them (`cache_dir=...`).

### Environment variables and optional configuration

- `OPENAI_API_KEY`: required for `OpenAIVLM` and OpenAI-backed `LLMTranslator` usage.
- `cache_dir` on `LLMTranslator`: enables on-disk caching of generated predicate translations.

## Testing

Install the development dependencies and run the test suite:

```bash
pip install -e ".[dev]"       # CPU: full fast suite; vLLM-dependent tests skip
pip install -e ".[dev-gpu]"   # CUDA hosts: adds vllm for the vLLM test coverage

pytest -m "not slow"          # fast suite: CPU only, no model downloads
pytest -m slow                # downloads and runs real models
pytest                        # everything
```

Continuous integration runs the fast suite on Python 3.10–3.14 for every pull request and every push to `main`. On a machine without a CUDA GPU, install CPU-only PyTorch first to avoid the much larger CUDA build: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.

How each part of the library can be verified without special hardware:

- **Engine, answer spaces, results, translators, calibration, PDDL grounding:** covered by the fast suite, which drives everything through deterministic fake backends (`tests/fakes.py`).
- **HuggingFace backend:** mocked in the fast suite; `pytest -m slow tests/backends/test_huggingface.py` and the Quick Start above run real small models on a CPU.
- **OpenAI backend:** mocked in the fast suite; real use needs an `OPENAI_API_KEY`.
- **vLLM backend:** mocked in the fast suite when `vllm` is installed; the real-engine tests (`pytest -m slow tests/backends/test_vllm.py`) need a CUDA GPU.

## Getting help and contributing

- **Questions and bug reports:** open an issue on the [issue tracker](https://github.com/CLAIR-LAB-TECHNION/s3e/issues/new/choose) (templates are provided for bugs, feature requests, and questions).
- **Contributing:** see [`CONTRIBUTING.md`](https://github.com/CLAIR-LAB-TECHNION/s3e/blob/main/CONTRIBUTING.md) for the development setup, test commands, and conventions. Pull requests are welcome.
- **Code of conduct:** this project follows the [Contributor Covenant](https://github.com/CLAIR-LAB-TECHNION/s3e/blob/main/CODE_OF_CONDUCT.md).
- **Changes between versions:** see [`CHANGELOG.md`](https://github.com/CLAIR-LAB-TECHNION/s3e/blob/main/CHANGELOG.md).

`s3e` is maintained by the [CLAIR Lab](https://github.com/CLAIR-LAB-TECHNION) at the Technion – Israel Institute of Technology, which uses it in its own research. Issues and pull requests are triaged by the maintainers on a best-effort basis.

## License

This project is licensed under the MIT License. See [`LICENSE`](https://github.com/CLAIR-LAB-TECHNION/s3e/blob/main/LICENSE) for details.

## Citation

If you use `s3e` in your research, please cite the S3E paper (GitHub's "Cite this repository" button reads the same metadata from [`CITATION.cff`](https://github.com/CLAIR-LAB-TECHNION/s3e/blob/main/CITATION.cff)):

```bibtex
@inproceedings{azran2025s3e,
  title     = {{S3E}: Semantic Symbolic State Estimation With Vision-Language Foundation Models},
  author    = {Azran, Guy and Goshen, Yuval and Yuan, Kai and Keren, Sarah},
  booktitle = {Workshop on Planning in the Era of LLMs (LM4Plan) at the AAAI Conference on Artificial Intelligence},
  year      = {2025},
  url       = {https://openreview.net/forum?id=gw4hYNFUIC}
}
```
