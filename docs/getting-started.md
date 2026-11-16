# Getting started

## Installation

S3E requires Python 3.10 or newer and is available on
[PyPI](https://pypi.org/project/s3e/):

```bash
pip install s3e
```

A bare install contains only the core (`numpy`, `Pillow`, `tqdm`): the
engine, result objects, answer spaces, and non-LLM translators. Add extras for
the pieces you need:

| Extra | Enables | Installs |
|---|---|---|
| `pddl` | {meth}`SemanticStateEstimator.from_pddl <s3e.SemanticStateEstimator.from_pddl>`, {mod}`s3e.pddl` | `unified-planning` |
| `hf` | {class}`~s3e.HuggingFaceVLM` | `torch`, `torchvision`, `transformers`, `accelerate` |
| `openai` | {class}`~s3e.OpenAIVLM`, OpenAI-backed {class}`~s3e.LLMTranslator` | `openai` |
| `vllm` | {class}`~s3e.VLLMBackend` (CUDA hosts only) | `vllm>=0.11.1` |
| `calibration` | {meth}`PlattCalibrator.fit <s3e.PlattCalibrator.fit>` | `scikit-learn` |
| `all` | everything except `vllm` | |

```bash
pip install "s3e[pddl,hf]"
```

Using a missing optional feature raises an `ImportError` naming the extra to
install.

FlashAttention can speed up supported HuggingFace models. Its installation
depends on the platform and hardware, so follow its
[installation guide](https://github.com/dao-ailab/flash-attention?tab=readme-ov-file#installation-and-features)
if your model and environment support it.

## Quick start

The snippets below run as-is, in order, on CPU or GPU, with `pip install "s3e[pddl,hf,calibration]"`. They use a tiny model (`HuggingFaceTB/SmolVLM-256M-Instruct`, ~500 MB, downloaded on first use) and a synthetic scene so nothing else needs to be on disk. A model this small is useful for trying the API, not for accurate estimates: expect poorly calibrated probabilities, which is what the calibration layer is for. On a GPU each snippet takes seconds; on a laptop-class CPU, expect a minute or more per query (SmolVLM splits each image into many tiles).

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

`QueryEngine` is the PDDL-free heart of S3E: images + queries + an answer space → predictions.

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

Convert the boolean state back into a Unified Planning state object:

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


The {doc}`user guide <user-guide>` explains how `collect` groups and filters
samples, and which checks decide whether a calibrator applies to a call.

## Without downloading a model

To exercise the library without downloading any model, implement a tiny
{class}`~s3e.VLMBackend` that returns fixed probabilities. The walkthrough
notebook does this throughout, and the test suite uses the same pattern
(`pytest -m "not slow"` runs on CPU with no downloads).
`examples/custom_backend.py` runs the whole pipeline this way (see
{doc}`examples`).

## Learning more

- The {doc}`walkthrough notebook <s3e_walkthrough>`
  ([run it in Colab](https://colab.research.google.com/github/CLAIR-LAB-TECHNION/s3e/blob/main/docs/s3e_walkthrough.ipynb))
  explains every layer step by step.
- The {doc}`user guide <user-guide>` covers every configuration option and
  how to read the results.
- The {doc}`api/index` documents every public class and function.
