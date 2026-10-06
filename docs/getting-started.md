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
| `vllm` | {class}`~s3e.VLLMBackend` (CUDA hosts only) | `vllm>=0.11.0` |
| `calibration` | {meth}`PlattCalibrator.fit <s3e.PlattCalibrator.fit>` | `scikit-learn` |
| `all` | everything except `vllm` | |

```bash
pip install "s3e[pddl,hf]"
```

Using a missing optional feature raises an `ImportError` naming the extra to
install.

## A first estimate

The example below runs a real (small) HuggingFace VLM on CPU or GPU. It
downloads the ~500 MB `HuggingFaceTB/SmolVLM-256M-Instruct` checkpoint on
first use and needs the `pddl` and `hf` extras.

```python
from PIL import Image, ImageDraw

from s3e import SemanticStateEstimator, TemplateTranslator

DOMAIN = """
(define (domain blocksworld)
  (:requirements :typing)
  (:types block)
  (:predicates (on ?x - block ?y - block) (clear ?x - block)))
"""
PROBLEM = """
(define (problem two-blocks)
  (:domain blocksworld)
  (:objects blue orange - block)
  (:init (on blue orange) (clear blue))
  (:goal (on orange blue)))
"""

# A synthetic scene: a blue block stacked on an orange block.
scene = Image.new("RGB", (256, 256), "white")
draw = ImageDraw.Draw(scene)
draw.rectangle((78, 60, 178, 130), fill="deepskyblue", outline="black", width=3)
draw.rectangle((78, 130, 178, 200), fill="orange", outline="black", width=3)

estimator = SemanticStateEstimator.from_pddl(
    DOMAIN,
    PROBLEM,
    vlm="HuggingFaceTB/SmolVLM-256M-Instruct",
    translator=TemplateTranslator({
        "on": "Is the {0} block on top of the {1} block?",
        "clear": "Is the {0} block clear?",
    }),
    prompt_template="Answer with exactly one word, yes or no: {query}",
)

results = estimator.estimate([scene])  # PredictionSet, one entry per predicate
print(results.probabilities())         # {"on(blue,orange)": 0.98..., ...}
print(results.to_state(confidence=0.5))
```

A 256M-parameter model is useful for checking an installation, not for
accurate estimates: expect poorly calibrated probabilities, which is exactly
what the calibration layer addresses.

To exercise the library without downloading any model, implement a tiny
{class}`~s3e.VLMBackend` that returns fixed probabilities. The walkthrough
notebook does this throughout, and the test suite uses the same pattern
(`pytest -m "not slow"` runs on CPU with no downloads).

## Learning more

- The [README](https://github.com/CLAIR-LAB-TECHNION/s3e#readme) covers
  categorical answers, relevant-atom subsets, multi-view averaging,
  calibration, and the vLLM and OpenAI backends.
- The
  [walkthrough notebook](https://github.com/CLAIR-LAB-TECHNION/s3e/blob/main/docs/s3e_walkthrough.ipynb)
  explains every layer step by step.
- The {doc}`api/index` documents every public class and function.
