# S3E: Semantic Symbolic State Estimation

[![Tests](https://github.com/CLAIR-LAB-TECHNION/s3e/actions/workflows/tests.yml/badge.svg)](https://github.com/CLAIR-LAB-TECHNION/s3e/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/s3e)](https://pypi.org/project/s3e/)
[![Python versions](https://img.shields.io/pypi/pyversions/s3e)](https://pypi.org/project/s3e/)
[![Documentation](https://readthedocs.org/projects/s3e/badge/?version=latest)](https://s3e.readthedocs.io)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/CLAIR-LAB-TECHNION/s3e/blob/main/LICENSE)

## Overview

S3E estimates grounded PDDL state predicates from images using vision-language models (VLMs). Given a planning domain, a problem, and images of a scene, it asks a VLM one question per grounded predicate (e.g. `on(a,b)` → "Is block a on top of block b?"), reads the probability the model assigns to the answer tokens instead of parsing its text, and returns a probability per predicate (optionally calibrated against labeled examples) plus a symbolic state you can hand back to a planner. It is built for research that grounds symbolic planners in perception, where parsing a model's text throws away its uncertainty and tying code to one provider makes comparing models laborious.

S3E is built as concentric, independently usable layers:

1. **Backends** (`s3e.backends`): a uniform `VLMBackend` interface over HuggingFace, OpenAI, and vLLM models.
2. **Engine** (`s3e.engine`): `QueryEngine` turns images + free-form queries + an answer space into `Prediction`s. No PDDL involved.
3. **Calibration** (`s3e.calibration`): fit a Platt-scaling calibrator on labeled examples, offline and VLM-free after data collection.
4. **State-estimation facade** (`s3e.estimator`): `SemanticStateEstimator` takes grounded predicates (an explicit list, or grounded from a PDDL domain/problem by `from_pddl`), translates them into queries, and drives a `QueryEngine`.

Each layer works standalone: you can use `QueryEngine` to answer arbitrary visual questions without PDDL, or use `SemanticStateEstimator.from_pddl` for the full predicate-grounding workflow.

## Features

- Answer free-form visual queries against a pluggable answer space (`BinaryAnswers`, `CategoricalAnswers`) with logprob or text-match scoring.
- Parse PDDL domains and problems from strings or `.pddl` files, and ground predicates over the problem's objects.
- Translate predicates with pluggable strategies: `IdentityTranslator`, `TemplateTranslator`, `PrewrittenTranslator`, and `LLMTranslator`.
- Use HuggingFace VLMs, OpenAI VLMs, vLLM-backed local models, or custom `VLMBackend` implementations.
- Query one scene at a time, or average predictions across several scenes of the same state (`estimate_averaged` / `PredictionSet.average`).
- Lazy, cached derivations on results (probability, decided answer, has-answer / null-domination, confidence), computed on demand without re-running inference.
- Offline calibration: collect VLM scores once, then fit/refit/apply a calibrator without querying the model again.
- Convert estimated states back into Unified Planning-compatible state objects.

## Installation

S3E requires Python 3.10 or newer. Install it from PyPI with the extras for PDDL grounding and HuggingFace models:

```bash
pip install "s3e[pddl,hf]"
```

A bare `pip install s3e` installs only the core (`Pillow`, `numpy`, `tqdm`). The other extras are `openai`, `vllm` (CUDA hosts only), `calibration`, and `all` (everything except `vllm`). The [getting-started guide](https://s3e.readthedocs.io/en/latest/getting-started.html) lists what each one enables. Using a feature whose extra is missing raises an `ImportError` that names the extra to install. For larger HuggingFace VLMs, a GPU-capable PyTorch environment is recommended.

To install from source, for development or for the latest unreleased version:

```bash
git clone https://github.com/CLAIR-LAB-TECHNION/s3e.git
cd s3e
pip install -e ".[pddl,hf]"     # or ".[dev]" / ".[dev-gpu]" for contributing
```

## Quick Start

This example runs as-is on CPU or GPU with `pip install "s3e[pddl,hf]"`. It uses a small model (`Qwen/Qwen3.5-0.8B`, ~1.7 GB, downloaded on first use) and a synthetic scene, so nothing else needs to be on disk. A model this small is useful for trying the API, not for accurate estimates.

```python
from PIL import Image, ImageDraw

from s3e import HuggingFaceVLM, SemanticStateEstimator, TemplateTranslator

domain_pddl = """
(define (domain blocksworld)
  (:requirements :typing)
  (:types block)
  (:predicates (on ?x - block ?y - block) (clear ?x - block)))
"""
problem_pddl = """
(define (problem two-blocks)
  (:domain blocksworld)
  (:objects blue orange - block)
  (:init (on blue orange) (clear blue))
  (:goal (on orange blue)))
"""

# A synthetic scene: a blue block stacked on an orange block.
# For real data, use Image.open("photo.png"). A scene is a list of images shown together.
image = Image.new("RGB", (256, 256), "white")
draw = ImageDraw.Draw(image)
draw.rectangle((78, 60, 178, 130), fill="deepskyblue", outline="black", width=3)
draw.rectangle((78, 130, 178, 200), fill="orange", outline="black", width=3)
scene = [image]

vlm = HuggingFaceVLM("Qwen/Qwen3.5-0.8B")  # load the model once, then share it
estimator = SemanticStateEstimator.from_pddl(
    domain_pddl,
    problem_pddl,
    vlm=vlm,
    translator=TemplateTranslator({
        "on": "Is the {0} block on top of the {1} block?",
        "clear": "Is the {0} block clear?",
    }),
    prompt_template="Answer with exactly one word, yes or no: {query}",
)

results = estimator.estimate(scene)  # PredictionSet: full detail per predicate
print(results.probabilities())       # {"on(blue,orange)": ..., ...}
state = results.to_state()          # dict[str, bool], ready for a planner
```

`QueryEngine` answers free-form visual questions the same way, with no PDDL:

```python
from s3e import QueryEngine

engine = QueryEngine(vlm)  # the same loaded model
predictions = engine.ask(scene, ["Is there a blue block?", "Is there a green block?"])
print(predictions.to_state())  # {'Is there a blue block?': True, ...}
```

## Documentation

- The [walkthrough notebook](https://github.com/CLAIR-LAB-TECHNION/s3e/blob/main/docs/s3e_walkthrough.ipynb) ([run it in Colab](https://colab.research.google.com/github/CLAIR-LAB-TECHNION/s3e/blob/main/docs/s3e_walkthrough.ipynb)) explains every layer step by step.
- The [getting-started guide](https://s3e.readthedocs.io/en/latest/getting-started.html) tours the API: categorical answers, predicate subsets, multi-view averaging, the OpenAI and vLLM backends, and calibration.
- The [user guide](https://s3e.readthedocs.io/en/latest/user-guide.html) covers every configuration option, thinking models, and how to read and calibrate the results.
- [`examples/`](https://github.com/CLAIR-LAB-TECHNION/s3e/tree/main/examples) holds runnable scripts, including a custom backend and a benchmark on synthetic Blocksworld scenes.
- The [API reference](https://s3e.readthedocs.io/en/latest/api/index.html) documents every public class and function.

## Testing

```bash
pip install -e ".[dev]"       # CPU: full fast suite; vLLM-dependent tests skip
pytest -m "not slow"          # fast suite: CPU only, no model downloads
pytest -m slow                # downloads and runs real models
```

Continuous integration runs the fast suite on Linux, macOS, and Windows, across Python 3.10–3.14, and at the lowest supported dependency versions, for every pull request. A weekly workflow runs the slow tests on CPU, including the Quick Start above, the getting-started guide, and the walkthrough notebook. The [developer guide](https://s3e.readthedocs.io/en/latest/developer-guide.html) describes how each backend is tested without special hardware.

## Getting help and contributing

- **Questions and bug reports:** open an issue on the [issue tracker](https://github.com/CLAIR-LAB-TECHNION/s3e/issues/new/choose) (templates are provided for bugs, feature requests, and questions).
- **Contributing:** see [`CONTRIBUTING.md`](https://github.com/CLAIR-LAB-TECHNION/s3e/blob/main/CONTRIBUTING.md) for the development setup, test commands, and conventions. Pull requests are welcome.
- **Code of conduct:** this project follows the [Contributor Covenant](https://github.com/CLAIR-LAB-TECHNION/s3e/blob/main/CODE_OF_CONDUCT.md).
- **Changes between versions:** see [`CHANGELOG.md`](https://github.com/CLAIR-LAB-TECHNION/s3e/blob/main/CHANGELOG.md).
- **Security issues:** report them privately as described in [`SECURITY.md`](https://github.com/CLAIR-LAB-TECHNION/s3e/blob/main/SECURITY.md).

S3E is maintained by the [CLAIR Lab](https://github.com/CLAIR-LAB-TECHNION) at the Technion – Israel Institute of Technology, which uses it in its own research. Issues and pull requests are triaged by the maintainers on a best-effort basis; [`GOVERNANCE.md`](https://github.com/CLAIR-LAB-TECHNION/s3e/blob/main/GOVERNANCE.md) describes how decisions and releases are made.

## License

This project is licensed under the MIT License. See [`LICENSE`](https://github.com/CLAIR-LAB-TECHNION/s3e/blob/main/LICENSE) for details.

## Citation

If you use S3E in your research, please cite the S3E paper (GitHub's "Cite this repository" button reads the same metadata from [`CITATION.cff`](https://github.com/CLAIR-LAB-TECHNION/s3e/blob/main/CITATION.cff)):

```bibtex
@inproceedings{azran2025s3e,
  title     = {{S3E}: Semantic Symbolic State Estimation With Vision-Language Foundation Models},
  author    = {Azran, Guy and Goshen, Yuval and Yuan, Kai and Keren, Sarah},
  booktitle = {Workshop on Planning in the Era of LLMs (LM4Plan) at the AAAI Conference on Artificial Intelligence},
  year      = {2025},
  url       = {https://openreview.net/forum?id=gw4hYNFUIC}
}
```

If you use S3E in a publication, please let us know by opening an issue, so we can list it in the [documentation](https://s3e.readthedocs.io/en/latest/citing.html).
