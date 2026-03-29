# S3E: Semantic Symbolic State Estimation

[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Overview

`s3e` is a Python package for estimating grounded PDDL state predicates from images using vision-language models (VLMs).

It is designed for workflows that need to connect visual observations to symbolic planning. Given a PDDL domain and problem, `s3e` enumerates grounded predicates, translates them into model-friendly queries, and returns either boolean state assignments, per-predicate probabilities, or normalized model outputs suitable for inspection and debugging.

The package integrates naturally with Unified Planning / PDDL-based systems and supports both HuggingFace and OpenAI-backed VLMs, as well as custom backends.

For a longer tutorial, see the [tutorial notebook](docs/s3e_walkthrough.ipynb).

## Features

- Estimate boolean symbolic states or probabilistic predicate values from one or more images.
- Parse PDDL domains and problems from strings or `.pddl` files.
- Automatically ground predicates over the current problem objects.
- Translate predicates with pluggable strategies: `IdentityTranslator`, `TemplateTranslator`, `PrewrittenTranslator`, and `LLMTranslator`.
- Use HuggingFace VLMs, OpenAI VLMs, or custom implementations via the `VLMBackend` interface.
- Support multi-image estimation with either single-pass or per-image averaging.
- Expose normalized `VLMOutput` objects for prompt tuning and backend inspection.
- Convert estimated states back into Unified Planning-compatible state objects.
- Cache LLM-generated predicate translations for reuse across runs.

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
pip install -e .
```

You can also install directly from the GitHub repository without cloning:

```bash
pip install "git+https://github.com/CLAIR-LAB-TECHNION/s3e.git"
```

### Optional dependencies

Install OpenAI support:

```bash
pip install -e '.[openai]'
```

Install development dependencies:

```bash
pip install -e '.[dev]'
```

Optional acceleration for supported HuggingFace models:

FlashAttention installation is platform- and hardware-dependent. If your chosen model and environment support it, follow the [installation guide](https://github.com/dao-ailab/flash-attention?tab=readme-ov-file#installation-and-features) to set it up.

## Quick Start / Usage

The example below uses a small HuggingFace model and template-based predicate translation.

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

estimator = SemanticStateEstimator(
    domain_pddl,
    problem_pddl,
    vlm="HuggingFaceTB/SmolVLM-256M-Instruct",
    query_translator=translator,
    user_prompt_template="Answer yes or no only: {query}",
)

images = [Image.open("scene.png")]

state = estimator(images)
probabilities = estimator.estimate_probabilities(images)

print(state)
print(probabilities)
```

You can also inspect normalized backend outputs directly:

```python
raw_outputs = estimator.estimate_raw(images)
print(raw_outputs["on(a,b)"])
```

To convert the boolean state back into a Unified Planning state object:

```python
from s3e.pddl.up_utils import state_dict_to_up_state

up_state = state_dict_to_up_state(estimator.up_problem, state)
```

For OpenAI-backed models, install the optional dependency and use an `OpenAI/`-prefixed model ID, for example `"OpenAI/gpt-4o"`.

## API Reference / Configuration

### Core estimator

`SemanticStateEstimator(domain, problem, vlm, ...)` is the main entry point.

Key arguments:

- `domain`, `problem`: PDDL domain and problem, provided either as strings or file paths.
- `vlm`: a `VLMBackend` instance or a model string. Strings prefixed with `OpenAI/` select the OpenAI backend; all other strings select the HuggingFace backend.
- `query_translator`: translation strategy used to convert grounded predicates into queries.
- `confidence`: default threshold used when converting probabilities into booleans.
- `multi_image_strategy`: either `"single"` or `"average"`.
- `probability_method`: either `"logprobs"` or `"text_match"`.
- `true_tokens`, `false_tokens`: optional token groups used for probability extraction.
- `batch_size`: number of predicate queries grouped into each backend batch.
- `user_prompt_template`: format string for each translated query; must contain `{query}`.
- `additional_instructions`: additional text appended to the system prompt.
- `vlm_kwargs`: keyword arguments forwarded when `vlm` is provided as a model string.

Common methods:

- `estimator(images) -> dict[str, bool]`: return a boolean symbolic state.
- `estimate_probabilities(images) -> dict[str, float]`: return per-predicate probabilities.
- `estimate_raw(images) -> dict[str, VLMOutput]`: return normalized backend outputs.
- `swap_problem(domain, problem)`: rebuild the estimator for a new planning problem.

### Translators

- `IdentityTranslator`: use grounded predicates as-is.
- `TemplateTranslator`: format grounded predicates with per-predicate templates.
- `PrewrittenTranslator`: provide explicit prompts for each grounded predicate.
- `LLMTranslator`: generate natural-language prompts with an LLM and optionally cache them.

### Environment variables and optional configuration

- `OPENAI_API_KEY`: required for `OpenAIVLM` and OpenAI-backed `LLMTranslator` usage.
- `cache_dir` on `LLMTranslator`: enables on-disk caching of generated predicate translations.

## Contributing

Install development dependencies:

```bash
pip install -e '.[dev]'
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
