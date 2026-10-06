# Developer guide

This page explains how S3E is put together and how to extend it. For the
development setup, test commands, style, and release process, see
[`CONTRIBUTING.md`](https://github.com/CLAIR-LAB-TECHNION/s3e/blob/main/CONTRIBUTING.md).

## Module map

| Module | Responsibility |
|---|---|
| `s3e/backends/` | The {class}`~s3e.VLMBackend` interface and {class}`~s3e.VLMOutput`; the HuggingFace, OpenAI, and vLLM backends; {func}`~s3e.resolve_backend` (model id → backend); a reverse token index shared by the local backends |
| `s3e/engine/` | {class}`~s3e.QueryEngine`; answer spaces ({class}`~s3e.BinaryAnswers`, {class}`~s3e.CategoricalAnswers`); results ({class}`~s3e.Prediction`, {class}`~s3e.PredictionSet`) |
| `s3e/calibration/` | The {class}`~s3e.Calibrator` interface, {class}`~s3e.PlattCalibrator`, and calibration data ({class}`~s3e.CalibrationSet`) |
| `s3e/translation/` | The {class}`~s3e.QueryTranslator` interface and its implementations; the JSON cache used by {class}`~s3e.LLMTranslator` |
| `s3e/pddl/` | Unified Planning helpers: parsing, grounding, state conversion, and the domain fingerprint stored with calibrators |
| `s3e/estimator.py` | {class}`~s3e.SemanticStateEstimator`, the state-estimation facade (predicates in, state out; PDDL via `from_pddl`) |
| `s3e/_deps.py` | `require(module, extra)`, which turns a missing optional dependency into an `ImportError` naming the extra to install |

Dependencies point downward: the estimator uses translation, the engine,
calibration, and the PDDL helpers; calibration and the engine use results and
answer spaces; nothing below the estimator knows about PDDL. The core imports
only NumPy, Pillow, and tqdm. Each optional dependency (torch, transformers,
openai, vllm, unified-planning, scikit-learn) is imported only by the module
that needs it, after `require()`; `tests/test_imports.py` checks this in fresh
subprocesses.

## What happens in one estimate

1. {meth}`SemanticStateEstimator.estimate <s3e.SemanticStateEstimator.estimate>`
   selects the requested predicates and looks up their queries, which the
   translator produced once, at construction or in `set_problem`. Predicates
   that share a query are asked once.
2. {meth}`QueryEngine.ask <s3e.QueryEngine.ask>` wraps each query in the
   prompt template and sends batches of `batch_size` prompts to the backend's
   `query_batch`, with the system prompt and, in `logprobs` mode, the answer
   space's interest tokens (in `text_match` mode, `generate=True` instead).
3. The answer space scores each `VLMOutput` into per-option masses, the null
   option's mass, and the unassigned remainder.
4. Each result becomes an immutable `Prediction`; probabilities, answers,
   and states are derived from the stored masses on first access. The engine
   emits one `UnmatchedAnswerWarning` per call if some queries matched no
   option.
5. With a calibrator, the estimator first checks the calibrator's recorded
   scoring mode, answer space, and domain fingerprint against its own, then
   returns `calibrator.apply(results)`.

## Extending S3E

### A new model backend

Subclass {class}`~s3e.VLMBackend` and implement `query`; override
`query_batch` when the model can batch, and `unsupported_interest_tokens`
when you can tell which answer tokens are not single vocabulary tokens.
The contract that matters most: when `interest_tokens` is given, `token_probs`
has exactly those keys (0.0 for tokens the model did not produce), and
`argmax_in_interest` says whether the model's most likely token was one of
them. `examples/custom_backend.py` is a complete, minimal backend.

`BackendContract` in `tests/backends/test_contract.py` encodes this
behavior. Subclass it in your backend's test module and provide a
`make_backend` fixture to run the whole contract against your backend.

### A new translator

Subclass {class}`~s3e.QueryTranslator` and implement
`translate(predicates, domain=None, problem=None)`, returning a dict from
each grounded predicate string to its query. Translators that work without
PDDL must accept `domain=None` and `problem=None`.

### A new answer space

Build an {class}`~s3e.AnswerSpace` from {class}`~s3e.AnswerOption` objects,
or use {class}`~s3e.CategoricalAnswers` with explicit options. Tokens must
not overlap between options. Serialization (`to_dict` / `from_dict`, used by
`PredictionSet` round-trips and calibrator metadata) supports the binary and
categorical spaces.

### A new calibrator

Subclass {class}`~s3e.Calibrator` and implement `apply`, `save`, and `load`.
`apply` must return a new `PredictionSet` and leave its input untouched;
`Prediction.with_probability` makes the calibrated copy of each prediction.
If the calibrator has a `meta` dict, the estimator compares any `scoring`,
`answers`, and `domain_fingerprint` it holds with its own before applying.

## Tests

- `tests/` mirrors the package layout. `tests/fakes.py` provides `FakeVLM`, a
  deterministic backend that satisfies the full backend contract, and
  `tests/conftest.py` holds shared fixtures and a Blocksworld domain.
- `pytest -m "not slow"` is the fast suite: no downloads, CPU only. It also
  runs the docstring examples in `s3e/` and the scripts in `examples/` (with
  fake backends).
- Tests marked `slow` download and run real models: backend integration
  tests, the README and getting-started code, and the walkthrough notebook. A
  scheduled workflow runs them weekly on CPU; the vLLM integration tests need
  a CUDA GPU.
- `tests/workflows/` holds end-to-end tests of usage patterns that span
  modules, such as sharing one backend between estimators or calibrating
  offline.

## Continuous integration

The `Tests` workflow runs on every pull request and push to `main`:

- the fast suite on Linux with Python 3.10–3.14, and on macOS and Windows with
  Python 3.10 and 3.14;
- the fast suite with every direct dependency at its declared minimum version;
- a coverage report over the whole package, including the vLLM backend;
- `ruff check .`, a strict documentation build, a packaging check, and
  validation of `CITATION.cff`.

The `Slow tests` workflow runs the real-model tests weekly and on demand, and
the `Draft PDF` workflow builds the JOSS paper when `paper/` changes.

## Documentation

The documentation is built with Sphinx from `docs/`: MyST Markdown pages,
autodoc API pages under `docs/api/`, and the walkthrough notebook (rendered
with its saved outputs). The statement of need and the user guide are
included from `README.md`, so edit them there. Build locally with:

```bash
pip install -e ".[docs]"
sphinx-build -W -b html docs docs/_build/html
```
