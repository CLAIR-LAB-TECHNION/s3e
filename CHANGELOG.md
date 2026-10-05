# Changelog

All notable changes to `s3e` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [semantic versioning](https://semver.org/) (pre-1.0: minor versions may
contain breaking changes). Entries for 0.1.0–0.4.1 were reconstructed from
the commit history and PyPI release dates when this file was introduced.

## Unreleased

### Added
- JOSS submission materials: `paper/paper.md` and `paper/paper.bib`, with a
  GitHub Actions workflow that builds the draft PDF.
- Sphinx API reference under `docs/` (Read the Docs configuration in
  `.readthedocs.yaml`) and docstrings for the remaining public calibration,
  answer-space, and serialization APIs.
- Continuous integration running the fast test suite on Python 3.10–3.14 and
  building the documentation.
- Community files: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, issue and pull
  request templates, and `CITATION.cff`.
- `.mailmap` consolidating the maintainer's commit identities.
- `MANIFEST.in` so the source distribution ships the full test suite.
- `Prediction.has_answer` and `Prediction.matched`, and an
  `UnmatchedAnswerWarning` emitted once per `QueryEngine.ask` call when some
  replies fall outside the answer space.

### Changed
- **Breaking:** `to_state` and `SemanticStateEstimator.__call__` always return
  a complete `dict[str, bool]`: a predicate is True iff P(true) >= confidence,
  never `None`. `Prediction.answer` is never `None`.
- **Breaking:** a prediction has no answer when the explicit null option is
  its top answer or no answer token received any mass. Its P(true) is 0.5 (a
  uniform distribution for categorical spaces), even when calibrated, so it is
  True at the default confidence of 0.5 and False above it.
- **Breaking:** `text_match` scoring matches only when the reply *starts* with
  an answer token (as a whole word, longest token first); previously a token
  anywhere in the reply matched.
- `Prediction.probability` is `T / (T + F)` without smoothing, so a matched
  `text_match` reply gives exactly 1.0. Calibration scores are unchanged.

### Fixed
- `CalibrationSet.collect` skips predictions with no answer, which are not
  valid training points.
- `PredictionSet.average` averages calibrated probabilities only over members
  that have an answer.

## 0.4.1 — 2026-08-30

Architecture redesign into independently usable layers. (The version was
bumped to 0.4.0 during development, but 0.4.0 was never published to PyPI;
its changes ship in 0.4.1.)

### Added
- `QueryEngine`: images + free-form queries + an answer space → predictions,
  with no PDDL involved.
- Answer spaces (`BinaryAnswers`, `CategoricalAnswers`, `AnswerOption`) with
  multi-token surface forms and an optional explicit null option.
- Lazy, immutable `Prediction` / `PredictionSet` result objects with JSON
  round-tripping and `PredictionSet.average` for multi-view estimation.
- `s3e.calibration` package: `Calibrator` interface, `PlattCalibrator`, and
  `CalibrationSet` / `CalibrationExample` / `CalibrationSample`.
- Public `resolve_backend()` and helpful `ImportError`s that name the extra to
  install for each optional dependency.
- `dev-gpu` extra; contract tests for every backend and for downstream
  research workflows.

### Changed
- **Breaking:** `SemanticStateEstimator` is now a thin facade
  ("predicates in, state out"); build it from PDDL with `from_pddl` or from an
  explicit predicate list.
- **Breaking:** `s3e.vlm` renamed to `s3e.backends`; vLLM is selected by
  passing a `VLLMBackend` instance instead of a flag.
- Optional dependencies are tiered into extras; `import s3e` no longer pulls
  in torch, Unified Planning, scikit-learn, or openai.

### Removed
- **Breaking:** `StateEstimator`, `ProbabilisticStateEstimator`,
  `PredicatePredictionDetails`, and `PlattCalibrationSample`, superseded by
  `QueryEngine`, `Prediction` / `PredictionSet`, and `CalibrationSample`.
- **Breaking:** estimator methods `swap_problem`, `estimate_prediction_details`,
  `estimate_probabilities`, `estimate_raw`, and the `*_platt_scaling*` family,
  superseded by `set_problem`, `estimate`, and the `s3e.calibration` package.

### Fixed
- Many input-validation fixes: configuration is validated before any model is
  loaded; duplicate or empty answer tokens are rejected; overflow-safe Platt
  sigmoid; the estimator's problem is restored after
  `CalibrationSet.collect`; only boolean fluents are grounded as predicates.

## 0.3.2 — 2026-08-04

### Added
- `interest_tokens` contract for VLM backends: backends report probability
  mass for exactly the requested answer tokens, matched by token id in the
  HuggingFace and vLLM backends.

### Changed
- `s3e.__version__` is read from the installed package metadata.

### Fixed
- Padding issues in multi-prompt HuggingFace batches.

## 0.3.1 — 2026-07-20

### Changed
- Requesting vLLM together with an OpenAI model now warns and is ignored
  instead of raising.

## 0.3.0 — 2026-07-15

### Added
- `VLLMBackend` for local, single-node multi-GPU inference (requires
  `vllm>=0.11.0`, the `vllm` extra).

### Changed
- **Breaking:** estimator configuration arguments are keyword-only.
- HuggingFace backend: uses `dtype` instead of the deprecated `torch_dtype`,
  runs under `torch.inference_mode`, and keeps only next-token logits to save
  memory.

## 0.2.0 — 2026-05-12

### Added
- Batched HuggingFace inference for both log-probability queries and text
  generation.
- A calibration-data workflow: collect labeled scores once, save and load
  them, and fit Platt scaling from the saved data without querying the VLM.
- `s3e.__version__`.

### Changed
- The HuggingFace backend returns probabilities over the full vocabulary by
  default.

## 0.1.0 — 2026-05-07

First release on PyPI. Development started in August 2024 as the
implementation behind the S3E workshop paper (AAAI 2025 Workshop LM4Plan).
The repository became public at `github.com/guyazran/s3e` in February 2026
and moved to `github.com/CLAIR-LAB-TECHNION/s3e` in April 2026.

### Added
- `SemanticStateEstimator`: PDDL grounding (via Unified Planning), per-predicate
  probabilities from answer-token masses, prediction details, querying a subset
  of predicates, and averaging across scenes.
- HuggingFace and OpenAI VLM backends.
- Identity, template, prewritten, and LLM-generated predicate translation.
- Optional null-token abstention.
- Platt-scaling calibration (global or lifted scope) with save and load.
