# Null Token Support Design

## Summary

Add optional null-token support to `SemanticStateEstimator` so the estimator can abstain with `None` when the model output indicates "not enough information", while preserving the existing binary probability API and the existing single-raw-call workflow for calibrated and uncalibrated probabilities.

## Context

`s3e/semantic_state_estimator.py` is currently binary:

- `estimate_probabilities()` returns binary `P(true)`.
- `probabilities_from_raw()` exists so callers can derive calibrated and uncalibrated binary probabilities from one `estimate_raw()` call.
- `__call__()` thresholds binary probabilities into booleans.
- Platt scaling operates only on true/false token groups and true/false ground-truth labels.

The new requirement is to support a third token group for "not enough information" without changing the meaning of `estimate_probabilities()` and without folding null labels into calibration.

## Goals

- Support an optional null token group, configured by the user.
- Allow `__call__()` to return `None` for predicates whose raw null mass is strictly larger than both raw true mass and raw false mass.
- Preserve the existing `estimate_raw()` -> `probabilities_from_raw()` workflow so callers can still derive calibrated and uncalibrated binary probabilities from the same VLM call.
- Expose a companion API that reports both the binary probability view and the raw null-signal metadata.
- Keep Platt scaling binary and compatible with existing saved profiles.

## Non-Goals

- Do not change the signature or meaning of `estimate_probabilities()`.
- Do not add null-valued labels to `CalibrationExample`.
- Do not train or serialize a three-way calibration model.
- Do not redefine calibration profile compatibility around null tokens.

## Chosen API

### Constructor

Add an optional constructor parameter:

- `null_tokens: list[str] | None = None`

`None` disables null-mode completely. In that case, the estimator behaves exactly as it does today.

### Existing Public Methods

These methods keep their current signatures and meanings:

- `estimate_raw(...) -> dict[str, VLMOutput]`
- `estimate_probabilities(...) -> dict[str, float]`
- `probabilities_from_raw(...) -> dict[str, float]`

`estimate_probabilities()` and `probabilities_from_raw()` continue to mean binary `P(true)` over the true/false token groups only. If calibration is enabled, they continue to return the calibrated binary probability.

### New Public Type

Add a public dataclass in `s3e/semantic_state_estimator.py` and re-export it from `s3e/__init__.py`:

- `PredicatePredictionDetails`

Planned fields:

- `probability: float`
- `raw_probability: float`
- `score: float`
- `raw_true_mass: float`
- `raw_false_mass: float`
- `raw_none_mass: float`
- `none_is_max_raw: bool`

Field meanings:

- `probability` is the binary `P(true)` selected by the `calibrated` flag.
- `raw_probability` is the uncalibrated binary `P(true)`.
- `score` is the existing true-vs-false grouped log-odds score used for Platt scaling.
- The raw mass fields expose grouped token mass before any calibration.
- `none_is_max_raw` is the abstain signal used by `__call__()`.

### New Public Methods

Add a structured companion API that mirrors the existing probability workflow:

- `estimate_prediction_details(...) -> dict[str, PredicatePredictionDetails]`
- `prediction_details_from_raw(...) -> dict[str, PredicatePredictionDetails]`

This preserves the current pattern:

1. call `estimate_raw(...)` once
2. derive as many views as needed from that raw output

That includes both:

- calibrated and uncalibrated binary probabilities through `probabilities_from_raw(...)`
- null-aware metadata through `prediction_details_from_raw(...)`

### `__call__()`

Change `SemanticStateEstimator.__call__()` to return `dict[str, bool | None]`.

Rule:

- return `None` when `none_is_max_raw` is `True`
- otherwise return `bool(details.probability >= threshold)`

The base estimator docstrings and type hints in `s3e/state_estimator.py` should be widened to allow `None`, while remaining compatible with implementations that still only return booleans.

## Computation Model

### `logprobs` mode

Introduce one shared internal pipeline that all public binary and structured methods use.

### Raw grouped masses

For each `VLMOutput`, compute:

- `raw_true_mass = sum(token_probs[token] for token in true_tokens)`
- `raw_false_mass = sum(token_probs[token] for token in false_tokens)`
- `raw_none_mass = sum(token_probs[token] for token in null_tokens)` when null-mode is enabled, otherwise `0.0`

### Binary probability

The binary probability remains defined only over the true/false token groups:

- `raw_total = raw_true_mass + raw_false_mass`
- if `raw_total == 0`, `raw_probability = 0.5`
- otherwise `raw_probability = raw_true_mass / raw_total`

The calibration score remains the current true-vs-false score:

- `score = grouped_log_odds(token_probs, true_tokens, false_tokens)`

Null mass does not participate in `raw_probability`, `score`, or Platt fitting.

### Abstain signal

The abstain signal is defined exactly as approved:

- `none_is_max_raw = raw_none_mass > raw_true_mass and raw_none_mass > raw_false_mass`

Ties do not abstain.

### Calibration

Calibration remains a second-stage projection on top of the binary true/false view only:

- if calibration is disabled, `probability = raw_probability`
- if calibration is enabled, `probability = calibrated_p_true`

Calibration never changes:

- `raw_true_mass`
- `raw_false_mass`
- `raw_none_mass`
- `none_is_max_raw`
- `raw_probability`

This keeps the calibrated/uncalibrated split aligned with the existing API design.

## `text_match` mode

`estimate_probabilities()` remains unchanged in meaning and behavior: it stays a binary projection, not a three-way API.

The companion details API treats generated text as a one-hot assignment over the configured token groups:

- if text matches a true token:
  - `raw_true_mass = 1.0`
  - `raw_false_mass = 0.0`
  - `raw_none_mass = 0.0`
  - `none_is_max_raw = False`
  - `raw_probability = 1.0`
- if text matches a false token:
  - `raw_true_mass = 0.0`
  - `raw_false_mass = 1.0`
  - `raw_none_mass = 0.0`
  - `none_is_max_raw = False`
  - `raw_probability = 0.0`
- if text matches a null token:
  - `raw_true_mass = 0.0`
  - `raw_false_mass = 0.0`
  - `raw_none_mass = 1.0`
  - `none_is_max_raw = True`
  - `raw_probability = 0.5`
- if text is missing or matches none of the configured groups:
  - all raw masses are `0.0`
  - `none_is_max_raw = False`
  - `raw_probability = 0.5`

Calibration remains unsupported for `text_match`, unchanged from today.

## Multi-Image Behavior

Preserve the current binary semantics for `multi_image_strategy="average"`:

- average per-image binary probabilities exactly as today
- average per-image calibration outputs exactly as today when calibration is enabled

For the new metadata:

- average `raw_true_mass`, `raw_false_mass`, and `raw_none_mass` across images
- average `raw_probability` across images
- average `score` across images
- compute `none_is_max_raw` from the averaged raw masses

This keeps the new abstain decision aligned with the existing averaged inference model.

## Calibration And Profile Compatibility

Platt scaling remains binary.

`CalibrationExample.state_dict` stays `dict[str, bool]`. Ground truth does not carry a null label.

No changes are planned for:

- `fit_platt_scaling(...)`
- calibration score computation
- calibration profile serialization format
- saved-profile schema version
- profile compatibility checks for true/false token groups

Changing `null_tokens` alone must not invalidate an otherwise compatible Platt profile, because null tokens do not participate in calibration.

## Validation And Error Handling

Add explicit validation that the configured token groups do not overlap:

- no token may appear in both true and false groups
- no token may appear in both null and true groups
- no token may appear in both null and false groups

Raise `ValueError` with a direct message when overlap exists.

Unknown predicates and all existing calibration-related errors should keep their current behavior.

## Internal Structure

Use one shared internal representation so the logic is not duplicated.

Planned layering:

1. raw details extraction from `VLMOutput`
2. optional calibration projection onto the binary probability
3. thin public projections:
   - `estimate_probabilities(...)`
   - `probabilities_from_raw(...)`
   - `estimate_prediction_details(...)`
   - `prediction_details_from_raw(...)`
   - `__call__(...)`

This preserves the original reason `probabilities_from_raw()` was added: callers can derive multiple views from one raw VLM query.

## Testing Plan

Update or add tests in `tests/test_semantic_state_estimator.py` and, if needed, `tests/test_calibration.py`.

Coverage should include:

- constructor support for `null_tokens`
- validation failure on overlapping token groups
- unchanged `estimate_probabilities()` behavior as a binary API, including when null tokens are configured
- unchanged `probabilities_from_raw()` behavior for calibrated and uncalibrated binary outputs from one raw VLM call, including when null tokens are configured
- new `prediction_details_from_raw()` behavior from one raw VLM call
- new `estimate_prediction_details()` convenience wrapper
- `__call__()` returning `None` only when raw null mass is strictly larger than raw true and raw false mass
- tie cases not abstaining
- `logprobs` details exposing raw grouped masses correctly
- `text_match` details treating true/false/null outputs as one-hot grouped masses
- `text_match` binary probability remaining unchanged
- multi-image averaging for both binary probability and null metadata
- calibration affecting only `probability`, not raw masses or `none_is_max_raw`
- compatibility of existing saved Platt profiles when null tokens are configured

## Implementation Notes

Keep the diff focused on the estimator API, shared extraction helpers, exports, and the directly affected tests. Do not expand this change into unrelated calibration refactors or broader API redesign.
