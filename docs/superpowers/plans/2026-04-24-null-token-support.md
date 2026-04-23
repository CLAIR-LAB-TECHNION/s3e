# Null Token Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional null-token support to `SemanticStateEstimator` so it can abstain with `None` while preserving the existing binary probability API and the single-raw-call calibrated/uncalibrated workflow.

**Architecture:** Replace the existing tuple-based extraction pipeline with a single shared `PredicatePredictionDetails` intermediate. Every public method projects from it. Keep calibration strictly binary over true/false token groups, expose null metadata beside the binary probability, and drive `__call__()` abstention from raw null dominance only.

**Key design principle:** One shared internal pipeline. `_extract_prediction_details()` is the single point of truth for converting a `VLMOutput` into structured details. All public methods — `estimate_probabilities()`, `probabilities_from_raw()`, `estimate_prediction_details()`, `prediction_details_from_raw()`, and `__call__()` — project from it. The old tuple-based methods (`_extract_probability`, `_extract_logprobs`, `_extract_text_match`, `_estimate_single`, `_estimate_average`, `_calibrate_prediction_details`) are deleted.

**Tech Stack:** Python 3.10+, pytest, NumPy, PIL fixtures, existing `s3e` calibration helpers

---

## File Map

- Modify: `s3e/semantic_state_estimator.py`
  Add `PredicatePredictionDetails`, `null_tokens`, token-group validation, `_extract_prediction_details()`, `prediction_details_from_raw()`, `estimate_prediction_details()`, `_average_prediction_details()`. Rewrite `probabilities_from_raw()`, `estimate_probabilities()`, and `__call__()` as thin projections from the details path. Update `_estimate_calibration_example()` and `fit_platt_scaling()` to use details. Delete `_extract_probability()`, `_extract_logprobs()`, `_extract_text_match()`, `_estimate_single()`, `_estimate_average()`, `_calibrate_prediction_details()`.
- Modify: `s3e/state_estimator.py`
  Widen the abstract/base `__call__()` return types and docs from `bool` to `bool | None` without changing the binary estimator fallback behavior.
- Modify: `s3e/__init__.py`
  Re-export `PredicatePredictionDetails` and update the quick-start comment to reflect `dict[str, bool | None]`.
- Modify: `tests/test_semantic_state_estimator.py`
  Add focused red/green coverage for null token configuration, raw details extraction, abstention behavior, binary calibration compatibility, `text_match`, and multi-image averaging.

`s3e/calibration.py` should stay untouched unless implementation reveals an unexpected coupling. The design explicitly keeps null tokens out of calibration fitting and profile serialization.

### Task 1: Add Raw Null-Token Details API

**Files:**
- Modify: `tests/test_semantic_state_estimator.py`
- Modify: `s3e/semantic_state_estimator.py`
- Modify: `s3e/__init__.py`

- [ ] **Step 1: Write the failing tests**

```python
from s3e import CalibrationExample, PredicatePredictionDetails


class TestNullTokenDetails:
    def test_custom_null_tokens(
        self, fake_vlm, blocksworld_domain, blocksworld_problem
    ):
        se = SemanticStateEstimator(
            blocksworld_domain,
            blocksworld_problem,
            vlm=fake_vlm,
            null_tokens=["null", "NULL"],
        )
        assert se.null_tokens == ["null", "NULL"]

    def test_overlapping_null_and_true_tokens_raise(
        self, fake_vlm, blocksworld_domain, blocksworld_problem
    ):
        with pytest.raises(ValueError, match="null_tokens and true_tokens"):
            SemanticStateEstimator(
                blocksworld_domain,
                blocksworld_problem,
                vlm=fake_vlm,
                true_tokens=["true"],
                false_tokens=["false"],
                null_tokens=["true"],
            )

    def test_prediction_details_from_raw_reports_raw_masses_and_none_flag(
        self, single_image, blocksworld_domain, blocksworld_problem
    ):
        vlm = FakeVLM(
            token_probs={"true": 0.2, "false": 0.1, "null": 0.6, "other": 0.1}
        )
        se = SemanticStateEstimator(
            blocksworld_domain,
            blocksworld_problem,
            vlm=vlm,
            true_tokens=["true"],
            false_tokens=["false"],
            null_tokens=["null"],
        )

        raw = se.estimate_raw(single_image, predicates=["on(a,b)"])
        details = se.prediction_details_from_raw(raw, calibrated=False)
        pred_details = details["on(a,b)"]

        assert isinstance(pred_details, PredicatePredictionDetails)
        assert pred_details.raw_true_mass == pytest.approx(0.2)
        assert pred_details.raw_false_mass == pytest.approx(0.1)
        assert pred_details.raw_none_mass == pytest.approx(0.6)
        assert pred_details.none_is_max_raw is True
        assert pred_details.raw_probability == pytest.approx(2.0 / 3.0)
        assert pred_details.probability == pytest.approx(2.0 / 3.0)

    def test_estimate_probabilities_ignores_null_mass(
        self, single_image, blocksworld_domain, blocksworld_problem
    ):
        vlm = FakeVLM(token_probs={"true": 0.2, "false": 0.1, "null": 0.6})
        se = SemanticStateEstimator(
            blocksworld_domain,
            blocksworld_problem,
            vlm=vlm,
            true_tokens=["true"],
            false_tokens=["false"],
            null_tokens=["null"],
        )

        probs = se.estimate_probabilities(single_image, calibrated=False)
        assert probs["on(a,b)"] == pytest.approx(2.0 / 3.0)
```

- [ ] **Step 2: Run the targeted tests to verify red**

Run:

```bash
pytest \
  tests/test_semantic_state_estimator.py::TestNullTokenDetails::test_custom_null_tokens \
  tests/test_semantic_state_estimator.py::TestNullTokenDetails::test_overlapping_null_and_true_tokens_raise \
  tests/test_semantic_state_estimator.py::TestNullTokenDetails::test_prediction_details_from_raw_reports_raw_masses_and_none_flag \
  tests/test_semantic_state_estimator.py::TestNullTokenDetails::test_estimate_probabilities_ignores_null_mass \
  -v
```

Expected: FAIL with `TypeError: SemanticStateEstimator.__init__() got an unexpected keyword argument 'null_tokens'` and `ImportError` for `PredicatePredictionDetails`.

- [ ] **Step 3: Write the minimal implementation**

In `s3e/semantic_state_estimator.py`, add the public dataclass, the new constructor argument, token-group validation, raw-mass extraction, and the raw-details public method:

```python
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class PredicatePredictionDetails:
    probability: float
    raw_probability: float
    score: float
    raw_true_mass: float
    raw_false_mass: float
    raw_none_mass: float
    none_is_max_raw: bool
```

Add `null_tokens` to `__init__`, after the existing `false_tokens` block:

```python
    null_tokens: list[str] | None = None,
```

At the end of the token-group section in `__init__`, add:

```python
    self.null_tokens = list(null_tokens or [])
    self._validate_token_groups()
```

Add the validation helper:

```python
def _validate_token_groups(self) -> None:
    overlaps = [
        ("true_tokens and false_tokens", set(self.true_tokens) & set(self.false_tokens)),
        ("null_tokens and true_tokens", set(self.null_tokens) & set(self.true_tokens)),
        ("null_tokens and false_tokens", set(self.null_tokens) & set(self.false_tokens)),
    ]
    for label, values in overlaps:
        if values:
            joined = ", ".join(sorted(values))
            raise ValueError(f"Token groups must be disjoint; overlap between {label}: {joined}")
```

Add the single shared extraction method that handles both `logprobs` and `text_match`. This method is self-contained — it does not delegate to any of the old extraction methods:

```python
def _extract_prediction_details(self, output: VLMOutput) -> PredicatePredictionDetails:
    if self.probability_method == "text_match":
        if output.text is None:
            return PredicatePredictionDetails(0.5, 0.5, 0.0, 0.0, 0.0, 0.0, False)
        text = output.text.strip().lower()
        true_lower = {t.lower() for t in self.true_tokens}
        probability = 1.0 if text in true_lower else 0.0
        return PredicatePredictionDetails(
            probability=probability,
            raw_probability=probability,
            score=0.0,
            raw_true_mass=0.0,
            raw_false_mass=0.0,
            raw_none_mass=0.0,
            none_is_max_raw=False,
        )

    raw_true_mass = sum(output.token_probs.get(tok, 0.0) for tok in self.true_tokens)
    raw_false_mass = sum(output.token_probs.get(tok, 0.0) for tok in self.false_tokens)
    raw_none_mass = sum(output.token_probs.get(tok, 0.0) for tok in self.null_tokens)
    raw_total = raw_true_mass + raw_false_mass
    raw_probability = 0.5 if raw_total == 0 else float(np.clip(raw_true_mass / raw_total, 0.0, 1.0))
    score = 0.0 if raw_total == 0 else grouped_log_odds(output.token_probs, self.true_tokens, self.false_tokens)
    return PredicatePredictionDetails(
        probability=raw_probability,
        raw_probability=raw_probability,
        score=score,
        raw_true_mass=raw_true_mass,
        raw_false_mass=raw_false_mass,
        raw_none_mass=raw_none_mass,
        none_is_max_raw=raw_none_mass > raw_true_mass and raw_none_mass > raw_false_mass,
    )
```

Add the structured details method with calibration support:

```python
def prediction_details_from_raw(
    self,
    raw_outputs: dict[str, VLMOutput],
    calibrated: bool | None = None,
) -> dict[str, PredicatePredictionDetails]:
    use_calibration = self._resolve_calibrated_flag(calibrated)
    details = {
        pred: self._extract_prediction_details(output)
        for pred, output in raw_outputs.items()
    }
    if not use_calibration:
        return details

    result: dict[str, PredicatePredictionDetails] = {}
    for pred, detail in details.items():
        calibrated_probability = self._apply_platt_profile(pred, detail.score)
        if calibrated_probability is None:
            result[pred] = detail
        else:
            result[pred] = replace(detail, probability=calibrated_probability)
    return result
```

Rewrite `probabilities_from_raw()` to project from the details path:

```python
def probabilities_from_raw(
    self,
    raw_outputs: dict[str, VLMOutput],
    calibrated: bool | None = None,
) -> dict[str, float]:
    """Derive probabilities from already-obtained raw VLM outputs.

    This avoids a second VLM invocation when you need both calibrated
    and uncalibrated probabilities for the same observation::

        raw = estimator.estimate_raw(images)
        uncalibrated = estimator.probabilities_from_raw(raw)
        calibrated = estimator.probabilities_from_raw(raw, calibrated=True)

    Args:
        raw_outputs: Mapping of grounded predicate strings to
            :class:`VLMOutput`, as returned by :meth:`estimate_raw`.
        calibrated: Whether to apply Platt scaling.  ``None`` auto-detects
            (apply if a profile is loaded), ``True`` requires a profile,
            ``False`` always returns uncalibrated probabilities.

    Returns:
        Mapping of grounded predicate strings to P(true).
    """
    details = self.prediction_details_from_raw(raw_outputs, calibrated=calibrated)
    return {pred: detail.probability for pred, detail in details.items()}
```

In `s3e/__init__.py`, re-export the new type:

```python
from .semantic_state_estimator import PredicatePredictionDetails, SemanticStateEstimator

__all__ = [
    "StateEstimator",
    "ProbabilisticStateEstimator",
    "SemanticStateEstimator",
    "PredicatePredictionDetails",
    "CalibrationExample",
    "VLMBackend",
    "VLMOutput",
    "HuggingFaceVLM",
    "OpenAIVLM",
    "QueryTranslator",
    "IdentityTranslator",
    "PrewrittenTranslator",
    "TemplateTranslator",
    "LLMTranslator",
]
```

- [ ] **Step 4: Run the targeted tests to verify green**

Run:

```bash
pytest \
  tests/test_semantic_state_estimator.py::TestNullTokenDetails::test_custom_null_tokens \
  tests/test_semantic_state_estimator.py::TestNullTokenDetails::test_overlapping_null_and_true_tokens_raise \
  tests/test_semantic_state_estimator.py::TestNullTokenDetails::test_prediction_details_from_raw_reports_raw_masses_and_none_flag \
  tests/test_semantic_state_estimator.py::TestNullTokenDetails::test_estimate_probabilities_ignores_null_mass \
  -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add s3e/semantic_state_estimator.py s3e/__init__.py tests/test_semantic_state_estimator.py
git commit -m "feat: add raw null-token prediction details"
```

### Task 2: Unify Pipeline And Add Null Abstention

This task completes the pipeline unification. After this task, every public method projects from `_extract_prediction_details()`. The old tuple-based extraction methods are deleted. `__call__()` gains abstention behavior for dominant null mass.

**Files:**
- Modify: `tests/test_semantic_state_estimator.py`
- Modify: `s3e/semantic_state_estimator.py`
- Modify: `s3e/state_estimator.py`

- [ ] **Step 1: Write the failing tests**

```python
class TestNullAwareCall:
    def test_call_returns_none_when_null_mass_is_strictly_largest(
        self, single_image, blocksworld_domain, blocksworld_problem
    ):
        vlm = FakeVLM(token_probs={"true": 0.2, "false": 0.1, "null": 0.6})
        se = SemanticStateEstimator(
            blocksworld_domain,
            blocksworld_problem,
            vlm=vlm,
            true_tokens=["true"],
            false_tokens=["false"],
            null_tokens=["null"],
        )

        state = se(single_image, predicates=["on(a,b)"])
        assert state == {"on(a,b)": None}

    def test_call_uses_threshold_when_null_mass_ties_true(
        self, single_image, blocksworld_domain, blocksworld_problem
    ):
        vlm = FakeVLM(token_probs={"true": 0.4, "false": 0.1, "null": 0.4})
        se = SemanticStateEstimator(
            blocksworld_domain,
            blocksworld_problem,
            vlm=vlm,
            confidence=0.5,
            true_tokens=["true"],
            false_tokens=["false"],
            null_tokens=["null"],
        )

        state = se(single_image, predicates=["on(a,b)"])
        assert state == {"on(a,b)": True}

    def test_estimate_prediction_details_matches_projection_from_raw(
        self, single_image, blocksworld_domain, blocksworld_problem
    ):
        vlm = FakeVLM(token_probs={"true": 0.6, "false": 0.2, "null": 0.1})
        se = SemanticStateEstimator(
            blocksworld_domain,
            blocksworld_problem,
            vlm=vlm,
            true_tokens=["true"],
            false_tokens=["false"],
            null_tokens=["null"],
        )

        direct = se.estimate_prediction_details(
            single_image,
            calibrated=False,
            predicates=["on(a,b)"],
        )
        raw = se.estimate_raw(single_image, predicates=["on(a,b)"])
        from_raw = se.prediction_details_from_raw(raw, calibrated=False)
        assert direct == from_raw
```

- [ ] **Step 2: Run the targeted tests to verify red**

Run:

```bash
pytest \
  tests/test_semantic_state_estimator.py::TestNullAwareCall::test_call_returns_none_when_null_mass_is_strictly_largest \
  tests/test_semantic_state_estimator.py::TestNullAwareCall::test_call_uses_threshold_when_null_mass_ties_true \
  tests/test_semantic_state_estimator.py::TestNullAwareCall::test_estimate_prediction_details_matches_projection_from_raw \
  -v
```

Expected: FAIL because `__call__()` still returns `bool` values and `estimate_prediction_details()` does not exist yet.

- [ ] **Step 3: Write the minimal implementation**

Add the multi-image averaging helper:

```python
def _average_prediction_details(
    self,
    per_image_details: list[dict[str, PredicatePredictionDetails]],
) -> dict[str, PredicatePredictionDetails]:
    predicates = list(per_image_details[0].keys())
    result: dict[str, PredicatePredictionDetails] = {}
    for pred in predicates:
        items = [details[pred] for details in per_image_details]
        raw_true_mass = float(np.mean([d.raw_true_mass for d in items]))
        raw_false_mass = float(np.mean([d.raw_false_mass for d in items]))
        raw_none_mass = float(np.mean([d.raw_none_mass for d in items]))
        result[pred] = PredicatePredictionDetails(
            probability=float(np.mean([d.probability for d in items])),
            raw_probability=float(np.mean([d.raw_probability for d in items])),
            score=float(np.mean([d.score for d in items])),
            raw_true_mass=raw_true_mass,
            raw_false_mass=raw_false_mass,
            raw_none_mass=raw_none_mass,
            none_is_max_raw=raw_none_mass > raw_true_mass and raw_none_mass > raw_false_mass,
        )
    return result
```

Add the convenience wrapper:

```python
def estimate_prediction_details(
    self,
    images: list[Image],
    calibrated: bool | None = None,
    predicates: list[str] | None = None,
) -> dict[str, PredicatePredictionDetails]:
    if self.multi_image_strategy != "average":
        raw = self.estimate_raw(images, predicates=predicates)
        return self.prediction_details_from_raw(raw, calibrated=calibrated)

    per_image = [
        self.prediction_details_from_raw(
            self.estimate_raw([img], predicates=predicates),
            calibrated=calibrated,
        )
        for img in images
    ]
    return self._average_prediction_details(per_image)
```

Rewrite `estimate_probabilities()` as a thin projection from the details path:

```python
def estimate_probabilities(
    self,
    images: list[Image],
    calibrated: bool | None = None,
    predicates: list[str] | None = None,
) -> dict[str, float]:
    """Estimate P(true) for each grounded predicate.

    Args:
        images: List of PIL images representing the current state.
        calibrated: Whether to apply Platt scaling.
        predicates: Optional list of grounded predicate strings to
            query.  When ``None`` (default), all predicates are
            queried.  Unknown predicates raise :class:`ValueError`.
    """
    details = self.estimate_prediction_details(
        images, calibrated=calibrated, predicates=predicates
    )
    return {pred: detail.probability for pred, detail in details.items()}
```

Rewrite `__call__()` to use the details path with abstention:

```python
def __call__(
    self,
    images: list[Image],
    confidence: float | None = None,
    calibrated: bool | None = None,
    predicates: list[str] | None = None,
) -> dict[str, bool | None]:
    details = self.estimate_prediction_details(
        images,
        calibrated=calibrated,
        predicates=predicates,
    )
    threshold = confidence if confidence is not None else self.confidence
    return {
        pred: None if detail.none_is_max_raw else bool(detail.probability >= threshold)
        for pred, detail in details.items()
    }
```

Update `_estimate_calibration_example()` to use the details path:

```python
def _estimate_calibration_example(
    self,
    example: CalibrationExample,
) -> list[dict[str, PredicatePredictionDetails]]:
    original_problem = self._problem
    try:
        if example.problem is not None:
            self.swap_problem(self._domain, example.problem)
        unknown = set(example.state_dict.keys()) - set(self.queries_dict)
        if unknown:
            raise ValueError(
                f"Calibration example contains predicate(s) not in the "
                f"current problem: {', '.join(sorted(unknown))}"
            )
        labeled = list(example.state_dict.keys())
        if self.multi_image_strategy == "average":
            return [
                self.prediction_details_from_raw(
                    self.estimate_raw([image], predicates=labeled),
                    calibrated=False,
                )
                for image in example.images
            ]
        return [
            self.prediction_details_from_raw(
                self.estimate_raw(example.images, predicates=labeled),
                calibrated=False,
            )
        ]
    finally:
        if example.problem is not None:
            self.swap_problem(self._domain, original_problem)
```

Update `fit_platt_scaling()` to read `.score` from the details objects. Change the inner loop from:

```python
for predicate, (_, score) in details.items():
```

to:

```python
for predicate, detail in details.items():
```

and replace `score` with `detail.score` in the body.

Delete the old tuple-based methods that are no longer called:

- `_extract_probability()`
- `_extract_logprobs()`
- `_extract_text_match()`
- `_estimate_single()`
- `_estimate_average()`
- `_calibrate_prediction_details()`

In `s3e/state_estimator.py`, widen the types and docs without changing the default thresholding behavior of the base class:

```python
@abstractmethod
def __call__(self, images: list[Image]) -> dict[str, bool | None]:
    """Estimate the current state from images.

    Returns:
        Dictionary mapping predicate strings to boolean values, or
        ``None`` when the estimator abstains.
    """
```

```python
def __call__(
    self, images: list[Image], confidence: float | None = None
) -> dict[str, bool | None]:
    probs = self.estimate_probabilities(images)
    threshold = confidence if confidence is not None else self.confidence
    return {pred: bool(prob >= threshold) for pred, prob in probs.items()}
```

- [ ] **Step 4: Run the targeted tests to verify green, then run full regression**

Run the new tests:

```bash
pytest \
  tests/test_semantic_state_estimator.py::TestNullAwareCall::test_call_returns_none_when_null_mass_is_strictly_largest \
  tests/test_semantic_state_estimator.py::TestNullAwareCall::test_call_uses_threshold_when_null_mass_ties_true \
  tests/test_semantic_state_estimator.py::TestNullAwareCall::test_estimate_prediction_details_matches_projection_from_raw \
  -v
```

Expected: PASS.

Then run the full non-slow suite to verify the pipeline unification preserved all existing behavior:

```bash
pytest -m "not slow" -v
```

Expected: all existing tests pass. This confirms that the old tuple-based methods were safely replaced.

- [ ] **Step 5: Commit**

```bash
git add s3e/semantic_state_estimator.py s3e/state_estimator.py tests/test_semantic_state_estimator.py
git commit -m "feat: unify pipeline through details path and add null abstention"
```

### Task 3: Preserve Binary Calibration And Single-Raw-Call Reuse

**Files:**
- Modify: `tests/test_semantic_state_estimator.py`
- Modify: `s3e/semantic_state_estimator.py`

- [ ] **Step 1: Write the failing tests**

```python
class TestNullTokenCalibrationCompatibility:
    def test_prediction_details_from_raw_supports_both_binary_views(
        self, single_image, blocksworld_domain, blocksworld_problem
    ):
        vlm = FakeVLM(token_probs={"true": 0.8, "false": 0.2, "null": 0.6})
        se = SemanticStateEstimator(
            blocksworld_domain,
            blocksworld_problem,
            vlm=vlm,
            true_tokens=["true"],
            false_tokens=["false"],
            null_tokens=["null"],
        )
        se._platt_scaling_profile = PlattScalingProfile(
            scope="global",
            probability_method="logprobs",
            true_tokens=["true"],
            false_tokens=["false"],
            domain_fingerprint="irrelevant-for-this-test",
            score_kind="grouped_log_odds",
            groups={
                GLOBAL_CALIBRATION_KEY: PlattParameters(
                    a=2.0,
                    b=0.0,
                    sample_count=8,
                    positive_count=4,
                    negative_count=4,
                )
            },
        )

        raw = se.estimate_raw(single_image, predicates=["on(a,b)"])
        uncalibrated = se.prediction_details_from_raw(raw, calibrated=False)["on(a,b)"]
        calibrated = se.prediction_details_from_raw(raw, calibrated=True)["on(a,b)"]

        assert calibrated.probability != pytest.approx(uncalibrated.probability)
        assert calibrated.raw_probability == pytest.approx(uncalibrated.raw_probability)
        assert calibrated.raw_none_mass == pytest.approx(uncalibrated.raw_none_mass)
        assert calibrated.none_is_max_raw is uncalibrated.none_is_max_raw

    def test_probabilities_from_raw_matches_estimate_probabilities_with_null_tokens(
        self, single_image, blocksworld_domain, blocksworld_problem
    ):
        vlm = FakeVLM(token_probs={"true": 0.8, "false": 0.2, "null": 0.6})
        se = SemanticStateEstimator(
            blocksworld_domain,
            blocksworld_problem,
            vlm=vlm,
            true_tokens=["true"],
            false_tokens=["false"],
            null_tokens=["null"],
        )

        raw = se.estimate_raw(single_image)
        from_raw = se.probabilities_from_raw(raw, calibrated=False)
        direct = se.estimate_probabilities(single_image, calibrated=False)
        assert from_raw == direct

    def test_load_platt_scaling_ignores_null_tokens(
        self, tmp_path, blocksworld_domain, blocksworld_problem
    ):
        path = save_global_platt_profile(tmp_path, blocksworld_domain, blocksworld_problem)
        se = SemanticStateEstimator(
            blocksworld_domain,
            blocksworld_problem,
            vlm=FakeVLM(),
            null_tokens=["null"],
        )

        se.load_platt_scaling(path)
        probs = se.estimate_probabilities(make_calibration_image(1), calibrated=True)
        assert "on(a,b)" in probs
```

- [ ] **Step 2: Run the targeted tests to verify red**

Run:

```bash
pytest \
  tests/test_semantic_state_estimator.py::TestNullTokenCalibrationCompatibility::test_prediction_details_from_raw_supports_both_binary_views \
  tests/test_semantic_state_estimator.py::TestNullTokenCalibrationCompatibility::test_probabilities_from_raw_matches_estimate_probabilities_with_null_tokens \
  tests/test_semantic_state_estimator.py::TestNullTokenCalibrationCompatibility::test_load_platt_scaling_ignores_null_tokens \
  -v
```

Expected: These tests should PASS immediately because Tasks 1 and 2 already built all the necessary infrastructure. If they pass, there is no Step 3 implementation needed — move directly to Step 4.

If any test fails, investigate and fix the details/calibration interaction in `prediction_details_from_raw()`. The `replace(detail, probability=calibrated_probability)` pattern should already preserve raw masses and `none_is_max_raw`.

- [ ] **Step 3: Verify green**

Run:

```bash
pytest \
  tests/test_semantic_state_estimator.py::TestNullTokenCalibrationCompatibility::test_prediction_details_from_raw_supports_both_binary_views \
  tests/test_semantic_state_estimator.py::TestNullTokenCalibrationCompatibility::test_probabilities_from_raw_matches_estimate_probabilities_with_null_tokens \
  tests/test_semantic_state_estimator.py::TestNullTokenCalibrationCompatibility::test_load_platt_scaling_ignores_null_tokens \
  -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_semantic_state_estimator.py
git commit -m "test: verify binary calibration compatibility with null tokens"
```

### Task 4: Support `text_match` And Multi-Image Null Details

**Files:**
- Modify: `tests/test_semantic_state_estimator.py`
- Modify: `s3e/semantic_state_estimator.py`

- [ ] **Step 1: Write the failing tests**

```python
class TestNullTokenTextMatch:
    def test_text_match_null_output_sets_full_none_mass(
        self, single_image, blocksworld_domain, blocksworld_problem
    ):
        se = SemanticStateEstimator(
            blocksworld_domain,
            blocksworld_problem,
            vlm=FakeVLM(text="null"),
            probability_method="text_match",
            true_tokens=["true"],
            false_tokens=["false"],
            null_tokens=["null"],
        )

        details = se.estimate_prediction_details(single_image, predicates=["on(a,b)"])
        pred_details = details["on(a,b)"]
        assert pred_details.raw_true_mass == pytest.approx(0.0)
        assert pred_details.raw_false_mass == pytest.approx(0.0)
        assert pred_details.raw_none_mass == pytest.approx(1.0)
        assert pred_details.none_is_max_raw is True
        assert pred_details.raw_probability == pytest.approx(0.5)
        assert pred_details.probability == pytest.approx(0.5)

    def test_text_match_null_output_keeps_binary_probability_api(
        self, single_image, blocksworld_domain, blocksworld_problem
    ):
        se = SemanticStateEstimator(
            blocksworld_domain,
            blocksworld_problem,
            vlm=FakeVLM(text="null"),
            probability_method="text_match",
            true_tokens=["true"],
            false_tokens=["false"],
            null_tokens=["null"],
        )

        probs = se.estimate_probabilities(single_image, predicates=["on(a,b)"])
        assert probs == {"on(a,b)": pytest.approx(0.5)}


class TestNullTokenAverageStrategy:
    def test_estimate_prediction_details_averages_raw_masses(
        self, blocksworld_domain, blocksworld_problem
    ):
        image_one = Image.new("RGB", (64, 64))
        image_two = Image.new("RGB", (64, 64))

        class ImageAwareVLM(FakeVLM):
            def __init__(self):
                super().__init__()
                self.token_probs_by_image_id = {}

            def query(
                self,
                images,
                prompt,
                system_prompt=None,
                generate=False,
                **inference_kwargs,
            ):
                del prompt
                del system_prompt
                del generate
                del inference_kwargs
                return VLMOutput(
                    token_probs=self.token_probs_by_image_id[id(images[0])]
                )

        vlm = ImageAwareVLM()
        vlm.token_probs_by_image_id[id(image_one)] = {
            "true": 0.2,
            "false": 0.1,
            "null": 0.7,
        }
        vlm.token_probs_by_image_id[id(image_two)] = {
            "true": 0.3,
            "false": 0.2,
            "null": 0.1,
        }

        se = SemanticStateEstimator(
            blocksworld_domain,
            blocksworld_problem,
            vlm=vlm,
            multi_image_strategy="average",
            true_tokens=["true"],
            false_tokens=["false"],
            null_tokens=["null"],
        )

        details = se.estimate_prediction_details([image_one, image_two], predicates=["on(a,b)"])
        pred_details = details["on(a,b)"]

        assert pred_details.raw_true_mass == pytest.approx(0.25)
        assert pred_details.raw_false_mass == pytest.approx(0.15)
        assert pred_details.raw_none_mass == pytest.approx(0.4)
        assert pred_details.none_is_max_raw is True
        assert pred_details.raw_probability == pytest.approx(((0.2 / 0.3) + (0.3 / 0.5)) / 2.0)
```

- [ ] **Step 2: Run the targeted tests to verify red**

Run:

```bash
pytest \
  tests/test_semantic_state_estimator.py::TestNullTokenTextMatch::test_text_match_null_output_sets_full_none_mass \
  tests/test_semantic_state_estimator.py::TestNullTokenTextMatch::test_text_match_null_output_keeps_binary_probability_api \
  tests/test_semantic_state_estimator.py::TestNullTokenAverageStrategy::test_estimate_prediction_details_averages_raw_masses \
  -v
```

Expected: FAIL because `text_match` details are still binary-only (no null-aware one-hot grouping). The multi-image averaging test should PASS because `_average_prediction_details()` already handles null masses from Task 2 — if it passes, that is fine; it was included here to confirm coverage.

- [ ] **Step 3: Write the minimal implementation**

Replace the text_match branch in `_extract_prediction_details()` with a dedicated null-aware one-hot method:

```python
def _extract_text_match_details(self, output: VLMOutput) -> PredicatePredictionDetails:
    if output.text is None:
        return PredicatePredictionDetails(
            probability=0.5,
            raw_probability=0.5,
            score=0.0,
            raw_true_mass=0.0,
            raw_false_mass=0.0,
            raw_none_mass=0.0,
            none_is_max_raw=False,
        )

    text = output.text.strip().lower()
    true_lower = {t.lower() for t in self.true_tokens}
    false_lower = {t.lower() for t in self.false_tokens}
    none_lower = {t.lower() for t in self.null_tokens}

    if text in true_lower:
        return PredicatePredictionDetails(1.0, 1.0, 0.0, 1.0, 0.0, 0.0, False)
    if text in false_lower:
        return PredicatePredictionDetails(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, False)
    if text in none_lower:
        return PredicatePredictionDetails(0.5, 0.5, 0.0, 0.0, 0.0, 1.0, True)
    return PredicatePredictionDetails(0.5, 0.5, 0.0, 0.0, 0.0, 0.0, False)
```

Update `_extract_prediction_details()` to delegate:

```python
def _extract_prediction_details(self, output: VLMOutput) -> PredicatePredictionDetails:
    if self.probability_method == "text_match":
        return self._extract_text_match_details(output)
    # ... logprobs path unchanged ...
```

- [ ] **Step 4: Run the targeted tests to verify green**

Run:

```bash
pytest \
  tests/test_semantic_state_estimator.py::TestNullTokenTextMatch::test_text_match_null_output_sets_full_none_mass \
  tests/test_semantic_state_estimator.py::TestNullTokenTextMatch::test_text_match_null_output_keeps_binary_probability_api \
  tests/test_semantic_state_estimator.py::TestNullTokenAverageStrategy::test_estimate_prediction_details_averages_raw_masses \
  -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add s3e/semantic_state_estimator.py tests/test_semantic_state_estimator.py
git commit -m "feat: support null details in text-match and average modes"
```

### Task 5: Polish Public Docs And Run Full Verification

**Files:**
- Modify: `s3e/__init__.py`
- Modify: `s3e/semantic_state_estimator.py`
- Modify: `s3e/state_estimator.py`

- [ ] **Step 1: Update the public docstrings and quick-start comments**

Refresh the public docs so they describe the new abstention mode without changing the binary probability contract:

```python
"""s3e — Semantic State Estimation using vision-language models.

Quick start::

    from s3e import SemanticStateEstimator

    se = SemanticStateEstimator(domain_pddl, problem_pddl, vlm="Qwen/Qwen2-VL-7B-Instruct")
    state = se(images)  # dict[str, bool | None]
"""
```

```python
class SemanticStateEstimator(ProbabilisticStateEstimator):
    """Vision-language model based state estimator.

    Args:
        domain: PDDL domain as a file path or string.
        problem: PDDL problem as a file path or string.
        vlm: A :class:`VLMBackend` instance, or a model ID string.
        query_translator: Strategy for converting predicates to queries.
        true_tokens: Token strings representing "true".
        false_tokens: Token strings representing "false".
        null_tokens: Token strings representing "not enough information".
            When configured, :meth:`__call__` may return ``None`` for
            predicates whose raw null token mass is strictly larger than
            the raw true and false token masses.
    """
```

```python
def estimate_probabilities(
    self,
    images: list[Image],
    calibrated: bool | None = None,
    predicates: list[str] | None = None,
) -> dict[str, float]:
    """Estimate binary P(true) for each grounded predicate.

    Null-token mass is available through :meth:`estimate_prediction_details`
    and does not change the meaning of this method.
    """
```

- [ ] **Step 2: Run focused regression tests for all new behavior**

Run:

```bash
pytest \
  tests/test_semantic_state_estimator.py::TestNullTokenDetails \
  tests/test_semantic_state_estimator.py::TestNullAwareCall \
  tests/test_semantic_state_estimator.py::TestNullTokenCalibrationCompatibility \
  tests/test_semantic_state_estimator.py::TestNullTokenTextMatch \
  tests/test_semantic_state_estimator.py::TestNullTokenAverageStrategy \
  -v
```

Expected: PASS.

- [ ] **Step 3: Run the fast full verification loop**

Run:

```bash
python -m compileall s3e tests
pytest -m "not slow" -v
```

Expected: `compileall` exits successfully without syntax errors, and `pytest -m "not slow" -v` exits successfully with all non-slow tests passing.

- [ ] **Step 4: Commit the final polish**

```bash
git add s3e/__init__.py s3e/semantic_state_estimator.py s3e/state_estimator.py tests/test_semantic_state_estimator.py
git commit -m "docs: clarify null-token abstention behavior"
```

## Self-Review

### Spec coverage

- `null_tokens` constructor support: Task 1
- `__call__()` returning `None`: Task 2
- preserve `estimate_raw()` -> `probabilities_from_raw()` reuse: Tasks 1 and 3
- companion details API: Tasks 1 and 2
- calibration stays binary: Task 3
- `text_match` one-hot null behavior: Task 4
- multi-image averaged raw null metadata: Task 4
- updated public docs/types: Tasks 2 and 5
- one shared internal pipeline: Task 2 (deletes old tuple-based methods, all public methods project from `_extract_prediction_details`)

No spec requirement is left without a task.

### Pipeline unification checklist

After Task 2 completes, the following old methods must be deleted:

- `_extract_probability()` — replaced by `_extract_prediction_details()`
- `_extract_logprobs()` — inlined into `_extract_prediction_details()`
- `_extract_text_match()` — inlined into `_extract_prediction_details()`, later replaced by `_extract_text_match_details()` in Task 4
- `_estimate_single()` — replaced by `prediction_details_from_raw(estimate_raw(...))`
- `_estimate_average()` — replaced by `_average_prediction_details()`
- `_calibrate_prediction_details()` — replaced by calibration logic in `prediction_details_from_raw()`

And the following methods must be rewritten to project from the details path:

- `estimate_probabilities()` — projects `detail.probability` from `estimate_prediction_details()`
- `probabilities_from_raw()` — projects `detail.probability` from `prediction_details_from_raw()`
- `__call__()` — projects from `estimate_prediction_details()` with abstention
- `_estimate_calibration_example()` — returns `list[dict[str, PredicatePredictionDetails]]`
- `fit_platt_scaling()` — reads `detail.score` instead of tuple index

### Placeholder scan

The plan contains exact file paths, concrete test names, concrete commands, and concrete code blocks. There are no `TODO`, `TBD`, or "similar to above" placeholders.

### Type consistency

The plan uses one consistent public details type name, `PredicatePredictionDetails`, and one consistent companion API pair, `estimate_prediction_details()` / `prediction_details_from_raw()`. The widened public state type is consistently `dict[str, bool | None]`.
