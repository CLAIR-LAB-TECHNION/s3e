# Null Token Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional null-token support to `SemanticStateEstimator` so it can abstain with `None` while preserving the existing binary probability API and the single-raw-call calibrated/uncalibrated workflow.

**Architecture:** Introduce a shared `PredicatePredictionDetails` intermediate object inside `SemanticStateEstimator` and make every public view project from it. Keep calibration strictly binary over true/false token groups, expose null metadata beside the binary probability, and drive `__call__()` abstention from raw null dominance only.

**Tech Stack:** Python 3.10+, pytest, NumPy, PIL fixtures, existing `s3e` calibration helpers

---

## File Map

- Modify: `s3e/semantic_state_estimator.py`
  Add `PredicatePredictionDetails`, `null_tokens`, token-group validation, raw-details extraction, the new companion details APIs, and the null-aware `__call__()` logic.
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

Expected: FAIL with `TypeError: SemanticStateEstimator.__init__() got an unexpected keyword argument 'null_tokens'` and `AttributeError: 'SemanticStateEstimator' object has no attribute 'prediction_details_from_raw'`.

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

```python
def __init__(
    self,
    domain: str,
    problem: str,
    vlm: Union[VLMBackend, str],
    query_translator: QueryTranslator | None = None,
    system_prompt: str | None = None,
    user_prompt_template: str | None = None,
    true_tokens: list[str] | None = None,
    false_tokens: list[str] | None = None,
    null_tokens: list[str] | None = None,
    confidence: float = 0.5,
    multi_image_strategy: str = "single",
    probability_method: str = "logprobs",
    batch_size: int = 8,
    additional_instructions: str | None = None,
    vlm_kwargs: dict | None = None,
    inference_kwargs: dict | None = None,
):
    super().__init__(domain, problem, confidence)
    if isinstance(vlm, str):
        self.vlm = self._build_vlm_from_string(vlm, vlm_kwargs or {})
    else:
        self.vlm = vlm
    self.inference_kwargs = inference_kwargs or {}
    self.query_translator = query_translator or IdentityTranslator()
    has_nl_translator = not isinstance(self.query_translator, IdentityTranslator)
    if true_tokens is not None:
        self.true_tokens = true_tokens
    else:
        self.true_tokens = (
            list(TRUE_TOKENS_WITH_TRANSLATION)
            if has_nl_translator
            else list(TRUE_TOKENS_NO_TRANSLATION)
        )
    if false_tokens is not None:
        self.false_tokens = false_tokens
    else:
        self.false_tokens = (
            list(FALSE_TOKENS_WITH_TRANSLATION)
            if has_nl_translator
            else list(FALSE_TOKENS_NO_TRANSLATION)
        )
    self.null_tokens = list(null_tokens or [])
    self._validate_token_groups()
```

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


def _extract_prediction_details(self, output: VLMOutput) -> PredicatePredictionDetails:
    if self.probability_method == "text_match":
        probability = self._extract_text_match(output)
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

Keep `estimate_probabilities()` binary by projecting from the new details map:

```python
def probabilities_from_raw(
    self,
    raw_outputs: dict[str, VLMOutput],
    calibrated: bool | None = None,
) -> dict[str, float]:
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

### Task 2: Make `__call__()` Abstain On Dominant Null Mass

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

In `s3e/semantic_state_estimator.py`, add the convenience wrapper and switch `__call__()` to the details path:

```python
def estimate_prediction_details(
    self,
    images: list[Image],
    calibrated: bool | None = None,
    predicates: list[str] | None = None,
) -> dict[str, PredicatePredictionDetails]:
    raw = self.estimate_raw(images, predicates=predicates)
    return self.prediction_details_from_raw(raw, calibrated=calibrated)
```

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

- [ ] **Step 4: Run the targeted tests to verify green**

Run:

```bash
pytest \
  tests/test_semantic_state_estimator.py::TestNullAwareCall::test_call_returns_none_when_null_mass_is_strictly_largest \
  tests/test_semantic_state_estimator.py::TestNullAwareCall::test_call_uses_threshold_when_null_mass_ties_true \
  tests/test_semantic_state_estimator.py::TestNullAwareCall::test_estimate_prediction_details_matches_projection_from_raw \
  -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add s3e/semantic_state_estimator.py s3e/state_estimator.py tests/test_semantic_state_estimator.py
git commit -m "feat: abstain when null token mass dominates"
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

Expected: FAIL because the details path does not yet preserve both calibrated and uncalibrated binary views while keeping raw masses stable.

- [ ] **Step 3: Write the minimal implementation**

Keep calibration binary-only and apply it by replacing only the `probability` field in the details object:

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

Do not change `fit_platt_scaling()`, profile serialization, or `_validate_platt_profile()` beyond ensuring they continue to compare only `true_tokens` and `false_tokens`.

- [ ] **Step 4: Run the targeted tests to verify green**

Run:

```bash
pytest \
  tests/test_semantic_state_estimator.py::TestNullTokenCalibrationCompatibility::test_prediction_details_from_raw_supports_both_binary_views \
  tests/test_semantic_state_estimator.py::TestNullTokenCalibrationCompatibility::test_probabilities_from_raw_matches_estimate_probabilities_with_null_tokens \
  tests/test_semantic_state_estimator.py::TestNullTokenCalibrationCompatibility::test_load_platt_scaling_ignores_null_tokens \
  -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add s3e/semantic_state_estimator.py tests/test_semantic_state_estimator.py
git commit -m "feat: preserve binary calibration with null-token metadata"
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

Expected: FAIL because `text_match` details are still binary-only and `estimate_prediction_details()` does not yet average structured details across images.

- [ ] **Step 3: Write the minimal implementation**

Teach the shared extraction path about one-hot `text_match` grouping and average structured details instead of just `(probability, score)` tuples:

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

```python
def _extract_prediction_details(self, output: VLMOutput) -> PredicatePredictionDetails:
    if self.probability_method == "text_match":
        return self._extract_text_match_details(output)
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

```python
def _average_prediction_details(
    self,
    per_image_details: list[dict[str, PredicatePredictionDetails]],
) -> dict[str, PredicatePredictionDetails]:
    predicates = list(per_image_details[0].keys())
    result: dict[str, PredicatePredictionDetails] = {}
    for pred in predicates:
        raw_true_mass = float(np.mean([details[pred].raw_true_mass for details in per_image_details]))
        raw_false_mass = float(np.mean([details[pred].raw_false_mass for details in per_image_details]))
        raw_none_mass = float(np.mean([details[pred].raw_none_mass for details in per_image_details]))
        result[pred] = PredicatePredictionDetails(
            probability=float(np.mean([details[pred].probability for details in per_image_details])),
            raw_probability=float(np.mean([details[pred].raw_probability for details in per_image_details])),
            score=float(np.mean([details[pred].score for details in per_image_details])),
            raw_true_mass=raw_true_mass,
            raw_false_mass=raw_false_mass,
            raw_none_mass=raw_none_mass,
            none_is_max_raw=raw_none_mass > raw_true_mass and raw_none_mass > raw_false_mass,
        )
    return result


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

No spec requirement is left without a task.

### Placeholder scan

The plan contains exact file paths, concrete test names, concrete commands, and concrete code blocks. There are no `TODO`, `TBD`, or “similar to above” placeholders.

### Type consistency

The plan uses one consistent public details type name, `PredicatePredictionDetails`, and one consistent companion API pair, `estimate_prediction_details()` / `prediction_details_from_raw()`. The widened public state type is consistently `dict[str, bool | None]`.
