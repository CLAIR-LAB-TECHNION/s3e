# S3E Architecture Redesign — Design Spec

**Date:** 2026-08-28
**Status:** Approved design, pending implementation plan
**Scope:** Sub-project 1 of 4 (architecture). CI/tooling, docs/community, and release/publication are separate follow-up sub-projects.

## Goals

Turn `s3e` into a codebase fit for a top-tier open-source research package
(JOSS-checklist-ready, demo-track-ready) by restructuring it into lean,
composable layers. The public API may break freely (0.3.x → next minor/major);
it is optimized for **human clarity**: a reader should be able to read a
10-line example and correctly guess how the rest of the library works.

Non-goals for this sub-project: CI/lint/typing setup, docs site, PyPI/Zenodo
release, JOSS paper, demo. Those build on the finished API.

## Design principles

1. **Layered composition.** Concentric layers, each usable on its own.
   Dependency direction is strictly downward; no layer knows about the layers
   above it.
2. **Named objects over flag soup.** Every capability a user can turn on or
   off is a class they can import. Constructor calls read as sentences.
   Config-style kwargs (`probability_method=`, `use_vllm=`,
   `multi_image_strategy=`) are replaced by objects or methods with names.
3. **Store little, derive lazily.** Results store raw probability masses only;
   everything else (argmax, calibrated values, null domination) is computed on
   demand and cached. Laziness never re-runs inference.
4. **Split inference from interpretation.** The expensive VLM pass and the
   cheap scoring/derivation pass are separate, publicly composable steps.
5. **PDDL is a front door, not a requirement.** The estimator's contract is
   "predicates in, state out"; PDDL grounding is one (well-supported) way to
   produce predicates.

## Package layout

```
s3e/
  __init__.py        # curated exports: the ~12 names users need
  _deps.py           # require(module, extra) helper (internal)
  backends/          # VLMBackend, VLMOutput, HuggingFaceVLM, OpenAIVLM,
                     # VLLMBackend, resolve_backend()
  engine/            # QueryEngine, AnswerSpace, AnswerOption, BinaryAnswers,
                     # CategoricalAnswers, Prediction, PredictionSet
  calibration/       # Calibrator, PlattCalibrator, CalibrationSet
  translation/       # QueryTranslator + Identity/Template/Prewritten/LLM,
                     # translation cache (absorbs cache.py)
  pddl/              # UP parsing, grounding, state conversion (up_utils API kept)
  estimator.py       # SemanticStateEstimator facade
```

Dependency direction: `estimator` → (`pddl`, `translation`, `engine`) →
`backends`; `calibration` depends only on `engine` result types. `s3e/vlm/` is
renamed to `s3e/backends/`.

## Layer 1: Backends (`s3e/backends/`)

Kept close to the current design — the best-factored part of the codebase.

- `VLMBackend.query_batch(images, prompts, interest_tokens=...) -> list[VLMOutput]`
  remains the single required method. The interest-token contract becomes the
  *primary* path: backends gather probability mass at known token ids and
  never materialize full-vocab distributions on the default path.
- `VLMOutput` remains the normalized record (generated text, token
  distributions, interest-token masses). Backends never know about answer
  spaces, calibration, or PDDL.
- Multi-image batching is part of the documented backend contract: a batch
  element is (scene, prompt) where a scene is one or more images; the HF
  padding behavior fixed in 871fb46 is pinned by contract tests.
- `resolve_backend(spec, **vlm_kwargs) -> VLMBackend` is the **public** model-string
  factory (`"OpenAI/..."` → OpenAIVLM; otherwise HuggingFaceVLM). It replaces
  the private `SemanticStateEstimator._build_vlm_from_string` that MLSS and
  ViPlan++ currently import. The `use_vllm=` flag is deleted; vLLM users
  construct `VLLMBackend(...)` explicitly.
- Base classes (`backend.py`) are importable with **no** optional heavy
  dependencies installed, so downstream code can `isinstance`-check without
  importing vLLM/torch (kills ViPlan++'s `type(vlm).__module__` sniffing).

## Layer 2: Engine (`s3e/engine/`)

### QueryEngine

The PDDL-free heart: images + queries + answer space → predictions.

```python
engine = QueryEngine(
    vlm,                          # VLMBackend instance or model string
    answers=BinaryAnswers(),      # default answer space
    system_prompt=...,            # optional; sensible default
    prompt_template="{query}",    # wrapper for each query; must contain {query}
    batch_size=8,
    inference_kwargs={...},       # default per-query kwargs, forwarded verbatim
    vlm_kwargs={...},             # used only when vlm is a model string
)
predictions = engine.ask(images, queries)                      # PredictionSet
predictions = engine.ask(images, queries, answers=...,         # per-call overrides
                         inference_kwargs=..., keep_raw=False)
```

- `images` is one *scene* (a list of images presented together — today's
  `"single"` strategy). Today's `"average"` strategy decomposes into two
  explicit pieces: `engine.ask_each(scenes, queries) -> list[PredictionSet]`
  (one prediction set per scene, batched through the same path) and
  `PredictionSet.average(sets) -> PredictionSet` (mean of stored masses).
  Multi-image strategy is no longer constructor state.
- Batching is defined over (scene, query) pairs and lives in the engine.
- `inference_kwargs` semantics per backend are documented on the backend
  classes (OpenAI: request args; HF: forward/generate args; vLLM:
  SamplingParams).

### Answer spaces

An answer **option** is a label plus the token strings that count as
expressing it:

```python
AnswerOption("red", tokens=["red", "Red", "RED", " red", ...])
```

Options auto-expand labels into case/leading-whitespace variants by default;
`tokens=` overrides. Spaces:

- `BinaryAnswers(true_label="yes", false_label="no", *, true_tokens=None,
  false_tokens=None)` — two options with boolean semantics.
  `BinaryAnswers("true", "false")` works out of the box.
- `CategoricalAnswers([...])` — N options; labels or explicit `AnswerOption`s.

Scoring lives in the answer space, in one place:

- **Logprob scoring** (default): sum backend-reported masses per option's
  single-token forms; leftover is null mass. Options whose every form
  tokenizes to multiple tokens are rejected with a clear `ValueError` in
  logprob mode. Multi-token sequence scoring is a documented future
  extension, not v1.
- **Text-match scoring**: match generated text against option forms. Replaces
  the estimator-level `probability_method="text_match"` flag.

### Results

Both immutable after creation.

**`Prediction`** — one query's outcome. Stores per-option masses, null mass,
generated text (if any), and a reference to its answer space. Lazy cached
derivations:

```python
p.masses          # {"yes": 0.71, "no": 0.09} — the stored data
p.null_mass       # 0.20
p.probability     # binary spaces: normalized P(true)
p.answer          # argmax label; bool for binary spaces
p.null_dominated  # null_mass > every option mass
p.confident(0.8)  # threshold check
p.raw             # underlying VLMOutput; only if keep_raw=True, else None
```

**`PredictionSet`** — ordered mapping of query (or predicate) → `Prediction`:

```python
results["on(a,b)"].probability
results.probabilities()             # dict[str, float]
results.to_state(confidence=0.85)   # dict[str, bool | None]
results.where(pred)                 # filtering/inspection
results.to_dict() / PredictionSet.from_dict(d)   # backend-free round trip
```

Efficiency requirements (from MLSS / ViPlan++ usage):

1. Predictions store a handful of floats by default; `keep_raw=True` is
   opt-in so long runs don't pin token tensors.
2. `to_dict`/`from_dict` reconstruct a `PredictionSet` with **no backend
   present** — the offline-recalibration workflow becomes supported API.
3. Serialized formats carry a `"format_version"` field.

## Layer 3: Calibration (`s3e/calibration/`)

Self-contained subpackage operating on prediction data, never on estimator
internals. Absorbs the ~450 lines of Platt plumbing currently in the
estimator; the estimator keeps zero calibration state.

```python
data = CalibrationSet.collect(estimator, labeled_examples)  # expensive, uses VLM
data.save("calib.json"); data = CalibrationSet.load("calib.json")

cal = PlattCalibrator.fit(data, scope="lifted")   # cheap, offline
cal.save("platt.json"); cal = PlattCalibrator.load("platt.json")

calibrated = cal.apply(results)    # new PredictionSet; original untouched
```

- `Calibrator` ABC: `fit`, `apply`, `save`/`load`. `TemperatureCalibrator` is
  a planned drop-in; conformal methods are a natural future implementation.
- Scopes `"global" | "lifted" | "grounded"` and existing validation/grouping
  behavior are ported, not redesigned.
- Convenience: `estimator.estimate(images, calibrator=cal)` — explicit at the
  call site, never implicit state.

## Layer 4: PDDL (`s3e/pddl/`) and the facade (`s3e/estimator.py`)

`s3e/pddl/` owns everything Unified-Planning-shaped; nothing else imports UP:

- `parse_domain_problem(domain, problem)` — strings or file paths.
- `ground_predicates(up_problem) -> list[str]`.
- Existing `up_utils` public surface kept (MLSS imports it directly).

`SemanticStateEstimator` is a thin facade (~150–200 lines) owning exactly a
predicate list, a translator, and a `QueryEngine`:

```python
estimator = SemanticStateEstimator.from_pddl(domain, problem, vlm=..., translator=...)
estimator = SemanticStateEstimator(predicates=[...], vlm=..., translator=...)

state = estimator(images)                          # sugar: .estimate(...).to_state()
results = estimator.estimate(images, predicates=subset, calibrator=None)
estimator.set_problem(domain, problem)             # re-grounds; backend untouched
estimator.to_up_state(state)                       # PDDL-constructed estimators only
```

- `estimate(images, predicates=subset)` queries only the subset
  (relevant-atom masking — both consumers depend on it).
- `set_problem` (renames `swap_problem`) rebuilds grounding + queries only.
- Shared backends are the documented pattern: build one `VLMBackend`, hand it
  to many engines/estimators.
- Translators keep their current interfaces; `cache.py` folds into
  `s3e/translation/`.

## Dependencies

### Tiers

- **Core** (`pip install s3e`): `Pillow`, `numpy`, `tqdm`. Buys the engine,
  result objects, answer spaces, non-LLM translators, and any backend whose
  deps are present.
- **Extras:**
  - `hf`: torch, torchvision, transformers, accelerate
  - `openai`: openai
  - `vllm`: vllm>=0.11.0
  - `pddl`: unified-planning>=1.3.0
  - `calibration`: scikit-learn
  - `all`: everything except `vllm` (platform-constrained)
  - `dev`: pytest + contributor tooling
- README quick start recommends `pip install "s3e[pddl,hf]"`.

### Import strategy (no guard clutter)

- Heavy imports appear as normal top-of-module imports **in leaf modules
  only**. No try/except per import.
- Nothing in core imports a leaf module at package-import time. `s3e/__init__.py`
  and subpackage `__init__`s expose optional-dependency names via module
  `__getattr__` (the existing `VLLMBackend` lazy pattern, applied uniformly).
- `s3e._deps.require(module, extra)` is called once per optional module (top
  of leaf module or class constructor) to convert `ModuleNotFoundError` into
  e.g. `ImportError: HuggingFaceVLM requires the 'hf' extra: pip install "s3e[hf]"`.
- `import s3e` on a bare install is clean and silent: no warnings, no heavy
  modules in `sys.modules`.

## Error handling

- `ValueError` with the offending value named: unknown predicate, answer
  option with no single-token form in logprob mode, template without
  `{query}`, malformed serialized payloads.
- `ImportError` with the exact extra to install, raised lazily at
  construction/first use, never at `import s3e`.
- Serialization formats (`PredictionSet`, `CalibrationSet`, calibrator
  profiles) include `"format_version"`; mismatches fail loudly with a clear
  message.

## Testing strategy

Test tree mirrors the package: `tests/engine/`, `tests/backends/`,
`tests/calibration/`, `tests/pddl/`, `tests/translation/`,
`tests/test_estimator.py`, plus `tests/test_imports.py` and
`tests/consumers/`. The 2,700-line estimator test monolith is split along the
same boundaries as the code.

1. **Shared fake backend.** The `FakeVLM`-style double becomes one documented
   fixture implementing the full `VLMBackend` contract (interest tokens,
   batching, multi-image padding), reused by engine/estimator/calibration
   tests.
2. **Backend contract suite.** One parametrized test suite every backend
   (real or fake) must pass, so HF/OpenAI/vLLM/fakes cannot drift apart.
   Slow/GPU marker convention stays; real-model tests remain `@pytest.mark.slow`.
3. **Import-hygiene tests** (`tests/test_imports.py`), using a meta-path
   blocker simulating missing packages, run in subprocesses for clean module
   caches:
   - `import s3e` on a bare install succeeds with zero warnings and no heavy
     modules in `sys.modules` (torch, transformers, unified_planning, ...).
   - Each optional feature imported without its dependency raises
     `ImportError` naming the exact extra.
   - Each extra installed alone actually enables its feature.
4. **Consumer contract tests** (`tests/consumers/`), written against the fake
   backend, one module per downstream project, each test documenting the
   workflow it protects:
   - `test_mlss_workflow.py`: long-lived estimator with `set_problem` per
     sample and no backend rebuild; `estimate(images, predicates=subset)`
     querying only the subset; details serialization to JSON and offline
     reconstruction via `from_dict`; calibration collect → save → refit
     without a VLM.
   - `test_viplan_workflow.py`: shared prebuilt backend across estimators;
     the adapter pattern (`prepare`/`estimate` with relevant-atom
     subsetting); every prediction-detail field ViPlan++'s payload builder
     reads; backend type detection without importing vLLM; public
     `resolve_backend`.
   These are the migration gate: if they pass, MLSS and ViPlan++ can migrate
   with mechanical renames.
5. **Serialization round-trip tests** for every format.
6. **Red/green discipline.** Implementation follows TDD: for each component,
   tests are written against the designed API first, confirmed failing (red),
   then implemented to green. Ported legacy tests (padding, interest-token
   math, Platt grouping — hard-won behavior) form the regression net during
   the port.

## Migration notes (for MLSS and ViPlan++)

| Old | New |
|---|---|
| `SemanticStateEstimator(domain, problem, vlm=...)` | `SemanticStateEstimator.from_pddl(domain, problem, vlm=...)` |
| `SemanticStateEstimator._build_vlm_from_string(...)` | `resolve_backend(...)` (public) |
| `swap_problem(...)` | `set_problem(...)` |
| `estimate_raw` / `estimate_probabilities` / `estimate_prediction_details` | `estimate(...)` returning `PredictionSet` (`.raw` via `keep_raw=True`, `.probabilities()`, per-`Prediction` details) |
| `PredicatePredictionDetails` | `Prediction` (lazy fields) |
| `probability_method="text_match"` | text-match scoring on the answer space |
| `use_vllm=True` | construct `VLLMBackend(...)` explicitly |
| `fit_platt_scaling*` / `save_platt_scaling*` on estimator | `CalibrationSet` + `PlattCalibrator` in `s3e.calibration` |
| module-name sniffing for vLLM detection | `isinstance` against always-importable base classes |

## Repo hygiene (folded into this sub-project)

Remove stray files (`hello.bash`, `slurm-1345506.out`, tracked `s3e.egg-info`
if any); fix `pyproject.toml` license file reference (`LICENSE.txt` →
`LICENSE`).

## Future extensions (designed for, not built)

- `TemperatureCalibrator`; conformal calibration.
- Multi-token sequence scoring for categorical answers.
- Additional answer spaces (e.g. numeric).
