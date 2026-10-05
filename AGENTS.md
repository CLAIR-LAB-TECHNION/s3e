# AGENTS.md

## Scope
- Repository: `s3e`, a Python package for semantic state estimation over PDDL using vision-language and language-model backends.
- Source: `s3e/`
- Tests: `tests/`
- Packaging/config: `pyproject.toml`, `pytest.ini`
- No repo-specific Cursor or Copilot rules were found:
  - no `.cursorrules`
  - no `.cursor/rules/`
  - no `.github/copilot-instructions.md`
- If any of those files appear later, treat them as higher-priority instructions and update this file.

## Layout
- `s3e/estimator.py`: `SemanticStateEstimator`, the thin PDDL-facade wiring predicates, translation, and a `QueryEngine`
- `s3e/engine/`: `QueryEngine`, answer spaces (`BinaryAnswers`, `CategoricalAnswers`), and result types (`Prediction`, `PredictionSet`) — the PDDL-free core
- `s3e/backends/`: `VLMBackend` interface, `VLMOutput`, `HuggingFaceVLM`, `OpenAIVLM`, `VLLMBackend`, `resolve_backend()`
- `s3e/calibration/`: `Calibrator`, `PlattCalibrator`, `CalibrationSet`/`CalibrationExample`/`CalibrationSample` — offline calibration over prediction data
- `s3e/translation/`: predicate-to-query translators, including `cache.py` (JSON cache helpers for `LLMTranslator`)
- `s3e/pddl/`: Unified Planning / PDDL helpers (parsing, grounding, state conversion)
- `s3e/_deps.py`: `require(module, extra)` helper for lazy, informative optional-dependency errors
- `tests/`: mirrors the package layout (`tests/engine/`, `tests/backends/`, `tests/calibration/`, `tests/pddl/`, `tests/workflows/`, `tests/test_estimator.py`, `tests/test_imports.py`, ...)
- `tests/conftest.py`: shared fixtures and Blocksworld sample data
- `tests/fakes.py`: shared `FakeVLM` double implementing the full `VLMBackend` contract
- `docs/`: Sphinx API reference (`conf.py`, MyST Markdown pages, `api/*.rst` autodoc pages) plus the walkthrough notebook `docs/s3e_walkthrough.ipynb`; built on Read the Docs via `.readthedocs.yaml`
- `paper/`: JOSS paper (`paper.md`, `paper.bib`, `s3e-pipeline.png`); keep it the only `paper.md` in the repo
- `.github/workflows/`: `tests.yml` (fast suite on Python 3.10–3.14, ruff, strict docs build) and `draft-pdf.yml` (JOSS draft PDF)
- Project metadata: `CHANGELOG.md`, `CITATION.cff`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `.mailmap`

## Environment
- Python requirement: `>=3.10`
- Build backend: setuptools
- The README documents usage; rely on source, tests, and `pyproject.toml` for actual conventions.

## Setup Commands
- Core editable install: `pip install -e .`
- Dev install (CPU, standard): `pip install -e '.[dev]'` — everything needed for the test suite except vLLM; vLLM-dependent tests skip.
- Dev install (CUDA hosts): `pip install -e '.[dev-gpu]'` — adds `vllm`; required for the vLLM unit tests and `pytest -m slow` vLLM coverage.
- Optional extras: `pddl` (PDDL grounding), `hf` (HuggingFace VLM backend), `openai` (OpenAI VLM backend), `vllm` (local multi-GPU inference), `calibration` (Platt scaling, scikit-learn), `all` (everything except `vllm`), `docs` (Sphinx toolchain) — e.g. `pip install -e '.[pddl,hf]'`

## Build Commands
- Packaging is configured through setuptools in `pyproject.toml`.
- Create wheel/sdist with `python -m build` (requires `pip install build`; it is not part of any extra).
- Do not document or automate a different build flow unless you add the necessary config in the same change.
- Docs: `pip install -e '.[docs]'` then `sphinx-build -W -b html docs docs/_build/html`. The build mocks torch/transformers/vllm, so new public modules must import cleanly under those mocks; add new public APIs to the matching `docs/api/*.rst` page.
- JOSS paper: the `Draft PDF` workflow builds `paper/paper.pdf` on changes under `paper/`. Keep the paper between 750 and 1750 words (the JOSS bot counts the whole file).

## Lint And Static Checks
- Linter: `ruff check .` (installed by the `dev` extra). The rule set in `pyproject.toml` (`[tool.ruff.lint]`) is correctness-focused (`E4`, `E7`, `E9`, `F`, `W`, `B`); `E402` is ignored because optional backends import their dependency after `require()`.
- There is no formatter and no type checker; do not run `ruff format` or reformat files.
- If you introduce or change a lint or type-check tool, update `pyproject.toml`, CI, and this file together.
- CI (`.github/workflows/tests.yml`) installs CPU-only torch, then `pip install -e '.[dev]'`, and runs `pytest -m "not slow"` on Python 3.10–3.14, plus `ruff check .` and the strict docs build.

## Test Commands
- Full suite: `pytest`
- Fast default loop: `pytest -m "not slow"`
- Slow tests only: `pytest -m slow`
- Verbose run: `pytest -v`
- Test discovery / node IDs: `pytest --collect-only -q`

## Single-Test Commands
- One file: `pytest tests/test_cache.py`
- One class: `pytest tests/engine/test_answers.py::TestBinaryAnswers`
- One test: `pytest tests/test_cache.py::TestMakeCacheKey::test_basic_key`
- Another example: `pytest tests/engine/test_answers.py::TestBinaryAnswers::test_default_yes_no`
- Prefer the narrowest relevant test first, then broaden scope only if needed.

## Test Suite Notes
- `pytest.ini` defines a `slow` marker for tests that download and run real HuggingFace models.
- Docstring examples are doctests: bare `pytest` collects `tests/` and `s3e/` with `--doctest-modules` (except `s3e/backends/`, which imports heavy dependencies). Keep examples to model-free APIs and round floats in their output.
- `pytest -m "not slow"` is the default verification command for normal development.
- Running the suite assumes a dev install (`dev` or `dev-gpu`). Without `vllm`, the vLLM unit tests skip with a reason; the slow vLLM integration tests additionally require CUDA. Other partial installs (missing torch/openai/unified-planning) are supported for the *library* (see `tests/test_imports.py`) but not for running the test suite itself.
- Reuse fixtures from `tests/conftest.py` and the shared `FakeVLM` double from `tests/fakes.py` instead of duplicating common setup.
- The test tree mirrors the package: `tests/engine/`, `tests/backends/`, `tests/calibration/`, `tests/pddl/`, plus `tests/test_estimator.py`, `tests/test_translators.py`, `tests/test_cache.py`, `tests/test_imports.py` (import-hygiene for bare/partial installs), and `tests/workflows/` (end-to-end usage-pattern tests that span modules).

## Code Style Guidelines

### General
- Follow the existing repository style; do not impose a new style system.
- Use 4-space indentation.
- Keep modules focused on one responsibility.
- Start modules with a concise top-level docstring.
- Add docstrings for public classes and important functions.
- Prefer clear names and small helpers over extra comments.

### Imports
- Group imports as: standard library, third-party, local package.
- Separate import groups with one blank line.
- Inside package code, prefer relative imports such as `from .backend import VLMBackend`.
- In tests, prefer absolute imports from `s3e...`; `from conftest import ...` is also used for shared fixtures/constants.
- Prefer explicit imports over wildcard imports.
- Use parenthesized multiline imports when local import lists get long.

### Formatting
- No formatter is enforced, so preserve the surrounding file's style.
- Keep line length readable; no hard limit is configured.
- Wrap long constructor calls and function calls across lines.
- The codebase mostly uses double quotes; keep file-local consistency.

### Types
- Type hints are expected on public APIs and common on internal helpers.
- Prefer built-in generics like `list[str]` and `dict[str, float]` in new code.
- Prefer PEP 604 unions like `str | None` in new code.
- Some existing files still use `Union` / `Optional`; do not churn them without a reason.
- Use concrete return types for helpers and backend interfaces when practical.
- Narrow `# type: ignore[...]` usage is acceptable for optional dependency compatibility shims.

### Naming
- `snake_case` for functions, methods, variables, and fixtures
- `PascalCase` for classes
- `UPPER_SNAKE_CASE` for constants and prompt templates
- Test classes named `Test...`
- Test methods named `test_...`
- Prefer explicit domain terms over vague names

### API Design
- Favor small composable abstractions.
- Use `ABC` and `@abstractmethod` for formal interfaces.
- Use `@dataclass` for lightweight structured outputs like `VLMOutput`.
- Update package re-exports and `__all__` when adding public APIs.
- Keep backend-specific logic inside backend modules, not generic estimator modules.

### Error Handling
- Raise explicit, informative exceptions.
- Use `ValueError` for invalid predicate/template/query inputs.
- Use `ImportError` with install guidance for optional dependencies like `openai` or `transformers`.
- Avoid broad exception handling unless it is a deliberate fallback or compatibility path.
- If you add a fallback, make the alternate behavior obvious and safe.
- Prefer exceptions over `assert` for user-facing validation.
- Reserve `assert` for internal invariants and tests.

### Dependency And Compatibility Conventions
- Optional integrations should fail lazily with a helpful message.
- Preserve the `OpenAI/` model-prefix convention used by OpenAI-backed code.
- Preserve compatibility branches for HuggingFace / transformers API differences when touching that code.

## Testing Conventions
- Use `pytest`, not `unittest.TestCase`.
- `unittest.mock.patch` and `MagicMock` are acceptable inside pytest tests.
- Prefer focused assertions over large opaque fixtures.
- Mark real-model or download-heavy tests with `@pytest.mark.slow`.
- When behavior changes, update the nearest relevant test module.
- Record user-facing changes under "Unreleased" in `CHANGELOG.md`.

## Agent Workflow
- Inspect the target module and its nearest tests before editing.
- After edits, run the narrowest relevant pytest command first.
- If a change crosses modules, run `pytest -m "not slow"` before finishing.
- Do not add new tooling or style rules unless the change truly requires them.
- Keep diffs minimal and aligned with the current structure.
- Do not revert unrelated user changes in a dirty worktree.

## Quick Reference
- Dev install: `pip install -e '.[dev]'` (CPU) or `pip install -e '.[dev-gpu]'` (CUDA hosts, adds vLLM)
- Fast verification: `pytest -m "not slow"` and `ruff check .`
- Single test: `pytest tests/test_cache.py::TestMakeCacheKey::test_basic_key`
- Test discovery: `pytest --collect-only -q`
- Optional package build: `python -m build`
