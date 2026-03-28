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
- `s3e/semantic_state_estimator.py`: main public estimator
- `s3e/state_estimator.py`: abstract estimator interfaces
- `s3e/vlm/`: VLM abstractions and implementations
- `s3e/translation/`: predicate-to-query translators
- `s3e/pddl/`: Unified Planning / PDDL helpers
- `s3e/cache.py`: JSON cache helpers
- `tests/conftest.py`: shared fixtures and Blocksworld sample data

## Environment
- Python requirement: `>=3.10`
- Build backend: setuptools
- The README is minimal; rely on source, tests, and `pyproject.toml` for actual conventions.

## Setup Commands
- Core editable install: `pip install -e .`
- Dev install: `pip install -e '.[dev]'`
- Optional OpenAI extras: `pip install -e '.[openai]'`

## Build Commands
- Packaging is configured through setuptools in `pyproject.toml`.
- If the `build` package is installed, create wheel/sdist with: `python -m build`
- In the analyzed environment, `python -m build` currently fails because `build` is not installed.
- Do not document or automate a different build flow unless you add the necessary config in the same change.

## Lint And Static Checks
- There is no configured linter or formatter in this repository.
- No repo config exists for `ruff`, `black`, `isort`, `flake8`, `mypy`, or `pyright`.
- Do not invent repo-standard lint commands.
- If you want lightweight validation, use:
  - `python -m compileall s3e tests`
  - `pytest -m "not slow"`
- If you introduce a lint or type-check tool, update `pyproject.toml`, CI, and this file together.

## Test Commands
- Full suite: `pytest`
- Fast default loop: `pytest -m "not slow"`
- Slow tests only: `pytest -m slow`
- Verbose run: `pytest -v`
- Test discovery / node IDs: `pytest --collect-only -q`

## Single-Test Commands
- One file: `pytest tests/test_cache.py`
- One class: `pytest tests/test_vlm_backends.py::TestOpenAIVLM`
- One test: `pytest tests/test_cache.py::TestMakeCacheKey::test_basic_key`
- Another example: `pytest tests/test_semantic_state_estimator.py::TestTextMatchMode::test_text_match_probability`
- Prefer the narrowest relevant test first, then broaden scope only if needed.

## Test Suite Notes
- `pytest.ini` defines a `slow` marker for tests that download and run real HuggingFace models.
- `pytest -m "not slow"` is the default verification command for normal development.
- Reuse fixtures from `tests/conftest.py` instead of duplicating common setup.
- Existing tests cover cache helpers, PDDL utilities, translators, VLM backends, and estimator flows.

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

## Agent Workflow
- Inspect the target module and its nearest tests before editing.
- After edits, run the narrowest relevant pytest command first.
- If a change crosses modules, run `pytest -m "not slow"` before finishing.
- Do not add new tooling or style rules unless the change truly requires them.
- Keep diffs minimal and aligned with the current structure.
- Do not revert unrelated user changes in a dirty worktree.

## Quick Reference
- Dev install: `pip install -e '.[dev]'`
- Fast verification: `pytest -m "not slow"`
- Single test: `pytest tests/test_cache.py::TestMakeCacheKey::test_basic_key`
- Test discovery: `pytest --collect-only -q`
- Optional package build: `python -m build`
