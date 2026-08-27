# S3E Architecture Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure `s3e` into the layered architecture from the approved spec: backends → engine (answer spaces + lazy results) → calibration/translation/PDDL → thin estimator facade, with tiered optional dependencies and consumer-contract tests.

**Architecture:** Strictly-downward layers. `s3e/backends/` (renamed from `s3e/vlm/`) keeps the existing VLM implementations. A new `s3e/engine/` provides `QueryEngine`, answer spaces, and lazy `Prediction`/`PredictionSet` objects. `s3e/calibration/` becomes a package holding all Platt machinery. `SemanticStateEstimator` is rewritten as a ~200-line facade in `s3e/estimator.py` with PDDL as an optional front door (`from_pddl`).

**Tech Stack:** Python ≥3.10, pytest, setuptools. Core deps: Pillow, numpy, tqdm. Extras: hf (torch/torchvision/transformers/accelerate), openai, vllm≥0.11.0, pddl (unified-planning≥1.3.0), calibration (scikit-learn).

**Spec:** `docs/superpowers/specs/2026-08-28-architecture-redesign-design.md` — read it before starting. It is the authority on API shape and rationale.

## Global Constraints

- Python `>= 3.10`; built-in generics (`list[str]`) and PEP 604 unions (`str | None`) in new code.
- 4-space indent, double quotes, module docstrings, docstrings on public classes/functions (see AGENTS.md).
- TDD every task: write the failing test, run it, watch it fail, implement, watch it pass, commit. Never write implementation before its failing test.
- After every task: `pytest -m "not slow"` must be green before committing (existing tests count; update imports in old tests when a task moves modules).
- Heavy imports (`torch`, `transformers`, `unified_planning`, `vllm`, `openai`, `sklearn`) may appear ONLY in leaf modules: `s3e/backends/huggingface.py`, `s3e/backends/openai.py`, `s3e/backends/vllm.py`, `s3e/pddl/*`, `s3e/calibration/platt.py` (inside `fit` only), `s3e/translation/llm.py` (inside constructor only). `import s3e` must never import any of them.
- No per-import try/except guards. Optional-dependency errors go through `s3e._deps.require` exactly once per optional boundary.
- Every serialization format includes `"format_version"`.
- `EPS = 1e-12` (ported `SCORE_EPS`) is the single epsilon used in probability/score math; it lives in `s3e/engine/results.py` and is imported everywhere else.
- Commit messages: conventional style as in recent history (`feat:`, `refactor:`, `test:`, `docs:`, `build:`).
- Do not modify `MLSS/` or `ViPlan-PlusPlus/` — they are reference consumers only.

---

### Task 1: Repo hygiene and dependency tiers

**Files:**
- Delete: `hello.bash`, `slurm-1345506.out`
- Modify: `pyproject.toml`, `.gitignore`

**Interfaces:**
- Consumes: nothing.
- Produces: extras `hf`, `openai`, `vllm`, `pddl`, `calibration`, `all`, `dev` in `pyproject.toml`; later tasks' error messages name these extras.

- [ ] **Step 1: Remove stray files**

```bash
rm hello.bash slurm-1345506.out
grep -qxF 'slurm-*.out' .gitignore || echo 'slurm-*.out' >> .gitignore
grep -qxF '*.egg-info/' .gitignore || echo '*.egg-info/' >> .gitignore
```

(`hello.bash` and `slurm-1345506.out` are untracked scratch; `rm` is enough. If `git ls-files | grep egg-info` shows tracked egg-info files, also `git rm -r --cached s3e.egg-info`.)

- [ ] **Step 2: Rewrite the `[project]` dependency sections of `pyproject.toml`**

Replace the `license`, `dependencies`, and `[project.optional-dependencies]` sections with:

```toml
license = { file = "LICENSE" }
dependencies = [
    "numpy",
    "tqdm",
    "Pillow",
]

[project.optional-dependencies]
hf = ["torch", "torchvision", "transformers", "accelerate"]
openai = ["openai"]
# VLLMBackend relies on SamplingParams(logprobs=-1) / LLM(max_logprobs=-1)
# for full-vocab logprobs, added in vLLM 0.11.0 (vllm-project/vllm#25031).
vllm = ["vllm>=0.11.0"]
pddl = ["unified-planning>=1.3.0"]
calibration = ["scikit-learn"]
# Everything except vllm, which has hard platform constraints.
all = ["s3e[hf,openai,pddl,calibration]"]
dev = ["pytest", "s3e[all]"]
```

(The old core list wrongly shipped torch/transformers/accelerate/torchvision and unified-planning to every user; `LICENSE.txt` was a broken reference — the file is `LICENSE`.)

- [ ] **Step 3: Verify the metadata parses and the suite is green**

Run: `python -c "import tomllib; tomllib.load(open('pyproject.toml','rb')); print('ok')"` → `ok`
Run: `pip install -e '.[dev]' -q && pytest -m "not slow" -q` → all pass (nothing is uninstalled by re-tiering; the dev environment keeps its packages).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "build: tier optional dependencies; repo hygiene"
```

---

### Task 2: `s3e/_deps.py` optional-dependency helper

**Files:**
- Create: `s3e/_deps.py`
- Test: `tests/test_deps.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `require(module_name: str, extra: str, feature: str) -> None` — raises `ImportError` naming the extra when `module_name` is not importable; no-op otherwise. Used by every optional leaf module in later tasks.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_deps.py
"""Tests for the optional-dependency helper."""

import pytest

from s3e._deps import require


class TestRequire:
    def test_noop_when_module_present(self):
        require("json", "someextra", "SomeFeature")  # stdlib always present

    def test_raises_naming_extra_when_missing(self):
        with pytest.raises(ImportError) as excinfo:
            require("s3e_definitely_not_a_module", "hf", "HuggingFaceVLM")
        message = str(excinfo.value)
        assert "HuggingFaceVLM" in message
        assert 'pip install "s3e[hf]"' in message

    def test_survives_meta_path_finders_that_raise(self):
        import sys

        class Blocker:
            def find_spec(self, name, path=None, target=None):
                if name == "blocked_module_xyz":
                    raise ModuleNotFoundError(name=name)
                return None

        sys.meta_path.insert(0, Blocker())
        try:
            with pytest.raises(ImportError, match=r"s3e\[vllm\]"):
                require("blocked_module_xyz", "vllm", "VLLMBackend")
        finally:
            sys.meta_path.pop(0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_deps.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 's3e._deps'`

- [ ] **Step 3: Implement `s3e/_deps.py`**

```python
# s3e/_deps.py
"""Optional-dependency checks with install guidance.

Leaf modules that need a heavy optional package call :func:`require` once,
before their normal top-of-module imports. Everything else in the package
imports freely — no per-import guards.
"""

import importlib.util


def require(module_name: str, extra: str, feature: str) -> None:
    """Raise a helpful ImportError when an optional module is missing.

    Args:
        module_name: Importable module to check for (e.g. ``"torch"``).
        extra: The s3e extra that provides it (e.g. ``"hf"``).
        feature: Human-readable feature name for the error message.
    """
    try:
        found = importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        # Meta-path finders may raise instead of returning None (our own
        # import-hygiene tests install one that does); treat that as absent.
        found = False
    if not found:
        raise ImportError(
            f'{feature} requires the {extra!r} extra: pip install "s3e[{extra}]"'
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_deps.py -v` → PASS

- [ ] **Step 5: Commit**

```bash
git add s3e/_deps.py tests/test_deps.py
git commit -m "feat: add optional-dependency require() helper"
```

---

### Task 3: Rename `s3e/vlm/` to `s3e/backends/`, add public `resolve_backend`

**Files:**
- Move: `s3e/vlm/` → `s3e/backends/` (all five modules, via `git mv`)
- Create: `s3e/backends/resolve.py`
- Modify: `s3e/backends/__init__.py`, `s3e/backends/huggingface.py`, `s3e/backends/openai.py`, `s3e/backends/vllm.py`, `s3e/semantic_state_estimator.py`, `s3e/__init__.py`, `s3e/translation/llm.py` (if it imports `s3e.vlm`), `tests/test_vlm_backends.py`, `tests/conftest.py` (any `s3e.vlm` imports)
- Test: `tests/backends/test_resolve.py`

**Interfaces:**
- Consumes: `s3e._deps.require` (Task 2).
- Produces: `s3e.backends.VLMBackend`, `s3e.backends.VLMOutput` (eager); `HuggingFaceVLM`, `OpenAIVLM`, `VLLMBackend` (lazy via module `__getattr__`); `resolve_backend(vlm: str | VLMBackend, **vlm_kwargs) -> VLMBackend`; `OPENAI_MODEL_IDENTIFIER = "OpenAI/"`. Signature of `VLMBackend.query_batch(images, prompts, system_prompt=None, generate=False, interest_tokens=None, **inference_kwargs) -> list[VLMOutput]` is unchanged and later tasks depend on it exactly as-is.

- [ ] **Step 1: Write the failing tests**

```python
# tests/backends/test_resolve.py
"""Tests for the public backend factory."""

import sys

import pytest

from s3e.backends import VLMBackend, VLMOutput, resolve_backend


class DummyBackend(VLMBackend):
    def query(self, images, prompt, system_prompt=None, generate=False,
              interest_tokens=None, **inference_kwargs):
        return VLMOutput()


class TestResolveBackend:
    def test_instance_passes_through(self):
        backend = DummyBackend()
        assert resolve_backend(backend) is backend

    def test_instance_with_vlm_kwargs_rejected(self):
        with pytest.raises(ValueError, match="vlm_kwargs"):
            resolve_backend(DummyBackend(), device_map="auto")

    def test_openai_prefix_selects_openai_backend(self, monkeypatch):
        from s3e.backends import openai as openai_module

        captured = {}

        class FakeOpenAIVLM:
            def __init__(self, model, **kwargs):
                captured["model"] = model
                captured["kwargs"] = kwargs

        monkeypatch.setattr(openai_module, "OpenAIVLM", FakeOpenAIVLM)
        resolve_backend("OpenAI/gpt-4o", api_key="k")
        assert captured == {"model": "gpt-4o", "kwargs": {"api_key": "k"}}

    def test_plain_string_selects_huggingface_backend(self, monkeypatch):
        from s3e.backends import huggingface as hf_module

        captured = {}

        class FakeHFVLM:
            def __init__(self, model, **kwargs):
                captured["model"] = model
                captured["kwargs"] = kwargs

        monkeypatch.setattr(hf_module, "HuggingFaceVLM", FakeHFVLM)
        resolve_backend("Qwen/Qwen2-VL-7B-Instruct", device_map="auto")
        assert captured["model"] == "Qwen/Qwen2-VL-7B-Instruct"
        assert captured["kwargs"] == {"device_map": "auto"}


class TestLazyExports:
    def test_base_import_does_not_pull_heavy_modules(self):
        # s3e.backends is already imported by this test module; the assertion
        # is that importing it never dragged torch in transitively.
        import s3e.backends  # noqa: F401
        assert "s3e.backends.base_marker" not in sys.modules  # sanity
        # HuggingFaceVLM is exposed lazily:
        import s3e.backends as b
        assert "HuggingFaceVLM" in b.__all__

    def test_unknown_attribute_raises(self):
        import s3e.backends as b
        with pytest.raises(AttributeError):
            b.NoSuchThing
```

Also create empty `tests/backends/__init__.py`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/backends/test_resolve.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 's3e.backends'`

- [ ] **Step 3: Move the package and fix imports**

```bash
git mv s3e/vlm s3e/backends
grep -rln "s3e\.vlm\|from \.vlm\|from \.\.vlm" s3e tests | xargs sed -i 's/s3e\.vlm/s3e.backends/g; s/from \.vlm/from .backends/g; s/from \.\.vlm/from ..backends/g'
```

Then review `git diff` — the substitutions must only touch import lines and docstrings. `ViPlan`'s module-sniffing string `"s3e.vlm.vllm"` does not exist in this repo; nothing else should reference `s3e.vlm`.

- [ ] **Step 4: Add `resolve.py`**

```python
# s3e/backends/resolve.py
"""Public factory turning model-id strings into VLM backends."""

from .backend import VLMBackend

OPENAI_MODEL_IDENTIFIER = "OpenAI/"


def resolve_backend(vlm: "str | VLMBackend", **vlm_kwargs) -> VLMBackend:
    """Resolve a model string or backend instance into a VLMBackend.

    Strings prefixed with ``"OpenAI/"`` select :class:`OpenAIVLM`; any other
    string selects :class:`HuggingFaceVLM`. For a vLLM engine, construct
    ``VLLMBackend(...)`` explicitly and pass the instance.

    Args:
        vlm: Backend instance (returned as-is) or model-id string.
        vlm_kwargs: Constructor kwargs, forwarded only on the string path.

    Raises:
        ValueError: If ``vlm`` is already an instance but ``vlm_kwargs``
            were provided (they would be silently dropped otherwise).
    """
    if isinstance(vlm, VLMBackend):
        if vlm_kwargs:
            raise ValueError(
                "vlm_kwargs are only used when vlm is a model string; "
                f"got a {type(vlm).__name__} instance plus kwargs "
                f"{sorted(vlm_kwargs)}"
            )
        return vlm
    if vlm.startswith(OPENAI_MODEL_IDENTIFIER):
        from .openai import OpenAIVLM

        return OpenAIVLM(vlm[len(OPENAI_MODEL_IDENTIFIER):], **vlm_kwargs)
    from .huggingface import HuggingFaceVLM

    return HuggingFaceVLM(vlm, **vlm_kwargs)
```

- [ ] **Step 5: Rewrite `s3e/backends/__init__.py` with lazy heavy exports**

```python
# s3e/backends/__init__.py
"""VLM backends: the abstract contract plus concrete implementations.

``VLMBackend``/``VLMOutput`` and :func:`resolve_backend` import without any
optional dependency. The concrete backends are exposed lazily so that
``import s3e.backends`` never imports torch, openai, or vllm.
"""

import importlib

from .backend import VLMBackend, VLMOutput
from .resolve import OPENAI_MODEL_IDENTIFIER, resolve_backend

__all__ = [
    "VLMBackend",
    "VLMOutput",
    "resolve_backend",
    "OPENAI_MODEL_IDENTIFIER",
    "HuggingFaceVLM",
    "OpenAIVLM",
    "VLLMBackend",
]

_LAZY_BACKENDS = {
    "HuggingFaceVLM": ".huggingface",
    "OpenAIVLM": ".openai",
    "VLLMBackend": ".vllm",
}


def __getattr__(name: str):
    """Lazily expose backends that need optional dependencies."""
    if name in _LAZY_BACKENDS:
        module = importlib.import_module(_LAZY_BACKENDS[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

- [ ] **Step 6: Route optional-dependency errors through `require`**

In each of `s3e/backends/huggingface.py`, `s3e/backends/openai.py`, `s3e/backends/vllm.py`: immediately after the module docstring and stdlib imports, and before the heavy imports, add the matching call and delete any existing `try/except ImportError` shim around those imports (keep narrow shims that handle *version* differences, e.g. transformers API compatibility branches):

```python
from .._deps import require

require("torch", "hf", "HuggingFaceVLM")          # huggingface.py
require("openai", "openai", "OpenAIVLM")           # openai.py
require("vllm", "vllm", "VLLMBackend")             # vllm.py
```

Update `s3e/__init__.py`'s existing lazy `__getattr__` so `VLLMBackend` (and now also `HuggingFaceVLM`, `OpenAIVLM`) resolve through `s3e.backends`; keep them out of eager imports:

```python
from .backends import VLMBackend, VLMOutput, resolve_backend

_LAZY_TOP_LEVEL = {"HuggingFaceVLM", "OpenAIVLM", "VLLMBackend"}


def __getattr__(name: str):
    """Lazily expose optional integrations without importing their packages."""
    if name in _LAZY_TOP_LEVEL:
        import s3e.backends as _backends

        return getattr(_backends, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

(Adjust the top of `s3e/__init__.py`: drop the eager `from .vlm import ... HuggingFaceVLM, OpenAIVLM` line; add `resolve_backend` to `__all__`.)

- [ ] **Step 7: Run the whole fast suite**

Run: `pytest tests/backends/test_resolve.py -v` → PASS
Run: `pytest -m "not slow" -q` → all pass (old `tests/test_vlm_backends.py` now imports `s3e.backends`).

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor: rename s3e.vlm to s3e.backends; add public resolve_backend"
```

---

### Task 4: Shared fake backend and backend contract suite

**Files:**
- Create: `tests/fakes.py`, `tests/backends/test_contract.py`
- Modify: `tests/test_vlm_backends.py` (replace its local `FakeVLM` with the shared one)

**Interfaces:**
- Consumes: `VLMBackend`, `VLMOutput` from Task 3.
- Produces: `tests.fakes.FakeVLM` — the one fake used by every later task's tests. Constructor: `FakeVLM(token_probs=None, text=None, argmax_in_interest=True)`; records every call in `self.calls` (list of dicts with keys `images`, `prompts`, `system_prompt`, `generate`, `interest_tokens`, `inference_kwargs`); `script_responses(mapping)` sets per-prompt-substring responses. Also `make_backend_contract_tests(backend_factory)` pattern: `tests/backends/test_contract.py` holds the parametrized contract suite class `BackendContract` that concrete-backend test modules subclass.

- [ ] **Step 1: Write `tests/fakes.py` (this is test infrastructure — written directly, exercised by the contract suite below)**

```python
# tests/fakes.py
"""Shared fake VLM backend implementing the full VLMBackend contract.

Used by engine, estimator, calibration, and consumer tests. Honors the
interest_tokens contract: when interest tokens are requested, the returned
token_probs contains exactly those keys (absent tokens get 0.0).
"""

from s3e.backends import VLMBackend, VLMOutput


class FakeVLM(VLMBackend):
    """Deterministic fake backend.

    Args:
        token_probs: Default token-string -> probability mapping returned
            for every query (before interest-token filtering).
        text: Default generated text returned when ``generate=True``.
        argmax_in_interest: Value reported when interest tokens are given.
    """

    def __init__(self, token_probs=None, text=None, argmax_in_interest=True):
        self.token_probs = dict(token_probs or {"yes": 0.7, "no": 0.2})
        self.text = text
        self.argmax_in_interest = argmax_in_interest
        self.calls: list[dict] = []
        self._scripted: dict[str, dict] = {}

    def script_responses(self, mapping: dict[str, dict]) -> None:
        """Per-query overrides: prompt-substring -> token_probs mapping."""
        self._scripted.update(mapping)

    def _probs_for(self, prompt: str) -> dict[str, float]:
        for needle, probs in self._scripted.items():
            if needle in prompt:
                return dict(probs)
        return dict(self.token_probs)

    def query(self, images, prompt, system_prompt=None, generate=False,
              interest_tokens=None, **inference_kwargs):
        self.calls.append(
            {
                "images": list(images),
                "prompts": [prompt],
                "system_prompt": system_prompt,
                "generate": generate,
                "interest_tokens": (
                    None if interest_tokens is None else list(interest_tokens)
                ),
                "inference_kwargs": dict(inference_kwargs),
            }
        )
        probs = self._probs_for(prompt)
        if interest_tokens is not None:
            token_probs = {t: probs.get(t, 0.0) for t in interest_tokens}
            argmax = self.argmax_in_interest
        else:
            token_probs = probs
            argmax = None
        return VLMOutput(
            token_probs=token_probs,
            text=self.text if generate else None,
            argmax_in_interest=argmax,
        )
```

- [ ] **Step 2: Write the failing contract suite**

```python
# tests/backends/test_contract.py
"""Contract tests every VLMBackend implementation must pass.

Concrete backend test modules subclass ``BackendContract`` and provide a
``make_backend`` fixture. This module applies the suite to FakeVLM so the
fake can never drift from the contract real backends implement.
"""

from PIL import Image
import pytest

from s3e.backends import VLMOutput

from conftest import make_blank_image  # add helper if absent; see step 3
from fakes import FakeVLM


class BackendContract:
    """Behavioral contract for VLMBackend implementations."""

    @pytest.fixture
    def images(self):
        return [make_blank_image()]

    def test_query_returns_vlm_output(self, make_backend, images):
        out = make_backend().query(images, "Is it red?")
        assert isinstance(out, VLMOutput)

    def test_interest_tokens_keys_are_exactly_the_request(self, make_backend, images):
        out = make_backend().query(
            images, "Is it red?", interest_tokens=["yes", "no", "zzz_absent"]
        )
        assert set(out.token_probs) == {"yes", "no", "zzz_absent"}
        assert out.token_probs["zzz_absent"] == 0.0
        assert out.argmax_in_interest is not None

    def test_interest_masses_are_probabilities(self, make_backend, images):
        out = make_backend().query(images, "q", interest_tokens=["yes", "no"])
        for mass in out.token_probs.values():
            assert 0.0 <= mass <= 1.0
        assert sum(out.token_probs.values()) <= 1.0 + 1e-9

    def test_query_batch_matches_sequential_query(self, make_backend, images):
        backend = make_backend()
        batch = backend.query_batch(images, ["a", "b"], interest_tokens=["yes", "no"])
        singles = [
            make_backend().query(images, p, interest_tokens=["yes", "no"])
            for p in ("a", "b")
        ]
        assert [o.token_probs for o in batch] == [o.token_probs for o in singles]

    def test_generate_mode_returns_text(self, make_backend, images):
        out = make_backend().query(images, "q", generate=True)
        assert out.text is None or isinstance(out.text, str)

    def test_multi_image_scene_accepted(self, make_backend):
        scene = [make_blank_image(), make_blank_image()]
        out = make_backend().query(scene, "q", interest_tokens=["yes", "no"])
        assert isinstance(out, VLMOutput)


class TestFakeVLMContract(BackendContract):
    @pytest.fixture
    def make_backend(self):
        return lambda: FakeVLM(text="yes")
```

- [ ] **Step 3: Ensure `make_blank_image` exists in `tests/conftest.py`**

If `tests/conftest.py` has no image helper, add:

```python
from PIL import Image as PILImage


def make_blank_image(size=(8, 8)):
    """Tiny RGB image for backend tests."""
    return PILImage.new("RGB", size, color=(127, 127, 127))
```

- [ ] **Step 4: Run and verify**

Run: `pytest tests/backends/test_contract.py -v` → PASS (fake satisfies the contract; if any contract test fails, fix `FakeVLM`, not the test).

- [ ] **Step 5: Point existing backend tests at the shared fake**

In `tests/test_vlm_backends.py`: delete the local `class FakeVLM(VLMBackend)` (lines ~12–29) and add `from fakes import FakeVLM`. Adapt call sites if the local fake's constructor differed (check its definition before deleting; keep its response semantics by passing `token_probs=`/`text=`).

Additionally, subclass the contract suite for the real backends inside their existing test classes' modules (mocked where the current tests are mocked, `@pytest.mark.slow` where they are slow) — add at the bottom of `tests/test_vlm_backends.py`:

```python
from test_contract_support import *  # only if needed; otherwise import directly:
```

```python
from tests.backends.test_contract import BackendContract


@pytest.mark.slow
class TestHuggingFaceVLMContract(BackendContract):
    @pytest.fixture
    def make_backend(self):
        from s3e.backends import HuggingFaceVLM
        backend = HuggingFaceVLM("HuggingFaceTB/SmolVLM-256M-Instruct")
        return lambda: backend
```

(Use the same small model id the existing slow HF integration tests use — check `TestHuggingFaceVLMIntegration` and reuse its model and any fixtures. Do the same for `VLLMBackend` next to `TestVLLMBackendIntegration`, marked slow. OpenAI gets no live contract subclass — its mocked tests stay as-is.)

Note: `tests/conftest.py` also defines its own legacy fake VLM class (lines ~39–92) with `fake_vlm`/`fake_images` fixtures used by the old estimator monolith. Leave that one alone here — it dies with the monolith in Task 11. Only `tests/test_vlm_backends.py`'s local fake is replaced now.

- [ ] **Step 6: Run the fast suite**

Run: `pytest -m "not slow" -q` → all pass.

- [ ] **Step 7: Commit**

```bash
git add tests/fakes.py tests/backends/test_contract.py tests/test_vlm_backends.py tests/conftest.py
git commit -m "test: shared FakeVLM and backend contract suite"
```

---

### Task 5: Answer spaces (`s3e/engine/answers.py`)

**Files:**
- Create: `s3e/engine/__init__.py`, `s3e/engine/answers.py`
- Test: `tests/engine/__init__.py` (empty), `tests/engine/test_answers.py`

**Interfaces:**
- Consumes: `VLMOutput` from `s3e.backends`.
- Produces (exact names later tasks use):
  - `expand_token_variants(label: str) -> tuple[str, ...]`
  - `AnswerOption(label: str, tokens: tuple[str, ...])`, `AnswerOption.make(label, tokens=None)`
  - `ScoredMasses(masses: dict[str, float], null_mass: float, unassigned_mass: float)`
  - `AnswerSpace(options, null_option=None)` with `.options`, `.null_option`, `.labels`, `.interest_tokens`, `.score(output: VLMOutput, scoring: str) -> ScoredMasses`, `.to_dict()`, `AnswerSpace.from_dict(d)`
  - `BinaryAnswers(true_label="yes", false_label="no", *, true_tokens=None, false_tokens=None, null_label="unknown", null_tokens=None)` with `.true_label`, `.false_label`
  - `CategoricalAnswers(options, *, null_label="unknown", null_tokens=None)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/engine/test_answers.py
"""Tests for answer options and answer spaces."""

import pytest

from s3e.backends import VLMOutput
from s3e.engine import (
    AnswerOption,
    AnswerSpace,
    BinaryAnswers,
    CategoricalAnswers,
    expand_token_variants,
)


class TestExpandTokenVariants:
    def test_case_and_leading_space_variants(self):
        variants = set(expand_token_variants("red"))
        assert {"red", "Red", "RED", " red", " Red", " RED"} <= variants

    def test_multiword_label_kept_verbatim_among_variants(self):
        assert "dark blue" in expand_token_variants("dark blue")


class TestAnswerOption:
    def test_make_auto_expands(self):
        option = AnswerOption.make("yes")
        assert "yes" in option.tokens and "Yes" in option.tokens

    def test_make_explicit_tokens_win(self):
        option = AnswerOption.make("yes", tokens=["y"])
        assert option.tokens == ("y",)
        assert option.label == "yes"


class TestBinaryAnswers:
    def test_default_yes_no(self):
        space = BinaryAnswers()
        assert space.true_label == "yes"
        assert space.false_label == "no"
        assert "Yes" in space.options[0].tokens

    def test_relabel_true_false(self):
        space = BinaryAnswers("true", "false")
        assert space.true_label == "true"
        assert "True" in space.options[0].tokens
        assert "FALSE" in space.options[1].tokens

    def test_explicit_token_overrides(self):
        space = BinaryAnswers(true_tokens=["yep"], false_tokens=["nope"])
        assert space.options[0].tokens == ("yep",)

    def test_null_tokens_create_null_option(self):
        space = BinaryAnswers(null_tokens=["unknown", "Unknown"])
        assert space.null_option is not None
        assert space.null_option.label == "unknown"
        assert "unknown" in space.interest_tokens

    def test_overlapping_tokens_rejected(self):
        with pytest.raises(ValueError, match="overlap"):
            BinaryAnswers(true_tokens=["yes"], false_tokens=["yes"])


class TestLogprobScoring:
    def test_masses_summed_per_option(self):
        space = BinaryAnswers(true_tokens=["yes", "Yes"], false_tokens=["no"])
        output = VLMOutput(token_probs={"yes": 0.5, "Yes": 0.2, "no": 0.1})
        scored = space.score(output, scoring="logprobs")
        assert scored.masses == {"yes": 0.7, "no": 0.1}
        assert scored.null_mass == 0.0
        assert scored.unassigned_mass == pytest.approx(0.2)

    def test_null_option_mass_separated(self):
        space = BinaryAnswers(
            true_tokens=["yes"], false_tokens=["no"], null_tokens=["unknown"]
        )
        output = VLMOutput(token_probs={"yes": 0.2, "no": 0.1, "unknown": 0.6})
        scored = space.score(output, scoring="logprobs")
        assert scored.null_mass == pytest.approx(0.6)
        assert scored.masses == {"yes": 0.2, "no": 0.1}


class TestTextMatchScoring:
    def test_matching_option_gets_full_mass(self):
        space = BinaryAnswers()
        output = VLMOutput(text="Yes, it is.")
        scored = space.score(output, scoring="text_match")
        assert scored.masses["yes"] == 1.0
        assert scored.masses["no"] == 0.0
        assert scored.unassigned_mass == 0.0

    def test_no_match_is_fully_unassigned(self):
        space = BinaryAnswers()
        output = VLMOutput(text="I cannot tell.")
        scored = space.score(output, scoring="text_match")
        assert scored.masses == {"yes": 0.0, "no": 0.0}
        assert scored.unassigned_mass == 1.0

    def test_null_option_matches_text(self):
        space = BinaryAnswers(null_tokens=["unknown"])
        output = VLMOutput(text="unknown")
        scored = space.score(output, scoring="text_match")
        assert scored.null_mass == 1.0


class TestCategoricalAnswers:
    def test_labels_from_strings(self):
        space = CategoricalAnswers(["red", "green", "blue"])
        assert space.labels == ["red", "green", "blue"]

    def test_scoring_over_three_options(self):
        space = CategoricalAnswers(["red", "green", "blue"])
        output = VLMOutput(token_probs={"red": 0.6, " green": 0.3, "blue": 0.05})
        scored = space.score(output, scoring="logprobs")
        assert scored.masses["red"] == pytest.approx(0.6)
        assert scored.masses["green"] == pytest.approx(0.3)


class TestSerialization:
    @pytest.mark.parametrize(
        "space",
        [
            BinaryAnswers("true", "false", null_tokens=["unknown"]),
            CategoricalAnswers(["red", "green"]),
        ],
    )
    def test_round_trip(self, space):
        restored = AnswerSpace.from_dict(space.to_dict())
        assert type(restored) is type(space)
        assert restored.to_dict() == space.to_dict()

    def test_binary_round_trip_keeps_semantics(self):
        restored = AnswerSpace.from_dict(BinaryAnswers("true", "false").to_dict())
        assert restored.true_label == "true"


class TestUnknownScoring:
    def test_unknown_scoring_mode_rejected(self):
        with pytest.raises(ValueError, match="scoring"):
            BinaryAnswers().score(VLMOutput(), scoring="magic")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/engine/test_answers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 's3e.engine'`

- [ ] **Step 3: Implement `s3e/engine/answers.py`**

```python
# s3e/engine/answers.py
"""Answer spaces: what counts as an answer and how model output scores it.

An :class:`AnswerOption` is a label plus the token strings that express it.
An :class:`AnswerSpace` is an ordered set of options (plus an optional
explicit null/abstain option) that can score a :class:`VLMOutput` either
from token masses ("logprobs") or from generated text ("text_match").
"""

from collections.abc import Sequence
from dataclasses import dataclass

from ..backends import VLMOutput

SCORING_MODES = ("logprobs", "text_match")


def expand_token_variants(label: str) -> tuple[str, ...]:
    """Case and leading-space variants of a label, label's own casing first."""
    seen: list[str] = []
    for base in (label, label.lower(), label.capitalize(), label.upper()):
        for variant in (base, " " + base):
            if variant not in seen:
                seen.append(variant)
    return tuple(seen)


@dataclass(frozen=True)
class AnswerOption:
    """One admissible answer: a label plus its accepted token strings."""

    label: str
    tokens: tuple[str, ...]

    @classmethod
    def make(cls, label: str, tokens: "Sequence[str] | None" = None) -> "AnswerOption":
        """Build an option, auto-expanding the label when tokens are omitted."""
        if tokens is None:
            return cls(label, expand_token_variants(label))
        return cls(label, tuple(tokens))


@dataclass(frozen=True)
class ScoredMasses:
    """Raw scoring result: per-option masses plus null and unassigned mass."""

    masses: dict[str, float]
    null_mass: float
    unassigned_mass: float


class AnswerSpace:
    """Ordered set of answer options with scoring behavior."""

    def __init__(
        self,
        options: Sequence[AnswerOption],
        null_option: "AnswerOption | None" = None,
    ):
        self.options = tuple(options)
        self.null_option = null_option
        labels = [o.label for o in self.options]
        if len(set(labels)) != len(labels):
            raise ValueError(f"Duplicate answer labels: {labels}")
        all_options = self.options + ((null_option,) if null_option else ())
        seen_tokens: dict[str, str] = {}
        for option in all_options:
            for token in option.tokens:
                if token in seen_tokens and seen_tokens[token] != option.label:
                    raise ValueError(
                        f"Token {token!r} would overlap between options "
                        f"{seen_tokens[token]!r} and {option.label!r}"
                    )
                seen_tokens[token] = option.label

    @property
    def labels(self) -> list[str]:
        return [o.label for o in self.options]

    @property
    def interest_tokens(self) -> list[str]:
        tokens: list[str] = []
        for option in self.options + (
            (self.null_option,) if self.null_option else ()
        ):
            tokens.extend(option.tokens)
        return tokens

    def score(self, output: VLMOutput, scoring: str) -> ScoredMasses:
        """Score a backend output into per-option masses."""
        if scoring == "logprobs":
            return self._score_logprobs(output)
        if scoring == "text_match":
            return self._score_text(output)
        raise ValueError(
            f"Unknown scoring mode {scoring!r}; expected one of {SCORING_MODES}"
        )

    def _score_logprobs(self, output: VLMOutput) -> ScoredMasses:
        probs = output.token_probs
        masses = {
            o.label: sum(probs.get(t, 0.0) for t in o.tokens) for o in self.options
        }
        null_mass = (
            sum(probs.get(t, 0.0) for t in self.null_option.tokens)
            if self.null_option
            else 0.0
        )
        unassigned = max(0.0, 1.0 - sum(masses.values()) - null_mass)
        return ScoredMasses(masses=masses, null_mass=null_mass, unassigned_mass=unassigned)

    def _score_text(self, output: VLMOutput) -> ScoredMasses:
        text = output.text or ""
        matched: "str | None" = None
        null_matched = False
        candidates = list(self.options) + (
            [self.null_option] if self.null_option else []
        )
        for option in candidates:
            if any(token.strip() and token.strip() in text for token in option.tokens):
                if self.null_option and option is self.null_option:
                    null_matched = True
                else:
                    matched = option.label
                break
        masses = {o.label: (1.0 if o.label == matched else 0.0) for o in self.options}
        null_mass = 1.0 if null_matched else 0.0
        unassigned = 0.0 if (matched or null_matched) else 1.0
        return ScoredMasses(masses=masses, null_mass=null_mass, unassigned_mass=unassigned)

    def to_dict(self) -> dict:
        return {
            "type": "categorical",
            "options": [
                {"label": o.label, "tokens": list(o.tokens)} for o in self.options
            ],
            "null_option": (
                {
                    "label": self.null_option.label,
                    "tokens": list(self.null_option.tokens),
                }
                if self.null_option
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AnswerSpace":
        if data["type"] == "binary":
            return BinaryAnswers._from_dict(data)
        if data["type"] == "categorical":
            return CategoricalAnswers._from_dict(data)
        raise ValueError(f"Unknown answer space type: {data['type']!r}")


def _null_option_from(
    null_label: str, null_tokens: "Sequence[str] | None"
) -> "AnswerOption | None":
    if null_tokens is None:
        return None
    return AnswerOption.make(null_label, null_tokens)


class BinaryAnswers(AnswerSpace):
    """Two-option answer space with boolean semantics (true first)."""

    def __init__(
        self,
        true_label: str = "yes",
        false_label: str = "no",
        *,
        true_tokens: "Sequence[str] | None" = None,
        false_tokens: "Sequence[str] | None" = None,
        null_label: str = "unknown",
        null_tokens: "Sequence[str] | None" = None,
    ):
        self.true_label = true_label
        self.false_label = false_label
        super().__init__(
            [
                AnswerOption.make(true_label, true_tokens),
                AnswerOption.make(false_label, false_tokens),
            ],
            null_option=_null_option_from(null_label, null_tokens),
        )

    def to_dict(self) -> dict:
        data = super().to_dict()
        data["type"] = "binary"
        data["true_label"] = self.true_label
        data["false_label"] = self.false_label
        return data

    @classmethod
    def _from_dict(cls, data: dict) -> "BinaryAnswers":
        options = data["options"]
        null = data.get("null_option")
        return cls(
            data["true_label"],
            data["false_label"],
            true_tokens=options[0]["tokens"],
            false_tokens=options[1]["tokens"],
            null_label=null["label"] if null else "unknown",
            null_tokens=null["tokens"] if null else None,
        )


class CategoricalAnswers(AnswerSpace):
    """N-option answer space built from labels or explicit options."""

    def __init__(
        self,
        options: "Sequence[str | AnswerOption]",
        *,
        null_label: str = "unknown",
        null_tokens: "Sequence[str] | None" = None,
    ):
        built = [
            o if isinstance(o, AnswerOption) else AnswerOption.make(o)
            for o in options
        ]
        super().__init__(built, null_option=_null_option_from(null_label, null_tokens))

    @classmethod
    def _from_dict(cls, data: dict) -> "CategoricalAnswers":
        null = data.get("null_option")
        return cls(
            [AnswerOption(o["label"], tuple(o["tokens"])) for o in data["options"]],
            null_label=null["label"] if null else "unknown",
            null_tokens=null["tokens"] if null else None,
        )
```

- [ ] **Step 4: Create `s3e/engine/__init__.py`**

```python
# s3e/engine/__init__.py
"""Query engine layer: answer spaces, lazy results, and the QueryEngine."""

from .answers import (
    AnswerOption,
    AnswerSpace,
    BinaryAnswers,
    CategoricalAnswers,
    ScoredMasses,
    expand_token_variants,
)

__all__ = [
    "AnswerOption",
    "AnswerSpace",
    "BinaryAnswers",
    "CategoricalAnswers",
    "ScoredMasses",
    "expand_token_variants",
]
```

(Extend `__all__` in Tasks 6 and 7 as `results.py` and `engine.py` land.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/engine/test_answers.py -v` → PASS
Run: `pytest -m "not slow" -q` → all pass.

- [ ] **Step 6: Commit**

```bash
git add s3e/engine tests/engine
git commit -m "feat: answer spaces with multi-token surface forms"
```

---

### Task 6: Lazy results (`s3e/engine/results.py`)

**Files:**
- Create: `s3e/engine/results.py`
- Modify: `s3e/engine/__init__.py`
- Test: `tests/engine/test_results.py`

**Interfaces:**
- Consumes: `AnswerSpace`, `BinaryAnswers`, `ScoredMasses` (Task 5); `VLMOutput` (Task 3).
- Produces (used verbatim by Tasks 7–13):
  - `EPS = 1e-12`
  - `Prediction(query, masses, null_mass, unassigned_mass, answers, *, text=None, argmax_in_interest=None, raw=None, probability_override=None)` with properties `.answer`, `.probability`, `.score`, `.null_dominated`, `.distribution()`, `.confident(threshold)`, `.with_probability(p)`, `.to_dict()`, `Prediction.from_dict(d)`
  - `PredictionSet(predictions: dict[str, Prediction])` — Mapping; `.probabilities()`, `.to_state(confidence=0.5)`, `.where(fn)`, `.to_dict()`, `PredictionSet.from_dict(d)`, `PredictionSet.average(sets)`
  - `PREDICTION_SET_FORMAT_VERSION = 1`

- [ ] **Step 1: Write the failing tests**

```python
# tests/engine/test_results.py
"""Tests for lazy Prediction / PredictionSet objects."""

import json
import math

import pytest

from s3e.engine import BinaryAnswers, CategoricalAnswers, Prediction, PredictionSet


def make_prediction(true_mass=0.7, false_mass=0.2, null_mass=0.0, **kwargs):
    space = kwargs.pop("answers", BinaryAnswers(null_tokens=["unknown"] if null_mass else None))
    unassigned = max(0.0, 1.0 - true_mass - false_mass - null_mass)
    return Prediction(
        query="on(a,b)",
        masses={space.true_label: true_mass, space.false_label: false_mass},
        null_mass=null_mass,
        unassigned_mass=unassigned,
        answers=space,
        **kwargs,
    )


class TestPrediction:
    def test_probability_normalizes_over_binary_masses(self):
        p = make_prediction(0.7, 0.2)
        assert p.probability == pytest.approx(0.7 / 0.9, rel=1e-6)

    def test_probability_with_zero_masses_is_half(self):
        p = make_prediction(0.0, 0.0)
        assert p.probability == pytest.approx(0.5)

    def test_answer_is_bool_for_binary(self):
        assert make_prediction(0.7, 0.2).answer is True
        assert make_prediction(0.1, 0.8).answer is False

    def test_score_is_grouped_log_odds(self):
        p = make_prediction(0.7, 0.2)
        assert p.score == pytest.approx(math.log((0.7 + 1e-12) / (0.2 + 1e-12)))

    def test_null_dominated_when_null_beats_all_options(self):
        assert make_prediction(0.2, 0.1, null_mass=0.6).null_dominated is True
        assert make_prediction(0.7, 0.2, null_mass=0.05).null_dominated is False

    def test_answer_none_when_null_dominated(self):
        assert make_prediction(0.2, 0.1, null_mass=0.6).answer is None

    def test_confident(self):
        p = make_prediction(0.9, 0.05)
        assert p.confident(0.8) is True
        assert p.confident(0.99) is False

    def test_probability_override_wins(self):
        p = make_prediction(0.7, 0.2).with_probability(0.42)
        assert p.probability == pytest.approx(0.42)
        # original untouched
        assert make_prediction(0.7, 0.2).probability != pytest.approx(0.42)

    def test_categorical_answer_is_argmax_label(self):
        space = CategoricalAnswers(["red", "green"])
        p = Prediction(
            query="color(a)",
            masses={"red": 0.1, "green": 0.6},
            null_mass=0.0,
            unassigned_mass=0.3,
            answers=space,
        )
        assert p.answer == "green"

    def test_categorical_probability_raises(self):
        space = CategoricalAnswers(["red", "green"])
        p = Prediction(
            query="q", masses={"red": 0.5, "green": 0.5},
            null_mass=0.0, unassigned_mass=0.0, answers=space,
        )
        with pytest.raises(ValueError, match="binary"):
            p.probability

    def test_distribution_normalizes(self):
        p = make_prediction(0.6, 0.2)
        dist = p.distribution()
        assert sum(dist.values()) == pytest.approx(1.0)
        assert dist["yes"] == pytest.approx(0.75)


class TestPredictionSet:
    def make_set(self):
        return PredictionSet(
            {
                "on(a,b)": make_prediction(0.9, 0.05),
                "on(b,a)": make_prediction(0.1, 0.85),
                "clear(a)": make_prediction(0.5, 0.45),
            }
        )

    def test_mapping_protocol(self):
        results = self.make_set()
        assert len(results) == 3
        assert list(results) == ["on(a,b)", "on(b,a)", "clear(a)"]
        assert results["on(a,b)"].answer is True

    def test_probabilities(self):
        probs = self.make_set().probabilities()
        assert set(probs) == {"on(a,b)", "on(b,a)", "clear(a)"}
        assert probs["on(a,b)"] > 0.9

    def test_to_state_three_way(self):
        state = self.make_set().to_state(confidence=0.8)
        assert state["on(a,b)"] is True
        assert state["on(b,a)"] is False
        assert state["clear(a)"] is None  # not confident either way

    def test_to_state_null_dominated_is_none(self):
        results = PredictionSet({"p(a)": make_prediction(0.2, 0.1, null_mass=0.6)})
        assert results.to_state()["p(a)"] is None

    def test_where(self):
        confident = self.make_set().where(lambda p: p.confident(0.8))
        assert set(confident) == {"on(a,b)", "on(b,a)"}


class TestSerialization:
    def test_round_trip_via_json(self):
        results = PredictionSet({"on(a,b)": make_prediction(0.7, 0.2)})
        payload = json.loads(json.dumps(results.to_dict()))
        restored = PredictionSet.from_dict(payload)
        assert restored["on(a,b)"].probability == pytest.approx(
            results["on(a,b)"].probability
        )
        assert restored["on(a,b)"].answer is True

    def test_format_version_present_and_checked(self):
        payload = PredictionSet({"q": make_prediction()}).to_dict()
        assert payload["format_version"] == 1
        payload["format_version"] = 999
        with pytest.raises(ValueError, match="format_version"):
            PredictionSet.from_dict(payload)

    def test_raw_not_serialized(self):
        p = make_prediction(raw=object())
        assert "raw" not in p.to_dict()


class TestAverage:
    def test_average_means_masses(self):
        a = PredictionSet({"q": make_prediction(0.8, 0.1)})
        b = PredictionSet({"q": make_prediction(0.4, 0.5)})
        avg = PredictionSet.average([a, b])
        assert avg["q"].masses["yes"] == pytest.approx(0.6)
        assert avg["q"].masses["no"] == pytest.approx(0.3)

    def test_average_requires_same_queries(self):
        a = PredictionSet({"q1": make_prediction()})
        b = PredictionSet({"q2": make_prediction()})
        with pytest.raises(ValueError, match="same queries"):
            PredictionSet.average([a, b])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/engine/test_results.py -v`
Expected: FAIL with `ImportError: cannot import name 'Prediction'`

- [ ] **Step 3: Implement `s3e/engine/results.py`**

```python
# s3e/engine/results.py
"""Lazy prediction objects: store masses, derive everything else on demand."""

import math
from collections.abc import Iterator, Mapping, Sequence
from functools import cached_property

from .answers import AnswerSpace, BinaryAnswers

EPS = 1e-12
PREDICTION_SET_FORMAT_VERSION = 1


class Prediction:
    """One query's outcome. Immutable; derived values are cached lazily.

    Stores per-option probability masses, the explicit-null option's mass,
    and the unassigned remainder. ``raw`` holds the backend
    :class:`~s3e.backends.VLMOutput` only when the engine was asked to keep
    it and is never serialized.
    """

    def __init__(
        self,
        query: str,
        masses: Mapping[str, float],
        null_mass: float,
        unassigned_mass: float,
        answers: AnswerSpace,
        *,
        text: "str | None" = None,
        argmax_in_interest: "bool | None" = None,
        raw=None,
        probability_override: "float | None" = None,
    ):
        self.query = query
        self.masses = dict(masses)
        self.null_mass = null_mass
        self.unassigned_mass = unassigned_mass
        self.answers = answers
        self.text = text
        self.argmax_in_interest = argmax_in_interest
        self.raw = raw
        self.probability_override = probability_override

    @cached_property
    def null_dominated(self) -> bool:
        """True when the explicit null option out-masses every answer option."""
        if not self.masses:
            return False
        return self.null_mass > max(self.masses.values())

    @cached_property
    def answer(self):
        """Argmax label; a bool for binary spaces; None when null-dominated."""
        if self.null_dominated:
            return None
        label = max(self.masses, key=self.masses.__getitem__)
        if isinstance(self.answers, BinaryAnswers):
            return label == self.answers.true_label
        return label

    @cached_property
    def probability(self) -> float:
        """Normalized P(true) for binary spaces (override wins when set)."""
        if self.probability_override is not None:
            return self.probability_override
        self._require_binary("probability")
        true_mass = self.masses[self.answers.true_label]
        false_mass = self.masses[self.answers.false_label]
        return (true_mass + EPS) / (true_mass + false_mass + 2 * EPS)

    @cached_property
    def score(self) -> float:
        """Grouped log-odds log(true_mass / false_mass) for binary spaces."""
        self._require_binary("score")
        true_mass = self.masses[self.answers.true_label]
        false_mass = self.masses[self.answers.false_label]
        return math.log((true_mass + EPS) / (false_mass + EPS))

    def distribution(self) -> dict[str, float]:
        """Masses normalized over the answer options."""
        total = sum(self.masses.values())
        if total <= 0.0:
            uniform = 1.0 / len(self.masses)
            return {label: uniform for label in self.masses}
        return {label: mass / total for label, mass in self.masses.items()}

    def confident(self, threshold: float) -> bool:
        """Whether either boolean outcome reaches the threshold."""
        return self.probability >= threshold or (1.0 - self.probability) >= threshold

    def with_probability(self, probability: float) -> "Prediction":
        """Copy of this prediction with an overriding probability (calibration)."""
        return Prediction(
            self.query,
            self.masses,
            self.null_mass,
            self.unassigned_mass,
            self.answers,
            text=self.text,
            argmax_in_interest=self.argmax_in_interest,
            raw=self.raw,
            probability_override=probability,
        )

    def _require_binary(self, what: str) -> None:
        if not isinstance(self.answers, BinaryAnswers):
            raise ValueError(
                f"{what} is only defined for binary answer spaces; "
                f"this prediction uses {type(self.answers).__name__}"
            )

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "masses": dict(self.masses),
            "null_mass": self.null_mass,
            "unassigned_mass": self.unassigned_mass,
            "text": self.text,
            "argmax_in_interest": self.argmax_in_interest,
            "probability_override": self.probability_override,
            "answers": self.answers.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Prediction":
        return cls(
            query=data["query"],
            masses=data["masses"],
            null_mass=data["null_mass"],
            unassigned_mass=data["unassigned_mass"],
            answers=AnswerSpace.from_dict(data["answers"]),
            text=data.get("text"),
            argmax_in_interest=data.get("argmax_in_interest"),
            probability_override=data.get("probability_override"),
        )


class PredictionSet(Mapping):
    """Ordered mapping of query (or predicate) to :class:`Prediction`."""

    def __init__(self, predictions: Mapping[str, Prediction]):
        self._predictions = dict(predictions)

    def __getitem__(self, key: str) -> Prediction:
        return self._predictions[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._predictions)

    def __len__(self) -> int:
        return len(self._predictions)

    def probabilities(self) -> dict[str, float]:
        """Per-query P(true) (binary spaces)."""
        return {key: p.probability for key, p in self._predictions.items()}

    def to_state(self, confidence: float = 0.5) -> "dict[str, bool | None]":
        """Threshold probabilities into a three-valued boolean state.

        True when P(true) >= confidence, False when P(false) >= confidence,
        None otherwise or when the prediction is null-dominated.
        """
        state: dict[str, bool | None] = {}
        for key, p in self._predictions.items():
            if p.null_dominated:
                state[key] = None
            elif p.probability >= confidence:
                state[key] = True
            elif (1.0 - p.probability) >= confidence:
                state[key] = False
            else:
                state[key] = None
        return state

    def where(self, predicate) -> "PredictionSet":
        """Subset of predictions for which ``predicate(prediction)`` is true."""
        return PredictionSet(
            {k: p for k, p in self._predictions.items() if predicate(p)}
        )

    def to_dict(self) -> dict:
        return {
            "format_version": PREDICTION_SET_FORMAT_VERSION,
            "predictions": {k: p.to_dict() for k, p in self._predictions.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PredictionSet":
        version = data.get("format_version")
        if version != PREDICTION_SET_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported PredictionSet format_version: {version!r} "
                f"(expected {PREDICTION_SET_FORMAT_VERSION})"
            )
        return cls(
            {k: Prediction.from_dict(d) for k, d in data["predictions"].items()}
        )

    @classmethod
    def average(cls, sets: "Sequence[PredictionSet]") -> "PredictionSet":
        """Mean of stored masses across prediction sets over the same queries."""
        if not sets:
            raise ValueError("Expected at least one PredictionSet to average.")
        keys = list(sets[0])
        for other in sets[1:]:
            if list(other) != keys:
                raise ValueError("All PredictionSets must cover the same queries.")
        count = len(sets)
        averaged: dict[str, Prediction] = {}
        for key in keys:
            members = [s[key] for s in sets]
            first = members[0]
            averaged[key] = Prediction(
                query=first.query,
                masses={
                    label: sum(m.masses[label] for m in members) / count
                    for label in first.masses
                },
                null_mass=sum(m.null_mass for m in members) / count,
                unassigned_mass=sum(m.unassigned_mass for m in members) / count,
                answers=first.answers,
                argmax_in_interest=first.argmax_in_interest,
            )
        return cls(averaged)
```

- [ ] **Step 4: Extend `s3e/engine/__init__.py`**

Add to the imports and `__all__`:

```python
from .results import EPS, PREDICTION_SET_FORMAT_VERSION, Prediction, PredictionSet
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/engine/test_results.py -v` → PASS
Run: `pytest -m "not slow" -q` → all pass.

- [ ] **Step 6: Commit**

```bash
git add s3e/engine tests/engine
git commit -m "feat: lazy Prediction and PredictionSet result objects"
```

---

### Task 7: `QueryEngine` (`s3e/engine/engine.py`)

**Files:**
- Create: `s3e/engine/engine.py`
- Modify: `s3e/engine/__init__.py`
- Test: `tests/engine/test_engine.py`

**Interfaces:**
- Consumes: `resolve_backend`, `VLMBackend` (Task 3); `AnswerSpace`, `BinaryAnswers` (Task 5); `Prediction`, `PredictionSet` (Task 6); `tests.fakes.FakeVLM` (Task 4).
- Produces: `QueryEngine(vlm, *, answers=None, scoring="logprobs", system_prompt=None, prompt_template="{query}", batch_size=8, inference_kwargs=None, vlm_kwargs=None)` with `.backend`, `.answers`, `.scoring`, `.ask(images, queries, *, answers=None, scoring=None, inference_kwargs=None, keep_raw=False) -> PredictionSet`, `.ask_each(scenes, queries, **same) -> list[PredictionSet]`. Also adds `VLMBackend.unsupported_interest_tokens(tokens) -> list[str]` (default `[]`) to `s3e/backends/backend.py`. Task 11's facade builds on exactly this.

- [ ] **Step 1: Write the failing tests**

```python
# tests/engine/test_engine.py
"""Tests for QueryEngine."""

import pytest

from s3e.engine import BinaryAnswers, CategoricalAnswers, PredictionSet, QueryEngine

from conftest import make_blank_image
from fakes import FakeVLM


@pytest.fixture
def images():
    return [make_blank_image()]


class TestAsk:
    def test_returns_prediction_set_keyed_by_query(self, images):
        engine = QueryEngine(FakeVLM({"yes": 0.8, "no": 0.1}))
        results = engine.ask(images, ["Is a on b?", "Is b clear?"])
        assert isinstance(results, PredictionSet)
        assert list(results) == ["Is a on b?", "Is b clear?"]
        assert results["Is a on b?"].answer is True

    def test_interest_tokens_passed_in_logprobs_mode(self, images):
        fake = FakeVLM()
        QueryEngine(fake).ask(images, ["q"])
        call = fake.calls[0]
        assert call["generate"] is False
        assert "yes" in call["interest_tokens"]
        assert "no" in call["interest_tokens"]

    def test_text_match_mode_generates(self, images):
        fake = FakeVLM(text="Yes.")
        results = QueryEngine(fake, scoring="text_match").ask(images, ["q"])
        assert fake.calls[0]["generate"] is True
        assert fake.calls[0]["interest_tokens"] is None
        assert results["q"].masses["yes"] == 1.0

    def test_prompt_template_applied(self, images):
        fake = FakeVLM()
        QueryEngine(fake, prompt_template="Answer yes or no: {query}").ask(
            images, ["Is it red?"]
        )
        assert fake.calls[0]["prompts"] == ["Answer yes or no: Is it red?"]

    def test_prompt_template_must_contain_query(self):
        with pytest.raises(ValueError, match="{query}"):
            QueryEngine(FakeVLM(), prompt_template="no placeholder")

    def test_system_prompt_forwarded(self, images):
        fake = FakeVLM()
        QueryEngine(fake, system_prompt="Be terse.").ask(images, ["q"])
        assert fake.calls[0]["system_prompt"] == "Be terse."

    def test_per_call_answer_space_override(self, images):
        fake = FakeVLM({"red": 0.6, "green": 0.2})
        engine = QueryEngine(fake)
        results = engine.ask(
            images, ["color?"], answers=CategoricalAnswers(["red", "green"])
        )
        assert results["color?"].answer == "red"

    def test_keep_raw(self, images):
        engine = QueryEngine(FakeVLM())
        assert engine.ask(images, ["q"])["q"].raw is None
        assert engine.ask(images, ["q"], keep_raw=True)["q"].raw is not None

    def test_option_with_no_single_token_form_rejected(self, images):
        class PickyFake(FakeVLM):
            def unsupported_interest_tokens(self, tokens):
                return [t for t in tokens if " " in t.strip()]

        space = CategoricalAnswers(["red", "dark blue"])
        with pytest.raises(ValueError, match="dark blue"):
            QueryEngine(PickyFake()).ask(images, ["color?"], answers=space)


class TestInferenceKwargs:
    def test_defaults_merged_with_per_call(self, images):
        fake = FakeVLM()
        engine = QueryEngine(fake, inference_kwargs={"temperature": 0.2, "seed": 1})
        engine.ask(images, ["q"], inference_kwargs={"seed": 7})
        assert fake.calls[0]["inference_kwargs"] == {"temperature": 0.2, "seed": 7}


class TestBatching:
    def test_queries_chunked_by_batch_size(self, images):
        fake = FakeVLM()
        QueryEngine(fake, batch_size=2).ask(images, ["a", "b", "c", "d", "e"])
        assert [call["prompts"] for call in fake.calls] == [
            ["a", "b"], ["c", "d"], ["e"]
        ]


class TestBackendResolution:
    def test_string_resolves_via_resolve_backend(self, monkeypatch, images):
        import s3e.engine.engine as engine_module

        fake = FakeVLM()
        monkeypatch.setattr(
            engine_module, "resolve_backend", lambda vlm, **kw: fake
        )
        engine = QueryEngine("some/model", vlm_kwargs={"device_map": "auto"})
        assert engine.backend is fake

    def test_instance_used_directly(self):
        fake = FakeVLM()
        assert QueryEngine(fake).backend is fake


class TestAskEach:
    def test_one_prediction_set_per_scene(self):
        fake = FakeVLM()
        scenes = [[make_blank_image()], [make_blank_image()]]
        sets = QueryEngine(fake).ask_each(scenes, ["q"])
        assert len(sets) == 2
        assert all(isinstance(s, PredictionSet) for s in sets)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/engine/test_engine.py -v`
Expected: FAIL with `ImportError: cannot import name 'QueryEngine'`

- [ ] **Step 3: Implement `s3e/engine/engine.py`**

```python
# s3e/engine/engine.py
"""QueryEngine: images + queries + an answer space -> predictions."""

from collections.abc import Sequence

from PIL.Image import Image

from ..backends import VLMBackend, resolve_backend
from .answers import AnswerSpace, BinaryAnswers
from .results import Prediction, PredictionSet


class QueryEngine:
    """Answers free-form queries about images against an answer space.

    Args:
        vlm: A :class:`VLMBackend` instance or a model-id string (resolved
            through :func:`resolve_backend`).
        answers: Default answer space (default: ``BinaryAnswers()``).
        scoring: ``"logprobs"`` (token masses) or ``"text_match"``
            (generated text).
        system_prompt: Optional system prompt forwarded to the backend.
        prompt_template: Wrapper for each query; must contain ``{query}``.
        batch_size: Number of queries per backend batch call.
        inference_kwargs: Default per-query kwargs forwarded verbatim to the
            backend (semantics are backend-specific; see the backend class).
        vlm_kwargs: Constructor kwargs, used only when ``vlm`` is a string.
    """

    def __init__(
        self,
        vlm: "VLMBackend | str",
        *,
        answers: "AnswerSpace | None" = None,
        scoring: str = "logprobs",
        system_prompt: "str | None" = None,
        prompt_template: str = "{query}",
        batch_size: int = 8,
        inference_kwargs: "dict | None" = None,
        vlm_kwargs: "dict | None" = None,
    ):
        self.backend = resolve_backend(vlm, **(vlm_kwargs or {}))
        self.answers = answers if answers is not None else BinaryAnswers()
        self.scoring = scoring
        self.system_prompt = system_prompt
        if "{query}" not in prompt_template:
            raise ValueError(
                f"prompt_template must contain '{{query}}'; got {prompt_template!r}"
            )
        self.prompt_template = prompt_template
        self.batch_size = batch_size
        self.inference_kwargs = dict(inference_kwargs or {})

    def ask(
        self,
        images: list[Image],
        queries: Sequence[str],
        *,
        answers: "AnswerSpace | None" = None,
        scoring: "str | None" = None,
        inference_kwargs: "dict | None" = None,
        keep_raw: bool = False,
    ) -> PredictionSet:
        """Answer each query about one scene (a list of images shown together)."""
        space = answers if answers is not None else self.answers
        mode = scoring if scoring is not None else self.scoring
        merged_kwargs = {**self.inference_kwargs, **(inference_kwargs or {})}
        generate = mode == "text_match"
        interest = None if generate else space.interest_tokens
        if interest is not None:
            self._reject_untokenizable_options(space)

        prompts = [self.prompt_template.format(query=q) for q in queries]
        outputs = []
        for start in range(0, len(prompts), self.batch_size):
            outputs.extend(
                self.backend.query_batch(
                    images,
                    prompts[start : start + self.batch_size],
                    system_prompt=self.system_prompt,
                    generate=generate,
                    interest_tokens=interest,
                    **merged_kwargs,
                )
            )

        predictions: dict[str, Prediction] = {}
        for query, output in zip(queries, outputs):
            scored = space.score(output, scoring=mode)
            predictions[query] = Prediction(
                query=query,
                masses=scored.masses,
                null_mass=scored.null_mass,
                unassigned_mass=scored.unassigned_mass,
                answers=space,
                text=output.text,
                argmax_in_interest=output.argmax_in_interest,
                raw=output if keep_raw else None,
            )
        return PredictionSet(predictions)

    def ask_each(
        self,
        scenes: Sequence[list[Image]],
        queries: Sequence[str],
        **ask_kwargs,
    ) -> list[PredictionSet]:
        """Run :meth:`ask` once per scene; combine with PredictionSet.average."""
        return [self.ask(scene, queries, **ask_kwargs) for scene in scenes]

    def _reject_untokenizable_options(self, space: AnswerSpace) -> None:
        """Reject options whose every token form the backend cannot score.

        Logprob scoring reads single-token masses; an option like
        ``"dark blue"`` whose every surface form is multi-token would
        silently score 0.0. Backends that can tell report such forms via
        ``unsupported_interest_tokens``; the default reports none.
        """
        unsupported = set(self.backend.unsupported_interest_tokens(space.interest_tokens))
        if not unsupported:
            return
        options = list(space.options) + (
            [space.null_option] if space.null_option else []
        )
        dead = [o.label for o in options if set(o.tokens) <= unsupported]
        if dead:
            raise ValueError(
                f"Answer options {dead} have no single-token form this backend "
                "can score in logprobs mode; use scoring='text_match' or "
                "provide single-token surface forms"
            )
```

Also add the default hook to `VLMBackend` in `s3e/backends/backend.py` (non-abstract, so existing backends keep working):

```python
    def unsupported_interest_tokens(self, tokens: Sequence[str]) -> list[str]:
        """Subset of ``tokens`` this backend cannot score as a single token.

        Default: assume everything is scorable (unknown strings already get
        0.0 mass under the interest-token contract). Backends with a reverse
        token index (HuggingFace, vLLM) should override this to report token
        strings that no single vocabulary id decodes to.
        """
        return []
```

In `s3e/backends/huggingface.py` and `s3e/backends/vllm.py`, override it using the module's existing reverse-index machinery (`build_token_reverse_index` / the backend's cached index): return the tokens missing from the index. Follow each backend's existing lazy-index initialization pattern; coverage for the overrides belongs to the existing mocked backend tests (add one test per backend module asserting a multi-token string is reported and a known single token is not).

- [ ] **Step 4: Extend `s3e/engine/__init__.py`**

```python
from .engine import QueryEngine
```

and add `"QueryEngine"` to `__all__`.

- [ ] **Step 5: Run tests, then fast suite**

Run: `pytest tests/engine/ -v` → PASS
Run: `pytest -m "not slow" -q` → all pass.

- [ ] **Step 6: Commit**

```bash
git add s3e/engine tests/engine
git commit -m "feat: QueryEngine with per-call answer spaces and batching"
```

---

### Task 8: PDDL layer additions (`s3e/pddl/`)

**Files:**
- Create: `s3e/pddl/fingerprint.py`
- Modify: `s3e/pddl/__init__.py`, `s3e/pddl/up_utils.py`, `s3e/calibration.py` (import fixups only), `tests/test_calibration.py` (import fixups only)
- Test: `tests/pddl/__init__.py` (empty), `tests/pddl/test_grounding.py`

**Interfaces:**
- Consumes: existing `up_utils` functions.
- Produces: `s3e.pddl.parse_domain_problem(domain, problem) -> Problem` (alias of `create_up_problem`, which remains); `s3e.pddl.ground_predicates(up_problem, objects=None) -> list[str]` (alias of `get_all_grounded_predicates_for_objects`, which remains); `s3e.pddl.compute_domain_fingerprint(domain) -> str` (moved from `s3e/calibration.py`). `unified_planning` is now imported ONLY under `s3e/pddl/`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/pddl/test_grounding.py
"""Tests for the public PDDL grounding surface."""

from s3e.pddl import compute_domain_fingerprint, ground_predicates, parse_domain_problem

from conftest import BLOCKSWORLD_DOMAIN, BLOCKSWORLD_PROBLEM  # existing fixtures/constants


class TestParseAndGround:
    def test_parse_domain_problem_returns_problem(self):
        up_problem = parse_domain_problem(BLOCKSWORLD_DOMAIN, BLOCKSWORLD_PROBLEM)
        assert up_problem.fluents

    def test_ground_predicates_enumerates_atoms(self):
        up_problem = parse_domain_problem(BLOCKSWORLD_DOMAIN, BLOCKSWORLD_PROBLEM)
        grounded = ground_predicates(up_problem)
        assert any(p.startswith("on(") for p in grounded)


class TestFingerprintMoved:
    def test_fingerprint_importable_from_pddl(self):
        assert len(compute_domain_fingerprint(BLOCKSWORLD_DOMAIN)) == 64
```

(Use the real constant names from `tests/conftest.py` — check them first; if the domain/problem constants have different names, import those.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/pddl/test_grounding.py -v`
Expected: FAIL with `ImportError: cannot import name 'compute_domain_fingerprint' from 's3e.pddl'`

- [ ] **Step 3: Implement**

1. Move the fingerprint block out of `s3e/calibration.py` into `s3e/pddl/fingerprint.py`: everything from `_COMMUTATIVE_BINARY_OPS` through `compute_domain_fingerprint` (functions `_canonicalize_fnode`, `_canonicalize_effect`, `_canonicalize_action_preconditions`, `_canonicalize_action_effects`, `_build_canonical_domain_string`, `compute_domain_fingerprint`), together with the `unified_planning` imports they need. Give the new module a docstring: `"""Canonical PDDL domain fingerprinting for calibration compatibility."""`
2. In `s3e/calibration.py`, delete the moved code and its UP imports; add `from .pddl.fingerprint import compute_domain_fingerprint` **only if** anything left in the module still references it — otherwise import nothing (Task 9 restructures this file anyway). Fix `tests/test_calibration.py` to import `compute_domain_fingerprint` from `s3e.pddl.fingerprint`.
3. In `s3e/pddl/up_utils.py` add the two aliases with docstrings:

```python
def parse_domain_problem(domain: str, problem: str) -> Problem:
    """Parse a PDDL domain and problem (file paths or strings). Alias of
    :func:`create_up_problem` under the public name."""
    return create_up_problem(domain, problem)


def ground_predicates(
    up_problem: Problem, objects: Optional[dict[str, list[str]]] = None
) -> list[str]:
    """Enumerate all grounded predicate strings. Alias of
    :func:`get_all_grounded_predicates_for_objects` under the public name."""
    return get_all_grounded_predicates_for_objects(up_problem, objects)
```

4. Re-export from `s3e/pddl/__init__.py`: `parse_domain_problem`, `ground_predicates`, `compute_domain_fingerprint`, plus the existing exports (keep whatever it currently exports).

- [ ] **Step 4: Run tests, then fast suite**

Run: `pytest tests/pddl/ tests/test_calibration.py tests/test_pddl_utils.py -v` → PASS
Run: `pytest -m "not slow" -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: move domain fingerprinting into s3e.pddl; public parse/ground names"
```

---

### Task 9: Calibration package (`s3e/calibration/`)

**Files:**
- Delete: `s3e/calibration.py` (contents redistributed)
- Create: `s3e/calibration/__init__.py`, `s3e/calibration/base.py`, `s3e/calibration/data.py`, `s3e/calibration/platt.py`
- Modify: `s3e/semantic_state_estimator.py` (import fixups only — legacy names stay re-exported until Task 11), `tests/test_calibration.py`
- Test: `tests/calibration/__init__.py` (empty), `tests/calibration/test_platt.py`, `tests/calibration/test_data.py`

**Interfaces:**
- Consumes: `Prediction`, `PredictionSet`, `EPS` (Task 6); `s3e._deps.require` (Task 2).
- Produces (used by Tasks 11 and 13):
  - `CalibrationExample(images, state_dict, problem=None)` (ported dataclass)
  - `CalibrationSample(predicate: str, score: float, label: bool, problem: str | None = None)` (ported `PlattCalibrationSample`, renamed; keeps `to_dict`/`from_dict`)
  - `CalibrationSet(samples: list[CalibrationSample], meta: dict)` with `.collect(estimator, examples) -> CalibrationSet` (classmethod; uses `estimator.estimate` and `estimator.calibration_meta()`), `.save(path)`, `CalibrationSet.load(path)`, `.to_dict()`/`.from_dict()` (`"format_version": 1`)
  - `Calibrator` ABC: `apply(results: PredictionSet) -> PredictionSet`, `save(path)`, classmethod `load(path)`
  - `PlattCalibrator.fit(data: CalibrationSet, scope="global", pass_through_single_class=False) -> PlattCalibrator`; `.apply(results)`; `.scope`; group keys: `"__global__"` / predicate name before `"("` (lifted) / full predicate string (grounded)
  - Ported internals in `platt.py`: `PlattParameters`, `fit_platt_parameters`, `apply_platt_scaling`, `grouped_log_odds` (all moved verbatim from old `calibration.py`; `require_sklearn()` replaced by `require("sklearn", "calibration", "Platt scaling fitting")`)

- [ ] **Step 1: Write the failing tests**

```python
# tests/calibration/test_platt.py
"""Tests for the Calibrator interface and PlattCalibrator."""

import json

import pytest

pytest.importorskip("sklearn")

from s3e.calibration import CalibrationSample, CalibrationSet, PlattCalibrator
from s3e.engine import BinaryAnswers, Prediction, PredictionSet


def make_samples(predicate="on(a,b)", n=20):
    """Well-separated scores: positives high, negatives low."""
    samples = []
    for i in range(n):
        label = i % 2 == 0
        score = 2.0 + 0.1 * i if label else -2.0 - 0.1 * i
        samples.append(CalibrationSample(predicate=predicate, score=score, label=label))
    return samples


def make_results(true_mass=0.7, false_mass=0.2, predicate="on(a,b)"):
    space = BinaryAnswers()
    return PredictionSet(
        {
            predicate: Prediction(
                query=predicate,
                masses={"yes": true_mass, "no": false_mass},
                null_mass=0.0,
                unassigned_mass=0.1,
                answers=space,
            )
        }
    )


class TestPlattFitApply:
    def test_fit_global_and_apply_overrides_probability(self):
        data = CalibrationSet(samples=make_samples(), meta={})
        cal = PlattCalibrator.fit(data, scope="global")
        results = make_results()
        calibrated = cal.apply(results)
        raw = results["on(a,b)"].probability
        assert calibrated["on(a,b)"].probability != pytest.approx(raw)
        assert 0.0 <= calibrated["on(a,b)"].probability <= 1.0
        # original untouched
        assert results["on(a,b)"].probability == pytest.approx(raw)

    def test_lifted_scope_groups_by_predicate_name(self):
        data = CalibrationSet(
            samples=make_samples("on(a,b)") + make_samples("on(b,a)"), meta={}
        )
        cal = PlattCalibrator.fit(data, scope="lifted")
        assert set(cal.group_keys()) == {"on"}

    def test_grounded_scope_groups_by_full_predicate(self):
        data = CalibrationSet(
            samples=make_samples("on(a,b)") + make_samples("clear(a)"), meta={}
        )
        cal = PlattCalibrator.fit(data, scope="grounded")
        assert set(cal.group_keys()) == {"on(a,b)", "clear(a)"}

    def test_apply_without_matching_group_keeps_raw(self):
        data = CalibrationSet(samples=make_samples("on(a,b)"), meta={})
        cal = PlattCalibrator.fit(data, scope="grounded")
        results = make_results(predicate="clear(c)")
        calibrated = cal.apply(results)
        assert calibrated["clear(c)"].probability == pytest.approx(
            results["clear(c)"].probability
        )

    def test_single_class_group_rejected_by_default(self):
        one_sided = [
            CalibrationSample(predicate="on(a,b)", score=1.0 + i, label=True)
            for i in range(5)
        ]
        with pytest.raises(ValueError, match="positive and negative"):
            PlattCalibrator.fit(
                CalibrationSet(samples=one_sided, meta={}), scope="global"
            )

    def test_invalid_scope_rejected(self):
        with pytest.raises(ValueError, match="scope"):
            PlattCalibrator.fit(
                CalibrationSet(samples=make_samples(), meta={}), scope="bogus"
            )


class TestPlattPersistence:
    def test_save_load_round_trip(self, tmp_path):
        cal = PlattCalibrator.fit(
            CalibrationSet(samples=make_samples(), meta={}), scope="global"
        )
        path = tmp_path / "platt.json"
        cal.save(path)
        restored = PlattCalibrator.load(path)
        results = make_results()
        assert restored.apply(results)["on(a,b)"].probability == pytest.approx(
            cal.apply(results)["on(a,b)"].probability
        )

    def test_saved_file_has_format_version(self, tmp_path):
        cal = PlattCalibrator.fit(
            CalibrationSet(samples=make_samples(), meta={}), scope="global"
        )
        path = tmp_path / "platt.json"
        cal.save(path)
        assert "format_version" in json.loads(path.read_text())
```

```python
# tests/calibration/test_data.py
"""Tests for CalibrationSet collection and persistence."""

import pytest

from s3e.calibration import CalibrationExample, CalibrationSample, CalibrationSet


class TestCalibrationSetPersistence:
    def test_round_trip(self, tmp_path):
        data = CalibrationSet(
            samples=[CalibrationSample("on(a,b)", 1.5, True, problem=None)],
            meta={"true_label": "yes", "false_label": "no"},
        )
        path = tmp_path / "calib.json"
        data.save(path)
        restored = CalibrationSet.load(path)
        assert restored.samples[0].predicate == "on(a,b)"
        assert restored.samples[0].score == pytest.approx(1.5)
        assert restored.meta["true_label"] == "yes"

    def test_bad_format_version_rejected(self, tmp_path):
        path = tmp_path / "calib.json"
        path.write_text('{"format_version": 99, "samples": [], "meta": {}}')
        with pytest.raises(ValueError, match="format_version"):
            CalibrationSet.load(path)
```

(`CalibrationSet.collect` needs the new estimator and is covered in Task 11's estimator tests and Task 13's consumer tests; here it exists but delegates to `estimator.estimate`.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/calibration/ -v`
Expected: FAIL with `ImportError` (no `CalibrationSample`/`CalibrationSet`/`PlattCalibrator` in `s3e.calibration`).

- [ ] **Step 3: Restructure into a package**

```bash
git mv s3e/calibration.py s3e/calibration_legacy_tmp.py
mkdir s3e/calibration
```

Distribute `calibration_legacy_tmp.py` (already stripped of fingerprinting in Task 8):

**`s3e/calibration/platt.py`** — move verbatim: `SCORE_EPS` (delete; import `EPS` from `..engine.results` instead and rename uses), `GLOBAL_CALIBRATION_KEY`, `PlattParameters`, `PlattScalingProfile` (keep as internal profile storage; keep its `to_dict`/`from_dict` but add `"format_version": 1` alongside the existing `schema_version` key), `grouped_log_odds`, `apply_platt_scaling`, `fit_platt_parameters`. Replace the top-level sklearn try/except and `require_sklearn()` with a lazy import inside `fit_platt_parameters`:

```python
def fit_platt_parameters(scores: list[float], labels: list[bool]) -> PlattParameters:
    from .._deps import require

    require("sklearn", "calibration", "Platt scaling fitting")
    from sklearn.linear_model import LogisticRegression
    ...
```

Then add the public calibrator in the same file:

```python
VALID_SCOPES = ("global", "lifted", "grounded")


def _group_key(predicate: str, scope: str) -> str:
    if scope == "global":
        return GLOBAL_CALIBRATION_KEY
    if scope == "lifted":
        return predicate.split("(", 1)[0]
    return predicate


class PlattCalibrator(Calibrator):
    """Per-group Platt scaling fitted on grouped log-odds scores."""

    def __init__(self, scope: str, groups: dict[str, PlattParameters], meta: dict):
        self.scope = scope
        self.groups = groups
        self.meta = meta

    @classmethod
    def fit(
        cls,
        data: "CalibrationSet",
        scope: str = "global",
        pass_through_single_class: bool = False,
    ) -> "PlattCalibrator":
        if scope not in VALID_SCOPES:
            raise ValueError(f"Unknown scope {scope!r}; expected one of {VALID_SCOPES}")
        grouped: dict[str, list] = {}
        for sample in data.samples:
            grouped.setdefault(_group_key(sample.predicate, scope), []).append(sample)
        groups: dict[str, PlattParameters] = {}
        for key, samples in grouped.items():
            labels = [s.label for s in samples]
            if pass_through_single_class and len(set(labels)) < 2:
                continue
            groups[key] = fit_platt_parameters([s.score for s in samples], labels)
        return cls(scope=scope, groups=groups, meta=dict(data.meta))

    def group_keys(self) -> list[str]:
        return list(self.groups)

    def apply(self, results: "PredictionSet") -> "PredictionSet":
        calibrated = {}
        for key, prediction in results.items():
            params = self.groups.get(_group_key(key, self.scope))
            if params is None:
                calibrated[key] = prediction
            else:
                calibrated[key] = prediction.with_probability(
                    apply_platt_scaling(prediction.score, params)
                )
        return type(results)(calibrated)

    PLATT_FORMAT_VERSION = 1

    def save(self, path: "str | Path") -> None:
        payload = {
            "format_version": self.PLATT_FORMAT_VERSION,
            "kind": "platt",
            "scope": self.scope,
            "meta": self.meta,
            "groups": {
                key: {
                    "a": params.a,
                    "b": params.b,
                    "sample_count": params.sample_count,
                    "positive_count": params.positive_count,
                    "negative_count": params.negative_count,
                }
                for key, params in self.groups.items()
            },
        }
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: "str | Path") -> "PlattCalibrator":
        data = json.loads(Path(path).read_text())
        version = data.get("format_version")
        if version != cls.PLATT_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported PlattCalibrator format_version: {version!r} "
                f"(expected {cls.PLATT_FORMAT_VERSION})"
            )
        return cls(
            scope=data["scope"],
            groups={
                key: PlattParameters(
                    a=float(g["a"]),
                    b=float(g["b"]),
                    sample_count=int(g["sample_count"]),
                    positive_count=int(g["positive_count"]),
                    negative_count=int(g["negative_count"]),
                )
                for key, g in data["groups"].items()
            },
            meta=data.get("meta", {}),
        )
```

(`import json` and `from pathlib import Path` at the top of `platt.py`. `PlattScalingProfile` is kept ported in this task only because the legacy estimator still imports it; Task 11 deletes it together with the legacy re-export block — its validation fields live on in `meta`.)

**`s3e/calibration/base.py`**:

```python
# s3e/calibration/base.py
"""Calibrator interface: fit offline, apply to prediction sets."""

from abc import ABC, abstractmethod
from pathlib import Path

from ..engine import PredictionSet


class Calibrator(ABC):
    """Transforms a PredictionSet's probabilities; never mutates inputs."""

    @abstractmethod
    def apply(self, results: PredictionSet) -> PredictionSet:
        """Return a new PredictionSet with calibrated probabilities."""

    @abstractmethod
    def save(self, path: "str | Path") -> None:
        """Persist this calibrator to a JSON file."""

    @classmethod
    @abstractmethod
    def load(cls, path: "str | Path") -> "Calibrator":
        """Restore a calibrator persisted by :meth:`save`."""
```

**`s3e/calibration/data.py`** — `CalibrationExample` moved verbatim; `CalibrationSample` is the old `PlattCalibrationSample` renamed (keep `to_dict`/`from_dict`); plus:

```python
CALIBRATION_SET_FORMAT_VERSION = 1


@dataclass
class CalibrationSet:
    """Scores + labels collected once, reusable for any calibrator fit."""

    samples: list[CalibrationSample]
    meta: dict

    @classmethod
    def collect(cls, estimator, examples: list[CalibrationExample]) -> "CalibrationSet":
        """Query the estimator's VLM on labeled examples (the expensive step)."""
        samples: list[CalibrationSample] = []
        for example in examples:
            if example.problem is not None:
                estimator.set_problem(*example.problem_pair())  # see note below
            results = estimator.estimate(
                example.images, predicates=list(example.state_dict)
            )
            for predicate, label in example.state_dict.items():
                samples.append(
                    CalibrationSample(
                        predicate=predicate,
                        score=results[predicate].score,
                        label=bool(label),
                        problem=example.problem,
                    )
                )
        return cls(samples=samples, meta=estimator.calibration_meta())

    def to_dict(self) -> dict:
        return {
            "format_version": CALIBRATION_SET_FORMAT_VERSION,
            "meta": self.meta,
            "samples": [s.to_dict() for s in self.samples],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CalibrationSet":
        version = data.get("format_version")
        if version != CALIBRATION_SET_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported CalibrationSet format_version: {version!r} "
                f"(expected {CALIBRATION_SET_FORMAT_VERSION})"
            )
        return cls(
            samples=[CalibrationSample.from_dict(s) for s in data["samples"]],
            meta=dict(data.get("meta", {})),
        )

    def save(self, path: "str | Path") -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: "str | Path") -> "CalibrationSet":
        return cls.from_dict(json.loads(Path(path).read_text()))
```

Note on `collect`: the old `CalibrationExample.problem` is a single problem-PDDL string; `set_problem` needs (domain, problem) — in the loop, replace the sketch's `estimator.set_problem(*example.problem_pair())` with `estimator.set_problem(estimator.domain_pddl, example.problem)` (the estimator keeps its domain; there is no `problem_pair()` helper).

**`s3e/calibration/__init__.py`**:

```python
# s3e/calibration/__init__.py
"""Calibration: collect scored examples once, fit and apply calibrators offline."""

from .base import Calibrator
from .data import CalibrationExample, CalibrationSample, CalibrationSet
from .platt import PlattCalibrator

# Legacy names still consumed by s3e/semantic_state_estimator.py until Task 11:
from .platt import (
    GLOBAL_CALIBRATION_KEY,
    PlattParameters,
    PlattScalingProfile,
    apply_platt_scaling,
    fit_platt_parameters,
    grouped_log_odds,
)
from .data import CalibrationSample as PlattCalibrationSample

__all__ = [
    "Calibrator",
    "CalibrationExample",
    "CalibrationSample",
    "CalibrationSet",
    "PlattCalibrator",
]
```

Then `git rm s3e/calibration_legacy_tmp.py` after everything is moved. Fix imports in `s3e/semantic_state_estimator.py` (it imported `SCORE_EPS`/`require_sklearn` etc. — point the survivors at the new locations; `SCORE_EPS` → `EPS` from `s3e.engine`; `require_sklearn()` call sites → `require("sklearn", "calibration", "Platt scaling fitting")`). Fix `tests/test_calibration.py` imports the same way; delete its `TestDomainFingerprint` only if Task 8 already relocated those tests, otherwise keep them pointing at `s3e.pddl.fingerprint`.

- [ ] **Step 4: Run tests, then fast suite**

Run: `pytest tests/calibration/ -v` → PASS
Run: `pytest -m "not slow" -q` → all pass (legacy estimator still works through the re-exports).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: calibration package with Calibrator ABC, PlattCalibrator, CalibrationSet"
```

---

### Task 10: Translation updates

**Files:**
- Move: `s3e/cache.py` → `s3e/translation/cache.py` (`git mv`)
- Modify: `s3e/translation/translator.py`, `s3e/translation/template.py`, `s3e/translation/llm.py`, `s3e/translation/__init__.py`, `tests/test_cache.py`, `tests/test_translators.py`
- Test: additions to `tests/test_translators.py`

**Interfaces:**
- Consumes: `s3e._deps.require` (Task 2).
- Produces: `QueryTranslator.translate(predicates: list[str], domain: str | None = None, problem: str | None = None) -> dict[str, str]` — domain/problem now optional. `TemplateTranslator` works without PDDL (positional/custom-name mapping only). `LLMTranslator` raises `ValueError` when domain/problem are `None`. Task 11's facade calls translators exactly through this signature.

- [ ] **Step 1: Write the failing tests (append to `tests/test_translators.py`)**

```python
class TestPddlFreeTranslation:
    def test_identity_without_domain(self):
        from s3e.translation import IdentityTranslator

        result = IdentityTranslator().translate(["on(a,b)"])
        assert result == {"on(a,b)": "on(a,b)"}

    def test_template_positional_without_domain(self):
        from s3e.translation import TemplateTranslator

        translator = TemplateTranslator({"on": "Is {0} on {1}?"})
        result = translator.translate(["on(a,b)"])
        assert result == {"on(a,b)": "Is a on b?"}

    def test_template_custom_names_without_domain(self):
        from s3e.translation import TemplateTranslator

        translator = TemplateTranslator({"on": "Is {top} on {bottom}?"})
        result = translator.translate(["on(a,b)"])
        assert result == {"on(a,b)": "Is a on b?"}

    def test_llm_translator_requires_domain(self):
        from s3e.translation import LLMTranslator

        translator = LLMTranslator.__new__(LLMTranslator)  # skip client setup
        import pytest

        with pytest.raises(ValueError, match="domain"):
            translator.translate(["on(a,b)"])
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_translators.py::TestPddlFreeTranslation -v`
Expected: FAIL with `TypeError: translate() missing 2 required positional arguments`

- [ ] **Step 3: Implement**

1. `s3e/translation/translator.py`: change the abstract signature to

```python
    @abstractmethod
    def translate(
        self,
        predicates: list[str],
        domain: "str | None" = None,
        problem: "str | None" = None,
    ) -> dict[str, str]:
        """Translate grounded predicates to query strings.

        ``domain``/``problem`` provide optional PDDL context; translators
        that can work without it must accept ``None``.
        """
```

2. `s3e/translation/template.py`: in `TemplateTranslator.translate`, make the signature-names map empty when domain is None:

```python
    def translate(self, predicates, domain=None, problem=None):
        result: dict[str, str] = {}
        predicate_arg_names = (
            _predicate_argument_names(domain, problem)
            if domain is not None and problem is not None
            else {}
        )
        ...  # rest unchanged
```

3. `s3e/translation/identity.py`: update the signature to match (its body ignores domain/problem already — verify and adjust defaults).
4. `s3e/translation/llm.py`: update the signature; at the top of `translate` add

```python
        if domain is None or problem is None:
            raise ValueError(
                "LLMTranslator needs the PDDL domain and problem for context; "
                "got None. Use TemplateTranslator or PrewrittenTranslator for "
                "PDDL-free estimation."
            )
```

Also: if `llm.py` imports `openai` at module top (guarded or not), move that import into the constructor preceded by `require("openai", "openai", "LLMTranslator")`, and update `s3e/cache.py` imports after the move (`from .cache import ...` inside the translation package).
5. `git mv s3e/cache.py s3e/translation/cache.py`; fix `tests/test_cache.py` imports to `from s3e.translation.cache import ...`.

- [ ] **Step 4: Run tests, then fast suite**

Run: `pytest tests/test_translators.py tests/test_cache.py -v` → PASS
Run: `pytest -m "not slow" -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: PDDL-optional translator interface; cache into translation package"
```

---

### Task 11: The `SemanticStateEstimator` facade

This is the pivotal task: the old god-object dies here.

**Files:**
- Create: `s3e/estimator.py`
- Delete: `s3e/semantic_state_estimator.py`, `s3e/state_estimator.py`, `tests/test_semantic_state_estimator.py` (superseded — see step 6)
- Modify: `s3e/__init__.py`, `s3e/constants.py`, `s3e/calibration/__init__.py` (drop the legacy re-export block), `s3e/calibration/data.py` (finalize `collect`)
- Test: `tests/test_estimator.py`

**Interfaces:**
- Consumes: `QueryEngine`, `BinaryAnswers`, `PredictionSet` (Tasks 5–7); translators (Task 10); `s3e.pddl` (Task 8); `Calibrator` (Task 9).
- Produces (final public estimator API; Tasks 12–14 and both consumers rely on it):
  - `SemanticStateEstimator(predicates, *, vlm, translator=None, answers=None, system_prompt=None, prompt_template=None, additional_instructions=None, confidence=0.5, scoring="logprobs", batch_size=8, vlm_kwargs=None, inference_kwargs=None, null_tokens=None, true_tokens=None, false_tokens=None)`
  - `SemanticStateEstimator.from_pddl(domain, problem, **same_keywords)`
  - `.predicates: list[str]`, `.queries: dict[str, str]`, `.engine: QueryEngine`, `.translator`, `.up_problem` (from_pddl only, else `None`), `.domain_pddl`/`.problem_pddl` (from_pddl only, else `None`), `.domain_fingerprint: str | None`, `.confidence`
  - `.estimate(images, *, predicates=None, calibrator=None, keep_raw=False, inference_kwargs=None) -> PredictionSet`
  - `.estimate_averaged(scenes, **same) -> PredictionSet`
  - `.__call__(images, confidence=None) -> dict[str, bool | None]`
  - `.set_problem(domain, problem) -> None` (raises `ValueError` on non-PDDL estimators)
  - `.set_predicates(predicates) -> None`
  - `.to_up_state(state) -> UPState` (raises `ValueError` on non-PDDL estimators)
  - `.calibration_meta() -> dict` (answer labels/tokens, scoring, domain_fingerprint)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_estimator.py
"""Tests for the SemanticStateEstimator facade."""

import pytest

from s3e import BinaryAnswers, PredictionSet, SemanticStateEstimator, TemplateTranslator

from conftest import BLOCKSWORLD_DOMAIN, BLOCKSWORLD_PROBLEM, make_blank_image
from fakes import FakeVLM


TEMPLATES = {"on": "Is {0} on {1}?", "clear": "Is {0} clear?"}


@pytest.fixture
def images():
    return [make_blank_image()]


def make_estimator(fake=None, **kwargs):
    return SemanticStateEstimator.from_pddl(
        BLOCKSWORLD_DOMAIN,
        BLOCKSWORLD_PROBLEM,
        vlm=fake or FakeVLM(),
        translator=TemplateTranslator(TEMPLATES),
        **kwargs,
    )


class TestConstructionFromPredicates:
    def test_predicates_without_pddl(self, images):
        estimator = SemanticStateEstimator(
            predicates=["on(a,b)", "clear(a)"],
            vlm=FakeVLM(),
            translator=TemplateTranslator(TEMPLATES),
        )
        results = estimator.estimate(images)
        assert set(results) == {"on(a,b)", "clear(a)"}

    def test_pddl_extras_raise_without_pddl(self, images):
        estimator = SemanticStateEstimator(
            predicates=["on(a,b)"], vlm=FakeVLM(),
            translator=TemplateTranslator(TEMPLATES),
        )
        with pytest.raises(ValueError, match="from_pddl"):
            estimator.set_problem(BLOCKSWORLD_DOMAIN, BLOCKSWORLD_PROBLEM)
        with pytest.raises(ValueError, match="from_pddl"):
            estimator.to_up_state({"on(a,b)": True})


class TestConstructionFromPddl:
    def test_grounds_all_predicates(self):
        estimator = make_estimator()
        assert any(p.startswith("on(") for p in estimator.predicates)
        assert any(p.startswith("clear(") for p in estimator.predicates)

    def test_queries_are_translated(self):
        estimator = make_estimator()
        grounded_on = next(p for p in estimator.predicates if p.startswith("on("))
        assert estimator.queries[grounded_on].startswith("Is ")

    def test_fingerprint_available(self):
        assert len(make_estimator().domain_fingerprint) == 64


class TestEstimate:
    def test_returns_prediction_set_keyed_by_predicate(self, images):
        results = make_estimator().estimate(images)
        assert isinstance(results, PredictionSet)
        assert set(results) == set(make_estimator().predicates)

    def test_predicate_subset_queries_only_subset(self, images):
        fake = FakeVLM()
        estimator = make_estimator(fake)
        subset = estimator.predicates[:2]
        results = estimator.estimate(images, predicates=subset)
        assert set(results) == set(subset)
        prompts = [p for call in fake.calls for p in call["prompts"]]
        assert len(prompts) == 2

    def test_unknown_predicate_rejected(self, images):
        with pytest.raises(ValueError, match="nope"):
            make_estimator().estimate(images, predicates=["nope(x)"])

    def test_call_thresholds(self, images):
        fake = FakeVLM({"yes": 0.9, "no": 0.05})
        state = make_estimator(fake)(images)
        assert all(value is True for value in state.values())

    def test_calibrator_applied_when_passed(self, images):
        class HalfCalibrator:
            def apply(self, results):
                return PredictionSet(
                    {k: p.with_probability(0.5) for k, p in results.items()}
                )

        results = make_estimator().estimate(images, calibrator=HalfCalibrator())
        assert all(p.probability == 0.5 for p in results.values())


class TestSetProblem:
    def test_regrounds_without_touching_backend(self, images):
        fake = FakeVLM()
        estimator = make_estimator(fake)
        backend_before = estimator.engine.backend
        estimator.set_problem(BLOCKSWORLD_DOMAIN, BLOCKSWORLD_PROBLEM)
        assert estimator.engine.backend is backend_before
        assert estimator.predicates  # re-grounded


class TestSharedBackend:
    def test_two_estimators_share_one_backend(self):
        fake = FakeVLM()
        a = make_estimator(fake)
        b = make_estimator(fake)
        assert a.engine.backend is b.engine.backend


class TestNullTokens:
    def test_null_dominated_predicate_is_none_in_state(self, images):
        fake = FakeVLM({"yes": 0.1, "no": 0.05, "unknown": 0.7})
        estimator = make_estimator(fake, null_tokens=["unknown"])
        state = estimator(images)
        assert all(value is None for value in state.values())


class TestCalibrationMeta:
    def test_meta_includes_labels_and_fingerprint(self):
        meta = make_estimator().calibration_meta()
        assert meta["true_label"] == "yes"
        assert meta["scoring"] == "logprobs"
        assert len(meta["domain_fingerprint"]) == 64


class TestCollectIntegration:
    def test_collect_produces_scored_samples(self, images):
        pytest.importorskip("sklearn")
        from s3e.calibration import CalibrationExample, CalibrationSet, PlattCalibrator

        estimator = make_estimator(FakeVLM({"yes": 0.8, "no": 0.1}))
        target = {p: (i % 2 == 0) for i, p in enumerate(estimator.predicates)}
        data = CalibrationSet.collect(
            estimator, [CalibrationExample(images=images, state_dict=target)]
        )
        assert len(data.samples) == len(target)
        assert data.meta["true_label"] == "yes"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_estimator.py -v`
Expected: FAIL with `ImportError` (no `SemanticStateEstimator` matching the new API in `s3e`).

- [ ] **Step 3: Implement `s3e/estimator.py`**

```python
# s3e/estimator.py
"""SemanticStateEstimator: a thin facade wiring predicates, translation,
and a QueryEngine into symbolic state estimation.

The estimator's contract is "predicates in, state out". PDDL is one way to
produce the predicates (:meth:`SemanticStateEstimator.from_pddl`); an
explicit list is another. Unified Planning is imported only on the PDDL
paths, keeping the core importable without it.
"""

from collections.abc import Sequence

from PIL.Image import Image

from .constants import (
    SYSTEM_PROMPT_ADDITIONAL_INSTRUCTIONS,
    SYSTEM_PROMPT_NO_TRANSLATION,
    SYSTEM_PROMPT_WITH_TRANSLATION,
)
from .engine import BinaryAnswers, PredictionSet, QueryEngine
from .translation import IdentityTranslator, QueryTranslator


class SemanticStateEstimator:
    """Estimates truth values for a set of grounded predicates from images.

    Args:
        predicates: The grounded predicate strings to estimate.
        vlm: A backend instance or model-id string (see ``resolve_backend``).
        translator: Predicate-to-query strategy (default: identity).
        answers: Answer space (default: ``BinaryAnswers()``; identity
            translation defaults to ``BinaryAnswers("true", "false")``).
        system_prompt: Overrides the auto-selected system prompt.
        prompt_template: Wrapper for each query; must contain ``{query}``.
        additional_instructions: Appended to the system prompt.
        confidence: Default threshold for :meth:`__call__`.
        scoring: ``"logprobs"`` or ``"text_match"``.
        batch_size / vlm_kwargs / inference_kwargs: Forwarded to
            :class:`QueryEngine`.
        true_tokens / false_tokens / null_tokens: Convenience overrides
            building the default binary answer space; ignored when
            ``answers`` is passed explicitly.
    """

    def __init__(
        self,
        predicates: Sequence[str],
        *,
        vlm,
        translator: "QueryTranslator | None" = None,
        answers=None,
        system_prompt: "str | None" = None,
        prompt_template: "str | None" = None,
        additional_instructions: "str | None" = None,
        confidence: float = 0.5,
        scoring: str = "logprobs",
        batch_size: int = 8,
        vlm_kwargs: "dict | None" = None,
        inference_kwargs: "dict | None" = None,
        true_tokens: "list[str] | None" = None,
        false_tokens: "list[str] | None" = None,
        null_tokens: "list[str] | None" = None,
    ):
        self.translator = translator or IdentityTranslator()
        identity = isinstance(self.translator, IdentityTranslator)

        if answers is None:
            if identity:
                answers = BinaryAnswers(
                    "true", "false",
                    true_tokens=true_tokens, false_tokens=false_tokens,
                    null_tokens=null_tokens,
                )
            else:
                answers = BinaryAnswers(
                    true_tokens=true_tokens, false_tokens=false_tokens,
                    null_tokens=null_tokens,
                )

        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT_WITH_TRANSLATION
        if additional_instructions:
            system_prompt += SYSTEM_PROMPT_ADDITIONAL_INSTRUCTIONS.format(
                additional_instructions=additional_instructions
            )

        self.confidence = confidence
        self.engine = QueryEngine(
            vlm,
            answers=answers,
            scoring=scoring,
            system_prompt=system_prompt,
            prompt_template=prompt_template or "{query}",
            batch_size=batch_size,
            inference_kwargs=inference_kwargs,
            vlm_kwargs=vlm_kwargs,
        )

        self.up_problem = None
        self.domain_pddl: "str | None" = None
        self.problem_pddl: "str | None" = None
        self.domain_fingerprint: "str | None" = None
        self.predicates: list[str] = []
        self.queries: dict[str, str] = {}
        self.set_predicates(predicates)

    @classmethod
    def from_pddl(cls, domain: str, problem: str, **kwargs) -> "SemanticStateEstimator":
        """Build an estimator by grounding a PDDL domain and problem.

        ``domain``/``problem`` are PDDL strings or ``.pddl`` file paths.
        Identity translation additionally gets a domain-aware system prompt.
        """
        from .pddl import (
            compute_domain_fingerprint,
            get_object_names_dict,
            get_pddl_strings,
            ground_predicates,
            parse_domain_problem,
        )

        up_problem = parse_domain_problem(domain, problem)
        translator = kwargs.get("translator")
        if kwargs.get("system_prompt") is None and (
            translator is None or isinstance(translator, IdentityTranslator)
        ):
            objects = get_object_names_dict(up_problem)
            objects_str = "\n".join(
                f"{key} type: {value}" for key, value in objects.items()
            )
            domain_str, _ = get_pddl_strings(up_problem)
            kwargs["system_prompt"] = SYSTEM_PROMPT_NO_TRANSLATION.format(
                domain=domain_str, objects=objects_str
            )

        estimator = cls(ground_predicates(up_problem), vlm=kwargs.pop("vlm"), **kwargs)
        estimator.up_problem = up_problem
        estimator.domain_pddl = domain
        estimator.problem_pddl = problem
        estimator.domain_fingerprint = compute_domain_fingerprint(up_problem)
        estimator._retranslate()
        return estimator

    # --- predicate/problem management ---

    def set_predicates(self, predicates: Sequence[str]) -> None:
        """Replace the predicate list and re-run translation."""
        self.predicates = list(predicates)
        self._retranslate()

    def set_problem(self, domain: str, problem: str) -> None:
        """Re-ground a new PDDL problem; the engine/backend is untouched."""
        self._require_pddl("set_problem")
        from .pddl import (
            compute_domain_fingerprint,
            ground_predicates,
            parse_domain_problem,
        )

        self.up_problem = parse_domain_problem(domain, problem)
        self.domain_pddl = domain
        self.problem_pddl = problem
        self.domain_fingerprint = compute_domain_fingerprint(self.up_problem)
        self.predicates = ground_predicates(self.up_problem)
        self._retranslate()

    def _retranslate(self) -> None:
        self.queries = self.translator.translate(
            self.predicates, self.domain_pddl, self.problem_pddl
        )

    def _require_pddl(self, method: str) -> None:
        if self.up_problem is None:
            raise ValueError(
                f"{method} is only available on estimators built with from_pddl"
            )

    # --- estimation ---

    def estimate(
        self,
        images: list[Image],
        *,
        predicates: "Sequence[str] | None" = None,
        calibrator=None,
        keep_raw: bool = False,
        inference_kwargs: "dict | None" = None,
    ) -> PredictionSet:
        """Estimate the selected predicates; returns a lazy PredictionSet."""
        selected = self._select(predicates)
        queries = [self.queries[p] for p in selected]
        answered = self.engine.ask(
            images, queries, keep_raw=keep_raw, inference_kwargs=inference_kwargs
        )
        results = PredictionSet(
            {p: answered[q] for p, q in zip(selected, queries)}
        )
        if calibrator is not None:
            results = calibrator.apply(results)
        return results

    def estimate_averaged(self, scenes, **estimate_kwargs) -> PredictionSet:
        """Estimate each scene separately and average the stored masses."""
        return PredictionSet.average(
            [self.estimate(scene, **estimate_kwargs) for scene in scenes]
        )

    def __call__(
        self, images: list[Image], confidence: "float | None" = None
    ) -> "dict[str, bool | None]":
        """Estimate and threshold into a boolean state."""
        threshold = confidence if confidence is not None else self.confidence
        return self.estimate(images).to_state(confidence=threshold)

    def _select(self, predicates: "Sequence[str] | None") -> list[str]:
        if predicates is None:
            return list(self.predicates)
        known = set(self.predicates)
        unknown = [p for p in predicates if p not in known]
        if unknown:
            raise ValueError(f"Unknown predicates requested: {unknown}")
        return list(predicates)

    # --- interop ---

    def to_up_state(self, state: dict[str, bool]):
        """Convert a boolean state dict into a Unified Planning UPState."""
        self._require_pddl("to_up_state")
        from .pddl import state_dict_to_up_state

        return state_dict_to_up_state(self.up_problem, state)

    def calibration_meta(self) -> dict:
        """Metadata stored alongside collected calibration data."""
        answers = self.engine.answers
        return {
            "true_label": getattr(answers, "true_label", None),
            "false_label": getattr(answers, "false_label", None),
            "scoring": self.engine.scoring,
            "domain_fingerprint": self.domain_fingerprint,
            "answers": answers.to_dict(),
        }
```

Notes for the implementer:
- `Prediction.query` will hold the translated query while `PredictionSet` is re-keyed by predicate — that is intentional; the prediction remembers what was actually asked.
- If a query string repeats across predicates, `engine.ask` deduplicates keys (dict). Guard: in `estimate`, if `len(set(queries)) != len(queries)`, fall back to per-predicate lookup by position — build `PredictionSet` from `zip(selected, queries)` reading `answered[q]`; identical queries share one prediction object, which is correct and efficient.
- Ensure `s3e/pddl/__init__.py` re-exports `get_object_names_dict`, `get_pddl_strings`, `state_dict_to_up_state` (they exist in `up_utils.py`).

- [ ] **Step 4: Rewire the public API**

Replace `s3e/__init__.py` with:

```python
# s3e/__init__.py
"""s3e — Semantic Symbolic State Estimation with vision-language models.

Quick start::

    from s3e import SemanticStateEstimator, TemplateTranslator

    estimator = SemanticStateEstimator.from_pddl(
        domain_pddl, problem_pddl,
        vlm="HuggingFaceTB/SmolVLM-256M-Instruct",
        translator=TemplateTranslator({"on": "Is {0} on {1}?"}),
    )
    state = estimator(images)  # dict[str, bool | None]
"""

from importlib.metadata import PackageNotFoundError, version

from .backends import VLMBackend, VLMOutput, resolve_backend
from .calibration import (
    CalibrationExample,
    CalibrationSample,
    CalibrationSet,
    Calibrator,
    PlattCalibrator,
)
from .engine import (
    AnswerOption,
    AnswerSpace,
    BinaryAnswers,
    CategoricalAnswers,
    Prediction,
    PredictionSet,
    QueryEngine,
)
from .estimator import SemanticStateEstimator
from .translation import (
    IdentityTranslator,
    LLMTranslator,
    PrewrittenTranslator,
    QueryTranslator,
    TemplateTranslator,
)

try:
    __version__ = version("s3e")
except PackageNotFoundError:  # running from a source tree
    __version__ = "0.0.0.dev0"

__all__ = [
    "SemanticStateEstimator",
    "QueryEngine",
    "AnswerOption",
    "AnswerSpace",
    "BinaryAnswers",
    "CategoricalAnswers",
    "Prediction",
    "PredictionSet",
    "VLMBackend",
    "VLMOutput",
    "resolve_backend",
    "HuggingFaceVLM",
    "OpenAIVLM",
    "VLLMBackend",
    "QueryTranslator",
    "IdentityTranslator",
    "TemplateTranslator",
    "PrewrittenTranslator",
    "LLMTranslator",
    "Calibrator",
    "PlattCalibrator",
    "CalibrationSet",
    "CalibrationSample",
    "CalibrationExample",
]

_LAZY_TOP_LEVEL = {"HuggingFaceVLM", "OpenAIVLM", "VLLMBackend"}


def __getattr__(name: str):
    """Lazily expose optional backends without importing their packages."""
    if name in _LAZY_TOP_LEVEL:
        import s3e.backends as _backends

        return getattr(_backends, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

Caution: `s3e.translation` must not import `openai` eagerly (Task 10 moved that into the `LLMTranslator` constructor) and `s3e.calibration` must not import `sklearn` eagerly (Task 9 moved it into `fit_platt_parameters`) — verify both, or `import s3e` breaks on bare installs.

Then delete the superseded modules and finalize:

```bash
git rm s3e/semantic_state_estimator.py s3e/state_estimator.py
```

Remove the legacy re-export block from `s3e/calibration/__init__.py` (the `PlattScalingProfile`/`PlattCalibrationSample` compatibility names — nothing imports them now; keep `grouped_log_odds`, `apply_platt_scaling`, `fit_platt_parameters`, `PlattParameters` exported from `s3e.calibration.platt` for tests). In `s3e/constants.py`, delete `TRUE_TOKENS_*`/`FALSE_TOKENS_*` (answer spaces own token defaults now) and `OPENAI_MODEL_IDENTIFIER` (lives in `backends/resolve.py`); keep the three prompt templates. Finalize `CalibrationSet.collect` in `s3e/calibration/data.py` against the real estimator API (per Task 9's note: `estimator.set_problem(estimator.domain_pddl, example.problem)` when `example.problem` is set).

- [ ] **Step 5: Port surviving estimator-test behaviors, delete the monolith**

`tests/test_semantic_state_estimator.py` (2,727 lines) is superseded. Its behaviors now live in: engine tests (Tasks 5–7: token groups, text match, interest tokens, null tokens, batching, probability clipping), calibration tests (Task 9: Platt fit/apply/persistence/errors), and `tests/test_estimator.py` (this task: construction, call, subsetting, set_problem, use-vllm routing is gone by design). Before deleting, skim each remaining test class against this map and port any *behavior* not yet covered into the appropriate new test module — expected gaps to port here into `tests/test_estimator.py`: prompt-template forwarding (old `TestUserPromptTemplate`), averaged multi-scene estimation (old `TestNullTokenAverageStrategy` → test `estimate_averaged` with a null-token fake), and text-match end-to-end through the estimator (old `TestTextMatchMode` → one test constructing the estimator with `scoring="text_match"` and a `FakeVLM(text="Yes.")`). Then:

```bash
git rm tests/test_semantic_state_estimator.py
```

- [ ] **Step 6: Run the full fast suite**

Run: `pytest tests/test_estimator.py -v` → PASS
Run: `pytest -m "not slow" -q` → all pass.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor!: SemanticStateEstimator as a thin facade; predicates-in, state-out"
```

---

### Task 12: Import-hygiene tests

**Files:**
- Test: `tests/test_imports.py`

**Interfaces:**
- Consumes: the final package layout (Tasks 1–11).
- Produces: the enforcement of the spec's import rules; no new runtime API.

- [ ] **Step 1: Write the tests (they are the deliverable; expected to pass if Tasks 2–11 were faithful — any failure is a real layering bug to fix in source, not in the test)**

```python
# tests/test_imports.py
"""Import hygiene: bare installs import cleanly; optional features fail with
the exact extra named; heavy modules are never imported eagerly.

Each test runs a fresh subprocess so module caches can't mask problems. A
meta-path blocker simulates missing optional packages even though the dev
environment has them installed.
"""

import subprocess
import sys

# NOTE: contains f-string braces, so it is filled via .replace(), not .format().
BLOCKER = """
import sys

class _Blocker:
    def __init__(self, blocked):
        self.blocked = set(blocked)

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.blocked:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None

sys.meta_path.insert(0, _Blocker(BLOCKED_LIST))
"""

HEAVY = ["torch", "torchvision", "transformers", "accelerate",
         "unified_planning", "vllm", "openai", "sklearn"]


def run_python(body: str, blocked=()) -> subprocess.CompletedProcess:
    script = BLOCKER.replace("BLOCKED_LIST", repr(list(blocked))) + body
    return subprocess.run(
        [sys.executable, "-W", "error::Warning", "-c", script],
        capture_output=True, text=True, timeout=120,
    )


class TestBareImport:
    def test_import_s3e_with_all_heavy_deps_blocked(self):
        result = run_python("import s3e", blocked=HEAVY)
        assert result.returncode == 0, result.stderr

    def test_import_s3e_pulls_no_heavy_modules(self):
        body = (
            "import s3e, sys\n"
            "heavy = " + repr(HEAVY) + "\n"
            "loaded = [m for m in heavy if m in sys.modules]\n"
            "assert not loaded, f'eagerly imported: {loaded}'"
        )
        result = run_python(body)
        assert result.returncode == 0, result.stderr

    def test_core_symbols_usable_without_heavy_deps(self):
        body = (
            "from s3e import BinaryAnswers, PredictionSet, QueryEngine, "
            "SemanticStateEstimator, TemplateTranslator\n"
            "space = BinaryAnswers('true', 'false')\n"
            "assert space.true_label == 'true'"
        )
        result = run_python(body, blocked=HEAVY)
        assert result.returncode == 0, result.stderr


class TestMissingDependencyErrors:
    def check(self, body: str, blocked: list, expected_extra: str):
        script = (
            "try:\n"
            + "".join("    " + line + "\n" for line in body.splitlines())
            + "except ImportError as e:\n"
            "    assert 's3e[" + expected_extra + "]' in str(e), str(e)\n"
            "else:\n"
            "    raise SystemExit('expected ImportError')\n"
        )
        result = run_python(script, blocked=blocked)
        assert result.returncode == 0, result.stderr or result.stdout

    def test_huggingface_names_hf_extra(self):
        self.check(
            "from s3e import HuggingFaceVLM", ["torch"], "hf"
        )

    def test_vllm_names_vllm_extra(self):
        self.check("from s3e import VLLMBackend", ["vllm"], "vllm")

    def test_openai_names_openai_extra(self):
        self.check("from s3e import OpenAIVLM", ["openai"], "openai")

    def test_pddl_names_pddl_extra(self):
        self.check(
            "import s3e.pddl", ["unified_planning"], "pddl"
        )

    def test_platt_fit_names_calibration_extra(self):
        self.check(
            "from s3e.calibration.platt import fit_platt_parameters\n"
            "fit_platt_parameters([1.0, -1.0], [True, False])",
            ["sklearn"],
            "calibration",
        )
```

- [ ] **Step 2: Run**

Run: `pytest tests/test_imports.py -v`
Expected: mostly PASS. Two likely legitimate failures to fix in source now:
1. `import s3e.pddl` with `unified_planning` blocked currently raises raw `ModuleNotFoundError` — add to the top of `s3e/pddl/__init__.py` (before other imports): `from .._deps import require` / `require("unified_planning", "pddl", "PDDL support (s3e.pddl)")`.
2. Any residual eager heavy import surfaced by `test_import_s3e_pulls_no_heavy_modules` — chase it via `python -X importtime -c "import s3e"` and push the import down into the leaf.

- [ ] **Step 3: Run the fast suite, commit**

Run: `pytest -m "not slow" -q` → all pass.

```bash
git add tests/test_imports.py s3e/pddl/__init__.py
git commit -m "test: import hygiene for bare installs and per-extra errors"
```

---

### Task 13: Consumer contract tests (MLSS and ViPlan++)

**Files:**
- Test: `tests/consumers/__init__.py` (empty), `tests/consumers/test_mlss_workflow.py`, `tests/consumers/test_viplan_workflow.py`

**Interfaces:**
- Consumes: the entire public API (Tasks 3–12).
- Produces: the migration gate for the two downstream repos. These tests encode each project's workflow shape; when they pass, migration is mechanical renames.

- [ ] **Step 1: Write the failing-or-passing tests (any failure is a source bug — the API contract was fixed in Task 11)**

```python
# tests/consumers/test_mlss_workflow.py
"""Contract tests for the MLSS workflow (make_predictions.py, calibrate_vlm.py).

MLSS pattern: one long-lived estimator per domain; per sample it swaps the
problem, estimates a relevant-atom subset, serializes prediction details to
JSON, and later refits calibration offline without a VLM.
"""

import json

import pytest

from s3e import PredictionSet, SemanticStateEstimator, TemplateTranslator

from conftest import BLOCKSWORLD_DOMAIN, BLOCKSWORLD_PROBLEM, make_blank_image
from fakes import FakeVLM

TEMPLATES = {"on": "Is {0} on {1}?", "clear": "Is {0} clear?"}


@pytest.fixture
def estimator():
    return SemanticStateEstimator.from_pddl(
        BLOCKSWORLD_DOMAIN,
        BLOCKSWORLD_PROBLEM,
        vlm=FakeVLM({"yes": 0.8, "no": 0.1}),
        translator=TemplateTranslator(TEMPLATES),
    )


class TestPerSampleLoop:
    def test_set_problem_then_subset_estimate(self, estimator):
        backend = estimator.engine.backend
        estimator.set_problem(BLOCKSWORLD_DOMAIN, BLOCKSWORLD_PROBLEM)
        assert estimator.engine.backend is backend  # never rebuilt

        subset = estimator.predicates[:3]
        results = estimator.estimate([make_blank_image()], predicates=subset)
        assert list(results) == subset
        prompts = [p for call in backend.calls for p in call["prompts"]]
        assert len(prompts) == len(subset)  # only the subset was queried


class TestDetailsSerialization:
    def test_details_to_json_and_back_without_backend(self, estimator):
        results = estimator.estimate([make_blank_image()])
        payload = json.dumps(results.to_dict())          # what MLSS writes
        restored = PredictionSet.from_dict(json.loads(payload))
        for predicate in results:
            assert restored[predicate].probability == pytest.approx(
                results[predicate].probability
            )
            assert restored[predicate].score == pytest.approx(
                results[predicate].score
            )
            # fields MLSS's payload builder reads:
            p = restored[predicate]
            assert p.masses is not None
            assert p.null_mass is not None
            assert p.argmax_in_interest is not None


class TestOfflineCalibrationRefit:
    def test_collect_save_refit_without_vlm(self, estimator, tmp_path):
        pytest.importorskip("sklearn")
        from s3e.calibration import CalibrationExample, CalibrationSet, PlattCalibrator

        # Half true, half false, with separated masses so a fit converges.
        estimator.engine.backend.script_responses(
            {"Is a": {"yes": 0.9, "no": 0.05}, "Is b": {"yes": 0.1, "no": 0.85}}
        )
        labels = {
            p: ("a" in p.split("(", 1)[1].split(",")[0])
            for p in estimator.predicates
        }
        examples = [
            CalibrationExample(images=[make_blank_image()], state_dict=labels)
        ]
        data = CalibrationSet.collect(estimator, examples)
        data.save(tmp_path / "calib.json")

        # Later, no VLM anywhere:
        reloaded = CalibrationSet.load(tmp_path / "calib.json")
        cal = PlattCalibrator.fit(reloaded, scope="global")
        cal.save(tmp_path / "platt.json")
        restored = PlattCalibrator.load(tmp_path / "platt.json")

        results = estimator.estimate([make_blank_image()])
        calibrated = restored.apply(results)
        assert all(0.0 <= p.probability <= 1.0 for p in calibrated.values())
```

```python
# tests/consumers/test_viplan_workflow.py
"""Contract tests for the ViPlan++ workflow (mpst_exp/predict.py, estimators.py).

ViPlan pattern: a shared prebuilt backend feeds estimators built per domain;
an adapter prepares (build or set_problem) per episode and estimates a
relevant-atom subset; payloads read per-predicate detail fields; backend
type checks must work without importing vllm.
"""

import json
import sys

import pytest

from s3e import SemanticStateEstimator, TemplateTranslator, resolve_backend
from s3e.backends import VLMBackend

from conftest import BLOCKSWORLD_DOMAIN, BLOCKSWORLD_PROBLEM, make_blank_image
from fakes import FakeVLM

TEMPLATES = {"on": "Is {0} on {1}?", "clear": "Is {0} clear?"}


def build_estimator(vlm):
    return SemanticStateEstimator.from_pddl(
        BLOCKSWORLD_DOMAIN, BLOCKSWORLD_PROBLEM,
        vlm=vlm, translator=TemplateTranslator(TEMPLATES),
        batch_size=1,
        inference_kwargs={"temperature": 0.0},
    )


class TestSharedBackend:
    def test_one_backend_many_estimators(self):
        shared = FakeVLM()
        a, b = build_estimator(shared), build_estimator(shared)
        assert a.engine.backend is b.engine.backend is shared

    def test_resolve_backend_is_public(self):
        assert callable(resolve_backend)
        assert resolve_backend(FakeVLM()) is not None


class TestAdapterPattern:
    def test_prepare_then_estimate_subset(self):
        """The S3EAdapter shape: lazy build, then set_problem per episode."""
        estimator = None
        for _episode in range(2):
            if estimator is None:
                estimator = build_estimator(FakeVLM())
            else:
                estimator.set_problem(BLOCKSWORLD_DOMAIN, BLOCKSWORLD_PROBLEM)
            relevant = estimator.predicates[:2]
            details = estimator.estimate([make_blank_image()], predicates=relevant)
            assert set(details) == set(relevant)

    def test_inference_kwargs_reach_backend(self):
        fake = FakeVLM()
        build_estimator(fake).estimate([make_blank_image()])
        assert fake.calls[0]["inference_kwargs"] == {"temperature": 0.0}


class TestPayloadFields:
    def test_every_field_the_payload_builder_reads(self):
        results = build_estimator(FakeVLM()).estimate([make_blank_image()])
        predicate = next(iter(results))
        p = results[predicate]
        payload = {
            "probability": p.probability,
            "score": p.score,
            "masses": p.masses,
            "null_mass": p.null_mass,
            "unassigned_mass": p.unassigned_mass,
            "null_dominated": p.null_dominated,
            "argmax_in_interest": p.argmax_in_interest,
            "answer": p.answer,
        }
        json.dumps(payload)  # must be JSON-serializable as ViPlan writes it


class TestBackendDetectionWithoutVllm:
    def test_isinstance_check_without_importing_vllm(self):
        backend = FakeVLM()
        assert isinstance(backend, VLMBackend)
        assert "vllm" not in sys.modules  # the check itself must not import it
```

- [ ] **Step 2: Run**

Run: `pytest tests/consumers/ -v` → PASS. Any failure is a defect in Tasks 5–11 — fix the source (never weaken the consumer test) and note the fix in the commit message.

- [ ] **Step 3: Run the fast suite, commit**

Run: `pytest -m "not slow" -q` → all pass.

```bash
git add tests/consumers
git commit -m "test: consumer contract suites for MLSS and ViPlan++ workflows"
```

---

### Task 14: README sync, version bump, final verification

**Files:**
- Modify: `README.md`, `pyproject.toml`, `AGENTS.md`, `docs/s3e_walkthrough.ipynb` (marker only)

**Interfaces:**
- Consumes: the finished API.
- Produces: a repo whose documentation matches its code; version `0.4.0`.

- [ ] **Step 1: Update README code samples to the new API**

Rewrite the Quick Start to the three-level story (engine-only, categorical, `from_pddl`) using the exact samples from the spec's "Layer 2"/"Layer 4" sections; update the installation section for the new extras (`pip install "s3e[pddl,hf]"` as the recommended quick start, one line per extra); replace the estimator argument reference (`query_translator` → `translator`, `probability_method` → `scoring`, drop `use_vllm`/`multi_image_strategy`, add `answers=`); replace the Platt section with the `CalibrationSet`/`PlattCalibrator` flow from the spec; keep the citation block. Every code sample in the README must actually run against the new API — verify each by pasting into `python` with a `FakeVLM`-style stub or the SmolVLM model if available.

- [ ] **Step 2: Update AGENTS.md layout section**

Update the Layout list to the new structure (`s3e/engine/`, `s3e/backends/`, `s3e/calibration/`, `s3e/estimator.py`) and the single-test examples to existing node IDs (e.g. `pytest tests/engine/test_answers.py::TestBinaryAnswers::test_default_yes_no`).

- [ ] **Step 3: Mark the walkthrough notebook as pending**

Add a leading markdown cell to `docs/s3e_walkthrough.ipynb`: "⚠️ This walkthrough targets the pre-0.4 API and is being rewritten for the new architecture (docs sub-project)." (Full notebook rewrite belongs to the docs sub-project.)

- [ ] **Step 4: Bump version**

In `pyproject.toml`: `version = "0.4.0"`.

- [ ] **Step 5: Full verification**

Run: `python -m compileall s3e tests` → no errors
Run: `pytest -m "not slow" -q` → all pass
Run: `pytest --collect-only -q | tail -3` → sane test count, no collection errors
If a GPU is available (see memory note `vllm-slow-test-cuda-toolchain.md` for env vars): `pytest -m slow -q` → pass; otherwise state plainly in the final report that slow tests were not run.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "docs: sync README and AGENTS.md with the 0.4 architecture; bump to 0.4.0"
```

---

## Post-plan follow-ups (out of scope here)

- Migrate MLSS and ViPlan++ against 0.4.0 (mechanical renames per the spec's migration table).
- Sub-project 2: CI, lint/format/type-check tooling.
- Sub-project 3: docs site, notebook rewrite, CONTRIBUTING/CODE_OF_CONDUCT.
- Sub-project 4: PyPI release, Zenodo DOI, JOSS paper, demo.
