# vLLM VLM Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a vLLM-powered `VLLMBackend` to `s3e`, activated by a `use_vllm` flag on `SemanticStateEstimator`, giving transparent single-node multi-GPU local inference while preserving the existing `VLMBackend`/`VLMOutput` contract.

**Architecture:** A new `VLLMBackend(VLMBackend)` in `s3e/vlm/vllm.py` (parallel to `huggingface.py` and `openai.py`) wraps a persistent `vllm.LLM` engine. It uses tensor parallelism across all local GPUs by default and `LLM.chat()` for inference so vLLM owns chat-template rendering and image-placeholder expansion. `SemanticStateEstimator._build_vlm_from_string` routes a non-OpenAI model string to it when `use_vllm=True`. Every non-obvious design choice is documented inline in the code, per the spec.

**Tech Stack:** Python ≥3.10, vLLM (optional dep), PyTorch, PIL, pytest (mocked unit tests; slow GPU integration test).

**Spec:** `docs/superpowers/specs/2026-05-26-vllm-backend-design.md`

**Branch:** `vllm-backend` (already created off `main`).

**Conventions to follow:**
- Run the fast suite with `pytest -m "not slow"` (per AGENTS.md). It must pass with no GPU and no `vllm` installed.
- Mocked tests patch `s3e.vlm.vllm.LLM` and `s3e.vlm.vllm.SamplingParams`, mirroring `TestHuggingFaceVLMMocked`/`TestOpenAIVLM` in `tests/test_vlm_backends.py`.
- Commit message trailer (every commit):
  ```
  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  ```

---

### Task 1: Add the `vllm` optional dependency

**Files:**
- Modify: `pyproject.toml:31-34`

- [ ] **Step 1: Add the optional-dependency entry**

In `pyproject.toml`, the current block is:

```toml
[project.optional-dependencies]
openai = ["openai"]
calibration = ["scikit-learn"]
dev = ["pytest"]
```

Change it to:

```toml
[project.optional-dependencies]
openai = ["openai"]
vllm = ["vllm"]
calibration = ["scikit-learn"]
dev = ["pytest"]
```

- [ ] **Step 2: Verify the edit landed**

Run: `grep -n 'vllm = \["vllm"\]' pyproject.toml`
Expected: prints the line, e.g. `32:vllm = ["vllm"]`
(Avoids `tomllib`, which is unavailable on Python 3.10 — the repo's minimum.)

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: add vllm optional dependency

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 2: Create `VLLMBackend` construction (module skeleton + `__init__`)

**Files:**
- Create: `s3e/vlm/vllm.py`
- Test: `tests/test_vlm_backends.py` (append a new `TestVLLMBackendMocked` class)

- [ ] **Step 1: Write the failing tests**

Append to the **end** of `tests/test_vlm_backends.py`:

```python
import math as _math  # noqa: E402  (module-level math for vLLM helpers)


def _make_logprob(token, logprob):
    """Build a mock vLLM Logprob (has .decoded_token and .logprob)."""
    item = MagicMock()
    item.decoded_token = token
    item.logprob = logprob
    return item


def _make_logprobs_output(token_logprobs):
    """Build a mock vLLM RequestOutput for logprobs mode.

    token_logprobs: list of (token_str, logprob_float). Keyed by fake ids.
    """
    completion = MagicMock()
    completion.logprobs = [
        {idx: _make_logprob(tok, lp) for idx, (tok, lp) in enumerate(token_logprobs)}
    ]
    output = MagicMock()
    output.outputs = [completion]
    return output


def _make_text_output(text):
    """Build a mock vLLM RequestOutput for text mode."""
    completion = MagicMock()
    completion.text = text
    output = MagicMock()
    output.outputs = [completion]
    return output


class TestVLLMBackendMocked:
    """Unit tests for VLLMBackend with vllm mocked out."""

    @patch("torch.cuda.device_count", return_value=3)
    @patch("s3e.vlm.vllm.SamplingParams")
    @patch("s3e.vlm.vllm.LLM")
    def test_tensor_parallel_defaults_to_all_local_gpus(
        self, mock_llm_cls, mock_sp_cls, mock_device_count
    ):
        from s3e.vlm.vllm import VLLMBackend

        VLLMBackend("test/model")

        kwargs = mock_llm_cls.call_args.kwargs
        assert kwargs["model"] == "test/model"
        assert kwargs["tensor_parallel_size"] == 3

    @patch("torch.cuda.device_count", return_value=8)
    @patch("s3e.vlm.vllm.SamplingParams")
    @patch("s3e.vlm.vllm.LLM")
    def test_tensor_parallel_explicit_override(
        self, mock_llm_cls, mock_sp_cls, mock_device_count
    ):
        from s3e.vlm.vllm import VLLMBackend

        VLLMBackend("test/model", tensor_parallel_size=2)

        assert mock_llm_cls.call_args.kwargs["tensor_parallel_size"] == 2

    @patch("torch.cuda.device_count", return_value=1)
    @patch("s3e.vlm.vllm.SamplingParams")
    @patch("s3e.vlm.vllm.LLM")
    def test_max_logprobs_full_vocab_by_default(
        self, mock_llm_cls, mock_sp_cls, mock_device_count
    ):
        from s3e.vlm.vllm import VLLMBackend

        VLLMBackend("test/model")

        assert mock_llm_cls.call_args.kwargs["max_logprobs"] == -1

    @patch("torch.cuda.device_count", return_value=1)
    @patch("s3e.vlm.vllm.SamplingParams")
    @patch("s3e.vlm.vllm.LLM")
    def test_max_logprobs_finite_when_num_logprobs_set(
        self, mock_llm_cls, mock_sp_cls, mock_device_count
    ):
        from s3e.vlm.vllm import VLLMBackend

        VLLMBackend("test/model", num_logprobs=5)

        assert mock_llm_cls.call_args.kwargs["max_logprobs"] == 5

    @patch("torch.cuda.device_count", return_value=1)
    @patch("s3e.vlm.vllm.SamplingParams")
    @patch("s3e.vlm.vllm.LLM")
    def test_engine_kwargs_forwarded(
        self, mock_llm_cls, mock_sp_cls, mock_device_count
    ):
        from s3e.vlm.vllm import VLLMBackend

        VLLMBackend("test/model", gpu_memory_utilization=0.5, max_model_len=2048)

        kwargs = mock_llm_cls.call_args.kwargs
        assert kwargs["gpu_memory_utilization"] == 0.5
        assert kwargs["max_model_len"] == 2048

    @patch("s3e.vlm.vllm.SamplingParams", None)
    @patch("s3e.vlm.vllm.LLM", None)
    def test_missing_vllm_raises_install_guidance(self):
        from s3e.vlm.vllm import VLLMBackend

        with pytest.raises(ImportError, match=r"pip install s3e\[vllm\]"):
            VLLMBackend("test/model")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_vlm_backends.py::TestVLLMBackendMocked -v`
Expected: FAIL — `ModuleNotFoundError: No module named 's3e.vlm.vllm'`

- [ ] **Step 3: Create the module with the lazy guard and `__init__`**

Create `s3e/vlm/vllm.py`:

```python
"""vLLM VLM backend.

A :class:`VLMBackend` implementation that runs a local HuggingFace
vision-language model through `vLLM <https://docs.vllm.ai/>`_ for transparent,
high-throughput, single-node multi-GPU inference. Selected by passing
``use_vllm=True`` to :class:`SemanticStateEstimator` with a non-OpenAI model id.

Design notes (kept in code on purpose so future maintainers can change the
design with full context):

* The module is named ``vllm.py`` to match the existing backend naming
  convention (``openai.py`` imports ``openai``; ``huggingface.py`` uses
  ``transformers``). Python 3 uses absolute imports, so ``from vllm import ...``
  below resolves to the installed *package*, not this module -- the same safe
  pattern ``openai.py`` already relies on.
* ``vllm`` is an optional dependency, imported lazily so importing s3e never
  requires it. Construction raises a helpful :class:`ImportError` when missing.
"""

import math

import torch

from .backend import VLMBackend, VLMOutput

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None  # type: ignore[assignment]
    SamplingParams = None  # type: ignore[assignment]


def _check_vllm_installed() -> None:
    if LLM is None or SamplingParams is None:
        raise ImportError(
            "The 'vllm' package is required for VLLMBackend. "
            "Install it with: pip install s3e[vllm]"
        )


class VLLMBackend(VLMBackend):
    """VLM backend backed by a local vLLM engine.

    The estimator always sends the *same* images with *many* prompts (one per
    grounded predicate); vLLM batches them in a single ``chat`` call and shards
    each layer across GPUs under the hood, so the user can treat a multi-GPU box
    as one big GPU.

    Args:
        model_id: HuggingFace model identifier loaded by vLLM.
        tensor_parallel_size: Number of GPUs to shard each layer across. When
            ``None`` (default), uses all locally visible GPUs
            (``torch.cuda.device_count()``), so the user need not specify a
            count. Multi-node execution is intentionally out of scope but can be
            reached by passing vLLM engine kwargs (e.g. ``pipeline_parallel_size``,
            ``distributed_executor_backend``) through ``engine_kwargs``.
        num_logprobs: Number of top tokens to include in ``token_probs``.
            ``None`` (default) returns the full-vocabulary distribution,
            mirroring :class:`HuggingFaceVLM`. This maps to vLLM ``logprobs=-1``
            with the engine built using ``max_logprobs=-1`` -- note the docs' OOM
            caveat for full-vocab logprobs. A finite ``k`` returns the top ``k``.
        **engine_kwargs: Forwarded verbatim to the ``vllm.LLM`` constructor
            (e.g. ``gpu_memory_utilization``, ``dtype``, ``max_model_len``,
            ``quantization``), mirroring how :class:`HuggingFaceVLM` forwards
            ``**model_kwargs``.

    Notes:
        There is intentionally no ``max_new_tokens`` parameter: the HF backend
        stores one but never forwards it, so it is dead state we do not
        reproduce. Generation length is controlled through ``inference_kwargs``
        (vLLM ``SamplingParams.max_tokens``).

        This backend relies on the model's chat template (see
        :meth:`query_batch`); a model without one is unsupported on this path.
    """

    def __init__(
        self,
        model_id: str,
        tensor_parallel_size: int | None = None,
        num_logprobs: int | None = None,
        **engine_kwargs,
    ):
        _check_vllm_installed()
        self.model_id = model_id
        self.num_logprobs = num_logprobs

        # Default to every locally visible GPU so the user does not have to
        # specify a count; an explicit value always wins. max(..., 1) keeps a
        # sane value when no CUDA device is visible.
        if tensor_parallel_size is None:
            tensor_parallel_size = max(torch.cuda.device_count(), 1)
        self.tensor_parallel_size = tensor_parallel_size

        # max_logprobs is fixed at engine-construction time, and num_logprobs is
        # also a constructor arg, so the two are always consistent. None means
        # "full vocab" (-1), matching HuggingFaceVLM's default; a finite k caps
        # at k. Requesting more logprobs per call than max_logprobs makes vLLM
        # raise, which is why we size the cap from num_logprobs here.
        self.llm = LLM(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            max_logprobs=(-1 if num_logprobs is None else num_logprobs),
            **engine_kwargs,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_vlm_backends.py::TestVLLMBackendMocked -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add s3e/vlm/vllm.py tests/test_vlm_backends.py
git commit -m "feat: add VLLMBackend construction with lazy vllm guard

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 3: Logprobs-mode inference (`query`, `query_batch`, sampling params, extraction)

**Files:**
- Modify: `s3e/vlm/vllm.py` (add methods to `VLLMBackend`)
- Test: `tests/test_vlm_backends.py` (add tests to `TestVLLMBackendMocked`)

- [ ] **Step 1: Write the failing tests**

Add these methods inside `TestVLLMBackendMocked` in `tests/test_vlm_backends.py`:

```python
    def _make_backend(self, mock_llm_cls, num_logprobs=None):
        """Construct a VLLMBackend whose engine is the mocked LLM instance."""
        from s3e.vlm.vllm import VLLMBackend

        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm
        with patch("torch.cuda.device_count", return_value=1):
            backend = VLLMBackend("test/model", num_logprobs=num_logprobs)
        return backend, mock_llm

    @patch("s3e.vlm.vllm.SamplingParams")
    @patch("s3e.vlm.vllm.LLM")
    def test_query_batch_builds_one_chat_call_with_conversation_structure(
        self, mock_llm_cls, mock_sp_cls
    ):
        backend, mock_llm = self._make_backend(mock_llm_cls)
        mock_llm.chat.return_value = [
            _make_logprobs_output([("yes", _math.log(0.8))]),
            _make_logprobs_output([("no", _math.log(0.6))]),
        ]
        img = Image.new("RGB", (64, 64))

        results = backend.query_batch([img], ["q1", "q2"], system_prompt="sys")

        assert mock_llm.chat.call_count == 1
        conversations = mock_llm.chat.call_args.args[0]
        assert len(conversations) == 2
        # Each conversation: system message then user message with image + text.
        first = conversations[0]
        assert first[0] == {"role": "system", "content": "sys"}
        assert first[1]["role"] == "user"
        assert first[1]["content"][0]["type"] == "image_pil"
        assert first[1]["content"][0]["image_pil"] is img
        assert first[1]["content"][-1] == {"type": "text", "text": "q1"}
        assert len(results) == 2

    @patch("s3e.vlm.vllm.SamplingParams")
    @patch("s3e.vlm.vllm.LLM")
    def test_query_batch_omits_system_message_when_absent(
        self, mock_llm_cls, mock_sp_cls
    ):
        backend, mock_llm = self._make_backend(mock_llm_cls)
        mock_llm.chat.return_value = [_make_logprobs_output([("yes", _math.log(0.5))])]

        backend.query_batch([], ["q1"])

        conversation = mock_llm.chat.call_args.args[0][0]
        assert all(m["role"] != "system" for m in conversation)

    @patch("s3e.vlm.vllm.SamplingParams")
    @patch("s3e.vlm.vllm.LLM")
    def test_logprobs_mode_sampling_params_defaults(
        self, mock_llm_cls, mock_sp_cls
    ):
        backend, mock_llm = self._make_backend(mock_llm_cls, num_logprobs=None)
        mock_llm.chat.return_value = [_make_logprobs_output([("yes", _math.log(0.5))])]

        backend.query_batch([], ["q1"])

        kwargs = mock_sp_cls.call_args.kwargs
        assert kwargs["max_tokens"] == 1
        assert kwargs["temperature"] == 0.0
        assert kwargs["logprobs"] == -1  # full vocab when num_logprobs is None

    @patch("s3e.vlm.vllm.SamplingParams")
    @patch("s3e.vlm.vllm.LLM")
    def test_logprobs_value_follows_num_logprobs(self, mock_llm_cls, mock_sp_cls):
        backend, mock_llm = self._make_backend(mock_llm_cls, num_logprobs=4)
        mock_llm.chat.return_value = [_make_logprobs_output([("yes", _math.log(0.5))])]

        backend.query_batch([], ["q1"])

        assert mock_sp_cls.call_args.kwargs["logprobs"] == 4

    @patch("s3e.vlm.vllm.SamplingParams")
    @patch("s3e.vlm.vllm.LLM")
    def test_max_tokens_is_overridable_in_logprobs_mode(
        self, mock_llm_cls, mock_sp_cls
    ):
        backend, mock_llm = self._make_backend(mock_llm_cls)
        mock_llm.chat.return_value = [_make_logprobs_output([("yes", _math.log(0.5))])]

        backend.query_batch([], ["q1"], max_tokens=7)

        assert mock_sp_cls.call_args.kwargs["max_tokens"] == 7

    @patch("s3e.vlm.vllm.SamplingParams")
    @patch("s3e.vlm.vllm.LLM")
    def test_logprobs_extraction_converts_to_probabilities(
        self, mock_llm_cls, mock_sp_cls
    ):
        backend, mock_llm = self._make_backend(mock_llm_cls)
        mock_llm.chat.return_value = [
            _make_logprobs_output([("yes", _math.log(0.7)), ("no", _math.log(0.3))])
        ]

        result = backend.query([], "q1")

        assert result.text is None
        assert result.token_probs["yes"] == pytest.approx(0.7)
        assert result.token_probs["no"] == pytest.approx(0.3)

    @patch("s3e.vlm.vllm.SamplingParams")
    @patch("s3e.vlm.vllm.LLM")
    def test_logprobs_extraction_sums_duplicate_tokens(
        self, mock_llm_cls, mock_sp_cls
    ):
        backend, mock_llm = self._make_backend(mock_llm_cls)
        mock_llm.chat.return_value = [
            _make_logprobs_output([("same", _math.log(0.4)), ("same", _math.log(0.25))])
        ]

        result = backend.query([], "q1")

        assert set(result.token_probs) == {"same"}
        assert result.token_probs["same"] == pytest.approx(0.65)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_vlm_backends.py::TestVLLMBackendMocked -v`
Expected: FAIL — `AttributeError: 'VLLMBackend' object has no attribute 'query_batch'` (the abstract `query` is unimplemented).

- [ ] **Step 3: Add the inference methods**

Append these methods to the `VLLMBackend` class in `s3e/vlm/vllm.py` (after `__init__`):

```python
    def query(self, images, prompt, system_prompt=None, generate=False, **inference_kwargs):
        """Send a single query to the vLLM engine."""
        return self.query_batch(
            images, [prompt], system_prompt, generate, **inference_kwargs
        )[0]

    def query_batch(self, images, prompts, system_prompt=None, generate=False, **inference_kwargs):
        """Send multiple prompts against the same images in one batched call.

        We use ``LLM.chat`` (not ``LLM.generate`` + ``multi_modal_data``) so
        vLLM applies the model's own chat template and performs the
        resolution-dependent image-placeholder expansion -- mirroring how
        :class:`HuggingFaceVLM` relies on ``processor.apply_chat_template`` and
        owning zero per-model placeholder logic. The trade-off: a model without
        a chat template is unsupported here.
        """
        if not prompts:
            return []

        sampling_params = self._build_sampling_params(generate, **inference_kwargs)

        # Every prompt shares the same images, so build the image content once
        # and reuse it across the per-prompt conversations.
        image_content = [{"type": "image_pil", "image_pil": image} for image in images]
        conversations = []
        for prompt in prompts:
            messages = []
            if system_prompt is not None:
                messages.append({"role": "system", "content": system_prompt})
            messages.append(
                {
                    "role": "user",
                    "content": image_content + [{"type": "text", "text": prompt}],
                }
            )
            conversations.append(messages)

        # chat() accepts a batched list of conversations and returns outputs in
        # input order.
        outputs = self.llm.chat(conversations, sampling_params)
        return [self._to_vlm_output(output, generate) for output in outputs]

    def _build_sampling_params(self, generate, **inference_kwargs):
        """Build vLLM SamplingParams for the requested mode.

        ``inference_kwargs`` are user overrides. We only *force* the settings
        the estimator's contract depends on and use ``setdefault`` for the rest
        -- the same pattern OpenAIVLM uses to force logprobs and HuggingFaceVLM
        uses to default ``logits_to_keep=1``.
        """
        if generate:
            # Text mode: do NOT bound max_tokens, so a model can reason as it
            # sees fit (the OpenAI backend documents the same philosophy).
            inference_kwargs.setdefault("temperature", 0.0)
        else:
            # Logprobs mode: one decode step is the memory-minimal default and
            # the direct analog of HuggingFaceVLM's logits_to_keep=1. It is a
            # setdefault, not a force, so a user may override it. Free-form
            # reasoning belongs in text mode, not here.
            inference_kwargs.setdefault("max_tokens", 1)
            inference_kwargs.setdefault("temperature", 0.0)
            # Force logprobs on (analog of OpenAIVLM forcing logprobs=True).
            # -1 == full vocab; otherwise the configured top-k.
            inference_kwargs["logprobs"] = (
                -1 if self.num_logprobs is None else self.num_logprobs
            )
        return SamplingParams(**inference_kwargs)

    def _to_vlm_output(self, output, generate):
        """Convert one vLLM RequestOutput into a :class:`VLMOutput`."""
        completion = output.outputs[0]
        if generate:
            return VLMOutput(token_probs=None, text=completion.text)

        # Logprobs mode reads the distribution over the next answer token. With
        # the default max_tokens=1 this is the only generated position and
        # reproduces HuggingFaceVLM's single-forward next-token distribution.
        logprobs_seq = completion.logprobs
        if not logprobs_seq or logprobs_seq[0] is None:
            # With logprobs forced and max_tokens>=1 this should never happen;
            # treat it as an internal inconsistency rather than silently
            # returning empty probabilities (see AGENTS.md error-handling rules).
            raise RuntimeError(
                "vLLM returned no logprobs for a request despite logprobs being "
                "requested. This indicates an internal inconsistency in the vLLM "
                "output; check the installed vLLM version and SamplingParams."
            )

        token_probs: dict[str, float] = {}
        for logprob in logprobs_seq[0].values():
            # Sum probabilities of duplicate decoded token strings, matching the
            # dedup HuggingFaceVLM and OpenAIVLM perform.
            token = logprob.decoded_token
            token_probs[token] = token_probs.get(token, 0.0) + math.exp(logprob.logprob)

        return VLMOutput(token_probs=token_probs, text=None)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_vlm_backends.py::TestVLLMBackendMocked -v`
Expected: PASS (all `TestVLLMBackendMocked` tests, including the new logprobs ones)

- [ ] **Step 5: Commit**

```bash
git add s3e/vlm/vllm.py tests/test_vlm_backends.py
git commit -m "feat: add vLLM logprobs-mode inference via LLM.chat

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 4: Text (generate) mode

**Files:**
- Test: `tests/test_vlm_backends.py` (add tests to `TestVLLMBackendMocked`)

Note: the implementation from Task 3 already covers text mode (`_build_sampling_params` and `_to_vlm_output` both branch on `generate`). These tests lock that behavior in. If they pass immediately, that is expected — still run them red-first by writing them before reading the Task 3 code, then confirm green.

- [ ] **Step 1: Write the failing tests**

Add inside `TestVLLMBackendMocked`:

```python
    @patch("s3e.vlm.vllm.SamplingParams")
    @patch("s3e.vlm.vllm.LLM")
    def test_text_mode_returns_text_and_no_token_probs(
        self, mock_llm_cls, mock_sp_cls
    ):
        backend, mock_llm = self._make_backend(mock_llm_cls)
        mock_llm.chat.return_value = [
            _make_text_output("yes"),
            _make_text_output("no"),
        ]

        results = backend.query_batch([], ["q1", "q2"], generate=True)

        assert [r.text for r in results] == ["yes", "no"]
        assert all(r.token_probs is None for r in results)

    @patch("s3e.vlm.vllm.SamplingParams")
    @patch("s3e.vlm.vllm.LLM")
    def test_text_mode_does_not_bound_or_request_logprobs(
        self, mock_llm_cls, mock_sp_cls
    ):
        backend, mock_llm = self._make_backend(mock_llm_cls)
        mock_llm.chat.return_value = [_make_text_output("yes")]

        backend.query_batch([], ["q1"], generate=True)

        kwargs = mock_sp_cls.call_args.kwargs
        assert "max_tokens" not in kwargs  # model may reason freely
        assert "logprobs" not in kwargs
        assert kwargs["temperature"] == 0.0
```

- [ ] **Step 2: Run tests to verify status**

Run: `pytest tests/test_vlm_backends.py::TestVLLMBackendMocked::test_text_mode_returns_text_and_no_token_probs tests/test_vlm_backends.py::TestVLLMBackendMocked::test_text_mode_does_not_bound_or_request_logprobs -v`
Expected: PASS (text mode implemented in Task 3). If a test FAILS, fix the Task 3 implementation to match.

- [ ] **Step 3: Commit**

```bash
git add tests/test_vlm_backends.py
git commit -m "test: cover vLLM text-generation mode

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 5: Edge cases — empty prompts and missing logprobs

**Files:**
- Test: `tests/test_vlm_backends.py` (add tests to `TestVLLMBackendMocked`)

- [ ] **Step 1: Write the failing tests**

Add inside `TestVLLMBackendMocked`:

```python
    @patch("s3e.vlm.vllm.SamplingParams")
    @patch("s3e.vlm.vllm.LLM")
    def test_empty_prompts_returns_empty_list_without_engine_call(
        self, mock_llm_cls, mock_sp_cls
    ):
        backend, mock_llm = self._make_backend(mock_llm_cls)

        assert backend.query_batch([Image.new("RGB", (8, 8))], []) == []
        mock_llm.chat.assert_not_called()

    @patch("s3e.vlm.vllm.SamplingParams")
    @patch("s3e.vlm.vllm.LLM")
    def test_missing_logprobs_raises_informative_error(
        self, mock_llm_cls, mock_sp_cls
    ):
        backend, mock_llm = self._make_backend(mock_llm_cls)
        completion = MagicMock()
        completion.logprobs = None  # vLLM returned no logprobs
        bad_output = MagicMock()
        bad_output.outputs = [completion]
        mock_llm.chat.return_value = [bad_output]

        with pytest.raises(RuntimeError, match="no logprobs"):
            backend.query([], "q1")
```

- [ ] **Step 2: Run tests to verify status**

Run: `pytest tests/test_vlm_backends.py::TestVLLMBackendMocked::test_empty_prompts_returns_empty_list_without_engine_call tests/test_vlm_backends.py::TestVLLMBackendMocked::test_missing_logprobs_raises_informative_error -v`
Expected: PASS (both behaviors implemented in Task 3). If either FAILS, fix the Task 3 implementation.

- [ ] **Step 3: Commit**

```bash
git add tests/test_vlm_backends.py
git commit -m "test: cover vLLM empty-prompt and missing-logprobs paths

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 6: Re-export `VLLMBackend`

**Files:**
- Modify: `s3e/vlm/__init__.py`
- Modify: `s3e/__init__.py:20` and `s3e/__init__.py:40-46`
- Test: `tests/test_vlm_backends.py` (add a small import test)

- [ ] **Step 1: Write the failing test**

Add at the **end** of `tests/test_vlm_backends.py`:

```python
def test_vllm_backend_is_exported():
    import s3e
    from s3e.vlm import VLLMBackend as FromVlm
    from s3e import VLLMBackend as FromTop

    assert FromVlm is FromTop
    assert "VLLMBackend" in s3e.__all__
    assert "VLLMBackend" in s3e.vlm.__all__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vlm_backends.py::test_vllm_backend_is_exported -v`
Expected: FAIL — `ImportError: cannot import name 'VLLMBackend' from 's3e.vlm'`

- [ ] **Step 3: Update `s3e/vlm/__init__.py`**

Replace its body with:

```python
"""VLM (Vision-Language Model) backends for s3e.

This subpackage provides the :class:`VLMBackend` abstraction and
concrete implementations for HuggingFace Transformers models, the
OpenAI API, and vLLM.
"""

from .backend import VLMBackend, VLMOutput
from .huggingface import HuggingFaceVLM
from .openai import OpenAIVLM
from .vllm import VLLMBackend

__all__ = ["VLMBackend", "VLMOutput", "HuggingFaceVLM", "OpenAIVLM", "VLLMBackend"]
```

- [ ] **Step 4: Update `s3e/__init__.py`**

Change the VLM import line (currently `from .vlm import VLMBackend, VLMOutput, HuggingFaceVLM, OpenAIVLM`) to:

```python
from .vlm import VLMBackend, VLMOutput, HuggingFaceVLM, OpenAIVLM, VLLMBackend
```

And in the `__all__` list, add `"VLLMBackend"` immediately after `"OpenAIVLM",`:

```python
    "OpenAIVLM",
    "VLLMBackend",
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_vlm_backends.py::test_vllm_backend_is_exported -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add s3e/vlm/__init__.py s3e/__init__.py tests/test_vlm_backends.py
git commit -m "feat: export VLLMBackend from s3e and s3e.vlm

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 7: Route `use_vllm` in `SemanticStateEstimator`

**Files:**
- Modify: `s3e/semantic_state_estimator.py` (`__init__` signature ~95-113; backend build ~116-120; `_build_vlm_from_string` ~195-203; class docstring Args ~91-93)
- Test: `tests/test_semantic_state_estimator.py` (append a new `TestUseVllmRouting` class)

- [ ] **Step 1: Write the failing tests**

Append to the **end** of `tests/test_semantic_state_estimator.py`:

```python
from unittest.mock import MagicMock, patch


class TestUseVllmRouting:
    def test_use_vllm_routes_string_model_to_vllm_backend(self):
        with patch("s3e.vlm.vllm.VLLMBackend") as mock_backend:
            mock_instance = MagicMock()
            mock_backend.return_value = mock_instance
            est = SemanticStateEstimator(
                BLOCKSWORLD_DOMAIN,
                BLOCKSWORLD_PROBLEM,
                vlm="some/local-model",
                use_vllm=True,
                vlm_kwargs={"tensor_parallel_size": 2},
            )
        mock_backend.assert_called_once_with(
            "some/local-model", tensor_parallel_size=2
        )
        assert est.vlm is mock_instance

    def test_use_vllm_with_openai_model_raises(self):
        with pytest.raises(ValueError, match="not compatible with OpenAI"):
            SemanticStateEstimator(
                BLOCKSWORLD_DOMAIN,
                BLOCKSWORLD_PROBLEM,
                vlm="OpenAI/gpt-4o",
                use_vllm=True,
            )

    def test_use_vllm_ignored_when_instance_passed(self):
        fake = FakeVLM()
        est = SemanticStateEstimator(
            BLOCKSWORLD_DOMAIN,
            BLOCKSWORLD_PROBLEM,
            vlm=fake,
            use_vllm=True,
        )
        assert est.vlm is fake
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_semantic_state_estimator.py::TestUseVllmRouting -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'use_vllm'`

- [ ] **Step 3: Add the `use_vllm` parameter to `__init__`**

In `s3e/semantic_state_estimator.py`, the signature currently ends:

```python
        vlm_kwargs: dict | None = None,
        inference_kwargs: dict | None = None,
    ):
```

Change it to:

```python
        vlm_kwargs: dict | None = None,
        use_vllm: bool = False,
        inference_kwargs: dict | None = None,
    ):
```

- [ ] **Step 4: Store and forward `use_vllm` at the backend-build site**

The current block is:

```python
        # --- VLM backend ---
        if isinstance(vlm, str):
            self.vlm = self._build_vlm_from_string(vlm, vlm_kwargs or {})
        else:
            self.vlm = vlm
```

Change it to:

```python
        # --- VLM backend ---
        # use_vllm only affects the string path; an explicit backend instance is
        # used as-is.
        self.use_vllm = use_vllm
        if isinstance(vlm, str):
            self.vlm = self._build_vlm_from_string(vlm, vlm_kwargs or {}, use_vllm)
        else:
            self.vlm = vlm
```

- [ ] **Step 5: Update `_build_vlm_from_string`**

The current method is:

```python
    @staticmethod
    def _build_vlm_from_string(vlm_id: str, vlm_kwargs: dict) -> VLMBackend:
        """Construct a VLM backend from a model ID string."""
        if vlm_id.startswith(OPENAI_MODEL_IDENTIFIER):
            from .vlm.openai import OpenAIVLM
            return OpenAIVLM(vlm_id, **vlm_kwargs)
        else:
            from .vlm.huggingface import HuggingFaceVLM
            return HuggingFaceVLM(vlm_id, **vlm_kwargs)
```

Replace it with:

```python
    @staticmethod
    def _build_vlm_from_string(
        vlm_id: str, vlm_kwargs: dict, use_vllm: bool = False
    ) -> VLMBackend:
        """Construct a VLM backend from a model ID string.

        ``use_vllm`` routes a non-OpenAI model through the vLLM engine instead
        of plain transformers. It is incompatible with ``OpenAI/`` models (which
        run against the hosted API, not a local engine).
        """
        if vlm_id.startswith(OPENAI_MODEL_IDENTIFIER):
            if use_vllm:
                raise ValueError(
                    "use_vllm=True is not compatible with OpenAI/ models."
                )
            from .vlm.openai import OpenAIVLM
            return OpenAIVLM(vlm_id, **vlm_kwargs)
        if use_vllm:
            from .vlm.vllm import VLLMBackend
            return VLLMBackend(vlm_id, **vlm_kwargs)
        from .vlm.huggingface import HuggingFaceVLM
        return HuggingFaceVLM(vlm_id, **vlm_kwargs)
```

- [ ] **Step 6: Document `use_vllm` in the class docstring**

In the `SemanticStateEstimator` docstring Args, the line is:

```python
        vlm_kwargs: Extra kwargs for VLM construction (only when vlm is a string).
```

Add immediately after it:

```python
        use_vllm: When ``vlm`` is a model-id string for a non-OpenAI model,
            route it through the vLLM engine (:class:`VLLMBackend`) instead of
            plain transformers. Ignored when ``vlm`` is a backend instance; an
            ``OpenAI/`` model with ``use_vllm=True`` raises ``ValueError``.
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_semantic_state_estimator.py::TestUseVllmRouting -v`
Expected: PASS (3 tests)

- [ ] **Step 8: Commit**

```bash
git add s3e/semantic_state_estimator.py tests/test_semantic_state_estimator.py
git commit -m "feat: route use_vllm to VLLMBackend in SemanticStateEstimator

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 8: Slow GPU integration test

**Files:**
- Test: `tests/test_vlm_backends.py` (append a `TestVLLMBackendIntegration` class)

- [ ] **Step 1: Write the integration test**

Append to the **end** of `tests/test_vlm_backends.py`:

```python
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="vLLM requires CUDA")
class TestVLLMBackendIntegration:
    """Integration test with a tiny real model via vLLM.

    Requires a GPU and an installed `vllm`. Skipped otherwise. Run with:
    pytest -m slow
    """

    TINY_VLM_ID = "katuni4ka/tiny-random-llava"

    def test_loads_and_queries_logprobs(self):
        from s3e.vlm.vllm import VLLMBackend

        backend = VLLMBackend(
            self.TINY_VLM_ID,
            tensor_parallel_size=1,
            num_logprobs=2,
            gpu_memory_utilization=0.3,
            max_model_len=2048,
        )
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        result = backend.query([img], "Is this a test?")

        assert isinstance(result, VLMOutput)
        assert isinstance(result.token_probs, dict)
        assert 0 < len(result.token_probs) <= 2
        assert all(p >= 0 for p in result.token_probs.values())
```

- [ ] **Step 2: Verify it is collected but skipped on this host**

Run: `pytest tests/test_vlm_backends.py::TestVLLMBackendIntegration -v`
Expected: SKIPPED (no CUDA on this host) — confirms the guard works and the fast suite is unaffected.

- [ ] **Step 3: Commit**

```bash
git add tests/test_vlm_backends.py
git commit -m "test: add slow GPU integration test for VLLMBackend

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 9: Final verification & documentation review

**Files:**
- Review only: `s3e/vlm/vllm.py`, `s3e/semantic_state_estimator.py`

- [ ] **Step 1: Confirm every design choice is documented in code**

Open `s3e/vlm/vllm.py` and confirm an inline comment or docstring explains each of these (per the spec's Documentation requirements). Add any that are missing:
- Why `LLM.chat()` over `LLM.generate()` + `multi_modal_data` (and the chat-template limitation) — in `query_batch` docstring.
- Why `max_tokens=1` is a `setdefault` (not forced) in logprobs mode, and that reasoning lives in text mode — in `_build_sampling_params`.
- Why `logprobs` is forced — in `_build_sampling_params`.
- The `num_logprobs` → `max_logprobs` mapping (`None` → `-1`), consistency at construction, OOM caveat — in `__init__` and the class docstring.
- Why `tensor_parallel_size` defaults to the local GPU count; multi-node out of scope but reachable via `engine_kwargs` — in the class docstring and `__init__`.
- Why there is no `max_new_tokens` parameter — in the class docstring Notes.
- Why missing logprobs raises rather than returning empty `token_probs` — in `_to_vlm_output`.
- The `import vllm` / module-name safety note — in the module docstring.
- Token-probability dedup matching the other backends and `VLMOutput` compatibility — in `_to_vlm_output`.

Then confirm `s3e/semantic_state_estimator.py` documents that `use_vllm` only affects the string path (docstring + the `_build_vlm_from_string` docstring + the inline comment at the build site).

- [ ] **Step 2: Run the full fast suite**

Run: `pytest -m "not slow" -v`
Expected: PASS — all existing tests plus the new `TestVLLMBackendMocked`, `test_vllm_backend_is_exported`, and `TestUseVllmRouting` tests. No GPU or `vllm` install required.

- [ ] **Step 3: Byte-compile the package**

Run: `python -m compileall s3e tests`
Expected: no errors.

- [ ] **Step 4: Commit any documentation fixes from Step 1**

```bash
git add s3e/vlm/vllm.py s3e/semantic_state_estimator.py
git commit -m "docs: document vLLM backend design choices inline

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

(If Step 1 found nothing to add, skip this commit.)

---

## Notes for the implementer

- **vLLM mocking gotcha:** Tests patch `s3e.vlm.vllm.LLM` and `s3e.vlm.vllm.SamplingParams` (the names bound *in this module*), never the real `vllm` package. `torch.cuda.device_count` is patched globally via `patch("torch.cuda.device_count", ...)`.
- **`SamplingParams` assertions:** because it is patched with a `MagicMock`, `SamplingParams(**kwargs)` records the call; assert on `mock_sp_cls.call_args.kwargs`.
- **`chat()` call shape:** the backend calls `self.llm.chat(conversations, sampling_params)` positionally, so conversations are at `mock_llm.chat.call_args.args[0]`.
- **Order independence:** every task repeats the exact code it touches; do not rely on having read earlier tasks.
