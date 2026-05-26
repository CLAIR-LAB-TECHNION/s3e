# vLLM VLM Backend — Design

- **Date:** 2026-05-26
- **Status:** Approved design, pending implementation plan
- **Scope:** Add a vLLM-powered VLM backend to `s3e`, activated by a `use_vllm`
  flag when using a HuggingFace model. Provides transparent, multi-GPU,
  high-throughput local inference while preserving the existing `VLMBackend`
  contract.

## Goal

Let a user run any local HuggingFace vision-language model through
[vLLM](https://docs.vllm.ai/en/stable/) instead of plain `transformers`, by
setting a single flag. The user treats a single multi-GPU machine as "one big
GPU": parallelism and batching are handled under the hood. By default the
backend uses **all local GPUs**; the user may override this and any other vLLM
engine setting.

The backend must be a drop-in `VLMBackend`: it returns the same `VLMOutput`
shape as the existing backends, so the estimator, probability extraction, and
Platt-scaling calibration all work without modification.

## Non-goals

- Multi-node / distributed (Ray) inference. Targeting a single node with one or
  more local GPUs. Advanced users can still pass through vLLM engine kwargs
  (e.g. `pipeline_parallel_size`, `distributed_executor_backend`) but the
  backend will not hard-code or default to a multi-node setup.
- A no-chat-template "raw prompt" escape hatch. The backend relies on the
  model's chat template (see Inference Path). This can be added later if a
  target model lacks a chat template (YAGNI for now).
- Changing the `VLMBackend` interface or `VLMOutput` dataclass.

## Background: how the estimator uses a backend

`SemanticStateEstimator.estimate_raw` chunks grounded-predicate prompts into
`batch_size` groups and calls `vlm.query_batch(images, batch_prompts,
system_prompt=..., generate=self._generate_mode, **inference_kwargs)`. Key
facts that shape this design:

- The **same images** are sent with **many prompts** (one per grounded
  predicate).
- Two modes:
  - **logprobs mode** (`generate=False`): the estimator reads
    `VLMOutput.token_probs` — a `dict[str, float]` over the next answer token —
    and sums probability mass across the true / false / null token groups.
  - **text_match mode** (`generate=True`): the estimator reads
    `VLMOutput.text`.
- Platt-scaling calibration consumes `token_probs` via `grouped_log_odds`, so
  the `token_probs` format must match the other backends exactly.

## Architecture & file layout

- New class `VLLMBackend(VLMBackend)` in **`s3e/vlm/vllm.py`**, parallel to
  `huggingface.py` and `openai.py`.
- Module name `vllm.py` mirrors the existing convention (`openai.py` does
  `import openai`; `huggingface.py` uses `transformers`). Python 3 absolute
  imports mean `from vllm import LLM, SamplingParams` inside
  `s3e/vlm/vllm.py` resolves to the installed package, not the module itself —
  the same pattern `openai.py` already uses safely.
- Export `VLLMBackend` from `s3e/vlm/__init__.py` and add it to `__all__`.
- Add `vllm = ["vllm"]` to `[project.optional-dependencies]` in
  `pyproject.toml`.
- Lazy import with an `ImportError` pointing to `pip install s3e[vllm]`,
  mirroring the existing `openai` guard:

  ```python
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
  ```

## Public API & activation

`SemanticStateEstimator.__init__` gains `use_vllm: bool = False`. Routing
happens in `_build_vlm_from_string`, which now takes `use_vllm`:

```python
@staticmethod
def _build_vlm_from_string(vlm_id, vlm_kwargs, use_vllm=False):
    if vlm_id.startswith(OPENAI_MODEL_IDENTIFIER):
        if use_vllm:
            raise ValueError("use_vllm=True is not compatible with OpenAI/ models.")
        from .vlm.openai import OpenAIVLM
        return OpenAIVLM(vlm_id, **vlm_kwargs)
    if use_vllm:
        from .vlm.vllm import VLLMBackend
        return VLLMBackend(vlm_id, **vlm_kwargs)
    from .vlm.huggingface import HuggingFaceVLM
    return HuggingFaceVLM(vlm_id, **vlm_kwargs)
```

- `use_vllm` affects **only the string path**. If a user passes an
  already-constructed `VLMBackend` instance as `vlm`, `use_vllm` is ignored
  (documented in the estimator docstring).
- The estimator stores `use_vllm` and forwards it when building from a string.

`VLLMBackend.__init__` signature:

```python
VLLMBackend(
    model_id: str,
    tensor_parallel_size: int | None = None,  # None -> all local GPUs
    num_logprobs: int | None = None,           # None -> full vocab (mirrors HF)
    **engine_kwargs,                           # forwarded to vllm.LLM(...)
)
```

- `tensor_parallel_size=None` resolves to `max(torch.cuda.device_count(), 1)`.
  An explicit value is respected.
- `**engine_kwargs` is forwarded to the `vllm.LLM` constructor, exposing
  `gpu_memory_utilization`, `dtype`, `max_model_len`, `quantization`, etc.,
  mirroring how `HuggingFaceVLM` forwards `**model_kwargs`.
- No `max_new_tokens` parameter. (`HuggingFaceVLM` stores `max_new_tokens` but
  never uses it — its generate path forwards only `inference_kwargs`. We do not
  reproduce that dead state.) Generation length is controlled through
  `inference_kwargs` → `SamplingParams.max_tokens`.

## Engine construction & logprobs configuration

In `__init__`, build one persistent engine:

```python
self.llm = LLM(
    model=model_id,
    tensor_parallel_size=resolved_tp,
    max_logprobs=(-1 if num_logprobs is None else num_logprobs),
    **engine_kwargs,
)
```

vLLM logprobs semantics (verified against the docs):

- `SamplingParams.logprobs`: integer `k` → top-`k` (plus the sampled token);
  `-1` → **all vocab logprobs**; `None` → none.
- Engine `max_logprobs` (constructor arg): default `20`; `-1` → no cap
  (full vocab, with an OOM caveat for large `output_length * vocab_size`).

Because both `num_logprobs` and `max_logprobs` are fixed at construction, they
are always consistent:

- `num_logprobs=None` → `max_logprobs=-1`, sampling `logprobs=-1` (full vocab —
  the same default and semantics as `HuggingFaceVLM`). Since logprobs mode uses
  `max_tokens=1`, the cost is one position's distribution, comparable to the HF
  backend's single-position extraction. The full-vocab OOM caveat is noted in
  the docstring.
- finite `k` → `max_logprobs=k`, sampling `logprobs=k`.

## Inference path

Use vLLM's offline **`LLM.chat()`** with one conversation per prompt, all
sharing the same `image_pil` content. This delegates chat-template rendering
and (resolution-dependent) image-placeholder expansion to vLLM/the model's own
processor — mirroring how `HuggingFaceVLM._render_prompt` relies on
`processor.apply_chat_template`, and owning zero per-model placeholder logic.

```python
def query(self, images, prompt, system_prompt=None, generate=False, **inference_kwargs):
    return self.query_batch(images, [prompt], system_prompt, generate, **inference_kwargs)[0]

def query_batch(self, images, prompts, system_prompt=None, generate=False, **inference_kwargs):
    if not prompts:
        return []
    sampling_params = self._build_sampling_params(generate, **inference_kwargs)
    image_content = [{"type": "image_pil", "image_pil": img} for img in images]
    conversations = []
    for prompt in prompts:
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {"role": "user", "content": image_content + [{"type": "text", "text": prompt}]}
        )
        conversations.append(messages)
    outputs = self.llm.chat(conversations, sampling_params)  # order preserved
    return [self._to_vlm_output(output, generate) for output in outputs]
```

`llm.chat()` accepts a batched list of conversations and a `SamplingParams`,
and returns outputs in input order.

### SamplingParams per mode

`inference_kwargs` are merged as overrides, mirroring how
`OpenAIVLM._set_inference_kwargs_defaults` forces logprobs and `HuggingFaceVLM`
forces `logits_to_keep=1`.

- **logprobs mode** (`generate=False`):
  - `inference_kwargs.setdefault("max_tokens", 1)` — one decode step. This is
    the memory-minimal default and the direct analog of HF's
    `logits_to_keep=1`. It is `setdefault`, not forced, so a user may override
    it.
  - force `logprobs = (-1 if num_logprobs is None else num_logprobs)` (the
    analog of OpenAI forcing `logprobs=True`).
  - `inference_kwargs.setdefault("temperature", 0.0)` for deterministic output.
- **text_match mode** (`generate=True`):
  - `inference_kwargs.setdefault("temperature", 0.0)`.
  - Do **not** force or default `max_tokens` — let the model generate (and
    reason) as it sees fit. This matches the OpenAI backend's documented
    philosophy of not bounding completion length so reasoning can proceed.

Free-form reasoning therefore lives in text mode; logprobs mode stays lean and
reproduces the HF backend's next-token distribution.

### Output extraction (`_to_vlm_output`)

- **logprobs mode**: read `output.outputs[0].logprobs[0]` (a
  `dict[token_id -> Logprob]` for the next answer token). Build
  `token_probs = {logprob.decoded_token: exp(logprob.logprob)}`, **summing
  duplicate decoded tokens** (the same dedup `HuggingFaceVLM` and `OpenAIVLM`
  perform). `text=None`.
- **text_match mode**: `token_probs=None`,
  `text = output.outputs[0].text`.

This keeps `VLMOutput` byte-for-byte compatible with the other backends, so
Platt-scaling calibration works unchanged.

## Error handling

- Missing `vllm` package → `ImportError` with `pip install s3e[vllm]`
  guidance, raised lazily at construction.
- `use_vllm=True` with an `OpenAI/` model → `ValueError`.
- Model without a chat template → surface vLLM's error clearly. No silent
  fallback; documented as a known limitation of the chat-based path.
- **Missing logprobs is an error, not a swallow.** With `logprobs` forced and
  `max_tokens >= 1`, an absent or `None` `outputs[0].logprobs[0]` is an
  internal-inconsistency edge case. Raise an informative exception (e.g.
  "vLLM returned no logprobs for a request despite logprobs being requested")
  rather than returning empty `token_probs`. This follows AGENTS.md's "raise
  explicit, informative exceptions" and "avoid broad exception handling."

## Testing

Implementation proceeds via **red/green TDD** throughout: write a failing test,
watch it fail (red), implement the minimal code to pass (green), then refactor.
Each behavior below is driven by a test written before its implementation.

Following the existing split in `tests/test_vlm_backends.py`:

- **Mocked unit tests** (patch `s3e.vlm.vllm.LLM` and `s3e.vlm.vllm.SamplingParams`,
  like `TestHuggingFaceVLMMocked` / `TestOpenAIVLM`):
  - Construction: `tensor_parallel_size=None` resolves to a mocked
    `torch.cuda.device_count()`; an explicit value is respected.
  - `max_logprobs`/`logprobs` wiring: `num_logprobs=None` → `max_logprobs=-1`
    and sampling `logprobs=-1`; finite `k` → both `k`.
  - One `chat()` call per `query_batch` with the correct conversation
    structure (optional system message, `image_pil` content per image, text).
  - logprobs extraction: `exp(logprob)` and duplicate-decoded-token summing.
  - text-mode extraction reads `outputs[0].text` and sets `token_probs=None`.
  - logprobs mode sets `max_tokens=1` by default but honors an override;
    text mode does not force `max_tokens`.
  - empty prompts → `[]` with no engine call.
  - missing/`None` logprobs raises an informative error.
  - `ImportError` guidance when `vllm` is absent.
- **Estimator routing tests** (in `tests/test_semantic_state_estimator.py`):
  - `use_vllm=True` routes a string model to `VLLMBackend`.
  - `use_vllm=True` with an `OpenAI/` model → `ValueError`.
  - `use_vllm` is ignored when an explicit backend instance is passed.
- **Slow integration test** marked `@pytest.mark.slow` and guarded by
  `pytest.mark.skipif(not torch.cuda.is_available())`. vLLM requires CUDA, so
  this is skipped on a non-GPU host but runs on a GPU machine.

This honors AGENTS.md: `pytest -m "not slow"` stays the fast verification path
and runs fully mocked, with no GPU or `vllm` install required.

## Documentation requirements

**Every design choice in this spec must be documented directly in the code, in
detail, so the next maintainer can make informed decisions about future design
changes.** The spec lives outside the source tree; the code is what gets read
when someone modifies the backend. Inline comments and docstrings must capture
not just *what* the code does but *why* — the rationale and the alternative
that was rejected — for each non-obvious choice. Concretely, document:

- **Why `LLM.chat()` over `LLM.generate()` + `multi_modal_data`** — chat
  delegates chat-template rendering and resolution-dependent image-placeholder
  expansion to the model's own processor, which VLMs almost always ship; the
  `generate()` path would force us to own per-model placeholder logic and risks
  token/feature-count mismatches. Note the accepted limitation: a model without
  a chat template is unsupported on this path.
- **Why `max_tokens=1` is a `setdefault` (not forced) in logprobs mode** — it
  is the memory-minimal default and the direct analog of `HuggingFaceVLM`'s
  `logits_to_keep=1`, while remaining overridable. Note that free-form
  reasoning is intended for text mode, which deliberately does not bound
  `max_tokens` (matching the OpenAI backend's documented philosophy).
- **Why `logprobs` is forced** in logprobs mode (the analog of the OpenAI
  backend forcing `logprobs=True`).
- **The `num_logprobs` → `max_logprobs` mapping** (`None` → `-1` = full vocab),
  why both must be set at construction to stay consistent, and the full-vocab
  OOM caveat.
- **Why `tensor_parallel_size` defaults to the local GPU count** and that
  multi-node is intentionally out of scope but reachable via `**engine_kwargs`.
- **Why there is no `max_new_tokens` parameter** — it is dead state in
  `HuggingFaceVLM` (stored, never forwarded), so it is not reproduced here.
- **Why missing logprobs raises** rather than returning empty `token_probs` —
  it is an internal-inconsistency edge case, not a normal path.
- **The `import vllm` / module-name `vllm.py` safety note** (absolute imports;
  mirrors `openai.py`).
- **Token-probability dedup** (summing duplicate decoded tokens) matching the
  other backends, and why `VLMOutput` must stay byte-for-byte compatible (Platt
  calibration consumes `token_probs`).

Additional touch-points:

- `VLLMBackend` class + method docstrings (args, the full-vocab OOM caveat, the
  chat-template requirement).
- Update `SemanticStateEstimator` docstring for the new `use_vllm` parameter,
  including that it only affects the string path.
- `pyproject.toml` optional-dependency entry.

## Open questions

None. Design approved in brainstorming on 2026-05-26, including: logprobs-mode
behavior confirmed as specified; `LLM.chat()` accepted as the right convenience
trade-off given VLMs' reliance on built-in chat templates for image handling.
