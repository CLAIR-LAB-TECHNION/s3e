# HuggingFace True Batching Design

## Context

`HuggingFaceVLM.query_batch()` currently loops over prompts and performs one processor call plus one model call per prompt. `SemanticStateEstimator.estimate_raw()` already groups predicate prompts into batches and passes each batch to `query_batch()`, so the backend should use those batches directly.

## Goals

- Make `HuggingFaceVLM.query_batch()` perform true batched HuggingFace inference.
- Keep the change model-agnostic and safe for supported HuggingFace VLMs.
- Preserve the public `VLMBackend` API and result ordering.
- Support both logprob mode (`generate=False`) and text generation mode (`generate=True`).
- Keep `query()` behavior unchanged by routing it through the batched implementation with one prompt.

## Non-goals

- No `SemanticStateEstimator` prompt cache or precomputation.
- No persistent token decode cache.
- No KV-cache reuse, including system-prompt KV caching.
- No model-family-specific paths or assumptions.
- No assumptions about where `{query}` appears in `user_prompt_template`.

## Architecture

The change is limited to `s3e/vlm/huggingface.py`.

`query_batch()` will:

1. Build a message list for each prompt using the existing `_build_messages()` structure.
2. Render each message list with `processor.apply_chat_template(..., tokenize=False)`.
3. Fall back per prompt to `system_prompt + "\n\n" + prompt` when chat-template rendering fails, matching current behavior.
4. Call the processor once with a list of rendered prompt strings.
5. Pass the same image set to every batch row by using `[images for _ in prompts]`; pass `None` when there are no images.
6. If a processor cannot handle the batched multimodal input format, fall back to the current per-prompt behavior for correctness rather than using model-specific workarounds.
7. Move tensor-like processor outputs to `self.model.device` while preserving non-tensor metadata.
8. Run one batched model forward call for logprob mode, or one batched `model.generate()` call for generation mode.
9. Return one `VLMOutput` per input prompt in the original order.

## Batched logprob extraction

For `generate=False`, `_get_next_token_probs()` will become batch-aware and return `list[dict[str, float]]`.

The method will:

- Run `self.model(**inputs, **inference_kwargs)` once under `torch.no_grad()`.
- Use `outputs.logits[:, -1, :]` for next-token logits for every row.
- Apply `torch.softmax(..., dim=-1)`.
- If `num_logprobs is None`, include the full vocabulary for each row.
- If `num_logprobs` is set, use `torch.topk(..., dim=-1)` for each row.
- Decode selected token IDs with batch-aware decoding when possible, then regroup decoded tokens by row.
- Sum probability mass when multiple selected token IDs decode to the same token string.

## Batched generation

For `generate=True`, `_generate_text()` will become batch-aware and return `list[str | None]`.

The method will:

- Call `self.model.generate(**inputs, **inference_kwargs)` once.
- For decoder-only models, trim the padded input length (`inputs["input_ids"].shape[-1]`) from each generated row, because HuggingFace `generate()` returns outputs aligned to the padded input tensor.
- For encoder-decoder models that return decoder outputs without the input prompt, skip input trimming.
- Avoid using `attention_mask.sum(dim=-1)` for trimming because left padding would make that index point into the prompt rather than the generated suffix.
- Decode generated suffixes with `processor.batch_decode(..., skip_special_tokens=True)` when available.
- Fall back to per-row `processor.decode(..., skip_special_tokens=True)` when batch decoding is unavailable.
- Preserve current failure behavior by returning `None` for each prompt if generation raises an exception.

## Error handling and compatibility

- Empty `prompts` should return an empty list without calling the processor or model.
- Single-prompt behavior should match current `query()` results.
- Existing inference kwargs forwarding should be preserved.
- Batched preprocessing fallback should preserve correctness for processors that do not support nested batched image inputs.
- Processor outputs may contain tensors and non-tensor metadata; only objects with `.to(...)` should be moved to the model device.
- The implementation should not rely on `past_key_values`, `cache_position`, image-grid internals, or any model-specific processor fields.

## Testing strategy

Development must follow red/green testing:

1. Add failing tests first that demonstrate current `query_batch()` is not truly batched.
2. Run the narrowest relevant tests and confirm the new tests fail for the expected reason.
3. Implement the backend batching change.
4. Re-run the narrow tests and confirm they pass.
5. Broaden to the non-slow test suite.

Mocked unit tests should cover:

- `query_batch()` calls the processor and model once for multiple prompts in logprob mode.
- Output count and order match input prompts.
- Batched logits produce different token-prob dictionaries per row.
- `num_logprobs` works independently for each row.
- Generate mode calls `model.generate()` once.
- Decoder-only generated outputs are trimmed after the padded input length, while encoder-decoder outputs are not incorrectly trimmed.
- `processor.batch_decode()` is preferred when available, with a per-row decode fallback.
- Batched preprocessing fallback preserves correctness when a mocked processor rejects nested batched image inputs.
- `query()` still returns one `VLMOutput` through the batched path.
- Empty prompt lists return `[]` without model work.

Slow integration tests may be added or adjusted if mocked tests are insufficient to validate real processor batching behavior. Any slow tests should use the existing `@pytest.mark.slow` convention and a tiny HuggingFace model, and should remain separate from the default `pytest -m "not slow"` verification path.

Recommended verification commands:

```bash
pytest tests/test_vlm_backends.py::TestHuggingFaceVLMMocked
pytest -m "not slow"
```

If slow integration coverage changes, also run:

```bash
pytest -m slow tests/test_vlm_backends.py::TestHuggingFaceVLMIntegration
```
