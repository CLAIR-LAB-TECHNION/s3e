# HuggingFace True Batching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `HuggingFaceVLM.query_batch()` run true batched HuggingFace processor/model calls while preserving model-agnostic behavior.

**Architecture:** Keep the change isolated to `s3e/vlm/huggingface.py`. Add mocked red/green tests in `tests/test_vlm_backends.py` for batched logprobs, batched generation, safe fallback, and slow integration coverage, then implement only safe batching with no persistent caches and no KV-cache reuse.

**Tech Stack:** Python 3.10+, PyTorch, HuggingFace Transformers `AutoProcessor` / VLM auto model classes, pytest, unittest.mock.

---

## File structure

- Modify `s3e/vlm/huggingface.py`
  - Responsibility: HuggingFace VLM backend implementation.
  - Add private helpers for prompt rendering, input preparation, sequential fallback, batched token-prob extraction, batched token decoding, batched generation trimming, and generated-text decoding.
  - Keep the public `query()` and `query_batch()` API unchanged.
- Modify `tests/test_vlm_backends.py`
  - Responsibility: backend unit and integration tests.
  - Add mocked tests that fail against the current sequential implementation before changing production code.
  - Tighten the existing slow HuggingFace `query_batch` integration test with bounded `num_logprobs` assertions.

## Scope guardrails

- Do not modify `s3e/semantic_state_estimator.py`.
- Do not add estimator prompt caching.
- Do not add persistent token decode caching.
- Do not add KV-cache reuse.
- Do not add model-family-specific branches.
- Preserve current inference kwargs forwarding.

---

### Task 1: Batched logprob tests and implementation

**Files:**
- Modify: `tests/test_vlm_backends.py`
- Modify: `s3e/vlm/huggingface.py`

- [ ] **Step 1: Add failing mocked tests for batched logprob behavior**

In `tests/test_vlm_backends.py`, inside `class TestHuggingFaceVLMMocked`, insert this helper immediately after the class docstring:

```python
    def _make_mock_hf_components(
        self,
        mock_model_cls,
        mock_proc_cls,
        logits=None,
        input_ids=None,
        vlm_kwargs=None,
    ):
        from s3e.vlm.huggingface import HuggingFaceVLM

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_model.device = torch.device("cpu")
        mock_model.config.is_encoder_decoder = False

        mock_processor = MagicMock()
        mock_proc_cls.from_pretrained.return_value = mock_processor

        if input_ids is None:
            batch_size = int(logits.shape[0]) if logits is not None else 1
            input_ids = torch.ones(batch_size, 5, dtype=torch.long)
        mock_processor.return_value = {"input_ids": input_ids}

        def render_chat(messages, add_generation_prompt=True, tokenize=False):
            del add_generation_prompt
            del tokenize
            user_text = messages[-1]["content"][-1]["text"]
            return f"chat:{user_text}"

        def decode_token(token_id, **kwargs):
            del kwargs
            if isinstance(token_id, torch.Tensor):
                token_id = token_id.item()
            return f"tok{int(token_id)}"

        def batch_decode_tokens(sequences, **kwargs):
            del kwargs
            decoded = []
            for sequence in sequences:
                if isinstance(sequence, torch.Tensor):
                    flat = sequence.flatten()
                    decoded.append(f"tok{int(flat[0])}" if len(flat) else "")
                else:
                    decoded.append(f"tok{int(sequence[0])}" if sequence else "")
            return decoded

        mock_processor.apply_chat_template.side_effect = render_chat
        mock_processor.decode.side_effect = decode_token
        mock_processor.batch_decode.side_effect = batch_decode_tokens

        if logits is not None:
            mock_output = MagicMock()
            mock_output.logits = logits
            mock_model.return_value = mock_output

        vlm = HuggingFaceVLM("test/model", **(vlm_kwargs or {}))
        return vlm, mock_model, mock_processor
```

Still inside `class TestHuggingFaceVLMMocked`, insert these tests after `test_query_limits_token_probs_when_num_logprobs_is_set`:

```python
    @patch("s3e.vlm.huggingface.AutoProcessor")
    @patch("s3e.vlm.huggingface._AutoModelClass")
    def test_query_batch_runs_single_forward_for_multiple_prompts(
        self, mock_model_cls, mock_proc_cls
    ):
        logits = torch.tensor(
            [
                [[0.0, 1.0, 2.0]],
                [[2.0, 1.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        vlm, mock_model, mock_processor = self._make_mock_hf_components(
            mock_model_cls, mock_proc_cls, logits=logits
        )
        img = Image.new("RGB", (64, 64))

        results = vlm.query_batch([img], ["q1", "q2"], system_prompt="sys")

        assert len(results) == 2
        assert mock_processor.call_count == 1
        assert mock_model.call_count == 1
        assert mock_processor.call_args.kwargs["text"] == ["chat:q1", "chat:q2"]
        processor_images = mock_processor.call_args.kwargs["images"]
        assert len(processor_images) == 2
        assert processor_images[0][0] is img
        assert processor_images[1][0] is img

        expected_probs = torch.softmax(logits[:, -1, :].float(), dim=-1)
        assert results[0].token_probs["tok2"] == pytest.approx(
            expected_probs[0, 2].item()
        )
        assert results[1].token_probs["tok0"] == pytest.approx(
            expected_probs[1, 0].item()
        )

    @patch("s3e.vlm.huggingface.AutoProcessor")
    @patch("s3e.vlm.huggingface._AutoModelClass")
    def test_query_batch_topk_is_computed_per_row(
        self, mock_model_cls, mock_proc_cls
    ):
        logits = torch.tensor(
            [
                [[0.0, 1.0, 2.0, 3.0]],
                [[3.0, 2.0, 1.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        vlm, _, _ = self._make_mock_hf_components(
            mock_model_cls,
            mock_proc_cls,
            logits=logits,
            vlm_kwargs={"num_logprobs": 2},
        )

        results = vlm.query_batch([], ["q1", "q2"])

        assert set(results[0].token_probs) == {"tok2", "tok3"}
        assert set(results[1].token_probs) == {"tok0", "tok1"}

    @patch("s3e.vlm.huggingface.AutoProcessor")
    @patch("s3e.vlm.huggingface._AutoModelClass")
    def test_query_batch_sums_duplicate_decoded_tokens(
        self, mock_model_cls, mock_proc_cls
    ):
        logits = torch.tensor([[[0.0, 1.0, 2.0, 3.0]]], dtype=torch.float32)
        vlm, _, mock_processor = self._make_mock_hf_components(
            mock_model_cls,
            mock_proc_cls,
            logits=logits,
            vlm_kwargs={"num_logprobs": 2},
        )
        mock_processor.decode.side_effect = None
        mock_processor.decode.return_value = "same"
        mock_processor.batch_decode.side_effect = None
        mock_processor.batch_decode.return_value = ["same", "same"]

        result = vlm.query([], "q1")

        expected_probs = torch.softmax(logits[0, -1, :].float(), dim=-1)
        assert set(result.token_probs) == {"same"}
        assert result.token_probs["same"] == pytest.approx(
            expected_probs[2].item() + expected_probs[3].item()
        )

    @patch("s3e.vlm.huggingface.AutoProcessor")
    @patch("s3e.vlm.huggingface._AutoModelClass")
    def test_query_batch_falls_back_to_sequential_when_batched_processor_rejects_images(
        self, mock_model_cls, mock_proc_cls
    ):
        logits = torch.tensor([[[0.0, 1.0]]], dtype=torch.float32)
        vlm, mock_model, mock_processor = self._make_mock_hf_components(
            mock_model_cls, mock_proc_cls, logits=logits
        )
        img = Image.new("RGB", (64, 64))

        def processor_call(*, text, images, return_tensors, padding):
            del images
            del return_tensors
            del padding
            if isinstance(text, list):
                raise ValueError("nested image batches unsupported")
            return {"input_ids": torch.ones(1, 5, dtype=torch.long)}

        mock_processor.side_effect = processor_call

        results = vlm.query_batch([img], ["q1", "q2"])

        assert len(results) == 2
        assert mock_processor.call_count == 3
        assert mock_model.call_count == 2
        assert all(isinstance(result.token_probs, dict) for result in results)

    @patch("s3e.vlm.huggingface.AutoProcessor")
    @patch("s3e.vlm.huggingface._AutoModelClass")
    def test_query_batch_empty_prompts_returns_empty_list(
        self, mock_model_cls, mock_proc_cls
    ):
        from s3e.vlm.huggingface import HuggingFaceVLM

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_processor = MagicMock()
        mock_proc_cls.from_pretrained.return_value = mock_processor

        vlm = HuggingFaceVLM("test/model")
        img = Image.new("RGB", (64, 64))

        assert vlm.query_batch([img], []) == []
        mock_processor.assert_not_called()
        mock_model.assert_not_called()
```

- [ ] **Step 2: Run the new logprob tests and verify red**

Run:

```bash
pytest tests/test_vlm_backends.py::TestHuggingFaceVLMMocked::test_query_batch_runs_single_forward_for_multiple_prompts \
       tests/test_vlm_backends.py::TestHuggingFaceVLMMocked::test_query_batch_topk_is_computed_per_row \
       tests/test_vlm_backends.py::TestHuggingFaceVLMMocked::test_query_batch_sums_duplicate_decoded_tokens \
       tests/test_vlm_backends.py::TestHuggingFaceVLMMocked::test_query_batch_falls_back_to_sequential_when_batched_processor_rejects_images \
       tests/test_vlm_backends.py::TestHuggingFaceVLMMocked::test_query_batch_empty_prompts_returns_empty_list \
       -v
```

Expected: at least `test_query_batch_runs_single_forward_for_multiple_prompts`, `test_query_batch_topk_is_computed_per_row`, and `test_query_batch_falls_back_to_sequential_when_batched_processor_rejects_images` fail against the current sequential implementation.

- [ ] **Step 3: Implement batched logprob query handling**

In `s3e/vlm/huggingface.py`, replace the existing `query_batch()` method with this method, and add the private helpers shown below it before `_build_messages()`:

```python
    def query_batch(self, images, prompts, system_prompt=None, generate=False, **inference_kwargs):
        """Send multiple queries against the same images."""
        if not prompts:
            return []

        text_inputs = [
            self._render_prompt(images, prompt, system_prompt)
            for prompt in prompts
        ]
        batched_images = self._build_batched_images(images, len(prompts))

        try:
            inputs = self._prepare_inputs(text_inputs, batched_images)
        except Exception:
            return self._query_batch_sequential(
                images, prompts, system_prompt, generate, **inference_kwargs
            )

        if generate:
            return self._query_batch_sequential(
                images, prompts, system_prompt, generate, **inference_kwargs
            )

        token_probs_batch = self._get_next_token_probs(inputs, **inference_kwargs)
        return [
            VLMOutput(token_probs=token_probs, text=None)
            for token_probs in token_probs_batch
        ]

    def _query_batch_sequential(
        self,
        images,
        prompts,
        system_prompt=None,
        generate=False,
        **inference_kwargs,
    ):
        """Correctness fallback for processors that cannot batch inputs."""
        results = []
        for prompt in prompts:
            text_input = self._render_prompt(images, prompt, system_prompt)
            inputs = self._prepare_inputs(text_input, images if images else None)

            if generate:
                generated_text = self._generate_text(inputs, **inference_kwargs)
                token_probs = None
            else:
                token_probs = self._get_next_token_probs(
                    inputs, **inference_kwargs
                )[0]
                generated_text = None

            results.append(VLMOutput(token_probs=token_probs, text=generated_text))
        return results

    def _render_prompt(self, images, prompt, system_prompt=None):
        """Render a prompt with the processor chat template when available."""
        full_prompt = prompt
        if system_prompt is not None:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        messages = self._build_messages(images, prompt, system_prompt)
        try:
            return self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        except Exception:
            return full_prompt

    @staticmethod
    def _build_batched_images(images, batch_size: int):
        """Repeat the same image set for every prompt row."""
        if not images:
            return None
        return [list(images) for _ in range(batch_size)]

    def _prepare_inputs(self, text, images):
        """Run the processor and move tensor-like inputs to the model device."""
        inputs = self.processor(
            text=text,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        return self._move_inputs_to_device(inputs)

    def _move_inputs_to_device(self, inputs):
        """Move tensor-like processor outputs while preserving metadata."""
        return {
            key: value.to(self.model.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
```

Still in `s3e/vlm/huggingface.py`, replace the existing `_get_next_token_probs()` method with this batch-aware version, and add `_decode_token_ids()` immediately after it:

```python
    def _get_next_token_probs(self, inputs, **inference_kwargs):
        with torch.no_grad():
            outputs = self.model(**inputs, **inference_kwargs)

        logits = outputs.logits[:, -1, :].float()
        probs = torch.softmax(logits, dim=-1)

        if self.num_logprobs is None:
            selected_probs = probs
            selected_indices = torch.arange(
                probs.shape[-1], device=probs.device
            ).expand(probs.shape[0], -1)
        else:
            selected_probs, selected_indices = torch.topk(
                probs, min(self.num_logprobs, probs.shape[-1]), dim=-1
            )

        selected_probs = selected_probs.detach().cpu().tolist()
        selected_indices = selected_indices.detach().cpu().tolist()
        flat_indices = [int(idx) for row in selected_indices for idx in row]
        decoded_tokens = self._decode_token_ids(flat_indices)

        results = []
        offset = 0
        for row_probs, row_indices in zip(selected_probs, selected_indices):
            row_tokens = decoded_tokens[offset : offset + len(row_indices)]
            offset += len(row_indices)

            token_probs = {}
            for token_str, prob in zip(row_tokens, row_probs):
                token_probs[token_str] = token_probs.get(token_str, 0.0) + float(prob)
            results.append(token_probs)

        return results

    def _decode_token_ids(self, token_ids: list[int]) -> list[str]:
        """Decode token IDs, preferring batch decoding when it behaves correctly."""
        if not token_ids:
            return []

        batch_decode = getattr(self.processor, "batch_decode", None)
        if callable(batch_decode):
            try:
                decoded = batch_decode(
                    [[int(token_id)] for token_id in token_ids],
                    skip_special_tokens=False,
                )
                if len(decoded) == len(token_ids):
                    return list(decoded)
            except Exception:
                pass

        return [self.processor.decode(int(token_id)) for token_id in token_ids]
```

- [ ] **Step 4: Run the logprob tests and verify green**

Run:

```bash
pytest tests/test_vlm_backends.py::TestHuggingFaceVLMMocked::test_query_batch_runs_single_forward_for_multiple_prompts \
       tests/test_vlm_backends.py::TestHuggingFaceVLMMocked::test_query_batch_topk_is_computed_per_row \
       tests/test_vlm_backends.py::TestHuggingFaceVLMMocked::test_query_batch_sums_duplicate_decoded_tokens \
       tests/test_vlm_backends.py::TestHuggingFaceVLMMocked::test_query_batch_falls_back_to_sequential_when_batched_processor_rejects_images \
       tests/test_vlm_backends.py::TestHuggingFaceVLMMocked::test_query_batch_empty_prompts_returns_empty_list \
       -v
```

Expected: all selected tests pass.

- [ ] **Step 5: Run all mocked HuggingFace VLM tests**

Run:

```bash
pytest tests/test_vlm_backends.py::TestHuggingFaceVLMMocked -v
```

Expected: existing generation tests still pass because generation remains on the sequential fallback until Task 2.

- [ ] **Step 6: Commit Task 1**

Run:

```bash
git add s3e/vlm/huggingface.py tests/test_vlm_backends.py
git commit -m "feat: batch huggingface logprob queries"
```

---

### Task 2: Batched generation tests and implementation

**Files:**
- Modify: `tests/test_vlm_backends.py`
- Modify: `s3e/vlm/huggingface.py`

- [ ] **Step 1: Add failing mocked tests for batched generation**

In `tests/test_vlm_backends.py`, inside `class TestHuggingFaceVLMMocked`, insert these tests after `test_query_generate_mode_sets_safe_defaults`:

```python
    @patch("s3e.vlm.huggingface.AutoProcessor")
    @patch("s3e.vlm.huggingface._AutoModelClass")
    def test_query_batch_generate_calls_generate_once_and_batch_decodes(
        self, mock_model_cls, mock_proc_cls
    ):
        input_ids = torch.ones(2, 3, dtype=torch.long)
        vlm, mock_model, mock_processor = self._make_mock_hf_components(
            mock_model_cls, mock_proc_cls, input_ids=input_ids
        )
        mock_model.generate.return_value = torch.tensor(
            [
                [1, 2, 3, 10, 11],
                [4, 5, 6, 20, 21],
            ],
            dtype=torch.long,
        )
        mock_processor.batch_decode.side_effect = None
        mock_processor.batch_decode.return_value = ["yes", "no"]

        results = vlm.query_batch(
            [], ["q1", "q2"], generate=True, max_new_tokens=2
        )

        assert mock_model.generate.call_count == 1
        assert mock_model.generate.call_args.kwargs["max_new_tokens"] == 2
        decoded_sequences = mock_processor.batch_decode.call_args.args[0]
        assert [seq.tolist() for seq in decoded_sequences] == [[10, 11], [20, 21]]
        assert [result.text for result in results] == ["yes", "no"]
        assert all(result.token_probs is None for result in results)

    @patch("s3e.vlm.huggingface.AutoProcessor")
    @patch("s3e.vlm.huggingface._AutoModelClass")
    def test_query_batch_generate_does_not_trim_encoder_decoder_outputs(
        self, mock_model_cls, mock_proc_cls
    ):
        input_ids = torch.ones(2, 3, dtype=torch.long)
        vlm, mock_model, mock_processor = self._make_mock_hf_components(
            mock_model_cls, mock_proc_cls, input_ids=input_ids
        )
        mock_model.config.is_encoder_decoder = True
        mock_model.generate.return_value = torch.tensor(
            [
                [10, 11],
                [20, 21],
            ],
            dtype=torch.long,
        )
        mock_processor.batch_decode.side_effect = None
        mock_processor.batch_decode.return_value = ["yes", "no"]

        results = vlm.query_batch([], ["q1", "q2"], generate=True)

        decoded_sequences = mock_processor.batch_decode.call_args.args[0]
        assert [seq.tolist() for seq in decoded_sequences] == [[10, 11], [20, 21]]
        assert [result.text for result in results] == ["yes", "no"]

    @patch("s3e.vlm.huggingface.AutoProcessor")
    @patch("s3e.vlm.huggingface._AutoModelClass")
    def test_query_batch_generate_falls_back_to_decode_when_batch_decode_fails(
        self, mock_model_cls, mock_proc_cls
    ):
        input_ids = torch.ones(2, 3, dtype=torch.long)
        vlm, mock_model, mock_processor = self._make_mock_hf_components(
            mock_model_cls, mock_proc_cls, input_ids=input_ids
        )
        mock_model.generate.return_value = torch.tensor(
            [
                [1, 2, 3, 10, 11],
                [4, 5, 6, 20, 21],
            ],
            dtype=torch.long,
        )
        mock_processor.batch_decode.side_effect = TypeError("no batch decode")
        mock_processor.decode.side_effect = (
            lambda sequence, **kwargs: f"decoded:{sequence.tolist()}"
        )

        results = vlm.query_batch([], ["q1", "q2"], generate=True)

        assert [result.text for result in results] == [
            "decoded:[10, 11]",
            "decoded:[20, 21]",
        ]
```

- [ ] **Step 2: Run the new generation tests and verify red**

Run:

```bash
pytest tests/test_vlm_backends.py::TestHuggingFaceVLMMocked::test_query_batch_generate_calls_generate_once_and_batch_decodes \
       tests/test_vlm_backends.py::TestHuggingFaceVLMMocked::test_query_batch_generate_does_not_trim_encoder_decoder_outputs \
       tests/test_vlm_backends.py::TestHuggingFaceVLMMocked::test_query_batch_generate_falls_back_to_decode_when_batch_decode_fails \
       -v
```

Expected: at least `test_query_batch_generate_calls_generate_once_and_batch_decodes` fails because generation still uses the sequential fallback from Task 1.

- [ ] **Step 3: Implement batched generation**

In `s3e/vlm/huggingface.py`, update the `if generate:` branch inside `query_batch()` to this code:

```python
        if generate:
            generated_texts = self._generate_text(inputs, **inference_kwargs)
            return [
                VLMOutput(token_probs=None, text=generated_text)
                for generated_text in generated_texts
            ]
```

In `_query_batch_sequential()`, update the generation branch to index the single-item list returned by `_generate_text()`:

```python
            if generate:
                generated_text = self._generate_text(inputs, **inference_kwargs)[0]
                token_probs = None
```

Replace the existing `_generate_text()` method with this batch-aware method, and add the helper methods immediately before or after it:

```python
    def _generate_text(self, inputs, **inference_kwargs) -> list[str | None]:
        """Generate text responses for a batch of prompts."""
        batch_size = self._infer_batch_size(inputs)
        try:
            output_ids = self.model.generate(**inputs, **inference_kwargs)
            generated_sequences = self._trim_generated_sequences(output_ids, inputs)
            return self._decode_generated_sequences(generated_sequences)
        except Exception:
            return [None for _ in range(batch_size)]

    @staticmethod
    def _infer_batch_size(inputs) -> int:
        input_ids = inputs.get("input_ids")
        if input_ids is not None and hasattr(input_ids, "shape") and input_ids.shape:
            return int(input_ids.shape[0])
        return 1

    def _trim_generated_sequences(self, output_ids, inputs):
        """Remove prompt tokens from decoder-only generation outputs."""
        if self._model_is_encoder_decoder():
            return [row for row in output_ids]
        input_len = inputs["input_ids"].shape[-1]
        return [row[input_len:] for row in output_ids]

    def _model_is_encoder_decoder(self) -> bool:
        config = getattr(self.model, "config", None)
        return getattr(config, "is_encoder_decoder", False) is True

    def _decode_generated_sequences(self, sequences) -> list[str]:
        """Decode generated token sequences, preferring batch decoding."""
        batch_decode = getattr(self.processor, "batch_decode", None)
        if callable(batch_decode):
            try:
                decoded = batch_decode(sequences, skip_special_tokens=True)
                if len(decoded) == len(sequences):
                    return list(decoded)
            except Exception:
                pass

        return [
            self.processor.decode(sequence, skip_special_tokens=True)
            for sequence in sequences
        ]
```

- [ ] **Step 4: Run the generation tests and verify green**

Run:

```bash
pytest tests/test_vlm_backends.py::TestHuggingFaceVLMMocked::test_query_batch_generate_calls_generate_once_and_batch_decodes \
       tests/test_vlm_backends.py::TestHuggingFaceVLMMocked::test_query_batch_generate_does_not_trim_encoder_decoder_outputs \
       tests/test_vlm_backends.py::TestHuggingFaceVLMMocked::test_query_batch_generate_falls_back_to_decode_when_batch_decode_fails \
       -v
```

Expected: all selected tests pass.

- [ ] **Step 5: Run all mocked HuggingFace VLM tests**

Run:

```bash
pytest tests/test_vlm_backends.py::TestHuggingFaceVLMMocked -v
```

Expected: all mocked HuggingFace backend tests pass, including existing `query()` tests.

- [ ] **Step 6: Commit Task 2**

Run:

```bash
git add s3e/vlm/huggingface.py tests/test_vlm_backends.py
git commit -m "feat: batch huggingface generation"
```

---

### Task 3: Tighten slow HuggingFace batch integration coverage

**Files:**
- Modify: `tests/test_vlm_backends.py`

- [ ] **Step 1: Update the slow `test_query_batch` integration test**

In `tests/test_vlm_backends.py`, replace `TestHuggingFaceVLMIntegration.test_query_batch` with this version:

```python
    def test_query_batch(self):
        from s3e.vlm.huggingface import HuggingFaceVLM

        vlm = HuggingFaceVLM(self.TINY_VLM_ID, device_map="cpu", num_logprobs=2)
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        results = vlm.query_batch([img], ["q1?", "q2?"])

        assert len(results) == 2
        assert all(isinstance(r, VLMOutput) for r in results)
        assert all(isinstance(r.token_probs, dict) for r in results)
        assert all(0 < len(r.token_probs) <= 2 for r in results)
        assert all(
            all(prob >= 0 for prob in r.token_probs.values())
            for r in results
        )
```

- [ ] **Step 2: Run non-slow backend tests**

Run:

```bash
pytest -m "not slow" tests/test_vlm_backends.py -v
```

Expected: all non-slow backend tests pass. The updated slow integration test is deselected by the marker expression.

- [ ] **Step 3: Run the updated slow integration test when the tiny model can be downloaded or is cached**

Run:

```bash
pytest -m slow tests/test_vlm_backends.py::TestHuggingFaceVLMIntegration::test_query_batch -v
```

Expected when HuggingFace model access is available: the test passes. If the environment cannot access or cache `katuni4ka/tiny-random-llava`, record the download/cache error in the final implementation notes and continue with the mandatory non-slow verification.

- [ ] **Step 4: Commit Task 3**

Run:

```bash
git add tests/test_vlm_backends.py
git commit -m "test: tighten huggingface batch integration"
```

---

### Task 4: Final verification

**Files:**
- Verify: `s3e/vlm/huggingface.py`
- Verify: `tests/test_vlm_backends.py`

- [ ] **Step 1: Run the full mocked HuggingFace backend suite**

Run:

```bash
pytest tests/test_vlm_backends.py::TestHuggingFaceVLMMocked -v
```

Expected: all tests in `TestHuggingFaceVLMMocked` pass.

- [ ] **Step 2: Run the default non-slow repository test suite**

Run:

```bash
pytest -m "not slow"
```

Expected: all non-slow tests pass.

- [ ] **Step 3: Run Python compilation check**

Run:

```bash
python -m compileall s3e tests
```

Expected: command exits successfully without syntax errors.

- [ ] **Step 4: Inspect git history and working tree**

Run:

```bash
git log --oneline -4
git status --short
```

Expected: the three implementation commits from Tasks 1-3 are visible near the top of the log, and `git status --short` prints no tracked-file modifications. If slow-test execution produced cache files, remove untracked cache artifacts before handing off.
