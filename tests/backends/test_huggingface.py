# SPDX-FileCopyrightText: CLAIR Lab Technion
# SPDX-License-Identifier: MIT

"""Tests for the HuggingFace VLM backend: mocked units, slow
integration against tiny real models, and the shared backend contract.
"""

import math
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

torch = pytest.importorskip("torch", reason="torch not installed (s3e[hf])")
pytest.importorskip("transformers", reason="transformers not installed (s3e[hf])")

from s3e.backends.backend import VLMOutput

from backends.test_contract import BackendContract


class TestHuggingFaceVLMMocked:
    """Unit tests for HuggingFaceVLM using mocked transformers."""

    def _make_mock_hf_components(
        self,
        mock_model_cls,
        mock_proc_cls,
        logits=None,
        input_ids=None,
        vlm_kwargs=None,
    ):
        from s3e.backends.huggingface import HuggingFaceVLM

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
            # Mirror the real transformers contract: logits_to_keep=n returns
            # only the final n sequence positions. A mock that ignores it would
            # hand back full-sequence logits the model never actually produces.
            def forward(*args, logits_to_keep=0, **kwargs):
                del args, kwargs
                sliced = (
                    logits
                    if not isinstance(logits_to_keep, int) or logits_to_keep == 0
                    else logits[:, -logits_to_keep:, :]
                )
                mock_output = MagicMock()
                mock_output.logits = sliced
                return mock_output

            mock_model.side_effect = forward

        vlm = HuggingFaceVLM("test/model", **(vlm_kwargs or {}))
        return vlm, mock_model, mock_processor

    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
    def test_construction(self, mock_model_cls, mock_proc_cls):
        from s3e.backends.huggingface import HuggingFaceVLM

        vlm = HuggingFaceVLM("test/model")
        mock_model_cls.from_pretrained.assert_called_once()
        mock_proc_cls.from_pretrained.assert_called_once()
        assert vlm.num_logprobs is None

    @pytest.mark.parametrize("num_logprobs", [0, -1, 1.5, True])
    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
    def test_invalid_num_logprobs_rejected_before_model_load(
        self, mock_model_cls, mock_proc_cls, num_logprobs
    ):
        from s3e.backends.huggingface import HuggingFaceVLM

        with pytest.raises(ValueError, match="num_logprobs"):
            HuggingFaceVLM("test/model", num_logprobs=num_logprobs)

        mock_model_cls.from_pretrained.assert_not_called()
        mock_proc_cls.from_pretrained.assert_not_called()

    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
    def test_no_max_new_tokens_constructor_state(self, mock_model_cls, mock_proc_cls):
        """Generation length is a per-call inference kwarg, never constructor
        state (mirroring VLLMBackend); the old ``max_new_tokens`` attribute
        was stored but never forwarded to generate()."""
        from s3e.backends.huggingface import HuggingFaceVLM

        mock_model_cls.from_pretrained.return_value = MagicMock()
        mock_proc_cls.from_pretrained.return_value = MagicMock()

        vlm = HuggingFaceVLM("test/model")

        assert not hasattr(vlm, "max_new_tokens")

    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
    def test_processor_loads_from_the_same_hub_snapshot(
        self, mock_model_cls, mock_proc_cls
    ):
        """A pinned revision must pin the processor too, without handing it
        model-only kwargs."""
        from s3e.backends.huggingface import HuggingFaceVLM

        mock_model_cls.from_pretrained.return_value = MagicMock()
        mock_proc_cls.from_pretrained.return_value = MagicMock()
        quantization = object()

        HuggingFaceVLM(
            "test/model",
            revision="abc123",
            cache_dir="/models",
            trust_remote_code=True,
            quantization_config=quantization,
        )

        mock_proc_cls.from_pretrained.assert_called_once_with(
            "test/model", revision="abc123", cache_dir="/models", trust_remote_code=True
        )
        model_kwargs = mock_model_cls.from_pretrained.call_args.kwargs
        assert model_kwargs["revision"] == "abc123"
        assert model_kwargs["quantization_config"] is quantization

    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
    def test_query_returns_vlm_output(self, mock_model_cls, mock_proc_cls):
        from s3e.backends.huggingface import HuggingFaceVLM

        # Set up mock model to return logits
        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_model.device = torch.device("cpu")

        # Mock processor
        mock_processor = MagicMock()
        mock_proc_cls.from_pretrained.return_value = mock_processor
        mock_processor.return_value = {"input_ids": torch.ones(1, 5, dtype=torch.long)}
        mock_processor.decode.return_value = "yes"

        # Mock model output: logits shape (batch=1, seq_len=1, vocab_size=100)
        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 1, 100)
        mock_model.return_value = mock_output

        # Mock tokenizer within processor
        mock_processor.tokenizer.convert_tokens_to_ids.return_value = 0
        mock_processor.tokenizer.vocab_size = 100

        vlm = HuggingFaceVLM("test/model")
        img = Image.new("RGB", (64, 64))
        result = vlm.query([img], "Is A on B?")

        assert isinstance(result, VLMOutput)
        assert isinstance(result.token_probs, dict)
        assert result.text is None
        mock_model.assert_called_once()
        mock_model.generate.assert_not_called()

    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
    def test_query_returns_all_token_probs_by_default(
        self, mock_model_cls, mock_proc_cls
    ):
        from s3e.backends.huggingface import HuggingFaceVLM

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_model.device = torch.device("cpu")

        mock_processor = MagicMock()
        mock_proc_cls.from_pretrained.return_value = mock_processor
        mock_processor.return_value = {"input_ids": torch.ones(1, 5, dtype=torch.long)}
        mock_processor.decode.side_effect = lambda idx: f"tok{idx}"

        logits = torch.arange(25, dtype=torch.float32).reshape(1, 1, 25)
        mock_output = MagicMock()
        mock_output.logits = logits
        mock_model.return_value = mock_output

        vlm = HuggingFaceVLM("test/model")
        img = Image.new("RGB", (64, 64))
        result = vlm.query([img], "Is A on B?")

        expected_probs = torch.softmax(logits[0, -1].float(), dim=-1)
        assert set(result.token_probs) == {f"tok{i}" for i in range(25)}
        for token_id, expected_prob in enumerate(expected_probs):
            assert result.token_probs[f"tok{token_id}"] == pytest.approx(
                expected_prob.item()
            )

    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
    def test_query_limits_token_probs_when_num_logprobs_is_set(
        self, mock_model_cls, mock_proc_cls
    ):
        from s3e.backends.huggingface import HuggingFaceVLM

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_model.device = torch.device("cpu")

        mock_processor = MagicMock()
        mock_proc_cls.from_pretrained.return_value = mock_processor
        mock_processor.return_value = {"input_ids": torch.ones(1, 5, dtype=torch.long)}
        mock_processor.decode.side_effect = lambda idx: f"tok{idx}"

        logits = torch.tensor([[[0.0, 1.0, 2.0, 3.0]]])
        mock_output = MagicMock()
        mock_output.logits = logits
        mock_model.return_value = mock_output

        vlm = HuggingFaceVLM("test/model", num_logprobs=2)
        img = Image.new("RGB", (64, 64))
        result = vlm.query([img], "Is A on B?")

        expected_probs = torch.softmax(logits[0, -1].float(), dim=-1)
        assert set(result.token_probs) == {"tok2", "tok3"}
        assert result.token_probs["tok3"] == pytest.approx(expected_probs[3].item())
        assert result.token_probs["tok2"] == pytest.approx(expected_probs[2].item())

    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
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

    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
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

    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
    def test_interest_tokens_return_exact_masses_and_bypass_topk(
        self, mock_model_cls, mock_proc_cls
    ):
        logits = torch.tensor([[[0.0, 1.0, 2.0, 3.0]]], dtype=torch.float32)
        # num_logprobs=2 would drop tok0 on the legacy top-k path; the
        # interest gather must be exact regardless.
        vlm, _, _ = self._make_mock_hf_components(
            mock_model_cls,
            mock_proc_cls,
            logits=logits,
            vlm_kwargs={"num_logprobs": 2},
        )

        result = vlm.query([], "q1", interest_tokens=["tok0", "tok3"])

        expected_probs = torch.softmax(logits[0, -1, :].float(), dim=-1)
        assert set(result.token_probs) == {"tok0", "tok3"}
        assert result.token_probs["tok0"] == pytest.approx(expected_probs[0].item())
        assert result.token_probs["tok3"] == pytest.approx(expected_probs[3].item())
        assert result.argmax_in_interest is True

    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
    def test_interest_argmax_false_when_top_token_outside_interest(
        self, mock_model_cls, mock_proc_cls
    ):
        logits = torch.tensor([[[0.0, 1.0, 2.0, 3.0]]], dtype=torch.float32)
        vlm, _, _ = self._make_mock_hf_components(
            mock_model_cls, mock_proc_cls, logits=logits
        )

        result = vlm.query([], "q1", interest_tokens=["tok0"])

        assert result.argmax_in_interest is False

    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
    def test_interest_unknown_token_gets_zero_mass(
        self, mock_model_cls, mock_proc_cls
    ):
        logits = torch.tensor([[[0.0, 1.0, 2.0, 3.0]]], dtype=torch.float32)
        vlm, _, _ = self._make_mock_hf_components(
            mock_model_cls, mock_proc_cls, logits=logits
        )

        result = vlm.query([], "q1", interest_tokens=["tok1", "unknown"])

        assert result.token_probs["unknown"] == 0.0
        expected_probs = torch.softmax(logits[0, -1, :].float(), dim=-1)
        assert result.token_probs["tok1"] == pytest.approx(expected_probs[1].item())

    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
    def test_interest_reverse_index_is_built_once(
        self, mock_model_cls, mock_proc_cls
    ):
        logits = torch.tensor([[[0.0, 1.0, 2.0, 3.0]]], dtype=torch.float32)
        vlm, _, mock_processor = self._make_mock_hf_components(
            mock_model_cls, mock_proc_cls, logits=logits
        )

        vlm.query([], "q1", interest_tokens=["tok0"])
        decode_calls_after_first = (
            mock_processor.batch_decode.call_count
            + mock_processor.decode.call_count
        )
        vlm.query([], "q2", interest_tokens=["tok0"])
        decode_calls_after_second = (
            mock_processor.batch_decode.call_count
            + mock_processor.decode.call_count
        )

        assert decode_calls_after_second == decode_calls_after_first

    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
    def test_interest_sums_duplicate_ids_decoding_to_same_string(
        self, mock_model_cls, mock_proc_cls
    ):
        logits = torch.tensor([[[0.0, 1.0, 2.0, 3.0]]], dtype=torch.float32)
        vlm, _, mock_processor = self._make_mock_hf_components(
            mock_model_cls, mock_proc_cls, logits=logits
        )
        mock_processor.decode.side_effect = None
        mock_processor.decode.return_value = "same"
        mock_processor.batch_decode.side_effect = None
        mock_processor.batch_decode.return_value = ["same", "same", "same", "same"]

        result = vlm.query([], "q1", interest_tokens=["same"])

        assert result.token_probs["same"] == pytest.approx(1.0)
        assert result.argmax_in_interest is True

    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
    def test_interest_batch_rows_get_row_wise_masses_and_flags(
        self, mock_model_cls, mock_proc_cls
    ):
        logits = torch.tensor(
            [
                [[0.0, 1.0, 2.0]],
                [[2.0, 1.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        vlm, _, _ = self._make_mock_hf_components(
            mock_model_cls, mock_proc_cls, logits=logits
        )

        results = vlm.query_batch([], ["q1", "q2"], interest_tokens=["tok2"])

        expected_probs = torch.softmax(logits[:, -1, :].float(), dim=-1)
        assert results[0].token_probs["tok2"] == pytest.approx(
            expected_probs[0, 2].item()
        )
        assert results[1].token_probs["tok2"] == pytest.approx(
            expected_probs[1, 2].item()
        )
        assert results[0].argmax_in_interest is True
        assert results[1].argmax_in_interest is False

    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
    def test_interest_respected_by_sequential_fallback(
        self, mock_model_cls, mock_proc_cls
    ):
        logits = torch.tensor([[[0.0, 1.0]]], dtype=torch.float32)
        vlm, _, mock_processor = self._make_mock_hf_components(
            mock_model_cls, mock_proc_cls, logits=logits
        )
        img = Image.new("RGB", (64, 64))

        def processor_call(*, text, images, return_tensors, padding):
            del images, return_tensors, padding
            if isinstance(text, list):
                raise ValueError("nested image batches unsupported")
            return {"input_ids": torch.ones(1, 5, dtype=torch.long)}

        mock_processor.side_effect = processor_call

        results = vlm.query_batch([img], ["q1", "q2"], interest_tokens=["tok1"])

        assert all(set(r.token_probs) == {"tok1"} for r in results)
        assert all(r.argmax_in_interest is True for r in results)

    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
    def test_generate_mode_ignores_interest_tokens(
        self, mock_model_cls, mock_proc_cls
    ):
        input_ids = torch.ones(1, 3, dtype=torch.long)
        vlm, mock_model, mock_processor = self._make_mock_hf_components(
            mock_model_cls, mock_proc_cls, input_ids=input_ids
        )
        mock_model.generate.return_value = torch.tensor(
            [[1, 2, 3, 10]], dtype=torch.long
        )
        mock_processor.batch_decode.side_effect = None
        mock_processor.batch_decode.return_value = ["yes"]

        result = vlm.query(
            [], "q1", generate=True, interest_tokens=["tok0"]
        )

        assert result.text == "yes"
        assert result.token_probs is None
        assert result.argmax_in_interest is None

    def _make_ragged_length_hf_components(
        self, mock_model_cls, mock_proc_cls, **vlm_kwargs
    ):
        """Mock a model whose prediction depends on the last real prompt token.

        Prompt ``qN`` tokenizes to ``2 + N`` copies of token ``100 + N``, so
        rows have genuinely different lengths and each row has a distinct
        correct answer, making a misread pad position observably wrong.
        """
        from s3e.backends.huggingface import HuggingFaceVLM

        vocab_size = 8
        pad_id = 0

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_model.device = torch.device("cpu")
        mock_model.config.is_encoder_decoder = False

        mock_processor = MagicMock()
        mock_proc_cls.from_pretrained.return_value = mock_processor
        mock_processor.tokenizer.padding_side = "right"

        def tokenize(text):
            index = int(text[-1])
            return [100 + index] * (2 + index)

        def run_processor(text=None, images=None, **kwargs):
            del images, kwargs
            texts = [text] if isinstance(text, str) else list(text)
            rows = [tokenize(item) for item in texts]
            width = max(len(row) for row in rows)
            left = mock_processor.tokenizer.padding_side == "left"

            input_ids, attention_mask = [], []
            for row in rows:
                pad = [pad_id] * (width - len(row))
                input_ids.append(pad + row if left else row + pad)
                mask = [0] * len(pad)
                attention_mask.append(
                    mask + [1] * len(row) if left else [1] * len(row) + mask
                )

            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }

        def forward(input_ids=None, logits_to_keep=0, **kwargs):
            del kwargs
            batch, length = input_ids.shape
            logits = torch.zeros((batch, length, vocab_size), dtype=torch.float32)
            for b in range(batch):
                for t in range(length):
                    logits[b, t, int(input_ids[b, t]) % vocab_size] = 50.0
            if isinstance(logits_to_keep, int) and logits_to_keep > 0:
                logits = logits[:, -logits_to_keep:, :]
            output = MagicMock()
            output.logits = logits
            return output

        mock_processor.side_effect = run_processor
        mock_processor.apply_chat_template.side_effect = (
            lambda messages, **kwargs: f"chat:{messages[-1]['content'][-1]['text']}"
        )
        mock_processor.batch_decode.side_effect = lambda seqs, **kwargs: [
            f"tok{int(torch.as_tensor(seq).flatten()[0])}" for seq in seqs
        ]
        mock_model.side_effect = forward

        return HuggingFaceVLM("test/model", **vlm_kwargs), mock_model

    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
    def test_batched_queries_agree_with_unbatched(
        self, mock_model_cls, mock_proc_cls
    ):
        """The invariant the padding bug violated: batching must not change answers."""
        vlm, mock_model = self._make_ragged_length_hf_components(
            mock_model_cls, mock_proc_cls
        )
        prompts = ["q0", "q1", "q2"]

        unbatched = [vlm.query([], prompt).token_probs for prompt in prompts]
        batched = [out.token_probs for out in vlm.query_batch([], prompts)]

        assert batched == unbatched
        # each row answers with its own last real token, not a neighbour's
        assert [max(probs, key=probs.get) for probs in batched] == [
            "tok4",
            "tok5",
            "tok6",
        ]

        # 3 unbatched + 1 batched + 1 invariance-check rerun of the shortest row
        assert mock_model.call_count == 5
        vlm.query_batch([], prompts)
        assert mock_model.call_count == 6  # the check does not run again

    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
    def test_right_padded_batch_fails_the_pad_invariance_check(
        self, mock_model_cls, mock_proc_cls
    ):
        """A processor that right-pads despite the configured side is caught.

        Right padding puts a pad token at the final position of every short
        row, so the batched answer diverges from the row's unbatched answer
        and must be rejected rather than returned.
        """
        vlm, _ = self._make_ragged_length_hf_components(mock_model_cls, mock_proc_cls)
        vlm.processor.tokenizer.padding_side = "right"

        with pytest.raises(ValueError, match="under padding"):
            vlm.query_batch([], ["q0", "q1", "q2"])

    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
    def test_pad_invariance_check_can_be_skipped(self, mock_model_cls, mock_proc_cls):
        vlm, mock_model = self._make_ragged_length_hf_components(
            mock_model_cls, mock_proc_cls, skip_pad_invariance_check=True
        )

        results = vlm.query_batch([], ["q0", "q1", "q2"])

        assert mock_model.call_count == 1  # no verification forward
        assert [max(r.token_probs, key=r.token_probs.get) for r in results] == [
            "tok4",
            "tok5",
            "tok6",
        ]

    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
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

    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
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

    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
    def test_unsupported_interest_tokens_reports_multi_token_strings(
        self, mock_model_cls, mock_proc_cls
    ):
        from s3e.backends.huggingface import HuggingFaceVLM

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_processor = MagicMock()
        mock_proc_cls.from_pretrained.return_value = mock_processor

        vocab = ["yes", "no", "dark", "blue"]
        mock_processor.tokenizer.__len__.return_value = len(vocab)
        mock_processor.batch_decode.side_effect = lambda seqs, **kwargs: [
            vocab[seq[0]] for seq in seqs
        ]

        vlm = HuggingFaceVLM("test/model")

        unsupported = vlm.unsupported_interest_tokens(["yes", "dark blue"])

        assert unsupported == ["dark blue"]

    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
    def test_query_batch_empty_prompts_returns_empty_list(
        self, mock_model_cls, mock_proc_cls
    ):
        from s3e.backends.huggingface import HuggingFaceVLM

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_processor = MagicMock()
        mock_proc_cls.from_pretrained.return_value = mock_processor

        vlm = HuggingFaceVLM("test/model")
        img = Image.new("RGB", (64, 64))

        assert vlm.query_batch([img], []) == []
        mock_processor.assert_not_called()
        mock_model.assert_not_called()

    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
    def test_query_generate_mode_forwards_inference_kwargs(
        self, mock_model_cls, mock_proc_cls
    ):
        from s3e.backends.huggingface import HuggingFaceVLM

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_model.device = torch.device("cpu")

        mock_processor = MagicMock()
        mock_proc_cls.from_pretrained.return_value = mock_processor
        mock_processor.return_value = {"input_ids": torch.ones(1, 5, dtype=torch.long)}
        mock_processor.decode.return_value = "yes"

        # Generate one new token after the 5-token prompt.
        mock_model.generate.return_value = torch.ones((1, 6), dtype=torch.long)

        vlm = HuggingFaceVLM("test/model")
        img = Image.new("RGB", (64, 64))
        result = vlm.query(
            [img],
            "Is A on B?",
            generate=True,
            max_new_tokens=7,
            do_sample=False,
        )

        assert isinstance(result, VLMOutput)
        assert result.text == "yes"
        mock_model.assert_not_called()
        mock_model.generate.assert_called_once()
        assert mock_model.generate.call_args.kwargs["max_new_tokens"] == 7
        assert mock_model.generate.call_args.kwargs["do_sample"] is False

    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
    def test_query_generate_mode_sets_safe_defaults(
        self, mock_model_cls, mock_proc_cls
    ):
        from s3e.backends.huggingface import HuggingFaceVLM

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_model.device = torch.device("cpu")

        mock_processor = MagicMock()
        mock_proc_cls.from_pretrained.return_value = mock_processor
        mock_processor.return_value = {"input_ids": torch.ones(1, 5, dtype=torch.long)}
        mock_processor.decode.return_value = "yes"
        mock_model.generate.return_value = torch.ones((1, 6), dtype=torch.long)

        vlm = HuggingFaceVLM("test/model")
        img = Image.new("RGB", (64, 64))
        _ = vlm.query([img], "Is A on B?", generate=True)

        mock_model.generate.assert_called_once()
        generate_kwargs = mock_model.generate.call_args.kwargs
        assert "max_new_tokens" not in generate_kwargs

    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
    def test_generate_mode_does_not_forward_logits_to_keep(
        self, mock_model_cls, mock_proc_cls
    ):
        """The logits_to_keep=1 memory default belongs to the logprobs
        forward pass only; generate() manages its own logits and must not
        receive it."""
        from s3e.backends.huggingface import HuggingFaceVLM

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_model.device = torch.device("cpu")

        mock_processor = MagicMock()
        mock_proc_cls.from_pretrained.return_value = mock_processor
        mock_processor.return_value = {"input_ids": torch.ones(1, 5, dtype=torch.long)}
        mock_processor.decode.return_value = "yes"
        mock_model.generate.return_value = torch.ones((1, 6), dtype=torch.long)

        vlm = HuggingFaceVLM("test/model")
        img = Image.new("RGB", (64, 64))
        _ = vlm.query([img], "Is A on B?", generate=True)

        mock_model.generate.assert_called_once()
        assert "logits_to_keep" not in mock_model.generate.call_args.kwargs

    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
    def test_generate_error_propagates(self, mock_model_cls, mock_proc_cls):
        """A failing generate() must raise, not silently produce None text."""
        from s3e.backends.huggingface import HuggingFaceVLM

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_model.device = torch.device("cpu")

        mock_processor = MagicMock()
        mock_proc_cls.from_pretrained.return_value = mock_processor
        mock_processor.return_value = {"input_ids": torch.ones(1, 5, dtype=torch.long)}
        mock_model.generate.side_effect = RuntimeError("CUDA out of memory")

        vlm = HuggingFaceVLM("test/model")
        img = Image.new("RGB", (64, 64))

        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            vlm.query([img], "Is A on B?", generate=True)

    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
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

    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
    def test_query_batch_generate_returns_one_output_per_prompt_when_multiple_sequences_returned(
        self, mock_model_cls, mock_proc_cls
    ):
        input_ids = torch.ones(2, 3, dtype=torch.long)
        vlm, mock_model, mock_processor = self._make_mock_hf_components(
            mock_model_cls, mock_proc_cls, input_ids=input_ids
        )
        mock_model.generate.return_value = torch.tensor(
            [
                [1, 2, 3, 10, 11],
                [1, 2, 3, 12, 13],
                [4, 5, 6, 20, 21],
                [4, 5, 6, 22, 23],
            ],
            dtype=torch.long,
        )
        mock_processor.batch_decode.side_effect = None
        mock_processor.batch_decode.return_value = ["first q1", "first q2"]

        results = vlm.query_batch(
            [], ["q1", "q2"], generate=True, num_return_sequences=2
        )

        decoded_sequences = mock_processor.batch_decode.call_args.args[0]
        assert [seq.tolist() for seq in decoded_sequences] == [[10, 11], [20, 21]]
        assert len(results) == 2
        assert [result.text for result in results] == ["first q1", "first q2"]

    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
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

    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
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


# Vocabulary for the thinking tests: id -> decoded string.
THINK_VOCAB = ["<pad>", "q", "x", "\n\n", "Yes", "No", "r", "<think>", "</think>"]
THINK_START, THINK_END = 7, 8


def _logits_for(probs_by_id):
    """A logits row whose softmax puts ``probs_by_id`` on those ids."""
    row = torch.full((len(THINK_VOCAB),), -30.0)
    for token_id, prob in probs_by_id.items():
        row[token_id] = math.log(prob)
    return row


class TestHuggingFaceVLMThinking:
    """Thinking blocks: detection, closing, thinking first, and text mode."""

    def _make_vlm(self, mock_model_cls, mock_proc_cls, prompt_ids, forward_steps,
                  generate=None, **vlm_kwargs):
        """A mocked HuggingFaceVLM over ``THINK_VOCAB``.

        ``prompt_ids`` is what the processor returns for every text;
        ``forward_steps`` are the next-token distributions of successive
        forward passes (the last one repeats); ``generate`` is the side
        effect of ``model.generate``.
        """
        from s3e.backends.huggingface import HuggingFaceVLM

        model = MagicMock()
        mock_model_cls.from_pretrained.return_value = model
        model.device = torch.device("cpu")
        model.config.is_encoder_decoder = False
        model.generation_config.eos_token_id = 0
        model.generation_config.pad_token_id = 0
        steps = [
            _logits_for(probs).expand(1, len(prompt_ids), -1) for probs in forward_steps
        ]
        model.side_effect = lambda *a, **k: MagicMock(
            logits=steps.pop(0) if len(steps) > 1 else steps[0]
        )
        if generate is not None:
            model.generate.side_effect = generate

        processor = MagicMock()
        mock_proc_cls.from_pretrained.return_value = processor
        processor.return_value = {"input_ids": torch.tensor([prompt_ids])}
        processor.apply_chat_template.side_effect = (
            lambda messages, **kwargs: f"chat:{messages[-1]['content'][-1]['text']}"
        )
        processor.batch_decode.side_effect = lambda seqs, **kwargs: [
            THINK_VOCAB[int(seq[0])] for seq in seqs
        ]
        processor.decode.side_effect = lambda ids, skip_special_tokens=False: "".join(
            THINK_VOCAB[int(i)]
            for i in ids
            if not (skip_special_tokens and int(i) == 0)
        )
        tokenizer = processor.tokenizer
        tokenizer.encode.side_effect = lambda text, add_special_tokens=False: (
            [THINK_VOCAB.index(text)] if text in THINK_VOCAB else [1, 2]
        )
        tokenizer.decode.side_effect = lambda ids: "".join(THINK_VOCAB[i] for i in ids)

        vlm = HuggingFaceVLM("test/model", **vlm_kwargs)
        return vlm, model, processor

    @staticmethod
    def _separator_then_answer(prompt_ids):
        """generate() output of the separator pass: "\\n\\n", then "Yes"."""
        return torch.tensor([prompt_ids + [3, 4]])

    def _texts_sent_to_processor(self, processor):
        return [c.kwargs["text"] for c in processor.call_args_list]

    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
    def test_open_prompt_is_closed_and_answer_read_after_block(
        self, mock_model_cls, mock_proc_cls
    ):
        from s3e.backends.utils.thinking import MAX_SEPARATOR_TOKENS

        prompt_ids = [1, THINK_START]
        vlm, model, processor = self._make_vlm(
            mock_model_cls, mock_proc_cls,
            prompt_ids=prompt_ids,
            forward_steps=[{6: 0.9, 4: 0.05, 5: 0.05}, {4: 0.8, 5: 0.2}],
            generate=lambda **kwargs: self._separator_then_answer(prompt_ids),
        )

        result = vlm.query([], "q", interest_tokens=["Yes", "No"])

        assert result.token_probs == pytest.approx({"Yes": 0.8, "No": 0.2}, abs=1e-6)
        assert result.argmax_in_interest is True
        assert result.reasoning == ""
        # The separator pass sees the closed block; the answer is read from a
        # forward pass over the block plus the separator it found.
        assert self._texts_sent_to_processor(processor)[-2:] == [
            "chat:q</think>",
            "chat:q</think>\n\n",
        ]
        assert model.generate.call_args.kwargs["max_new_tokens"] == MAX_SEPARATOR_TOKENS
        assert model.generate.call_args.kwargs["do_sample"] is False
        assert model.call_args.kwargs["logits_to_keep"] == 1

    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
    def test_model_opening_a_block_gets_a_whole_empty_block(
        self, mock_model_cls, mock_proc_cls
    ):
        prompt_ids = [1, 2]
        vlm, _, processor = self._make_vlm(
            mock_model_cls, mock_proc_cls,
            prompt_ids=prompt_ids,
            forward_steps=[{THINK_START: 0.9, 4: 0.1}, {4: 0.3, 5: 0.7}],
            generate=lambda **kwargs: self._separator_then_answer(prompt_ids),
        )

        result = vlm.query([], "q", interest_tokens=["Yes", "No"])

        assert result.token_probs == pytest.approx({"Yes": 0.3, "No": 0.7}, abs=1e-6)
        assert self._texts_sent_to_processor(processor)[-1] == (
            "chat:q<think>\n\n</think>\n\n"
        )

    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
    def test_answer_right_after_block_needs_no_separator(
        self, mock_model_cls, mock_proc_cls
    ):
        prompt_ids = [1, THINK_START]
        vlm, _, processor = self._make_vlm(
            mock_model_cls, mock_proc_cls,
            prompt_ids=prompt_ids,
            forward_steps=[{6: 1.0}, {4: 0.6, 5: 0.4}],
            generate=lambda **kwargs: torch.tensor([prompt_ids + [5]]),
        )

        vlm.query([], "q", interest_tokens=["Yes", "No"])

        assert self._texts_sent_to_processor(processor)[-1] == "chat:q</think>"

    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
    def test_rows_without_a_block_are_untouched(self, mock_model_cls, mock_proc_cls):
        vlm, model, _ = self._make_vlm(
            mock_model_cls, mock_proc_cls,
            prompt_ids=[1, 2],
            forward_steps=[{4: 0.6, 5: 0.4}],
        )

        result = vlm.query([], "q", interest_tokens=["Yes", "No"])

        assert result.token_probs == pytest.approx({"Yes": 0.6, "No": 0.4}, abs=1e-6)
        assert result.reasoning is None
        model.generate.assert_not_called()
        assert model.call_count == 1

    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
    def test_delimiters_none_turns_handling_off(self, mock_model_cls, mock_proc_cls):
        vlm, model, _ = self._make_vlm(
            mock_model_cls, mock_proc_cls,
            prompt_ids=[1, THINK_START],
            forward_steps=[{6: 0.9, 4: 0.1}],
            thinking_delimiters=None,
        )

        result = vlm.query([], "q", interest_tokens=["Yes", "No"])

        assert result.reasoning is None
        assert result.argmax_in_interest is False
        model.generate.assert_not_called()

    @pytest.mark.parametrize(
        "generated, continuation, reasoning",
        [
            ([6, 6, THINK_END], "chat:qrr</think>", "rr"),
            ([6, 6, 6], "chat:qrrr\n</think>", "rrr"),  # budget ran out
        ],
    )
    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
    def test_think_mode_generates_reasoning_then_scores(
        self, mock_model_cls, mock_proc_cls, generated, continuation, reasoning
    ):
        from s3e.backends.utils.thinking import MAX_SEPARATOR_TOKENS

        prompt_ids = [1, THINK_START]

        def generate(**kwargs):
            if kwargs["max_new_tokens"] == MAX_SEPARATOR_TOKENS:
                return self._separator_then_answer(prompt_ids)
            return torch.tensor([prompt_ids + generated])

        vlm, model, processor = self._make_vlm(
            mock_model_cls, mock_proc_cls,
            prompt_ids=prompt_ids,
            forward_steps=[{6: 1.0}, {4: 0.9, 5: 0.1}],
            generate=generate,
            thinking="think",
            max_thinking_tokens=3,
        )

        result = vlm.query([], "q", interest_tokens=["Yes", "No"])

        think_call = model.generate.call_args_list[0].kwargs
        assert think_call["max_new_tokens"] == 3
        assert think_call["eos_token_id"] == [THINK_END, 0]
        assert think_call["pad_token_id"] == 0
        assert self._texts_sent_to_processor(processor)[-2:] == [
            continuation,
            continuation + "\n\n",
        ]
        assert result.reasoning == reasoning
        assert result.token_probs == pytest.approx({"Yes": 0.9, "No": 0.1}, abs=1e-6)

    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
    def test_think_mode_cuts_a_row_at_its_end_of_sequence(
        self, mock_model_cls, mock_proc_cls
    ):
        """Without a pad id, generate() pads finished rows with the first
        stop id, the closing delimiter; a row that ended on EOS is cut there
        and closed, not continued through the EOS."""
        from s3e.backends.utils.thinking import MAX_SEPARATOR_TOKENS

        prompt_ids = [1, THINK_START]

        def generate(**kwargs):
            if kwargs["max_new_tokens"] == MAX_SEPARATOR_TOKENS:
                return self._separator_then_answer(prompt_ids)
            return torch.tensor([prompt_ids + [6, 0, THINK_END]])

        vlm, model, processor = self._make_vlm(
            mock_model_cls, mock_proc_cls,
            prompt_ids=prompt_ids,
            forward_steps=[{6: 1.0}, {4: 0.9, 5: 0.1}],
            generate=generate,
            thinking="think",
        )
        model.generation_config.pad_token_id = None

        result = vlm.query([], "q", interest_tokens=["Yes", "No"])

        assert model.generate.call_args_list[0].kwargs["pad_token_id"] == 0
        assert self._texts_sent_to_processor(processor)[-1] == (
            "chat:qr\n</think>\n\n"
        )
        assert result.reasoning == "r"

    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
    def test_think_mode_opens_the_block_the_model_was_about_to_open(
        self, mock_model_cls, mock_proc_cls
    ):
        from s3e.backends.utils.thinking import MAX_SEPARATOR_TOKENS

        prompt_ids = [1, 2]

        def generate(**kwargs):
            if kwargs["max_new_tokens"] == MAX_SEPARATOR_TOKENS:
                return self._separator_then_answer(prompt_ids)
            return torch.tensor([prompt_ids + [6, THINK_END]])

        vlm, _, processor = self._make_vlm(
            mock_model_cls, mock_proc_cls,
            prompt_ids=prompt_ids,
            forward_steps=[{THINK_START: 0.6, 4: 0.4}, {4: 0.9, 5: 0.1}],
            generate=generate,
            thinking="think",
        )

        result = vlm.query([], "q", interest_tokens=["Yes", "No"])

        assert self._texts_sent_to_processor(processor)[1:3] == [
            "chat:q<think>",
            "chat:q<think>r</think>",
        ]
        assert result.reasoning == "r"

    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
    def test_answer_after_block_is_scored_with_the_query_kwargs(
        self, mock_model_cls, mock_proc_cls
    ):
        prompt_ids = [1, THINK_START]
        vlm, model, _ = self._make_vlm(
            mock_model_cls, mock_proc_cls,
            prompt_ids=prompt_ids,
            forward_steps=[{6: 1.0}, {4: 0.6, 5: 0.4}],
            generate=lambda **kwargs: self._separator_then_answer(prompt_ids),
        )

        vlm.query([], "q", interest_tokens=["Yes", "No"], use_cache=False)

        first_pass, answer_pass = (c.kwargs for c in model.call_args_list)
        assert answer_pass["use_cache"] is first_pass["use_cache"] is False
        assert answer_pass["logits_to_keep"] == 1
        assert "use_cache" not in model.generate.call_args.kwargs

    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
    def test_delimiter_in_the_users_text_opens_nothing(
        self, mock_model_cls, mock_proc_cls
    ):
        vlm, model, _ = self._make_vlm(
            mock_model_cls, mock_proc_cls,
            prompt_ids=[THINK_START, 1, 2],
            forward_steps=[{4: 0.6, 5: 0.4}],
        )

        result = vlm.query([], "q", interest_tokens=["Yes", "No"])

        assert result.reasoning is None
        model.generate.assert_not_called()

    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
    def test_sequential_fallback_skips_each_rows_block(
        self, mock_model_cls, mock_proc_cls
    ):
        prompt_ids = [1, THINK_START]
        vlm, _, processor = self._make_vlm(
            mock_model_cls, mock_proc_cls,
            prompt_ids=prompt_ids,
            forward_steps=[{6: 1.0}, {4: 0.8, 5: 0.2}, {6: 1.0}, {4: 0.3, 5: 0.7}],
            generate=lambda **kwargs: self._separator_then_answer(prompt_ids),
        )
        single_row = processor.return_value

        def process(text, **kwargs):
            if isinstance(text, list) and len(text) > 1:
                raise ValueError("cannot batch")
            return single_row

        processor.side_effect = process

        results = vlm.query_batch([], ["a", "b"], interest_tokens=["Yes", "No"])

        assert [r.reasoning for r in results] == ["", ""]
        assert results[0].token_probs == pytest.approx({"Yes": 0.8, "No": 0.2}, abs=1e-6)
        assert results[1].token_probs == pytest.approx({"Yes": 0.3, "No": 0.7}, abs=1e-6)

    @pytest.mark.parametrize(
        "prompt_ids, generated, text, reasoning",
        [
            ([1, THINK_START], [6, THINK_END, 3, 4], "Yes", "r"),
            ([1, 2], [THINK_START, 6, 6, THINK_END, 3, 5], "No", "rr"),
            ([1, THINK_START], [6, 6, 0], None, "rr"),
            ([1, 2], [4, 3], "Yes", None),  # no block: decoded as before
        ],
    )
    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
    def test_text_mode_moves_block_to_reasoning(
        self, mock_model_cls, mock_proc_cls, prompt_ids, generated, text, reasoning
    ):
        vlm, _, _ = self._make_vlm(
            mock_model_cls, mock_proc_cls,
            prompt_ids=prompt_ids,
            forward_steps=[{4: 1.0}],
            generate=lambda **kwargs: torch.tensor([prompt_ids + generated]),
        )

        result = vlm.query([], "q", generate=True, max_new_tokens=10)

        assert result.text == text
        assert result.reasoning == reasoning

    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
    def test_chat_template_kwargs_reach_the_template(
        self, mock_model_cls, mock_proc_cls
    ):
        vlm, _, processor = self._make_vlm(
            mock_model_cls, mock_proc_cls,
            prompt_ids=[1, 2],
            forward_steps=[{4: 1.0}],
            chat_template_kwargs={"enable_thinking": False},
        )

        vlm.query([], "q", interest_tokens=["Yes", "No"])

        assert processor.apply_chat_template.call_args.kwargs["enable_thinking"] is False

    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
    def test_template_rejecting_its_kwargs_raises(self, mock_model_cls, mock_proc_cls):
        """A prompt without the template would silently ignore the kwargs."""
        vlm, _, processor = self._make_vlm(
            mock_model_cls, mock_proc_cls,
            prompt_ids=[1, 2],
            forward_steps=[{4: 1.0}],
            chat_template_kwargs={"enable_thinking": False},
        )
        processor.apply_chat_template.side_effect = ValueError("template error")

        with pytest.raises(ValueError, match="template error"):
            vlm.query([], "q", interest_tokens=["Yes", "No"])

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"thinking": "skip"},
            {"max_thinking_tokens": 0},
            {"thinking_delimiters": ("<think>",)},
        ],
    )
    @patch("s3e.backends.huggingface.AutoProcessor")
    @patch("s3e.backends.huggingface._AutoModelClass")
    def test_invalid_thinking_options_rejected_before_model_load(
        self, mock_model_cls, mock_proc_cls, kwargs
    ):
        from s3e.backends.huggingface import HuggingFaceVLM

        with pytest.raises(ValueError):
            HuggingFaceVLM("test/model", **kwargs)

        mock_model_cls.from_pretrained.assert_not_called()


@pytest.mark.slow
class TestHuggingFaceVLMIntegration:
    """Integration tests with a tiny real HF model.

    These tests download a small model and run actual inference.
    Skip with: pytest -m "not slow"
    """

    TINY_VLM_ID = "katuni4ka/tiny-random-llava"

    def test_loads_and_queries(self):
        from s3e.backends.huggingface import HuggingFaceVLM

        vlm = HuggingFaceVLM(self.TINY_VLM_ID, device_map="cpu", torch_dtype=torch.float32)
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        result = vlm.query([img], "Is this a test?")

        assert isinstance(result, VLMOutput)
        assert isinstance(result.token_probs, dict)
        assert len(result.token_probs) > 0
        assert all(p >= 0 for p in result.token_probs.values())

    def test_interest_tokens_parity_with_full_vocab(self):
        """Interest-mode masses must equal the full-vocabulary path's."""
        from s3e.backends.huggingface import HuggingFaceVLM

        vlm = HuggingFaceVLM(self.TINY_VLM_ID, device_map="cpu", torch_dtype=torch.float32)
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        prompt = "Is this a test?"
        interest = ["Yes", "No", "yes", "no"]

        full = vlm.query([img], prompt)
        gathered = vlm.query([img], prompt, interest_tokens=interest)

        assert set(gathered.token_probs) == set(interest)
        for token in interest:
            assert gathered.token_probs[token] == pytest.approx(
                full.token_probs.get(token, 0.0), abs=1e-9
            )
        assert gathered.argmax_in_interest == (
            max(full.token_probs, key=full.token_probs.get) in set(interest)
        )

    def test_query_batch(self):
        from s3e.backends.huggingface import HuggingFaceVLM

        vlm = HuggingFaceVLM(
            self.TINY_VLM_ID, device_map="cpu", torch_dtype=torch.float32, num_logprobs=2
        )
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



@pytest.mark.slow
class TestHuggingFaceVLMThinkingIntegration:
    """Thinking handling against a tiny random Qwen3.5.

    Its chat template opens a ``<think>`` block by default, like the real
    Qwen3.5-4B and larger. The weights are random, so the tests check the
    mechanics (where the answer is read, what lands in ``reasoning``), not
    the answers.
    """

    TINY_THINKING_VLM_ID = (
        "onnx-internal-testing/tiny-random-Qwen3_5ForConditionalGeneration"
    )

    def _backend(self, **kwargs):
        from s3e.backends.huggingface import HuggingFaceVLM

        return HuggingFaceVLM(
            self.TINY_THINKING_VLM_ID,
            device_map="cpu",
            torch_dtype=torch.float32,
            **kwargs,
        )

    @pytest.fixture(scope="class")
    def image(self):
        return Image.new("RGB", (64, 64), color=(200, 30, 30))

    def test_logprobs_mode_closes_the_open_block(self, image):
        result = self._backend().query(
            [image], "Is it red? Answer Yes or No.", interest_tokens=["Yes", "No"]
        )

        assert result.reasoning == ""
        assert set(result.token_probs) == {"Yes", "No"}
        assert 0.0 <= sum(result.token_probs.values()) <= 1.0 + 1e-6

    def test_thinking_off_through_the_template(self, image):
        backend = self._backend(chat_template_kwargs={"enable_thinking": False})

        result = backend.query([image], "Is it red?", interest_tokens=["Yes", "No"])

        assert result.reasoning is None

    def test_think_mode_generates_reasoning(self, image):
        backend = self._backend(thinking="think", max_thinking_tokens=4)

        result = backend.query([image], "Is it red?", interest_tokens=["Yes", "No"])

        assert isinstance(result.reasoning, str)
        assert set(result.token_probs) == {"Yes", "No"}

    def test_text_mode_moves_unfinished_block_to_reasoning(self, image):
        result = self._backend().query(
            [image], "Is it red?", generate=True, max_new_tokens=6, do_sample=False
        )

        assert result.text is None
        assert result.reasoning


@pytest.mark.slow
class TestHuggingFaceVLMContract(BackendContract):
    """Applies the shared backend contract to a real, tiny HF model."""

    @pytest.fixture(scope="class")
    def make_backend(self):
        from s3e.backends.huggingface import HuggingFaceVLM

        backend = HuggingFaceVLM(
            TestHuggingFaceVLMIntegration.TINY_VLM_ID, device_map="cpu", torch_dtype=torch.float32
        )
        return lambda: backend

