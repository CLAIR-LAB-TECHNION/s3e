"""Tests for VLM backends."""

import importlib.util
import os

import pytest
from PIL import Image

from s3e.vlm.backend import VLMBackend, VLMOutput


class FakeVLM(VLMBackend):
    """A fake VLM that returns configurable token probabilities."""

    def __init__(
        self, token_probs: dict[str, float] | None = None, text: str | None = None
    ):
        self.token_probs = token_probs or {"yes": 0.8, "no": 0.2}
        self.text = text
        self.call_count = 0

    def query(
        self, images, prompt, system_prompt=None, generate=False, **inference_kwargs
    ):
        del generate
        del inference_kwargs
        self.call_count += 1
        return VLMOutput(token_probs=dict(self.token_probs), text=self.text)


class TestVLMOutput:
    def test_creation(self):
        output = VLMOutput(token_probs={"yes": 0.9, "no": 0.1})
        assert output.token_probs["yes"] == 0.9
        assert output.text is None

    def test_with_text(self):
        output = VLMOutput(token_probs={"yes": 0.9}, text="yes")
        assert output.text == "yes"

    def test_argmax_in_interest_defaults_to_none(self):
        output = VLMOutput(token_probs={"yes": 0.9})
        assert output.argmax_in_interest is None

    def test_argmax_in_interest_is_settable(self):
        output = VLMOutput(token_probs={"yes": 0.9}, argmax_in_interest=True)
        assert output.argmax_in_interest is True


class TestVLMBackend:
    def test_query_returns_vlm_output(self):
        vlm = FakeVLM()
        img = Image.new("RGB", (64, 64))
        result = vlm.query([img], "Is block A on block B?")
        assert isinstance(result, VLMOutput)
        assert "yes" in result.token_probs

    def test_query_batch_default_loops(self):
        vlm = FakeVLM()
        img = Image.new("RGB", (64, 64))
        results = vlm.query_batch([img], ["q1", "q2", "q3"])
        assert len(results) == 3
        assert vlm.call_count == 3
        assert all(isinstance(r, VLMOutput) for r in results)

    def test_query_batch_passes_system_prompt(self):
        class TrackingVLM(VLMBackend):
            def __init__(self):
                self.received_system_prompts = []

            def query(
                self,
                images,
                prompt,
                system_prompt=None,
                generate=False,
                **inference_kwargs,
            ):
                del generate
                del inference_kwargs
                self.received_system_prompts.append(system_prompt)
                return VLMOutput(token_probs={"yes": 0.5})

        vlm = TrackingVLM()
        img = Image.new("RGB", (64, 64))
        vlm.query_batch([img], ["q1", "q2"], system_prompt="Be helpful.")
        assert vlm.received_system_prompts == ["Be helpful.", "Be helpful."]

    def test_query_batch_forwards_interest_tokens(self):
        class TrackingVLM(VLMBackend):
            def __init__(self):
                self.received_interest_tokens = []

            def query(
                self,
                images,
                prompt,
                system_prompt=None,
                generate=False,
                interest_tokens=None,
                **inference_kwargs,
            ):
                del generate
                del inference_kwargs
                self.received_interest_tokens.append(interest_tokens)
                return VLMOutput(token_probs={"yes": 0.5})

        vlm = TrackingVLM()
        img = Image.new("RGB", (64, 64))
        vlm.query_batch([img], ["q1", "q2"], interest_tokens=["yes", "no"])
        assert vlm.received_interest_tokens == [["yes", "no"], ["yes", "no"]]


from unittest.mock import MagicMock, patch
from s3e.vlm.openai import OpenAIVLM


class TestOpenAIVLM:
    def _make_mock_response(self, token_logprobs):
        """Create a mock OpenAI response with given token->logprob pairs."""
        import math

        mock_top_logprobs = []
        for token, logprob in token_logprobs:
            item = MagicMock()
            item.token = token
            item.logprob = logprob
            mock_top_logprobs.append(item)

        mock_content = MagicMock()
        mock_content.top_logprobs = mock_top_logprobs

        mock_choice = MagicMock()
        mock_choice.logprobs.content = [mock_content]
        mock_choice.message.content = "yes"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        return mock_response

    @patch("s3e.vlm.openai.openai")
    def test_query_returns_vlm_output(self, mock_openai_module):
        import math

        mock_client = MagicMock()
        mock_openai_module.OpenAI.return_value = mock_client

        # ln(0.8) ≈ -0.223
        mock_client.chat.completions.create.return_value = self._make_mock_response(
            [("yes", math.log(0.8)), ("no", math.log(0.2))]
        )

        vlm = OpenAIVLM("gpt-4o")
        img = Image.new("RGB", (64, 64))
        result = vlm.query([img], "Is A on B?")

        assert isinstance(result, VLMOutput)
        assert "yes" in result.token_probs
        assert "no" in result.token_probs
        assert result.text == "yes"

    @patch("s3e.vlm.openai.openai")
    def test_strips_openai_prefix(self, mock_openai_module):
        mock_client = MagicMock()
        mock_openai_module.OpenAI.return_value = mock_client

        vlm = OpenAIVLM("OpenAI/gpt-4o")
        assert vlm.model_id == "gpt-4o"

    @patch("s3e.vlm.openai.openai")
    def test_query_batch_calls_query_per_prompt(self, mock_openai_module):
        import math

        mock_client = MagicMock()
        mock_openai_module.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._make_mock_response(
            [("yes", math.log(0.7)), ("no", math.log(0.3))]
        )

        vlm = OpenAIVLM("gpt-4o")
        img = Image.new("RGB", (64, 64))
        results = vlm.query_batch([img], ["q1", "q2"])
        assert len(results) == 2

    @patch("s3e.vlm.openai.openai")
    def test_interest_tokens_filter_and_backfill(self, mock_openai_module):
        import math

        mock_client = MagicMock()
        mock_openai_module.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._make_mock_response(
            [
                ("yes", math.log(0.6)),
                ("no", math.log(0.3)),
                ("maybe", math.log(0.1)),
            ]
        )

        vlm = OpenAIVLM("gpt-4o")
        img = Image.new("RGB", (64, 64))
        result = vlm.query(
            [img], "Is A on B?", interest_tokens=["yes", "no", "null"]
        )

        assert set(result.token_probs) == {"yes", "no", "null"}
        assert result.token_probs["yes"] == pytest.approx(0.6)
        assert result.token_probs["no"] == pytest.approx(0.3)
        assert result.token_probs["null"] == 0.0
        assert result.argmax_in_interest is True

    @patch("s3e.vlm.openai.openai")
    def test_interest_argmax_false_when_top_entry_outside_interest(
        self, mock_openai_module
    ):
        import math

        mock_client = MagicMock()
        mock_openai_module.OpenAI.return_value = mock_client
        # Deliberately unsorted: the highest-probability entry is listed
        # second, so an implementation that trusts list order is caught.
        mock_client.chat.completions.create.return_value = self._make_mock_response(
            [("yes", math.log(0.4)), ("maybe", math.log(0.5))]
        )

        vlm = OpenAIVLM("gpt-4o")
        img = Image.new("RGB", (64, 64))
        result = vlm.query([img], "Is A on B?", interest_tokens=["yes", "no"])

        assert result.argmax_in_interest is False
        assert result.token_probs["yes"] == pytest.approx(0.4)

    @patch("s3e.vlm.openai.openai")
    def test_interest_tokens_sum_duplicate_entries(self, mock_openai_module):
        import math

        mock_client = MagicMock()
        mock_openai_module.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._make_mock_response(
            [("yes", math.log(0.3)), ("yes", math.log(0.2))]
        )

        vlm = OpenAIVLM("gpt-4o")
        img = Image.new("RGB", (64, 64))
        result = vlm.query([img], "Is A on B?", interest_tokens=["yes"])

        assert result.token_probs["yes"] == pytest.approx(0.5)

    @patch("s3e.vlm.openai.openai")
    def test_no_interest_tokens_keeps_full_dict_and_none_flag(
        self, mock_openai_module
    ):
        import math

        mock_client = MagicMock()
        mock_openai_module.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._make_mock_response(
            [("yes", math.log(0.6)), ("maybe", math.log(0.4))]
        )

        vlm = OpenAIVLM("gpt-4o")
        img = Image.new("RGB", (64, 64))
        result = vlm.query([img], "Is A on B?")

        assert set(result.token_probs) == {"yes", "maybe"}
        assert result.argmax_in_interest is None


import torch


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

    @patch("s3e.vlm.huggingface.AutoProcessor")
    @patch("s3e.vlm.huggingface._AutoModelClass")
    def test_construction(self, mock_model_cls, mock_proc_cls):
        from s3e.vlm.huggingface import HuggingFaceVLM

        vlm = HuggingFaceVLM("test/model")
        mock_model_cls.from_pretrained.assert_called_once()
        mock_proc_cls.from_pretrained.assert_called_once()
        assert vlm.max_new_tokens == 10
        assert vlm.num_logprobs is None

    @patch("s3e.vlm.huggingface.AutoProcessor")
    @patch("s3e.vlm.huggingface._AutoModelClass")
    def test_custom_max_new_tokens(self, mock_model_cls, mock_proc_cls):
        from s3e.vlm.huggingface import HuggingFaceVLM

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_proc_cls.from_pretrained.return_value = MagicMock()

        vlm = HuggingFaceVLM("test/model", max_new_tokens=42)

        assert vlm.max_new_tokens == 42

    @patch("s3e.vlm.huggingface.AutoProcessor")
    @patch("s3e.vlm.huggingface._AutoModelClass")
    def test_query_returns_vlm_output(self, mock_model_cls, mock_proc_cls):
        from s3e.vlm.huggingface import HuggingFaceVLM

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

        vlm = HuggingFaceVLM("test/model", max_new_tokens=100)
        img = Image.new("RGB", (64, 64))
        result = vlm.query([img], "Is A on B?")

        assert isinstance(result, VLMOutput)
        assert isinstance(result.token_probs, dict)
        assert result.text is None
        mock_model.assert_called_once()
        mock_model.generate.assert_not_called()

    @patch("s3e.vlm.huggingface.AutoProcessor")
    @patch("s3e.vlm.huggingface._AutoModelClass")
    def test_query_returns_all_token_probs_by_default(
        self, mock_model_cls, mock_proc_cls
    ):
        from s3e.vlm.huggingface import HuggingFaceVLM

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

    @patch("s3e.vlm.huggingface.AutoProcessor")
    @patch("s3e.vlm.huggingface._AutoModelClass")
    def test_query_limits_token_probs_when_num_logprobs_is_set(
        self, mock_model_cls, mock_proc_cls
    ):
        from s3e.vlm.huggingface import HuggingFaceVLM

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

    @patch("s3e.vlm.huggingface.AutoProcessor")
    @patch("s3e.vlm.huggingface._AutoModelClass")
    def test_interest_argmax_false_when_top_token_outside_interest(
        self, mock_model_cls, mock_proc_cls
    ):
        logits = torch.tensor([[[0.0, 1.0, 2.0, 3.0]]], dtype=torch.float32)
        vlm, _, _ = self._make_mock_hf_components(
            mock_model_cls, mock_proc_cls, logits=logits
        )

        result = vlm.query([], "q1", interest_tokens=["tok0"])

        assert result.argmax_in_interest is False

    @patch("s3e.vlm.huggingface.AutoProcessor")
    @patch("s3e.vlm.huggingface._AutoModelClass")
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

    @patch("s3e.vlm.huggingface.AutoProcessor")
    @patch("s3e.vlm.huggingface._AutoModelClass")
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

    @patch("s3e.vlm.huggingface.AutoProcessor")
    @patch("s3e.vlm.huggingface._AutoModelClass")
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

    @patch("s3e.vlm.huggingface.AutoProcessor")
    @patch("s3e.vlm.huggingface._AutoModelClass")
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

    @patch("s3e.vlm.huggingface.AutoProcessor")
    @patch("s3e.vlm.huggingface._AutoModelClass")
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

    @patch("s3e.vlm.huggingface.AutoProcessor")
    @patch("s3e.vlm.huggingface._AutoModelClass")
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
        from s3e.vlm.huggingface import HuggingFaceVLM

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

    @patch("s3e.vlm.huggingface.AutoProcessor")
    @patch("s3e.vlm.huggingface._AutoModelClass")
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

    @patch("s3e.vlm.huggingface.AutoProcessor")
    @patch("s3e.vlm.huggingface._AutoModelClass")
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

    @patch("s3e.vlm.huggingface.AutoProcessor")
    @patch("s3e.vlm.huggingface._AutoModelClass")
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

    @patch("s3e.vlm.huggingface.AutoProcessor")
    @patch("s3e.vlm.huggingface._AutoModelClass")
    def test_query_generate_mode_forwards_inference_kwargs(
        self, mock_model_cls, mock_proc_cls
    ):
        from s3e.vlm.huggingface import HuggingFaceVLM

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

    @patch("s3e.vlm.huggingface.AutoProcessor")
    @patch("s3e.vlm.huggingface._AutoModelClass")
    def test_query_generate_mode_sets_safe_defaults(
        self, mock_model_cls, mock_proc_cls
    ):
        from s3e.vlm.huggingface import HuggingFaceVLM

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


@pytest.mark.slow
class TestHuggingFaceVLMIntegration:
    """Integration tests with a tiny real HF model.

    These tests download a small model and run actual inference.
    Skip with: pytest -m "not slow"
    """

    TINY_VLM_ID = "katuni4ka/tiny-random-llava"

    def test_loads_and_queries(self):
        from s3e.vlm.huggingface import HuggingFaceVLM

        vlm = HuggingFaceVLM(self.TINY_VLM_ID, device_map="cpu")
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        result = vlm.query([img], "Is this a test?")

        assert isinstance(result, VLMOutput)
        assert isinstance(result.token_probs, dict)
        assert len(result.token_probs) > 0
        assert all(p >= 0 for p in result.token_probs.values())

    def test_interest_tokens_parity_with_full_vocab(self):
        """Interest-mode masses must equal the full-vocabulary path's."""
        from s3e.vlm.huggingface import HuggingFaceVLM

        vlm = HuggingFaceVLM(self.TINY_VLM_ID, device_map="cpu")
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


import math as _math  # noqa: E402  (module-level math for vLLM helpers)


def _make_logprob(token, logprob, rank=None):
    """Build a mock vLLM Logprob (has .decoded_token, .logprob and .rank)."""
    item = MagicMock()
    item.decoded_token = token
    item.logprob = logprob
    item.rank = rank
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


def _make_id_logprobs_output(id_logprobs):
    """Build a mock vLLM RequestOutput keyed by explicit token ids.

    Mirrors ``detokenize=False`` output: ``decoded_token`` is None, the
    logprobs dict keys are real token ids, and every entry carries its
    vocab ``rank`` (1 = highest probability), as real vLLM entries do.
    """
    ranked_ids = sorted(id_logprobs, key=id_logprobs.get, reverse=True)
    completion = MagicMock()
    completion.logprobs = [
        {
            token_id: _make_logprob(
                None, logprob, rank=ranked_ids.index(token_id) + 1
            )
            for token_id, logprob in id_logprobs.items()
        }
    ]
    output = MagicMock()
    output.outputs = [completion]
    return output


def _make_id_tokenizer(vocab):
    """Build a mock tokenizer whose id ``i`` decodes to ``vocab[i]``."""
    tokenizer = MagicMock()
    tokenizer.__len__.return_value = len(vocab)
    tokenizer.batch_decode.side_effect = lambda sequences, **kwargs: [
        vocab[sequence[0]] for sequence in sequences
    ]
    tokenizer.decode.side_effect = lambda ids, **kwargs: vocab[
        ids[0] if isinstance(ids, list) else ids
    ]
    return tokenizer


class TestVLLMBackendMocked:
    """Unit tests for VLLMBackend with vllm mocked out."""

    def _make_backend(self, mock_llm_cls, num_logprobs=None):
        """Construct a VLLMBackend whose engine is the mocked LLM instance."""
        from s3e.vlm.vllm import VLLMBackend

        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm
        with patch("torch.cuda.device_count", return_value=1):
            backend = VLLMBackend("test/model", num_logprobs=num_logprobs)
        return backend, mock_llm

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
        assert kwargs["logprobs"] == -1

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

    @patch("s3e.vlm.vllm.SamplingParams")
    @patch("s3e.vlm.vllm.LLM")
    def test_interest_mode_requests_detokenize_false(
        self, mock_llm_cls, mock_sp_cls
    ):
        backend, mock_llm = self._make_backend(mock_llm_cls)
        mock_llm.get_tokenizer.return_value = _make_id_tokenizer(["yes", "no"])
        mock_llm.chat.return_value = [
            _make_id_logprobs_output({0: _math.log(0.8), 1: _math.log(0.2)})
        ]

        backend.query([], "q1", interest_tokens=["yes", "no"])

        kwargs = mock_sp_cls.call_args.kwargs
        assert kwargs["detokenize"] is False
        assert kwargs["logprobs"] == -1
        assert kwargs["max_tokens"] == 1

    @patch("s3e.vlm.vllm.SamplingParams")
    @patch("s3e.vlm.vllm.LLM")
    def test_no_interest_mode_does_not_touch_detokenize(
        self, mock_llm_cls, mock_sp_cls
    ):
        backend, mock_llm = self._make_backend(mock_llm_cls)
        mock_llm.chat.return_value = [_make_logprobs_output([("yes", _math.log(0.5))])]

        backend.query([], "q1")

        assert "detokenize" not in mock_sp_cls.call_args.kwargs

    @patch("s3e.vlm.vllm.SamplingParams")
    @patch("s3e.vlm.vllm.LLM")
    def test_interest_matches_by_token_id_and_backfills(
        self, mock_llm_cls, mock_sp_cls
    ):
        backend, mock_llm = self._make_backend(mock_llm_cls)
        mock_llm.get_tokenizer.return_value = _make_id_tokenizer(
            ["yes", "no", "maybe"]
        )
        mock_llm.chat.return_value = [
            _make_id_logprobs_output(
                {0: _math.log(0.6), 1: _math.log(0.3), 2: _math.log(0.1)}
            )
        ]

        result = backend.query([], "q1", interest_tokens=["yes", "no", "null"])

        assert set(result.token_probs) == {"yes", "no", "null"}
        assert result.token_probs["yes"] == pytest.approx(0.6)
        assert result.token_probs["no"] == pytest.approx(0.3)
        assert result.token_probs["null"] == 0.0
        assert result.argmax_in_interest is True
        assert result.text is None

    @patch("s3e.vlm.vllm.SamplingParams")
    @patch("s3e.vlm.vllm.LLM")
    def test_interest_sums_duplicate_ids_decoding_to_same_string(
        self, mock_llm_cls, mock_sp_cls
    ):
        backend, mock_llm = self._make_backend(mock_llm_cls)
        mock_llm.get_tokenizer.return_value = _make_id_tokenizer(["yes", "yes"])
        mock_llm.chat.return_value = [
            _make_id_logprobs_output({0: _math.log(0.4), 1: _math.log(0.2)})
        ]

        result = backend.query([], "q1", interest_tokens=["yes"])

        assert result.token_probs["yes"] == pytest.approx(0.6)

    @patch("s3e.vlm.vllm.SamplingParams")
    @patch("s3e.vlm.vllm.LLM")
    def test_interest_argmax_false_when_top_id_outside_interest(
        self, mock_llm_cls, mock_sp_cls
    ):
        backend, mock_llm = self._make_backend(mock_llm_cls)
        mock_llm.get_tokenizer.return_value = _make_id_tokenizer(["maybe", "yes"])
        mock_llm.chat.return_value = [
            _make_id_logprobs_output({0: _math.log(0.5), 1: _math.log(0.4)})
        ]

        result = backend.query([], "q1", interest_tokens=["yes"])

        assert result.argmax_in_interest is False
        assert result.token_probs["yes"] == pytest.approx(0.4)

    @patch("s3e.vlm.vllm.SamplingParams")
    @patch("s3e.vlm.vllm.LLM")
    def test_interest_reverse_index_is_built_once(
        self, mock_llm_cls, mock_sp_cls
    ):
        backend, mock_llm = self._make_backend(mock_llm_cls)
        tokenizer = _make_id_tokenizer(["yes", "no"])
        mock_llm.get_tokenizer.return_value = tokenizer
        mock_llm.chat.return_value = [
            _make_id_logprobs_output({0: _math.log(0.8), 1: _math.log(0.2)})
        ]

        backend.query([], "q1", interest_tokens=["yes"])
        backend.query([], "q2", interest_tokens=["yes"])

        assert mock_llm.get_tokenizer.call_count == 1
        assert tokenizer.batch_decode.call_count == 1

    @patch("s3e.vlm.vllm.SamplingParams")
    @patch("s3e.vlm.vllm.LLM")
    def test_interest_with_stop_strings_keeps_detokenization(
        self, mock_llm_cls, mock_sp_cls
    ):
        """detokenize=False forbids stop strings, so stop wins."""
        backend, mock_llm = self._make_backend(mock_llm_cls)
        mock_llm.get_tokenizer.return_value = _make_id_tokenizer(["yes"])
        mock_llm.chat.return_value = [
            _make_id_logprobs_output({0: _math.log(0.8)})
        ]

        backend.query([], "q1", interest_tokens=["yes"], stop=["\n"])

        assert "detokenize" not in mock_sp_cls.call_args.kwargs

    @patch("s3e.vlm.vllm.SamplingParams")
    @patch("s3e.vlm.vllm.LLM")
    def test_generate_mode_ignores_interest_tokens(
        self, mock_llm_cls, mock_sp_cls
    ):
        backend, mock_llm = self._make_backend(mock_llm_cls)
        mock_llm.chat.return_value = [_make_text_output("yes")]

        result = backend.query([], "q1", generate=True, interest_tokens=["yes"])

        assert result.text == "yes"
        assert result.token_probs is None
        assert result.argmax_in_interest is None
        kwargs = mock_sp_cls.call_args.kwargs
        assert "detokenize" not in kwargs
        assert "logprobs" not in kwargs
        mock_llm.get_tokenizer.assert_not_called()

    @patch("s3e.vlm.vllm.SamplingParams", None)
    @patch("s3e.vlm.vllm.LLM", None)
    def test_missing_vllm_raises_install_guidance(self):
        from s3e.vlm.vllm import VLLMBackend

        with pytest.raises(ImportError, match=r"pip install s3e\[vllm\]"):
            VLLMBackend("test/model")

    def test_installed_vllm_import_failure_is_not_masked(self):
        import builtins
        import importlib
        import sys

        module_name = "s3e.vlm.vllm"
        parent_module = sys.modules.get("s3e.vlm")
        original_module = sys.modules.pop(module_name, None)
        had_parent_attr = parent_module is not None and hasattr(parent_module, "vllm")
        original_parent_attr = (
            getattr(parent_module, "vllm", None) if had_parent_attr else None
        )
        if had_parent_attr:
            delattr(parent_module, "vllm")

        real_import = builtins.__import__

        def import_with_broken_vllm(
            name, globals=None, locals=None, fromlist=(), level=0
        ):
            if name == "vllm":
                raise ModuleNotFoundError(
                    "No module named 'vllm_dependency'", name="vllm_dependency"
                )
            return real_import(name, globals, locals, fromlist, level)

        try:
            with patch("builtins.__import__", side_effect=import_with_broken_vllm):
                with pytest.raises(ModuleNotFoundError, match="vllm_dependency"):
                    importlib.import_module(module_name)
        finally:
            sys.modules.pop(module_name, None)
            if original_module is not None:
                sys.modules[module_name] = original_module
            if parent_module is not None:
                if had_parent_attr:
                    setattr(parent_module, "vllm", original_parent_attr)
                elif hasattr(parent_module, "vllm"):
                    delattr(parent_module, "vllm")


def test_vllm_backend_is_exported():
    import s3e
    from s3e.vlm import VLLMBackend as FromVlm
    from s3e import VLLMBackend as FromTop

    assert FromVlm is FromTop
    assert "VLLMBackend" in s3e.__all__
    assert "VLLMBackend" in s3e.vlm.__all__


def test_import_s3e_does_not_touch_broken_vllm_dependency():
    import subprocess
    import sys

    script = """
import builtins

real_import = builtins.__import__


def import_with_broken_vllm(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "vllm":
        raise ModuleNotFoundError(
            "No module named 'vllm_dependency'", name="vllm_dependency"
        )
    return real_import(name, globals, locals, fromlist, level)


builtins.__import__ = import_with_broken_vllm

import s3e

assert "VLLMBackend" in s3e.__all__

try:
    from s3e import VLLMBackend  # noqa: F401
except ModuleNotFoundError as exc:
    assert exc.name == "vllm_dependency", exc.name
else:
    raise AssertionError("Explicit VLLMBackend access should surface broken vLLM")
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("vllm") is None,
    reason="vLLM requires CUDA and an installed vllm package",
)
class TestVLLMBackendIntegration:
    """Integration test with a small real VLM via vLLM.

    Requires a GPU and an installed ``vllm``; skipped otherwise. The host must
    also expose a CUDA dev toolchain (``nvcc`` + CUDA headers) because vLLM's
    default logprobs sampler (FlashInfer) JIT-compiles a kernel on first use.
    Run with: ``pytest -m slow``.

    The model is the small ``SmolVLM-256M-Instruct`` rather than a degenerate
    ``tiny-random`` stub: such stubs use a head dim (e.g. 4) below the minimum
    that CUDA attention kernels accept, so they cannot actually run on a GPU.
    ``enforce_eager=True`` skips torch.compile / CUDA-graph capture, keeping the
    smoke test fast and free of compile-time backend surprises.
    """

    SMALL_VLM_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"

    @pytest.fixture(scope="class")
    def backend(self):
        # Evaluating this class's skipif guard (torch.cuda.is_available()) at
        # collection time initializes CUDA in the pytest process. vLLM's engine
        # core subprocess uses the fork start method by default, and CUDA
        # cannot re-initialize in a forked child ("Cannot re-initialize CUDA in
        # forked subprocess"), so force spawn. setdefault keeps any explicit
        # user choice.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        from s3e.vlm.vllm import VLLMBackend

        # num_logprobs=None exercises the full-vocab path (max_logprobs=-1 /
        # logprobs=-1, the vLLM >= 0.11.0 feature the pyproject pin exists for)
        # and makes the token-string assertions below deterministic: the full
        # vocabulary necessarily contains the tokens they look for.
        # Prefix caching is off so repeated queries with the same prompt (the
        # parity test) recompute their logits identically.
        return VLLMBackend(
            self.SMALL_VLM_ID,
            tensor_parallel_size=1,
            num_logprobs=None,
            gpu_memory_utilization=0.3,
            max_model_len=4096,
            enforce_eager=True,
            enable_prefix_caching=False,
        )

    def test_loads_and_queries_logprobs(self, backend):
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        result = backend.query([img], "Is this a test?")

        assert isinstance(result, VLMOutput)
        assert isinstance(result.token_probs, dict)
        assert len(result.token_probs) > 2  # full-vocab distribution
        assert all(p >= 0 for p in result.token_probs.values())

        # Token-string parity with HuggingFaceVLM: keys must be decoded text
        # (e.g. "Yes"), not raw tokenizer symbols ("▁Yes" / "ĠYes"). The
        # estimator matches token_probs keys against plain strings like
        # "Yes"/"true", so a raw-symbol format would silently zero out the
        # true/false token masses instead of failing loudly — catch it here.
        assert "Yes" in result.token_probs
        assert not any(key.startswith(("▁", "Ġ")) for key in result.token_probs)

    def test_interest_tokens_parity_with_full_vocab(self, backend):
        """Interest-mode masses must equal the full-vocabulary path's."""
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        prompt = "Is this a test?"
        interest = ["Yes", "No", "yes", "no"]

        full = backend.query([img], prompt)
        gathered = backend.query([img], prompt, interest_tokens=interest)

        assert set(gathered.token_probs) == set(interest)
        for token in interest:
            assert gathered.token_probs[token] == pytest.approx(
                full.token_probs.get(token, 0.0), abs=1e-9
            )
        assert gathered.argmax_in_interest == (
            max(full.token_probs, key=full.token_probs.get) in set(interest)
        )
