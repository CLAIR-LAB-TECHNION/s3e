"""Tests for VLM backends."""

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
            mock_output = MagicMock()
            mock_output.logits = logits
            mock_model.return_value = mock_output

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
    def test_query_batch_uses_attention_mask_for_last_prompt_token(
        self, mock_model_cls, mock_proc_cls
    ):
        logits = torch.zeros((2, 4, 4), dtype=torch.float32)
        logits[0, 2, 2] = 10.0
        logits[0, 3, 0] = 20.0
        logits[1, 3, 1] = 10.0
        vlm, _, mock_processor = self._make_mock_hf_components(
            mock_model_cls,
            mock_proc_cls,
            logits=logits,
            input_ids=torch.ones(2, 4, dtype=torch.long),
            vlm_kwargs={"num_logprobs": 1},
        )
        mock_processor.return_value = {
            "input_ids": torch.ones(2, 4, dtype=torch.long),
            "attention_mask": torch.tensor(
                [
                    [1, 1, 1, 0],
                    [1, 1, 1, 1],
                ],
                dtype=torch.long,
            ),
        }

        results = vlm.query_batch([], ["short", "long"])

        assert set(results[0].token_probs) == {"tok2"}
        assert set(results[1].token_probs) == {"tok1"}

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
