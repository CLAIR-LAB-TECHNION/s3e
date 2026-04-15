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

    @patch("s3e.vlm.huggingface.AutoProcessor")
    @patch("s3e.vlm.huggingface._AutoModelClass")
    def test_construction(self, mock_model_cls, mock_proc_cls):
        from s3e.vlm.huggingface import HuggingFaceVLM

        vlm = HuggingFaceVLM("test/model")
        mock_model_cls.from_pretrained.assert_called_once()
        mock_proc_cls.from_pretrained.assert_called_once()
        assert vlm.max_new_tokens == 10

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

        vlm = HuggingFaceVLM(self.TINY_VLM_ID, device_map="cpu")
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        results = vlm.query_batch([img], ["q1?", "q2?"])

        assert len(results) == 2
        assert all(isinstance(r, VLMOutput) for r in results)
