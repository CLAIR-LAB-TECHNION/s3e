"""Tests for the OpenAI VLM backend (client mocked; no API calls)."""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

pytest.importorskip("openai", reason="openai not installed (s3e[openai])")

from s3e.backends.backend import VLMOutput
from s3e.backends.openai import OpenAIVLM


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

    @patch("s3e.backends.openai.openai")
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

    @patch("s3e.backends.openai.openai")
    def test_strips_openai_prefix(self, mock_openai_module):
        mock_client = MagicMock()
        mock_openai_module.OpenAI.return_value = mock_client

        vlm = OpenAIVLM("OpenAI/gpt-4o")
        assert vlm.model_id == "gpt-4o"

    @patch("s3e.backends.openai.openai")
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

    @patch("s3e.backends.openai.openai")
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

    @patch("s3e.backends.openai.openai")
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

    @patch("s3e.backends.openai.openai")
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

    @patch("s3e.backends.openai.openai")
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

    @patch("s3e.backends.openai.openai")
    def test_generate_mode_skips_logprobs(self, mock_openai_module):
        """Generate mode must not request logprobs (reasoning models reject
        them) and returns text only, like the other backends."""
        mock_client = MagicMock()
        mock_openai_module.OpenAI.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = "yes, it is"
        mock_choice.logprobs = None
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        vlm = OpenAIVLM("gpt-4o")
        img = Image.new("RGB", (64, 64))
        result = vlm.query([img], "Is A on B?", generate=True)

        request_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "logprobs" not in request_kwargs
        assert "top_logprobs" not in request_kwargs
        assert result.token_probs is None
        assert result.text == "yes, it is"
        assert result.argmax_in_interest is None

    @patch("s3e.backends.openai.openai")
    def test_missing_logprobs_raises_informative_error(self, mock_openai_module):
        """Models that return no logprobs must fail with guidance, not an
        opaque AttributeError."""
        mock_client = MagicMock()
        mock_openai_module.OpenAI.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = "yes"
        mock_choice.logprobs = None
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        vlm = OpenAIVLM("gpt-4o")
        img = Image.new("RGB", (64, 64))

        with pytest.raises(ValueError, match="logprobs.*text_match"):
            vlm.query([img], "Is A on B?")

    @patch("s3e.backends.openai.openai")
    def test_empty_logprobs_content_raises_informative_error(
        self, mock_openai_module
    ):
        mock_client = MagicMock()
        mock_openai_module.OpenAI.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = "yes"
        mock_choice.logprobs.content = []
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        vlm = OpenAIVLM("gpt-4o")
        img = Image.new("RGB", (64, 64))

        with pytest.raises(ValueError, match="logprobs.*text_match"):
            vlm.query([img], "Is A on B?")
