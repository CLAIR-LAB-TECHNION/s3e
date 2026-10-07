# SPDX-FileCopyrightText: CLAIR Lab Technion
# SPDX-License-Identifier: MIT

"""Tests for the backend-agnostic thinking-block helpers."""

import pytest

import warnings

from s3e.backends.utils.thinking import (
    KNOWN_THINKING_DELIMITERS,
    MAX_SEPARATOR_TOKENS,
    ThinkingTokens,
    leading_whitespace_tokens,
    resolve_thinking_tokens,
    validate_thinking_options,
)

START, END = 7, 8
TOKENS = ThinkingTokens("<think>", "</think>", START, END)
THINK_PAIR = (("<think>", "</think>"),)


class FakeTokenizer:
    """Encodes known strings to fixed ids; everything else to two ids."""

    def __init__(self, vocab):
        self.vocab = dict(vocab)
        self.ids = {token_id: text for text, token_id in self.vocab.items()}

    def encode(self, text, add_special_tokens=False):
        return [self.vocab[text]] if text in self.vocab else [1, 2]

    def decode(self, ids):
        return "".join(self.ids[token_id] for token_id in ids)


class TestValidateThinkingOptions:
    def test_auto_tries_the_known_pairs(self):
        assert (
            validate_thinking_options("close", 1024, "auto") == KNOWN_THINKING_DELIMITERS
        )

    def test_a_given_pair_is_the_only_candidate(self):
        assert validate_thinking_options("close", 1024, ["<t>", "</t>"]) == (
            ("<t>", "</t>"),
        )

    def test_none_delimiters_turn_handling_off(self):
        assert validate_thinking_options("think", 1, None) is None

    @pytest.mark.parametrize("mode", ["skip", "", None, "Close"])
    def test_unknown_mode_rejected(self, mode):
        with pytest.raises(ValueError, match="thinking must be one of"):
            validate_thinking_options(mode, 1024, "auto")

    @pytest.mark.parametrize("budget", [0, -1, 1.5, True, None])
    def test_invalid_budget_rejected(self, budget):
        with pytest.raises(ValueError, match="max_thinking_tokens"):
            validate_thinking_options("close", budget, "auto")

    @pytest.mark.parametrize(
        "delimiters",
        [
            "<think>",
            "Auto",
            ("<think>",),
            ("<think>", "</think>", "x"),
            ("a", "a"),
            ("", "b"),
            {"<a>": 1, "<b>": 2},
            {"<a>", "<b>"},
        ],
    )
    def test_invalid_delimiters_rejected(self, delimiters):
        with pytest.raises(ValueError, match="thinking_delimiters"):
            validate_thinking_options("close", 1024, delimiters)


class TestResolveThinkingTokens:
    def test_single_token_delimiters_resolve(self):
        tokenizer = FakeTokenizer({"<think>": START, "</think>": END})
        assert resolve_thinking_tokens(THINK_PAIR, tokenizer) == TOKENS

    def test_multi_token_delimiter_turns_handling_off(self):
        tokenizer = FakeTokenizer({"<think>": START})
        with pytest.warns(UserWarning, match="not single tokens"):
            assert resolve_thinking_tokens(THINK_PAIR, tokenizer) is None

    def test_delimiter_that_does_not_round_trip_turns_handling_off(self):
        tokenizer = FakeTokenizer({"<think>": START, "</think>": END})
        tokenizer.ids[END] = "<unk>"
        with pytest.warns(UserWarning, match="not single tokens"):
            assert resolve_thinking_tokens(THINK_PAIR, tokenizer) is None

    def test_none_delimiters_or_tokenizer(self):
        tokenizer = FakeTokenizer({"<think>": START, "</think>": END})
        assert resolve_thinking_tokens(None, tokenizer) is None
        with pytest.warns(UserWarning, match="not single tokens"):
            assert resolve_thinking_tokens(THINK_PAIR, None) is None

    def test_auto_without_a_usable_pair_is_silent(self):
        """Most models have no thinking delimiters; "auto" says nothing."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert resolve_thinking_tokens(
                KNOWN_THINKING_DELIMITERS, FakeTokenizer({"<think>": START})
            ) is None

    @pytest.mark.parametrize("start, end", KNOWN_THINKING_DELIMITERS)
    def test_auto_finds_each_known_family(self, start, end):
        tokenizer = FakeTokenizer({start: 100, end: 101})
        assert resolve_thinking_tokens(KNOWN_THINKING_DELIMITERS, tokenizer) == (
            ThinkingTokens(start, end, 100, 101)
        )

    def test_auto_uses_the_first_usable_pair(self):
        tokenizer = FakeTokenizer(
            {"[THINK]": 34, "[/THINK]": 35, "<think>": START, "</think>": END}
        )
        assert resolve_thinking_tokens(KNOWN_THINKING_DELIMITERS, tokenizer) == TOKENS

    def test_auto_skips_a_half_present_pair(self):
        tokenizer = FakeTokenizer({"<think>": START, "[THINK]": 34, "[/THINK]": 35})
        assert resolve_thinking_tokens(KNOWN_THINKING_DELIMITERS, tokenizer) == (
            ThinkingTokens("[THINK]", "[/THINK]", 34, 35)
        )

    def test_tokenizer_errors_turn_handling_off(self):
        class Broken:
            def encode(self, text, add_special_tokens=False):
                raise TypeError("no encode")

        assert resolve_thinking_tokens(KNOWN_THINKING_DELIMITERS, Broken()) is None


class TestOpensBlock:
    VOCAB = {1: "q", 2: "x", 3: "\n", 9: "<|im_end|>", START: "<think>", END: "</think>"}

    def _decode(self, ids):
        return "".join(self.VOCAB[token_id] for token_id in ids)

    @pytest.mark.parametrize(
        "ids, expected",
        [
            ([1, 2], False),
            ([1, START], True),
            ([1, START, 3], True),
            ([1, START, END], False),
            ([1, START, END, 3], False),
            ([START, END, START], True),
            ([END], False),
            ([], False),
            # A delimiter the user's text mentions, followed by more text or
            # by the template's turn markers, opens nothing.
            ([1, START, 2, 3], False),
            ([1, START, 9], False),
            ([1, START] + [3] * (MAX_SEPARATOR_TOKENS + 1), False),
        ],
    )
    def test_open_block_detection(self, ids, expected):
        assert TOKENS.opens_block(ids, self._decode) is expected

    def test_closed_block_needs_no_decode(self):
        """The empty block of a thinking-off template is read from the ids."""

        def decode(ids):
            raise AssertionError(f"decoded {ids}")

        assert TOKENS.opens_block([1, START, 3, END, 3], decode) is False


class TestClosingText:
    def test_open_prompt_mirrors_trailing_whitespace(self):
        prompt = "<|im_start|>assistant\n<think>\n"
        assert prompt + TOKENS.closing_text(prompt, prompt_open=True) == (
            "<|im_start|>assistant\n<think>\n\n</think>"
        )

    def test_open_prompt_without_whitespace(self):
        assert TOKENS.closing_text("x<think>", prompt_open=True) == "</think>"

    def test_closed_prompt_gets_a_whole_empty_block(self):
        assert TOKENS.closing_text("x", prompt_open=False) == "<think>\n\n</think>"


class TestSplit:
    def test_no_block(self):
        assert TOKENS.split([3, 4], prompt_open=False) == (None, [3, 4])

    def test_block_opened_by_prompt(self):
        assert TOKENS.split([6, 6, END, 3, 4], prompt_open=True) == ([6, 6], [3, 4])

    def test_block_opened_by_model(self):
        assert TOKENS.split([START, 6, END, 4], prompt_open=False) == ([6], [4])

    def test_unclosed_block(self):
        assert TOKENS.split([6, 6], prompt_open=True) == ([6, 6], None)
        assert TOKENS.split([START, 6], prompt_open=False) == ([6], None)

    def test_close_without_open_is_not_a_block(self):
        assert TOKENS.split([6, END, 4], prompt_open=False) == (None, [6, END, 4])


class TestLeadingWhitespaceTokens:
    @pytest.mark.parametrize(
        "steps, expected",
        [
            (["Yes"], 0),
            (["\n\n", "Yes"], 1),
            (["\n", " ", "No"], 2),
            (["\n\n", "\n"], 2),
            ([], 0),
        ],
    )
    def test_counts_the_separator(self, steps, expected):
        assert leading_whitespace_tokens(steps) == expected
