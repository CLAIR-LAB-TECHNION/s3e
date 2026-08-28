"""Tests for the token reverse-index helper."""

from unittest.mock import MagicMock

from s3e.backends.token_index import (
    build_token_reverse_index,
    decode_single_token_ids,
)


class TestBuildTokenReverseIndex:
    def test_maps_each_string_to_its_ids(self):
        decoded = ["a", "b", "c"]
        index = build_token_reverse_index(
            lambda ids: [decoded[i] for i in ids], len(decoded)
        )
        assert index == {"a": [0], "b": [1], "c": [2]}

    def test_groups_duplicate_strings_in_id_order(self):
        decoded = ["yes", "no", "yes", "yes"]
        index = build_token_reverse_index(
            lambda ids: [decoded[i] for i in ids], len(decoded)
        )
        assert index == {"yes": [0, 2, 3], "no": [1]}

    def test_decode_called_once_with_all_ids(self):
        calls = []

        def decode(ids):
            calls.append(list(ids))
            return [f"tok{i}" for i in ids]

        build_token_reverse_index(decode, 5)
        assert calls == [[0, 1, 2, 3, 4]]

    def test_zero_vocab_returns_empty_index(self):
        assert build_token_reverse_index(lambda ids: [], 0) == {}


class TestDecodeSingleTokenIds:
    def test_prefers_batch_decode_with_one_id_per_sequence(self):
        tokenizer = MagicMock()
        tokenizer.batch_decode.return_value = ["a", "b"]

        assert decode_single_token_ids(tokenizer, [3, 7]) == ["a", "b"]
        sequences = tokenizer.batch_decode.call_args.args[0]
        assert sequences == [[3], [7]]
        assert tokenizer.batch_decode.call_args.kwargs["skip_special_tokens"] is False

    def test_falls_back_to_per_id_decode_when_batch_decode_fails(self):
        tokenizer = MagicMock()
        tokenizer.batch_decode.side_effect = TypeError("no batch decode")
        tokenizer.decode.side_effect = lambda ids, **kwargs: f"tok{ids[0]}"

        assert decode_single_token_ids(tokenizer, [1, 2]) == ["tok1", "tok2"]

    def test_empty_ids(self):
        assert decode_single_token_ids(MagicMock(), []) == []
