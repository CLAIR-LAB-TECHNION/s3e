"""Tests for the token reverse-index helper."""

from s3e.vlm.token_index import build_token_reverse_index


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
