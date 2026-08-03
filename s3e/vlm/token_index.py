"""Reverse token index: decoded string -> all vocabulary ids producing it.

Backends that can address their probability tensors by token id use this to
resolve caller-supplied token strings (``interest_tokens``) into id lists
once, instead of detokenizing the whole vocabulary on every query. Summing
mass over every id that decodes to the same string reproduces exactly the
duplicate-string aggregation the full-vocabulary string path performs.
"""

from collections.abc import Callable


def build_token_reverse_index(
    decode_ids: Callable[[list[int]], list[str]],
    vocab_size: int,
) -> dict[str, list[int]]:
    """Decode the whole vocabulary once and invert it.

    Args:
        decode_ids: Maps a list of token ids to their decoded strings,
            one string per id (e.g. ``HuggingFaceVLM._decode_token_ids``).
        vocab_size: Number of ids to decode, ``range(vocab_size)``.

    Returns:
        Mapping from decoded string to the ascending list of ids that
        decode to it.
    """
    decoded = decode_ids(list(range(vocab_size)))
    index: dict[str, list[int]] = {}
    for token_id, token_str in enumerate(decoded):
        index.setdefault(token_str, []).append(token_id)
    return index
