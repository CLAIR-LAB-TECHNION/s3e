# SPDX-FileCopyrightText: CLAIR Lab Technion
# SPDX-License-Identifier: MIT

"""Thinking blocks: detecting and skipping a model's reasoning.

Thinking models wrap their reasoning in a pair of delimiter tokens, most
often ``<think>`` and ``</think>`` (Qwen3 and Qwen3.5, DeepSeek-R1, GLM,
MiniMax, ...). Checkpoints do not declare the pair in a standard way, so
backends try the pairs of known model families (:data:`KNOWN_THINKING_DELIMITERS`)
or one the caller names, and switch this logic on only when both delimiters
are single tokens of the model's vocabulary. A model without them (SmolVLM,
LLaVA, ...) behaves exactly as if this module did not exist.

Backends take a binary ``thinking`` choice. Many chat templates have a
variable that turns thinking on or off (``enable_thinking`` for Qwen3.5 and
Gemma 4); backends find it among :data:`KNOWN_THINKING_SWITCHES` by
rendering a probe conversation (:func:`find_thinking_switch`) and set it.
Where no switch exists, a block can still appear: the chat template opens it
(Qwen3-VL-*-Thinking, DeepSeek-R1 distills), or the model does. A query is
*thinking* when its prompt ends with an opening delimiter or the model's
most likely next token opens one. With thinking off, backends then close
the block at once and read the answer after the empty block; with thinking
on, the model reasons first and the answer is read after its reasoning. The
answer is always read from a forward pass over the prompt, the closed
block, and the whitespace the model puts after it, like the pass that
scores every other query: reading it after a decode step instead can differ
numerically, since some engines (e.g. vLLM with Qwen3.5's linear-attention
layers) use different kernels for prefill and decode.
"""

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass

#: Delimiter pairs of known model families, tried in order by
#: ``thinking_delimiters="auto"``; the first pair whose two delimiters are
#: single tokens of the model's vocabulary is used.
KNOWN_THINKING_DELIMITERS = (
    ("<think>", "</think>"),  # Qwen3/3.5, DeepSeek-R1, GLM, MiniMax, ...
    ("<|channel>", "<channel|>"),  # Gemma 4
    ("[THINK]", "[/THINK]"),  # Mistral reasoning models (HF tokenizer files)
    ("<seed:think>", "</seed:think>"),  # Seed
)
#: Chat-template variables known to switch thinking, tried in order as
#: ``(name, value turning thinking on, value turning it off)``. Templates
#: ignore variables they do not read, so a backend uses the first one whose
#: two values render a probe conversation differently.
KNOWN_THINKING_SWITCHES = (
    ("enable_thinking", True, False),  # Qwen3/3.5, Gemma 4, GLM, ...
    ("thinking", True, False),  # DeepSeek-V3.1, Granite 3
    ("thinking_budget", -1, 0),  # Seed-OSS (-1: no limit)
)
DEFAULT_MAX_THINKING_TOKENS = 1024
# Most whitespace tokens a chat template puts next to a delimiter: after the
# opening one that ends a prompt ("<think>\n"), or between a closed block and
# the answer ("\n\n" after Qwen's "</think>"), which this many greedy steps
# look for.
MAX_SEPARATOR_TOKENS = 8


def validate_thinking_options(
    thinking: bool,
    max_thinking_tokens: int,
    thinking_delimiters: "tuple[str, str] | str | None",
) -> "tuple[tuple[str, str], ...] | None":
    """Validate a backend's thinking parameters.

    Returns the delimiter pairs to try, in order: the known pairs for
    ``"auto"``, the one given pair, or None when handling is off.
    """
    if not isinstance(thinking, bool):
        raise ValueError(f"thinking must be True or False; got {thinking!r}")
    if (
        isinstance(max_thinking_tokens, bool)
        or not isinstance(max_thinking_tokens, int)
        or max_thinking_tokens < 1
    ):
        raise ValueError(
            "max_thinking_tokens must be a positive int; "
            f"got {max_thinking_tokens!r}"
        )
    if thinking_delimiters is None:
        if thinking:
            raise ValueError(
                "thinking=True needs thinking_delimiters to find the answer "
                "after the model's reasoning; got None"
            )
        return None
    if thinking_delimiters == "auto":
        return KNOWN_THINKING_DELIMITERS
    if (
        not isinstance(thinking_delimiters, (tuple, list))
        or len(thinking_delimiters) != 2
        or not all(isinstance(d, str) and d for d in thinking_delimiters)
        or thinking_delimiters[0] == thinking_delimiters[1]
    ):
        raise ValueError(
            "thinking_delimiters must be \"auto\", None, or two distinct "
            f"non-empty strings; got {thinking_delimiters!r}"
        )
    return (tuple(thinking_delimiters),)


@dataclass(frozen=True)
class ThinkingTokens:
    """A delimiter pair resolved to single token ids of one tokenizer."""

    start: str
    end: str
    start_id: int
    end_id: int

    def opens_block(
        self, token_ids: Sequence[int], decode: Callable[[list[int]], str]
    ) -> bool:
        """Whether ``token_ids`` end with an opening delimiter.

        This is how a chat template opens the block for the model (e.g.
        Qwen3.5's ``"<think>\\n"``): only whitespace, checked with
        ``decode`` (token ids to text, special tokens kept), may follow the
        last opening delimiter. A delimiter earlier in the prompt, such as
        one the user's text mentions, does not open a block.
        """
        ids = list(token_ids)
        # list.index on the reversed ids finds the last opening delimiter in
        # C, which matters for long multimodal prompts.
        reversed_ids = ids[::-1]
        if self.start_id not in reversed_ids:
            return False
        after = reversed_ids.index(self.start_id)
        if after == 0:
            return True
        tail = ids[-after:]
        # A closing delimiter is not whitespace, so finding it spares the
        # decode for the empty block thinking-off templates render.
        if after > MAX_SEPARATOR_TOKENS or self.end_id in tail:
            return False
        return not decode(tail).strip()

    def closing_text(self, prompt_text: str, prompt_open: bool) -> str:
        """Text that closes the block at once, leaving it empty.

        Repeats the whitespace that follows an opening delimiter at the end
        of the prompt, so ``"<think>\\n"`` becomes ``"<think>\\n\\n</think>"``,
        the empty block chat templates render when thinking is off. When the
        model was about to open the block itself, the whole empty block is
        appended in that same form.
        """
        if prompt_open:
            return prompt_text[len(prompt_text.rstrip()):] + self.end
        return f"{self.start}\n\n{self.end}"

    def split(
        self, generated_ids: Sequence[int], prompt_open: bool
    ) -> "tuple[list[int] | None, list[int] | None]":
        """Split generated ids into ``(reasoning, answer)`` id lists.

        ``reasoning`` is None when no block was involved (the prompt did not
        open one and the generation does not start with one). ``answer`` is
        None when the block never closed.
        """
        ids = [int(token_id) for token_id in generated_ids]
        started = bool(ids) and ids[0] == self.start_id
        if not (prompt_open or started):
            return None, ids
        body = ids[1:] if started else ids
        if self.end_id not in body:
            return body, None
        cut = body.index(self.end_id)
        return body[:cut], body[cut + 1:]


def resolve_thinking_tokens(
    candidates: "Sequence[tuple[str, str]] | None", tokenizer
) -> "ThinkingTokens | None":
    """Resolve the first usable delimiter pair against ``tokenizer``.

    A pair is usable when both delimiters encode to a single token that
    decodes back to the delimiter. Returns None (thinking support off) when
    no pair is, with a warning unless the candidates are the known pairs
    tried by ``"auto"``.
    """
    if candidates is None:
        return None
    if tokenizer is not None:
        for delimiters in candidates:
            tokens = _resolve_pair(delimiters, tokenizer)
            if tokens is not None:
                return tokens
    if tuple(candidates) != KNOWN_THINKING_DELIMITERS:
        warnings.warn(
            f"Thinking delimiters {list(candidates)} are not single tokens of "
            "the model's vocabulary, so thinking handling is off.",
            stacklevel=2,
        )
    return None


def _resolve_pair(delimiters: "tuple[str, str]", tokenizer) -> "ThinkingTokens | None":
    """Resolve one delimiter pair, or None unless both are single tokens."""
    token_ids = []
    for delimiter in delimiters:
        try:
            encoded = list(tokenizer.encode(delimiter, add_special_tokens=False))
            if len(encoded) != 1:
                return None
            token_id = int(encoded[0])
            if tokenizer.decode([token_id]) != delimiter:
                return None
        except Exception:
            return None
        token_ids.append(token_id)
    return ThinkingTokens(delimiters[0], delimiters[1], token_ids[0], token_ids[1])


@dataclass(frozen=True)
class ThinkingSwitch:
    """A chat-template variable that turns thinking on or off."""

    name: str
    on: object
    off: object

    def template_kwargs(self, thinking: bool) -> dict:
        """The template variable set to turn thinking on or off."""
        return {self.name: self.on if thinking else self.off}


def find_thinking_switch(render: Callable[[dict], str]) -> "ThinkingSwitch | None":
    """The first of :data:`KNOWN_THINKING_SWITCHES` this chat template reads.

    ``render`` renders a probe conversation with the given template
    variables. A switch counts when its two values render differently; one
    whose rendering raises is skipped. Returns None when none counts.
    """
    for name, on, off in KNOWN_THINKING_SWITCHES:
        try:
            if render({name: on}) != render({name: off}):
                return ThinkingSwitch(name, on, off)
        except Exception:
            continue
    return None


def thinking_template_kwargs(
    thinking: bool,
    tokens: "ThinkingTokens | None",
    switch: "ThinkingSwitch | None",
    chat_template_kwargs: dict,
) -> dict:
    """The chat-template variables a backend renders queries with.

    The switch (if any) turns thinking on only when the block can be
    skipped (``tokens`` resolved), and off otherwise; the caller's own
    ``chat_template_kwargs`` override it.
    """
    switched = {} if switch is None else switch.template_kwargs(
        thinking and tokens is not None
    )
    return {**switched, **chat_template_kwargs}


def warn_if_thinking_unavailable(
    model_id: str,
    thinking: bool,
    tokens: "ThinkingTokens | None",
    switch: "ThinkingSwitch | None",
    template_kwargs: dict,
    rendered_probe: "str | None",
) -> None:
    """Warn when ``thinking=True`` cannot make this model think.

    Either the block cannot be skipped (no delimiters resolved), or nothing
    turns thinking on: no switch set to on, and the probe rendered with
    ``template_kwargs`` does not end by opening a block.
    """
    if not thinking:
        return
    if tokens is None:
        warnings.warn(
            f"thinking=True has no effect for {model_id}: none of the "
            "thinking delimiters is a single token of its vocabulary, so the "
            "answer after its reasoning cannot be found. It answers without "
            "thinking.",
            stacklevel=3,
        )
        return
    switched_on = switch is not None and template_kwargs.get(switch.name) == switch.on
    template_opens = rendered_probe is not None and rendered_probe.rstrip().endswith(
        tokens.start
    )
    if not (switched_on or template_opens):
        warnings.warn(
            f"thinking=True found no way to turn thinking on for {model_id}: "
            "its chat template has no known thinking switch set to on and "
            "does not open a thinking block. It thinks only where it opens "
            "a block itself.",
            stacklevel=3,
        )


def leading_whitespace_tokens(step_tokens: Sequence[str]) -> int:
    """How many generated tokens come before the first non-whitespace one.

    Backends append these tokens (the separator a chat template puts after a
    closed block) to the continuation before scoring the answer token.
    """
    for step, token in enumerate(step_tokens):
        if token.strip():
            return step
    return len(step_tokens)
