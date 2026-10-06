# SPDX-FileCopyrightText: CLAIR Lab Technion
# SPDX-License-Identifier: MIT

"""vLLM VLM backend.

A :class:`VLMBackend` implementation that runs a local HuggingFace
vision-language model through `vLLM <https://docs.vllm.ai/>`_ for transparent,
high-throughput, single-node multi-GPU inference. Construct
``VLLMBackend(...)`` explicitly and pass it as the ``vlm`` argument to
:class:`QueryEngine` or :class:`SemanticStateEstimator`.

Design notes (kept in code on purpose so future maintainers can change the
design with full context):

* The module is named ``vllm.py`` to match the existing backend naming
  convention (``openai.py`` imports ``openai``; ``huggingface.py`` uses
  ``transformers``). Python 3 uses absolute imports, so ``from vllm import ...``
  below resolves to the installed *package*, not this module -- the same safe
  pattern ``openai.py`` already relies on.
* ``vllm`` is an optional dependency; this module is imported lazily (from
  ``s3e.backends``) so importing s3e never requires it. Importing this module
  directly raises a helpful :class:`ImportError` when it is missing.
* Requires vLLM >= 0.11.0: full-vocab logprobs via ``SamplingParams(logprobs=-1)``
  and ``LLM(max_logprobs=-1)`` landed in 0.11.0 (vllm-project/vllm#25031); older
  engines reject ``-1`` at construction. The pin lives in ``pyproject.toml``.
* Thinking models (see :mod:`s3e.backends.thinking`): rows whose answer would
  come after a thinking block are re-run through ``LLM.generate`` on a text
  prompt. ``LLM.chat`` cannot continue a conversation from text we append, and
  re-sending the token ids it returns does not work either: they hold the
  *expanded* image placeholders, which vLLM would expand a second time. The
  text prompt is ``RequestOutput.prompt``, the conversation exactly as
  ``LLM.chat`` rendered it (placeholders unexpanded), plus the closed block,
  with the images as ``multi_modal_data``, vLLM's documented offline
  multimodal input. Re-rendering with the tokenizer's chat template is only
  a fallback for outputs without that text: vLLM's chat renderer differs
  from it (e.g. it puts a newline after Qwen's image placeholder), and the
  answer then shifts. Since the engine tokenizes the text again, the
  continuation's token ids are checked against those ``LLM.chat`` scored,
  and a mismatch raises instead of reading the answer from another prompt:
  older vLLM versions return no prompt text, or text holding the expanded
  placeholders, and a text-only engine's tokenizer may add a second BOS
  token to a text prompt. Only thinking rows pay for the extra calls; other
  models never leave the ``LLM.chat`` path.
"""

import math

from .._deps import require

require("vllm", "vllm", "VLLMBackend")

import torch
from vllm import LLM, SamplingParams

from .backend import VLMBackend, VLMOutput, _validate_num_logprobs
from .thinking import (
    DEFAULT_MAX_THINKING_TOKENS,
    MAX_SEPARATOR_TOKENS,
    leading_whitespace_tokens,
    resolve_thinking_tokens,
    validate_thinking_options,
)
from .token_index import build_token_reverse_index, decode_single_token_ids


class VLLMBackend(VLMBackend):
    """VLM backend backed by a local vLLM engine.

    The estimator always sends the *same* images with *many* prompts (one per
    grounded predicate); vLLM batches them in a single ``chat`` call and shards
    each layer across GPUs under the hood, so the user can treat a multi-GPU box
    as one big GPU.

    Args:
        model_id: HuggingFace model identifier loaded by vLLM.
        tensor_parallel_size: Number of GPUs to shard each layer across. When
            ``None`` (default), uses all locally visible GPUs
            (``torch.cuda.device_count()``), so the user need not specify a
            count. Multi-node execution is intentionally out of scope but can be
            reached by passing vLLM engine kwargs (e.g. ``pipeline_parallel_size``,
            ``distributed_executor_backend``) through ``engine_kwargs``.
        num_logprobs: Number of top tokens to include in ``token_probs``.
            ``None`` (default) returns the full-vocabulary distribution,
            mirroring :class:`HuggingFaceVLM`. This maps to vLLM ``logprobs=-1``
            with the engine built using ``max_logprobs=-1`` -- note the docs' OOM
            caveat for full-vocab logprobs. A finite ``k`` returns the top ``k``.
            With ``interest_tokens`` queries this stays the exactness knob:
            ``None`` keeps the id-matched masses exact over the full
            vocabulary, while a finite ``k`` bounds the omitted interest
            mass by the ``k``-th ranked probability. Interest mode also requests
            ``detokenize=False``; whether the installed vLLM honors it for
            logprobs can be smoke-tested by checking that returned
            ``Logprob.decoded_token`` values are ``None``.
        chat_template_kwargs: Extra variables for the chat template, e.g.
            ``{"enable_thinking": False}`` to turn a thinking model's
            reasoning off. Defaults to None.
        thinking: What logprobs-mode queries do when the model starts a
            thinking block: ``"close"`` (default) closes it at once and
            scores the answer after the empty block; ``"think"`` generates
            the reasoning first (up to ``max_thinking_tokens``) and scores
            the answer after it. Text mode always generates freely and moves
            the block to :attr:`VLMOutput.reasoning`.
        max_thinking_tokens: Reasoning budget for ``thinking="think"``; a
            block still open after this many tokens is closed and the answer
            scored. The prompt plus this budget must fit in the engine's
            ``max_model_len``. Defaults to 1024.
        thinking_delimiters: The tokens that open and close a thinking
            block, active only when both are single tokens of the model's
            vocabulary. ``"auto"`` (default) uses the first such pair among
            those of known model families: ``<think>``/``</think>`` (Qwen3,
            DeepSeek-R1, GLM, ...), ``<|channel>``/``<channel|>`` (Gemma 4),
            ``[THINK]``/``[/THINK]`` (Mistral, with HF tokenizer files), and
            ``<seed:think>``/``</seed:think>``. Pass a pair to use only that
            one, or ``None`` to turn thinking handling off.
        **engine_kwargs: Forwarded verbatim to the ``vllm.LLM`` constructor
            (e.g. ``gpu_memory_utilization``, ``dtype``, ``max_model_len``,
            ``quantization``), mirroring how :class:`HuggingFaceVLM` forwards
            ``**model_kwargs``.

    Notes:
        There is intentionally no ``max_new_tokens`` parameter (the HF backend
        makes the same choice): generation length is controlled per call
        through ``inference_kwargs`` (vLLM ``SamplingParams.max_tokens``).

        This backend relies on the model's chat template (see
        :meth:`query_batch`); a model without one is unsupported on this path.
    """

    def __init__(
        self,
        model_id: str,
        tensor_parallel_size: int | None = None,
        num_logprobs: int | None = None,
        chat_template_kwargs: dict | None = None,
        thinking: str = "close",
        max_thinking_tokens: int = DEFAULT_MAX_THINKING_TOKENS,
        thinking_delimiters: tuple[str, str] | str | None = "auto",
        **engine_kwargs,
    ):
        self.model_id = model_id
        self.num_logprobs = _validate_num_logprobs(num_logprobs)
        self.thinking_delimiters = validate_thinking_options(
            thinking, max_thinking_tokens, thinking_delimiters
        )
        self.thinking = thinking
        self.max_thinking_tokens = max_thinking_tokens
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self._token_reverse_index: dict[str, list[int]] | None = None
        self._thinking_tokens = None
        self._thinking_tokens_resolved = False
        self._tokenizer = None

        # Default to every locally visible GPU so the user does not have to
        # specify a count; an explicit value always wins. max(..., 1) keeps a
        # sane value when no CUDA device is visible.
        if tensor_parallel_size is None:
            tensor_parallel_size = max(torch.cuda.device_count(), 1)
        elif (
            isinstance(tensor_parallel_size, bool)
            or not isinstance(tensor_parallel_size, int)
            or tensor_parallel_size < 1
        ):
            raise ValueError(
                "tensor_parallel_size must be a positive int or None; "
                f"got {tensor_parallel_size!r}"
            )
        self.tensor_parallel_size = tensor_parallel_size

        # max_logprobs is fixed at engine-construction time, and num_logprobs is
        # also a constructor arg, so the two are always consistent. None means
        # "full vocab" (-1), matching HuggingFaceVLM's default; a finite k caps
        # at k. Requesting more logprobs per call than max_logprobs makes vLLM
        # raise, which is why we size the cap from num_logprobs here.
        self.llm = LLM(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            max_logprobs=(-1 if num_logprobs is None else num_logprobs),
            **engine_kwargs,
        )

    def query(
        self,
        images: list,
        prompt: str,
        system_prompt: str | None = None,
        generate: bool = False,
        interest_tokens=None,
        **inference_kwargs,
    ) -> VLMOutput:
        """Send a single query to the vLLM engine."""
        return self.query_batch(
            images,
            [prompt],
            system_prompt,
            generate,
            interest_tokens=interest_tokens,
            **inference_kwargs,
        )[0]

    def query_batch(
        self,
        images: list,
        prompts: list[str],
        system_prompt: str | None = None,
        generate: bool = False,
        interest_tokens=None,
        **inference_kwargs,
    ) -> list[VLMOutput]:
        """Send multiple prompts against the same images in one batched call.

        We use ``LLM.chat`` (not ``LLM.generate`` + ``multi_modal_data``) so
        vLLM applies the model's own chat template and performs the
        resolution-dependent image-placeholder expansion -- mirroring how
        :class:`HuggingFaceVLM` relies on ``processor.apply_chat_template`` and
        owning zero per-model placeholder logic. The trade-off: a model without
        a chat template is unsupported here. Rows that would answer after a
        thinking block are re-scored through ``LLM.generate`` (see the module
        notes).
        """
        if not prompts:
            return []

        interest_mode = interest_tokens is not None and not generate
        sampling_params = self._build_sampling_params(
            generate, interest_mode, **inference_kwargs
        )

        # Every prompt shares the same images, so build the image content once
        # and reuse it across the per-prompt conversations.
        image_content = [{"type": "image_pil", "image_pil": image} for image in images]
        conversations = []
        for prompt in prompts:
            messages = []
            if system_prompt is not None:
                messages.append({"role": "system", "content": system_prompt})
            messages.append(
                {
                    "role": "user",
                    "content": image_content + [{"type": "text", "text": prompt}],
                }
            )
            conversations.append(messages)

        # chat() accepts a batched list of conversations and returns outputs in
        # input order. Template kwargs are passed only when set, so engines
        # and templates that predate them see the same call as before.
        chat_kwargs = (
            {"chat_template_kwargs": self.chat_template_kwargs}
            if self.chat_template_kwargs
            else {}
        )
        outputs = self.llm.chat(conversations, sampling_params, **chat_kwargs)
        results = [
            self._to_vlm_output(output, generate, interest_tokens)
            for output in outputs
        ]
        if generate:
            return results
        return self._skip_thinking(
            images, prompts, system_prompt, outputs, results,
            interest_tokens, inference_kwargs,
        )

    def _get_tokenizer(self):
        """The engine's tokenizer, fetched once."""
        if self._tokenizer is None:
            self._tokenizer = self.llm.get_tokenizer()
        return self._tokenizer

    def _get_thinking_tokens(self):
        """The resolved thinking delimiters, or None when inactive (cached)."""
        if not self._thinking_tokens_resolved:
            self._thinking_tokens = resolve_thinking_tokens(
                self.thinking_delimiters, self._get_tokenizer()
            )
            self._thinking_tokens_resolved = True
        return self._thinking_tokens

    def _skip_thinking(
        self, images, prompts, system_prompt, outputs, results,
        interest_tokens, inference_kwargs,
    ):
        """Re-score the rows whose answer would come after a thinking block.

        A row thinks when its prompt ends with an opening delimiter or its
        most likely next token opens one. Those rows are re-run through
        ``LLM.generate`` on the rendered prompt plus the closed block (empty,
        or the generated reasoning), and read at the answer after it.
        """
        tokens = self._get_thinking_tokens()
        if tokens is None:
            return results

        rows, prompt_open = [], []
        for row, output in enumerate(outputs):
            is_open = tokens.opens_block(
                output.prompt_token_ids or [], self._decode_with_special_tokens
            )
            start_entry = output.outputs[0].logprobs[0].get(tokens.start_id)
            if is_open or (start_entry is not None and start_entry.rank == 1):
                rows.append(row)
                prompt_open.append(is_open)
        if not rows:
            return results

        prompt_texts = [
            outputs[row].prompt
            if isinstance(outputs[row].prompt, str)
            else self._render_prompt(images, prompts[row], system_prompt)
            for row in rows
        ]
        if self.thinking == "think":
            # Open the block where the model was about to, so the reasoning
            # is generated inside it even if generate() starts differently.
            continuations, reasonings = self._think(
                images,
                [
                    text if is_open else text + tokens.start
                    for text, is_open in zip(prompt_texts, prompt_open)
                ],
                tokens,
            )
        else:
            continuations = [
                text + tokens.closing_text(text, is_open)
                for text, is_open in zip(prompt_texts, prompt_open)
            ]
            reasonings = [""] * len(rows)

        answers = self._answer_outputs_after(
            images,
            continuations,
            [outputs[row].prompt_token_ids or [] for row in rows],
            interest_tokens,
            inference_kwargs,
        )
        results = list(results)
        for row, answer, reasoning in zip(rows, answers, reasonings):
            answer.reasoning = reasoning
            results[row] = answer
        return results

    def _render_prompt(self, images, prompt, system_prompt=None) -> str:
        """Render one conversation to text with the tokenizer's chat template.

        A fallback for outputs that lack ``LLM.chat``'s own rendering, which
        it may not match exactly (see the module notes). Images become
        ``{"type": "image"}`` parts, which multimodal chat templates turn
        into the model's placeholder text; ``LLM.generate`` expands the
        placeholders when the images come with the prompt.
        """
        messages = []
        if system_prompt is not None:
            # A plain string, as query_batch sends it: some templates render
            # the list form differently (Gemma 4 adds a space).
            messages.append({"role": "system", "content": system_prompt})
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})
        return self._get_tokenizer().apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            **self.chat_template_kwargs,
        )

    @staticmethod
    def _engine_prompt(text: str, images) -> dict:
        """A text prompt for ``LLM.generate``, with the scene's images."""
        if not images:
            return {"prompt": text}
        return {"prompt": text, "multi_modal_data": {"image": list(images)}}

    def _think(self, images, prompt_texts, tokens):
        """Generate each prompt's reasoning, up to ``max_thinking_tokens``.

        Every prompt ends inside an open block. Returns one continuation per
        prompt (the prompt plus its reasoning, ending with the closing
        delimiter, which is appended when the budget ran out or the model
        stopped first) and the reasoning texts.
        """
        tokenizer = self._get_tokenizer()
        sampling_params = SamplingParams(
            max_tokens=self.max_thinking_tokens,
            temperature=0.0,
            stop_token_ids=[tokens.end_id],
        )
        outputs = self.llm.generate(
            [self._engine_prompt(text, images) for text in prompt_texts],
            sampling_params,
        )
        special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])

        continuations, reasonings = [], []
        for text, output in zip(prompt_texts, outputs):
            completion = output.outputs[0]
            ids = [int(token_id) for token_id in completion.token_ids]
            if tokens.end_id in ids:
                ids = ids[: ids.index(tokens.end_id) + 1]
            elif completion.stop_reason == tokens.end_id:
                ids.append(tokens.end_id)
            else:
                while ids and ids[-1] in special_ids:
                    ids.pop()
            generated = tokenizer.decode(ids, skip_special_tokens=False)
            if not ids or ids[-1] != tokens.end_id:
                if not generated[-1:].isspace():
                    generated += "\n"
                generated += tokens.end
            reasoning_ids, _ = tokens.split(ids, prompt_open=True)
            continuations.append(text + generated)
            reasonings.append(
                tokenizer.decode(reasoning_ids or [], skip_special_tokens=True).strip()
            )
        return continuations, reasonings

    def _answer_outputs_after(
        self, images, continuations, chat_prompt_ids, interest_tokens,
        inference_kwargs,
    ) -> list[VLMOutput]:
        """Score the answer token after each continuation's closed block.

        Chat templates put whitespace between a closed block and the answer
        (``"\\n\\n"`` for Qwen). A few greedy steps find it, and the answer
        is then scored with the same one-token pass as every other query, on
        the continuation plus that whitespace. Reading it after a decode step
        instead would differ: vLLM's prefill and decode kernels disagree
        slightly for some architectures (e.g. Qwen3.5's linear attention).
        ``chat_prompt_ids`` are the prompt ids ``LLM.chat`` scored for each
        continuation, which it must extend.
        """
        tokenizer = self._get_tokenizer()
        separator_outputs = self.llm.generate(
            [self._engine_prompt(text, images) for text in continuations],
            SamplingParams(max_tokens=MAX_SEPARATOR_TOKENS, temperature=0.0),
        )
        for prompt_ids, output in zip(chat_prompt_ids, separator_outputs):
            self._check_continuation(prompt_ids, output.prompt_token_ids or [])
        texts = []
        for text, output in zip(continuations, separator_outputs):
            ids = [int(token_id) for token_id in output.outputs[0].token_ids]
            count = leading_whitespace_tokens(decode_single_token_ids(tokenizer, ids))
            texts.append(text + tokenizer.decode(ids[:count], skip_special_tokens=False))

        sampling_params = self._build_sampling_params(
            False, interest_tokens is not None, **inference_kwargs
        )
        outputs = self.llm.generate(
            [self._engine_prompt(text, images) for text in texts], sampling_params
        )
        return [
            self._to_vlm_output(output, False, interest_tokens) for output in outputs
        ]

    def _check_continuation(self, chat_prompt_ids, continuation_ids) -> None:
        """Raise unless a continuation's ids extend those ``LLM.chat`` scored.

        All but the prompt's last token must match: text appended to it may
        merge with that token (Qwen's trailing ``"\\n"`` becomes ``"\\n\\n"``).
        """
        kept = max(len(chat_prompt_ids) - 1, 0)
        if list(continuation_ids[:kept]) == list(chat_prompt_ids[:kept]):
            return
        raise RuntimeError(
            f"vLLM tokenized the prompt continued past {self.model_id}'s "
            "thinking block differently from the one LLM.chat scored, so the "
            "answer would be read from another prompt. This happens with vLLM "
            "versions whose RequestOutput.prompt is missing or holds expanded "
            "image placeholders, and with text-only engines whose tokenizer "
            "adds a BOS token. Turn thinking off through chat_template_kwargs "
            "(e.g. {'enable_thinking': False}) where the template supports "
            "it, or use HuggingFaceVLM."
        )

    def _decode_with_special_tokens(self, token_ids) -> str:
        """Decode token ids into text, special tokens included."""
        return self._get_tokenizer().decode(
            list(token_ids), skip_special_tokens=False
        )

    def _build_sampling_params(
        self, generate: bool, interest_mode: bool = False, **inference_kwargs
    ) -> "SamplingParams":
        """Build vLLM SamplingParams for the requested mode.

        ``inference_kwargs`` are user overrides. We only *force* the settings
        the estimator's contract depends on and use ``setdefault`` for the rest
        -- the same pattern OpenAIVLM uses to force logprobs and
        HuggingFaceVLM uses to default ``logits_to_keep=1``.
        """
        if generate:
            # Text mode: do NOT bound max_tokens, so a model can reason as it
            # sees fit (the OpenAI backend documents the same philosophy).
            inference_kwargs.setdefault("temperature", 0.0)
        else:
            # Logprobs mode: one decode step is the memory-minimal default and
            # the direct analog of HuggingFaceVLM's logits_to_keep=1. It is a
            # setdefault, not a force, so a user may override it. Free-form
            # reasoning belongs in text mode, not here.
            inference_kwargs.setdefault("max_tokens", 1)
            inference_kwargs.setdefault("temperature", 0.0)
            # Force logprobs on (analog of OpenAIVLM forcing logprobs=True).
            # -1 == full vocab; otherwise the configured top-k.
            inference_kwargs["logprobs"] = (
                -1 if self.num_logprobs is None else self.num_logprobs
            )
            if interest_mode and "stop" not in inference_kwargs:
                # Interest tokens are matched by token id, so the returned
                # logprob entries never need decoded strings. Skipping
                # detokenization removes vLLM's per-id string conversion on
                # the output-processor thread — the dominant cost of
                # full-vocabulary logprobs. Engines that ignore the flag for
                # logprobs still return id-keyed entries, so results stay
                # correct either way, just slower. Stop strings require
                # detokenization, so an explicit ``stop`` wins.
                inference_kwargs.setdefault("detokenize", False)
        return SamplingParams(**inference_kwargs)

    def _to_vlm_output(
        self, output, generate: bool, interest_tokens=None
    ) -> VLMOutput:
        """Convert one vLLM RequestOutput into a :class:`VLMOutput`."""
        completion = output.outputs[0]
        if generate:
            return self._text_output(output)

        # Logprobs mode reads the distribution over the next answer token. With
        # the default max_tokens=1 this is the only generated position and
        # reproduces HuggingFaceVLM's single-forward next-token distribution.
        logprobs_seq = completion.logprobs
        if not logprobs_seq or logprobs_seq[0] is None:
            # With logprobs forced and max_tokens>=1 this should never happen.
            # Treat it as an internal inconsistency rather than silently
            # returning empty probabilities, because downstream calibration
            # expects token_probs to be the requested next-token distribution.
            raise RuntimeError(
                "vLLM returned no logprobs for a request despite logprobs being "
                "requested. This indicates an internal inconsistency in the vLLM "
                "output; check the installed vLLM version and SamplingParams."
            )

        if interest_tokens is not None:
            return self._interest_output(logprobs_seq[0], interest_tokens)

        token_probs: dict[str, float] = {}
        for logprob in logprobs_seq[0].values():
            # Sum probabilities of duplicate decoded token strings, matching the
            # dedup HuggingFaceVLM and OpenAIVLM perform. Keeping the same
            # VLMOutput.token_probs shape is important because Platt calibration
            # consumes these per-token probabilities without backend-specific
            # handling.
            token = logprob.decoded_token
            token_probs[token] = token_probs.get(token, 0.0) + math.exp(
                logprob.logprob
            )

        return VLMOutput(token_probs=token_probs, text=None)

    def _text_output(self, output) -> VLMOutput:
        """Text-mode output, with any thinking block moved to ``reasoning``.

        A reply whose block never closed gets ``text=None``.
        """
        completion = output.outputs[0]
        tokens = self._get_thinking_tokens()
        if tokens is None:
            return VLMOutput(token_probs=None, text=completion.text)
        reasoning_ids, answer_ids = tokens.split(
            completion.token_ids,
            tokens.opens_block(
                output.prompt_token_ids or [], self._decode_with_special_tokens
            ),
        )
        if reasoning_ids is None:
            return VLMOutput(token_probs=None, text=completion.text)
        tokenizer = self._get_tokenizer()
        text = (
            None
            if answer_ids is None
            else tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        )
        reasoning = tokenizer.decode(reasoning_ids, skip_special_tokens=True)
        return VLMOutput(token_probs=None, text=text, reasoning=reasoning.strip())

    def _interest_output(self, logprob_entries, interest_tokens) -> VLMOutput:
        """Build a VLMOutput by looking up interest ids in the logprob entries.

        Matching by token id (the logprobs dict key) instead of
        ``decoded_token`` works with ``detokenize=False`` output and sums
        duplicate ids decoding to the same string, exactly like the
        string-keyed path. vLLM ranks every returned entry (rank 1 = the
        model's most likely token, always included), so both the masses and
        the argmax flag come from interest-id lookups alone -- no scan of
        the returned distribution.
        """
        index = self._get_token_reverse_index()

        token_probs: dict[str, float] = {}
        argmax_in_interest = False
        for token in dict.fromkeys(interest_tokens):
            mass = 0.0
            for token_id in index.get(token, []):
                entry = logprob_entries.get(token_id)
                if entry is None:
                    continue
                mass += math.exp(entry.logprob)
                if entry.rank == 1:
                    argmax_in_interest = True
            token_probs[token] = mass

        return VLMOutput(
            token_probs=token_probs,
            text=None,
            argmax_in_interest=argmax_in_interest,
        )

    def unsupported_interest_tokens(self, tokens):
        """Subset of ``tokens`` with no single-token surface form here.

        Reuses the same reverse index (:meth:`_get_token_reverse_index`,
        built once from the engine's tokenizer and cached) the interest-mode
        id-lookup path uses.
        """
        index = self._get_token_reverse_index()
        return [t for t in dict.fromkeys(tokens) if t not in index]

    def _get_token_reverse_index(self) -> dict[str, list[int]]:
        """Build (once) the decoded-string -> ids index from the tokenizer."""
        if self._token_reverse_index is None:
            tokenizer = self._get_tokenizer()
            self._token_reverse_index = build_token_reverse_index(
                lambda ids: decode_single_token_ids(tokenizer, ids),
                len(tokenizer),
            )
        return self._token_reverse_index
