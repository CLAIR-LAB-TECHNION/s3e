# SPDX-FileCopyrightText: CLAIR Lab Technion
# SPDX-License-Identifier: MIT

"""HuggingFace Transformers VLM backend.

This module provides a :class:`VLMBackend` implementation that uses
HuggingFace's Auto classes and ``AutoProcessor`` to support vision-language
models loadable through them (exercised with SmolVLM and LLaVA-architecture
checkpoints in the tests and walkthrough notebook).
"""

from .._deps import require

require("torch", "hf", "HuggingFaceVLM")
require("transformers", "hf", "HuggingFaceVLM")

import torch
from transformers import AutoProcessor

from .backend import VLMBackend, VLMOutput, _validate_num_logprobs
from .thinking import (
    DEFAULT_MAX_THINKING_TOKENS,
    MAX_SEPARATOR_TOKENS,
    first_answer_step,
    resolve_thinking_tokens,
    validate_thinking_options,
)
from .token_index import build_token_reverse_index

# transformers 5.x renamed AutoModelForVision2Seq to AutoModelForImageTextToText
try:
    from transformers import AutoModelForImageTextToText as _AutoModelClass
except ImportError:
    from transformers import AutoModelForVision2Seq as _AutoModelClass

# from_pretrained kwargs that select *which* files to load; the processor
# shares them with the model.
_HUB_KWARGS = (
    "revision",
    "cache_dir",
    "token",
    "local_files_only",
    "force_download",
    "proxies",
    "trust_remote_code",
)


class HuggingFaceVLM(VLMBackend):
    """VLM backend using HuggingFace Transformers Auto classes.

    Args:
        model_id: HuggingFace model identifier.
        torch_dtype: PyTorch dtype for model weights. Defaults to
            ``torch.float16`` when CUDA is available, else ``torch.float32``.
            The default follows CUDA *visibility*, not ``device_map``: on a
            GPU host, ``device_map="cpu"`` still loads float16 weights, which
            run slowly on CPU. Pass ``torch_dtype=torch.float32`` explicitly
            when pinning a model to CPU on such a host.
        device_map: Device placement strategy. Defaults to ``"auto"``.
        attn_implementation: Attention implementation to use. ``None`` uses default.
        num_logprobs: Number of top tokens to include in token_probs. ``None``
            returns probabilities for all tokens. Defaults to ``None``.
            Irrelevant when a query passes ``interest_tokens``: that path
            gathers exact masses at known token ids and never truncates.
        skip_pad_invariance_check: Skip the one-time check that padded batches
            reproduce unbatched answers, for models already known to be
            pad-invariant. Defaults to False.
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
            scored. Defaults to 1024.
        thinking_delimiters: The tokens that open and close a thinking
            block, active only when both are single tokens of the model's
            vocabulary. ``"auto"`` (default) uses the first such pair among
            those of known model families: ``<think>``/``</think>`` (Qwen3,
            DeepSeek-R1, GLM, ...), ``<|channel>``/``<channel|>`` (Gemma 4),
            ``[THINK]``/``[/THINK]`` (Mistral, with HF tokenizer files), and
            ``<seed:think>``/``</seed:think>``. Pass a pair to use only that
            one, or ``None`` to turn thinking handling off.
        **model_kwargs: Additional kwargs for the model's from_pretrained().
            Hub kwargs that select the files to load (``revision``,
            ``cache_dir``, ``token``, ``local_files_only``, ``force_download``,
            ``proxies``, ``trust_remote_code``) are also passed to the
            processor, so a pinned revision pins both.

    Notes:
        There is intentionally no ``max_new_tokens`` constructor parameter
        (mirroring :class:`VLLMBackend`): generation length is controlled
        per call through ``inference_kwargs`` (transformers ``generate()``
        kwargs, e.g. ``max_new_tokens=10``).
    """

    def __init__(
        self,
        model_id: str,
        torch_dtype=None,
        device_map: str = "auto",
        attn_implementation: str | None = None,
        num_logprobs: int | None = None,
        skip_pad_invariance_check: bool = False,
        chat_template_kwargs: dict | None = None,
        thinking: str = "close",
        max_thinking_tokens: int = DEFAULT_MAX_THINKING_TOKENS,
        thinking_delimiters: tuple[str, str] | str | None = "auto",
        **model_kwargs,
    ):
        self.model_id = model_id
        self.num_logprobs = _validate_num_logprobs(num_logprobs)
        self.thinking_delimiters = validate_thinking_options(
            thinking, max_thinking_tokens, thinking_delimiters
        )
        self.thinking = thinking
        self.max_thinking_tokens = max_thinking_tokens
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self._pad_invariance_checked = bool(skip_pad_invariance_check)
        self._token_reverse_index: dict[str, list[int]] | None = None
        self._thinking_tokens = None
        self._thinking_tokens_resolved = False

        if torch_dtype is None:
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        load_kwargs = dict(
            dtype=torch_dtype,
            device_map=device_map,
            **model_kwargs,
        )
        if attn_implementation is not None:
            load_kwargs["attn_implementation"] = attn_implementation

        self.model = _AutoModelClass.from_pretrained(model_id, **load_kwargs)
        # The processor must come from the same snapshot as the weights (e.g.
        # a pinned ``revision``), but not get model-only kwargs.
        processor_kwargs = {
            key: model_kwargs[key] for key in _HUB_KWARGS if key in model_kwargs
        }
        self.processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)
        self.model.eval()

        # Batched inference reads logits[:, -1, :]; left padding makes that
        # position the last real token of every row.
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"

    def query(
        self,
        images,
        prompt,
        system_prompt=None,
        generate=False,
        interest_tokens=None,
        **inference_kwargs,
    ):
        """Send a single query to the HuggingFace VLM."""
        results = self.query_batch(
            images,
            [prompt],
            system_prompt,
            generate,
            interest_tokens=interest_tokens,
            **inference_kwargs,
        )
        return results[0]

    def query_batch(
        self,
        images,
        prompts,
        system_prompt=None,
        generate=False,
        interest_tokens=None,
        **inference_kwargs,
    ):
        """Send multiple queries against the same images."""
        if not prompts:
            return []

        # by default, only keep the last token's log probabilities to avoid
        # OOM; generate() manages its own logits, so only the forward pass
        # gets this default
        if not generate and "logits_to_keep" not in inference_kwargs:
            inference_kwargs["logits_to_keep"] = 1

        text_inputs = [
            self._render_prompt(images, prompt, system_prompt)
            for prompt in prompts
        ]
        batched_images = self._build_batched_images(images, len(prompts))

        try:
            inputs = self._prepare_inputs(text_inputs, batched_images)
        except Exception:
            return self._query_batch_sequential(
                images,
                prompts,
                system_prompt,
                generate,
                interest_tokens=interest_tokens,
                **inference_kwargs,
            )

        if generate:
            return self._generate_outputs(inputs, **inference_kwargs)

        probs = self._forward_next_token_probs(inputs, **inference_kwargs)
        self._check_pad_invariance(images, text_inputs, inputs, probs, **inference_kwargs)
        probs, reasonings = self._skip_thinking(images, text_inputs, inputs, probs)
        return self._outputs_from_probs(probs, interest_tokens, reasonings)

    def _query_batch_sequential(
        self,
        images,
        prompts,
        system_prompt=None,
        generate=False,
        interest_tokens=None,
        **inference_kwargs,
    ):
        """Correctness fallback for processors that cannot batch inputs."""
        results = []
        for prompt in prompts:
            text_input = self._render_prompt(images, prompt, system_prompt)
            inputs = self._prepare_inputs(text_input, images if images else None)

            if generate:
                results.append(self._generate_outputs(inputs, **inference_kwargs)[0])
            else:
                probs = self._forward_next_token_probs(inputs, **inference_kwargs)
                probs, reasonings = self._skip_thinking(
                    images, [text_input], inputs, probs
                )
                results.append(
                    self._outputs_from_probs(probs, interest_tokens, reasonings)[0]
                )
        return results

    def _outputs_from_probs(self, probs, interest_tokens, reasonings):
        """Build one VLMOutput per probability row.

        With ``interest_tokens``, mass is gathered at precomputed ids and
        nothing is detokenized; otherwise the legacy full/top-k path decodes
        the returned distribution into strings.
        """
        if interest_tokens is not None:
            return [
                VLMOutput(
                    token_probs=token_probs,
                    text=None,
                    argmax_in_interest=argmax_in_interest,
                    reasoning=reasoning,
                )
                for (token_probs, argmax_in_interest), reasoning in zip(
                    self._gather_interest_probs(probs, interest_tokens), reasonings
                )
            ]
        return [
            VLMOutput(token_probs=token_probs, text=None, reasoning=reasoning)
            for token_probs, reasoning in zip(
                self._format_token_probs(probs), reasonings
            )
        ]

    def _get_thinking_tokens(self):
        """The resolved thinking delimiters, or None when inactive (cached)."""
        if not self._thinking_tokens_resolved:
            self._thinking_tokens = resolve_thinking_tokens(
                self.thinking_delimiters, getattr(self.processor, "tokenizer", None)
            )
            self._thinking_tokens_resolved = True
        return self._thinking_tokens

    def _skip_thinking(self, images, text_inputs, inputs, probs):
        """Re-score the rows whose answer would come after a thinking block.

        A row thinks when its prompt ends inside an open block or its most
        likely next token opens one. Returns the probabilities with those
        rows replaced by the answer distribution after the block, plus one
        ``reasoning`` entry per row (None for rows without a block).
        """
        reasonings = [None] * probs.shape[0]
        tokens = self._get_thinking_tokens()
        input_ids = inputs.get("input_ids")
        if tokens is None or not isinstance(input_ids, torch.Tensor):
            return probs, reasonings

        argmax_ids = probs.argmax(dim=-1).tolist()
        rows, prompt_open = [], []
        for row, ids in enumerate(input_ids.tolist()):
            is_open = tokens.opens_block(ids)
            if is_open or argmax_ids[row] == tokens.start_id:
                rows.append(row)
                prompt_open.append(is_open)
        if not rows:
            return probs, reasonings

        prompts = [text_inputs[row] for row in rows]
        if self.thinking == "think":
            continuations, row_reasonings = self._think(
                images, prompts, prompt_open, tokens
            )
        else:
            continuations = [
                prompt + tokens.closing_text(prompt, is_open)
                for prompt, is_open in zip(prompts, prompt_open)
            ]
            row_reasonings = [""] * len(rows)

        probs = probs.clone()
        probs[rows] = self._answer_probs_after(images, continuations).to(
            device=probs.device, dtype=probs.dtype
        )
        for row, reasoning in zip(rows, row_reasonings):
            reasonings[row] = reasoning
        return probs, reasonings

    def _think(self, images, prompts, prompt_open, tokens):
        """Generate each prompt's reasoning, up to ``max_thinking_tokens``.

        Returns one continuation per prompt (the prompt plus its reasoning,
        ending with the closing delimiter, which is appended when the budget
        ran out or the model stopped first) and the reasoning texts.
        """
        inputs = self._prepare_rows(prompts, images)
        eos_ids = self._eos_token_ids()
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_thinking_tokens,
            do_sample=False,
            eos_token_id=[tokens.end_id, *eos_ids],
        )
        sequences = self._trim_generated_sequences(output_ids, inputs)
        pad_id = getattr(getattr(self.model, "generation_config", None), "pad_token_id", None)
        stop_ids = set(eos_ids) | ({int(pad_id)} if pad_id is not None else set())

        continuations, reasonings = [], []
        for prompt, is_open, sequence in zip(prompts, prompt_open, sequences):
            ids = [int(token_id) for token_id in sequence]
            if tokens.end_id in ids:
                ids = ids[: ids.index(tokens.end_id) + 1]
                generated = self._decode_sequence(ids, skip_special_tokens=False)
            else:
                stop = next((i for i, t in enumerate(ids) if t in stop_ids), len(ids))
                ids = ids[:stop]
                generated = self._decode_sequence(ids, skip_special_tokens=False)
                if not generated[-1:].isspace():
                    generated += "\n"
                generated += tokens.end
            reasoning_ids, _ = tokens.split(ids, is_open)
            continuations.append(prompt + generated)
            reasonings.append(
                self._decode_sequence(reasoning_ids or [], skip_special_tokens=True).strip()
            )
        return continuations, reasonings

    def _answer_probs_after(self, images, continuations):
        """Next-token probabilities at each continuation's answer position.

        Chat templates put whitespace between a closed block and the answer
        (``"\\n\\n"`` for Qwen), so a few greedy steps are generated and each
        row is read at its first token that is not whitespace.
        """
        inputs = self._prepare_rows(continuations, images)
        output = self.model.generate(
            **inputs,
            max_new_tokens=MAX_SEPARATOR_TOKENS,
            do_sample=False,
            output_logits=True,
            return_dict_in_generate=True,
        )
        new_ids = self._trim_generated_sequences(output.sequences, inputs)
        rows = []
        for row, ids in enumerate(new_ids):
            ids = [int(token_id) for token_id in ids][: len(output.logits)]
            step = first_answer_step(self._decode_token_ids(ids))
            rows.append(output.logits[step][row])
        return torch.softmax(torch.stack(rows).float(), dim=-1)

    def _prepare_rows(self, texts, images):
        """Processor inputs for prompts that all show the same images."""
        if len(texts) == 1:
            return self._prepare_inputs(texts[0], images if images else None)
        return self._prepare_inputs(
            texts, self._build_batched_images(images, len(texts))
        )

    def _eos_token_ids(self) -> list[int]:
        """The model's end-of-sequence ids from its generation config."""
        config = getattr(self.model, "generation_config", None)
        eos = getattr(config, "eos_token_id", None)
        if eos is None:
            return []
        eos = eos if isinstance(eos, (list, tuple)) else [eos]
        return [int(token_id) for token_id in eos]

    def unsupported_interest_tokens(self, tokens):
        """Subset of ``tokens`` with no single-token surface form here.

        Reuses the same reverse index (:meth:`_get_token_reverse_index`,
        built once and cached) the interest-token gather path uses, sized
        from the tokenizer's vocabulary -- the same cap that path applies
        when the tokenizer length is known.
        """
        tokenizer = getattr(self.processor, "tokenizer", None)
        vocab_size = len(tokenizer) if tokenizer is not None else 0
        index = self._get_token_reverse_index(vocab_size)
        return [t for t in dict.fromkeys(tokens) if t not in index]

    def _get_token_reverse_index(self, vocab_size: int) -> dict[str, list[int]]:
        """Build (once) and return the decoded-string -> ids index.

        The logits row can be longer than the tokenizer's vocabulary (padded
        embedding matrices); ids beyond the tokenizer cannot decode to an
        interest token, so the index is capped at the tokenizer length when
        it is known.
        """
        if self._token_reverse_index is None:
            vocab_limit = vocab_size
            tokenizer = getattr(self.processor, "tokenizer", None)
            if tokenizer is not None:
                try:
                    tokenizer_len = len(tokenizer)
                except TypeError:
                    tokenizer_len = 0
                if tokenizer_len > 0:
                    vocab_limit = min(vocab_limit, tokenizer_len)
            self._token_reverse_index = build_token_reverse_index(
                self._decode_token_ids, vocab_limit
            )
        return self._token_reverse_index

    def _gather_interest_probs(self, probs, interest_tokens):
        """Sum each row's mass over the ids decoding to each interest token.

        Returns one ``(token_probs, argmax_in_interest)`` pair per row.
        Exact: no top-k truncation, and duplicate ids decoding to the same
        string are summed just like the full-vocabulary string path.
        """
        index = self._get_token_reverse_index(int(probs.shape[-1]))
        interest = list(dict.fromkeys(interest_tokens))

        all_ids: list[int] = []
        spans: list[tuple[str, int, int]] = []
        for token in interest:
            ids = [i for i in index.get(token, []) if i < probs.shape[-1]]
            spans.append((token, len(all_ids), len(all_ids) + len(ids)))
            all_ids.extend(ids)

        # float64 so summing duplicate ids matches the full-vocab path, which
        # accumulates Python floats.
        selected = probs[:, all_ids].double().cpu() if all_ids else None
        argmax_ids = probs.argmax(dim=-1).cpu().tolist()
        interest_id_set = set(all_ids)

        results = []
        for row in range(probs.shape[0]):
            token_probs = {
                token: (
                    float(selected[row, start:end].sum()) if end > start else 0.0
                )
                for token, start, end in spans
            }
            results.append((token_probs, argmax_ids[row] in interest_id_set))
        return results

    def _render_prompt(self, images, prompt, system_prompt=None):
        """Render a prompt with the processor chat template when available."""
        full_prompt = prompt
        if system_prompt is not None:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        messages = self._build_messages(images, prompt, system_prompt)
        try:
            return self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                **self.chat_template_kwargs,
            )
        except Exception:
            return full_prompt

    @staticmethod
    def _build_batched_images(images, batch_size: int):
        """Repeat the same image set for every prompt row."""
        if not images:
            return None
        return [list(images) for _ in range(batch_size)]

    def _prepare_inputs(self, text, images):
        """Run the processor and move tensor-like inputs to the model device."""
        inputs = self.processor(
            text=text,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        return self._move_inputs_to_device(inputs)

    def _move_inputs_to_device(self, inputs):
        """Move tensor-like processor outputs while preserving metadata."""
        return {
            key: value.to(self.model.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    def _build_messages(self, images, prompt, system_prompt=None):
        """Build a chat-format message list."""
        messages = []
        if system_prompt is not None:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            )

        user_content = []
        for _ in images:
            user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": prompt})

        messages.append({"role": "user", "content": user_content})
        return messages

    def _forward_next_token_probs(self, inputs, **inference_kwargs):
        """Run the model and return next-token probabilities for each row.

        The final logit position is the last real token of every row because
        the tokenizer pads on the left.
        """
        with torch.inference_mode():
            outputs = self.model(**inputs, **inference_kwargs)
        return torch.softmax(outputs.logits[:, -1, :].float(), dim=-1)

    def _check_pad_invariance(
        self, images, text_inputs, inputs, batched_probs, **inference_kwargs
    ):
        """One-time check that padding does not change a row's answer.

        Reruns the most-padded row of the first padded batch on its own and
        compares the two distributions, catching models whose answers depend
        on where padding sits (e.g. learned absolute position embeddings, or
        a processor that ignores ``padding_side``).
        """
        if self._pad_invariance_checked:
            return
        attention_mask = inputs.get("attention_mask")
        if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2:
            return
        if bool(attention_mask.all()):
            return

        row = int(attention_mask.sum(dim=1).argmin())
        row_inputs = self._prepare_inputs(text_inputs[row], images if images else None)
        solo_probs = self._forward_next_token_probs(row_inputs, **inference_kwargs)[0]

        max_diff = float((solo_probs - batched_probs[row]).abs().max())
        if max_diff > 0.05:
            raise ValueError(
                f"{self.model_id} changes its next-token distribution under "
                f"padding (max probability difference {max_diff:.3f}), so "
                f"batched results cannot be trusted. Use batch_size=1."
            )
        self._pad_invariance_checked = True

    def _format_token_probs(self, probs):
        """Convert per-row probability tensors into token->probability dicts."""
        if self.num_logprobs is None:
            selected_probs = probs
            selected_indices = torch.arange(
                probs.shape[-1], device=probs.device
            ).expand(probs.shape[0], -1)
        else:
            selected_probs, selected_indices = torch.topk(
                probs, min(self.num_logprobs, probs.shape[-1]), dim=-1
            )

        selected_probs = selected_probs.detach().cpu().tolist()
        selected_indices = selected_indices.detach().cpu().tolist()
        flat_indices = [int(idx) for row in selected_indices for idx in row]
        decoded_tokens = self._decode_token_ids(flat_indices)

        results = []
        offset = 0
        for row_probs, row_indices in zip(selected_probs, selected_indices):
            row_tokens = decoded_tokens[offset : offset + len(row_indices)]
            offset += len(row_indices)

            token_probs = {}
            for token_str, prob in zip(row_tokens, row_probs):
                token_probs[token_str] = token_probs.get(token_str, 0.0) + float(prob)
            results.append(token_probs)

        return results

    def _decode_token_ids(self, token_ids: list[int]) -> list[str]:
        """Decode token IDs, preferring batch decoding when it behaves correctly."""
        if not token_ids:
            return []

        batch_decode = getattr(self.processor, "batch_decode", None)
        if callable(batch_decode):
            try:
                decoded = batch_decode(
                    [[int(token_id)] for token_id in token_ids],
                    skip_special_tokens=False,
                )
                if len(decoded) == len(token_ids):
                    return list(decoded)
            except Exception:
                pass

        return [self.processor.decode(int(token_id)) for token_id in token_ids]

    def _generate_outputs(self, inputs, **inference_kwargs) -> list[VLMOutput]:
        """Generate text responses for a batch of prompts.

        A thinking block is cut out of each reply into ``reasoning``; a
        reply whose block never closed gets ``text=None``. Generation errors
        propagate to the caller. ``text`` is also None when the generated
        sequences cannot be aligned one-to-one with the input prompts (e.g.
        multi-sequence decoding strategies).
        """
        batch_size = self._infer_batch_size(inputs)
        output_ids = self.model.generate(**inputs, **inference_kwargs)
        generated_sequences = self._trim_generated_sequences(output_ids, inputs)
        generated_sequences = self._select_generated_sequences_for_prompts(
            generated_sequences, batch_size
        )
        if generated_sequences is None:
            return [VLMOutput(token_probs=None, text=None) for _ in range(batch_size)]

        tokens = self._get_thinking_tokens()
        input_ids = inputs.get("input_ids")
        if tokens is None or not isinstance(input_ids, torch.Tensor):
            return [
                VLMOutput(token_probs=None, text=text)
                for text in self._decode_generated_sequences(generated_sequences)
            ]

        outputs = []
        for prompt_ids, sequence in zip(input_ids.tolist(), generated_sequences):
            reasoning_ids, answer_ids = tokens.split(
                sequence.tolist(), tokens.opens_block(prompt_ids)
            )
            if reasoning_ids is None:
                text = self._decode_generated_sequences([sequence])[0]
                outputs.append(VLMOutput(token_probs=None, text=text))
                continue
            text = (
                None
                if answer_ids is None
                else self._decode_sequence(answer_ids, skip_special_tokens=True).strip()
            )
            reasoning = self._decode_sequence(reasoning_ids, skip_special_tokens=True)
            outputs.append(
                VLMOutput(token_probs=None, text=text, reasoning=reasoning.strip())
            )
        return outputs

    def _decode_sequence(self, token_ids, skip_special_tokens: bool) -> str:
        """Decode one token-id sequence into a single string."""
        if not token_ids:
            return ""
        return self.processor.decode(
            [int(token_id) for token_id in token_ids],
            skip_special_tokens=skip_special_tokens,
        )

    @staticmethod
    def _infer_batch_size(inputs) -> int:
        input_ids = inputs.get("input_ids")
        if input_ids is not None and hasattr(input_ids, "shape") and input_ids.shape:
            return int(input_ids.shape[0])
        return 1

    def _trim_generated_sequences(self, output_ids, inputs):
        """Remove prompt tokens from decoder-only generation outputs."""
        if self._model_is_encoder_decoder():
            return [row for row in output_ids]
        input_len = inputs["input_ids"].shape[-1]
        return [row[input_len:] for row in output_ids]

    @staticmethod
    def _select_generated_sequences_for_prompts(sequences, batch_size: int):
        """Keep one generated sequence per input prompt."""
        sequence_count = len(sequences)
        if sequence_count == batch_size:
            return sequences
        if batch_size > 0 and sequence_count > 0 and sequence_count % batch_size == 0:
            group_size = sequence_count // batch_size
            return [sequences[i * group_size] for i in range(batch_size)]
        return None

    def _model_is_encoder_decoder(self) -> bool:
        config = getattr(self.model, "config", None)
        return getattr(config, "is_encoder_decoder", False) is True

    def _decode_generated_sequences(self, sequences) -> list[str]:
        """Decode generated token sequences, preferring batch decoding."""
        batch_decode = getattr(self.processor, "batch_decode", None)
        if callable(batch_decode):
            try:
                decoded = batch_decode(sequences, skip_special_tokens=True)
                if len(decoded) == len(sequences):
                    return list(decoded)
            except Exception:
                pass

        return [
            self.processor.decode(sequence, skip_special_tokens=True)
            for sequence in sequences
        ]
