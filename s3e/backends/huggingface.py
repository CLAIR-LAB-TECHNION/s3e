"""HuggingFace Transformers VLM backend.

This module provides a :class:`VLMBackend` implementation that uses
HuggingFace's Auto classes and ``AutoProcessor`` to support any standard
vision-language model (LLaVA, Qwen2-VL, InternVL, etc.).
"""

from .._deps import require

require("torch", "hf", "HuggingFaceVLM")
require("transformers", "hf", "HuggingFaceVLM")

import torch
import numpy as np
from transformers import AutoProcessor

from .backend import VLMBackend, VLMOutput
from .token_index import build_token_reverse_index

# transformers 5.x renamed AutoModelForVision2Seq to AutoModelForImageTextToText
try:
    from transformers import AutoModelForImageTextToText as _AutoModelClass
except ImportError:
    from transformers import AutoModelForVision2Seq as _AutoModelClass


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
        **model_kwargs: Additional kwargs for from_pretrained().

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
        **model_kwargs,
    ):
        self.model_id = model_id
        self.num_logprobs = num_logprobs
        self._pad_invariance_checked = bool(skip_pad_invariance_check)
        self._token_reverse_index: dict[str, list[int]] | None = None

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
        self.processor = AutoProcessor.from_pretrained(model_id)
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
            generated_texts = self._generate_text(inputs, **inference_kwargs)
            return [
                VLMOutput(token_probs=None, text=generated_text)
                for generated_text in generated_texts
            ]

        probs = self._forward_next_token_probs(inputs, **inference_kwargs)
        self._check_pad_invariance(images, text_inputs, inputs, probs, **inference_kwargs)
        return self._outputs_from_probs(probs, interest_tokens)

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
                generated_text = self._generate_text(inputs, **inference_kwargs)[0]
                results.append(VLMOutput(token_probs=None, text=generated_text))
            else:
                probs = self._forward_next_token_probs(inputs, **inference_kwargs)
                results.append(self._outputs_from_probs(probs, interest_tokens)[0])
        return results

    def _outputs_from_probs(self, probs, interest_tokens):
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
                )
                for token_probs, argmax_in_interest in self._gather_interest_probs(
                    probs, interest_tokens
                )
            ]
        return [
            VLMOutput(token_probs=token_probs, text=None)
            for token_probs in self._format_token_probs(probs)
        ]

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
                messages, add_generation_prompt=True, tokenize=False
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

    def _generate_text(self, inputs, **inference_kwargs) -> list[str | None]:
        """Generate text responses for a batch of prompts.

        Generation errors propagate to the caller. ``None`` entries appear
        only when the generated sequences cannot be aligned one-to-one with
        the input prompts (e.g. multi-sequence decoding strategies).
        """
        batch_size = self._infer_batch_size(inputs)
        output_ids = self.model.generate(**inputs, **inference_kwargs)
        generated_sequences = self._trim_generated_sequences(output_ids, inputs)
        generated_sequences = self._select_generated_sequences_for_prompts(
            generated_sequences, batch_size
        )
        if generated_sequences is None:
            return [None for _ in range(batch_size)]
        return self._decode_generated_sequences(generated_sequences)

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
