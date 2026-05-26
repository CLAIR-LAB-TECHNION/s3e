"""HuggingFace Transformers VLM backend.

This module provides a :class:`VLMBackend` implementation that uses
HuggingFace's Auto classes and ``AutoProcessor`` to support any standard
vision-language model (LLaVA, Qwen2-VL, InternVL, etc.).
"""

import torch
import numpy as np

from .backend import VLMBackend, VLMOutput

# transformers 5.x renamed AutoModelForVision2Seq to AutoModelForImageTextToText
_AutoModelClass = None
AutoProcessor = None  # type: ignore[assignment]

try:
    from transformers import AutoProcessor  # type: ignore[no-redef]

    try:
        from transformers import AutoModelForImageTextToText as _AutoModelClass  # type: ignore[no-redef]
    except ImportError:
        from transformers import AutoModelForVision2Seq as _AutoModelClass  # type: ignore[no-redef]
except ImportError:
    pass


def _check_hf_imports() -> None:
    if _AutoModelClass is None or AutoProcessor is None:
        raise ImportError(
            "Neither AutoModelForImageTextToText nor AutoModelForVision2Seq "
            "are available in your version of transformers. "
            "Install a compatible version with: pip install 'transformers>=4.36'"
        )


class HuggingFaceVLM(VLMBackend):
    """VLM backend using HuggingFace Transformers Auto classes.

    Args:
        model_id: HuggingFace model identifier.
        torch_dtype: PyTorch dtype for model weights. Defaults to
            ``torch.float16`` when CUDA is available, else ``torch.float32``.
        device_map: Device placement strategy. Defaults to ``"auto"``.
        attn_implementation: Attention implementation to use. ``None`` uses default.
        num_logprobs: Number of top tokens to include in token_probs. ``None``
            returns probabilities for all tokens. Defaults to ``None``.
        max_new_tokens: Maximum number of new tokens to generate. Defaults to 10.
        **model_kwargs: Additional kwargs for from_pretrained(). ``max_new_tokens``
            is consumed from this mapping and used for text generation.
    """

    def __init__(
        self,
        model_id: str,
        torch_dtype=None,
        device_map: str = "auto",
        attn_implementation: str | None = None,
        num_logprobs: int | None = None,
        max_new_tokens: int = 10,
        **model_kwargs,
    ):
        _check_hf_imports()
        self.model_id = model_id
        self.num_logprobs = num_logprobs
        self.max_new_tokens = max_new_tokens

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

    def query(self, images, prompt, system_prompt=None, generate=False, **inference_kwargs):
        """Send a single query to the HuggingFace VLM."""
        results = self.query_batch(images, [prompt], system_prompt, generate, **inference_kwargs)
        return results[0]

    def query_batch(self, images, prompts, system_prompt=None, generate=False, **inference_kwargs):
        """Send multiple queries against the same images."""
        if not prompts:
            return []

        # by default, only keep the last token's log probabilities to avoid OOM
        if "logits_to_keep" not in inference_kwargs:
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
                images, prompts, system_prompt, generate, **inference_kwargs
            )

        if generate:
            generated_texts = self._generate_text(inputs, **inference_kwargs)
            return [
                VLMOutput(token_probs=None, text=generated_text)
                for generated_text in generated_texts
            ]

        token_probs_batch = self._get_next_token_probs(inputs, **inference_kwargs)
        return [
            VLMOutput(token_probs=token_probs, text=None)
            for token_probs in token_probs_batch
        ]

    def _query_batch_sequential(
        self,
        images,
        prompts,
        system_prompt=None,
        generate=False,
        **inference_kwargs,
    ):
        """Correctness fallback for processors that cannot batch inputs."""
        results = []
        for prompt in prompts:
            text_input = self._render_prompt(images, prompt, system_prompt)
            inputs = self._prepare_inputs(text_input, images if images else None)

            if generate:
                generated_text = self._generate_text(inputs, **inference_kwargs)[0]
                token_probs = None
            else:
                token_probs = self._get_next_token_probs(
                    inputs, **inference_kwargs
                )[0]
                generated_text = None

            results.append(VLMOutput(token_probs=token_probs, text=generated_text))
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

    def _select_next_token_logits(self, logits, inputs):
        """Select logits after the last non-padding token in each batch row."""
        attention_mask = inputs.get("attention_mask")
        if (
            isinstance(attention_mask, torch.Tensor)
            and attention_mask.ndim == 2
            and tuple(attention_mask.shape) == tuple(logits.shape[:2])
        ):
            mask = attention_mask.to(device=logits.device).bool()
            positions = torch.arange(logits.shape[1], device=logits.device).expand(
                logits.shape[0], -1
            )
            last_indices = positions.masked_fill(~mask, -1).max(dim=1).values
            last_indices = torch.where(
                last_indices >= 0,
                last_indices,
                torch.full_like(last_indices, logits.shape[1] - 1),
            )
            batch_indices = torch.arange(logits.shape[0], device=logits.device)
            return logits[batch_indices, last_indices, :]

        return logits[:, -1, :]

    def _get_next_token_probs(self, inputs, **inference_kwargs):
        with torch.inference_mode():
            outputs = self.model(**inputs, **inference_kwargs)

        logits = self._select_next_token_logits(outputs.logits, inputs).float()
        probs = torch.softmax(logits, dim=-1)

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
        """Generate text responses for a batch of prompts."""
        batch_size = self._infer_batch_size(inputs)
        try:
            output_ids = self.model.generate(**inputs, **inference_kwargs)
            generated_sequences = self._trim_generated_sequences(output_ids, inputs)
            generated_sequences = self._select_generated_sequences_for_prompts(
                generated_sequences, batch_size
            )
            if generated_sequences is None:
                return [None for _ in range(batch_size)]
            return self._decode_generated_sequences(generated_sequences)
        except Exception:
            return [None for _ in range(batch_size)]

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
