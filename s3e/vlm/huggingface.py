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
            torch_dtype=torch_dtype,
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
            return self._query_batch_sequential(
                images, prompts, system_prompt, generate, **inference_kwargs
            )

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
                generated_text = self._generate_text(inputs, **inference_kwargs)
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

    def _get_next_token_probs(self, inputs, **inference_kwargs):
        with torch.no_grad():
            outputs = self.model(**inputs, **inference_kwargs)

        logits = outputs.logits[:, -1, :].float()
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

    def _generate_text(self, inputs, **inference_kwargs) -> str | None:
        """Generate a short text response for the text_match probability method."""
        try:
            output_ids = self.model.generate(**inputs, **inference_kwargs)
            # Trim the input tokens from the output
            input_len = inputs["input_ids"].shape[-1]
            generated_ids = output_ids[0, input_len:]
            return self.processor.decode(generated_ids, skip_special_tokens=True)
        except Exception:
            return None
