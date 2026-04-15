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
        num_logprobs: Number of top tokens to include in token_probs. Defaults to 20.
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
        num_logprobs: int = 20,
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
        results = []
        for prompt in prompts:
            full_prompt = prompt
            if system_prompt is not None:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            # Build conversation format for chat models
            messages = self._build_messages(images, prompt, system_prompt)

            try:
                text_input = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
            except Exception:
                # Fallback for models without chat template
                text_input = full_prompt

            inputs = self.processor(
                text=text_input,
                images=images if images else None,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            if generate:
                # Generate text response
                generated_text = self._generate_text(inputs, **inference_kwargs)
                token_probs = None
            else:
                # Get next-token probabilities
                token_probs = self._get_next_token_probs(inputs, **inference_kwargs)
                generated_text = None

            results.append(VLMOutput(token_probs=token_probs, text=generated_text))

        return results

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

            # Get next-token logits (last position in sequence)
        logits = outputs.logits[:, -1, :].float()
        probs = torch.softmax(logits, dim=-1)

        # Extract top-k token probabilities
        top_probs, top_indices = torch.topk(
            probs[0], min(self.num_logprobs, probs.shape[-1])
        )
        token_probs = {}
        for prob, idx in zip(top_probs, top_indices):
            token_str = self.processor.decode(idx.item())
            token_probs[token_str] = prob.item()
        return token_probs

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
