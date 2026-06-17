"""VLM (Vision-Language Model) backends for s3e.

This subpackage provides the :class:`VLMBackend` abstraction and
concrete implementations for HuggingFace Transformers models, the
OpenAI API, and vLLM.
"""

from .backend import VLMBackend, VLMOutput
from .huggingface import HuggingFaceVLM
from .openai import OpenAIVLM
from .vllm import VLLMBackend

__all__ = ["VLMBackend", "VLMOutput", "HuggingFaceVLM", "OpenAIVLM", "VLLMBackend"]
