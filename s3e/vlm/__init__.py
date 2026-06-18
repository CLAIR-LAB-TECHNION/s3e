"""VLM (Vision-Language Model) backends for s3e.

This subpackage provides the :class:`VLMBackend` abstraction and
concrete implementations for HuggingFace Transformers models, the
OpenAI API, and vLLM.
"""

from .backend import VLMBackend, VLMOutput
from .huggingface import HuggingFaceVLM
from .openai import OpenAIVLM

__all__ = ["VLMBackend", "VLMOutput", "HuggingFaceVLM", "OpenAIVLM", "VLLMBackend"]


def __getattr__(name: str):
    """Lazily expose optional backends without importing their dependencies."""
    if name == "VLLMBackend":
        from .vllm import VLLMBackend

        return VLLMBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
