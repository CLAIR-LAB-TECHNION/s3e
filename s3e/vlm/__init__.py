"""VLM (Vision-Language Model) backends for s3e.

This subpackage provides the :class:`VLMBackend` abstraction and
concrete implementations for HuggingFace Transformers models and
the OpenAI API.
"""

from .backend import VLMBackend, VLMOutput
from .huggingface import HuggingFaceVLM
from .openai import OpenAIVLM

__all__ = ["VLMBackend", "VLMOutput", "HuggingFaceVLM", "OpenAIVLM"]
