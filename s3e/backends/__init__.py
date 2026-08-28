"""VLM backends: the abstract contract plus concrete implementations.

``VLMBackend``/``VLMOutput`` and :func:`resolve_backend` import without any
optional dependency. The concrete backends are exposed lazily so that
``import s3e.backends`` never imports torch, openai, or vllm.
"""

import importlib

from .backend import VLMBackend, VLMOutput
from .resolve import OPENAI_MODEL_IDENTIFIER, resolve_backend

__all__ = [
    "VLMBackend",
    "VLMOutput",
    "resolve_backend",
    "OPENAI_MODEL_IDENTIFIER",
    "HuggingFaceVLM",
    "OpenAIVLM",
    "VLLMBackend",
]

_LAZY_BACKENDS = {
    "HuggingFaceVLM": ".huggingface",
    "OpenAIVLM": ".openai",
    "VLLMBackend": ".vllm",
}


def __getattr__(name: str):
    """Lazily expose backends that need optional dependencies."""
    if name in _LAZY_BACKENDS:
        module = importlib.import_module(_LAZY_BACKENDS[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
