"""Public factory turning model-id strings into VLM backends."""

from .backend import VLMBackend

OPENAI_MODEL_IDENTIFIER = "OpenAI/"


def resolve_backend(vlm: "str | VLMBackend", **vlm_kwargs) -> VLMBackend:
    """Resolve a model string or backend instance into a VLMBackend.

    Strings prefixed with ``"OpenAI/"`` select :class:`OpenAIVLM`; any other
    string selects :class:`HuggingFaceVLM`. For a vLLM engine, construct
    ``VLLMBackend(...)`` explicitly and pass the instance.

    Args:
        vlm: Backend instance (returned as-is) or model-id string.
        vlm_kwargs: Constructor kwargs, forwarded only on the string path.

    Raises:
        ValueError: If ``vlm`` is already an instance but ``vlm_kwargs``
            were provided (they would be silently dropped otherwise).
    """
    if isinstance(vlm, VLMBackend):
        if vlm_kwargs:
            raise ValueError(
                "vlm_kwargs are only used when vlm is a model string; "
                f"got a {type(vlm).__name__} instance plus kwargs "
                f"{sorted(vlm_kwargs)}"
            )
        return vlm
    if not isinstance(vlm, str):
        raise TypeError(
            "vlm must be a model-id string or VLMBackend instance; "
            f"got {type(vlm).__name__}"
        )
    if vlm.startswith(OPENAI_MODEL_IDENTIFIER):
        from .openai import OpenAIVLM

        return OpenAIVLM(vlm[len(OPENAI_MODEL_IDENTIFIER):], **vlm_kwargs)
    from .huggingface import HuggingFaceVLM

    return HuggingFaceVLM(vlm, **vlm_kwargs)
