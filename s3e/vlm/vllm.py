"""vLLM VLM backend.

A :class:`VLMBackend` implementation that runs a local HuggingFace
vision-language model through `vLLM <https://docs.vllm.ai/>`_ for transparent,
high-throughput, single-node multi-GPU inference. Selected by passing
``use_vllm=True`` to :class:`SemanticStateEstimator` with a non-OpenAI model id.

Design notes (kept in code on purpose so future maintainers can change the
design with full context):

* The module is named ``vllm.py`` to match the existing backend naming
  convention (``openai.py`` imports ``openai``; ``huggingface.py`` uses
  ``transformers``). Python 3 uses absolute imports, so ``from vllm import ...``
  below resolves to the installed *package*, not this module -- the same safe
  pattern ``openai.py`` already relies on.
* ``vllm`` is an optional dependency, imported lazily so importing s3e never
  requires it. Construction raises a helpful :class:`ImportError` when missing.
"""

import math

import torch

from .backend import VLMBackend, VLMOutput

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None  # type: ignore[assignment]
    SamplingParams = None  # type: ignore[assignment]


def _check_vllm_installed() -> None:
    if LLM is None or SamplingParams is None:
        raise ImportError(
            "The 'vllm' package is required for VLLMBackend. "
            "Install it with: pip install s3e[vllm]"
        )


class VLLMBackend(VLMBackend):
    """VLM backend backed by a local vLLM engine.

    The estimator always sends the *same* images with *many* prompts (one per
    grounded predicate); vLLM batches them in a single ``chat`` call and shards
    each layer across GPUs under the hood, so the user can treat a multi-GPU box
    as one big GPU.

    Args:
        model_id: HuggingFace model identifier loaded by vLLM.
        tensor_parallel_size: Number of GPUs to shard each layer across. When
            ``None`` (default), uses all locally visible GPUs
            (``torch.cuda.device_count()``), so the user need not specify a
            count. Multi-node execution is intentionally out of scope but can be
            reached by passing vLLM engine kwargs (e.g. ``pipeline_parallel_size``,
            ``distributed_executor_backend``) through ``engine_kwargs``.
        num_logprobs: Number of top tokens to include in ``token_probs``.
            ``None`` (default) returns the full-vocabulary distribution,
            mirroring :class:`HuggingFaceVLM`. This maps to vLLM ``logprobs=-1``
            with the engine built using ``max_logprobs=-1`` -- note the docs' OOM
            caveat for full-vocab logprobs. A finite ``k`` returns the top ``k``.
        **engine_kwargs: Forwarded verbatim to the ``vllm.LLM`` constructor
            (e.g. ``gpu_memory_utilization``, ``dtype``, ``max_model_len``,
            ``quantization``), mirroring how :class:`HuggingFaceVLM` forwards
            ``**model_kwargs``.

    Notes:
        There is intentionally no ``max_new_tokens`` parameter: the HF backend
        stores one but never forwards it, so it is dead state we do not
        reproduce. Generation length is controlled through ``inference_kwargs``
        (vLLM ``SamplingParams.max_tokens``).

        This backend relies on the model's chat template (see
        :meth:`query_batch`); a model without one is unsupported on this path.
    """

    def __init__(
        self,
        model_id: str,
        tensor_parallel_size: int | None = None,
        num_logprobs: int | None = None,
        **engine_kwargs,
    ):
        _check_vllm_installed()
        self.model_id = model_id
        self.num_logprobs = num_logprobs

        # Default to every locally visible GPU so the user does not have to
        # specify a count; an explicit value always wins. max(..., 1) keeps a
        # sane value when no CUDA device is visible.
        if tensor_parallel_size is None:
            tensor_parallel_size = max(torch.cuda.device_count(), 1)
        self.tensor_parallel_size = tensor_parallel_size

        # max_logprobs is fixed at engine-construction time, and num_logprobs is
        # also a constructor arg, so the two are always consistent. None means
        # "full vocab" (-1), matching HuggingFaceVLM's default; a finite k caps
        # at k. Requesting more logprobs per call than max_logprobs makes vLLM
        # raise, which is why we size the cap from num_logprobs here.
        self.llm = LLM(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            max_logprobs=(-1 if num_logprobs is None else num_logprobs),
            **engine_kwargs,
        )

    def query(
        self,
        images,
        prompt,
        system_prompt=None,
        generate=False,
        **inference_kwargs,
    ) -> VLMOutput:
        """Send a single query to the vLLM backend."""
        del images
        del prompt
        del system_prompt
        del generate
        del inference_kwargs
        raise NotImplementedError("VLLMBackend.query will be implemented in Task 3.")
