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
except ModuleNotFoundError as exc:
    if exc.name != "vllm":
        raise
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
        """Send a single query to the vLLM engine."""
        return self.query_batch(
            images, [prompt], system_prompt, generate, **inference_kwargs
        )[0]

    def query_batch(
        self,
        images,
        prompts,
        system_prompt=None,
        generate=False,
        **inference_kwargs,
    ) -> list[VLMOutput]:
        """Send multiple prompts against the same images in one batched call.

        We use ``LLM.chat`` (not ``LLM.generate`` + ``multi_modal_data``) so
        vLLM applies the model's own chat template and performs the
        resolution-dependent image-placeholder expansion -- mirroring how
        :class:`HuggingFaceVLM` relies on ``processor.apply_chat_template`` and
        owning zero per-model placeholder logic. The trade-off: a model without
        a chat template is unsupported here.
        """
        if not prompts:
            return []

        sampling_params = self._build_sampling_params(generate, **inference_kwargs)

        # Every prompt shares the same images, so build the image content once
        # and reuse it across the per-prompt conversations.
        image_content = [{"type": "image_pil", "image_pil": image} for image in images]
        conversations = []
        for prompt in prompts:
            messages = []
            if system_prompt is not None:
                messages.append({"role": "system", "content": system_prompt})
            messages.append(
                {
                    "role": "user",
                    "content": image_content + [{"type": "text", "text": prompt}],
                }
            )
            conversations.append(messages)

        # chat() accepts a batched list of conversations and returns outputs in
        # input order.
        outputs = self.llm.chat(conversations, sampling_params)
        return [self._to_vlm_output(output, generate) for output in outputs]

    def _build_sampling_params(self, generate, **inference_kwargs):
        """Build vLLM SamplingParams for the requested mode.

        ``inference_kwargs`` are user overrides. We only *force* the settings
        the estimator's contract depends on and use ``setdefault`` for the rest
        -- the same pattern OpenAIVLM uses to force logprobs and
        HuggingFaceVLM uses to default ``logits_to_keep=1``.
        """
        if generate:
            # Text mode: do NOT bound max_tokens, so a model can reason as it
            # sees fit (the OpenAI backend documents the same philosophy).
            inference_kwargs.setdefault("temperature", 0.0)
        else:
            # Logprobs mode: one decode step is the memory-minimal default and
            # the direct analog of HuggingFaceVLM's logits_to_keep=1. It is a
            # setdefault, not a force, so a user may override it. Free-form
            # reasoning belongs in text mode, not here.
            inference_kwargs.setdefault("max_tokens", 1)
            inference_kwargs.setdefault("temperature", 0.0)
            # Force logprobs on (analog of OpenAIVLM forcing logprobs=True).
            # -1 == full vocab; otherwise the configured top-k.
            inference_kwargs["logprobs"] = (
                -1 if self.num_logprobs is None else self.num_logprobs
            )
        return SamplingParams(**inference_kwargs)

    def _to_vlm_output(self, output, generate):
        """Convert one vLLM RequestOutput into a :class:`VLMOutput`."""
        completion = output.outputs[0]
        if generate:
            return VLMOutput(token_probs=None, text=completion.text)

        # Logprobs mode reads the distribution over the next answer token. With
        # the default max_tokens=1 this is the only generated position and
        # reproduces HuggingFaceVLM's single-forward next-token distribution.
        logprobs_seq = completion.logprobs
        if not logprobs_seq or logprobs_seq[0] is None:
            # With logprobs forced and max_tokens>=1 this should never happen.
            # Treat it as an internal inconsistency rather than silently
            # returning empty probabilities, because downstream calibration
            # expects token_probs to be the requested next-token distribution.
            raise RuntimeError(
                "vLLM returned no logprobs for a request despite logprobs being "
                "requested. This indicates an internal inconsistency in the vLLM "
                "output; check the installed vLLM version and SamplingParams."
            )

        token_probs: dict[str, float] = {}
        for logprob in logprobs_seq[0].values():
            # Sum probabilities of duplicate decoded token strings, matching the
            # dedup HuggingFaceVLM and OpenAIVLM perform. Keeping the same
            # VLMOutput.token_probs shape is important because Platt calibration
            # consumes these per-token probabilities without backend-specific
            # handling.
            token = logprob.decoded_token
            token_probs[token] = token_probs.get(token, 0.0) + math.exp(
                logprob.logprob
            )

        return VLMOutput(token_probs=token_probs, text=None)
