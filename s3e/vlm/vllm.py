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
* Requires vLLM >= 0.11.0: full-vocab logprobs via ``SamplingParams(logprobs=-1)``
  and ``LLM(max_logprobs=-1)`` landed in 0.11.0 (vllm-project/vllm#25031); older
  engines reject ``-1`` at construction. The pin lives in ``pyproject.toml``.
"""

import math

import torch

from .backend import VLMBackend, VLMOutput
from .token_index import build_token_reverse_index, decode_single_token_ids

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
            With ``interest_tokens`` queries this stays the throughput knob:
            ``None`` keeps the id-matched masses exact over the full
            vocabulary (at the cost of a Python loop over every returned
            entry), while a finite ``k`` bounds the omitted interest mass by
            the ``k``-th ranked probability. Interest mode also requests
            ``detokenize=False``; whether the installed vLLM honors it for
            logprobs can be smoke-tested by checking that returned
            ``Logprob.decoded_token`` values are ``None``.
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
        self._token_reverse_index: dict[str, list[int]] | None = None

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
        images: list,
        prompt: str,
        system_prompt: str | None = None,
        generate: bool = False,
        interest_tokens=None,
        **inference_kwargs,
    ) -> VLMOutput:
        """Send a single query to the vLLM engine."""
        return self.query_batch(
            images,
            [prompt],
            system_prompt,
            generate,
            interest_tokens=interest_tokens,
            **inference_kwargs,
        )[0]

    def query_batch(
        self,
        images: list,
        prompts: list[str],
        system_prompt: str | None = None,
        generate: bool = False,
        interest_tokens=None,
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

        interest_mode = interest_tokens is not None and not generate
        sampling_params = self._build_sampling_params(
            generate, interest_mode, **inference_kwargs
        )

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
        return [
            self._to_vlm_output(output, generate, interest_tokens)
            for output in outputs
        ]

    def _build_sampling_params(
        self, generate: bool, interest_mode: bool = False, **inference_kwargs
    ) -> "SamplingParams":
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
            if interest_mode and "stop" not in inference_kwargs:
                # Interest tokens are matched by token id, so the returned
                # logprob entries never need decoded strings. Skipping
                # detokenization removes vLLM's per-id string conversion on
                # the output-processor thread — the dominant cost of
                # full-vocabulary logprobs. Engines that ignore the flag for
                # logprobs still return id-keyed entries, so results stay
                # correct either way, just slower. Stop strings require
                # detokenization, so an explicit ``stop`` wins.
                inference_kwargs.setdefault("detokenize", False)
        return SamplingParams(**inference_kwargs)

    def _to_vlm_output(
        self, output, generate: bool, interest_tokens=None
    ) -> VLMOutput:
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

        if interest_tokens is not None:
            return self._interest_output(logprobs_seq[0], interest_tokens)

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

    def _interest_output(self, logprob_entries, interest_tokens) -> VLMOutput:
        """Build a VLMOutput by matching id-keyed logprob entries.

        Matching by token id (the logprobs dict key) instead of
        ``decoded_token`` works with ``detokenize=False`` output and sums
        duplicate ids decoding to the same string, exactly like the
        string-keyed path.
        """
        interest = list(dict.fromkeys(interest_tokens))
        id_map = self._get_interest_id_map(interest)

        token_probs = {token: 0.0 for token in interest}
        best_id = None
        best_logprob = -math.inf
        for token_id, logprob in logprob_entries.items():
            if logprob.logprob > best_logprob:
                best_id, best_logprob = token_id, logprob.logprob
            token = id_map.get(token_id)
            if token is not None:
                token_probs[token] += math.exp(logprob.logprob)

        return VLMOutput(
            token_probs=token_probs,
            text=None,
            argmax_in_interest=best_id in id_map,
        )

    def _get_token_reverse_index(self) -> dict[str, list[int]]:
        """Build (once) the decoded-string -> ids index from the tokenizer."""
        if self._token_reverse_index is None:
            tokenizer = self.llm.get_tokenizer()
            self._token_reverse_index = build_token_reverse_index(
                lambda ids: decode_single_token_ids(tokenizer, ids),
                len(tokenizer),
            )
        return self._token_reverse_index

    def _get_interest_id_map(self, interest: list[str]) -> dict[int, str]:
        """Map each vocabulary id decoding to an interest token onto it."""
        index = self._get_token_reverse_index()
        id_map: dict[int, str] = {}
        for token in interest:
            for token_id in index.get(token, []):
                id_map[token_id] = token
        return id_map
