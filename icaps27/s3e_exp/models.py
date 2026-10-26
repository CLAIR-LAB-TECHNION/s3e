# SPDX-License-Identifier: MIT
"""Model registry, backend construction, inference timing, and a fake model.

``configs/models.json`` names every model the experiments use, with its
exact Hugging Face revision, so a run can cite "model id + commit". Any
other model id can be passed directly (it then runs at the Hub's current
revision, which the provenance records).
"""

import json
import time
import zlib
from pathlib import Path

from s3e import VLMBackend, VLMOutput

REGISTRY_PATH = Path(__file__).resolve().parent.parent / "configs" / "models.json"
BACKENDS = ("hf", "vllm", "openai", "fake")
_DTYPES = ("float16", "bfloat16", "float32")


def load_registry(path: "Path | None" = None) -> dict:
    data = json.loads(Path(path or REGISTRY_PATH).read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def model_spec(name: str, registry: "dict | None" = None) -> dict:
    """The registry entry for ``name``, or an ad-hoc spec for a raw model id."""
    registry = load_registry() if registry is None else registry
    if name in registry:
        return {"name": name, **registry[name]}
    if name == "fake" or name.startswith("fake:"):
        return {"name": name, "id": name, "backend": "fake"}
    backend = "openai" if name.startswith("OpenAI/") else "hf"
    return {"name": name.replace("/", "--"), "id": name, "backend": backend}


def _torch_dtype(value):
    if isinstance(value, str) and value in _DTYPES:
        import torch

        return getattr(torch, value)
    return value


def build_backend(spec: dict, backend: "str | None" = None, **overrides) -> VLMBackend:
    """Construct the backend ``spec`` describes (``backend`` overrides its kind)."""
    kind = backend or spec.get("backend", "hf")
    if kind not in BACKENDS:
        raise ValueError(f"Unknown backend {kind!r}; expected one of {BACKENDS}")
    if kind == "fake":
        return OracleVLM(**{**spec.get("fake", {}), **overrides})
    if kind == "openai":
        from s3e import OpenAIVLM

        return OpenAIVLM(spec["id"], **{**spec.get("openai_kwargs", {}), **overrides})
    kwargs = dict(spec.get(f"{kind}_kwargs", {}))
    if spec.get("revision"):
        kwargs.setdefault("revision", spec["revision"])
    kwargs.update(overrides)
    if kind == "hf":
        from s3e import HuggingFaceVLM

        if "torch_dtype" in kwargs:
            kwargs["torch_dtype"] = _torch_dtype(kwargs["torch_dtype"])
        return HuggingFaceVLM(spec["id"], **kwargs)
    import os

    # vLLM's FlashInfer sampler JIT-compiles CUDA code (needs ninja and nvcc);
    # the PyTorch sampler needs neither and S3E reads logprobs, not samples.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from s3e import VLLMBackend

    return VLLMBackend(spec["id"], **kwargs)


def default_generation_kwargs(backend: VLMBackend, max_tokens: int = 8) -> dict:
    """A cap on generated replies for text scoring; the kwarg name is backend specific."""
    name = type(getattr(backend, "inner", backend)).__name__
    if name == "HuggingFaceVLM":
        return {"max_new_tokens": max_tokens}
    if name == "VLLMBackend":
        return {"max_tokens": max_tokens}
    if name == "OpenAIVLM":
        return {"max_completion_tokens": max_tokens}
    return {}


class TimedBackend(VLMBackend):
    """Wraps a backend and accumulates time spent inside it.

    The difference between an estimate's wall clock and ``seconds`` is the
    library's own overhead (grounding, translation, scoring).
    """

    def __init__(self, inner: VLMBackend):
        self.inner = inner
        self.seconds = 0.0
        self.prompts = 0
        self.calls = 0

    def __getattr__(self, name):
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    def reset(self) -> None:
        self.seconds, self.prompts, self.calls = 0.0, 0, 0

    def query(self, images, prompt, system_prompt=None, generate=False,
              interest_tokens=None, **inference_kwargs):
        return self.query_batch(images, [prompt], system_prompt, generate,
                                interest_tokens=interest_tokens, **inference_kwargs)[0]

    def query_batch(self, images, prompts, system_prompt=None, generate=False,
                    interest_tokens=None, **inference_kwargs):
        start = time.perf_counter()
        outputs = self.inner.query_batch(images, prompts, system_prompt, generate,
                                         interest_tokens=interest_tokens, **inference_kwargs)
        self.seconds += time.perf_counter() - start
        self.prompts += len(prompts)
        self.calls += 1
        return outputs

    def unsupported_interest_tokens(self, tokens):
        return self.inner.unsupported_interest_tokens(tokens)


_TRUE_WORDS = {"yes", "true"}
_FALSE_WORDS = {"no", "false"}


class OracleVLM(VLMBackend):
    """A deterministic stand-in model that knows the truth, for tests and dry runs.

    Before each estimate the harness calls :meth:`prepare` with the truth
    of every prompt. Each prompt is answered correctly with probability
    ``accuracy`` (decided by a hash of the prompt and ``seed``, so runs are
    reproducible), with ``confidence`` of the mass on the chosen answer and a
    little mass left outside the answer tokens. In generate mode it replies
    with the word, or with a sentence that contains it (``chatty`` share).
    """

    model_id = "fake-oracle"

    def __init__(self, accuracy: float = 0.9, confidence: float = 0.85,
                 chatty: float = 0.2, seed: int = 0):
        self.accuracy = accuracy
        self.confidence = confidence
        self.chatty = chatty
        self.seed = seed
        self.truth: dict[str, bool] = {}

    def prepare(self, truth_by_prompt: dict[str, bool]) -> None:
        self.truth.update(truth_by_prompt)

    def _draw(self, prompt: str, salt: str) -> float:
        return zlib.crc32(f"{self.seed}:{salt}:{prompt}".encode()) / 2**32

    def _answer(self, images, prompt: str) -> bool:
        truth = self.truth.get(prompt, False)
        # Images matter: different views give partly independent errors.
        view = ",".join(f"{zlib.crc32(img.tobytes()):x}" for img in images)
        correct = self._draw(prompt, f"correct:{view}") < self.accuracy
        return truth if correct else not truth

    def query(self, images, prompt, system_prompt=None, generate=False,
              interest_tokens=None, **inference_kwargs):
        answer = self._answer(images, prompt)
        if generate:
            word = "Yes" if answer else "No"
            chatty = self._draw(prompt, "chatty") < self.chatty
            return VLMOutput(token_probs=None,
                             text=f"I think the answer is {word.lower()}." if chatty else word)
        mass = self.confidence * (0.8 + 0.2 * self._draw(prompt, "mass"))
        other = 0.95 - mass
        probs = {}
        for token in interest_tokens or ("yes", "no"):
            word = token.strip().lower()
            if word in _TRUE_WORDS:
                probs[token] = mass if answer else other
            elif word in _FALSE_WORDS:
                probs[token] = other if answer else mass
            else:
                probs[token] = 0.0
        # Mass is split over a label's variants, as a real tokenizer would.
        totals: dict[str, int] = {}
        for token in probs:
            totals[token.strip().lower()] = totals.get(token.strip().lower(), 0) + 1
        probs = {t: p / totals[t.strip().lower()] for t, p in probs.items()}
        return VLMOutput(token_probs=probs, argmax_in_interest=True)
