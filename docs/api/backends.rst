Backends
========

.. currentmodule:: s3e

Every model is accessed through the :class:`VLMBackend` interface. Pass a
backend instance, or a model-id string resolved by :func:`resolve_backend`
(``OpenAI/``-prefixed ids select :class:`OpenAIVLM`; any other string selects
:class:`HuggingFaceVLM`). Construct :class:`VLLMBackend` explicitly.

Interface
---------

.. autoclass:: VLMBackend

.. autoclass:: VLMOutput

.. autofunction:: resolve_backend

Implementations
---------------

:class:`HuggingFaceVLM` needs the ``hf`` extra, :class:`OpenAIVLM` the
``openai`` extra, and :class:`VLLMBackend` the ``vllm`` extra (CUDA only).

.. autoclass:: HuggingFaceVLM

.. autoclass:: OpenAIVLM

.. autoclass:: VLLMBackend
