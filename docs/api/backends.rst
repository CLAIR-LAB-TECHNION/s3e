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

Thinking models
---------------

Some models reason in a delimited block before they answer (e.g.
``<think>...</think>``), opened by the chat template (Qwen3.5-4B and larger),
by a default system prompt (Mistral reasoning models), or on request (Gemma 4
with ``enable_thinking=True``). :class:`HuggingFaceVLM` and
:class:`VLLMBackend` detect the block and read the answer after it; their
``chat_template_kwargs``, ``thinking``, ``max_thinking_tokens``, and
``thinking_delimiters`` arguments control how, and
:attr:`VLMOutput.reasoning` holds the block's text. The user guide's
*Thinking models* section covers the options and the models they do not
cover.

.. autodata:: s3e.backends.utils.thinking.KNOWN_THINKING_DELIMITERS
   :no-value:

Helpers for backends that skip thinking blocks (see the developer guide):

.. autofunction:: s3e.backends.utils.thinking.resolve_thinking_tokens

.. autoclass:: s3e.backends.utils.thinking.ThinkingTokens
   :members: opens_block, closing_text, split

.. autofunction:: s3e.backends.utils.thinking.leading_whitespace_tokens
