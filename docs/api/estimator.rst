PDDL facade
===========

.. currentmodule:: s3e

:class:`SemanticStateEstimator` wires grounded predicates, a
:class:`QueryTranslator`, and a :class:`QueryEngine` into symbolic state
estimation. Build it from PDDL with :meth:`SemanticStateEstimator.from_pddl`
(requires the ``pddl`` extra) or from an explicit list of grounded predicate
strings.

.. autoclass:: SemanticStateEstimator(predicates, *, vlm, translator=None, answers=None, system_prompt=None, prompt_template=None, additional_instructions=None, confidence=0.5, scoring="logprobs", batch_size=8, vlm_kwargs=None, inference_kwargs=None, true_tokens=None, false_tokens=None, null_tokens=None)
   :special-members: __call__
