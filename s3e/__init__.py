"""s3e — Semantic State Estimation using vision-language models.

This package provides tools for estimating the boolean truth values of
PDDL predicates from images using vision-language models (VLMs).

Quick start::

    from s3e import SemanticStateEstimator

    se = SemanticStateEstimator(domain_pddl, problem_pddl, vlm="Qwen/Qwen2-VL-7B-Instruct")
    state = se(images)  # dict[str, bool | None]

See the README for full documentation and examples.
"""

from .state_estimator import StateEstimator, ProbabilisticStateEstimator
from .semantic_state_estimator import PredicatePredictionDetails, SemanticStateEstimator
from .calibration import CalibrationExample, PlattCalibrationSample
from .vlm import VLMBackend, VLMOutput, HuggingFaceVLM, OpenAIVLM, VLLMBackend
from .translation import (
    QueryTranslator,
    IdentityTranslator,
    PrewrittenTranslator,
    TemplateTranslator,
    LLMTranslator,
)

__version__ = "0.2.0"

__all__ = [
    "StateEstimator",
    "ProbabilisticStateEstimator",
    "SemanticStateEstimator",
    "PredicatePredictionDetails",
    "CalibrationExample",
    "PlattCalibrationSample",
    "VLMBackend",
    "VLMOutput",
    "HuggingFaceVLM",
    "OpenAIVLM",
    "VLLMBackend",
    "QueryTranslator",
    "IdentityTranslator",
    "PrewrittenTranslator",
    "TemplateTranslator",
    "LLMTranslator",
]
