"""s3e — Semantic Symbolic State Estimation with vision-language models.

Quick start::

    from s3e import SemanticStateEstimator, TemplateTranslator

    estimator = SemanticStateEstimator.from_pddl(
        domain_pddl, problem_pddl,
        vlm="HuggingFaceTB/SmolVLM-256M-Instruct",
        translator=TemplateTranslator({"on": "Is {0} on {1}?"}),
    )
    state = estimator(images)  # dict[str, bool | None]
"""

from importlib.metadata import PackageNotFoundError, version

from .backends import VLMBackend, VLMOutput, resolve_backend
from .calibration import (
    CalibrationExample,
    CalibrationSample,
    CalibrationSet,
    Calibrator,
    PlattCalibrator,
)
from .engine import (
    AnswerOption,
    AnswerSpace,
    BinaryAnswers,
    CategoricalAnswers,
    Prediction,
    PredictionSet,
    QueryEngine,
)
from .estimator import SemanticStateEstimator
from .translation import (
    IdentityTranslator,
    LLMTranslator,
    PrewrittenTranslator,
    QueryTranslator,
    TemplateTranslator,
)

try:
    __version__ = version("s3e")
except PackageNotFoundError:  # running from a source tree
    __version__ = "0.0.0.dev0"

__all__ = [
    "SemanticStateEstimator",
    "QueryEngine",
    "AnswerOption",
    "AnswerSpace",
    "BinaryAnswers",
    "CategoricalAnswers",
    "Prediction",
    "PredictionSet",
    "VLMBackend",
    "VLMOutput",
    "resolve_backend",
    "HuggingFaceVLM",
    "OpenAIVLM",
    "VLLMBackend",
    "QueryTranslator",
    "IdentityTranslator",
    "TemplateTranslator",
    "PrewrittenTranslator",
    "LLMTranslator",
    "Calibrator",
    "PlattCalibrator",
    "CalibrationSet",
    "CalibrationSample",
    "CalibrationExample",
]

_LAZY_TOP_LEVEL = {"HuggingFaceVLM", "OpenAIVLM", "VLLMBackend"}


def __getattr__(name: str):
    """Lazily expose optional backends without importing their packages."""
    if name in _LAZY_TOP_LEVEL:
        import s3e.backends as _backends

        return getattr(_backends, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
