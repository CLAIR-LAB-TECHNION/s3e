# s3e/calibration/__init__.py
"""Calibration: collect scored examples once, fit and apply calibrators offline."""

from .base import Calibrator
from .data import CalibrationExample, CalibrationSample, CalibrationSet
from .platt import PlattCalibrator

# Legacy names still consumed by s3e/semantic_state_estimator.py until Task 11:
from .platt import (
    GLOBAL_CALIBRATION_KEY,
    PLATT_CALIBRATION_DATA_SCHEMA_VERSION,
    PlattParameters,
    PlattScalingProfile,
    apply_platt_scaling,
    fit_platt_parameters,
    grouped_log_odds,
)
from .data import CalibrationSample as PlattCalibrationSample
from ..pddl.fingerprint import compute_domain_fingerprint

__all__ = [
    "Calibrator",
    "CalibrationExample",
    "CalibrationSample",
    "CalibrationSet",
    "PlattCalibrator",
]
