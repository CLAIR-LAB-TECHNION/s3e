Calibration
===========

.. currentmodule:: s3e

Calibration is split into an expensive step that queries the VLM once on
labeled examples (:meth:`CalibrationSet.collect`) and a cheap, offline,
VLM-free fit (:meth:`PlattCalibrator.fit`, requires the ``calibration``
extra). Both the collected data and fitted calibrators round-trip through
JSON.

.. autoclass:: Calibrator

.. autoclass:: PlattCalibrator

.. autoclass:: CalibrationExample

.. autoclass:: CalibrationSample

.. autoclass:: CalibrationSet

Platt scaling helpers
---------------------

.. currentmodule:: s3e.calibration.platt

.. autoclass:: PlattParameters

.. autofunction:: grouped_log_odds

.. autofunction:: apply_platt_scaling

.. autofunction:: fit_platt_parameters
