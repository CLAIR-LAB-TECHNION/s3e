Engine
======

.. currentmodule:: s3e

The PDDL-free core: a :class:`QueryEngine` answers free-form queries about a
scene against an answer space and returns lazily evaluated predictions.

Query engine
------------

.. autoclass:: QueryEngine

.. autoclass:: UnmatchedAnswerWarning

.. autoclass:: ThinkingTruncatedWarning

Answer spaces
-------------

.. autoclass:: AnswerOption

.. autoclass:: AnswerSpace

.. autoclass:: BinaryAnswers

.. autoclass:: CategoricalAnswers

Results
-------

.. autoclass:: Prediction

.. autoclass:: PredictionSet
