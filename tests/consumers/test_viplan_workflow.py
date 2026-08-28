"""Contract tests for the ViPlan++ workflow (mpst_exp/predict.py, estimators.py).

ViPlan pattern: a shared prebuilt backend feeds estimators built per domain;
an adapter prepares (build or set_problem) per episode and estimates a
relevant-atom subset; payloads read per-predicate detail fields; backend
type checks must work without importing vllm.
"""

import json
import sys

import pytest

from s3e import SemanticStateEstimator, TemplateTranslator, resolve_backend
from s3e.backends import VLMBackend

from conftest import BLOCKSWORLD_DOMAIN, BLOCKSWORLD_PROBLEM, make_blank_image
from fakes import FakeVLM

TEMPLATES = {"on": "Is {0} on {1}?", "clear": "Is {0} clear?"}


def build_estimator(vlm):
    return SemanticStateEstimator.from_pddl(
        BLOCKSWORLD_DOMAIN, BLOCKSWORLD_PROBLEM,
        vlm=vlm, translator=TemplateTranslator(TEMPLATES),
        batch_size=1,
        inference_kwargs={"temperature": 0.0},
    )


class TestSharedBackend:
    def test_one_backend_many_estimators(self):
        shared = FakeVLM()
        a, b = build_estimator(shared), build_estimator(shared)
        assert a.engine.backend is b.engine.backend is shared

    def test_resolve_backend_is_public(self):
        assert callable(resolve_backend)
        assert resolve_backend(FakeVLM()) is not None


class TestAdapterPattern:
    def test_prepare_then_estimate_subset(self):
        """The S3EAdapter shape: lazy build, then set_problem per episode."""
        estimator = None
        for _episode in range(2):
            if estimator is None:
                estimator = build_estimator(FakeVLM())
            else:
                estimator.set_problem(BLOCKSWORLD_DOMAIN, BLOCKSWORLD_PROBLEM)
            relevant = estimator.predicates[:2]
            details = estimator.estimate([make_blank_image()], predicates=relevant)
            assert set(details) == set(relevant)

    def test_inference_kwargs_reach_backend(self):
        fake = FakeVLM()
        build_estimator(fake).estimate([make_blank_image()])
        assert fake.calls[0]["inference_kwargs"] == {"temperature": 0.0}


class TestPayloadFields:
    def test_every_field_the_payload_builder_reads(self):
        results = build_estimator(FakeVLM()).estimate([make_blank_image()])
        predicate = next(iter(results))
        p = results[predicate]
        payload = {
            "probability": p.probability,
            "score": p.score,
            "masses": p.masses,
            "null_mass": p.null_mass,
            "unassigned_mass": p.unassigned_mass,
            "null_dominated": p.null_dominated,
            "argmax_in_interest": p.argmax_in_interest,
            "answer": p.answer,
        }
        json.dumps(payload)  # must be JSON-serializable as ViPlan writes it


class TestBackendDetectionWithoutVllm:
    def test_isinstance_check_without_importing_vllm(self):
        # Another test in the session may already have imported vllm; the
        # guarantee here is that the isinstance check itself never does.
        vllm_already_loaded = "vllm" in sys.modules
        backend = FakeVLM()
        assert isinstance(backend, VLMBackend)
        assert ("vllm" in sys.modules) == vllm_already_loaded
