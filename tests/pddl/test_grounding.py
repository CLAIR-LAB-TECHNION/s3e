"""Tests for the public PDDL grounding surface."""

from s3e.pddl import compute_domain_fingerprint, ground_predicates, parse_domain_problem

from conftest import BLOCKSWORLD_DOMAIN, BLOCKSWORLD_PROBLEM  # existing fixtures/constants


class TestParseAndGround:
    def test_parse_domain_problem_returns_problem(self):
        up_problem = parse_domain_problem(BLOCKSWORLD_DOMAIN, BLOCKSWORLD_PROBLEM)
        assert up_problem.fluents

    def test_ground_predicates_enumerates_atoms(self):
        up_problem = parse_domain_problem(BLOCKSWORLD_DOMAIN, BLOCKSWORLD_PROBLEM)
        grounded = ground_predicates(up_problem)
        assert any(p.startswith("on(") for p in grounded)


class TestFingerprintMoved:
    def test_fingerprint_importable_from_pddl(self):
        assert len(compute_domain_fingerprint(BLOCKSWORLD_DOMAIN)) == 64
