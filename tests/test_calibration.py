"""Tests for low-level Platt calibration helpers."""

import math

import pytest
from unified_planning.io import PDDLReader

from s3e.calibration import (
    CalibrationExample,
    PlattParameters,
    PlattScalingProfile,
    apply_platt_scaling,
    compute_domain_fingerprint,
    grouped_log_odds,
)


class TestGroupedLogOdds:
    def test_uses_configured_true_and_false_groups(self):
        token_probs = {"true": 0.7, "TRUE": 0.2, "false": 0.1, "other": 0.5}
        score = grouped_log_odds(token_probs, ["true", "TRUE"], ["false"])
        assert score == pytest.approx(math.log(0.9 / 0.1))

    def test_clamps_zero_mass_with_epsilon(self):
        score = grouped_log_odds({"true": 1.0}, ["true"], ["false"])
        assert math.isfinite(score)
        assert score > 0


class TestSigmoidApplication:
    def test_matches_platt_formula(self):
        params = PlattParameters(
            a=1.5,
            b=-0.5,
            sample_count=8,
            positive_count=3,
            negative_count=5,
        )
        probability = apply_platt_scaling(0.7, params)
        expected = 1.0 / (1.0 + math.exp(1.5 * 0.7 - 0.5))
        assert probability == pytest.approx(expected)


_MINIMAL_DOMAIN = (
    "(define (domain blocksworld) (:requirements :strips) (:predicates (p)))"
)


class TestProfileSerialization:
    def test_profile_round_trips_through_dict(self):
        profile = PlattScalingProfile(
            scope="global",
            probability_method="logprobs",
            true_tokens=["true"],
            false_tokens=["false"],
            domain_fingerprint=compute_domain_fingerprint(_MINIMAL_DOMAIN),
            score_kind="grouped_log_odds",
            groups={
                "__global__": PlattParameters(
                    a=0.9,
                    b=-0.2,
                    sample_count=6,
                    positive_count=2,
                    negative_count=4,
                )
            },
        )
        restored = PlattScalingProfile.from_dict(profile.to_dict())
        assert restored == profile

    def test_rejects_unsupported_schema_version(self):
        profile = PlattScalingProfile(
            scope="global",
            probability_method="logprobs",
            true_tokens=["true"],
            false_tokens=["false"],
            domain_fingerprint=compute_domain_fingerprint(_MINIMAL_DOMAIN),
            score_kind="grouped_log_odds",
            groups={},
        )
        payload = profile.to_dict()
        payload["schema_version"] = 99

        with pytest.raises(ValueError, match="Unsupported calibration schema version: 99"):
            PlattScalingProfile.from_dict(payload)


def _make_blocksworld(
    actions: str = "",
    constants: str = "",
    requirements: str = ":strips :typing",
    domain_name: str = "blocksworld",
) -> str:
    constants_section = f"(:constants {constants})" if constants else ""
    return (
        f"(define (domain {domain_name})"
        f"  (:requirements {requirements})"
        "  (:types block)"
        f"  {constants_section}"
        "  (:predicates (on ?x - block ?y - block) (clear ?x - block))"
        f"  {actions})"
    )


class TestDomainFingerprint:
    DOMAIN_A = _make_blocksworld()

    def test_ignores_whitespace_differences(self):
        spaced = (
            "(define  (domain blocksworld)\n"
            "  (:requirements :strips :typing)\n"
            "  (:types block)\n"
            "  (:predicates\n"
            "    (on ?x - block ?y - block)\n"
            "    (clear ?x - block)\n"
            "  )\n"
            ")"
        )
        assert compute_domain_fingerprint(self.DOMAIN_A) == compute_domain_fingerprint(spaced)

    def test_ignores_domain_name(self):
        other_name = _make_blocksworld(domain_name="blocks")
        assert compute_domain_fingerprint(self.DOMAIN_A) == compute_domain_fingerprint(other_name)

    def test_ignores_comments(self):
        commented = "; blocksworld domain\n" + self.DOMAIN_A
        assert compute_domain_fingerprint(self.DOMAIN_A) == compute_domain_fingerprint(commented)

    def test_ignores_predicate_order(self):
        reordered = (
            "(define (domain blocksworld)"
            "  (:requirements :strips :typing)"
            "  (:types block)"
            "  (:predicates (clear ?x - block) (on ?x - block ?y - block)))"
        )
        assert compute_domain_fingerprint(self.DOMAIN_A) == compute_domain_fingerprint(reordered)

    def test_ignores_action_order(self):
        action_a = "(:action a :parameters (?x - block) :precondition (clear ?x) :effect (not (clear ?x)))"
        action_b = "(:action b :parameters (?x - block) :precondition (not (clear ?x)) :effect (clear ?x))"
        domain_ab = _make_blocksworld(action_a + action_b)
        domain_ba = _make_blocksworld(action_b + action_a)
        assert compute_domain_fingerprint(domain_ab) == compute_domain_fingerprint(domain_ba)

    def test_ignores_and_conjunct_order(self):
        action_xy = (
            "(:action a :parameters (?x - block ?y - block)"
            "  :precondition (and (clear ?x) (clear ?y))"
            "  :effect (and (not (clear ?x)) (not (clear ?y))))"
        )
        action_yx = (
            "(:action a :parameters (?x - block ?y - block)"
            "  :precondition (and (clear ?y) (clear ?x))"
            "  :effect (and (not (clear ?y)) (not (clear ?x))))"
        )
        assert compute_domain_fingerprint(_make_blocksworld(action_xy)) == compute_domain_fingerprint(
            _make_blocksworld(action_yx)
        )

    def test_ignores_type_declaration_order(self):
        domain_ab = (
            "(define (domain d) (:requirements :strips :typing)"
            "  (:types a b - object) (:predicates (p ?x - a)))"
        )
        domain_ba = (
            "(define (domain d) (:requirements :strips :typing)"
            "  (:types b a - object) (:predicates (p ?x - a)))"
        )
        assert compute_domain_fingerprint(domain_ab) == compute_domain_fingerprint(domain_ba)

    def test_ignores_case_differences(self):
        upper = (
            "(define (domain BLOCKSWORLD)"
            "  (:requirements :strips :typing)"
            "  (:types Block)"
            "  (:predicates (On ?X - Block ?Y - Block) (Clear ?X - Block)))"
        )
        assert compute_domain_fingerprint(self.DOMAIN_A) == compute_domain_fingerprint(upper)

    def test_ignores_forall_bound_variable_order(self):
        reqs = ":strips :typing :universal-preconditions"
        action_xy = (
            "(:action a :parameters ()"
            "  :precondition (forall (?x - block ?y - block) (on ?x ?y))"
            "  :effect (clear b1))"
        )
        action_yx = (
            "(:action a :parameters ()"
            "  :precondition (forall (?y - block ?x - block) (on ?x ?y))"
            "  :effect (clear b1))"
        )
        assert compute_domain_fingerprint(
            _make_blocksworld(actions=action_xy, constants="b1 - block", requirements=reqs)
        ) == compute_domain_fingerprint(
            _make_blocksworld(actions=action_yx, constants="b1 - block", requirements=reqs)
        )

    def test_ignores_conditional_effect_condition_conjunct_order(self):
        reqs = ":strips :typing :conditional-effects"
        action_pq = (
            "(:action a :parameters (?x - block ?y - block)"
            "  :precondition ()"
            "  :effect (when (and (clear ?x) (clear ?y)) (on ?x ?y)))"
        )
        action_qp = (
            "(:action a :parameters (?x - block ?y - block)"
            "  :precondition ()"
            "  :effect (when (and (clear ?y) (clear ?x)) (on ?x ?y)))"
        )
        assert compute_domain_fingerprint(
            _make_blocksworld(actions=action_pq, requirements=reqs)
        ) == compute_domain_fingerprint(
            _make_blocksworld(actions=action_qp, requirements=reqs)
        )

    def test_unreferenced_constants_skipped_calibration_compat(self):
        with_const = _make_blocksworld(constants="unused - block")
        without_const = self.DOMAIN_A
        assert compute_domain_fingerprint(with_const) == compute_domain_fingerprint(without_const)

    def test_problem_input_matches_string_input(self):
        problem = PDDLReader().parse_problem_string(self.DOMAIN_A, None)
        assert compute_domain_fingerprint(self.DOMAIN_A) == compute_domain_fingerprint(problem)

    def test_problem_input_invariant_to_problem_objects(self):
        problem_str = (
            "(define (problem p) (:domain blocksworld)"
            "  (:objects a b c - block) (:init (clear a)) (:goal (clear b)))"
        )
        full = PDDLReader().parse_problem_string(self.DOMAIN_A, problem_str)
        domain_only = PDDLReader().parse_problem_string(self.DOMAIN_A, None)
        assert compute_domain_fingerprint(full) == compute_domain_fingerprint(domain_only)

    def test_different_predicates_differ(self):
        different = (
            "(define (domain blocksworld) (:requirements :strips :typing)"
            "  (:types block) (:predicates (holding ?x - block)))"
        )
        assert compute_domain_fingerprint(self.DOMAIN_A) != compute_domain_fingerprint(different)

    def test_different_action_effects_differ(self):
        action_p = "(:action a :parameters (?x - block) :precondition () :effect (clear ?x))"
        action_q = "(:action a :parameters (?x - block) :precondition () :effect (not (clear ?x)))"
        assert compute_domain_fingerprint(_make_blocksworld(action_p)) != compute_domain_fingerprint(
            _make_blocksworld(action_q)
        )

    def test_different_predicate_arity_differs(self):
        unary = (
            "(define (domain d) (:requirements :strips :typing)"
            "  (:types block) (:predicates (p ?x - block)))"
        )
        binary = (
            "(define (domain d) (:requirements :strips :typing)"
            "  (:types block) (:predicates (p ?x - block ?y - block)))"
        )
        assert compute_domain_fingerprint(unary) != compute_domain_fingerprint(binary)


class TestCalibrationExample:
    def test_problem_defaults_to_none(self):
        example = CalibrationExample(images=[], state_dict={"on(a,b)": True})
        assert example.problem is None
