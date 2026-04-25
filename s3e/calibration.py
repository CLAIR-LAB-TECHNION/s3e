"""Helpers and data structures for Platt scaling."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from PIL.Image import Image
from unified_planning.io import PDDLReader
from unified_planning.model.effect import Effect, EffectKind
from unified_planning.model.fnode import FNode
from unified_planning.model.operators import OperatorKind
from unified_planning.shortcuts import Problem

try:
    from sklearn.linear_model import LogisticRegression
except ImportError:
    LogisticRegression = None  # type: ignore[assignment]


CALIBRATION_SCHEMA_VERSION = 1
SCORE_EPS = 1e-12
GLOBAL_CALIBRATION_KEY = "__global__"


@dataclass(frozen=True)
class CalibrationExample:
    images: list[Image]
    state_dict: dict[str, bool]
    problem: str | None = None


@dataclass(frozen=True)
class PlattParameters:
    a: float
    b: float
    sample_count: int
    positive_count: int
    negative_count: int


@dataclass(frozen=True)
class PlattScalingProfile:
    scope: str
    probability_method: str
    true_tokens: list[str]
    false_tokens: list[str]
    domain_fingerprint: str
    score_kind: str
    groups: dict[str, PlattParameters]
    schema_version: int = CALIBRATION_SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope,
            "probability_method": self.probability_method,
            "true_tokens": list(self.true_tokens),
            "false_tokens": list(self.false_tokens),
            "domain_fingerprint": self.domain_fingerprint,
            "score_kind": self.score_kind,
            "groups": {
                key: {
                    "a": value.a,
                    "b": value.b,
                    "sample_count": value.sample_count,
                    "positive_count": value.positive_count,
                    "negative_count": value.negative_count,
                }
                for key, value in self.groups.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PlattScalingProfile":
        schema_version = int(data["schema_version"])
        if schema_version != CALIBRATION_SCHEMA_VERSION:
            raise ValueError(f"Unsupported calibration schema version: {schema_version}")
        return cls(
            schema_version=schema_version,
            scope=data["scope"],
            probability_method=data["probability_method"],
            true_tokens=list(data["true_tokens"]),
            false_tokens=list(data["false_tokens"]),
            domain_fingerprint=data["domain_fingerprint"],
            score_kind=data["score_kind"],
            groups={
                key: PlattParameters(
                    a=float(value["a"]),
                    b=float(value["b"]),
                    sample_count=int(value["sample_count"]),
                    positive_count=int(value["positive_count"]),
                    negative_count=int(value["negative_count"]),
                )
                for key, value in data["groups"].items()
            },
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "PlattScalingProfile":
        return cls.from_dict(json.loads(Path(path).read_text()))


_COMMUTATIVE_BINARY_OPS = frozenset({OperatorKind.AND, OperatorKind.OR, OperatorKind.EQUALS, OperatorKind.IFF})


def _canonicalize_fnode(node: FNode) -> str:
    """Produce a deterministic canonical string from a UP FNode tree.

    Commutative operators (AND, OR, EQUALS, IFF) have their operands sorted
    lexicographically on the canonical child string. All other operators
    preserve argument order.
    """
    kind = node.node_type

    if kind == OperatorKind.FLUENT_EXP:
        name = node.fluent().name
        args = ",".join(_canonicalize_fnode(a) for a in node.args)
        return f"{name}({args})" if args else name

    if kind == OperatorKind.PARAM_EXP:
        return f"?{node.parameter().name}"

    if kind == OperatorKind.VARIABLE_EXP:
        return f"?{node.variable().name}"

    if kind == OperatorKind.OBJECT_EXP:
        return node.object().name

    if kind == OperatorKind.BOOL_CONSTANT:
        return "true" if node.bool_constant_value() else "false"

    if kind == OperatorKind.INT_CONSTANT:
        return str(node.int_constant_value())

    if kind == OperatorKind.REAL_CONSTANT:
        return str(node.real_constant_value())

    if kind == OperatorKind.NOT:
        return f"(not {_canonicalize_fnode(node.args[0])})"

    if kind in (OperatorKind.EXISTS, OperatorKind.FORALL):
        quantifier = "exists" if kind == OperatorKind.EXISTS else "forall"
        sorted_vars = sorted(f"?{v.name}:{v.type.name}" for v in node.variables())
        body = _canonicalize_fnode(node.args[-1])
        return f"({quantifier} ({' '.join(sorted_vars)}) {body})"

    op_name = kind.name.lower()
    children = [_canonicalize_fnode(a) for a in node.args]
    if kind in _COMMUTATIVE_BINARY_OPS:
        children.sort()
    return f"({op_name} {' '.join(children)})"


def _canonicalize_effect(effect: Effect) -> str:
    """Canonical string for one UP Effect, normalized for fingerprinting."""
    fluent_str = _canonicalize_fnode(effect.fluent)

    if effect.kind == EffectKind.ASSIGN and effect.value.is_bool_constant():
        base = fluent_str if effect.value.bool_constant_value() else f"(not {fluent_str})"
    else:
        value_str = _canonicalize_fnode(effect.value)
        base = f"({effect.kind.name.lower()} {fluent_str} {value_str})"

    if effect.is_conditional():
        cond_str = _canonicalize_fnode(effect.condition)
        base = f"(when {cond_str} {base})"

    if effect.is_forall():
        sorted_vars = sorted(f"?{v.name}:{v.type.name}" for v in effect.forall)
        base = f"(forall ({' '.join(sorted_vars)}) {base})"

    return base


def _canonicalize_action_preconditions(preconditions: list[FNode]) -> str:
    """Canonical string for an action's preconditions list (implicit AND).

    UP exposes preconditions as a list of FNodes that combine implicitly via
    AND. Both the list itself and any nested AND/OR sub-trees are sorted.
    """
    conjuncts = sorted(_canonicalize_fnode(p) for p in preconditions)
    if not conjuncts:
        return "()"
    if len(conjuncts) == 1:
        return conjuncts[0]
    return "(and " + " ".join(conjuncts) + ")"


def _canonicalize_action_effects(effects: list[Effect]) -> str:
    """Canonical string for an action's effects list (implicit AND, sorted)."""
    pieces = sorted(_canonicalize_effect(e) for e in effects)
    if not pieces:
        return "()"
    if len(pieces) == 1:
        return pieces[0]
    return "(and " + " ".join(pieces) + ")"


def _build_canonical_domain_string(problem: Problem) -> str:
    """Build a deterministic canonical string from a UP Problem for fingerprinting.

    Every semantically unordered component is sorted:

    * ``types`` — sorted by name; supertype encoded as ``name<:parent``.
    * ``fluents`` — sorted by name; signature is positional.
    * ``actions`` — sorted by name; parameters are positional; preconditions
      and effects are sorted (the precondition list is an implicit AND, and
      nested AND/OR/EQUALS/IFF sub-trees are sorted by ``_canonicalize_fnode``).

    Constants/objects are intentionally **not** included — see
    :func:`compute_domain_fingerprint` for the calibration-compatibility scope.
    """
    type_entries = sorted(
        f"{t.name}<:{t.father.name}" if getattr(t, "father", None) is not None else t.name
        for t in problem.user_types
    )

    fluent_entries = []
    for f in sorted(problem.fluents, key=lambda f: f.name):
        sig = ",".join(p.type.name for p in f.signature)
        fluent_entries.append(f"{f.name}({sig})")

    action_entries = []
    for a in sorted(problem.actions, key=lambda a: a.name):
        params = ",".join(f"{p.name}:{p.type.name}" for p in a.parameters)
        prec = _canonicalize_action_preconditions(list(a.preconditions))
        eff = _canonicalize_action_effects(list(a.effects))
        action_entries.append(f"{a.name}[{params}]pre:{prec}eff:{eff}")

    return "\n".join(
        [
            "types:" + ",".join(type_entries),
            "fluents:" + ",".join(fluent_entries),
            "actions:" + "|".join(action_entries),
        ]
    )


def compute_domain_fingerprint(domain: str | Problem) -> str:
    """Hash a PDDL domain into a stable fingerprint.

    Accepts either a PDDL domain string or an already-parsed UP ``Problem``.
    Domains differing only by harmless serialization artifacts — whitespace,
    comments, domain name, or ordering of unordered declarations
    (types, predicates, actions, ``and`` conjuncts, quantifier-bound
    variables) — produce the same fingerprint.

    Scope is **calibration compatibility**, not full PDDL semantic
    equivalence. Constants/objects are excluded because UP cannot reliably
    distinguish domain constants from problem objects on a parsed Problem,
    and a calibration profile must remain valid across problem instances
    that share a domain. Constants referenced in actions are still captured
    in the action's FNode canonicalization.
    """
    if isinstance(domain, str):
        problem = PDDLReader().parse_problem_string(domain, None)
    else:
        problem = domain
    canonical = _build_canonical_domain_string(problem)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def grouped_log_odds(
    token_probs: dict[str, float],
    true_tokens: list[str],
    false_tokens: list[str],
    eps: float = SCORE_EPS,
) -> float:
    true_mass = sum(token_probs.get(token, 0.0) for token in true_tokens)
    false_mass = sum(token_probs.get(token, 0.0) for token in false_tokens)
    return math.log((true_mass + eps) / (false_mass + eps))


def apply_platt_scaling(score: float, params: PlattParameters) -> float:
    return 1.0 / (1.0 + math.exp(params.a * score + params.b))


def require_sklearn() -> None:
    if LogisticRegression is None:
        raise ImportError(
            "Platt scaling fitting requires scikit-learn. "
            "Install it with: pip install 's3e[calibration]'"
        )


def fit_platt_parameters(scores: list[float], labels: list[bool]) -> PlattParameters:
    require_sklearn()
    if not scores:
        raise ValueError("Expected at least one calibration sample.")

    positives = sum(bool(label) for label in labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("Platt scaling requires both positive and negative labels.")

    model = LogisticRegression(random_state=0)
    model.fit([[score] for score in scores], labels)

    coef = float(model.coef_[0][0])
    intercept = float(model.intercept_[0])
    return PlattParameters(
        a=-coef,
        b=-intercept,
        sample_count=len(scores),
        positive_count=positives,
        negative_count=negatives,
    )
