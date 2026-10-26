# SPDX-License-Identifier: MIT
"""Relevant-atom masking: which grounded atoms to ask the VLM about.

S3E estimates whatever predicate list it is given (``predicates=``); by
default every grounded boolean fluent. Three masks are compared:

* ``none``: every grounded atom (S3E's default grounding);
* ``dynamic``: only atoms of fluents some action changes; static atoms
  (e.g. adjacency, object categories) are read from the problem file, and
  atoms the domain declares impossible (e.g. ``on(x,x)``) are fixed to False;
* ``relevant``: ``dynamic`` restricted to atoms that can matter for the
  goal (backward relevance, Nebel et al. 1997): the goal's atoms, closed
  under the precondition and effect-condition atoms of every grounded
  action with an effect on a relevant atom. Other atoms cannot affect
  whether a plan reaches the goal and are fixed to False.

The masks use the domain, the objects, the goal, and static facts only, so
they never look at the true initial values of dynamic atoms.
"""

from collections.abc import Callable

from s3e.pddl import ground_predicate_str_to_fnode, ground_predicates

MASKS = ("none", "dynamic", "relevant")


def _fluents_in(expression) -> set[str]:
    extractor = expression.environment.free_vars_extractor
    return {node.fluent().name for node in extractor.get(expression)}


def fluent_analysis(problem) -> dict[str, set[str]]:
    """Lifted fluent roles: ``dynamic``, ``static``, ``goal``, ``relevant``."""
    dynamic: set[str] = set()
    depends: dict[str, set[str]] = {}
    for action in problem.actions:
        pre: set[str] = set()
        for condition in action.preconditions:
            pre |= _fluents_in(condition)
        for effect in action.effects:
            name = effect.fluent.fluent().name
            dynamic.add(name)
            depends.setdefault(name, set()).update(
                pre | _fluents_in(effect.condition) | _fluents_in(effect.value)
            )
    goal: set[str] = set()
    for condition in problem.goals:
        goal |= _fluents_in(condition)
    relevant, frontier = set(goal), list(goal)
    while frontier:
        for name in depends.get(frontier.pop(), ()):
            if name not in relevant:
                relevant.add(name)
                frontier.append(name)
    names = {f.name for f in problem.fluents if f.type.is_bool_type()}
    return {
        "dynamic": dynamic & names,
        "static": names - dynamic,
        "goal": goal & names,
        "relevant": relevant & names,
    }


def atom_string(fnode) -> str:
    return f"{fnode.fluent().name}({','.join(str(a) for a in fnode.args)})"


_RELEVANCE_CACHE: dict = {}


def relevant_atoms(problem) -> set[str]:
    """Grounded backward relevance from the goal (cached per problem text)."""
    from unified_planning.io import PDDLWriter
    from unified_planning.shortcuts import CompilationKind, Compiler

    writer = PDDLWriter(problem)
    key = (writer.get_domain(), _without_dynamic_init(problem))
    if key in _RELEVANCE_CACHE:
        return _RELEVANCE_CACHE[key]
    try:
        with Compiler(name="up_grounder", problem_kind=problem.kind,
                      compilation_kind=CompilationKind.GROUNDING) as grounder:
            grounded = grounder.compile(problem, CompilationKind.GROUNDING).problem
    except Exception as error:  # deliberate fallback: lifted relevance is a superset
        import warnings

        warnings.warn(f"grounding failed ({error!r}); using lifted relevance instead",
                      stacklevel=2)
        lifted = fluent_analysis(problem)["relevant"]
        relevant = {a for a in ground_predicates(problem) if fluent_of(a) in lifted}
        _RELEVANCE_CACHE[key] = relevant
        return relevant
    extractor = grounded.environment.free_vars_extractor
    supports: dict[str, set[str]] = {}
    for action in grounded.actions:
        needs = set()
        for condition in action.preconditions:
            needs |= {atom_string(f) for f in extractor.get(condition)}
        for effect in action.effects:
            conditional = {atom_string(f) for f in extractor.get(effect.condition)}
            target = atom_string(effect.fluent)
            supports.setdefault(target, set()).update(needs | conditional)
    relevant = set()
    for goal in problem.goals:
        relevant |= {atom_string(f) for f in extractor.get(goal)}
    frontier = list(relevant)
    while frontier:
        for atom in supports.get(frontier.pop(), ()):
            if atom not in relevant:
                relevant.add(atom)
                frontier.append(atom)
    _RELEVANCE_CACHE[key] = relevant
    return relevant


def _without_dynamic_init(problem) -> str:
    """The problem's objects, goal, and static facts: what relevance depends on."""
    static = fluent_analysis(problem)["static"]
    facts = sorted(str(k) for k, v in problem.explicit_initial_values.items()
                   if k.fluent().name in static and v.is_true())
    return repr((sorted(str(o) for o in problem.all_objects), [str(g) for g in problem.goals],
                 facts))


def fluent_of(atom: str) -> str:
    return atom.split("(", 1)[0]


def select_atoms(
    problem,
    mask: str = "none",
    impossible: "Callable[[str], bool] | None" = None,
) -> tuple[list[str], dict[str, bool]]:
    """Split the problem's grounded atoms into those to estimate and fixed ones.

    Returns ``(to_estimate, fixed)``: ``fixed`` maps every other grounded
    atom to the value it takes without vision (static atoms from the
    problem's initial state, impossible and irrelevant atoms False).
    """
    if mask not in MASKS:
        raise ValueError(f"Unknown mask {mask!r}; expected one of {MASKS}")
    atoms = ground_predicates(problem)
    if mask == "none":
        return atoms, {}
    roles = fluent_analysis(problem)
    relevant = relevant_atoms(problem) if mask == "relevant" else None
    estimate, fixed = [], {}
    for atom in atoms:
        name = fluent_of(atom)
        if name in roles["static"]:
            fnode = ground_predicate_str_to_fnode(problem, atom)
            fixed[atom] = problem.initial_value(fnode).bool_constant_value()
        elif (impossible and impossible(atom)) or (relevant is not None and atom not in relevant):
            fixed[atom] = False
        else:
            estimate.append(atom)
    return estimate, fixed
