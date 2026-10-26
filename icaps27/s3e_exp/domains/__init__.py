# SPDX-License-Identifier: MIT
"""Domain adapters.

An adapter is a module exposing ``NAME``, ``TEMPLATES`` (lifted predicate ->
question template), ``PROMPT_TEMPLATE``, and ``impossible(atom)``. Adapters
that can make scenes expose ``generate(size, count, seed)``; adapters that
can run closed loop expose ``World(scene)`` (``observe``, ``state``,
``apply``, ``goal_reached``); adapters with a consistent-state decoder
expose ``map_decode(probabilities, objects)``.
"""

import importlib

ADAPTERS = {
    "blocksworld": "s3e_exp.domains.blocksworld",
    "viplan-prb": "s3e_exp.domains.viplan",
    "viplan-gw": "s3e_exp.domains.viplan",
    "viplan-alf": "s3e_exp.domains.viplan",
}


def get(name: str):
    """The adapter module for ``name`` (ViPlan++ adapters take the env as a suffix)."""
    if name not in ADAPTERS:
        raise ValueError(f"Unknown domain {name!r}; expected one of {sorted(ADAPTERS)}")
    module = importlib.import_module(ADAPTERS[name])
    if hasattr(module, "adapter"):
        return module.adapter(name)
    return module
