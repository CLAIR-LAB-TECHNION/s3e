"""Optional-dependency checks with install guidance.

Leaf modules that need a heavy optional package call :func:`require` once,
before their normal top-of-module imports. Everything else in the package
imports freely — no per-import guards.
"""

import importlib.util


def require(module_name: str, extra: str, feature: str) -> None:
    """Raise a helpful ImportError when an optional module is missing.

    Args:
        module_name: Importable module to check for (e.g. ``"torch"``).
        extra: The s3e extra that provides it (e.g. ``"hf"``).
        feature: Human-readable feature name for the error message.
    """
    try:
        found = importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        # Meta-path finders may raise instead of returning None (our own
        # import-hygiene tests install one that does); treat that as absent.
        found = False
    if not found:
        raise ImportError(
            f'{feature} requires the {extra!r} extra: pip install "s3e[{extra}]"'
        )
