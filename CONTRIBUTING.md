# Contributing to s3e

Thanks for your interest in `s3e`! This document explains how to get help,
report problems, and contribute changes. By participating in this project you
agree to abide by its [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting help

- **Usage questions**: open a
  [GitHub issue](https://github.com/CLAIR-LAB-TECHNION/s3e/issues/new/choose)
  using the *Question* template. Please include the `s3e` version
  (`python -c "import s3e; print(s3e.__version__)"`), the backend you are using
  (HuggingFace, OpenAI, vLLM, or a custom `VLMBackend`), and a minimal code
  snippet.
- **Documentation**: start with the [README](README.md), the
  [API reference](https://s3e.readthedocs.io), and the
  [walkthrough notebook](docs/s3e_walkthrough.ipynb).

## Reporting bugs

Please [open a bug report](https://github.com/CLAIR-LAB-TECHNION/s3e/issues/new/choose)
and include:

1. what you did (a minimal, runnable example if possible — the `FakeVLM` in
   [`tests/fakes.py`](tests/fakes.py) is handy for reproducing engine or
   estimator bugs without downloading a model),
2. what you expected to happen and what happened instead (full traceback),
3. your environment: OS, Python version, `s3e` version, and the versions of
   the relevant optional dependencies (`torch`, `transformers`, `openai`,
   `vllm`, `unified-planning`, `scikit-learn`).

Security-sensitive reports should be sent privately to the maintainer at
guy.azran@campus.technion.ac.il instead of a public issue.

## Suggesting features

Feature requests are welcome as issues using the *Feature request* template.
Describe the use case first: what you are trying to estimate, with which
model, and why the current API does not cover it.

## Contributing code

### Development setup

```bash
git clone https://github.com/CLAIR-LAB-TECHNION/s3e.git
cd s3e
pip install -e ".[dev]"        # CPU: full fast suite; vLLM-dependent tests skip
pip install -e ".[dev-gpu]"    # CUDA hosts: adds vllm for the vLLM test coverage
```

To build the documentation locally:

```bash
pip install -e ".[docs]"
sphinx-build -W -b html docs docs/_build/html
```

### Running the tests

```bash
pytest -m "not slow"   # fast loop: no model downloads, runs on CPU
pytest -m slow         # downloads and runs real HuggingFace models
pytest                 # everything
```

Tests marked `slow` download real models; the vLLM integration tests also
need a CUDA GPU. Continuous integration runs `pytest -m "not slow"` on every
pull request and every push to `main`.

### Workflow

1. Fork the repository and create a feature branch from `main`.
2. Make your change, keeping the diff focused on one concern.
3. Add or update tests next to the code you changed — the `tests/` tree
   mirrors the package layout. Reuse the fixtures in `tests/conftest.py` and
   the `FakeVLM` double in `tests/fakes.py` instead of duplicating setup.
4. Run `pytest -m "not slow"` and make sure it passes.
5. Open a pull request describing the change and its motivation, and link any
   related issue.

### Style

There is no enforced formatter or linter; follow the style of the surrounding
code:

- 4-space indentation, double quotes, and a concise docstring at the top of
  every module.
- Docstrings (Google style, with `Args:` / `Returns:` sections) on public
  classes and functions — they are rendered into the API reference.
- Type hints on public APIs, using built-in generics (`list[str]`) and PEP 604
  unions (`str | None`) in new code.
- Raise explicit, informative exceptions: `ValueError` for invalid inputs, and
  `ImportError` with install guidance for missing optional dependencies (see
  `s3e._deps.require`).
- Keep backend-specific logic inside `s3e/backends/`; the engine and
  estimator stay backend-agnostic.
- When adding a public API, update the relevant `__init__.py` re-exports and
  `__all__`, and add it to the API reference under `docs/api/`.

### Adding a new VLM backend

Subclass `s3e.backends.VLMBackend` and implement `query` (and optionally
`query_batch` and `unsupported_interest_tokens`). The behavior every backend
must satisfy is captured by `BackendContract` in
[`tests/backends/test_contract.py`](tests/backends/test_contract.py): subclass
it in your backend's test module and provide a `make_backend` fixture to run
the whole contract suite against your implementation.
