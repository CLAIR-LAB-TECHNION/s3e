# AGENTS.md

## Verify changes
- Run the narrowest relevant test first, e.g. `pytest tests/test_cache.py::TestMakeCacheKey::test_basic_key`.
- Before finishing a change, run `pytest -m "not slow"` and `ruff check .`.
- `pytest -m slow` downloads and runs real models; run it only when asked.

## Don't
- Don't run `ruff format` or reformat files. The repo has no formatter, so match the surrounding style.
- Don't remove the `OpenAI/` model-name prefix convention or the compatibility branches for different transformers versions. Both are intentional.
- Don't use em-dashes, semicolons, or unnecessary colons in documentation.
- Don't edit `README.md` unless installation, the quick start, or the feature list changes. Behavior details, new arguments, and edge cases go in docstrings and `docs/user-guide.md`.

## Documentation
- Document a change once, in the docstring of the API it changes. Add to `docs/user-guide.md` only when the change affects how users configure S3E or read its results.
- Give each explanation one home, next to the code it is about. Copies in other comments, docstrings, or docs drift apart when only one is updated, so don't restate it elsewhere, even reworded or summarized. When you edit an explanation that already exists in several places, merge the copies into its home.
- Before writing a pointer to another file ("see X", "lives in X"), ask whether a reader of this code needs X to use or change it correctly. If not, leave the pointer out. A reader who does need X is already looking there.
- State user-facing changes in the PR summary. That is where the changelog is written from.

## Gotchas
- The docs build (`sphinx-build -W -b html docs docs/_build/html`) mocks torch, torchvision, transformers, accelerate and vllm (see `docs/conf.py`). New public modules must import cleanly under those mocks and be added to the matching `docs/api/*.rst` page.
- `paper/paper.md` (JOSS) must stay between 750 and 1750 words, counted over the whole file, and must be the only `paper.md` in the repo.

## Releases
- Only when cutting a release, follow the "Releasing" section of `CONTRIBUTING.md`.
