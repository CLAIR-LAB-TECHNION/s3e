# SPDX-FileCopyrightText: CLAIR Lab Technion
# SPDX-License-Identifier: MIT

"""The code shown in the README, the getting-started guide, and the
walkthrough notebook runs as documented, against a real (small) model.

All tests here are slow: they download ``HuggingFaceTB/SmolVLM-256M-Instruct``
(~500 MB) and run it, on CPU if no GPU is available.
"""

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
GETTING_STARTED = ROOT / "docs" / "getting-started.md"
NOTEBOOK = ROOT / "docs" / "s3e_walkthrough.ipynb"

pytestmark = pytest.mark.slow


def python_blocks(path: Path, section: str | None = None) -> list[str]:
    """The ```python blocks of a Markdown file, optionally of one ## section."""
    text = path.read_text(encoding="utf-8")
    if section is not None:
        start = text.index(f"\n## {section}\n")
        end = text.find("\n## ", start + 1)
        text = text[start:end if end != -1 else None]
    return re.findall(r"^```python\n(.*?)^```$", text, re.M | re.S)


def run_blocks(blocks: list[str], source: str) -> dict:
    namespace: dict = {"__name__": "__main__"}
    for block in blocks:
        exec(compile(block, source, "exec"), namespace)
    return namespace


def test_readme_quick_start_runs_in_order(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("unified_planning")
    pytest.importorskip("sklearn")
    # The vLLM snippet needs a CUDA host and the vllm extra.
    blocks = [b for b in python_blocks(README, "Quick Start") if "VLLMBackend(" not in b]
    assert len(blocks) >= 8
    monkeypatch.chdir(tmp_path)  # the calibration snippet writes JSON files

    namespace = run_blocks(blocks, str(README))

    assert set(namespace["state"].values()) <= {True, False}
    assert (tmp_path / "platt-profile.json").exists()


@pytest.mark.skipif(not GETTING_STARTED.exists(), reason="docs/ is not in the sdist")
def test_getting_started_example_runs():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("unified_planning")
    (block,) = [b for b in python_blocks(GETTING_STARTED) if "from_pddl" in b]

    namespace = run_blocks([block], str(GETTING_STARTED))

    assert set(namespace["results"].probabilities()) == {
        "on(blue,blue)", "on(blue,orange)", "on(orange,blue)", "on(orange,orange)",
        "clear(blue)", "clear(orange)",
    }


@pytest.mark.skipif(not NOTEBOOK.exists(), reason="docs/ is not in the sdist")
def test_walkthrough_notebook_executes(tmp_path, monkeypatch):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")

    # Run the kernel with this interpreter: an installed "python3" kernelspec
    # may point at another environment.
    kernel_dir = tmp_path / "kernels" / "s3e-tests"
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "kernel.json").write_text(
        json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "s3e tests",
            "language": "python",
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("JUPYTER_PATH", str(tmp_path))

    notebook = nbformat.read(NOTEBOOK, as_version=4)
    # Cells tagged "skip-execution" (the %pip install cell) are skipped, so
    # the notebook runs against this checkout, not the PyPI release.
    client = nbclient.NotebookClient(
        notebook,
        timeout=3600,
        kernel_name="s3e-tests",
        resources={"metadata": {"path": str(NOTEBOOK.parent)}},
    )
    client.execute()
