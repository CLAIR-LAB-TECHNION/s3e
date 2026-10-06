# SPDX-FileCopyrightText: CLAIR Lab Technion
# SPDX-License-Identifier: MIT

"""Release metadata stays consistent across pyproject.toml, CITATION.cff, and
the JOSS paper. (Zenodo archives releases from CITATION.cff, and JOSS requires
the archive's title and authors to match the paper's.)

Parsed with regular expressions: Python 3.10 has no tomllib, and PyYAML is
not a dependency.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
CITATION = ROOT / "CITATION.cff"
PAPER = ROOT / "paper" / "paper.md"


def read(path: Path) -> str:
    if not path.exists():
        pytest.skip(f"{path.relative_to(ROOT)} is not in this source tree")
    return path.read_text(encoding="utf-8")


def citation_top_level() -> str:
    """CITATION.cff without its preferred-citation block (the S3E paper)."""
    return read(CITATION).split("\npreferred-citation:", 1)[0]


def test_citation_version_matches_package_version():
    package = re.search(r'^version = "([^"]+)"', read(PYPROJECT), re.M).group(1)
    citation = re.search(r"^version: (\S+)", citation_top_level(), re.M).group(1)
    assert citation == package


def test_citation_title_matches_paper_title():
    citation = re.search(r'^title: "(.+)"$', citation_top_level(), re.M).group(1)
    paper = re.search(r"^title: '(.+)'$", read(PAPER), re.M).group(1)
    assert citation == paper


def test_citation_authors_match_paper_authors_in_order():
    citation = re.findall(
        r"- family-names: (.+)\n\s+given-names: (.+)", citation_top_level()
    )
    authors_block = read(PAPER).split("\nauthors:\n", 1)[1].split("\naffiliations:\n", 1)[0]
    paper = re.findall(r"^  - name: (.+)$", authors_block, re.M)
    assert paper
    assert [f"{given} {family}" for family, given in citation] == paper
