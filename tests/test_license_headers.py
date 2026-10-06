# SPDX-FileCopyrightText: CLAIR Lab Technion
# SPDX-License-Identifier: MIT

"""Every Python source file names its license (an SPDX header)."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCES = sorted(
    path
    for folder in ("s3e", "tests", "examples")
    for path in (ROOT / folder).rglob("*.py")
) + [path for path in [ROOT / "docs" / "conf.py"] if path.exists()]


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(ROOT)))
def test_source_file_has_spdx_header(path):
    head = path.read_text(encoding="utf-8").splitlines()[:3]
    assert "# SPDX-License-Identifier: MIT" in head, (
        f"{path.relative_to(ROOT)} must start with the SPDX header:\n"
        "# SPDX-FileCopyrightText: CLAIR Lab Technion\n"
        "# SPDX-License-Identifier: MIT"
    )
