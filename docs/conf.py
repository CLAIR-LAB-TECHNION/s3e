# SPDX-FileCopyrightText: CLAIR Lab Technion
# SPDX-License-Identifier: MIT

"""Sphinx configuration for the s3e documentation."""

from importlib.metadata import PackageNotFoundError, version

project = "s3e"
author = "Guy Azran and contributors"
copyright = "2024-2026, CLAIR Lab, Technion"

try:
    release = version("s3e")
except PackageNotFoundError:  # building from a source tree without installing
    release = "0.0.0.dev0"
version = release

extensions = [
    "myst_nb",  # MyST Markdown pages and the walkthrough notebook
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

exclude_patterns = ["_build", "**/.ipynb_checkpoints"]

# The heavy optional backends need GPUs or large downloads; mock them so the
# API reference builds with only the lightweight extras installed.
autodoc_mock_imports = ["torch", "torchvision", "transformers", "accelerate", "vllm"]
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autoclass_content = "class"

napoleon_google_docstring = True
napoleon_numpy_docstring = False

myst_heading_anchors = 3

# Render the walkthrough notebook with its saved outputs; executing it needs
# model downloads (the slow tests execute it instead). Its stderr is log noise.
nb_execution_mode = "off"
nb_output_stderr = "remove"

html_theme = "furo"
html_title = f"s3e {release}"
