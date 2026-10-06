# SPDX-FileCopyrightText: CLAIR Lab Technion
# SPDX-License-Identifier: MIT

"""Helpers shared by the backends, not backends themselves.

:mod:`.resolve` turns model-id strings into backends (re-exported as
:func:`s3e.resolve_backend`), :mod:`.token_index` maps token strings to
vocabulary ids, and :mod:`.thinking` detects and skips thinking blocks. None
of them imports an optional dependency.
"""
