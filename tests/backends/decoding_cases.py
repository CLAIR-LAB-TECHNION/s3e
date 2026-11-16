# SPDX-FileCopyrightText: CLAIR Lab Technion
# SPDX-License-Identifier: MIT

"""Caller ``inference_kwargs`` -> the decoding of every token the library generates.

Temperature 0 is greedy; ``top_k=None`` cuts nothing.
"""

from collections import namedtuple

Decoding = namedtuple("Decoding", "temperature top_p top_k repetition_penalty",
                      defaults=(0.0, 1.0, None, 1.0))

DECODING_CASES = [
    ({}, Decoding()),
    ({"temperature": 0}, Decoding()),
    ({"top_p": 0.5, "top_k": 5}, Decoding()),  # truncation alone: still greedy
    ({"repetition_penalty": 1.2}, Decoding(repetition_penalty=1.2)),
    ({"temperature": 0.7}, Decoding(0.7)),
    ({"temperature": 1.0, "top_p": 0.9, "top_k": 20}, Decoding(1.0, 0.9, 20)),
]
