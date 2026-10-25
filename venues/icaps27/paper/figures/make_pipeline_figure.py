# SPDX-FileCopyrightText: CLAIR Lab Technion
# SPDX-License-Identifier: MIT

"""Render the S3E pipeline figure (layers, data flow, planner hand-off).

Drawn for the JOSS paper (``paper/s3e-pipeline.png``); reuse or adapt it for
the ICAPS 2027 tool paper's Figure 1. Colors follow the Okabe-Ito palette.

Usage::

    python make_pipeline_figure.py out.png      # or out.pdf
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

# Okabe-Ito colorblind-safe palette: (edge, light fill)
LAYERS = {
    "input": ("#555555", "#F2F2F2"),
    "PDDL facade (SemanticStateEstimator)": ("#0072B2", "#DCEBF5"),
    "Backends (VLMBackend)": ("#009E73", "#D9F0E9"),
    "Engine (QueryEngine, answer spaces, results)": ("#D55E00", "#F9E3D6"),
    "Calibration (optional, PlattCalibrator)": ("#CC79A7", "#F6E6EF"),
}
FACADE, BACKEND, ENGINE, CALIB = list(LAYERS)[1:]

W, H = 7.0, 4.15
TITLE_PT, BODY_PT, LABEL_PT = 9.5, 8, 8

fig, ax = plt.subplots(figsize=(W, H), dpi=300)
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def box(x, y, w, h, title, body, layer, dashed=False):
    edge, fill = LAYERS[layer]
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.07",
            linewidth=1.4, edgecolor=edge, facecolor=fill,
            linestyle=(0, (4, 2)) if dashed else "solid",
        )
    )
    ax.text(x, y + h / 2 - 0.2, title, ha="center", va="center",
            fontsize=TITLE_PT, fontweight="bold", color="#1A1A1A")
    ax.text(x, y - 0.12, body, ha="center", va="center", fontsize=BODY_PT,
            color="#1A1A1A", family="DejaVu Sans Mono", linespacing=1.2)


def arrow(p, q):
    ax.add_patch(
        FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11,
                        linewidth=1.2, color="#333333")
    )


def label(x, y, text):
    ax.text(x, y, text, ha="center", va="center", fontsize=LABEL_PT,
            color="#333333", style="italic", linespacing=1.1)


# Row 1: the symbolic side.
y1, w1, h1 = 3.45, 1.78, 0.95
x1 = [0.95, 3.5, 6.05]
box(x1[0], y1, w1, h1, "PDDL problem", "(:objects a b)\n(on ?x ?y) …", "input")
box(x1[1], y1, w1, h1, "Grounded facts", "on(a,b)  on(b,a)\nclear(a)  …", FACADE)
box(x1[2], y1, w1, h1, "Queries", "\"Is block a on\ntop of block b?\"", FACADE)
for left, right, text in [(0, 1, "ground\n(Unified\nPlanning)"), (1, 2, "translate")]:
    arrow((x1[left] + w1 / 2, y1), (x1[right] - w1 / 2, y1))
    label((x1[left] + x1[right]) / 2, y1 + 0.38 if left == 0 else y1 + 0.2, text)

# Row 2: perception and probabilities.
y2, w2, h2 = 1.5, 1.5, 0.95
x2 = [0.8, 2.6, 4.4, 6.2]
box(x2[0], y2, w2, h2, "VLM backend", "HuggingFace\nvLLM · OpenAI", BACKEND)
box(x2[1], y2, w2, h2, "Token mass", "P(yes) P(no)\nP(unknown)", ENGINE)
box(x2[2], y2, w2, h2, "Calibration", "Platt scaling\nfitted offline", CALIB, dashed=True)
box(x2[3], y2, w2, h2, "State", "on(a,b): True\nclear(b): None", ENGINE)
for i in range(3):
    arrow((x2[i] + w2 / 2, y2), (x2[i + 1] - w2 / 2, y2))

# Queries wrap down to the backend; images feed it from below.
y_mid = (y1 - h1 / 2 + y2 + h2 / 2) / 2
ax.plot([x1[2], x1[2], x2[0]], [y1 - h1 / 2, y_mid, y_mid], color="#333333",
        linewidth=1.2, solid_capstyle="butt")
arrow((x2[0], y_mid + 0.006), (x2[0], y2 + h2 / 2))
label((x1[2] + x2[0]) / 2, y_mid + 0.14, "query each fact (batched)")
ax.text(x2[0], 0.6, "images", ha="center", va="center", fontsize=LABEL_PT, color="#333333")
arrow((x2[0], 0.72), (x2[0], y2 - h2 / 2))
ax.text(x2[3], 0.6, "planner", ha="center", va="center", fontsize=LABEL_PT, color="#333333")
arrow((x2[3], y2 - h2 / 2), (x2[3], 0.72))

handles = [Patch(facecolor=fill, edgecolor=edge, label=name)
           for name, (edge, fill) in LAYERS.items() if name != "input"]
ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2,
          fontsize=7.5, frameon=False, handlelength=1.5, columnspacing=1.2)

fig.savefig(sys.argv[1], bbox_inches="tight", pad_inches=0.04, facecolor="white")
print("wrote", sys.argv[1])
