# Venue material

Everything about publishing S3E as a software/tool paper lives here, on the `icaps27-tool`
branch, so that the material is in one place without touching `main`. Only what the
**ICAPS 2027 tool-paper submission** needs is tracked; material for the other venues is kept
in this tree but ignored by git (see the `venues/` block at the end of `.gitignore`). Delete
those ignore lines if you later want the other venues tracked too.

| Path | Tracked | What it is |
|---|---|---|
| `../icaps27-tool.md` | yes | The strategy: rules, acceptance drivers, paper outline, experiments, artifact, timeline, risks, checklists. Start here. |
| `venue-requirements.md` | yes | Requirements of JOSS, SoftwareX, JMLR MLOSS, and ICAPS 2027 as collected on 2026-10-05, with a status note on what has since been re-verified. |
| `icaps27/` | yes | The submission itself: abstract, paper sources, figure scripts, artifact README template; later the author response and reviews. See `icaps27/README.md`. |
| `joss/` | no | JOSS notes and the gap analysis behind PR #2. The JOSS paper itself is `../paper/` on `main`. |
| `softwarex/` | no | SoftwareX notes and the official templates and reviewer form (downloaded 2026-10-26). |
| `jmlr/` | no | JMLR MLOSS notes. |
| `archive/` | no | Superseded documents: the venue-readiness gap analysis behind PR #3 and the first draft of the ICAPS plan. |

## Rules for this branch

- **Do not push this branch to the public `origin` while the ICAPS paper is under
  double-blind review (Dec 14, 2026 to Feb 26, 2027).** Once `icaps27/paper/` holds the
  manuscript, a public push de-anonymizes it. Back it up to a private remote if needed.
- Nothing here is part of the package, the docs build, the sdist, or the test suite
  (`pytest` collects only `tests/` and `s3e/`). CI's `ruff check .` does lint any Python file
  here, so keep scripts ruff-clean.
- `.gitignore` ignores `*.json`, `*.png`, `*.pdf`, and `*.log` repository-wide. Figures under
  `icaps27/paper/figures/` are exempted; raw experiment results (JSON) belong in the separate
  experiments repository described in `../icaps27-tool.md`, Section 7, not here.

## Sequencing (decided 2026-10-25)

1. ICAPS 2027 tool paper: abstract Dec 7, paper Dec 14, 2026; notification Feb 26, 2027.
2. JOSS: submit only after Feb 26, 2027, declaring the ICAPS paper (JOSS allows companion
   papers; ICAPS forbids concurrent journal review).
3. ICAPS 2027 System Demonstration: call expected around March 2027.
4. SoftwareX and JMLR MLOSS: deferred. Both are journals (no concurrent submission with
   ICAPS), SoftwareX would duplicate the JOSS software paper, and JMLR MLOSS requires that the
   software not have been published elsewhere.

The lab's other ICAPS 2027 submission (the ViPlan++ narrative) lives in its own repository
and is not S3E venue material.
