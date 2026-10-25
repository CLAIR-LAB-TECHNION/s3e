# ICAPS 2027 tool-paper submission

Deadlines (23:59 UTC-12): abstract **Dec 7, 2026**, paper and supplementary material
**Dec 14, 2026**, author response Feb 1–4, 2027, notification Feb 26, 2027. Main track, paper
type "Tool paper", long format (8 pages + references), AAAI-27 author kit. The strategy and
checklists are in `../../icaps27-tool.md`.

## Layout

```
icaps27/
  README.md                     this file
  abstract.md                   title, draft abstract, keywords, paper type and tags (for Dec 7)
  paper/                        LaTeX sources once the AAAI-27 kit is in place (main.tex, sections/, refs.bib)
    figures/                    figure sources and rendered figures (png/pdf are un-ignored here)
      make_pipeline_figure.py   draws the pipeline figure used as Figure 1
  artifact/
    README-template.md          README for the experiments/artifact repository
  response/                     author-response drafts (Feb 1–4, 2027)       [create when needed]
  reviews/                      reviews received (Feb 26, 2027)              [create when needed]
```

The experiments and the artifact are a **separate repository** (`s3e-icaps27-experiments`,
see the strategy, Section 7): it is what gets mirrored anonymously at submission, it pins
`s3e` from PyPI rather than copying the library, and its raw JSON results would be swallowed
by this repository's `*.json` ignore rule.

## Double-blind reminders

- Name the tool S3E; cite Azran et al. (2025, 2026) in the third person; never name the lab
  or institution; scrub PDF and figure metadata.
- Links in the paper point to the anonymized experiments mirror; linked content is frozen
  after Dec 14; reviewers are not required to open it, so the 8 pages must stand alone.
- No arXiv posting between Nov 30, 2026 and Feb 26, 2027.
