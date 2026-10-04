---
title: 's3e: A Python library for probabilistic symbolic state estimation with vision-language models'
tags:
  - Python
  - automated planning
  - PDDL
  - vision-language models
  - state estimation
  - calibration
  - robotics
authors:
  - name: Guy Azran
    orcid: 0000-0002-2840-2584
    corresponding: true
    affiliation: 1
  - name: Yuval Goshen
    orcid: 0009-0000-4540-4621
    affiliation: 1
  - name: Kai Yuan
    affiliation: 2
  - name: Sarah Keren
    orcid: 0000-0001-7211-753X
    affiliation: 1
affiliations:
  - name: Taub Faculty of Computer Science, Technion – Israel Institute of Technology, Israel
    index: 1
    ror: 03qryx823
  - name: Intel Labs
    index: 2
date: 5 October 2026
bibliography: paper.bib
---

<!--
  TODO(authors) before submission:
  - Confirm the author list (JOSS: general supervision alone does not qualify
    for authorship), ORCID iDs, and affiliations. The 2025 workshop abstract
    also lists Guy Azran at Intel Labs; add the country for Intel Labs.
  - Confirm or complete the research-impact claims marked below.
  - Complete the AI usage disclosure with exact model versions.
  - Add funding sources and the sponsor-involvement statement.
  Keep CITATION.cff (title, authors) in sync with this file.
-->

# Summary

Automated planners reason over *symbolic states*: sets of facts such as
`on(a,b)` ("block a is on block b"), written in the Planning Domain Definition
Language [PDDL, @mcdermott1998pddl; @haslum2019pddl]. A robot, however, sees
images, not facts. `s3e` (Semantic Symbolic State Estimation) is a Python
library that turns images into such facts using vision-language models (VLMs),
AI models that answer questions about pictures. Given a planning problem and
one or more images of a scene, it lists every fact that could hold, asks the
model about each one, and reports how likely each fact is, together with a
true, false, or undecided verdict that a planner can use. Rather than reading
the model's written answer, `s3e` reads the probability the model assigns to
answer words such as "yes" and "no" (and optionally "unknown"). These scores
can be corrected against labeled examples, combined across camera views, or
passed to planners that handle uncertainty.

# Statement of need

Combining foundation models with sound classical planners is an active
research direction [@kambhampati2024llmmodulo]. In its visual
"VLM-as-grounder" form, a VLM grounds observations into symbolic predicates
and a planner reasons over them [@zhang2023tpvqa; @merler2025viplan]. Every
project that adopts this pattern needs the same plumbing: grounding predicates
over the problem's objects, phrasing a question per predicate, prompting a
particular model API with images, mapping outputs to truth values, and
handling answers that fall outside the expected vocabulary. Ad-hoc
implementations typically parse generated text, which discards the model's
uncertainty, and tie experiments to one model provider, which makes
cross-model comparisons and calibration studies laborious.

`s3e` packages this pipeline as a reusable, backend-agnostic library. Its
target audience is researchers in task planning, task-and-motion planning,
embodied AI, and neuro-symbolic reasoning who need symbolic state estimates
from images, and, because its query engine works without PDDL, anyone who needs
probabilistic yes/no or multiple-choice answers from VLMs. It treats the
uncertainty of an estimate as a first-class output: probabilities derived from
answer-token mass [@kadavath2022know], an optional explicit abstention option,
and optional post-hoc Platt scaling [@platt2000probabilities] fitted offline,
since neural-network confidences are often miscalibrated [@guo2017calibration].

# State of the field

Symbolic-planning libraries such as Unified Planning [@micheli2025unified] and
PDDLGym [@silver2020pddlgym] model, ground, simulate, or solve planning
problems, but assume the symbolic state is given. VLM evaluation harnesses
such as LMMs-Eval [@zhang2025lmmseval] and VLMEvalKit [@duan2024vlmevalkit]
run models on benchmarks and report aggregate metrics; LMMs-Eval can score
answer log-likelihoods, but only inside benchmark task definitions. The
closest tool is `t2v_metrics`, the package behind VQAScore
[@lin2024vqascore], which scores the probability of "Yes" for user-supplied
images and question templates across local and API models; it targets the
evaluation of text-to-visual generation and has no predicate grounding,
multi-option answer spaces, abstention, or calibration. Constrained-generation
libraries such as Outlines [@willard2023outlines] restrict what a model may
generate but do not report per-option probabilities. Research systems that
ground planners with VLM questions, including TP-VQA [@zhang2023tpvqa],
DKPROMPT [@zhang2024dkprompt], the VLM-as-grounder methods evaluated in ViPlan
[@merler2025viplan], and predicate-learning approaches
[@liang2025visualpredicator; @athalye2026pixels], implement the idea inside
experiment code tied to particular domains and benchmarks rather than as an
installable library.

None of these provides the combination `s3e` targets: PDDL grounding,
pluggable predicate-to-question translation, token-probability scoring over
configurable answer spaces, interchangeable local and hosted model backends,
and offline calibration behind one interface. Adding planning-specific
grounding and calibration to an evaluation harness or a benchmark codebase
would conflict with their purpose, so `s3e` was built as a separate library
that builds on existing ecosystems instead of reimplementing them: PDDL
parsing and grounding are delegated to Unified Planning, inference to Hugging
Face Transformers [@wolf2020transformers], vLLM [@kwon2023vllm], or the
OpenAI API, and fitting to scikit-learn [@pedregosa2011sklearn]. Estimated
states can be converted to Unified Planning state objects.

# Software design

`s3e` is organized as four layers, each usable without the layers above it
(\autoref{fig:pipeline}): *backends* expose Hugging Face, vLLM, and OpenAI
models through one `VLMBackend` interface; the *engine* (`QueryEngine`)
answers free-form queries about images against an answer space, with no PDDL
involved; *calibration* fits and applies calibrators to prediction data; and a
thin *PDDL facade* (`SemanticStateEstimator`) grounds a domain and problem,
translates predicates into queries, and drives the engine. Four design
decisions shape the library.

![The `s3e` pipeline. Colors mark the layer that implements each step; each
layer can be used without the layers above it, and calibration is
optional.\label{fig:pipeline}](s3e-pipeline.png)

**Probabilities instead of parsing.** An answer space (`BinaryAnswers`,
`CategoricalAnswers`) maps each option to the token strings that express it,
automatically expanding labels into case and leading-space variants because
tokenizers distinguish "Yes", " yes", and "YES". Backends report probability
mass for exactly the requested tokens and whether the model's most likely
token was among them, a contract checked by shared tests for the local
backends. Predictions therefore record when a model answered outside the
answer space instead of silently misreading it, and, for local backends,
options with no single-token form are rejected before inference. Reading one
next-token distribution is cheap and batches well, at the cost of requiring
answers expressible as single tokens; a text-matching mode remains for APIs
that do not expose log-probabilities.

**Store data, derive views.** A `Prediction` stores raw masses (per option,
for the explicit null option, and the unassigned remainder) plus any
calibrated probability; probabilities, argmax answers, abstentions, and
thresholded states are derived on demand. Because no derived view requires
re-running the model, users can change thresholds, recalibrate, or average
predictions across views after the fact, and results serialize to JSON
without backend dependencies. This matters when VLM inference dominates the
cost of an experiment.

**Offline calibration with provenance.** Calibration is split into an
expensive step that queries the model once on labeled scenes and a cheap,
repeatable fit of Platt scaling per global, lifted (per predicate name), or
grounded predicate group. Fitted calibrators record the scoring mode, the
answer space, and a fingerprint of the PDDL domain, and the estimator refuses
to apply a calibrator whose recorded scoring mode, answer space, or domain
fingerprint differs from its own.

**A lightweight core with optional backends.** The core depends only on NumPy,
Pillow, and tqdm; model backends, Unified Planning, and scikit-learn are
optional extras imported lazily, with errors naming the extra to install. This
keeps the engine usable without GPUs or PDDL, lets users add a model by
implementing a single `query` method, and lets the test suite run on a CPU
without downloads.

# Research impact statement

`s3e` is the reference implementation of S3E [@azran2025s3e;
@azran2025s3eicra], which estimates the full PDDL state by asking a VLM about
every grounded predicate; the library grew out of that work's research code
and has been released on PyPI since May 2026. S3E is cited by the
independently developed ViPlan benchmark for VLM-grounded planning
[@merler2025viplan]. Our follow-up work [@azran2026bridging] extends
VLM-as-grounder planning to belief-space planning over the Most Likely Subset
of States (MLSS), using the next-token probabilities of predicate queries.
<!-- TODO(authors): if true, state that the experiments in azran2026bridging
were run with s3e (give the version and a code link), and cite any other
papers, preprints, or groups that use s3e. JOSS requires demonstrated
research use; contract tests alone show API compatibility, not use. -->
Within our group, `s3e` is the state-estimation component of two research
workflows, the MLSS prediction-and-calibration pipeline and a ViPlan-based
evaluation pipeline, whose usage patterns are pinned by contract tests in the
repository so that library changes cannot silently break them.

# AI usage disclosure

Generative AI coding assistants were used in developing `s3e`. Of the 195
commits made between March and August 2026, 99 were made with Anthropic's
Claude Code, and so were all commits of the October 2026 work preparing this
submission; each records the assisting model in a `Co-Authored-By` trailer.
This assistance covered implementing the 0.4 architecture redesign, tests,
documentation, the walkthrough notebook, docstrings, packaging, continuous
integration, community guidelines, and the first draft of this paper. An
initial implementation of the vLLM backend was drafted with OpenAI Codex. The
authors reviewed all AI-assisted code, tests, and text. The CPU-only test
suite (348 tests) runs in continuous integration on every change, and
real-model tests are run manually.
<!-- TODO(authors): (1) list the exact model versions from the Co-Authored-By
trailers (`git log`) and the Codex version; (2) state whether any AI tools
were used before March 2026 and whether any other tools were used; (3) confirm
that the authors made the core design decisions (e.g. of the 0.4
architecture). -->

# Acknowledgements

We thank Adi Abargil for contributions to the early research codebase.
<!-- TODO(authors): acknowledge all sources of financial support and state
whether the sponsors had any involvement in the work. -->

# References
