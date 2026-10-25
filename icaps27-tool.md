# S3E as an ICAPS 2027 Tool Paper: Strategy for Maximal Chance of Acceptance

Status: planning document, written 2026-10-25, kept on the `icaps27-tool` branch together with `venues/` (see `venues/README.md` for the layout and for what is tracked versus ignored). Not part of the package or the docs build. Do not push this branch to the public remote while the paper is under double-blind review.

Deadline arithmetic: today is Oct 25, 2026. The abstract is due Dec 7 (6 weeks), the paper Dec 14 (7 weeks), both at 23:59 UTC-12.

---

## 0. Executive summary

- **Where to submit.** ICAPS 2027 main track, paper type "Tool paper", long format (8 pages + references), AAAI-27 author kit. There is no separate tools track and no artifact-evaluation committee in 2027; tool papers are reviewed by the regular PC "according to their nature and submission category".
- **What the reviewers will check.** The paper must cover: the problem and its importance, related work, the techniques and their novelty, the tool's construction and unique features, how it is used, and an evaluation of usefulness with comparisons to other tools and to earlier versions (the ICAPS 2025 tools-call list, still the de-facto rubric). Reviewers imported from TACAS habits will additionally judge maturity, user base, availability, and robustness, and will try to install the tool.
- **Where S3E is strong.** Engineering maturity, documentation, tests, CI, packaging, reproducibility provenance, and a clear set of engineering techniques that no other tool combines.
- **Where S3E is weak.** No experiment puts a planner in the loop; no numbers on scaling; almost no visible users outside the authors; no Zenodo DOI; the one external citation (ViPlan) criticizes S3E as "not scalable" and for not measuring planning success.
- **What to do, with zero changes to `s3e/`.** (1) Frame the paper around the library and its engineering techniques, not the 2025 method. (2) Run a closed-loop evaluation (image → S3E → Unified Planning → plan validity) plus scaling, scoring-mode, calibration, thinking, cross-tool, and cross-version experiments from scripts in a separate experiments repository. (3) Ship a Docker-based artifact with a CPU smoke test and cached results, anonymized at submission, with a Zenodo DOI at camera-ready. (4) Fix repository metadata and build evidence of external use before the Feb 1–4 author response. (5) Hold the JOSS paper until after the Feb 26 notification; submit an ICAPS 2027 System Demonstration in spring as well.

---

## 1. The rules (verified against the ICAPS 2027 call on 2026-10-25)

### 1.1 Dates and venue

| Item | Date |
|---|---|
| Abstract submission | Dec 7, 2026, 23:59 UTC-12 |
| Paper submission (and supplementary material) | Dec 14, 2026, 23:59 UTC-12 |
| Author response period | Feb 1–4, 2027 |
| Author notification | Feb 26, 2027 |
| Workshops and tutorials | Jun 28–29, 2027 |
| Main conference | Jun 30 – Jul 2, 2027, Columbia, SC, USA |
| Camera-ready, demos call | not yet published (2026's demo deadline was Apr 6) |

At least one author must register and present in person, or the paper is removed from the proceedings. Start travel/visa logistics early.

### 1.2 The tool-paper category (verbatim from the call)

> Tool papers, which describe systems that are useful to and of interest to the planning and scheduling community, and which are built using novel algorithmic and engineering techniques. Examples include: integrated planning systems, model checkers and synthesis tools, libraries for constructing, managing, and transforming representations of planning and scheduling problems, and applications for visualizing, benchmarking, and comparing planners or other types of tools.

Implications: the paper must name the *techniques* (algorithmic or engineering) and argue *usefulness to the P&S community*. S3E fits two of the call's own examples: a library that constructs a planning representation (the initial state / belief) from images, and a harness for benchmarking and comparing VLM grounders.

Track: main track. The "Planning Under a Different Name" special track is for work framed in another community's terminology and benchmarks; S3E is framed in PDDL and Unified Planning terms and should be judged by ICAPS reviewers.

### 1.3 Format and length

- AAAI-27 author kit. Long: 8 pages + unlimited references. Short: 4 pages + references. Over-length (content on page 9 / page 5 other than references and the optional ethics statement) is rejected without review.
- "Short papers may be narrower in scope than long papers, yet still make a complete and significant contribution." Planner Museum (ICAPS 2026) was accepted as a 6-page short tool paper, so a 4-page fallback is credible.

### 1.4 Double-blind rules

- Omit names and affiliations; refer to own prior work in the third person; acknowledgments must not identify authors.
- "Links included in submissions must be anonymized; camera-ready versions should use a persistent archive with a DOI."
- "Anonymized external links (e.g., anonymous code repositories or datasets) may be included in the paper or supplementary material ... The linked content must not be modified after the submission deadline, and reviewers are not required to visit external links."
- arXiv posting is discouraged from two weeks before the deadline (Nov 30, 2026) until notification (Feb 26, 2027).

### 1.5 Supplementary material

Optional; anonymized; total ≤ 100 MB; due with the paper; not in the proceedings. "Reviewers are not required to read or evaluate supplementary material, so the main paper must be self-contained." (ICAPS 2026 exempted tools papers from this; the 2027 call does not. Plan for the 8 pages to carry every claim.)

### 1.6 Reproducibility requirements (new in 2027, stated in the main paper)

- Hardware and OS; time and memory limits per instance.
- The exact benchmark set, how instances were selected, where to obtain them.
- The version (release or commit) of every planner, solver, or library, and all non-default parameters.
- "For methods using foundation models, the exact model identifiers and versions, decoding parameters, the prompts, and the dates on which the models were queried." Long prompts may go to the supplement if the main paper describes their structure.
- Randomness: aggregate over multiple runs; report the number of runs and a dispersion measure; do not claim improvements within the reported variation.
- "We strongly encourage authors to release code, learned models, benchmark instances, and raw per-instance results, and to include a brief availability statement."
- "Reviewers will be asked whether a paper reports enough detail for an independent group to reproduce its main results."

### 1.7 Dual submission and scope

- "Papers submitted to ICAPS-27 may not be submitted to other conferences or journals during the ICAPS-27 review period, nor may they be already under review, accepted, or published in other conferences or journals."
- Work that does not clearly connect to automated planning or scheduling is rejected without review.

---

## 2. What makes tool papers get accepted

### 2.1 The rubric (ICAPS 2025 Call for Tools Papers, the only ICAPS text that spells it out)

> Papers in this category must clearly describe the problem to be solved, its importance, related work, the techniques used in the tool and their novelty, the construction of the tool, its unique features, a discussion of how the tool is used and an evaluation of the usefulness of the tool along with a comparison with other tools and previous versions of the tool.

The same call says tools papers may be "refinements or optimizations of existing algorithms or techniques" that are "novel from an engineering standpoint" (its example: two-watched literals in SAT solving). Artifact criteria: consistency with and replicability of the paper's results; completeness; documentation and ease of use; availability in a permanent online repository; broad platform support (Ubuntu 22.04+, Windows, macOS); a README with an install script (Docker), a quick functional check, and replication instructions with resource estimates.

### 2.2 How reviewers actually score

- ICAPS 2025 acceptance questionnaire (area-chair meta-review form): technical soundness; applicability to the chosen topic; context with earlier work; novelty ("not previously published by others or by the same authors in an archival venue"); completeness (definitions, experimental data, literature, motivations, implications); importance; plus form questions (abstract reflects content, what is new is clear, why it matters is clear, figures and examples help). "Papers that receive more positive responses than negative responses ... will be more likely to be accepted."
- ICAPS 2026 reviewing guidance: "Significance involves not only novelty but potential impact on the field." Reviewers must list the questions whose answers would change their recommendation, which authors answer in the response period.
- TACAS tool-paper criteria that ICAPS borrowed: maturity as a software product (implementation, documentation, development process), user base, availability and licensing, robustness; reviewers are asked to download the tool and try the examples.

### 2.3 Precedents at ICAPS

- 2025: Posthoc (visualisation platform for search), HDDLGym (HDDL problems as Gym environments).
- 2026: Planner Museum (historical IPC planners kept runnable, with a 28-year evaluation; a short paper).

Common pattern: a clear community need, a runnable system people can adopt the same day, and an evaluation that teaches the community something it did not know. A feature list alone does not get in.

### 2.4 Translated into a scoring strategy for S3E

| Reviewer question | What must be in the paper |
|---|---|
| Is this planning? | PDDL in, Unified Planning state out, a planner in the evaluation loop, planning success as a headline metric. |
| What is novel? | Engineering techniques (Section 3.3), stated as problem → technique → consequence, with a version history showing they did not exist in 0.2. |
| Who needs it? | The plumbing every VLM-as-grounder project re-implements (ViPlan, TP-VQA, DKPROMPT, predicate-learning work), with citations, and the feature matrix showing no existing tool provides it. |
| Who uses it? | Research pipelines on the current API, the ViPlan citation, any external adopters recruited before Feb 1 (Section 6). |
| Can I reproduce it? | Provenance for every run (the benchmark script already writes it), pinned versions, multi-seed dispersion, availability statement, Docker artifact with a CPU smoke test. |
| Is it mature? | Tests, CI matrix, platforms, releases, docs, governance (Section 3.1 facts). |

---

## 3. Where S3E stands (2026-10-25)

### 3.1 Facts to cite in the paper (all verified today)

- 306 commits since Aug 5, 2024; public repository since Apr 15, 2026; 7 PyPI releases (0.1.0 → 0.5.0); MIT.
- ~5,100 lines of library code, ~7,600 lines of tests, 688 tests (654 fast, 34 real-model).
- CI on every push: Linux Python 3.10–3.14, macOS and Windows (3.10, 3.14), a lowest-dependency job, a coverage job with vLLM that fails below 97%, ruff, build + twine check, CITATION.cff validation, strict Sphinx build. Weekly real-model workflow executes the README Quick Start, getting-started page, and tutorial notebook.
- Docs live at s3e.readthedocs.io (user guide, developer guide with extension points, API reference, tutorial notebook, Colab link). `examples/custom_backend.py` runs the whole pipeline offline in seconds; `examples/blocksworld_benchmark.py` measures a model and writes per-predicate results with raw masses plus provenance (library versions, model id and resolved revision, full prompts, answer tokens, decoding settings, calibrator, dates, hardware).
- Project hygiene: CHANGELOG (Keep a Changelog), CONTRIBUTING with release process, SECURITY, GOVERNANCE, CODE_OF_CONDUCT, SPDX headers enforced by a test, CITATION.cff, issue templates.

### 3.2 Adoption signals (the weak spot)

- GitHub: 1 star, 0 forks, 0 external issues or pull requests; no description, topics, or homepage set; no branch protection; no Zenodo record.
- PyPI: 1,090 downloads in 102 days without mirrors; 156 in the last month.
- External citation: ViPlan (Merler et al., EMNLP 2026 Findings) cites S3E twice: in related work ("enumerates all possible predicate-object combinations", "not scalable", and it "does not measure planning success rates with the estimated states") and in limitations (as the way to automate predicate-to-question templates with an LLM).
- Own work: LM4Plan@AAAI 2025 and ICRA 2025 workshop papers (the method), NeSy 2026 (belief-space planning on S3E probabilities). Two in-group pipelines use pre-0.4 releases.

### 3.3 The engineering techniques to present as the contribution

Each is a problem → technique → consequence paragraph in the paper; none is in the 2025 workshop papers.

1. **Answer spaces over token mass.** Problem: tokenizers split "Yes", " yes", "YES" differently and APIs differ. Technique: `BinaryAnswers`/`CategoricalAnswers` map labels to token sets with automatic variant expansion, single-token validation for local backends, an explicit null ("unknown") option, and tracking of unassigned mass and whether the argmax was an answer token. Consequence: one scoring contract across HuggingFace, vLLM, and OpenAI; "no answer" is a first-class outcome instead of a parsing failure.
2. **Store data, derive views.** Problem: VLM inference dominates experiment cost, yet thresholds, calibration, and aggregation change afterwards. Technique: `Prediction` stores raw masses and any calibrated override; probability, answer, has-answer, confidence, distributions, and thresholded states are derived lazily; `PredictionSet` averages across views and round-trips to JSON without a backend. Consequence: recalibrate, re-threshold, and re-aggregate offline.
3. **Scoped offline calibration with provenance.** Technique: collect once, fit Platt scaling per global / lifted / grounded group, store scoring mode, answer space, and a canonical domain fingerprint; refuse to apply a mismatched calibrator. Consequence: calibration studies without re-querying; safety against silent misuse.
4. **Thinking-model handling.** Problem: reasoning models put the next token at the start of a thought, not the answer. Technique: probe the chat template for its thinking switch by rendering, close the block when thinking is off, budgeted greedy reasoning then scoring when on, delimiter auto-detection across model families, reasoning text exposed. Consequence: logprob scoring works on Qwen3.5, Gemma 4, Mistral reasoning models.
5. **Full-vocabulary logprobs under vLLM** with tokenization-consistency checks, so multi-GPU scoring returns the same contract as the HuggingFace backend.
6. **Relevant-atom masking and multi-view averaging** (`predicates=`, `estimate_averaged`), the two levers against cost and single-view error.
7. **Lightweight core, lazy optional backends**; a new model is one `query` method; the test suite runs on CPU with deterministic fakes.

### 3.4 Limitations to state candidly (reviewers reward this)

Boolean predicates only (no numeric fluents or functions); static grounding over all objects unless masked; logprob mode needs single-token answers; single-shot estimation with no temporal filtering across steps; small open models are poorly calibrated and sometimes wrong, which is what the calibration layer is for and what the evaluation should show rather than hide.

---

## 4. The paper

### 4.1 Working title and contribution statement

Title: "S3E: A Library for Probabilistic Symbolic State Estimation from Images for PDDL Planning" (the name S3E may be used; the paper cites Azran et al. 2025 in the third person for the method and positions itself as the library paper).

Contributions (listed in the introduction):
1. A backend-agnostic library that turns a PDDL domain, a problem, and images into probabilities over grounded predicates and a planner-ready state.
2. Engineering techniques (Section 3.3) that make token-probability scoring reliable across model families and APIs, and keep expensive inference separate from cheap offline analysis.
3. An evaluation of usefulness: planning success in the loop, scaling behavior, probabilities versus parsing, calibration, reasoning models, and comparisons with other tools and earlier versions.
4. A reproducible artifact and a tested, documented, multi-platform release.

### 4.2 Outline and page budget (8 pages, AAAI two-column)

| Section | Pages | Content |
|---|---|---|
| 1 Introduction | 0.75 | The VLM-as-grounder plumbing every project re-implements; why parsing text loses uncertainty; contributions. |
| 2 Related tools | 0.75 | Feature matrix (Table 1): Unified Planning, PDDLGym, LMMs-Eval, VLMEvalKit, t2v_metrics, Outlines, research code (TP-VQA, DKPROMPT, ViPlan grounders, predicate-learning). Rows: PDDL grounding, pluggable translation, per-option probabilities, null option, multi-backend, offline calibration, planner hand-off. |
| 3 S3E in use | 1.0 | Figure 1 (pipeline, from `paper/s3e-pipeline.png`), the Quick Start as a listing, the closed-loop snippet (estimate → `to_up_state` → set initial values → `OneshotPlanner`), the engine-only and categorical uses. |
| 4 Design and techniques | 2.0 | Layered architecture; the seven techniques as problem → technique → consequence; one small figure for the prediction data model or the thinking-block handling. |
| 5 Evaluation | 2.5 | E1–E7 below, with the reproducibility paragraph (hardware, versions, model identifiers and revisions, decoding parameters, prompt structure, query dates, seeds). |
| 6 Development, versions, availability | 0.75 | 0.2 → 0.5 comparison (Table), tests/CI/platforms, release and governance process, availability statement, limitations. |
| 7 Conclusion | 0.25 | What the community can now do that it could not before. |

Figures/tables: Fig 1 pipeline; Fig 2 planning success vs predicate accuracy across models; Fig 3 scaling (queries and time vs objects, with and without masking; HF vs vLLM); Table 1 feature matrix; Table 2 logprobs vs text parsing; Table 3 calibration and views; Table 4 versions and engineering facts.

### 4.3 Draft abstract (anonymized; refine after results)

> Planners that act from camera images need the symbolic state that PDDL assumes given. Research systems that ground classical planners with vision-language models (VLMs) re-implement the same pipeline: ground predicates over the problem's objects, phrase a question per predicate, prompt a model with images, and turn its output into truth values. Most parse generated text, which discards the model's uncertainty, and bind experiments to one model API. We present S3E, an open-source Python library that turns a PDDL domain, a problem, and one or more images into a probability for every grounded predicate and a Unified-Planning-compatible state. S3E reads the probability mass of answer tokens over configurable answer spaces with an explicit "unknown" option, across HuggingFace, vLLM, and OpenAI models, keeps raw masses so that thresholds, calibration, and multi-view aggregation can be changed offline, fits scoped Platt calibrators with provenance checks, and handles reasoning models whose next token is a thought rather than an answer. We evaluate usefulness by closing the loop with a planner on rendered and photographic scenes, measuring plan validity alongside predicate accuracy, scaling with the number of objects, the cost and benefit of token-probability scoring, calibration, and reasoning, and we compare with existing tools and earlier versions. S3E is tested on Linux, macOS, and Windows, documented, and released on PyPI with a reproducible artifact.

### 4.4 Double-blind handling for a public, named tool

- The paper names S3E and cites the 2025 workshop papers and the 2026 NeSy paper in the third person. It never names the institution or lab.
- The anonymized artifact is the *experiments* repository, which pins `s3e==0.5.x` from PyPI, plus a supplementary ZIP. Do **not** mirror the library source anonymously: every file carries an SPDX copyright line naming the lab, `CITATION.cff` names the authors, and stripping copyright notices from an MIT-licensed copy would be improper. ICAPS 2025 explicitly treated tools papers as "de-facto single-blind" for this reason; 2027 asks for best-effort anonymization of links. Residual risk: a reviewer who installs the package sees the copyright line. Accept it (it is the norm for tool papers) or email the program chairs before Dec 7 to confirm.
- Do not post the manuscript to arXiv between Nov 30 and Feb 26; do not put the manuscript, the experiments, or an "icaps" branch in the public `s3e` repository during review; do not change anything the anonymized links point to after Dec 14.
- Check PDF metadata, figure metadata, and notebook outputs for names and paths.

---

## 5. Evaluation of usefulness (all scripts live in a separate experiments repository)

General protocol: every run goes through `examples/blocksworld_benchmark.py` or a sibling script that writes the same provenance JSON; three seeds minimum for scene generation and model sampling; per-instance raw results kept; report mean and standard deviation (or min–max) and never claim a difference inside the spread; record model identifiers and HuggingFace revisions, decoding parameters, prompts, and query dates; fix per-query time and GPU memory limits and state them.

Models (choose four to six): SmolVLM-256M-Instruct (CPU smoke test and the "small model" row), one or two mid-size open VLMs already exercised by the code base (Qwen2-VL-7B-Instruct or Qwen3.5-4B; Gemma 4 E2B/E4B), one reasoning model (Ministral-3-3B-Reasoning or Qwen3.5 with thinking), one hosted model through the OpenAI backend.

Domains: (a) rendered Blocksworld from the existing generator at 3, 4, 5, 6, 7 blocks (ground truth is exact); (b) one photo-realistic or real domain so that reviewers do not dismiss the results as synthetic: the public ViPlan Blocksworld/household images if their license allows redistribution of the subset used, otherwise lab photographs of colored blocks with hand-labeled states. Publish the instances or the generator with seeds.

- **E1 Closed-loop planning (headline).** For each scene: estimate the state, build a Unified Planning problem whose initial values are the estimate, solve with Fast Downward through Unified Planning (or another UP engine), validate the plan against the ground-truth initial state with UP's plan validator, and check goal achievement. Report plan-validity rate and goal-achievement rate next to predicate accuracy, state exact-match rate, and F1 on true predicates (class imbalance makes accuracy flattering). Show that calibration or masking that improves predicate metrics also improves planning outcomes, or report honestly where it does not. This answers the ViPlan criticism directly and makes the paper a planning paper.
- **E2 Scale.** Number of grounded predicates and wall-clock per scene versus object count, with and without relevant-atom masking; throughput versus batch size; HuggingFace single GPU versus vLLM tensor-parallel on the same model. Peak memory per mode (the benchmark script records it).
- **E3 Probabilities versus parsing.** On the same models and scenes: `logprobs` versus `text_match` versus a naive regex-over-generated-text baseline implemented in the experiments repo (stand-in for ad-hoc code). Metrics: accuracy, no-answer rate, Brier, ECE, time per query.
- **E4 Calibration and views.** ECE and Brier before and after Platt scaling at global, lifted, and grounded scope; transfer across problem instances (fit on some problems, test on others); multi-view averaging with two to four camera views.
- **E5 Reasoning models.** Thinking off (switch), off (block closed), and on with a token budget: accuracy versus time per query; the notebook's Section 11 protocol scaled to the full scene set.
- **E6 Other tools.** Where another tool can do the same job, run it on the same images and questions: t2v_metrics' P("Yes") scoring on a model both support; LMMs-Eval log-likelihood scoring via a custom task if cheap. Otherwise the feature matrix stands, with a sentence on why each tool cannot be used for this job.
- **E7 Earlier versions.** Install 0.2.0 and 0.5.0 in separate environments; compare API surface, feature coverage, and throughput on the same scenes; summarize the 0.4 redesign and the 0.5 additions from the CHANGELOG. (This is the "previous versions" requirement.)

Compute plan: on the lab cluster, a mid-size VLM at batch 8–16 handles tens of queries per second on one GPU; 100 scenes × 5 block counts × ~30–100 predicates × 2 scoring modes is on the order of 10^5 queries per model, i.e. hours, not days. Run E1–E5 first; E6 and E7 are cheap and can be trimmed if time runs short.

Availability statement template (for Section 6): "S3E is released under the MIT license on PyPI and GitHub with documentation and a tested tutorial. The experiment scripts, generated instances, per-instance raw results, fitted calibrators, prompts, and a Dockerfile that reproduces every table and figure are available at [anonymized link; DOI at camera-ready]."

---

## 6. Adoption and visibility (repository settings and outreach, no code)

Do this week (owner: maintainer; each is minutes):
- Set the GitHub repository description, topics (automated-planning, pddl, vision-language-models, state-estimation, calibration, unified-planning, robotics), and homepage (readthedocs).
- Enable the Zenodo GitHub integration so the next release receives a DOI (needed for the camera-ready link and for JOSS later).
- Enable branch protection on `main`, private vulnerability reporting, and Discussions.
- Add a "Used in" list to `docs/citing.md` once the in-group pipelines are on 0.5 (keep it generic, no project internals).

Before Dec 14 (so the paper can cite them) and again before Feb 1 (so the author response can):
- Move the lab's NeSy/MLSS and ViPlan-based pipelines to 0.5 so the paper can truthfully say research pipelines use the current API.
- Contribute an S3E-backed grounder to the ViPlan benchmark repository (external code, their license). It turns the one public critic into a user and gives a concrete scalability answer in their own setting.
- Announce the 0.5/0.6 release with the tutorial on the planning-list mailing list, the Unified Planning community channels, and the Hugging Face community; ask two or three groups doing VLM-as-grounder work to try it and open issues.
- Give a short talk or demo in the lab's wider network; a recorded 10-minute video doubles as the ICAPS 2027 System Demonstration submission.

Honesty rule: cite only users you can name or link; do not inflate the user base in the paper or the response.

---

## 7. The artifact

Repository `s3e-icaps27-experiments` (private until Dec 7, then mirrored anonymously; public with a Zenodo DOI at camera-ready). Contents:

```
README.md          ABSTRACT, LICENSE, link to the artifact, install, smoke test, replication, resources
Dockerfile         Ubuntu 22.04, CPU PyTorch, s3e==0.5.x pinned, unified-planning + up-fast-downward
environment.lock   exact pip freeze used for the paper runs
Makefile           make smoke | make reproduce | make figures
experiments/       e1_closed_loop.py ... e7_versions.py (each writes provenance JSON)
instances/         generated scenes and seeds; photo subset with labels and license
results/raw/       per-instance JSON from the paper runs (regenerate figures without a GPU)
results/figures/   the paper's figures and tables, regenerated by make figures
prompts/           every prompt and answer-token set used
```

README skeleton (follows the ICAPS 2025 artifact template, which reviewers still expect):

```
# Artifact for "S3E: A Library for ..."
## Abstract           what the artifact contains and which claims (tables/figures) it supports
## License            MIT; permission for PC members to evaluate
## Requirements       Docker, 8 GB RAM for the smoke test; GPU (e.g. 24 GB) for full replication
## Quick check (10 min, CPU)   docker build . && docker run ... make smoke   (SmolVLM on 3 scenes, E1 on one scene)
## Full replication   per experiment: command, expected runtime, hardware used in the paper
## Regenerate figures from the shipped raw results (no GPU)
## Mapping            claim in paper -> script -> output file
```

Supplementary ZIP (≤ 100 MB): the README, full prompts, the appendix PDF with extra tables, and the raw results if they fit; otherwise the anonymized link carries them. Verify the Docker image builds and `make smoke` passes on a clean machine (and on macOS without Docker via `pip install`) before Dec 7.

---

## 8. Sequencing with the other venues

- **JOSS** (paper already drafted in `paper/`): JOSS welcomes companion publications and only asks that related publications be declared; it requires more than six months of public history (met since Oct 15, 2026) and evidence of research use. ICAPS forbids a paper "already under review ... in other ... journals". Hold the JOSS submission until after Feb 26, 2027, then submit it declaring the ICAPS paper. The ICAPS paper and the ViPlan citation strengthen JOSS's "being used for research" gate.
- **ICAPS 2027 System Demonstrations** (call expected around March 2027, deadline around early April): a 2-page abstract plus a ≤ 10-minute video weighted as heavily as the abstract. Submit regardless of the paper outcome; the closed-loop demo (camera → S3E → planner → executed plan) is the natural video.
- **If the tool paper is rejected:** use the reviews, submit JOSS immediately, submit the demo, and target the next ICAPS or a KEPS/PRL-style workshop with the strengthened user base.

---

## 9. Timeline

| By | Deliverable |
|---|---|
| Oct 31 | Decide long vs short, author list, models and domains; create the experiments repo and Dockerfile; set repo metadata, Zenodo, branch protection; obtain the AAAI-27 author kit. |
| Nov 7 | E1 pipeline running end to end on Blocksworld with one model; photo domain decided and licensed; feature matrix and related-tools section drafted. |
| Nov 21 | E1–E5 complete on the cluster for all models; E6/E7 run; design/techniques section drafted; `make smoke` passes in Docker on a clean machine. |
| Nov 28 | All figures and tables; reproducibility paragraph; limitations; internal read-through by all authors; arXiv blackout begins Nov 30. |
| Dec 7 | Abstract submitted on the submission site; anonymous mirror of the experiments repo and the supplementary ZIP frozen. |
| Dec 14 | Paper submitted; final check that linked content is unchanged. |
| Dec 15 – Jan 31 | Adoption work (Section 6); demo video; JOSS paper refreshed but not submitted. |
| Feb 1–4 | Author response: prepared answers on user base, scalability, novelty versus the 2025 papers, and reproducibility. |
| Feb 26 | Notification. Then: Zenodo release of the artifact, camera-ready with DOI links, JOSS submission, demo submission. |

---

## 10. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| "Not a planning paper" desk or soft reject | medium | E1 as the headline; PDDL/UP framing in the title, abstract, and first paragraph; Planning-and-Scheduling subject tags. |
| "Who uses it?" | high | Section 6 actions; honest, named users; emphasize maturity and reproducibility, which reviewers can verify themselves. |
| "Novelty versus the authors' 2025 workshop papers" | medium | Contribution statement is the library and its techniques; version comparison shows what did not exist before; workshop papers are cited as the method's origin. |
| "Just a wrapper around transformers" | medium | Techniques section with the failure modes each one removes (tokenizer variants, unknown answers, thinking blocks, vLLM tokenization drift). |
| Results on small models look bad | medium | Include mid-size and hosted models; present the tool as the instrument that measures and calibrates, not as a claim that VLMs are accurate. |
| Reproducibility nitpicks on VLM runs | low | Provenance JSON per run, pinned revisions, query dates, seeds, dispersion. |
| Time: 7 weeks | medium | Start E1 this week; E6/E7 are optional; short-paper fallback decided by Nov 28. |
| De-anonymization via the package | low | Accepted norm for tool papers; no lab name in the paper; optional email to the PC chairs. |

---

## 11. Optional library changes (only if the no-code-change preference is relaxed)

Nothing above requires edits under `s3e/`. Two small additive items would simplify the paper's story and could ship as 0.6.0, the version the paper pins:
- A helper that returns a copy of the Unified Planning problem with the estimated state as its initial values (today five lines of user code in the experiments repo).
- A `python -m s3e.benchmark` entry point wrapping `examples/blocksworld_benchmark.py`, so reviewers run the smoke test without cloning.

Do not add anything else for the paper's sake; reviewers reward a focused, stable tool over last-minute features.

---

## 12. Checklists

Submission day:
- [ ] Paper type "Tool paper", main track, Planning-and-Scheduling subject tags selected
- [ ] 8 pages + references, AAAI-27 kit, no names/affiliations, third-person self-citations, metadata scrubbed
- [ ] Reproducibility paragraph present in the main paper (hardware, limits, versions, model ids and revisions, decoding, prompts, dates, seeds, dispersion)
- [ ] Availability statement with anonymized link; supplementary ZIP ≤ 100 MB uploaded
- [ ] Anonymous mirror frozen; no further pushes until Feb 26

Artifact:
- [ ] Docker build and `make smoke` pass on a clean Ubuntu 22.04 machine; `pip` path tested on macOS
- [ ] Claim → script → output mapping complete; raw results shipped; figures regenerate without a GPU
- [ ] Resource estimates per experiment stated

Repository (public `s3e`):
- [ ] Description, topics, homepage set; Zenodo enabled; branch protection on; Discussions on
- [ ] No manuscript, experiments, or ICAPS branch pushed during review

---

## Sources

- ICAPS 2027: home https://icaps27.icaps-conference.org/ · news https://icaps27.icaps-conference.org/news · dates https://icaps27.icaps-conference.org/dates/ · call for papers https://icaps27.icaps-conference.org/calls/cfp · special track https://icaps27.icaps-conference.org/calls/special_tracks
- ICAPS 2025: Call for Tools Papers https://icaps25.icaps-conference.org/calls/tools_papers_call/ · Conditions for Acceptance https://icaps25.icaps-conference.org/organisation/conditions_for_acceptance · Paper Classification System https://icaps25.icaps-conference.org/organisation/paper_classification_system/ · accepted papers https://icaps25.icaps-conference.org/program/accepted/
- ICAPS 2026: CFP https://icaps26.icaps-conference.org/calls/cfp/ · demos https://icaps26.icaps-conference.org/calls/demos/ · reviewing https://icaps26.icaps-conference.org/organisation/reviewing/ · accepted papers https://icaps26.icaps-conference.org/program/accepted/
- Planner Museum (ICAPS 2026) https://ojs.aaai.org/index.php/ICAPS/article/view/42852
- TACAS tool-paper guidelines https://etaps.org/2013/tacas/tacas13-tool-papers-menu · TACAS 2025 artifact evaluation https://tacas.info/artifacts-25.php · PEPM tool paper advice https://conf.researchr.org/attending/pepm2015/ToolPaperAdvice
- JOSS submission policy https://joss.readthedocs.io/en/latest/submitting.html
- ViPlan (cites S3E) https://arxiv.org/html/2505.13180
- Repository metrics: GitHub API for CLAIR-LAB-TECHNION/s3e, pypistats.org for `s3e`, s3e.readthedocs.io, all on 2026-10-25
