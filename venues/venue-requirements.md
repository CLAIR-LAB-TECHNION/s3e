> **Status note (2026-10-26).** Recovered from the venue-readiness session of 2026-10-05
> (the file lived in that session's scratchpad, which no longer exists). The ICAPS section
> predates the verified 2027 call: item I4's "reviewers must consider supplementary material
> for tool papers" does **not** appear in the 2027 call, and 2027 has no artifact-evaluation
> step; the authoritative ICAPS 2027 rules are in `../icaps27-tool.md`, Section 1. The JOSS
> screening gates (J2) were re-verified on 2026-10-25.

# Venue requirements as currently understood (to be verified)

## JOSS (sources: joss.readthedocs.io submitting / review_criteria / paper / policies; joss-reviews .buffy reviewer_checklist.md)
J1. OSI license; plain-text LICENSE/COPYING file with full license text.
J2. Screening gates (desk-reject if any fails, added Mar 2026): (a) repo public >6 months before submission with active development spanning that period (commit distribution checked); (b) demonstrated research use ("at minimum by the developers themselves"; aspirational use insufficient); (c) open-source practices — single-author projects need multiple indicators: public commit history, tagged releases or changelog, tests and CI, clear docs, CONTRIBUTING, stated support or governance expectations; (d) iterative development, not a single burst.
J3. Out of scope: thin API clients, minor utilities, single-function packages, pre-trained models, notebooks.
J4. Repo: README with statement of need; pip-installable; example usage (ideally real-world); API docs ("Good": all functions documented incl. example inputs and outputs); automated tests hooked to CI; tagged versions or formal release process; community guidelines (contribute / report issues / seek support); issue tracker readable without registration.
J5. Paper: 750–1750 words; sections in order Summary, Statement of need, State of the field, Software design, Research impact statement, AI usage disclosure, Acknowledgements, References; YAML front matter (title, tags, authors w/ ORCID+affiliation, affiliations w/ optional ROR, date "%e %B %Y", bibliography); no API docs in paper; references with full venue names, DOIs where available; funding + sponsor-involvement statement.
J6. AI disclosure: tools/models AND versions, where used (code, paper, docs), nature/scope of assistance, assertion that authors reviewed/validated outputs and made core design decisions; if none used, say so.
J7. At acceptance: tagged release, archive (Zenodo/figshare) with DOI; archive title + author list (incl. ORCIDs) and license must match paper/software.
J8. Message to editors must disclose prior/planned publication elsewhere and conflicts of interest.

## SoftwareX (sources: official OSP template v6 Mar 2026 docx/tex from legacyfileshare.elsevier.com; reviewer form PDF 2024; Elsevier GenAI policy June 2026; guide-for-authors only via search snippets — sciencedirect blocked)
S1. Template mandatory; 5 required sections: Motivation and significance; Software description (architecture w/ pictorial overview, functionalities, optional code snippets); Illustrative examples (>=1); Impact (main section; uptake, publications); Conclusions.
S2. 4000-word limit (excl. metadata tables, refs), max 6 pages, max 6 figures, ~100-word abstract, <=6 keywords.
S3. Code metadata table C1–C8: current code version; permanent GitHub link to code for this version; legal code license; versioning system; languages/tools/services; compilation requirements/operating environments/dependencies; link to developer docs/manual; support email.
S4. GitHub repository mandatory (forked into ElsevierSoftwareX org on acceptance); repo must have well-documented README.md and "a Licence.txt file"; license from pre-approved list (MIT is listed).
S5. Reviewer form: runs as described; reviewer installs and reproduces results; good coding standards; proper OSS license; API docs; operating environment/dependencies complete; user manual/install guide; automated testing; portable to other platforms; test programs exercising all main features; readable/maintainable code; "licensing terms clearly stated in the source package and mentioned in each source file"; non-code artifacts quality.
S6. Declarations: competing interest, CRediT, data availability, funding, Elsevier GenAI declaration section before references.

## JMLR MLOSS (source: jmlr.org/mloss/mloss-info.html, author-info.html, format pages)
M1. Recognised OSI open-source license; all components comply; "licensing terms ... clearly stated in the source package and mentioned in each source file"; source package contains no extra files (VCS files, '._' files); code compiles and runs on all supported platforms.
M2. Paper: <=4 pages JMLR LaTeX style (jmlr2e) + references; abstract <=200 words; 5 keywords; running title <=50 chars.
M3. Cover letter: intended for MLOSS; license; project URL; version to be reviewed; evidence of active user community (stars, developers); suggested reviewers; plus general JMLR items (overlapping publications, co-author consent, COI, 3–5 action editors, 3–5 reviewers, keywords).
M4. Review criteria (ideal, not strict): quality of 4-page description; novelty/breadth/significance; active user community; openness (public repo, bug tracker, mailing list/forum); comparison to related implementations w.r.t. run-time, memory, features; user docs (installation, tutorials, non-trivial examples, full API docs, ease of viewing docs); developer docs enabling modification/extension with API reference; clarity of design; ease of contribution; "extensive unit and integration testing, and report of code coverage ... expected close to 100%"; CI ideally on all supported platforms and with multiple versions of dependencies; breadth of platforms incl. an open-source OS; freedom from proprietary software.
M5. Prior publication: method may be previously published; software must not have been published elsewhere; if software was main content of a prior peer-reviewed paper, repo must contain a document listing improvements/extensions. No simultaneous submission to other journals/conferences.

## ICAPS 2027 tool paper (sources: icaps27.icaps-conference.org/calls/cfp/ published 6 Sep 2026; icaps25 roles_and_guidelines & conditions_for_acceptance; eapls.org artifact badges)
I1. ICAPS-27 (Columbia SC, Jun 27–Jul 2 2027); abstracts Dec 7 2026, papers Dec 14 2026; "Tool papers" is a main-track category (no separate tools track).
I2. Long 8 pages + refs or short 4 pages + refs; AAAI format; double-blind; links in submission must be anonymized; camera-ready should use a persistent archive with a DOI.
I3. Reproducibility section of CfP: hardware/OS/limits; benchmark set; release/commit of every planner/solver/library; for foundation-model methods: exact model identifiers and versions, decoding parameters, full prompts, query dates; multiple runs with dispersion for randomized methods; code release strongly encouraged with availability statement.
I4. 2025 precedent (not restated in 2026/2027): tool papers submit an artefact; reviewers assess against EAPLS badges — Artifacts Available (archival DOI repo), Artifacts Evaluated Functional (documented, consistent, complete, exercisable) / Reusable (carefully documented, well-structured, adheres to norms), Results Validated. 2025/2026: reviewers must consider supplementary material for tool papers; supplementary material must be anonymised.
I5. CfP bars papers under review/accepted/published elsewhere.
