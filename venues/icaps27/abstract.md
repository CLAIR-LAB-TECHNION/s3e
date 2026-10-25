# Abstract submission (due Dec 7, 2026, 23:59 UTC-12)

**Working title.** S3E: A Library for Probabilistic Symbolic State Estimation from Images for
PDDL Planning

**Track / type / length.** Main track; paper type "Tool paper"; long (8 pages + references).

**Subject tags.** Select at least one Planning and Scheduling tag (papers without one were
auto-flagged for desk rejection in 2025); add Robotics and Uncertainty-in-AI tags as offered.

**Keywords.** symbolic state estimation; vision-language models; PDDL; grounding; calibration;
Unified Planning; open-source library

## Draft abstract (anonymized; refine after the experiments)

Planners that act from camera images need the symbolic state that PDDL assumes given.
Research systems that ground classical planners with vision-language models (VLMs)
re-implement the same pipeline: ground predicates over the problem's objects, phrase a
question per predicate, prompt a model with images, and turn its output into truth values.
Most parse generated text, which discards the model's uncertainty, and bind experiments to one
model API. We present S3E, an open-source Python library that turns a PDDL domain, a problem,
and one or more images into a probability for every grounded predicate and a
Unified-Planning-compatible state. S3E reads the probability mass of answer tokens over
configurable answer spaces with an explicit "unknown" option, across HuggingFace, vLLM, and
OpenAI models; keeps raw masses so that thresholds, calibration, and multi-view aggregation
can be changed offline; fits scoped Platt calibrators with provenance checks; and handles
reasoning models whose next token is a thought rather than an answer. We evaluate usefulness
by closing the loop with a planner on rendered and photographic scenes, measuring plan
validity alongside predicate accuracy, scaling with the number of objects, the cost and
benefit of token-probability scoring, calibration, and reasoning, and we compare with
existing tools and earlier versions. S3E is tested on Linux, macOS, and Windows, documented,
and released on PyPI with a reproducible artifact.

## Before submitting the abstract

- [ ] Author list and order fixed (authors cannot be added or removed after Dec 14).
- [ ] Each author is on at most 10 ICAPS-27 submissions across both tracks.
- [ ] Conflicts of interest entered on the submission site.
- [ ] Title and abstract contain no identifying information.
