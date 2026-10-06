# Related software

S3E sits between symbolic-planning libraries, which assume the state is
given, and tools for running vision-language models, which know nothing about
planning problems. The closest tools are summarized below.

**Planning libraries.** [Unified Planning](https://github.com/aiplan4eu/unified-planning)
and [PDDLGym](https://github.com/tomsilver/pddlgym) model, ground, simulate, or
solve planning problems, but take the symbolic state as given. S3E builds on
Unified Planning instead of reimplementing it: PDDL parsing and grounding are
delegated to it, and estimated states convert back to Unified Planning state
objects.

**VLM evaluation harnesses.** [LMMs-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)
and [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) run models on
benchmarks and report aggregate metrics. LMMs-Eval can score answer
log-likelihoods, but only inside benchmark task definitions.

**Probability-of-"yes" scoring.** The closest tool is
[`t2v_metrics`](https://github.com/linzhiqiu/t2v_metrics), the package behind
VQAScore. It scores the probability of "Yes" for user-supplied images and
question templates across local and API models, and targets the evaluation of
text-to-visual generation. It has no predicate grounding, multi-option answer
spaces, explicit "unknown" answer, or calibration.

**Constrained generation.** Libraries such as
[Outlines](https://github.com/dottxt-ai/outlines) restrict what a model may
generate, but do not report per-option probabilities.

**Research systems.** Systems that ground planners with VLM questions, such as
TP-VQA, DKPROMPT, the VLM-as-grounder methods evaluated in ViPlan, and
predicate-learning approaches, implement the idea inside experiment code tied
to particular domains and benchmarks, rather than as an installable library.

None of these combines what S3E provides behind one interface: PDDL
grounding, pluggable predicate-to-question translation, token-probability
scoring over configurable answer spaces with an optional "unknown" option,
interchangeable local and hosted model backends, and offline calibration.

The [software paper](https://github.com/CLAIR-LAB-TECHNION/s3e/blob/main/paper/paper.md)
cites each of these works.
