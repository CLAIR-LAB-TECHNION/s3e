# Artifact for "S3E: A Library for Probabilistic Symbolic State Estimation from Images for PDDL Planning"

<!-- Template for the README of the experiments/artifact repository. It follows the
     ICAPS 2025 tools-track artifact requirements, which reviewers still expect:
     abstract, license, link, install script, quick functional check, replication
     instructions with resource estimates, broad platform support. Fill in every
     <placeholder>; delete this comment. -->

## Abstract

This artifact contains the experiment scripts, generated instances, prompts, raw per-instance
results, fitted calibrators, and figure-generation code behind every table and figure of the
paper. It pins the S3E library at version <x.y.z> from PyPI. Claims supported: Tables <..>,
Figures <..>.

## License

MIT (see `LICENSE`). Program-committee members may evaluate the artifact without restriction.

## Artifact location

<anonymized URL at submission; Zenodo DOI at camera-ready>. The content is frozen as of
<date>.

## Requirements

- Quick check: Docker (or Python 3.10+ with `pip`), 8 GB RAM, no GPU, about 10 minutes.
- Full replication: one CUDA GPU with <N> GB memory for the mid-size models, an OpenAI API key
  for the hosted-model rows (optional; shipped raw results cover them), about <H> GPU-hours.

## Quick functional check (about 10 minutes, CPU)

```bash
docker build -t s3e-artifact .
docker run --rm s3e-artifact make smoke
```

`make smoke` runs the smallest model (`HuggingFaceTB/SmolVLM-256M-Instruct`) on three rendered
Blocksworld scenes, estimates the state, plans with Unified Planning and Fast Downward from
the estimate, validates the plan against the ground-truth state, and prints predicate
accuracy, plan validity, and timing. Expected output: <paste>.

Without Docker:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.lock
make smoke
```

## Replicating the paper

| Claim | Command | Hardware used in the paper | Runtime |
|---|---|---|---|
| Table 2, Fig. 2 (closed-loop planning) | `make e1` | <GPU> | <h> |
| Fig. 3 (scaling) | `make e2` | <GPU> | <h> |
| Table 3 (probabilities vs parsing) | `make e3` | <GPU> | <h> |
| Table 4 (calibration and views) | `make e4` | CPU (offline fit) | <min> |
| Table 5 (reasoning models) | `make e5` | <GPU> | <h> |
| Table 1 rows run head-to-head (other tools) | `make e6` | <GPU> | <h> |
| Table 6 (earlier versions) | `make e7` | <GPU> | <h> |

Every run writes a JSON file with full provenance: library versions, model identifier and
resolved revision, prompts, answer tokens, decoding parameters, seeds, dates, and hardware.

## Regenerating figures and tables from the shipped raw results (no GPU)

```bash
make figures
```

## Reproducibility details

- Models, revisions, decoding parameters, prompts, and query dates: `provenance/`.
- Seeds: <list>; results are reported as mean and standard deviation over <n> seeds.
- Per-instance time and memory limits: <values>.
- Instance generation: `instances/README.md` (generator, seeds, selection rule).

## Layout

```
Dockerfile  Makefile  requirements.lock  LICENSE
experiments/   e1_closed_loop.py ... e7_versions.py
instances/     generated scenes and seeds; photo subset with labels and license
prompts/       every prompt and answer-token set
results/raw/   per-instance JSON from the paper runs
results/figures/
provenance/
```
