# Artifact: experiments for the S3E tool paper

Experiments for "S3E: A Library for Probabilistic Symbolic State Estimation
from Images for PDDL Planning" (ICAPS 2027 tool paper). The library under
evaluation is the released `s3e==0.5.0` from PyPI; nothing here modifies it.
This directory is self-contained: it has its own `pyproject.toml`, lock file,
tests, and Makefile, and can be split into its own repository unchanged.

## Abstract

The artifact contains the code that produces every table and figure of the
evaluation (E1-E7), the scene generator with its seeds, adapters for three
ViPlan++ environments, the prompts and answer tokens, the 2025 workshop
paper's results transcribed for comparison, and a Docker image for a CPU
quick check. It supports these claims:

| Claim | Experiment | Script | Output |
|---|---|---|---|
| Planning success from S3E states, next to predicate accuracy; probabilities enable consistent-state decoding and calibration that change planning outcomes | E1 | `experiments/e1_closed_loop.py plan` | `e1_planning.{md,pdf}` |
| Closed-loop execution (estimate, plan, act, re-observe) succeeds, and execution monitoring on a few atoms costs fewer queries | E1 | `experiments/e1_closed_loop.py online` | `e1_online.md` |
| Masking cuts queried atoms; time per scene and throughput vs objects, batch size, and backend (HF vs vLLM) | E2 | `experiments/e2_scale.py` | `e2_atoms`, `e2_time` |
| Token probabilities beat parsing text (accuracy, calibration, no-answer rate, time) | E3 | `experiments/e3_scoring.py` | `e3_scoring.md` |
| Offline Platt calibration (scopes, transfer across problems) and multi-view averaging, without re-querying | E4 | `experiments/e4_calibration.py` | `e4_scopes`, `e4_transfer`, `e4_views`, `e4_reliability` |
| Logprob scoring of reasoning models needs thinking handling | E5 | `experiments/e5_reasoning.py` | `e5_reasoning.md` |
| Comparison with a hand-written scorer and t2v_metrics; feature matrix of related tools | E6 | `experiments/e6_tools.py` | `e6_tools.md`, `feature_matrix.md` |
| Earlier versions (0.2.0 vs 0.5.0): features, API, throughput; the 2025 setting with the current library | E7 | `experiments/e7_versions.py` | `e7_*.md` |

Outputs go to `results/figures/` (`make figures`, from `results/raw/`);
`results/figures/MANIFEST.md` lists each file with its inputs.

## License

MIT. Program committee members may download, run, and evaluate the artifact.
The prompts in `prompts/viplan/` are from ViPlan++ (MIT).

## Layout

```
Makefile            env | smoke | instances | collect | e1 ... e8 | vev | figures | reproduce
Dockerfile          Ubuntu 22.04, CPU torch, the lock's exact versions
pyproject.toml      s3e==0.5.0 (PyPI), unified-planning, up-fast-downward, ...; uv.lock
configs/            models.json (ids + pinned Hub revisions), feature_matrix.json
prompts/viplan/     ViPlan++ question templates and per-domain instructions
s3e_exp/            the harness
  domains/          blocksworld.py (generator, renderer, simulator, MAP decoder), viplan.py
  estimation.py     run S3E, store raw predictions (resumable JSONL + provenance)
  analysis.py       offline: rows, cross-fitted calibration, decoders, regex baseline
  masking.py        none / dynamic / relevant atom masks
  planning.py       plan from an estimate with Fast Downward via UP, replay in the truth
  closed_loop.py    open-loop / replan / monitor policies
  metrics.py, provenance.py, store.py, models.py (registry, timing, fake model)
  workers/          scripts that run outside this environment (other S3E versions,
                    t2v_metrics, the hand-written scorer)
experiments/        one script per stage (instances, collect, e1 ... e8, figures)
results/external/   llm4plan2025.json: the 2025 workshop paper's tables
slurm/run.sbatch    run a Makefile target on a GPU node
tests/              harness tests with the fake model (CPU, ~2 min)
```

## Requirements

- Python 3.12 via [uv](https://docs.astral.sh/uv/) (`make env`), or Docker.
- Quick check: CPU, 8 GB RAM. With `SMOKE_MODEL=fake` it needs no download.
- Full replication: one CUDA GPU with 48 GB (the paper used an NVIDIA L40S);
  Qwen3-VL-8B runs in float32 on the HuggingFace backend (see "Notes").
  ViPlan++ adapters additionally need its environment (below).

## Quick check (10 min)

```bash
make env            # .venv with s3e==0.5.0 from PyPI
make test           # harness tests with the fake model
make smoke-fake     # the E1-E4 pipeline end to end, no model download (1 min)
make smoke          # the same with SmolVLM-256M on two tiny scenes (GPU: 2 min; CPU: 10-40 min)
```

With Docker: `docker build -t s3e-artifact . && docker run --rm s3e-artifact`
(runs `make smoke` on CPU; `docker run --rm s3e-artifact make smoke-fake` for
the 1-minute version).

## Full replication

Every stage resumes after an interruption (records are appended per scene),
so Slurm `--requeue` is safe: `sbatch -p <partition> --export=ALL,TARGET=collect slurm/run.sbatch`.

| Step | Command | Hardware | Notes |
|---|---|---|---|
| Instances | `make instances` | CPU, minutes | 1,500 rendered Blocksworld scenes (3-7 blocks x 3 seeds x 100), 4 views each |
| ViPlan++ frames | `make instances-viplan VIPLAN_GW=... VIPLAN_PRB=... VIPLAN_ALF=...` | CPU | exports recorded episodes; see below |
| Predictions | `make collect MODELS="..."` | GPU | logprobs on 4 single views + all views in one prompt, text mode (16 tokens), dynamic mask |
| E1 offline | `make e1` | CPU (4 workers) | Fast Downward on every estimate; oracle and flipped-atom references |
| E1 online | `make e1-online` | GPU | 20 episodes per model and policy (4 per block count) |
| E2 | `make e2` (+ `--backend vllm` runs after `make env-vllm`) | GPU | atom counts are CPU-only |
| E3, E4 | `make e3 e4` | CPU | offline from the collected predictions |
| E5 | `make e5` | GPU | builds the thinking overlay (S3E main, commit `2a09b7e`) |
| E6 | `make env-t2v && make e6` | GPU | the t2v_metrics environment is separate (pinned commit); all scorers unbatched, bfloat16, no system prompt (t2v_metrics' own setting) |
| E7 | `make e7` | GPU | overlays for S3E 0.2.0 and main |
| Vev | `make vev` | GPU, then CPU | Vev-4B with S3E defaults and its native prompt, and its base model |
| E8 | `make e8` | GPU, then CPU | categorical answers; Vev-4B needs the S3E main overlay |
| Figures | `make figures` | CPU | from `results/raw/` only |

`make reproduce` runs them all in order.

### Measured costs (one NVIDIA L40S, 2026-10-26)

| What | Throughput |
|---|---|
| Qwen3-VL-8B, HuggingFace, float32, batch 16, one view | 9.5 / 6.7 / 4.0 queries/s at 3 / 5 / 7 blocks |
| same, four views in one prompt | 1.6 queries/s |
| Qwen3-VL-8B, vLLM, top-20 logprobs, batch 64 | 170-210 queries/s |
| Qwen3-VL-8B, vLLM, full-vocabulary logprobs (S3E 0.5.0's default) | ~5 queries/s |
| LLaVA-OneVision-0.5B, HF float32, GroceriesWorld 640x480 frames | ~4 queries/s |
| Fast Downward per Blocksworld problem (E1) | ~0.3 s |

S3E's own overhead (grounding, translation, scoring) is under 0.5% of the
wall clock. The full Blocksworld collection is about 250,000 single-image
and 21,500 four-image queries per model: about 15 GPU-hours on the
HuggingFace backend, under 2 on vLLM (`make collect BACKEND=vllm`). vLLM
top-20 probabilities matched HuggingFace float32 on the pilot (mean |dP|
0.010, 98.9% same decisions over 1,048 atoms), so `configs/models.json`
uses top-20 for vLLM; E2 also times the full-vocabulary mode.

### Models

`configs/models.json` pins every model to a Hub commit. Defaults:
SmolVLM-256M (small), Qwen3-VL-4B and -8B, Gemma-4-E4B (mid-size open),
Qwen3.5-0.8B / -4B and Ministral-3-3B-Reasoning (E5), GPT-4.1-mini through the
OpenAI backend (needs `OPENAI_API_KEY`; billed, so not in the default `MODELS`).
Any Hugging Face id also works (`--model org/name`, at the Hub's current
revision, which the provenance records).

### Domains

- **Rendered Blocksworld** (`s3e_exp/domains/blocksworld.py`): colored blocks,
  a gripper that may hold one, four camera views (front, back, wide,
  degraded); exact ground truth for every grounded atom; any state can be
  rendered, so it also serves as the simulator for closed-loop runs. The front
  view follows the S3E repository's `examples/blocksworld_benchmark.py`.
- **ViPlan++** (`s3e_exp/domains/viplan.py`): GroceriesWorld (`viplan-gw`,
  MuJoCo, 25 atoms, the setting of the 2025 workshop paper's experiment 1),
  photorealistic Blocksworld (`viplan-prb`, Blender), and ALFWorld
  (`viplan-alf`, AI2-THOR, 1,000-2,600 atoms).
  - *Recorded episodes* (no ViPlan++ install needed): `python experiments/instances.py viplan-gw --export <data>/episodes/gw --out instances/viplan-gw`
    reads `<cell>/episodes/episode_*/{episode.json,domain.pddl,problem.pddl,observations/*.npz}`;
    one scene per frame, the problem's `:init` replaced by that frame's true state.
  - *Live environments* (fresh procedurally generated problems, multi-camera
    GW views, and closed loop): run under ViPlan++'s own environment with this
    package and S3E 0.5.0 on `PYTHONPATH`:
    ```bash
    make overlay-s3e REF=0.5.0
    export MUJOCO_GL=egl          # GW; also put ViPlan++'s libOpenGL on LD_LIBRARY_PATH
    PYTHONPATH=envs/s3e-0.5.0:. $VIPLAN_ROOT/.venv/bin/python experiments/instances.py \
        viplan-gw --sizes 2 3 --seeds 0 1 2 --count 20 --options '{"random_steps": 3}' \
        --out instances/viplan-gw-live
    PYTHONPATH=envs/s3e-0.5.0:. $VIPLAN_ROOT/.venv/bin/python experiments/e1_closed_loop.py \
        online --instances instances/viplan-gw-live --model qwen3-vl-8b --policy monitor
    ```
    `size` is the goal horizon for GW, the number of shapes for PRB (Blender
    on `PATH`, ~20 s per render on 4 CPU cores), and ignored for ALF (tasks in
    order; needs `ALFWORLD_DATA` and Xvfb). Noise injection is off; GW uses its
    deterministic closed-form executor.

### S3E overlays (other versions without touching the installed one)

`make overlay-s3e REF=<version or commit>` installs that S3E into
`envs/s3e-<REF>` with `--no-deps`; putting the directory first on
`PYTHONPATH` selects it, so torch and transformers stay the same. E7 runs
0.2.0 this way; E5 runs S3E main (`2a09b7e`), because thinking-model support
landed after 0.5.0.

## Regenerate figures from shipped raw results (no GPU)

```bash
make figures RESULTS=results/raw
```

## What each run records

`run.json` next to every predictions file: start/finish and query dates
(UTC), command line, package versions (s3e, torch, transformers, vLLM,
unified-planning, up-fast-downward, ...), the S3E source path and version,
hardware (CPU model, GPUs, CUDA), the model id with the pinned and the
resolved revision, backend class and kwargs, the system prompt, prompt
template, every query, the answer space with its token variants, decoding
settings, and peak memory. Host and user names are not recorded.

Randomness: scene sets use seeds 0, 1, 2; tables report mean and standard
deviation across seeds (and across view subsets or calibration folds where
stated). Model scoring is deterministic (logprobs, greedy text).

## Notes and limitations

- **Batched HuggingFace inference.** S3E 0.5.0 checks once per backend that
  a padded batch reproduces unbatched answers, and refuses to batch
  otherwise. Qwen3-VL-8B in bfloat16 failed that check here (max P
  difference 0.185), so HF runs of Qwen3-VL use float32 (`configs/models.json`);
  vLLM does not pad.
- **vLLM sampling.** The harness sets `VLLM_USE_FLASHINFER_SAMPLER=0`:
  vLLM's FlashInfer sampler JIT-compiles CUDA code (needs `ninja` and
  `nvcc`), and S3E reads logprobs, so the PyTorch sampler is equivalent.
- **Unanswered atoms** (no mass on any answer token, or a text reply that
  matches none) are False when decoding states, for every method
  (logprobs, text, regex, other tools), and metrics also report accuracy on
  answered atoms. S3E itself gives them P(true) = 0.5, which a 0.5
  threshold would turn into True.
- **Resuming** is refused when a run directory's configuration differs
  (model, backend, kwargs, scoring, prompts, mask, ...); add views or
  scenes freely, and use `--tag` for a new configuration.
- **Not included.** ViPlan (Merler et al.) ships no pre-rendered images
  (Blocksworld renders through Blender 3.0; household needs licensed iGibson
  assets), so it is compared in the feature matrix and related work only.
  LMMs-Eval's image log-likelihood path is unavailable for the Qwen-VL and
  Gemma wrappers and scores mean cross-entropy over the chat-templated
  continuation, so it cannot score the same answer token; it is in the
  feature matrix only.
- **2025 workshop numbers** (`results/external/llm4plan2025.json`) were
  transcribed from the PDF (LLaVA-OneVision 0.5B/7B/72B, LLM translation with
  LLaMA 3 70B); the paper does not give exact model revisions, so E7's
  workshop table compares settings, not identical runs.
