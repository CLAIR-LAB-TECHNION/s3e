# Examples

The [`examples/`](https://github.com/CLAIR-LAB-TECHNION/s3e/tree/main/examples)
directory of the repository holds runnable scripts. The test suite runs both
of them with fake backends on every change.

## A custom backend, end to end, without downloads

`examples/custom_backend.py` implements a {class}`~s3e.VLMBackend`: a
simulated model that answers correctly 80% of the time but is always 95% sure.
It then runs the whole pipeline: PDDL grounding, calibration data collection,
Platt scaling, a held-out comparison of accuracy and Brier score before and
after calibration, and conversion to a Unified Planning state. It needs
`s3e[pddl,calibration]` and runs in seconds:

```bash
python examples/custom_backend.py
```

```text
15 grounded predicates, e.g. ['on(a,a)', 'on(a,b)', 'on(a,c)']
fitted Platt scaling on 600 samples

held-out scenes: 20 x 15 predicates
                accuracy     Brier
uncalibrated       0.770     0.209
calibrated         0.770     0.165
...
```

Calibration leaves accuracy unchanged (Platt scaling is monotone) and lowers
the Brier score, because the simulated model's 95% confidence overstates its
80% accuracy. To use a real model, pass a model id or backend instance as
`vlm=`; nothing else changes.

```{literalinclude} ../examples/custom_backend.py
:language: python
:start-at: class SimulatedVLM
:end-before: def score
```

## Evaluating a model on synthetic Blocksworld scenes

`examples/blocksworld_benchmark.py` renders random Blocksworld arrangements
(colored blocks on a table) with known ground truth and asks a model about
every grounded predicate. For each scoring mode (`logprobs` and `text_match`)
it reports accuracy, Brier score, expected calibration error, the share of
predictions with no answer, time per query, and peak memory (per mode on a
GPU; on CPU, the process's peak so far, model included). With
`--calibrate-fraction`, it also fits Platt scaling offline on that share of
the scenes and reports the remaining scenes before and after calibration:

```bash
pip install "s3e[pddl,hf,calibration]"
python examples/blocksworld_benchmark.py --model HuggingFaceTB/SmolVLM-256M-Instruct \
    --num-scenes 10 --calibrate-fraction 0.5 --output blocksworld-results.json
```

Run `python examples/blocksworld_benchmark.py --help` for every option. A GPU
is strongly recommended; on CPU even a 256M-parameter VLM can take tens of
seconds per query. The scenes are deliberately simple, so treat the script as
a template for evaluating your own models and domains, not as a published
benchmark.

## Reporting experiments

Results from foundation models are hard to reproduce unless the exact setup is
reported. The benchmark's JSON output records each item below (raw results as
per-predicate masses, the calibrator as its fitted parameters); when you write
your own experiment, this is where `s3e` exposes each of them:

| What to report | Where to find it |
|---|---|
| Library versions | `s3e.__version__`, `importlib.metadata.version(...)` for `torch`, `transformers`, `vllm`, `openai` |
| Model and exact revision | the model id; pin a Hugging Face commit with `HuggingFaceVLM(..., revision=...)` and read the loaded one from `backend.model.config._commit_hash` |
| Full prompts | `estimator.engine.system_prompt`, `estimator.engine.prompt_template`, and `estimator.queries` (predicate → question) |
| Answer tokens | `estimator.engine.answers.to_dict()` |
| Scoring and decoding settings | `estimator.engine.scoring` and `estimator.engine.inference_kwargs` |
| Calibration data and fit | `CalibrationSet.save(...)` and `PlattCalibrator.save(...)` (which records the scoring mode, answer space, and domain fingerprint) |
| Raw per-instance results | `PredictionSet.to_dict()` for each scene, reloadable with `PredictionSet.from_dict` |
| Dates and hardware | when the model was queried, and the GPU or CPU it ran on |
