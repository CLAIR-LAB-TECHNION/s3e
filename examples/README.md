# Examples

Runnable scripts that exercise `s3e` end to end. Run them from the repository
root; each script's docstring explains what it shows.

| Script | What it shows | Needs |
|---|---|---|
| [`custom_backend.py`](custom_backend.py) | Implementing a `VLMBackend` (a simulated, overconfident model), then the full pipeline: PDDL grounding, calibration data collection, Platt scaling, held-out accuracy and Brier score before/after calibration, and conversion to a Unified Planning state. Downloads nothing; runs in seconds. | `s3e[pddl,calibration]` |
| [`blocksworld_benchmark.py`](blocksworld_benchmark.py) | Evaluating a real model on rendered Blocksworld scenes with known ground truth: accuracy, Brier score, calibration error, unanswered-query rate, time per query, and peak memory for `logprobs` vs `text_match` scoring, optionally after offline Platt scaling. Writes per-predicate results plus provenance (versions, model revision, prompts, decoding settings, dates, hardware) to JSON. | `s3e[pddl,hf,calibration]`; a GPU is strongly recommended |

```bash
python examples/custom_backend.py
python examples/blocksworld_benchmark.py --model HuggingFaceTB/SmolVLM-256M-Instruct \
    --num-scenes 10 --calibrate-fraction 0.5 --output blocksworld-results.json
```

`blocksworld_benchmark.py --help` lists every option; pass `--model OpenAI/<model>`
for the OpenAI backend (set `OPENAI_API_KEY`) or `--vlm-kwargs '{"revision": "<commit>"}'`
to pin a Hugging Face model revision. The benchmark is a template for your own
evaluation, not a published result: the scenes are deliberately simple.

The [walkthrough notebook](../docs/s3e_walkthrough.ipynb) covers every layer of
the library step by step, and the test suite (`tests/test_examples.py`) runs
both scripts with a fake backend on every change.
