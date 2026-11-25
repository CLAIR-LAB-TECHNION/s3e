#!/bin/bash
# Precision-matched BF16 HF vs vLLM timing & agreement experiment on S3E 0.6.1.
# Submit from inside a Slurm session with:
#   env $(env | grep -o '^SLURM_[A-Z_]*' | sed 's/^/-u /') \
#     sbatch -A clair -p clair --gres=gpu:1 --gres-flags=disable-binding \
#            --cpus-per-task=8 --mem=64G --time=01:30:00 \
#            -D /home/guy.azran/workspace/s3e/icaps27 slurm/run_e2_bf16.sbatch
#SBATCH --job-name=s3e-e2-bf16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/home/guy.azran/workspace/s3e/icaps27/results/raw/e2-0.6.1/slurm-%j.out

set -euo pipefail
cd /home/guy.azran/workspace/s3e/icaps27

export PYTHONPATH="$PWD/envs/s3e-0.6.1"
export HF_HUB_CACHE="/home/guy.azran/icaps27-runs/hf-cache"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

OUT="results/raw/e2-0.6.1"
mkdir -p "$OUT/predictions"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "host: $(hostname -s) | gpu: $GPU_NAME | date: $(date -u -Iseconds)"
if [[ "$GPU_NAME" != "NVIDIA L40S" ]]; then
  echo "ERROR: Expected NVIDIA L40S, got '$GPU_NAME'" >&2
  exit 1
fi

PY=".venv/bin/python"
"$PY" -B -c '
import importlib.metadata as m, s3e, torch
print("s3e:", s3e.__file__, s3e.__version__, m.version("s3e"))
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
assert "envs/s3e-0.6.1" in s3e.__file__
assert s3e.__version__ == "0.6.1" and m.version("s3e") == "0.6.1"
'

echo "=== Stage 1: HF bfloat16 timing (batch 16, 32, 64, 128; mask none) ==="
"$PY" experiments/e2_scale.py time \
  --instances instances/blocksworld --seeds 0 --limit 10 \
  --model qwen3-vl-8b --backend hf \
  --vlm-kwargs '{"torch_dtype": "bfloat16"}' \
  --batch-sizes 16 32 64 128 --masks none \
  --out "$OUT/time-hf-bf16.jsonl"

echo "=== Stage 2: vLLM bfloat16 top-20 timing (batch 16, 32, 64, 128; mask none) ==="
"$PY" experiments/e2_scale.py time \
  --instances instances/blocksworld --seeds 0 --limit 10 \
  --model qwen3-vl-8b --backend vllm \
  --vlm-kwargs '{"dtype": "bfloat16", "num_logprobs": 20}' \
  --batch-sizes 16 32 64 128 --masks none \
  --out "$OUT/time-vllm-bf16-top20.jsonl"

echo "=== Stage 3: HF float32 anchor timing (batch 16; mask none) ==="
"$PY" experiments/e2_scale.py time \
  --instances instances/blocksworld --seeds 0 --limit 10 \
  --model qwen3-vl-8b --backend hf \
  --vlm-kwargs '{"torch_dtype": "float32"}' \
  --batch-sizes 16 --masks none \
  --out "$OUT/time-hf-fp32.jsonl"

echo "=== Stage 4: Prediction quality & agreement (same 50 scenes) ==="
"$PY" experiments/collect.py \
  --instances instances/blocksworld --seeds 0 --limit 10 \
  --model qwen3-vl-8b --backend hf \
  --vlm-kwargs '{"torch_dtype": "bfloat16"}' \
  --batch-size 1 --tag bf16-b1 --out "$OUT/predictions"

"$PY" experiments/collect.py \
  --instances instances/blocksworld --seeds 0 --limit 10 \
  --model qwen3-vl-8b --backend hf \
  --vlm-kwargs '{"torch_dtype": "bfloat16"}' \
  --batch-size 16 --tag bf16-b16 --out "$OUT/predictions"

"$PY" experiments/collect.py \
  --instances instances/blocksworld --seeds 0 --limit 10 \
  --model qwen3-vl-8b --backend hf \
  --vlm-kwargs '{"torch_dtype": "bfloat16"}' \
  --batch-size 128 --tag bf16-b128 --out "$OUT/predictions"

"$PY" experiments/collect.py \
  --instances instances/blocksworld --seeds 0 --limit 10 \
  --model qwen3-vl-8b --backend vllm \
  --vlm-kwargs '{"dtype": "bfloat16", "num_logprobs": 20}' \
  --batch-size 128 --tag vllm-bf16-top20-b128 --out "$OUT/predictions"

echo "=== Stage 5: Timing summary ==="
"$PY" experiments/e2_scale.py summary \
  "$OUT/time-hf-bf16.jsonl" \
  "$OUT/time-vllm-bf16-top20.jsonl" \
  "$OUT/time-hf-fp32.jsonl" \
  --json "$OUT/timing_summary.json"

echo "=== Stage 6: Pairwise prediction agreement & accuracy vs truth ==="
"$PY" - "$OUT" <<'PYEOF'
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from s3e_exp import analysis, metrics
from s3e_exp.estimation import load_predictions

out_dir = Path(sys.argv[1])
truth = analysis.load_truth(Path("instances/blocksworld"))

runs = {
    "hf-fp32-b16 (stored)": Path("results/raw/predictions/blocksworld/qwen3-vl-8b/logprobs-none"),
    "hf-bf16-b1": out_dir / "predictions/blocksworld/qwen3-vl-8b/logprobs-none-bf16-b1",
    "hf-bf16-b16": out_dir / "predictions/blocksworld/qwen3-vl-8b/logprobs-none-bf16-b16",
    "hf-bf16-b128": out_dir / "predictions/blocksworld/qwen3-vl-8b/logprobs-none-bf16-b128",
    "vllm-bf16-top20-b128": out_dir / "predictions/blocksworld/qwen3-vl-8b/logprobs-none-vllm-bf16-top20-b128-vllm",
}

# Restrict stored hf-fp32-b16 to the exact 50 scenes in bf16-b16.
ref_records = [r for r in load_predictions(runs["hf-bf16-b16"]) if r["view"] == "0"]
target_scenes = {r["scene"] for r in ref_records}
assert len(target_scenes) == 50, f"expected 50 scenes, got {len(target_scenes)}"

loaded = {}
summary = {"scenes": len(target_scenes), "per_run": {}, "pairwise": {}}
for name, path in runs.items():
    recs = [r for r in load_predictions(path) if r["view"] == "0" and r["scene"] in target_scenes]
    recs.sort(key=lambda r: r["scene"])
    assert len(recs) == 50, f"{name}: expected 50 scenes, got {len(recs)}"
    loaded[name] = {(r["scene"], atom): pred for r in recs for atom, pred in r["results"].items()}
    rows = [row for r in recs for row in analysis.rows(r, truth[r["scene"]])]
    summary["per_run"][name] = {**metrics.atom_metrics(rows), **metrics.scene_metrics(rows)}

pairs = [
    ("hf-bf16-b16", "hf-bf16-b1"),
    ("hf-bf16-b128", "hf-bf16-b1"),
    ("hf-bf16-b16", "hf-fp32-b16 (stored)"),
    ("hf-bf16-b128", "hf-fp32-b16 (stored)"),
    ("vllm-bf16-top20-b128", "hf-bf16-b16"),
    ("vllm-bf16-top20-b128", "hf-bf16-b128"),
    ("vllm-bf16-top20-b128", "hf-fp32-b16 (stored)"),
]
for a, b in pairs:
    keys = sorted(loaded[a].keys())
    assert keys == sorted(loaded[b].keys())
    diffs = [abs(loaded[a][k].probability - loaded[b][k].probability) for k in keys]
    agree = sum(
        (loaded[a][k].has_answer and loaded[a][k].probability >= 0.5)
        == (loaded[b][k].has_answer and loaded[b][k].probability >= 0.5)
        for k in keys
    ) / len(keys)
    summary["pairwise"][f"{a} vs {b}"] = {
        "atoms": len(keys),
        "mean_abs_dp": sum(diffs) / len(diffs),
        "max_abs_dp": max(diffs),
        "p95_abs_dp": sorted(diffs)[int(0.95 * len(diffs))],
        "decision_agreement": agree,
    }

(out_dir / "agreement_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PYEOF
echo "DONE $(date -u -Iseconds)"
