#!/bin/bash

# processing power
#SBATCH --partition=g48
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --constraint=ampere

# output files
#SBATCH --output=dprocess-72B_%A_%a.out
#SBATCH --error=dprocess-72B_%A_%a.err

python -u prb_process_datapoints.py --out_dir="72B" --vqa_model_id="lmms-lab/llava-onevision-qwen2-72b-ov"
