#!/bin/bash

# processing power
#SBATCH --partition=g24
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --constraint=ampere

# output files
#SBATCH --output=dprocess-7B_%A_%a.out
#SBATCH --error=dprocess-7B_%A_%a.err

python -u prb_process_datapoints.py --out_dir="7B" --vqa_model_id="lmms-lab/llava-onevision-qwen2-7b-ov"
