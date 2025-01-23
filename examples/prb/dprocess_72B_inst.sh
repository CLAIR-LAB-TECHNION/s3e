#!/bin/bash

# processing power
#SBATCH --partition=g48
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --constraint=ampere

# output files
#SBATCH --output=dprocess-72B-inst_%A_%a.out
#SBATCH --error=dprocess-72B-inst_%A_%a.err

python -u prb_process_datapoints.py --out_dir="72B + Instruct" --vqa_model_id="lmms-lab/llava-onevision-qwen2-72b-ov" --additional_instructions="You will be asked questions about the state of blocks in a given image.\nA block can be a cube, cylinder, or sphere.\nA block is considered on the table if it is not on top of any other block.\nBlocks come in one of two materials, rubber and metal. Rubber blocks have a matte finish while metal objects are glossy and reflective."
