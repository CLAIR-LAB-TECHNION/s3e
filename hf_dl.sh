#!/bin/bash
#SBATCH --job-name=dl_llama
#SBATCH --output=dl_llama.out
#SBATCH --error=dl_llama.err

huggingface-cli download meta-llama/Meta-Llama-3-70B-Instruct --include "original/*"
