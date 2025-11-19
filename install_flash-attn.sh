#!/bin/bash
#SBATCH --job-name=flash_attn
#SBATCH --nodelist=plato1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --output=flash_attn_install.out
#SBATCH --error=flash_attn_install.err

export CXXFLAGS="${CXXFLAGS} -fopenmp"
export LDFLAGS="${LDFLAGS} -L${CONDA_PREFIX}/lib"
export CPPFLAGS="${CPPFLAGS} -I${CONDA_PREFIX}/include"

pip install flash-attn --no-build-isolation

