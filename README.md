# Semantic State Estimator


## Installation

core packages:
```bash
pip install -e .
```

flash attn:
- requires nvcc
```bash
conda install nvidia cuda-toolkit=11.8
```
- requires llvm?
```bash
conda install -c conda-forge ninja cmake llvm-openmp
```