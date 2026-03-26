# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**s3e** (Semantic State Estimator) uses vision-language models (LLaVA, CLIP, OpenAI GPT-4o/4.1) to estimate environment state from images, outputting PDDL-compatible logical predicates for task and motion planning. Developed at CLAIR Lab, Technion.

## Commands

```bash
# Install (editable mode)
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Flash attention (optional, requires CUDA)
conda install nvidia cuda-toolkit=11.8
conda install -c conda-forge ninja cmake llvm-openmp
pip install flash-attn
```

## Architecture

### Class Hierarchy

```
StateEstimator (ABC)                       # s3e/state_estimator.py
├── ProbabilisticStateEstimator (ABC)      #   Adds confidence thresholding
│   └── SemanticStateEstimator             # s3e/semantic_state_estimator.py
│       ├── SemanticEstimatorMultiImageRun  #   Averages predictions across images
│       ├── SemanticEstimatorWithCLIP       #   Uses CLIP for vision-language matching
│       ├── SemanticStateEstimatorDetProbs  #   Binary yes/no from model response text
│       │   └── ...DetProbsMultiImageRun    #   Multi-image variant
│       └── ...Conditional                  #   Evaluates only conditional predicates
└── PredFnStateEstimator (ABC)             #   Maps predicates to Python methods
```

### Core Flow

1. A `StateEstimator` receives a PDDL domain + problem and a set of images
2. PDDL predicates are optionally converted to natural language via `PDDL2NLQueryConverter` (using LLaMA or OpenAI)
3. A VQA model (LLaVA, CLIP, or OpenAI) evaluates each predicate against the images
4. Token logit probabilities (true/false or yes/no) are extracted and thresholded to produce boolean state

### VQA Model Backends (`s3e/utils/`)

- **LLaVA legacy** (`llava_utils.py`) - requires `transformers==4.37.2`
- **LLaVA-OneVision** (`llava_next_utils.py`) - requires `transformers==4.40.0.dev0`
- **OpenAI** (`open_ai_utils.py`) - API-based, model IDs prefixed with `"OpenAI/"`
- **CLIP** - imported directly from transformers (fallback when version doesn't match LLaVA)

The `transformers` version determines which LLaVA backend is loaded at import time (see `semantic_state_estimator.py` lines 38-46).

### Key Utilities

- `up_utils.py` - Unified Planning integration: PDDL parsing, problem creation, predicate grounding, state conversion
- `pddl2nl_query_converter.py` - Converts grounded PDDL predicates to natural language yes/no questions; caches results to `s3e/cache/nl-predicates/`
- `misc.py` - `load_se_from_args()` factory function instantiates state estimators from config dicts; `load_from_entrypoint()` for dynamic module loading

### External Dependencies

- **ICAPS-24** (`gymjoco` branch) and **clair-robotics-stack** - CLAIR Lab custom forks installed from git
- **unified-planning[fast-downward]** - PDDL planning framework

## Known Issues

- `pyproject.toml` has `include = ['semantic_state_estimator']` in `[tool.setuptools.packages.find]` but the package directory is `s3e/` - this may cause the package not to be found correctly after install
- `s3e/__init__.py` is empty - nothing is exported at the package level
- `tests/` directory exists but contains no test files
