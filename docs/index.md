# S3E: Semantic Symbolic State Estimation

S3E estimates the truth values of grounded PDDL predicates from images using
vision-language models (VLMs). Given a planning domain, a problem, and one or
more images of a scene, it returns a probability (optionally calibrated) for every
grounded predicate (e.g. `on(a,b)`) and a symbolic state that can be handed
back to a planner.

## Statement of need

Research that grounds symbolic planners in perception (task planning, task-and-motion planning, embodied AI, neuro-symbolic reasoning) keeps re-implementing the same pipeline: enumerate grounded predicates, phrase a question for each, prompt a particular model API with images, and turn the output into truth values. Parsing generated text throws away the model's uncertainty, and hard-wiring one provider makes cross-model comparisons and calibration studies laborious. S3E packages this pipeline as a backend-agnostic library with probabilities as a first-class output: answer-token probability scoring, an optional explicit "unknown" answer, multi-view averaging, and offline calibration, over HuggingFace, vLLM, and OpenAI models. Its query engine also works without PDDL, for anyone who needs probabilistic yes/no or multiple-choice answers from a VLM.

## Design

S3E is built as independently usable layers:

1. **Backends** ({doc}`s3e.backends <api/backends>`): a uniform {class}`~s3e.VLMBackend`
   interface over HuggingFace, OpenAI, and vLLM models.
2. **Engine** ({doc}`s3e.engine <api/engine>`): {class}`~s3e.QueryEngine` turns images +
   free-form queries + an answer space into predictions. No PDDL involved.
3. **Calibration** ({doc}`s3e.calibration <api/calibration>`): fit a Platt-scaling calibrator
   on labeled examples, offline and VLM-free after data collection.
4. **State-estimation facade** ({doc}`s3e.estimator <api/estimator>`):
   {class}`~s3e.SemanticStateEstimator` takes grounded predicates (an explicit
   list, or grounded from a PDDL domain/problem by `from_pddl`), translates
   them into queries, and drives a `QueryEngine`.

The {doc}`developer guide <developer-guide>` explains how the layers fit
together and how to extend each one.

```{toctree}
:maxdepth: 2
:caption: Using S3E

getting-started
Tutorial <s3e_walkthrough>
user-guide
examples
api/index
```

```{toctree}
:maxdepth: 2
:caption: Project

developer-guide
related-software
citing
contributing
changelog
```

## Links

- Source code and issue tracker: <https://github.com/CLAIR-LAB-TECHNION/s3e>
- Package: <https://pypi.org/project/s3e/>
- Run the tutorial notebook in Google Colab:
  <https://colab.research.google.com/github/CLAIR-LAB-TECHNION/s3e/blob/main/docs/s3e_walkthrough.ipynb>
