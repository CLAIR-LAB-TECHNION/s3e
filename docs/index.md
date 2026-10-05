# s3e: Semantic Symbolic State Estimation

`s3e` estimates the truth values of grounded PDDL predicates from images using
vision-language models (VLMs). Given a planning domain, a problem, and one or
more images of a scene, it returns a probability (optionally calibrated) for every
grounded predicate (e.g. `on(a,b)`) and a symbolic state that can be handed
back to a planner.

It is built as independently usable layers:

1. **Backends** ({doc}`s3e.backends <api/backends>`) — a uniform {class}`~s3e.VLMBackend`
   interface over HuggingFace, OpenAI, and vLLM models.
2. **Engine** ({doc}`s3e.engine <api/engine>`) — {class}`~s3e.QueryEngine`: images +
   free-form queries + an answer space → predictions. No PDDL involved.
3. **Calibration** ({doc}`s3e.calibration <api/calibration>`) — fit a Platt-scaling calibrator
   on labeled examples, offline and VLM-free after data collection.
4. **PDDL facade** ({doc}`s3e.estimator <api/estimator>`) —
   {class}`~s3e.SemanticStateEstimator`: grounds a PDDL domain/problem into
   predicates, translates them into queries, and drives a `QueryEngine`.

```{toctree}
:maxdepth: 2

getting-started
api/index
contributing
changelog
```

## Links

- Source code and issue tracker: <https://github.com/CLAIR-LAB-TECHNION/s3e>
- Package: <https://pypi.org/project/s3e/>
- Walkthrough notebook (covers every layer):
  [`docs/s3e_walkthrough.ipynb`](https://github.com/CLAIR-LAB-TECHNION/s3e/blob/main/docs/s3e_walkthrough.ipynb)
