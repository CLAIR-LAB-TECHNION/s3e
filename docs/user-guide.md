# User guide

How to configure S3E and read its results: the estimator and engine
arguments, thinking models, answer spaces, the decision rule behind each
prediction, calibration, the translators, and environment settings. For a
hands-on introduction, start with {doc}`getting-started` and the
{doc}`walkthrough notebook <s3e_walkthrough>`. Every class and function is in
the {doc}`API reference <api/index>`.

## `SemanticStateEstimator`

`SemanticStateEstimator(predicates, vlm=..., translator=...)` builds from an explicit predicate list; `SemanticStateEstimator.from_pddl(domain, problem, vlm=..., translator=...)` grounds predicates from PDDL. Key arguments:

- `predicates` (constructor only): grounded predicate strings to estimate.
- `domain`, `problem` (`from_pddl` only): PDDL domain and problem, as strings or `.pddl` file paths.
- `vlm`: a `VLMBackend` instance or a model-id string (see `resolve_backend`). Strings prefixed with `OpenAI/` select `OpenAIVLM`; any other string selects `HuggingFaceVLM`. For vLLM, construct `VLLMBackend(...)` explicitly and pass the instance.
- `translator`: predicate-to-query strategy (default: `IdentityTranslator`).
- `answers`: the answer space (default: `BinaryAnswers()`; identity translation defaults to `BinaryAnswers("true", "false")`).
- `confidence`: default acceptance threshold used by `__call__`/`to_state`, for binary answer spaces only. A predicate is `True` when `P(true) >= confidence` and `False` otherwise, so every predicate gets a value. A predicate with no answer (see [Results](#results)) has `P(true) = 0.5`: it is `True` at the default `0.5` and `False` for any higher threshold.
- `scoring`: `"logprobs"` (default) scores the masses of the answer tokens; `"text_match"` lets the model generate and matches the reply's *start* against the answer tokens (whole words, case-sensitive, surrounding whitespace ignored, longest token first). A reply such as `"Answer: yes"`, `"**Yes**"`, or one that reasons before answering (e.g. a `<think>` block) matches nothing; prompt for a bare answer.
- `system_prompt`, `prompt_template`, `additional_instructions`: prompt construction; `prompt_template` must contain `{query}`.
- `true_tokens`, `false_tokens`, `null_tokens`: convenience overrides for the default binary answer space; ignored when `answers` is passed explicitly.
- `batch_size`, `vlm_kwargs`, `inference_kwargs`: forwarded to the underlying `QueryEngine` (see below).

Common methods:

- `estimator(images, confidence=None) -> dict[str, bool]`: estimate and threshold into a boolean state.
- `estimator.estimate(images, *, predicates=None, calibrator=None, keep_raw=False, inference_kwargs=None, allow_calibrator_policy_mismatch=False) -> PredictionSet`: full per-predicate detail.
- `estimator.estimate_averaged(scenes, **estimate_kwargs) -> PredictionSet`: estimate each scene separately and average the stored masses.
- `estimator.set_problem(domain, problem)`: re-ground a new PDDL problem; the engine/backend is untouched.
- `estimator.to_up_state(state)`: convert a boolean state dict into a Unified Planning state object (PDDL-built estimators only).

## `QueryEngine`

`QueryEngine(vlm, *, answers=None, scoring="logprobs", system_prompt=None, prompt_template="{query}", batch_size=8, inference_kwargs=None, vlm_kwargs=None)` is the PDDL-free engine `SemanticStateEstimator` is built on. `resolve_backend(vlm, **vlm_kwargs)` is the public model-string-to-backend factory it uses internally.

- `engine.ask(images, queries, *, answers=None, scoring=None, inference_kwargs=None, keep_raw=False) -> PredictionSet`: answer each query about one scene (a list of images shown together).
- `engine.ask_each(scenes, queries, **ask_kwargs) -> list[PredictionSet]`: run `ask` once per scene; combine with `PredictionSet.average(sets)`.

`vlm_kwargs` and `inference_kwargs` are intentionally different:

- `vlm_kwargs` configure backend/client construction, used only when `vlm` is a model string.
  - OpenAI backend: forwarded to `openai.OpenAI(...)` (e.g. `api_key`, `base_url`, `timeout`).
  - HuggingFace backend: forwarded to model construction (e.g. `device_map`, `torch_dtype`, `attn_implementation`).
  - vLLM backend: pass these directly to `VLLMBackend(...)` (e.g. `tensor_parallel_size`, `gpu_memory_utilization`).
- `inference_kwargs` configure runtime inference and are forwarded on every query.
  - OpenAI: request arguments for `chat.completions.create` (e.g. `temperature`, `max_completion_tokens`).
  - HuggingFace: forwarded to `model(...)` in logprobs mode and `model.generate(...)` in generation (`text_match`) mode.
  - vLLM: forwarded to `vllm.SamplingParams` (e.g. `temperature`, `max_tokens`).
  - In `text_match` mode, a query that sets no length gets 256 new tokens on both local backends (`DEFAULT_MAX_REPLY_TOKENS`), not the frameworks' defaults of 20 (transformers) and 16 (vLLM).

Decoding follows one rule on every backend: `temperature` is the switch (absent or 0 decodes greedily; above 0 samples from the full, untruncated distribution), and other decoding settings (`top_p`, `top_k`, penalties, ...) are off unless passed. `HuggingFaceVLM` ignores the decoding settings in a checkpoint's `generation_config.json`, keeping only its token ids, length and token constraints. Every token the library generates, including a thinking model's reasoning, follows this rule, while a logprobs-mode answer is an exact read of the next-token probabilities. OpenAI models that reject `temperature` are queried without it, with a warning.

## Thinking models

Some VLMs reason in a delimited block (e.g. `<think>...</think>`) before answering, and some do so by default: Qwen3.5-4B and larger through their chat template, Mistral reasoning models through their default system prompt (which any `system_prompt`, including the estimator's automatic one, replaces). Read naively, the next token is then the start of the reasoning, not the answer. `HuggingFaceVLM` and `VLLMBackend` take a `thinking` flag:

- `thinking=False` (default): the model answers without reasoning. Where the chat template has a thinking switch, the backend turns it off. It finds the switch at construction by rendering a probe conversation with each known template variable (`enable_thinking` for Qwen3/3.5 and Gemma 4, `thinking`, and Seed's `thinking_budget`) and keeping the one that changes the rendering. Where there is none, a thinking block that still appears is closed at once, so the answer follows an empty block. A block counts when the prompt ends by opening it or the model's most likely next token opens one; a delimiter that only appears in the prompt's text does not count. In logprobs mode, closing costs a few generated steps per affected query, which find the whitespace after the block. `HuggingFaceVLM` reads the answer at the step after it, and `VLLMBackend` in one more scoring pass. In text mode the reply is generated after the closed block.
- `thinking=True`, `max_thinking_tokens=1024`: the backend turns the switch on and the model reasons first, with the query's decoding settings (greedy by default) and up to the budget, after which the block is closed for it. The answer token is then scored, or the reply generated, after its reasoning; the query's length applies to the reply only. Expect a full generation per query. Greedy reasoning can repeat itself until the budget runs out, and an answer read after a block closed at the budget follows unfinished reasoning: `VLMOutput.thinking_truncated` is then `True`, and the engine emits one `ThinkingTruncatedWarning` per call with the count. With `VLLMBackend`, the prompt plus `max_thinking_tokens` (plus the reply, in text mode) must fit in `max_model_len`. A warning says when the model offers no way to think, or when the answer after its block cannot be found.
- `thinking_truncation_notice=None`: text the backend puts inside a block the budget cut off, on its own line before the closing delimiter, e.g. `"THINKING TRUNCATED"`. The default closes the block with nothing added. A block the model closed itself never gets the notice, and `VLMOutput.reasoning` never contains it. The notice conditions the answer, so it shifts logprobs-mode probabilities and text-mode replies, and whether it helps depends on the model and the string. On Blocksworld queries cut off at 64 reasoning tokens, `"(Reasoning cut short; answering now.)"` raised Qwen3.5-4B's accuracy by about 4 points and lowered Qwen3.5-0.8B's by about 30. Measure it on held-out data before using it. A calibrator records the notice in its generation policy and is refused when it changes.
- `chat_template_kwargs`: extra chat-template variables, applied over the switch. For example, `chat_template_kwargs={"enable_thinking": True}` keeps a Qwen3.5 template thinking even with `thinking=False`, and the backend then closes the block.
- The block's text is in `VLMOutput.reasoning` (via `keep_raw=True`): `""` when the backend closed the block without reasoning, `None` when it had no block to handle (including the empty block a template renders when its switch turns thinking off).
- `thinking_delimiters="auto"` (default): uses the first of the known pairs whose delimiters are both single tokens of the model's vocabulary: `<think>`/`</think>` (Qwen3/3.5, DeepSeek-R1, GLM, ...), `<|channel>`/`<channel|>` (Gemma 4), `[THINK]`/`[/THINK]` (Mistral reasoning models), and `<seed:think>`/`</seed:think>` (Seed). Models without any of them are unaffected. Pass one pair to use only that (with a warning when it is not single tokens of the vocabulary), or `None` to turn the handling off, switch included (which requires `thinking=False`).
- Not covered: tokenizers that do not expose the delimiters as single tokens, notably vLLM's Mistral-native tokenizer mode (`tokenizer_mode="mistral"`); and formats that end reasoning with a token sequence instead of one token, such as gpt-oss's harmony channels. So Mistral reasoning models are supported through `HuggingFaceVLM` only: with vLLM, Magistral-Small-2509 ships only Mistral-native tokenizer files, and Ministral-3 models fail inside vLLM's own image processing in HF tokenizer mode (vLLM 0.21).

The extra passes that skip a block follow the query's decoding settings, with the length and stop tokens the backend sets; the answer after the block is then read exactly, like any other query. `VLLMBackend` checks that the engine tokenized each continued prompt like the one `LLM.chat` scored, and raises a `RuntimeError` otherwise: older vLLM versions return no prompt text or text holding the expanded image placeholders, and a text-only engine's tokenizer may add a second BOS token.

Thinking trades compute for a chance at better answers, and the {doc}`walkthrough notebook <s3e_walkthrough>` measures both sides on a toy scene. Keep the default unless held-out labeled data shows that thinking pays for its cost on your task.

How checked models behave (yes/no questions about a toy scene; `HuggingFaceVLM` on CPU and GPU, `VLLMBackend` on GPU except for Ministral):

- Qwen3.5-4B: its chat template thinks by default. With `thinking=False` the switch turns thinking off; closing the block instead (with thinking forced on through `chat_template_kwargs`) gives exactly the same answers; with neither, almost no probability fell on the answer tokens.
- Gemma 4 (E2B): its template leaves thinking off unless the switch turns it on. With `thinking=True` the model opens the block itself and answers correctly after reasoning; `reasoning` starts with the model's `thought` label.
- Ministral-3-3B-Reasoning: has no switch and opens `[THINK]` by default. `thinking=True` answers correctly and confidently. With `thinking=False` its block is closed, and the model picks the right answer but puts only about a third of the probability on the answer tokens, because after an empty block it prefers a full sentence. Use `thinking=True`, or a prompt that asks for a one-word answer.

## Answer spaces

- `BinaryAnswers(true_label="yes", false_label="no", *, true_tokens=None, false_tokens=None, null_label="unknown", null_tokens=None)`: two options with boolean semantics.
- `CategoricalAnswers(options, *, null_label="unknown", null_tokens=None)`: N options, given as labels or explicit `AnswerOption`s.
- `AnswerOption(label, tokens)` / `AnswerOption.make(label, tokens=None)`: a label plus the token strings that express it; labels auto-expand into case/leading-whitespace variants when `tokens` is omitted.

## Results

`Prediction` (one query's outcome) and `PredictionSet` (an ordered mapping of query/predicate → `Prediction`) are both immutable, with lazily cached derivations:

- `prediction.masses`, `.null_mass`, `.unassigned_mass`, `.probability_override` (a calibrator's output): the stored raw data, never altered by the decision rule.
- `prediction.probability`, `.answer`, `.has_answer`, `.null_dominated`, `.matched`, `.confident(threshold)`, `.distribution()`, `.score`, `.log_odds(eps)`: derived on demand.
- `prediction_set.probabilities()`, `.to_state(confidence=0.5)`, `.where(predicate)`.
- `prediction_set.to_dict()` / `PredictionSet.from_dict(d)`: backend-free round trip (e.g. for offline recalibration).

`probability` is `T / (T + F)` over the true and false masses, or the calibrated value when a calibrator was applied. A prediction has **no answer** (`has_answer` is `False`) when the null option is the model's top answer (`null_dominated`) or no answer token got any mass (`matched` is `False`, e.g. a `text_match` reply that starts with none of the tokens). Then `probability` is `0.5` even after calibration, `distribution()` is uniform, and `answer` falls to the first option on the resulting tie (`True` for binary). The engine emits one `UnmatchedAnswerWarning` per call that had unmatched queries; replies that match the null option do not warn.

Because the stored data is untouched, other rules can be derived from it. For example, to count null mass as half true and half false instead: `(T + n/2) / (T + F + n)` from `masses` and `null_mass`, or `w * probability_override + (1 - w) * 0.5` with `w = (T + F) / (T + F + n)` for calibrated predictions (use `0.5` when `T + F + n == 0`). Apply such rules per scene, before `PredictionSet.average`: averaging already applies the built-in rule to which overrides it keeps.

## Calibration

Calibration maps a predicate's score to a probability, fitted on labeled
examples. {doc}`getting-started` shows the whole cycle of collecting data,
fitting a calibrator, and applying it. This section gives the rules each step
follows.

`CalibrationSet.collect` skips predictions with no answer: their probability stays `0.5` whatever the calibrator says, so they are not training points.

`scope` groups samples for fitting: `"global"` (one calibrator for everything), `"lifted"` (one per predicate name, e.g. all `on(...)` instances share a fit), or `"grounded"` (one per fully-grounded predicate). Every group needs both true and false labels and at least two distinct scores, unless you pass `pass_through_single_class=True` to leave such groups uncalibrated; since `collect` drops unanswered predictions, check this on small datasets. When examples span multiple problem instances, set `CalibrationExample.problem` on each; `CalibrationSet.collect` re-grounds the estimator against that problem before querying it, and the saved sample carries the problem string alongside its score and label.

Calibration works the same way in both scoring modes. A calibrator applies only in the mode its data was collected in.

Scores also depend on how the model generates, so `collect` records a generation policy (`QueryEngine.generation_policy()`): the backend's `model_id` when it has one, `thinking` and `max_thinking_tokens`, the `thinking_truncation_notice` when one is set, the `temperature`, and whichever decoding settings (`do_sample`, `top_k`, `top_p`, ...) are set. `estimate(..., calibrator=...)` refuses, with a `ValueError` naming each difference, a calibrator fitted under another policy than the call's (constructor `inference_kwargs` merged with the call's). `allow_calibrator_policy_mismatch=True` applies it anyway with a warning, e.g. for a local copy of the same model; it never bypasses the scoring-mode, domain or answer-space checks. Calibrators saved without a policy apply unchecked.


## Translators

- `IdentityTranslator`: use grounded predicates as-is.
- `TemplateTranslator`: format grounded predicates with per-predicate templates.
- `PrewrittenTranslator`: provide explicit prompts for each grounded predicate.
- `LLMTranslator`: generate natural-language prompts with an LLM and optionally cache them (`cache_dir=...`).

## Environment variables and optional configuration

- `OPENAI_API_KEY`: required for `OpenAIVLM` and OpenAI-backed `LLMTranslator` usage.
- `cache_dir` on `LLMTranslator`: enables on-disk caching of generated predicate translations.
