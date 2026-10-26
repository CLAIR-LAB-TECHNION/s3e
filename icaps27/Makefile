# Experiments for the S3E tool paper. See README.md for what each target runs.

PY ?= .venv/bin/python
RESULTS ?= results/raw
INSTANCES ?= instances
# Models: names from configs/models.json (or Hugging Face ids).
MODELS ?= smolvlm-256m qwen3-vl-4b qwen3-vl-8b gemma-4-e4b
SCALE_MODEL ?= qwen3-vl-8b
REASONING_MODELS ?= qwen3.5-4b qwen3.5-0.8b ministral-3-3b-reasoning
TOOLS_MODEL ?= qwen3-vl-8b
T2V_MODEL ?= qwen3-vl-8b
# The S3E source for overlays of unreleased commits (anonymized mirrors: override).
S3E_GIT ?= https://github.com/CLAIR-LAB-TECHNION/s3e
# Thinking-model support is on S3E main after 0.5.0 (E5 needs it).
THINKING_REF ?= 2a09b7e
T2V_GIT ?= https://github.com/linzhiqiu/t2v_metrics
T2V_REF ?= 6ecb74f92028f42c7e64546d3a71e98c8c73068f
REF ?= 0.2.0
# Backend for collection (hf or vllm); empty uses each model's registry default.
BACKEND ?=
BACKEND_ARG = $(if $(BACKEND),--backend $(BACKEND))
SMOKE ?= results/smoke
SMOKE_MODEL ?= smolvlm-256m
# Recorded ViPlan++ episodes (directories holding <cell>/episodes/episode_*).
VIPLAN_GW ?=
VIPLAN_PRB ?=
VIPLAN_ALF ?=

.PHONY: help env env-vllm env-t2v overlay-s3e test lint smoke smoke-fake instances \
	instances-viplan collect collect-viplan e1 e1-online e2 e3 e4 e5 e6 e7 workshop reproduce figures

help:
	@sed -n 's/^## //p' $(MAKEFILE_LIST)

## env            create .venv (Python 3.12, s3e==0.5.0 from PyPI, Fast Downward)
env:
	uv sync --python 3.12 --extra dev

## env-vllm       .venv plus vLLM (CUDA hosts; E2's HF-vs-vLLM rows)
env-vllm:
	uv sync --python 3.12 --extra dev --extra vllm

## env-t2v        separate environment for t2v_metrics (E6)
env-t2v:
	uv venv envs/t2v --python 3.12
	uv pip install --python envs/t2v/bin/python "t2v-metrics @ git+$(T2V_GIT)@$(T2V_REF)" imageio-ffmpeg audioread
	# t2v_metrics refuses to import without an ffmpeg executable on PATH (and imports
	# audioread, which its requirements omit at this commit).
	ln -sf "$$(envs/t2v/bin/python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" \
		envs/t2v/bin/ffmpeg

## overlay-s3e    install S3E REF (a version like 0.2.0 or a commit) into envs/s3e-REF
overlay-s3e:
	@if echo "$(REF)" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$$'; then \
		spec="s3e==$(REF)"; else spec="s3e @ git+$(S3E_GIT)@$(REF)"; fi; \
	uv pip install --python $(PY) --no-deps --target envs/s3e-$(REF) "$$spec"

## test           the harness's own tests (fake model, CPU, seconds)
test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check s3e_exp experiments tests

## smoke-fake     the whole pipeline with the fake model (1 minute, no download)
smoke-fake:
	$(MAKE) smoke SMOKE_MODEL=fake SMOKE=results/smoke-fake

## smoke          E1-E4 on two tiny scenes with SmolVLM-256M (CPU: ~10-40 min)
smoke:
	$(PY) experiments/instances.py blocksworld --sizes 2 --seeds 0 --count 2 --out $(SMOKE)/instances
	$(PY) experiments/collect.py --instances $(SMOKE)/instances --model $(SMOKE_MODEL) \
		--views 0 1 --out $(SMOKE)/predictions
	$(PY) experiments/collect.py --instances $(SMOKE)/instances --model $(SMOKE_MODEL) \
		--scoring text_match --out $(SMOKE)/predictions
	$(PY) experiments/e1_closed_loop.py plan --instances $(SMOKE)/instances --folds 2 \
		--predictions $(SMOKE)/predictions/blocksworld/$(SMOKE_MODEL)/logprobs-none \
		--out $(SMOKE)/e1/plan.jsonl
	$(PY) experiments/e1_closed_loop.py summary $(SMOKE)/e1/plan.jsonl
	$(PY) experiments/e1_closed_loop.py online --instances $(SMOKE)/instances --limit 1 \
		--model $(SMOKE_MODEL) --policy monitor --out $(SMOKE)/e1/online
	$(PY) experiments/e2_scale.py count --instances $(SMOKE)/instances --out $(SMOKE)/e2/count.jsonl
	$(PY) experiments/e3_scoring.py --instances $(SMOKE)/instances \
		--runs $(SMOKE)/predictions/blocksworld/$(SMOKE_MODEL)/logprobs-none \
		       $(SMOKE)/predictions/blocksworld/$(SMOKE_MODEL)/text_match-none \
		--out $(SMOKE)/e3.json
	$(PY) experiments/e4_calibration.py scopes --instances $(SMOKE)/instances --folds 2 \
		--run $(SMOKE)/predictions/blocksworld/$(SMOKE_MODEL)/logprobs-none --out $(SMOKE)/e4.json
	@echo "smoke test passed: outputs in $(SMOKE)"

## instances      the rendered Blocksworld set: 3-7 blocks x 3 seeds x 100 scenes, 4 views
instances:
	$(PY) experiments/instances.py blocksworld --sizes 3 4 5 6 7 --seeds 0 1 2 --count 100 \
		--out $(INSTANCES)/blocksworld

## instances-viplan  export recorded ViPlan++ episodes (set VIPLAN_GW/PRB/ALF)
instances-viplan:
	@if [ -n "$(VIPLAN_GW)" ]; then $(PY) experiments/instances.py viplan-gw \
		--export $(VIPLAN_GW) --out $(INSTANCES)/viplan-gw; fi
	@if [ -n "$(VIPLAN_PRB)" ]; then $(PY) experiments/instances.py viplan-prb \
		--export $(VIPLAN_PRB) --out $(INSTANCES)/viplan-prb; fi
	@if [ -n "$(VIPLAN_ALF)" ]; then $(PY) experiments/instances.py viplan-alf \
		--export $(VIPLAN_ALF) --every 2 --out $(INSTANCES)/viplan-alf; fi

## collect        per model: logprobs on view 0 (all scenes) and views 1-3 + all-in-one
##                (seed 0), text predictions, and the dynamic mask (GPU; BACKEND=hf|vllm)
collect:
	for model in $(MODELS); do \
		$(PY) experiments/collect.py --instances $(INSTANCES)/blocksworld --model $$model \
			$(BACKEND_ARG) --views 0 --out $(RESULTS)/predictions || exit 1; \
		$(PY) experiments/collect.py --instances $(INSTANCES)/blocksworld --model $$model \
			$(BACKEND_ARG) --seeds 0 --views 1 2 3 all --out $(RESULTS)/predictions || exit 1; \
		$(PY) experiments/collect.py --instances $(INSTANCES)/blocksworld --model $$model \
			$(BACKEND_ARG) --scoring text_match --max-new-tokens 16 \
			--out $(RESULTS)/predictions || exit 1; \
		$(PY) experiments/collect.py --instances $(INSTANCES)/blocksworld --model $$model \
			$(BACKEND_ARG) --mask dynamic --out $(RESULTS)/predictions || exit 1; \
	done

## collect-viplan logprobs and text predictions on exported ViPlan++ frames (GPU);
##                ALF uses the relevant mask (it has 1,000-2,600 atoms per frame)
collect-viplan:
	for d in viplan-gw viplan-prb viplan-alf; do [ -d $(INSTANCES)/$$d ] || continue; \
		if [ $$d = viplan-alf ]; then masks=relevant; else masks="none dynamic"; fi; \
		for model in $(MODELS); do for mask in $$masks; do \
			$(PY) experiments/collect.py --instances $(INSTANCES)/$$d --model $$model \
				--mask $$mask --out $(RESULTS)/predictions || exit 1; done; \
			$(PY) experiments/collect.py --instances $(INSTANCES)/$$d --model $$model \
				--mask $${masks%% *} --scoring text_match --max-new-tokens 16 \
				--out $(RESULTS)/predictions || exit 1; done; done

## e1             offline planning from every collected logprobs run (CPU)
e1:
	$(PY) experiments/e1_closed_loop.py plan --instances $(INSTANCES)/blocksworld \
		--predictions $(wildcard $(RESULTS)/predictions/blocksworld/*/logprobs-*) \
		--out $(RESULTS)/e1/plan-blocksworld.jsonl
	$(PY) experiments/e1_closed_loop.py summary $(RESULTS)/e1/plan-blocksworld.jsonl --by-size \
		--json $(RESULTS)/e1/plan-blocksworld.summary.json
	for d in viplan-gw viplan-prb viplan-alf; do [ -d $(RESULTS)/predictions/$$d ] || continue; \
		$(PY) experiments/e1_closed_loop.py plan --instances $(INSTANCES)/$$d \
			--decoders threshold calibrated \
			--predictions $(RESULTS)/predictions/$$d/*/logprobs-* \
			--out $(RESULTS)/e1/plan-$$d.jsonl || exit 1; done

## workshop       the 2025 workshop setting: LLaVA-OneVision on GroceriesWorld frames (GPU)
workshop:
	for size in 0.5b 7b; do \
		$(PY) experiments/collect.py --instances $(INSTANCES)/viplan-gw --model llava-ov-$$size \
			--out $(RESULTS)/predictions || exit 1; \
		$(PY) experiments/e7_versions.py workshop --instances $(INSTANCES)/viplan-gw \
			--run $(RESULTS)/predictions/viplan-gw/llava-ov-$$size/logprobs-none \
			--row "$$(echo $$size | tr b B) + Pose" \
			--out $(RESULTS)/e7/workshop-gw-llava-ov-$$size.json || exit 1; done

## e1-online      closed loop with the model in the loop (GPU): per model and policy,
##                20 episodes (4 per block count, seed 0)
e1-online:
	for model in $(MODELS); do for policy in open-loop replan monitor; do \
		$(PY) experiments/e1_closed_loop.py online --instances $(INSTANCES)/blocksworld \
			--seeds 0 --limit 4 --model $$model --policy $$policy \
			--out $(RESULTS)/e1/online || exit 1; done; \
		$(PY) experiments/e1_closed_loop.py online --instances $(INSTANCES)/blocksworld \
			--seeds 0 --limit 4 --model $$model --policy monitor --check action \
			--out $(RESULTS)/e1/online || exit 1; \
		$(PY) experiments/e1_closed_loop.py online --instances $(INSTANCES)/blocksworld \
			--seeds 0 --limit 4 --model $$model --policy monitor --decoder map \
			--out $(RESULTS)/e1/online || exit 1; done

## e2             atom counts (CPU) and timing vs objects / batch size / backend (GPU)
e2:
	$(PY) experiments/e2_scale.py count --instances $(INSTANCES)/blocksworld \
		--out $(RESULTS)/e2/count-blocksworld.jsonl
	for d in viplan-gw viplan-prb viplan-alf; do if [ -d $(INSTANCES)/$$d ]; then \
		$(PY) experiments/e2_scale.py count --instances $(INSTANCES)/$$d \
			--out $(RESULTS)/e2/count-$$d.jsonl; fi; done
	$(PY) experiments/e2_scale.py time --instances $(INSTANCES)/blocksworld --seeds 0 --limit 10 \
		--model $(SCALE_MODEL) --backend hf --batch-sizes 1 8 16 32 64 --masks none dynamic \
		--out $(RESULTS)/e2/time-blocksworld-hf.jsonl
	@if $(PY) -c "import vllm" 2>/dev/null; then \
		$(PY) experiments/e2_scale.py time --instances $(INSTANCES)/blocksworld --seeds 0 --limit 10 \
			--model $(SCALE_MODEL) --backend vllm --batch-sizes 8 16 32 64 128 --masks none dynamic \
			--out $(RESULTS)/e2/time-blocksworld-vllm.jsonl && \
		$(PY) experiments/e2_scale.py time --instances $(INSTANCES)/blocksworld --seeds 0 --limit 10 \
			--model $(SCALE_MODEL) --backend vllm --vlm-kwargs '{"num_logprobs": null}' \
			--batch-sizes 64 --masks none --out $(RESULTS)/e2/time-blocksworld-vllm-fullvocab.jsonl; \
	else echo "vLLM not installed (make env-vllm): skipping the vLLM rows"; fi

## e3             scoring modes from the collected runs (CPU)
e3:
	for model in $(MODELS); do \
		$(PY) experiments/e3_scoring.py --instances $(INSTANCES)/blocksworld \
			--runs $(RESULTS)/predictions/blocksworld/$$model/logprobs-none \
			       $(RESULTS)/predictions/blocksworld/$$model/text_match-none \
			--out $(RESULTS)/e3/$$model.json || exit 1; done

## e4             calibration scopes, transfer, and views from the collected runs (CPU)
e4:
	for model in $(MODELS); do run=$(RESULTS)/predictions/blocksworld/$$model/logprobs-none; \
		$(PY) experiments/e4_calibration.py scopes --instances $(INSTANCES)/blocksworld \
			--run $$run --out $(RESULTS)/e4/scopes-$$model.json || exit 1; \
		$(PY) experiments/e4_calibration.py transfer --instances $(INSTANCES)/blocksworld \
			--run $$run --out $(RESULTS)/e4/transfer-$$model.json || exit 1; \
		$(PY) experiments/e4_calibration.py views --instances $(INSTANCES)/blocksworld \
			--run $$run --calibrate --out $(RESULTS)/e4/views-$$model.json || exit 1; done

## e5             reasoning models (GPU; needs the thinking overlay, built here)
e5:
	$(MAKE) overlay-s3e REF=$(THINKING_REF)
	for model in $(REASONING_MODELS); do \
		PYTHONPATH=envs/s3e-$(THINKING_REF) $(PY) experiments/e5_reasoning.py predict \
			--instances $(INSTANCES)/blocksworld --sizes 3 4 5 --seeds 0 --limit 20 \
			--model $$model --out $(RESULTS)/e5 || exit 1; \
		$(PY) experiments/e5_reasoning.py summary $(RESULTS)/e5/blocksworld/$$model \
			--instances $(INSTANCES)/blocksworld --json $(RESULTS)/e5/$$model.json || exit 1; done

## e6             S3E vs a hand-written scorer and t2v_metrics (GPU; env-t2v for t2v).
##                All three run unbatched in bfloat16 with no system prompt, as t2v_metrics does.
e6:
	$(PY) experiments/collect.py --instances $(INSTANCES)/blocksworld --seeds 0 --limit 10 \
		--model $(TOOLS_MODEL) --system-prompt '' --batch-size 1 \
		--vlm-kwargs '{"torch_dtype": "bfloat16"}' --tag e6 --out $(RESULTS)/predictions
	$(PY) experiments/e6_tools.py export --instances $(INSTANCES)/blocksworld --seeds 0 --limit 10 \
		--system-prompt '' --out $(RESULTS)/e6/queries.jsonl
	$(PY) experiments/e6_tools.py adhoc --queries $(RESULTS)/e6/queries.jsonl --dtype bfloat16 \
		--model $(TOOLS_MODEL) --out $(RESULTS)/e6/adhoc-$(TOOLS_MODEL).jsonl
	if [ -x envs/t2v/bin/python ]; then $(PY) experiments/e6_tools.py t2v \
		--queries $(RESULTS)/e6/queries.jsonl --t2v-model $(T2V_MODEL) \
		--out $(RESULTS)/e6/t2v-$(T2V_MODEL).jsonl; fi
	$(PY) experiments/e6_tools.py compare --queries $(RESULTS)/e6/queries.jsonl \
		--s3e $(RESULTS)/predictions/blocksworld/$(TOOLS_MODEL)/logprobs-none-e6 \
		--tool adhoc=$(RESULTS)/e6/adhoc-$(TOOLS_MODEL).jsonl \
		$(if $(wildcard envs/t2v/bin/python),t2v_metrics=$(RESULTS)/e6/t2v-$(T2V_MODEL).jsonl) \
		--out $(RESULTS)/e6/compare-$(TOOLS_MODEL).json
	$(PY) experiments/e6_tools.py features

## e7             S3E 0.2.0 vs the installed 0.5.0 (GPU) and the API surface (CPU)
e7:
	$(MAKE) overlay-s3e REF=0.2.0
	$(MAKE) overlay-s3e REF=$(THINKING_REF)
	$(PY) experiments/e7_versions.py api --versions 0.2.0 installed $(THINKING_REF) \
		--out $(RESULTS)/e7
	$(PY) experiments/e7_versions.py summary $(RESULTS)/e7/api --json $(RESULTS)/e7/api.summary.json
	$(PY) experiments/e7_versions.py run --instances $(INSTANCES)/blocksworld --seeds 0 \
		--sizes 3 5 7 --limit 10 --model $(SCALE_MODEL) --versions 0.2.0 installed \
		--out $(RESULTS)/e7
	$(PY) experiments/e7_versions.py summary $(RESULTS)/e7/$(SCALE_MODEL) \
		--json $(RESULTS)/e7/$(SCALE_MODEL).summary.json

## reproduce      everything (GPU, many hours); each step resumes
reproduce: instances collect collect-viplan e1 e1-online e2 e3 e4 e5 e6 e7 workshop figures

## figures        tables and figures from results/raw (CPU, no model)
figures:
	$(PY) experiments/figures.py --results $(RESULTS) --out results/figures
