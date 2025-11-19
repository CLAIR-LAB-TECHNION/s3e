import os
# os.environ["TRANSFORMERS_CACHE"] = "/datasets/huggingface"
# os.environ["HF_HOME"] = "/home/guy.azran/.cache/huggingface"

import warnings

import fire

from semantic_state_estimator.semantic_state_estimator import SemanticStateEstimatorWithLLaMA



def get_queries(domain, problem, llama_model="meta-llama/Meta-Llama-3-70B-Instruct"):
    warnings.filterwarnings("ignore")

    SemanticStateEstimatorWithLLaMA(
        domain=domain,
        problem=problem,
        nl_converter_model_id=llama_model,
        vqa_model_id="lmms-lab/llava-onevision-qwen2-0.5b-ov"  # smallest model possible. don't care if it loads
    )

if __name__ == "__main__":
    fire.Fire(get_queries)
