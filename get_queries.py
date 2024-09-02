import warnings
from semantic_state_estimator.semantic_state_estimator import SemanticStateEstimatorWithLLaMA

warnings.filterwarnings("ignore")

LLAMA_ID = "meta-llama/Meta-Llama-3-70B-Instruct"
LLAVA_ID = "lmms-lab/llava-onevision-qwen2-7b-ov"

EXAMPLE = 'blocksworld'

se = SemanticStateEstimatorWithLLaMA(
    domain=f'examples/{EXAMPLE}/domain.pddl',
    problem=f'examples/{EXAMPLE}/problem.pddl',
    nl_converter_model_id=LLAMA_ID,
    vqa_model_id=LLAVA_ID
)