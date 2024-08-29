import os

# models
LLAMA_70B_INSTRUCT = "meta-llama/Meta-Llama-3-70B-Instruct"
LLAVA_7B_OV = "lmms-lab/llava-onevision-qwen2-7b-ov"

# literals map for processed data
LIT_MAP_FILE_NAME = "litmap.json"

# special keys for processed data
RENDERS_KEY = "renders"
TRUE_STATE_KEY = "true_state"
TRUE_STATE_ARR_KEY = "true_state_array"
ESTIMATED_STATE_DICT_PROB_KEY = "estimated_state_dict_prob"
ESTIMATED_STATE_ARR_PROB_KEY = "est_state_array_prob"

# cache directories
CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "cache"))
NL_PREDICATES_CACHE_DIR = os.path.join(CACHE_DIR, "nl-predicates")
