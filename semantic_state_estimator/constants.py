import os

# models
LLAMA_70B_INSTRUCT = "meta-llama/Meta-Llama-3-70B-Instruct"
LLAVA_7B_OV = "lmms-lab/llava-onevision-qwen2-7b-ov"
LLAVA_72B_OV = "lmms-lab/llava-onevision-qwen2-72b-ov"

# literals map for processed data
LIT_MAP_FILE_NAME = "litmap.json"

# directories and filenames for saving datapoints
RENDERS_DIR = 'renders'
TRUE_STATES_DIR = 'true_states'
PROCESSED_DIR = 'processed'
DP_FNAME_FORMAT = "data_point_{}"
EPISODES_DIR = 'episodes'
TRAJECTORY_STEP_FNAME_FORMAT = "STEP_{}"

# directories only for PRB
SCENES_DIR = 'scenes'

# cache directories
CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "cache"))
NL_PREDICATES_CACHE_DIR = os.path.join(CACHE_DIR, "nl-predicates")



