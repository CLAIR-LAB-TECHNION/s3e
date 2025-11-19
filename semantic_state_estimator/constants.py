import os

# models
LLAMA_70B_INSTRUCT = "meta-llama/Meta-Llama-3-70B-Instruct"
LLAVA_7B_OV = "lmms-lab/llava-onevision-qwen2-7b-ov-sft"
LLAVA_72B_OV = "lmms-lab/llava-onevision-qwen2-72b-ov-sft"
GPT_4O_LATEST = "OpenAI/gpt-4o-2024-11-20"
GPT_41_LATEST = "OpenAI/gpt-4.1-2025-04-14"

# literals map for processed data
LIT_MAP_FILE_NAME = "litmap.json"

# directories and filenames for saving datapoints
RENDERS_DIR = 'renders'
TRUE_STATES_DIR = 'true_states'
PROCESSED_DIR = 'processed'
DP_FNAME_FORMAT = "data_point_{}"
EPISODES_DIR = 'episodes'
TRAJECTORY_STEP_FNAME_FORMAT = "STEP_{}"
DONEFILE_NAME = 'DONE'

# directories only for PRB
SCENES_DIR = 'scenes'

# cache directories
CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "cache"))
NL_PREDICATES_CACHE_DIR = os.path.join(CACHE_DIR, "nl-predicates")

#OpenAI model identifier
OPENAI_MODEL_IDENTIFIER = "OpenAI/"

# prompts
SYSTEM_PROMPT_NO_TRANSLATION = """The following is a PDDL domain
{domain}
Here are the names of all the objects in the current problem, sorted by their type:
{objects}
Given a grounded predicate with concrete variables, state whether the statement is true or false.
Respond only with a "true" or "false" response and nothing else."""

SYSTEM_PROMPT_WITH_TRANSLATION = (
    "A curious human is asking an artificial intelligence assistant yes or no questions. "
    "The assistant answers with one of two responses: YES or NO. "
    "The assistant's response should not include any additional text."
)

SYSTEM_PROMPT_ADDITIONAL_INSTRUCTIONS = "\nAdditional Instructions and clarifications:\n{additional_instructions}"

