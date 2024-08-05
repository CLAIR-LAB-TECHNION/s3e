from llava.mm_utils import get_model_name_from_path
from llava_with_system_prompt import eval_model

# model_path = "liuhaotian/llava-v1.5-7b"
# model_path = "liuhaotian/llava-v1.6-vicuna-7b"
# model_path = 'liuhaotian/llava-v1.6-vicuna-13b'
# model_path = 'liuhaotian/llava-v1.6-mistral-7b'
model_path = "liuhaotian/llava-v1.6-34b"
system = ("A curious human is asking an artificial intelligence assistant yes or no questions. "
          "The assistant answers with one of three responses: YES, NO, or UNDECIDED. "
          "The assistant's response should not include any additional text.")
prompt = "Is the robot holding the carton of milk?"
image_file = "frame_0000.png"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512,

    # my custom args
    "system_override": True,
    "system": system
})()

eval_model(args)