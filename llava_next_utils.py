from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IGNORE_INDEX,
)
from llava.conversation import conv_templates, SeparatorStyle

from misc import remove_from_gpu_memory
from PIL import Image
import torch
import copy
import warnings


class LlavaOVModel:
    def __init__(
        self, model_id, system=None, **inference_kwargs
    ):
        warnings.filterwarnings("ignore")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        (self.tokenizer, self.model, self.image_processor, self.context_len) = (
            load_pretrained_model(model_id, None, "llava_qwen", device_map="auto",
                                  attn_implementation="flash_attention_2" if self.device == 'cuda' else None)
        )

        self.system_prompt = system

        self.inference_kwargs = inference_kwargs

    def __call__(self, images, query, get_probs=False):
        

        if isinstance(query, list):
            multi_prompt = True
        else:
            multi_prompt = False

        image_tensor = process_images(images, self.image_processor, self.model.config)
        image_tensor = [
            _image.to(dtype=torch.float16, device=self.device) for _image in image_tensor
        ]

        prompt = query_to_prompt(query, "qwen_2", len(images), self.system_prompt)

        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.device)
        )
        image_sizes = [_image.size for _image in images]

        if get_probs:
            with torch.inference_mode():
                outputs = self.model.forward(input_ids,
                                        images=image_tensor,
                                        image_sizes=image_sizes,
                                        use_cache=True)
            out = torch.softmax(outputs.logits, dim=-1).cpu()

        else:
            cont = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                max_new_tokens=4096,
            )
            out = self.tokenizer.batch_decode(cont, skip_special_tokens=True)

        remove_from_gpu_memory(image_tensor)

        return out[0]  #TODO handle multiple inputs at once

    # def process_batch(self, images_batch, query_batch, get_probs=False):
    #     input_ids = (
    #         tokenizer_image_token(
    #             prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    #         )
    #         .unsqueeze(0)
    #         .to(self.device)
    #     )
    #     image_sizes = [_image.size for _image in images]

    #     if get_probs:
    #         with torch.inference_mode():
    #             outputs = self.model.forward(input_ids,
    #                                     images=image_tensor,
    #                                     image_sizes=image_sizes,
    #                                     use_cache=True)
    #         out = torch.softmax(outputs.logits, dim=-1).cpu()

    #     else:
    #         cont = self.model.generate(
    #             input_ids,
    #             images=image_tensor,
    #             image_sizes=image_sizes,
    #             do_sample=False,
    #             temperature=0,
    #             max_new_tokens=4096,
    #         )
    #         out = self.tokenizer.batch_decode(cont, skip_special_tokens=True)

    #     remove_from_gpu_memory(image_tensor)

        return out

    def __del__(self):
        remove_from_gpu_memory(self.tokenizer, self.model, self.image_processor)


def query_to_prompt(query, conv_mode, num_images, system_prompt):
    image_tokens = '\n'.join([DEFAULT_IMAGE_TOKEN] * num_images)
    question = image_tokens + "\n" + query
    conv = copy.deepcopy(conv_templates[conv_mode])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    conv.system += system_prompt
    return conv.get_prompt()
