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

    def __call__(self, images, query, get_logits=False):
        # force queries to be a list batch
        if isinstance(query, str):
            multi_prompt = False
            query = [query]
        else:
            multi_prompt = True

        input_ids, image_tensor, image_sizes = self.prep_inputs(query, images)

        # run inference
        if get_logits:
            # run model feed-forward and get logits
            with torch.inference_mode():
                outputs = self.model.forward(input_ids,
                                             images=image_tensor,
                                             image_sizes=image_sizes,
                                             use_cache=True,
                                             dpo_forward=True)

            out = outputs[0]

        else:
            # generate text output
            cont = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                max_new_tokens=4096,
            )
            out = self.tokenizer.batch_decode(cont, skip_special_tokens=True)

        # output according to input arity
        if multi_prompt:
            return out
        else:
            return out[0]

    def prep_inputs(self, queries, images):
        # force images to be a list of lists, each list belongs to a corresponding query
        num_queries = len(queries)
        if not images:
            images = [[]] * num_queries
            image_tensor = None
            image_sizes = None
        else:
            if not isinstance(images[0], list):
                images = [images] * num_queries  # image separation for image-count per query
            assert len(images) == num_queries, 'must provide an image list for each query'
        
            # images are just called in order, flatten list
            images_flat = [img for img_list in images for img in img_list]
            
            # process images
            image_tensor = process_images(images_flat, self.image_processor, self.model.config)
            image_tensor = [
                img.to(dtype=torch.float16, device=self.device) for img in image_tensor
            ]
            image_sizes = [img.size for img in images_flat]

        # convert queries to prompts with image tokens
        prompts = []
        for imgs, q in zip(images, queries):
            prompts.append(query_to_prompt(q, "qwen_2", len(imgs), self.system_prompt))

        # tokenize prompts
        tokenized_prompts = tokenized_prompts = [
            tokenizer_image_token(p, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            for p in prompts
        ]

        # pad and stack tokenized prompts
        input_ids = []
        max_len_prompt = max([len(tp) for tp in tokenized_prompts])
        pad_token = self.tokenizer.encode(self.tokenizer.pad_token)
        for tp in tokenized_prompts:
            padding = torch.tensor(pad_token * (max_len_prompt - len(tp)), dtype=tp.dtype)
            padded_tp = torch.cat((padding, tp))
            input_ids.append(padded_tp)
        input_ids = torch.stack(input_ids).to(self.device)

        return input_ids, image_tensor, image_sizes

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
