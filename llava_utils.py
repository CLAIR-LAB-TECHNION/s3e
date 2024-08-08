import torch
from misc import remove_from_gpu_memory

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import re


class LlavaModel:
    def __init__(self, model_id, model_base=None, conv_mode=None, system=None, system_override=False, 
                 **inference_kwargs):
        (self.tokenizer,
         self.model,
         self.image_processor,
         self.context_len) = load_model(model_id, model_base)
        
        self.conv_mode = get_conv_mode(model_id, conv_mode)

        self.system_prompt = system
        self.system_override = system_override

        self.inference_kwargs = inference_kwargs

    def __call__(self, image_files, query, file_sep=',', get_logits=False, get_probs=False,
                 return_prompt=False):
        prompt, output = eval_preloaded_model(
            self.tokenizer,
            self.model,
            self.image_processor,
            image_files,
            query,
            self.conv_mode,
            self.system_prompt,
            self.system_override,
            file_sep,
            get_logits=get_logits,
            get_probs=get_probs,
            **self.inference_kwargs
        )
        
        if return_prompt:
            return prompt, output
        else:
            return output
    
    def __del__(self):
        remove_from_gpu_memory(self.tokenizer, self.model, self.image_processor)


# ===== Load Model =====

def load_model(model_path, model_base=None):
    disable_torch_init()

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name
    )

    return tokenizer, model, image_processor, context_len


# ===== Handle Images =====

def image_parser(image_file, sep=","):
    out = image_file.split(sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def load_image_tensors(images, image_processor, model, sep=","):
    if isinstance(images, str):
        images = image_parser(images, sep)
        pil_images = load_images(images)
    else:
        if images.ndim == 3:
            pil_images = [Image.fromarray(images)]
        else:
            pil_images = [Image.fromarray(img) for img in images]
    image_sizes = [x.size for x in pil_images]
    images_tensor = process_images(pil_images, image_processor, model.config).to(
        model.device, dtype=torch.float16
    )

    return images_tensor, image_sizes


def set_image_token(model, qs):
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    return qs


# ===== Handle Conversation =====


def get_prompt(qs, conv_mode, system=None, system_override=False):
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)

    # update system prompt
    if system is not None:
        if system_override:
            conv.system = system
        else:
            conv.system += f" {system}."

    return conv.get_prompt()


def get_conv_mode(model_path, set_conv_mode=None):
    model_name = get_model_name_from_path(model_path)

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if set_conv_mode is not None and conv_mode != set_conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, set_conv_mode, set_conv_mode
            )
        )
        conv_mode = set_conv_mode

    return conv_mode


# ===== Inference =====

def run_inference(tokenizer, model, prompt, images_tensor, image_sizes,
                  temperature=0, top_p=None, num_beams=1, max_new_tokens=512,
                  get_logits=False, get_probs=False):
    #TODO support running with multiple images for the same query

    if isinstance(prompt, list):
        multi_prompt = True
        tokenized_prompts = [
            tokenizer_image_token(p, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            for p in prompt
        ]
        max_len_prompt = max([len(tp) for tp in tokenized_prompts])
        pad_token = tokenizer.encode(tokenizer.pad_token)

        input_ids = []
        for tp in tokenized_prompts:
            padding = torch.tensor(pad_token * (max_len_prompt - len(tp)), dtype=torch.int)
            padded_tp = torch.cat((tp, padding))
            input_ids.append(padded_tp)
        input_ids = torch.stack(input_ids).cuda()

        if images_tensor.shape[0] == 1:
            images_tensor = torch.stack([images_tensor[0]] * len(prompt))
            image_sizes = image_sizes * len(prompt)
    else:
        multi_prompt = False
        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

    if get_logits or get_probs:
        with torch.inference_mode():
            outputs = model.forward(input_ids,
                                    images=images_tensor,
                                    image_sizes=image_sizes,
                                    use_cache=True)

        out = outputs.logits
        if get_probs:
            out = torch.softmax(out, dim=-1)
        out = out.cpu()
        if multi_prompt:
            return out
        else:
            return out[0]

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    if multi_prompt:
        return [s.strip() for s in tokenizer.batch_decode(output_ids, skip_special_tokens=True)]
    else:
        return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def eval_preloaded_model(tokenizer, model, image_processor, images, query, conv_mode, system=None,
                         system_override=False, sep=',', temperature=0,
                         top_p=None, num_beams=1, max_new_tokens=512, get_logits=False,
                         get_probs=False):
    if isinstance(query, str):
        qs = set_image_token(model, query)
        prompt = get_prompt(qs, conv_mode, system, system_override)
    else:
        qs = [set_image_token(model, q) for q in query]
        prompt = [get_prompt(q, conv_mode, system, system_override) for q in qs]
    
    if isinstance(images, (list | tuple)):
        images_tensor, image_sizes = zip(*[load_image_tensors(imgs, image_processor, model, sep) for imgs in images])
        images_tensor = torch.cat(images_tensor)
        image_sizes = [s[0] for s in image_sizes]
    else:
        images_tensor, image_sizes = load_image_tensors(images, image_processor, model, sep)
    
    outputs = run_inference(tokenizer, model, prompt, images_tensor, image_sizes,
                            temperature, top_p, num_beams, max_new_tokens, get_logits, get_probs)
                            
    return prompt, outputs


def eval_model(model_path, images, query, conv_mode=None, system=None, system_override=False,
               sep=',', model_base=None, temperature=0, top_p=None, num_beams=1,
               max_new_tokens=512, get_logits=False, get_probs=False):
    tokenizer, model, image_processor, context_len = load_model(model_path, model_base)
    conv_mode = get_conv_mode(model_path, conv_mode)

    return eval_preloaded_model(tokenizer, model, image_processor, images, query, conv_mode,
                                system, system_override, sep, temperature, top_p, num_beams,
                                max_new_tokens, get_logits, get_probs)


# ===== Results Display =====

def pretty_print_results(prompt, output):
    print('==== PROMPT ====')
    print(prompt)
    print('================')
    print()
    print('==== RESPONSE ====')
    print(output)
    print('==================')


def display_res(query, output, image_file, save_path=None):
    img = Image.open(image_file)
    imd = ImageDraw.Draw(img)
    fnt = ImageFont.truetype("LiberationMono-Regular.ttf", 30)
    imd.text((28, 36), f'{query}: {output}', font=fnt, fill=(255, 0, 255))
    if save_path is None:
        img.show()
    else:
        img.save(save_path)
