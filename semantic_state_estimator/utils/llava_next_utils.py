"""Module for handling LLaVA (Large Language and Vision Assistant) model interactions.

This module provides utilities for working with LLaVA models that can process both text
and image inputs, including support for system prompts, caching, and batch processing.
"""

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

from .misc import remove_from_gpu_memory
from PIL import Image
import torch
import copy
import warnings

try:
    import flash_attn
    flash_attn_installed = True
except ImportError:
    flash_attn_installed = False



class LlavaOVModel:
    """A class for handling LLaVA model interactions with both text and image inputs.
    
    This class provides a comprehensive interface for working with LLaVA models,
    including support for system prompts, image processing, and efficient caching.
    
    Attributes:
        device: The device to run the model on (cuda or cpu).
        tokenizer: The model's tokenizer.
        model: The LLaVA model instance.
        image_processor: The image processor for handling visual inputs.
        context_len: The maximum context length for the model.
        system_prompt: Optional system prompt for the model.
        system_images: Optional list of system images.
        inference_kwargs: Additional keyword arguments for model inference.
        system_cache: Cached system context for improved efficiency.
    """

    def __init__(self, model_id, system=None, system_images=None, **inference_kwargs):
        """Initialize the LLaVA model wrapper.
        
        Args:
            model_id: The identifier of the model to load.
            system: Optional system prompt.
            system_images: Optional list of system images.
            **inference_kwargs: Additional keyword arguments for model inference.
        """
        warnings.filterwarnings("ignore")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        (self.tokenizer, self.model, self.image_processor, self.context_len) = (
            load_pretrained_model(
                model_id,
                None,
                "llava_qwen",
                device_map="auto",
                attn_implementation=(
                    "flash_attention_2" if self.device == "cuda" and flash_attn_installed else None
                ),
            )
        )
        self.model.eval()

        self.system_prompt = system
        self.system_images = system_images or []

        self.inference_kwargs = inference_kwargs

        self.system_cache = None

    def __call__(self, images, query):
        """Run forward pass of the model on the given inputs.
        
        Args:
            images: Image or list of images to process.
            query: Query or list of queries to process.
            
        Returns:
            Model outputs for the given inputs.
        """
        # force queries to be a list batch
        if isinstance(query, str):
            multi_prompt = False
            queries = [query]
        else:
            queries = query
            multi_prompt = True

        input_ids, image_tensor, image_sizes = self.prep_inputs(
            queries, images, using_system_cache=self.system_cache is not None
        )

        # expand system cache to match batch size
        if self.system_cache:
            batch_size = len(queries)
            expanded_cache = [
                (k.repeat(batch_size, 1, 1, 1), v.repeat(batch_size, 1, 1, 1))
                for k, v in self.system_cache
            ]
            modalities = ["image"] * batch_size
        else:
            expanded_cache = None
            modalities = ["image"]

        # run model feed-forward and get logits
        with torch.inference_mode():
            outputs = self.model.forward(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                past_key_values=expanded_cache,
                use_cache=True,
                dpo_forward=True,
                modalities=modalities
            )

        out = outputs[0]

        # output according to input arity
        if multi_prompt:
            return out
        else:
            return out[0]

    def generate(self, images, query):
        """Generate text output from the model.
        
        Args:
            images: Image or list of images to process.
            query: Query or list of queries to process.
            
        Returns:
            Generated text responses.
        """
        # force queries to be a list batch
        if isinstance(query, str):
            multi_prompt = False
            query = [query]
        else:
            multi_prompt = True

        input_ids, image_tensor, image_sizes = self.prep_inputs(query, images)

        # generate text output
        with torch.inference_mode():
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

    def prep_inputs(
        self, queries, images, using_system_cache=False, remove_assistant=False
    ):
        """Prepare inputs for the model.
        
        Args:
            queries: List of queries to process.
            images: List of images to process.
            using_system_cache: Whether to use cached system context.
            remove_assistant: Whether to remove assistant responses from prompts.
            
        Returns:
            Tuple of (input_ids, image_tensor, image_sizes).
        """
        # force images to be a list of lists, each list belongs to a corresponding query
        num_queries = len(queries)
        if not images or using_system_cache:
            images = [[]] * num_queries
            image_tensor = None
            image_sizes = None
        else:
            if not isinstance(images[0], list):
                images = [
                    images
                ] * num_queries  # image separation for image-count per query
            assert (
                len(images) == num_queries
            ), "must provide an image list for each query"

            # images are just called in order, flatten list
            images_flat = [img for img_list in images for img in img_list]

            # process images
            image_tensor = process_images(
                images_flat, self.image_processor, self.model.config
            )
            image_tensor = [
                img.to(dtype=torch.float16, device=self.device) for img in image_tensor
            ]
            image_sizes = [img.size for img in images_flat]

        # convert queries to prompts with image tokens
        prompts = []
        for imgs, q in zip(images, queries):
            prompts.append(
                query_to_prompt(
                    q,
                    "qwen_2",
                    len(imgs),
                    self.system_prompt,
                    using_system_cache,
                    remove_assistant,
                )
            )

        # tokenize prompts
        tokenized_prompts = tokenized_prompts = [
            tokenizer_image_token(
                p, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            for p in prompts
        ]

        # pad and stack tokenized prompts
        input_ids = []
        max_len_prompt = max([len(tp) for tp in tokenized_prompts])
        pad_token = self.tokenizer.encode(self.tokenizer.pad_token)
        for tp in tokenized_prompts:
            padding = torch.tensor(
                pad_token * (max_len_prompt - len(tp)), dtype=tp.dtype
            )
            padded_tp = torch.cat((padding, tp))
            input_ids.append(padded_tp)
        input_ids = torch.stack(input_ids).to(self.device)

        return input_ids, image_tensor, image_sizes

    def generate_system_cache_with_images(self, images):
        """Generate and cache system context with images.
        
        Args:
            images: List of images to include in system context.
        """
        # prep inputs without a user query
        input_ids, image_tensor, image_sizes = self.prep_inputs(
            [""], self.system_images + images, remove_assistant=True
        )

        with torch.inference_mode():
            outputs = self.model.forward(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                use_cache=True,
                **self.inference_kwargs
            )
        self.system_cache = outputs.past_key_values
        remove_from_gpu_memory(input_ids, image_tensor)

    def clear_system_cache(self):
        """Clear the cached system context."""
        remove_from_gpu_memory(self.system_cache)
        self.system_cache = None

    def __del__(self):
        """Clean up GPU memory when the instance is deleted."""
        remove_from_gpu_memory(self.tokenizer, self.model, self.image_processor)


def query_to_prompt(
    query,
    conv_mode,
    num_images,
    system_prompt,
    using_system_cache=False,
    remove_assistant=False,
):
    """Convert a query to a prompt with image tokens.
    
    Args:
        query: The query to convert.
        conv_mode: The conversation mode to use.
        num_images: Number of images in the query.
        system_prompt: Optional system prompt.
        using_system_cache: Whether to use cached system context.
        remove_assistant: Whether to remove assistant responses from prompts.
        
    Returns:
        The formatted prompt with image tokens.
    """
    # append image tokens at the start of the query
    # these signal the model to take the next image of the input into consideration
    image_tokens = "\n".join([DEFAULT_IMAGE_TOKEN] * num_images)
    query = image_tokens + "\n" + query

    # prepare conversation template
    conv = copy.deepcopy(conv_templates[conv_mode])
    conv.append_message(conv.roles[0], query)  # add
    conv.append_message(conv.roles[1], None)  # empty assistant prompt
    conv.system += system_prompt  # add system prompt

    # convert conv template to prompt
    prompt = conv.get_prompt()

    # remove start if using system cache
    # system cache includes the system prompt and the image prompts
    if using_system_cache:
        user_role = conv.roles[0]
        user_prompt_start = prompt.find(user_role) + len(user_role)
        prompt = prompt[user_prompt_start:]

    # remove assistant prompt if required
    if remove_assistant:
        prompt = prompt.replace(conv.roles[1], "")

    return prompt
