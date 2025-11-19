import base64
from collections import defaultdict
from io import BytesIO
from typing import Optional

from .vqa_utils import VQAInterface
from ..constants import OPENAI_MODEL_IDENTIFIER

import openai
import numpy as np

TRANSLATION_KEY = "..."
ESTIMATION_KEY = "..."

translation_client = openai.OpenAI(api_key=TRANSLATION_KEY)
estimation_client = openai.OpenAI(api_key=ESTIMATION_KEY)


class OpenAIVQA(VQAInterface):
    def __init__(self, model_id, system, **inference_kwargs):
        self.model_id = model_id.strip(OPENAI_MODEL_IDENTIFIER)
        self.system_prompt = system
        self.inference_kwargs = inference_kwargs

    def __call__(self, images, query_batch, *token_groups_of_interest):
        batch_responses = [
            estimation_query(self.model_id,
                             images,
                             query,
                             system=self.system_prompt,
                             **self.inference_kwargs)
            for query in query_batch
        ]

        return np.array(
            [
                self.extract_token_group_probs(resp, *token_groups_of_interest)
                for resp in batch_responses
            ]
        ).transpose(1, 0)

    def extract_token_group_probs(self, model_output, *token_groups):
        top_logprobs = model_output.choices[0].logprobs.content[0].top_logprobs

        # map tokens to their probabilities
        tok_to_prob = defaultdict(int)
        for item in top_logprobs:
            tok_to_prob[item.token] = np.exp(item.logprob)

        # get probs for tokens of interest
        token_groups_probs = [
            np.sum([tok_to_prob[token]for token in token_group])
            for token_group in token_groups
        ]

        # normalize each group vs all groups
        return [prob / np.sum(token_groups_probs, axis=0) for prob in token_groups_probs]

    # Caching happens automatically in OpenAI API
    
    def generate_system_cache_with_images(self, images):
        pass

    def clear_system_cache(self):
        pass


def translation_query(model_id: str, query: str, system: Optional[str] = None, max_new_tokens: Optional[int] = None,
                      temperature: int = 1.0, top_p=1.0):
    """
    runs a text-to-text query using the openAI API.
    default argument values are set to the defaults of the openAI API.

    Args:
        model_id:
        query:
        system:
        max_new_tokens:
        temperature:
        top_p:

    Returns:

    """
    response = translation_client.responses.create(
        input=query,
        model=model_id,
        instructions=system,
        max_output_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p
    )

    return response.output_text


def estimation_query(model_id, images, query, system=None, max_new_tokens=512, temperature=1.0, top_p=1.0):
        processed_images = [_preprocess_image(image) for image in images]
        images_as_urls = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                          for base64_image in processed_images]
        input_message_content = [{"type": "text", "text": query}] + images_as_urls
        input_message = [
            {"role": "user", "content": input_message_content},
        ]
        if system is not None:
            input_message = [{"role": "developer", "content": system}] + input_message

        response = estimation_client.chat.completions.create(
            messages=input_message,
            model=model_id,
            max_completion_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            logprobs=True,
            top_logprobs=20
        )

        return response


def _preprocess_image(image):
    # Create a BytesIO object
    buffered = BytesIO()

    # Save the image to the BytesIO stream in JPEG format

    image.convert('RGB').save(buffered, format="JPEG")

    # Retrieve the byte data from the BytesIO stream
    img_bytes = buffered.getvalue()

    # Encode the byte data to Base64
    img_base64 = base64.b64encode(img_bytes)

    # Decode the Base64 bytes to a string
    return img_base64.decode('utf-8')
