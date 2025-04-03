"""Module for handling LLaMA language model interactions.

This module provides utilities for loading and running inference with LLaMA language models,
including support for chat-style interactions and batch processing.
"""

import transformers
import torch


def load_model(model_id):
    """Load a LLaMA model for text generation.
    
    Args:
        model_id: The identifier of the model to load (e.g., model name or path).
        
    Returns:
        A transformers pipeline configured for text generation.
    """
    return transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto"
    )


def run_inference_on_query(pipeline, query, system=None, max_new_tokens=512, do_sample=True,
                           temperature=0.6, top_p=0.9, return_as_dialog=False):
    """Run inference on a query using the provided pipeline.
    
    This function supports both single queries and batch processing, with optional
    system prompts and various generation parameters.
    
    Args:
        pipeline: The transformers pipeline to use for inference.
        query: The query or list of queries to process.
        system: Optional system prompt or list of system prompts.
        max_new_tokens: Maximum number of new tokens to generate.
        do_sample: Whether to use sampling during generation.
        temperature: Temperature parameter for generation (higher values = more random).
        top_p: Top-p parameter for nucleus sampling.
        return_as_dialog: Whether to return results in dialog format.
        
    Returns:
        If return_as_dialog is False:
            - For single query: The generated text response
            - For batch queries: List of generated text responses
        If return_as_dialog is True:
            - For single query: List of dialog turns including the response
            - For batch queries: List of dialog turns for each query
    """
    is_single_query = isinstance(query, str)
    
    if is_single_query:
        chat_dialog = [{"role": "user", "content": query}]
    else:
        chat_dialog = [
            [{"role": "user", "content": q}] for q in query
        ]

    if system:
        is_single_system = isinstance(system, str)
        if is_single_query and is_single_system:
            chat_dialog = [{"role": "system", "content": system}] + chat_dialog
        elif is_single_system:
            chat_dialog = [
                [{"role": "system", "content": system}] + c for c in chat_dialog
            ]
        else:
            assert not is_single_query, 'Cannot run a single query with multiple system prompts'
            assert len(chat_dialog) == len(system), 'number of queries and system prompts must match'
            chat_dialog = [
                [{"role": "system", "content": s}] + c for c, s in zip(chat_dialog, system)
            ]

    if is_single_query:
        prompts = [pipeline.tokenizer.apply_chat_template(chat_dialog, tokenize=False)]
    else:
        prompts = [
            pipeline.tokenizer.apply_chat_template(d, tokenize=False)
            for d in chat_dialog
        ]
    
    with torch.inference_mode():
        outputs = pipeline(
            prompts,
            max_new_tokens=max_new_tokens,
            eos_token_id=[
                pipeline.tokenizer.eos_token_id,
                pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ],
            pad_token_id=pipeline.tokenizer.eos_token_id,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p
        )

    out = []
    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        response = output[0]["generated_text"][len(prompt) + 11:]

        if return_as_dialog:
            if is_single_query:
                d = chat_dialog
            else:
                d = chat_dialog[i]
            out.append(d + [{"role": "assistant", "content": response}])
        else:
            out.append(response)

    if is_single_query:
        return out[0]
    else:
        return out
