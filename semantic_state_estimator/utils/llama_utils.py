import transformers
import torch


def load_model(model_id):
    return transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto"
    )


def run_inference_on_query(pipeline, query, system=None, max_new_tokens=512, do_sample=True,
                           temperature=0.6, top_p=0.9, return_as_dialog=False):
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
