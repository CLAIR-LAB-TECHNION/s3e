import os
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor

MODEL_DIR = "/export/work/gazran/hf_models"
MODEL_ID = "LlamaFinetuneBase/Meta-Llama-3.2-90B-Vision-Instruct"
# MODEL_ID = "LlamaFinetuneBase/Meta-Llama-3.2-11B-Vision"



class LLaMA32Model:
    def __init__(self, model_id=MODEL_ID, system=None, system_images=None, **inference_kwargs):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_path = os.path.join(MODEL_DIR, model_id)

        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer

        self.system_prompt = system

        self.inference_kwargs = inference_kwargs


    def __call__(self, images, query):
        # force queries to be a list batch
        if isinstance(query, str):
            multi_prompt = False
            queries = [query]
        else:
            queries = query
            multi_prompt = True

        input_text = []
        for query in queries:
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{self.system_prompt} {query}"}
                ]}
            ]
            input_text.append(self.processor.apply_chat_template(messages))

        inputs = self.processor(images, input_text, return_tensors="pt").to(self.device)

        # run model feed-forward and get logits
        with torch.inference_mode():
            outputs = self.model(
                **inputs,
                use_cache=True,
            )

        out = outputs.logits

        # output according to input arity
        if multi_prompt:
            return out
        else:
            return out[0]

    def generate_system_cache_with_images(*args, **kwargs):
        pass