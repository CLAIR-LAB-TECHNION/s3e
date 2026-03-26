"""LLM-driven translator — uses a language model to generate natural language queries.

Supports both HuggingFace text models (via ``AutoModelForCausalLM``) and
OpenAI API models (identified by the ``"OpenAI/"`` prefix).
"""

from tqdm.auto import tqdm

from .translator import QueryTranslator
from ..cache import make_cache_key, load_cache, save_cache
from ..constants import OPENAI_MODEL_IDENTIFIER
from ..pddl.up_utils import create_up_problem, get_object_names_dict, get_all_grounded_predicates_for_objects


_NL_SYSTEM_PROMPT_TEMPLATE = """The following is a PDDL domain:
{domain}
Here are the names of all the objects in the current problem, sorted by their type:
{objects_by_type}
Given a grounded predicate with concrete variables, write a natural language yes-no query whose answer determines the truth value of the predicate.
Respond only with this natural language query and nothing else."""


def _openai_translate(model_id: str, predicate: str, system_prompt: str, **kwargs) -> str:
    """Translate a single predicate using the OpenAI API."""
    try:
        import openai
    except ImportError:
        raise ImportError(
            "The 'openai' package is required for LLMTranslator with OpenAI models. "
            "Install it with: pip install s3e[openai]"
        )

    client = openai.OpenAI()
    response = client.responses.create(
        input=predicate,
        model=model_id,
        instructions=system_prompt,
        **kwargs,
    )
    return response.output_text


def _huggingface_translate(
    model, tokenizer, predicates: list[str], system_prompt: str, **kwargs
) -> list[str]:
    """Translate predicates using a HuggingFace causal language model."""
    import torch

    results = []
    kwargs.setdefault("max_new_tokens", 128)

    for predicate in tqdm(predicates, desc="Translating predicates"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": predicate},
        ]

        try:
            input_text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        except Exception:
            input_text = f"{system_prompt}\n\n{predicate}"

        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, **kwargs)

        # Trim input tokens
        generated_ids = output_ids[0, inputs["input_ids"].shape[-1]:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        results.append(text)

    return results


class LLMTranslator(QueryTranslator):
    """Translator that uses a language model to generate natural language queries.

    For OpenAI models, the ``model_id`` must start with ``"OpenAI/"``
    (e.g. ``"OpenAI/gpt-4o"``). For HuggingFace models, pass any valid
    model identifier.

    Args:
        model_id: Model identifier. ``"OpenAI/..."`` uses the OpenAI API;
            anything else loads a HuggingFace ``AutoModelForCausalLM``.
        cache_dir: Directory for caching translations. ``None`` disables caching.
        **inference_kwargs: Additional kwargs passed to the model during generation.
    """

    def __init__(self, model_id: str, cache_dir: str | None = None, **inference_kwargs):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.inference_kwargs = inference_kwargs
        self._is_openai = model_id.startswith(OPENAI_MODEL_IDENTIFIER)

        # Load HF model eagerly (OpenAI is stateless)
        self._hf_model = None
        self._hf_tokenizer = None
        if not self._is_openai:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._hf_tokenizer = AutoTokenizer.from_pretrained(model_id)
            self._hf_model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map="auto"
            )

    def translate(self, predicates, domain, problem):
        """Translate predicates to natural language queries.

        Uses cache when available. Only calls the LLM for predicates not already cached.
        """
        # Build system prompt from PDDL context
        up_problem = create_up_problem(domain, problem)
        objects = get_object_names_dict(up_problem)
        objects_by_type = "\n".join(
            f"{key} type: {list(map(str, value))}"
            for key, value in objects.items()
        )
        system_prompt = _NL_SYSTEM_PROMPT_TEMPLATE.format(
            domain=domain, objects_by_type=objects_by_type
        )

        # Try loading from cache
        cached: dict[str, str] = {}
        cache_key = None
        if self.cache_dir is not None:
            cache_key = make_cache_key(
                self.model_id, up_problem.name, **self.inference_kwargs
            )
            try:
                cached = load_cache(self.cache_dir, cache_key)
            except FileNotFoundError:
                pass

        # Determine which predicates still need translation
        missing = [p for p in predicates if p not in cached]

        if missing:
            new_translations = self._translate_batch(missing, system_prompt)

            # Update cache
            if self.cache_dir is not None and cache_key is not None:
                save_cache(self.cache_dir, cache_key, new_translations)

            cached.update(new_translations)

        return {p: cached[p] for p in predicates}

    def _translate_batch(
        self, predicates: list[str], system_prompt: str
    ) -> dict[str, str]:
        """Translate a batch of predicates (uncached) using the configured model."""
        if self._is_openai:
            openai_model_id = self.model_id.removeprefix(OPENAI_MODEL_IDENTIFIER)
            translations = {}
            for pred in tqdm(predicates, desc="Translating predicates (OpenAI)"):
                query = _openai_translate(
                    openai_model_id, pred, system_prompt, **self.inference_kwargs
                )
                translations[pred] = query
            return translations
        else:
            assert self._hf_model is not None and self._hf_tokenizer is not None
            queries = _huggingface_translate(
                self._hf_model, self._hf_tokenizer, predicates, system_prompt,
                **self.inference_kwargs,
            )
            return dict(zip(predicates, queries))
