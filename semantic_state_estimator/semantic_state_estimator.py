import os
from .utils.pddl2nl_query_converter import PDDL2NLQueryConverter
from .utils.misc import model_and_kwargs_to_filename
from .constants import NL_PREDICATES_CACHE_DIR
from tqdm.auto import tqdm
import json
from .utils.up_utils import create_up_problem, get_all_grounded_predicates_for_objects
import transformers
import math
import torch
import numpy as np
from .state_estimator import ProbabilisticStateEstimator

if transformers.__version__ == "4.37.2":
    legacy = True
    from .utils.llava_utils import LlavaModel
elif transformers.__version__ == "4.40.0.dev0":
    legacy = False
    from .utils.llava_next_utils import LlavaOVModel as LlavaModel
else:
    from transformers import CLIPProcessor, CLIPModel


class SemanticStateEstimator(ProbabilisticStateEstimator):
    def __init__(
        self,
        domain,
        problem,
        vqa_model_id,
        vqa_kwargs=None,
        additional_instructions=None,
        confidence=0.5
    ):
        super().__init__(domain, problem, confidence)

        pddl2nl = PDDL2NLQueryConverter.from_uninitialized(None, domain, problem)
        self.predicates = pddl2nl.all_grounded_predicates
        
        system = f"""The following is a PDDL domain
{pddl2nl.domain}
Here are the names of all the objects in the current problem, sorted by their type:
{pddl2nl.objects_by_type}
Given a grounded predicate with concrete variables, state whether the statement is true or false.
Respond only with a "true" or "false" response and nothing else."""

        if additional_instructions:
            system += f'\nAdditional Instructions and clarifications:\n{additional_instructions}'

        self.vqa_model = LlavaModel(vqa_model_id, system=system, **(vqa_kwargs or {}))

        # get tokens for words of interest (yes and no)
        self.true_tokens = list(
            map(
                lambda x: x[0],
                self.vqa_model.tokenizer(["true", "True", "TRUE"])["input_ids"],
            )
        )
        self.false_tokens = list(
            map(
                lambda x: x[0],
                self.vqa_model.tokenizer(["false", "False", "FALSE"])["input_ids"],
            )
        )

    def estimate_state(self, images):
        return self.estimate_state_par(images, batch_size=1)

        # state = set()
        # for pred, query in tqdm(self.queries_dict.items()):
        #     response = self.vqa_model.generate(images, query)
        #     if response.lower() == 'yes':
        #         state.add(pred)
        # return state

    def estimate_state_par(self, images, batch_size=8):
        # cache repeated parts of the prompt, including images
        self.vqa_model.generate_system_cache_with_images(images)

        out = {}
        num_batches = int(math.ceil(len(self.predicates) / batch_size))
        for i in tqdm(range(num_batches)):
            # get next batch of queries and corresponding predicates
            p_batch = self.predicates[i * batch_size : (i + 1) * batch_size]

            # get logits of next token
            # convert to bigger float (output is 16 bits)
            logits = self.vqa_model(images, p_batch)[:, -1].float()

            # get logits for "yes" and "no" tokens
            true_logits = logits[:, self.true_tokens].sum(dim=-1)
            false_logits = logits[:, self.false_tokens].sum(dim=-1)

            # calculate normalized probability for yes and no.
            # skip softmax by directly calculating normalized exp values
            # skip operating on ENITIRE VOCAB.
            exp_true = torch.exp(true_logits)
            exp_false = torch.exp(false_logits)
            normalized_true_probs = exp_true / (exp_true + exp_false)

            # update with batch info
            out.update(dict(zip(p_batch, normalized_true_probs.tolist())))

        # empty GPU cache
        self.vqa_model.clear_system_cache()

        return out


class SemanticEstimatorMultiImageRunNoLLaMA(SemanticStateEstimator):
        def estimate_state_par(self, images, batch_size=8):
            outputs_per_image = []
            for img in tqdm(images):
                out = super().estimate_state_par([img], batch_size)
                outputs_per_image.append(out)
            
            out = {
                predicate: np.mean([outputs_per_image[i][predicate] for i in range(len(images))])
                for predicate in outputs_per_image[0]  # expected at least 1 image as input
            }

            return out

    

class SemanticStateEstimatorWithLLaMA(ProbabilisticStateEstimator):
    def __init__(
        self,
        domain,
        problem,
        nl_converter_model_id,
        vqa_model_id,
        nl_converter_kwargs=None,
        vqa_kwargs=None,
        additional_instructions=None,
        additional_images=None,
        confidence=0.5
    ):
        super().__init__(domain, problem, confidence)

        self.queries_dict = self.get_queries_dict(
            domain, problem, nl_converter_model_id, nl_converter_kwargs
        )

        # setup VQA model with system prompt
        system = (
            "A curious human is asking an artificial intelligence assistant yes or no questions. "
            "The assistant answers with one of three responses: YES or NO. "
            "The assistant's response should not include any additional text."
        )
        if additional_instructions:
            system += f'\nAdditional Instructions and clarifications:\n{additional_instructions}'

        self.vqa_model = LlavaModel(vqa_model_id, system=system, system_images=additional_images or [],
                                    **(vqa_kwargs or {}))

        # get tokens for words of interest (yes and no)
        self.yes_tokens = list(
            map(
                lambda x: x[0],
                self.vqa_model.tokenizer(["yes", "YES", "Yes"])["input_ids"],
            )
        )
        self.no_tokens = list(
            map(
                lambda x: x[0],
                self.vqa_model.tokenizer(["no", "NO", "No"])["input_ids"],
            )
        )

    def estimate_state(self, images):
        return self.estimate_state_par(images, batch_size=1)

        # state = set()
        # for pred, query in tqdm(self.queries_dict.items()):
        #     response = self.vqa_model.generate(images, query)
        #     if response.lower() == 'yes':
        #         state.add(pred)
        # return state

    def estimate_state_par(self, images, batch_size=8):
        # cache repeated parts of the prompt, including images
        self.vqa_model.generate_system_cache_with_images(images)

        out = {}
        num_batches = int(math.ceil(len(self.queries_dict) / batch_size))
        preds, queries = zip(*self.queries_dict.items())
        for i in tqdm(range(num_batches)):
            # get next batch of queries and corresponding predicates
            q_batch = queries[i * batch_size : (i + 1) * batch_size]
            p_batch = preds[i * batch_size : (i + 1) * batch_size]

            # get logits of next token
            # convert to bigger float (output is 16 bits)
            logits = self.vqa_model(images, q_batch)[:, -1].float()

            # get probability of "yes" as opposed to "no"
            normalized_yes_probs = self.logits_to_yes_no_probs(logits)

            # update with batch info
            out.update(dict(zip(p_batch, normalized_yes_probs.tolist())))

        # empty GPU cache
        self.vqa_model.clear_system_cache()

        return out

    def logits_to_yes_no_probs(self, logits):
        # get logits for "yes" and "no" tokens
        yes_logits = logits[:, self.yes_tokens].sum(dim=-1)
        no_logits = logits[:, self.no_tokens].sum(dim=-1)

        # calculate normalized probability for yes and no.
        # skip softmax by directly calculating normalized exp values
        # skip operating on ENITIRE VOCAB.
        exp_yes = torch.exp(yes_logits)
        exp_no = torch.exp(no_logits)
        normalized_yes_probs = exp_yes / (exp_yes + exp_no)

        return normalized_yes_probs

    def swap_queries(
        self, domain, problem, nl_converter_model_id, nl_converter_kwargs=None
    ):
        self.queries_dict = self.get_queries_dict(
            domain, problem, nl_converter_model_id, nl_converter_kwargs
        )

    @classmethod
    def get_queries_dict(
        cls, domain, problem, nl_converter_model_id, nl_converter_kwargs=None
    ):
        try:
            queries_dict = cls.load_queries_dict_from_cache(
                domain, problem, nl_converter_model_id, nl_converter_kwargs
            )
            print("predicate queries loaded from cache")
        except (FileNotFoundError, KeyError) as e:
            print("loading predicate queries with LLM")
            queries_dict = cls.load_queries_dict_with_model(
                domain, problem, nl_converter_model_id, nl_converter_kwargs
            )

        return queries_dict

    @staticmethod
    def load_queries_dict_with_model(
        domain, problem, nl_converter_model_id, nl_converter_kwargs=None
    ):
        # set kwargs to default values
        nl_converter_kwargs = nl_converter_kwargs or {}

        # create pddl predicate-to-NL converter
        pddl2nl = PDDL2NLQueryConverter.from_uninitialized(
            nl_converter_model_id, domain, problem, **nl_converter_kwargs
        )

        # convert predicates to NL queries
        queries = pddl2nl.convert_to_nl(pddl2nl.all_grounded_predicates)
        queries_dict = {
            pred: query for pred, query in zip(pddl2nl.all_grounded_predicates, queries)
        }

        # cache qeuries dict
        problem_name = pddl2nl.up_problem.name
        cache_fname = (
            model_and_kwargs_to_filename(
                nl_converter_model_id, pddl_problem=problem_name, **nl_converter_kwargs
            )
            + ".json"
        )
        os.makedirs(NL_PREDICATES_CACHE_DIR, exist_ok=True)
        cache_file_path = os.path.join(NL_PREDICATES_CACHE_DIR, cache_fname)

        # if already exists, update the file and don't delete existing
        # this is good in case we have the same problem with different objects
        if os.path.exists(cache_file_path):
            with open(cache_file_path, 'r') as f:
                old_queries_dict = json.load(f)
            old_queries_dict.update(queries_dict)  # update old with new
            queries_dict = old_queries_dict

        with open(cache_file_path, "w") as f:
            json.dump(queries_dict, f, indent=4)

        # remove pddl predicate-to-NL converter from memory
        del pddl2nl

        return queries_dict

    @staticmethod
    def load_queries_dict_from_cache(
        domain, problem, nl_converter_model_id, nl_converter_kwargs=None
    ):
        up_problem = create_up_problem(domain, problem)
        problem_name = up_problem.name
        cache_fname = (
            model_and_kwargs_to_filename(
                nl_converter_model_id,
                pddl_problem=problem_name,
                **(nl_converter_kwargs or {})
            )
            + ".json"
        )
        with open(os.path.join(NL_PREDICATES_CACHE_DIR, cache_fname), "r") as f:
            queries_dict = json.load(f)
        
        # filter out irrelevant predicates
        grounded_predicates = get_all_grounded_predicates_for_objects(up_problem)
        queries_dict = {predicate: queries_dict[predicate] for predicate in grounded_predicates}

        return queries_dict


class SemanticEstimatorMultiImageRun(SemanticStateEstimatorWithLLaMA):
        def estimate_state_par(self, images, batch_size=8):
            outputs_per_image = []
            for img in tqdm(images):
                out = super().estimate_state_par([img], batch_size)
                outputs_per_image.append(out)
            
            out = {
                predicate: np.mean([outputs_per_image[i][predicate] for i in range(len(images))])
                for predicate in outputs_per_image[0]  # expected at least 1 image as input
            }

            return out



class SemanticEstimatorWithCLIP(SemanticStateEstimatorWithLLaMA):
    def __init__(self, domain,
                 problem,
                 nl_converter_model_id,
                 vqa_model_id="openai/clip-vit-base-patch32",
                 nl_converter_kwargs=None,
                 vqa_kwargs=None):
        super(ProbabilisticStateEstimator, self).__init__(domain, problem)
        self.queries_dict = self.get_queries_dict(
            domain, problem, nl_converter_model_id, nl_converter_kwargs
        )

        self.vqa_model = CLIPModel.from_pretrained(
            vqa_model_id,
            # attn_implementation="flash_attention_2",
            device_map='auto',
            torch_dtype=torch.float16
        )
        self.processor = CLIPProcessor.from_pretrained(vqa_model_id)
    
    def estimate_state(self, images):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        out = {}
        for pred, query in tqdm(self.queries_dict.items()):
            inputs = self.processor(
                text=[f"{query} Yes.", "{query} No."],
                images=images,
                return_tensors="pt",
                padding=True
            )
            inputs.to('cuda')
            with torch.no_grad():
                with torch.autocast(device):
                    outputs = self.vqa_model(**inputs)
            
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

            out[pred] = probs[0][0].item()
        
        return out
