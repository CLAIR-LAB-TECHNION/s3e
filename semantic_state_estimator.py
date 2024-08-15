import os
from pddl2nl_query_converter import PDDL2NLQueryConverter
from misc import model_and_kwargs_to_filename, NL_PREDICATES_CACHE_DIR
from tqdm.auto import tqdm
import json
from up_utils import create_up_problem
import transformers
import math
import torch

if transformers.__version__ == "4.37.2":
    legacy = True
    from llava_utils import LlavaModel
else:
    legacy = False
    from llava_next_utils import LlavaOVModel as LlavaModel


class SemanticStateEstimator:
    def __init__(
        self,
        domain,
        problem,
        nl_converter_model_id,
        vqa_model_id,
        nl_converter_kwargs=None,
        vqa_kwargs=None,
    ):
        self.queries_dict = self.get_queries_dict(domain,
                                                  problem,
                                                  nl_converter_model_id,
                                                  nl_converter_kwargs)

        # setup VQA model with system prompt
        system = (
            "A curious human is asking an artificial intelligence assistant yes or no questions. "
            "The assistant answers with one of three responses: YES or NO. "
            "The assistant's response should not include any additional text."
        )
        self.vqa_model = LlavaModel(
            vqa_model_id, system=system, **(vqa_kwargs or {})
        )

        # get tokens for words of interest (yes and no)
        self.yes_tokens = list(map(lambda x: x[0],
                               self.vqa_model.tokenizer(['yes', 'YES', 'Yes'])['input_ids']))
        self.no_tokens = list(map(lambda x: x[0],
                               self.vqa_model.tokenizer(['no', 'NO', 'No'])['input_ids']))

    def estimate_state(self, images):
        state_sym_probs_map = {}
        for pred, query in tqdm(self.queries_dict.items()):
            logits = self.vqa_model(images, query, get_logits=True)[-1].float()  # get probs for next token
            yes_logits = logits[self.yes_tokens].sum()  # get yes tokens prob
            no_logits = logits[self.no_tokens].sum()  # get no tokens prob
            exp_yes = torch.exp(yes_logits)
            exp_no = torch.exp(no_logits)
            state_sym_probs_map[pred] = (
                exp_yes / (exp_yes + exp_no)
            ).item()  # map probability of "yes" to the predicate
        return state_sym_probs_map

        # state = set()
        # for pred, query in tqdm(self.queries_dict.items()):
        #     response = self.vqa_model(images, query)
        #     if response.lower() == 'yes':
        #         state.add(pred)
        # return state

    def estimate_state_par(self, images, batch_size=8):
        out = {}
        num_batches = int(math.ceil(len(self.queries_dict) / batch_size))
        preds, queries = zip(*self.queries_dict.items())
        for i in tqdm(range(num_batches)):
            # get next batch of queries and corresponding predicates
            q_batch = queries[i*batch_size:(i+1)*batch_size]
            p_batch = preds[i*batch_size:(i+1)*batch_size]

            # get logits of next token
            # convert to bigger float (output is 16 bits)
            logits = self.vqa_model(images, q_batch, get_logits=True)[:, -1].float()
            
            # get logits for "yes" and "no" tokens
            yes_logits = logits[:, self.yes_tokens].sum(dim=-1)
            no_logits = logits[:, self.no_tokens].sum(dim=-1)
            
            # calculate normalized probability for yes and no.
            # skip softmax by directly calculating normalized exp values
            # skip operating on ENITIRE VOCAB.
            exp_yes = torch.exp(yes_logits)
            exp_no = torch.exp(no_logits)
            normalized_yes_probs = exp_yes / (exp_yes + exp_no)

            # update with batch info
            out.update(
                dict(zip(p_batch, normalized_yes_probs.tolist()))
            )

        return out

    def swap_queries(self, domain, problem, nl_converter_model_id, nl_converter_kwargs=None):
        self.queries_dict = self.get_queries_dict(domain, problem, nl_converter_model_id,
                                                  nl_converter_kwargs)

    @classmethod
    def get_queries_dict(cls, domain, problem, nl_converter_model_id, nl_converter_kwargs=None):
        try:
            queries_dict = cls.load_queries_dict_from_cache(
                domain, problem, nl_converter_model_id, nl_converter_kwargs
            )
            print('predicate queries loaded from cache')
        except FileNotFoundError:
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
                nl_converter_model_id,
                pddl_problem=problem_name,
                **nl_converter_kwargs
            )
            + ".json"
        )
        os.makedirs(NL_PREDICATES_CACHE_DIR, exist_ok=True)
        with open(os.path.join(NL_PREDICATES_CACHE_DIR, cache_fname), "w") as f:
            json.dump(queries_dict, f)

        # remove pddl predicate-to-NL converter from memory
        del pddl2nl

        return queries_dict

    @staticmethod
    def load_queries_dict_from_cache(
        domain, problem, nl_converter_model_id, nl_converter_kwargs=None
    ):
        problem_name = create_up_problem(domain, problem).name
        cache_fname = (
            model_and_kwargs_to_filename(
                nl_converter_model_id,
                pddl_problem=problem_name,
                **(nl_converter_kwargs or {})
            )
            + ".json"
        )
        with open(os.path.join(NL_PREDICATES_CACHE_DIR, cache_fname), "r") as f:
            return json.load(f)
