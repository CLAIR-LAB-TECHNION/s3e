from pddl2nl_query_converter import PDDL2NLQueryConverter
from llava_utils import LlavaModel
import torch
from tqdm.auto import tqdm


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
        # set kwargs to default values
        nl_converter_kwargs = nl_converter_kwargs or {}
        vqa_kwargs = vqa_kwargs or {}

        # create pddl predicate-to-NL converter
        pddl2nl = PDDL2NLQueryConverter.from_uninitialized(
            nl_converter_model_id, domain, problem, **nl_converter_kwargs
        )

        # convert predicates to NL queries
        queries = pddl2nl.convert_to_nl(pddl2nl.all_grounded_predicates)
        self.queries_dict = {
            pred: query for pred, query in zip(pddl2nl.all_grounded_predicates, queries)
        }

        # remove pddl predicate-to-NL converter from memory
        del pddl2nl

        # setup VQA model with system prompt
        system = (
            "A curious human is asking an artificial intelligence assistant yes or no questions. "
            "The assistant answers with one of three responses: YES or NO. "
            "The assistant's response should not include any additional text."
        )
        self.vqa_model = LlavaModel(
            vqa_model_id, system=system, system_override=True, **vqa_kwargs
        )

        # get tokens for words of interest (yes and no)
        self.tokens_of_interest = self.vqa_model.tokenizer.encode(['yes', 'no'])
        self.yes_tokens = self.vqa_model.tokenizer.encode(['yes', 'YES', 'Yes'])
        self.no_tokens = self.vqa_model.tokenizer.encode(['no', 'NO', 'No'])

    def estimate_state(self, images):
        # TODO receive simulation state instead of renderings.

        state_sym_probs_map = {}
        for pred, query in tqdm(self.queries_dict.items()):
            probs = self.vqa_model(images, query, get_probs=True)[-1]  # get probs for next token
            yes_probs = probs[self.yes_tokens].sum()  # get yes tokens prob
            no_probs = probs[self.no_tokens].sum()  # get no tokens prob
            state_sym_probs_map[pred] = (yes_probs / (yes_probs + no_probs)).item()  # map probability of "yes" to the predicate
        return state_sym_probs_map

        # state = set()
        # for pred, query in tqdm(self.queries_dict.items()):
        #     response = self.vqa_model(images, query)
        #     if response.lower() == 'yes':
        #         state.add(pred)
        # return state
        
        #TODO conform to batch size to preserve memory
        # preds, queries = zip(*self.queries_dict.items())
        # probs = self.vqa_model(images, queries, get_probs=True)[:, -1]  # get probs of next token
        # yes_probs = probs[:, self.yes_tokens].sum(dim=-1)
        # no_probs = probs[:, self.no_tokens].sum(dim=-1)
        # normalized_yes_probs = yes_probs / (yes_probs + no_probs)
        # return dict(zip(preds, normalized_yes_probs.tolist()))
