from pddl2nl_query_converter import PDDL2NLQueryConverter
from llava_utils import LlavaModel
import torch


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
        self.tokens_of_interest = self.vqa_model.tokenizer.encode('yes', 'no')

    def estimate_state(self, images):
        # TODO receive simulation state instead of renderings.

        # state_sym_probs_map = {}
        # for pred, query in self.queries_dict.items():
        #     logits = self.vqa_model(images, query, get_logits=True)  # get logits
        #     yes_no_logits = logits[-1][self.tokens_of_interest]  # filter out logits for "yes" and "no" words
        #     probs = torch.softmax(yes_no_logits, dim=-1)  # get probabilities from logits
        #     state_sym_probs_map[pred] = probs[0].item()  # map probability of "yes" to the predicate
        preds, queries = zip(*self.queries_dict.items())
        logits = self.vqa_model(images, queries, get_logits=True)  # get logits
        yes_no_logits = logits[:, -1, self.tokens_of_interest]
        probs = torch.softmax(yes_no_logits, dim=-1)
        
        return dict(zip(preds, probs[:, 0].tolist()))
