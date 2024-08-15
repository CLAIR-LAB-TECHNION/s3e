from up_utils import *
from llama_utils import *
from misc import remove_from_gpu_memory


class PDDL2NLQueryConverter:
    def __init__(self, model, up_problem, **inference_kwargs):
        self.model = model
        self.up_problem = up_problem
        self.domain, self.problem = get_pddl_files_str(up_problem)
        self.objects = get_object_names_dict(up_problem)
        self.all_grounded_predicates = get_all_grounded_predicates_for_objects(
            up_problem
        )

        objects_by_type = "\n".join([f"{key} type: {list(map(str, value))}" for key, value in self.objects.items()])
        self.system_prompt = f"""The following is a PDDL domain
{self.domain}
Here are the names of all the objects in the current problem, sorted by their type:
{objects_by_type}
Given a grounded predicate with concrete variables, write a natural language yes-no query whose answer determines the truth value of the predicate.
Respond only with this natural language query and nothing else."""

        self.inference_kwargs = inference_kwargs

    @classmethod
    def from_uninitialized(cls, model, domain_or_up_problem, problem=None, **inference_kwargs):
        if isinstance(model, str):
            model = load_model(model)
        
        if isinstance(domain_or_up_problem, str):
            assert (problem is not None), "if domain is specified, problem must also be specified"
            up_problem = create_up_problem(domain_or_up_problem, problem)
        else:
            up_problem = domain_or_up_problem

        return cls(model, up_problem, **inference_kwargs)

    def convert_to_nl(self, grounded_predicate):
        return run_inference_on_query(
            self.model,
            grounded_predicate,
            self.system_prompt,
            **self.inference_kwargs
        )

    def __del__(self):
        remove_from_gpu_memory(self.model)
