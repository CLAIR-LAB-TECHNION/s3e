"""Module for converting PDDL predicates to natural language queries.

This module provides functionality to convert PDDL (Planning Domain Definition Language)
predicates into natural language queries that can be used to determine the truth value
of predicates in a given planning domain.
"""

from .up_utils import *
from .llama_utils import *
from .misc import remove_from_gpu_memory


class PDDL2NLQueryConverter:
    """A class that converts PDDL predicates to natural language queries.
    
    This class uses a language model to convert PDDL predicates into human-readable
    questions that can be used to determine the truth value of predicates in a planning domain.
    
    Attributes:
        model: The language model used for generating natural language queries.
        up_problem: The Unified Planning problem instance.
        domain: The PDDL domain string.
        problem: The PDDL problem string.
        objects: Dictionary mapping object types to lists of object names.
        all_grounded_predicates: List of all possible grounded predicates.
        objects_by_type: String representation of objects grouped by type.
        system_prompt: The system prompt used for the language model.
        inference_kwargs: Additional keyword arguments for model inference.
    """

    def __init__(self, model, up_problem, **inference_kwargs):
        """Initialize the PDDL2NLQueryConverter.
        
        Args:
            model: The language model to use for query generation.
            up_problem: The Unified Planning problem instance.
            **inference_kwargs: Additional keyword arguments for model inference.
        """
        self.model = model
        self.up_problem = up_problem
        self.domain, self.problem = get_pddl_files_str(up_problem)
        self.objects = get_object_names_dict(up_problem)
        self.all_grounded_predicates = get_all_grounded_predicates_for_objects(
            up_problem
        )

        self.objects_by_type = "\n".join([f"{key} type: {list(map(str, value))}" for key, value in self.objects.items()])
        self.system_prompt = f"""The following is a PDDL domain:
{self.domain}
Here are the names of all the objects in the current problem, sorted by their type:
{self.objects_by_type}
Given a grounded predicate with concrete variables, write a natural language yes-no query whose answer determines the truth value of the predicate.
Respond only with this natural language query and nothing else."""

        self.inference_kwargs = inference_kwargs

    @classmethod
    def from_uninitialized(cls, model, domain_or_up_problem, problem=None, **inference_kwargs):
        """Create a PDDL2NLQueryConverter instance from uninitialized components.
        
        This class method provides a convenient way to create an instance when starting
        with raw PDDL files or strings.
        
        Args:
            model: The language model to use for query generation.
            domain_or_up_problem: Either a Unified Planning problem instance or a PDDL domain string.
            problem: PDDL problem string (required if domain_or_up_problem is a string).
            **inference_kwargs: Additional keyword arguments for model inference.
            
        Returns:
            A new PDDL2NLQueryConverter instance.
            
        Raises:
            AssertionError: If domain is specified as a string but problem is not provided.
        """
        if isinstance(model, str):
            model = load_model(model)
        
        if isinstance(domain_or_up_problem, str):
            assert (problem is not None), "if domain is specified, problem must also be specified"
            up_problem = create_up_problem(domain_or_up_problem, problem)
        else:
            up_problem = domain_or_up_problem

        return cls(model, up_problem, **inference_kwargs)

    def convert_to_nl(self, grounded_predicate):
        """Convert a grounded PDDL predicate to a natural language query.
        
        Args:
            grounded_predicate: The grounded PDDL predicate to convert.
            
        Returns:
            A natural language query that determines the truth value of the predicate.
        """
        return run_inference_on_query(
            self.model,
            grounded_predicate,
            self.system_prompt,
            **self.inference_kwargs
        )

    def __del__(self):
        """Clean up GPU memory when the instance is deleted."""
        remove_from_gpu_memory(self.model)
