"""Default system prompts for s3e.

Answer token groups live with the answer spaces (:mod:`s3e.engine.answers`)
and the OpenAI model-id prefix lives with backend resolution
(:mod:`s3e.backends.resolve`); this module holds prompts only.
"""

SYSTEM_PROMPT_NO_TRANSLATION = """The following is a PDDL domain
{domain}
Here are the names of all the objects in the current problem, sorted by their type:
{objects}
Given a grounded predicate with concrete variables, state whether the statement is true or false.
Respond only with a "true" or "false" response and nothing else."""

SYSTEM_PROMPT_IDENTITY = """Given a grounded predicate with concrete variables, state whether the statement is true or false.
Respond only with a "true" or "false" response and nothing else."""

SYSTEM_PROMPT_WITH_TRANSLATION = (
    "A curious human is asking an artificial intelligence assistant yes or no questions. "
    "The assistant answers with one of two responses: YES or NO. "
    "The assistant's response should not include any additional text."
)

SYSTEM_PROMPT_ADDITIONAL_INSTRUCTIONS = (
    "\nAdditional Instructions and clarifications:\n{additional_instructions}"
)
