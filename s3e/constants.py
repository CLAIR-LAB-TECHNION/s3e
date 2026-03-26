"""Default prompts and token groups for s3e.

This module contains the default system prompts and token groups used by
SemanticStateEstimator. Users can override any of these when constructing
an estimator.
"""

# OpenAI model identifier prefix
OPENAI_MODEL_IDENTIFIER = "OpenAI/"

# --- Default system prompts ---

SYSTEM_PROMPT_NO_TRANSLATION = """The following is a PDDL domain
{domain}
Here are the names of all the objects in the current problem, sorted by their type:
{objects}
Given a grounded predicate with concrete variables, state whether the statement is true or false.
Respond only with a "true" or "false" response and nothing else."""

SYSTEM_PROMPT_WITH_TRANSLATION = (
    "A curious human is asking an artificial intelligence assistant yes or no questions. "
    "The assistant answers with one of two responses: YES or NO. "
    "The assistant's response should not include any additional text."
)

SYSTEM_PROMPT_ADDITIONAL_INSTRUCTIONS = (
    "\nAdditional Instructions and clarifications:\n{additional_instructions}"
)

# --- Default token groups for probability extraction ---

TRUE_TOKENS_NO_TRANSLATION = ["true", "True", "TRUE"]
FALSE_TOKENS_NO_TRANSLATION = ["false", "False", "FALSE"]

TRUE_TOKENS_WITH_TRANSLATION = ["yes", "Yes", "YES"]
FALSE_TOKENS_WITH_TRANSLATION = ["no", "No", "NO"]
