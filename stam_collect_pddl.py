from semantic_state_estimator.utils.up_utils import *

problem = create_up_problem('examples/gw_grip/domain.pddl', 'examples/gw_grip/problem.pddl')

collect_pddl_datapoints(problem, 10_000, 'stam', max_episode_steps=50)
