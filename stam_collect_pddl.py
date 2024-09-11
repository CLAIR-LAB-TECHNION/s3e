from semantic_state_estimator.utils.up_utils import *

problem = create_up_problem('examples/real_robot/domain.pddl', 'examples/real_robot/problem.pddl')

collect_pddl_datapoints(problem, 10_000, 'stam2', max_episode_steps=50)
