import os

from tamp_helpers.table_sampling import sample_free_spot_on_table_for_block
from unified_planning.engines.sequential_simulator import UPSequentialSimulator

from prb_env import PRBEnv, properties, random
from prb_gt_state_estimator import PRBGTStateEstimator
from semantic_state_estimator.skill_executer import SkillExecuter, ActionReturnType
from semantic_state_estimator.utils.up_utils import state_dict_to_up_state



DOMAIN_PTH = os.path.join(os.path.dirname(__file__), "domain.pddl")
with open(DOMAIN_PTH, 'r') as f:
    DOMAIN_STR = f.read()


class PRBSkillExecuter(SkillExecuter):
    def __init__(self, env: PRBEnv):
        super().__init__(env)
        self.env = env

    def reset_env(self):
        super().reset_env()

    def go_home(self) -> ActionReturnType:
        return True, []

    def wait(self, n_steps) -> ActionReturnType:
        return True, []

    def _check_preconditions(self, action_str: str, *args_str) -> bool:
        # create GT state estimator
        problem_str = self.env.get_problem_file_str()
        gt_se = PRBGTStateEstimator(DOMAIN_STR, problem_str)
        problem = gt_se.up_problem  # need problem externally

        # get the current PDDL state
        state = self.env.state
        pddl_state_dict = gt_se(state)
        up_state = state_dict_to_up_state(problem, pddl_state_dict)

        # initialize problem simulator
        problem_sim = UPSequentialSimulator(problem)

        # initialize action instance
        action = problem.action(action_str)
        args = [problem.object(arg) for arg in args_str]
        action_instance = action(*args)

        # test applicability
        is_applicable = problem_sim.is_applicable(up_state, action_instance)

        if not is_applicable:
            # check if the action is applicable in the current state
            pretty_state = '{\n\t' + '\n\t'.join([f'{k}: {v}' for k, v in pddl_state_dict.items()]) + '\n}'
            print(f"Action {action_str}({','.join(args_str)}) is not applicable in state:\n{pretty_state}\nwiggling environment")
            state.wiggle()

        return is_applicable

    def execute_action(self, action, parameters):
        if not self._check_preconditions(action, *parameters):
            return False, []
        return super().execute_action(action, parameters)

    def __change_block_location_to_table(self, block_obj):
        # convenience variable
        table_size = self.env.state.table_size

        # get maximum X value (apparently y is ignored?)
        unit = max(properties['sizes'].values())
        max_x = unit * 2 * table_size

        trial = 0
        fail = True
        while fail and trial < 100:
            fail = False
            block_obj.x = max_x * ((random.randint(0, table_size - 1) / (table_size - 1)) - 1 / 2)
            block_obj.z = 0
            for other_block in self.env.state.objects:
                if block_obj.overlap(other_block):
                    fail = True
                    break
            block_obj.z += block_obj.size
            trial += 1

        return not fail

    def move_from_block_to_table(self, s, b):
        shape_obj = self.env.state.get_object_by_name(s)

        index = self.env.state.objects.index(shape_obj)
        self.env.state.objects.remove(shape_obj)
        suc = self.__change_block_location_to_table(shape_obj)
        self.env.state.objects.insert(index, shape_obj)

        return suc, []

    def move_from_table_to_block(self, s, b):
        s_obj = self.env.state.get_object_by_name(s)
        b_obj = self.env.state.get_object_by_name(b)

        # set x on top of block
        loc_size = s_obj.size / 2
        s_obj.x = random.uniform(b_obj.x - loc_size, b_obj.x + loc_size)

        # set z directly above block
        s_obj.z = s_obj.z + b_obj.z + b_obj.size

        return True, []

    def move_from_block_to_block(self, s, b1, b2):
        self.move_from_table_to_block(s, b2)  # identical implementation
        return True, []

