import numpy as np
from tamp_helpers.table_sampling import sample_free_spot_on_table_for_block

from examples.prb.prb_env import PRBEnv, properties, random
from semantic_state_estimator.skill_executer import SkillExecuter, ActionReturnType


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

    def move_from_block_to_table(self, b1, b2):
        block_obj = self.env.state.get_object_by_name(b1)

        index = self.env.state.objects.index(block_obj)
        self.env.state.objects.remove(block_obj)
        self.__change_block_location_to_table(block_obj)
        self.env.state.objects.insert(index, block_obj)

        return True, []

    def move_from_table_to_block(self, b1, b2):
        b1_obj = self.env.state.get_object_by_name(b1)
        b2_obj = self.env.state.get_object_by_name(b2)

        # set x on top of block
        loc_size = b1_obj.size / 2
        b1_obj.x = random.uniform(b2_obj.x - loc_size, b2_obj.x + loc_size)

        # set z directly above block
        b1_obj.z = b1_obj.z + b2_obj.z

        return True, []

    def move_from_block_to_block(self, b1, b2):
        self.move_from_table_to_block(b1, b2)  # identical implementation
        return True, []

