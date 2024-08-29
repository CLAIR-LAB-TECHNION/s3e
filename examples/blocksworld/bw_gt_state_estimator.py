from unified_planning.model import FNode
from tamp_helpers.pred_utils import entity_is_on_entity

from semantic_state_estimator.state_estimator import PredFnStateEstimator
from semantic_state_estimator.utils.up_utils import get_all_grounded_predicates_for_objects


OBJECT_PDDL_ID_TO_MUJOCO_NAME = {
    # tables
    'brown-table': 'table_brown_top',  # these are specifically the names for the table tops, not the entire table
    'black-table': 'table_black_top',
    'white-table': 'table_white_top',

    # blocks
    'red_block': 'cubeA',
    'green_block': 'cubeB',
    'blue_block': 'cubeC',
    'yellow_block': 'cubeD',
    'cyan_block': 'cubeE',
    'purple_block': 'cubeF',
}

TABLE_PDDL_IDS = [
    pddl_id
    for pddl_id in OBJECT_PDDL_ID_TO_MUJOCO_NAME.keys()
    if pddl_id.endswith('table')
]
BLOCK_PDDL_IDS = [
    pddl_id
    for pddl_id in OBJECT_PDDL_ID_TO_MUJOCO_NAME.keys()
    if pddl_id.endswith('block')
]


def pddl_id_to_mujoco_name(pddl_object_id):
  # the PDDL object ID is converted to lower case to support PDDL's case-insensitivity
  return OBJECT_PDDL_ID_TO_MUJOCO_NAME[pddl_object_id.lower()]


def pddl_id_to_mujoco_entity(object_id, sim):
  # get mujoco identifier from PDDL ID
  object_name = pddl_id_to_mujoco_name(object_id)

  # get the entity associated with this object name
  return sim.get_entity(object_name, 'geom')


class BlocksWorldGTStateEstimator(PredFnStateEstimator):
    def __init__(self, domain, problem, env):
        super().__init__(domain, problem)
        self.env = env

    def robot_gripping(self, block_id, state):
        block_name = pddl_id_to_mujoco_name(block_id)
        return state['grasped_object'] == block_name

    def on_top_of(self, block1_id, block2_id, state):
        if self.robot_gripping(block1_id, state):
            return False

        block1 = pddl_id_to_mujoco_entity(block1_id, self.env._env.sim)
        block2 = pddl_id_to_mujoco_entity(block2_id, self.env._env.sim)

        return entity_is_on_entity(block1, block2, state)

    def on_table(self, block_id, table_id, state):
        if self.robot_gripping(block_id, state):
            return False

        block = pddl_id_to_mujoco_entity(block_id, self.env._env.sim)
        table = pddl_id_to_mujoco_entity(table_id, self.env._env.sim)

        return entity_is_on_entity(block, table, state)

    def clear_on_top(self, block_id, state):
        return not self.robot_gripping(block_id, state) and not any(
            self.on_top_of(other_block_id, block_id, state)
            for other_block_id in BLOCK_PDDL_IDS
        )

    def robot_gripper_empty(self, state):
        return not any(
            self.robot_gripping(block_id, state)
            for block_id in BLOCK_PDDL_IDS
        )
