from PIL.Image import Image
from unified_planning.model import FNode
from tamp_helpers.pred_utils import entity_is_on_entity

from semantic_state_estimator.state_estimator import PredFnStateEstimator


OBJECT_PDDL_ID_TO_MUJOCO_NAME = {
    # tables
    'wood-table': 'table_wood_top',  # these are specifically the names for the table tops, not the entire table
    'white-table': 'table_white_top',
    'black-table': 'table_black_top',

    # items
    'green-bottle': 'bottle',
    'loaf-of-bread': 'bread',
    'red-can-of-soda': 'can',
    'red-box-of-cereal': 'cereal',
    'lemon': 'lemon',
    'milk-carton': 'milk',
}

TABLE_PDDL_IDS = [
    pddl_id
    for pddl_id in OBJECT_PDDL_ID_TO_MUJOCO_NAME.keys()
    if pddl_id.endswith('table')
]
ITEM_PDDL_IDS = [
    pddl_id
    for pddl_id in OBJECT_PDDL_ID_TO_MUJOCO_NAME.keys()
    if not pddl_id.endswith('table')
]


def pddl_id_to_mujoco_name(pddl_object_id):
  # the PDDL object ID is converted to lower case to support PDDL's case-insensitivity
  return OBJECT_PDDL_ID_TO_MUJOCO_NAME[pddl_object_id.lower()]


def pddl_id_to_mujoco_entity(object_id, sim):
  # get mujoco identifier from PDDL ID
  object_name = pddl_id_to_mujoco_name(object_id)

  # get the entity associated with this object name
  return sim.get_entity(object_name, 'geom')


class GroceriesWorldGTStateEstimator(PredFnStateEstimator):
    def __init__(self, domain, problem, env):
        super().__init__(domain, problem)
        self.env = env

    def robot_gripping(self, item_id, state):
        item_name = pddl_id_to_mujoco_name(item_id)
        return state['grasped_object'] == item_name

    def on_table(self, item_id, table_id, state):
        if self.robot_gripping(item_id, state):
            return False

        block = pddl_id_to_mujoco_entity(item_id, self.env._env.sim)
        table = pddl_id_to_mujoco_entity(table_id, self.env._env.sim)

        return entity_is_on_entity(block, table, state)

    def robot_gripper_empty(self, state):
        return not any(
            self.robot_gripping(item_id, state)
            for item_id in ITEM_PDDL_IDS
        )
