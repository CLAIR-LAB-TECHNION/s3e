import os

import numpy as np
from gymjoco.episode.specs import ObjectSpec, JointSpec, TaskSpec
from gymjoco.tasks import NullTask
from n_table_blocks_world.n_table_blocks_world import NTableBlocksWorld
from n_table_blocks_world.configurations_and_constants import ROBOTIQ_2F85_BODY
from tamp_helpers.table_sampling import sample_on_table

from gw_gt_state_estimator import pddl_id_to_mujoco_entity


GRASPS_JOINTS = {
    'bottle': 0.6,
    'bread': 0.7,
    'can': 0.7,
    'cereal': 0.0,
    'lemon': 0.6,
    'milk': 0.6,
}

GRASP_OFFSETS = {
    'bottle': 0.18,
    'bread': 0.15,
    'can': 0.15,
    'cereal': 0.2,
    'lemon': 0.15,
    'milk': 0.18,
}


class ObjectPositionTask(NullTask):
    def reset(self, table_ids, obj_ids) -> None:
        self.table_ids = table_ids
        self.obj_ids = obj_ids

        self._sample_obj_poses()

    def _sample_obj_poses(self):
        for obj_id in self.obj_ids:
            obj_geom = self.sim.get_entity(obj_id[:-1], 'geom')
            table_id = np.random.choice(self.table_ids)
            table_pos = sample_on_table(table_id, self.sim, pddl_id_to_mujoco_entity, obj_geom.size[-1])
            obj_pos = np.concatenate([table_pos, [1, 0, 0, 0]])
            self.sim.get_entity(obj_id, 'body').set_state(position=obj_pos)


def get_env():
    ENV_CFG = dict(
        scene=dict(
            resource='housetableworld',
            objects=(
                ObjectSpec('bottle', base_joints=JointSpec('free')),
                ObjectSpec('bread', base_joints=JointSpec('free')),
                ObjectSpec('can', base_joints=JointSpec('free')),
                ObjectSpec('cereal', base_joints=JointSpec('free')),
                ObjectSpec('lemon', base_joints=JointSpec('free')),
                ObjectSpec('milk', base_joints=JointSpec('free')),
            ),
            # render_camera='frontview',
            # renderer_cfg=dict(width=1440, height=1080),
            init_keyframe='home'
        ),
        robot=dict(
            resource='ur5e',
            mount='rethink_stationary',
            privileged_info=True,
            attachments=os.path.join(os.path.dirname(__file__), 'robotiq_2f85', '2f85.xml'),
        ),
        task=TaskSpec(cls=ObjectPositionTask,
                      params=dict(table_ids=['wood-table', 'black-table', 'white-table'],
                                  obj_ids=['bottle/', 'bread/', 'can/', 'cereal/', 'lemon/', 'milk/'])),
    )

    return NTableBlocksWorld(cfg=ENV_CFG, render_mode='rgb_array', ee_name=ROBOTIQ_2F85_BODY,
                             grasp_joints=GRASPS_JOINTS, grasp_offsets=GRASP_OFFSETS)
