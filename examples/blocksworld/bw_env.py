from copy import copy

from bw_gt_state_estimator import pddl_id_to_mujoco_entity

import numpy as np
from gymjoco.tasks import NullTask
from gymjoco.episode.specs import ObjectSpec, JointSpec, TaskSpec
from tamp_helpers.table_sampling import sample_on_table
from n_table_blocks_world.n_table_blocks_world import NTableBlocksWorld


class BlocksPositionTask(NullTask):
    def reset(self, table_pddl_ids, obj_ids) -> None:
        self.table_pddl_ids = table_pddl_ids
        self.obj_ids = obj_ids

        self._sample_obj_poses()

    def _sample_obj_poses(self):
        objs_remaining = copy(self.obj_ids)
        while objs_remaining:
            stack_size = np.random.randint(1, len(objs_remaining) + 1)
            table_id = np.random.choice(self.table_pddl_ids)
            table_pos = sample_on_table(table_id, self.sim, pddl_id_to_mujoco_entity)

            for i in range(stack_size):
                random_idx = np.random.randint(len(objs_remaining))
                obj_id = objs_remaining.pop(random_idx)
                obj_pos = table_pos + np.array([0, 0, i * 0.1])
                obj_pos = np.concatenate([obj_pos, [1, 0, 0, 0]])
                self.sim.get_entity(obj_id, 'body').set_state(position=obj_pos)

def get_env():
    ENV_CFG = dict(
        scene=dict(
            resource='3tableworld',
            objects=(
                ObjectSpec('cubeA', base_joints=JointSpec('free')),
                ObjectSpec('cubeB', base_joints=JointSpec('free')),
                ObjectSpec('cubeC', base_joints=JointSpec('free')),
                ObjectSpec('cubeD', base_joints=JointSpec('free')),
                ObjectSpec('cubeE', base_joints=JointSpec('free')),
                ObjectSpec('cubeF', base_joints=JointSpec('free')),
            ),
            render_camera='frontview',
            # renderer_cfg=dict(width=1440, height=1080),
            init_keyframe='home'
        ),
        robot=dict(
            resource='ur5e',
            mount='rethink_stationary',
            privileged_info=True,
            attachments=['adhesive_gripper'],
            # cameras=['frontview', 'rightangleview', 'leftangleview', 'rightsideview', 'leftsideview', 'birdview'],
        ),
        task=TaskSpec(cls=BlocksPositionTask,
                      params=dict(table_pddl_ids=['brown-table', 'black-table', 'white-table'],
                                  obj_ids=['cubeA/', 'cubeB/', 'cubeC/', 'cubeD/', 'cubeE/', 'cubeF/'])),
    )

    return NTableBlocksWorld(cfg=ENV_CFG, render_mode='rgb_array')



