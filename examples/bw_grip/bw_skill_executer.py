from collections import defaultdict
import numpy as np
from motion_planning.motion_executer import NTableBlocksWorldMotionExecuter
from tamp_helpers.table_sampling import sample_free_spot_on_table_for_block

from semantic_state_estimator.skill_executer import SkillExecuter, ActionReturnType

UR5_HOME_CONFIG = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0])


OFFSETS_PICK = defaultdict(lambda: 0.22)


OFFSETS_PLACE = defaultdict(lambda: 0.28)


class BlocksWorldSkillExecuter(SkillExecuter):
    def __init__(self, env, pddl_id_to_mujoco_name, pddl_id_to_mujoco_entity, pickable_item_ids, home_config=None):
        super().__init__(env)
        self.motion_exec = NTableBlocksWorldMotionExecuter(env)
        self.pddl_id_to_mujoco_name = pddl_id_to_mujoco_name
        self.pddl_id_to_mujoco_entity = pddl_id_to_mujoco_entity
        self.pickable_item_ids = pickable_item_ids
        self.home_config = home_config or UR5_HOME_CONFIG

    def reset_env(self):
        super().reset_env()
        self.motion_exec.update_blocks_positions()

    def go_home(self) -> ActionReturnType:
        return self.motion_exec.move_to(self.home_config, render_freq=self.render_freq)

    def wait(self, n_steps) -> ActionReturnType:
        return self.motion_exec.wait(n_steps, render_freq=self.render_freq)

    def pick_up(self, item_id, table_id):
        # NOTE: we accept the table_id parameter but don't use it.
        # we do not need this parameter because we already have the position of each block from the simulator.
        # we keep the argument for compatibility with the PDDL action.

        # get block identifier in MuJoCo
        item_name = self.pddl_id_to_mujoco_name(item_id)

        # move end-effector above the block
        offset = OFFSETS_PICK[item_name]
        move_suc, move_frames = self.motion_exec.move_above_block(item_name, offset=offset, render_freq=self.render_freq)

        # activate the gripper to grasp the object
        grasp_suc, grasp_frames = self.motion_exec.activate_grasp(render_freq=self.render_freq)

        return move_suc and grasp_suc, np.concatenate([move_frames, grasp_frames])

    def put_down(self, block_id, table_id):
        # sample a collision free spot on the table
        item_name = self.pddl_id_to_mujoco_name(block_id)
        offset = OFFSETS_PLACE[item_name]
        table_pos = sample_free_spot_on_table_for_block(table_id, block_id, self.env._env.sim,
                                                        self.pddl_id_to_mujoco_entity,
                                                        z_offset=offset,
                                                        ids=self.pickable_item_ids)

        # move end-effector with the block above the sampled spot on the table.
        move_suc, move_frames = self.motion_exec.move_to_pose(table_pos, render_freq=self.render_freq)

        # deactivate the girpper to release the object
        grasp_suc, grasp_frames = self.motion_exec.deactivate_grasp(render_freq=self.render_freq)

        return move_suc and grasp_suc, np.concatenate([move_frames, grasp_frames])

    def stack(self, block1_id, block2_id):
        # get identifier in MuJoCo for the block on which to stack
        block2_name = self.pddl_id_to_mujoco_name(block2_id)
        offset = OFFSETS_PLACE[block2_name]

        # move end-effector with the destination block
        move_suc, move_frames = self.motion_exec.move_above_block(block2_name, offset=offset, render_freq=self.render_freq)

        # deactivate the gripper to release the object
        grasp_suc, grasp_frames = self.motion_exec.deactivate_grasp(render_freq=self.render_freq)

        return move_suc and grasp_suc, np.concatenate([move_frames, grasp_frames])

    def unstack(self, block1_id, block2_id):
        # we can just pick up the block from its location without considering the other block or the table it is on.
        return self.pick_up(block1_id, None)  # remember, we are ignoring the table ID in the pick_up method.
