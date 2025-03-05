from semantic_state_estimator.skill_executer import SkillExecuter, ActionReturnType

from perform_act import pick_up_item, put_down_item


class RWRobotSkillExecuter(SkillExecuter):
    def __init__(self, env, workspace_path):
        super().__init__(env)

        #TODO load worksapce
        self.workspace = ...

        #TODO load  motion planner and robot
        self.mp = ...
        self.gt = ...
        self.robot = ...
        ...

    def pick_up(self, item_id, table_id):
        #TODO
        #prepare inputs for pick_up_item
        inputs = ...
        pick_up_item(*inputs)

    def put_down(self, item_id, table_id):
        #TODO
        #prepare inputs for put_down_item
        inputs = ...
        put_down_item(*inputs)

    def go_home(self) -> ActionReturnType:
        pass

    def wait(self, n_steps) -> ActionReturnType:
        pass
