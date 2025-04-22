import json
import logging
from tinydb import TinyDB, Query

from semantic_state_estimator.skill_executer import SkillExecuter, ActionReturnType

from clair_robotics_stack.ur.lab_setup.manipulation.manipulation_controller_2fg import ManipulationController2FG
from clair_robotics_stack.ur.lab_setup.robot_inteface.robots_metadata import ur5e_2
from clair_robotics_stack.ur.lab_setup.manipulation.robot_with_motion_planning import RobotInterfaceWithMP
from clair_robotics_stack.planning.motion.motion_planner import MotionPlanner
from clair_robotics_stack.planning.motion.geometry_and_transforms import GeometryAndTransforms
import numpy as np


from clair-robotics-stack import *
from perform_act import pick_up_item, put_down_item


class RWRobotSkillExecuter(SkillExecuter):
    """
    RWRobotSkillExecuter implements robot skill execution for pick and place tasks.
    
    This class provides methods to interact with a robot to pick up and put down items
    in a workspace, maintaining a database of item locations and states.
    
    Attributes:
        db (TinyDB): Database for storing workspace information
        workspace: Container for database tables
        mp (MotionPlanner): Motion planner for robot movement
        gt (GeometryAndTransforms): Geometry and transforms utility
        robot (ManipulationController2FG): Robot controller
    """
    def __init__(self, env, workspace_path):
        super().__init__(env)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        #TODO load worksapce
        # Initialize database
        self.db = TinyDB(workspace_path)
        self.workspace = type('', (), {})()  # Create an empty object to attach attributes
        self.workspace.items = self.db.table('items')
        self.workspace.robot = self.db.table('robot')
        self.workspace.board = self.db.table('board_locations')


        #TODO load  motion planner and robot
        self.mp = MotionPlanner()
        self.gt = GeometryAndTransforms.from_motion_planner(self.mp)
        self.robot = ManipulationController2FG.build_from_robot_name_and_ip(ur5e_2["ip"], ur5e_2["name"])


    def pick_up(self, item_id, table_id):
        try:
            #prepare inputs for pick_up_item
            inputs = [self.robot, self.workspace.items.get(doc_id=item_id)]
            pick_up_item(*inputs)
            
            """
            # Update database and state
            item = self.workspace.items.get(doc_id=item_id)
            table = self.workspace.board.get(doc_id=table_id)
            self.workspace.robot.update({'holding': item.get("name")}, self.workspace.robot.get(doc_id=ur5e_2["name"]))
            self.update_item_location(item.get("name"), table.get("name"))
            """
            
        except Exception as e:
            logging.error(f"Failed to pick up item '{item_id}': {e}")
            raise
        
        #TODO: attach the item to the robot config in the motion planner
        #TODO: update the database
        #TODO: save the json file

    def put_down(self, item_id, table_id):
        try:
            #prepare inputs for put_down_item
            inputs = [self.robot, self.workspace.items.get(doc_id=item_id), self.workspace.board.get(doc_id=table_id)]
            put_down_item(*inputs)
            
            """
            # Update database and state
            item = self.workspace.items.get(doc_id=item_id)
            table = self.workspace.board.get(doc_id=table_id)
            self.update_item_location(item.get("name"), table.get("name"))
            """

        except Exception as e:
            logging.error(f"Failed to drop down item '{item_id}': {e}")
            raise
        
        #TODO: detach the item from the the robot config in the motion planner
        #TODO: update the database
        #TODO: save the json file


    def go_home(self) -> ActionReturnType:
        pass

    def wait(self, n_steps) -> ActionReturnType:
        pass


    def update_item_location(self, item_name, location):
        """Update item location in the database."""
        Item = Query()
        self.workspace.items.update({'board_location': location}, Item.name == item_name)

    def get_item(self, item_name):
        """Get item data from the database."""
        Item = Query()
        return self.workspace.items.get(Item.name == item_name)

    def get_board_location(self, item_name):
        """Get board location from the database."""
        Item = Query()
        item = self.workspace.items.get(Item.name == item_name)
        if item:
            return item.get("board_location")
        else:
            logging.error(f"Item '{item_name}' not found in the database.")
            return None

    
def pick_up_item(robot, item):
    """
    Picks up an item based on its metadata from the workspace.

    Parameters:
    - robot: Instance of ManipulationController controlling the robot.
    - item: Dictionary containing item's metadata (name, coordinates, height, etc.).

    Steps:
    1. Extract item's position and height from metadata.
    2. Move to a position above the item safely.
    3. Execute the pick_up function for the final pick-up action.
    """
    # Extract item properties
    robot_name = robot.robot_name
    item_name = item.get("name", "unknown")
    location = item.get("coordinates")
    height = item.get("height")  # Mandatory field
    rz = 0.0  # Default rotation around z-axis

    #Todo: check status of the gripper

    try:

        # Validate mandatory fields
        if not location:
            raise ValueError(f"Item '{item_name}' has no coordinates specified.")
        if height is None:
            raise ValueError(f"Item '{item_name}' is missing a height value. This is critical to avoid collisions.")

        # Calculate the position above the item for a safe approach
        pick_up_point = location[:]
        pick_up_point[2] += height * 1.1  # Lift up for safety
        ee_pose = robot.gt.get_gripper_facing_downwards_6d_pose_robot_frame(robot.robot_name, pick_up_point, rz)

        # Find the IK solution for the above position
        logging.info(f"Calculating IK solution for position above '{item_name}' at [{pick_up_point}].")
        pick_up_config = robot.find_ik_solution(ee_pose)
        if not pick_up_config:
            raise ValueError(f"Failed to find an IK solution for '{item_name}' at {pick_up_point}.")

        # Plan and move to the position above the item
        logging.info(f"Moving to position above '{item_name}' at {pick_up_point}.")
        if not robot.plan_and_moveJ(pick_up_config):
            raise RuntimeError(f"Failed to move to the position above '{item_name}'.")

        # Check if the gripper is engaged and release it if necessary
        if not robot.is_gripper_open():
            logging.info("Gripper is colosed. Releasing it before starting the pick-up operation.")
            robot.release_grasp()

        # Perform the pick-up action
        logging.info(f"Executing pick-up action for '{item_name}'.")
        robot.pick_up(x=location[0], y=location[1], rz=rz, start_height=height)
        robot.motion_planner.remove_object(item_name)
        robot_model = robot.motion_planner.robot_name_mapping[robot_name]
        robot.motion_planner._add_attachments(robot_model, [item_name])  # Todo: change the ur5e_2 to the robot name
        # robot.motion_planner.visualize(window_name="workspace")


    except Exception as e:
        logging.error(f"Failed to pick up item '{item_name}': {e}")
        raise

def put_down_item(robot, item, goal):
    """
    Drops down an item at the specified coordinates in the workspace.

    Parameters:
    - robot: Instance of ManipulationController controlling the robot.
    - item: Dictionary containing item's metadata (name, height, etc.).
    - drop_coordinates: List of [x, y, z] coordinates where the item should be placed.

    Steps:
    1. Verify if the gripper is holding an item. If yes, release it.
    2. Move to a position above the drop point safely.
    3. Execute the drop-down action by moving down and releasing the gripper.
    """
    # Extract item properties
    robot_name = robot.robot_name
    item_name = item.get("name", "unknown")
    height = item.get("height")  # Mandatory field
    rz = 0.0  # Default rotation around z-axis

    # Todo: check status of the gripper

    try:
        # Validate mandatory fields
        if height is None:
            raise ValueError(f"Item '{item_name}' is missing a height value. This is critical to avoid collisions.")

        # Calculate the position above the drop point for a safe approach
        goal[2] = height * 1.1  # Lift up for safety
        ee_pose = robot.gt.get_gripper_facing_downwards_6d_pose_robot_frame(robot.robot_name, goal, rz)

        # Find the IK solution for the above position
        logging.info(f"Calculating IK solution for position above '{item_name}' at [{goal}].")
        drop_config = robot.find_ik_solution(ee_pose)
        if not drop_config:
            raise ValueError(f"Failed to find an IK solution for '{item_name}' at {goal}.")

        # Plan and move to the position above the drop point
        logging.info(f"Moving to position above '{item_name}' at {goal}.")
        if not robot.plan_and_moveJ(drop_config):
            raise RuntimeError(f"Failed to move to the position above '{item_name}'.")

        # Perform the drop-down action
        logging.info(f"Executing drop-down action for '{item_name}'.")
        robot.put_down(x=goal[0], y=goal[1], rz=rz, start_height=height * 1.1)
        robot.motion_planner.add_object_to_world(item_name, item)
        robot_model = robot.motion_planner.robot_name_mapping[robot_name]
        robot.motion_planner.remove_attachments(robot_model)



    except Exception as e:
        logging.error(f"Failed to drop down item '{item_name}': {e}")
        raise