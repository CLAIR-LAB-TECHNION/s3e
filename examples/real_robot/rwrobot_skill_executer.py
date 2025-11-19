import json
import logging
import random
from tinydb import TinyDB, Query
from semantic_state_estimator.skill_executer import SkillExecuter, ActionReturnType
from clair_robotics_stack.ur.lab_setup.manipulation.manipulation_controller_2fg import ManipulationController2FG
from clair_robotics_stack.ur.lab_setup.robot_inteface.robots_metadata import ur5e_2
from clair_robotics_stack.ur.lab_setup.manipulation.robot_with_motion_planning import RobotInterfaceWithMP
from clair_robotics_stack.planning.motion.motion_planner import MotionPlanner
from clair_robotics_stack.planning.motion.geometry_and_transforms import GeometryAndTransforms
import numpy as np


from clair_robotics_stack import *
#from perform_act import pick_up_item, put_down_item


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
    def __init__(self, env):
        super().__init__(env)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Initialize database
        workspace_path = "workspace.json"
        self.db = TinyDB(workspace_path)
        self.workspace = type('', (), {})()  # Create an empty object to attach attributes
        self.workspace.items = self.db.table('items')
        self.workspace.robot = self.db.table('robot')
        self.workspace.board = self.db.table('board_locations')

        self.robot = ManipulationController2FG.build_from_robot_name_and_ip(ur5e_2["ip"], ur5e_2["name"])

        # self.robot.move_home(speed=2, acceleration=1.5)

        #TODO: add items to the world need to thing of smart way to do this

        # items = self.workspace.items.all()
        # for item in items:
        #     dict_item = dict(item)
        #     item_id = item.get("name")
        #     if item_id and item.get("board_location"):
        #         self.robot.motion_planner.add_object_to_world(item_id, dict_item)
        #         logging.info(f"Added item '{item_id}' to the robot workspace.")


    def pick_up(self, item_name, table_id=None):
        item_id = self.get_item_id_by_name(item_name)

        try:
            pick_up_item(self.robot, self.workspace.items.get(doc_id=item_id))

            # Update the database
            self.update_database_after_action("pickup", item_id)

        except Exception as e:
            logging.error(f"Failed to pick up item '{item_id}': {e}")
            raise


    def put_down(self, item_name, table_id):
        item_id = self.get_item_id_by_name(item_name)
        try:
            failures = 0
            while failures < 5:
                try:
                    # Find an available zone on the specified board
                    zone_id = self.find_available_zone(table_id)
                    if not zone_id:
                        raise ValueError(f"No available zones on {table_id} board")

                    # Get zone coordinates
                    zone = self.workspace.board.get(doc_id=zone_id)
                    item = self.workspace.items.get(doc_id=item_id)

                    # Execute put down operation
                    put_down_item(self.robot, item, zone["coordinates"])
                except ValueError:
                    failures += 1
                    logging.warning(f"Failed to find an available zone. Attempt {failures}/5.")
                    if failures == 5:
                        raise ValueError("Failed to find an available zone after 5 attempts.")

                break

            # Update the database
            self.update_database_after_action("putdown", item_id, zone_id)
        except Exception as e:
            logging.error(f"Failed to put down item {item_id}: {e}")
            raise


    def go_home(self):
        self.robot.move_home(speed=2, acceleration=2)
        pass

    def wait(self, n_steps) -> ActionReturnType:
        pass

    def update_item_location(self, item_name, location):
        """Update item location in the database."""
        Item = Query()
        self.workspace.items.update({'board_location': location}, Item.name == item_name)

    def get_item_by_name(self, item_name):
        """Get item data from the database."""
        Item = Query()
        return self.workspace.items.get(Item.name == item_name)

    def get_item_id_by_name(self, item_name):
        item = self.get_item_by_name(item_name)
        return item.doc_id

    def get_board_location(self, item_name):
        """Get board location from the database."""
        Item = Query()
        item = self.workspace.items.get(Item.name == item_name)
        if item:
            return item.get("board_location")
        else:
            logging.error(f"Item '{item_name}' not found in the database.")
            return None

    def find_available_zone(self, board_name):
        """
        Find an available (unoccupied) zone on a specified board.

        Parameters:
            board_name (str): Name/color of the board (e.g., "white", "blue")

        Returns:
            int: doc_id of the available zone, or None if no zone is available
        """
        # Query for locations on the specified board that are not occupied
        Location = Query()
        available_locations = self.workspace.board.search(
            (Location.name == board_name) & (Location.occupied == False)
        )

        if not available_locations:
            logging.warning(f"No available zones found on the {board_name} board")
            return None

        # Return the doc_id of the first available location
        # You could implement different strategies here (closest to robot, etc.)
        return random.choice(available_locations[:]).doc_id


    def update_database_after_action(self, action_type, item_id, zone_id=None):
        """
        Update database state after robot actions.

        Parameters:
            action_type (str): Either "pickup" or "putdown"
            item_id (int): The doc_id of the item being manipulated
            zone_id (int, optional): The doc_id of the zone for putdown
        """
        item = self.workspace.items.get(doc_id=item_id)

        if action_type.lower() == "pickup":
            # 1. Get the zone the item was on
            board_location = item.get('board_location', {})
            if board_location:
                # Find the zone the item was on by name and zone_id
                Location = Query()
                zones = self.workspace.board.search(
                    (Location.name == board_location.get('name')) &
                    (Location.zone_id == board_location.get('zone_id'))
                )
                if zones:
                    # Mark zone as unoccupied
                    self.workspace.board.update({'occupied': False}, doc_ids=[zones[0].doc_id])

            # 2. Update robot state to holding this item
            self.workspace.robot.update({'holding': item.get('name')}, doc_ids=[1])

            # 3. Update item to show it's not on any board
            self.workspace.items.update({'board_location': None}, doc_ids=[item_id])

        elif action_type.lower() == "putdown":
            if not zone_id:
                logging.error("Zone ID required for putdown database update")
                return

            # 1. Get zone information
            zone = self.workspace.board.get(doc_id=zone_id)
            zone_id = int(zone_id)
            item_id = int(item_id)

            # 2. Mark zone as occupied
            self.workspace.board.update({'occupied': True}, doc_ids=[zone_id])

            # 3. Update item location
            self.workspace.items.update(
            {'board_location': {'name': zone['name'], 'zone_id': zone['zone_id']},
             'coordinates': zone['coordinates']}, doc_ids=[item_id])

            # 4. Update robot state (no longer holding item)
            self.workspace.robot.update({'holding': None}, doc_ids=[1])

        # Save changes to database
        logging.info(f"Database updated after {action_type} operation for item {item_id}")


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
    rz = 0.0 if item.doc_id == 4 else np.pi / 2  # Default rotation around z-axis

    try:
        # Validate mandatory fields
        if not location:
            raise ValueError(f"Item '{item_name}' has no coordinates specified.")
        if height is None:
            raise ValueError(f"Item '{item_name}' is missing a height value. This is critical to avoid collisions.")

        # Calculate the position above the item for a safe approach
        pick_up_point = location[:]
        pick_up_point[2] += height + 0.1  # Lift up for safety
        ee_pose = robot.gt.get_gripper_facing_downwards_6d_pose_robot_frame(robot.robot_name, pick_up_point, rz)

        # Find the IK solution for the above position
        logging.info(f"Calculating IK solution for position above '{item_name}' at [{pick_up_point}].")
        pick_up_config = robot.find_ik_solution(ee_pose, max_tries=20)
        if not pick_up_config:
            raise ValueError(f"Failed to find an IK solution for '{item_name}' at {pick_up_point}.")

        # Plan and move to the position above the item
        logging.info(f"Moving to position above '{item_name}' at {pick_up_point}.")
        if not robot.plan_and_moveJ(pick_up_config):
            raise RuntimeError(f"Failed to move to the position above '{item_name}'.")

        # Check if the gripper is engaged and release it if necessary
        # if not robot.is_gripper_open():
        #     logging.info("Gripper is colosed. Releasing it before starting the pick-up operation.")
        #     robot.release_grasp()

        # Perform the pick-up action
        logging.info(f"Executing pick-up action for '{item_name}'.")
        robot.pick_up(x=location[0], y=location[1], rz=rz, start_height=height * 1.2)
        robot.moveL_relative([0, 0, 0.2])
        robot.motion_planner.remove_object(item_name)
        robot_model = robot.motion_planner.robot_name_mapping[robot_name]
        robot.motion_planner.add_attachments(robot_model, [item_name], item)

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
    rz = 0.0 if item.doc_id == 4 else np.pi / 2  # Default rotation around z-axis

    try:
        # Validate mandatory fields
        if height is None:
            raise ValueError(f"Item '{item_name}' is missing a height value. This is critical to avoid collisions.")

        # Calculate the position above the drop point for a safe approach
        goal[2] = height + 0.3 # Lift up for safety
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
        robot.moveL_relative([0, 0, 0.08])

        #update item coordinates in the database
        item["coordinates"] = goal

        # Update the database with the new coordinates
        robot.motion_planner.add_object_to_world(item_name, item)
        robot_model = robot.motion_planner.robot_name_mapping[robot_name]
        robot.motion_planner.remove_attachments(robot_model)

    except Exception as e:
        logging.error(f"Failed to drop down item '{item_name}': {e}")
        raise