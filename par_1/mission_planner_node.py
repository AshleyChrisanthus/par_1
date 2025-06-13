#!/usr/bin/env python3

from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
import rclpy
from rclpy.node import Node

class MissionPlanner(Node):
    """
    A class to create a ROS2 node that plans and executes a mission by sending
    a sequence of waypoints to the Nav2 stack.
    """
    def __init__(self):
        """
        Initializes the MissionPlanner node.
        """
        super().__init__('mission_planner_node')
        self.navigator = BasicNavigator()

        # --- Your Proven Waypoints ---
        # A list of dictionaries, where each dictionary represents a waypoint.
        self.mission_points = [
            {'x': 2.2, 'y': 0.0, 'z': 0.0,
             'ox': 0.0, 'oy': 0.0, 'oz': 0.0, 'ow': 1.0},        # P1
            # {'x': 2.2, 'y': 0.6, 'z': 0.0,
            #  'ox': 0.0, 'oy': 0.0, 'oz': 0.7071, 'ow': 0.7071},  # P2
            # {'x': 0.0, 'y': 0.6, 'z': 0.0,
            #  'ox': 0.0, 'oy': 0.0, 'oz': 1.0, 'ow': 0.0}         # P3
        ]

    def run_mission(self):
        """
        Executes the main mission logic.
        """
        self.get_logger().info("Waiting for Nav2 to activate...")
        self.navigator.waitUntilNav2Active()
        self.get_logger().info("Nav2 is active. Starting mission.")

        # Convert the dictionary of points to a list of PoseStamped objects
        mission_goals = self._create_goal_poses()

        # --- Execute the Mission Sequentially ---
        for i, goal in enumerate(mission_goals):
            self.get_logger().info(f"--- Sending Goal {i+1}/{len(mission_goals)}: "
                                   f"(x={goal.pose.position.x:.2f}, y={goal.pose.position.y:.2f}) ---")
            self.navigator.goToPose(goal)

            # Wait until the current goal is complete
            while not self.navigator.isTaskComplete():
                # You can get feedback here if you want (e.g., distance remaining)
                # feedback = self.navigator.getFeedback()
                pass

            # Check the result of the completed goal
            result = self.navigator.getResult()
            if result == TaskResult.SUCCEEDED:
                self.get_logger().info(f"Goal {i+1} succeeded!")
            else:
                self.get_logger().error(f"Goal {i+1} failed with status: {result.name}. Aborting mission.")
                # Exit the loop if any goal fails
                break
        
        self.get_logger().info("🎉 Mission Complete! 🎉")

    def _create_goal_poses(self):
        """
        A helper method to convert the internal list of points
        into a list of PoseStamped messages.
        """
        goal_poses = []
        for p in self.mission_points:
            goal_pose = PoseStamped()
            goal_pose.header.frame_id = 'map'
            goal_pose.header.stamp = self.get_clock().now().to_msg()
            goal_pose.pose.position.x = p['x']
            goal_pose.pose.position.y = p['y']
            goal_pose.pose.orientation.x = p['ox']
            goal_pose.pose.orientation.y = p['oy']
            goal_pose.pose.orientation.z = p['oz']
            goal_pose.pose.orientation.w = p['ow']
            goal_poses.append(goal_pose)
        return goal_poses


def main(args=None):
    """
    The main entry point for the script.
    """
    rclpy.init(args=args)

    # Create an instance of the MissionPlanner class and run the mission
    mission_planner = MissionPlanner()
    mission_planner.run_mission()

    # Shutdown the ROS client library
    mission_planner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
