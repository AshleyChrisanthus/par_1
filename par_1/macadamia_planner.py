#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
import tf_transformations

"""
A ROS2 script to command a ROSBot 2 to navigate a pre-defined serpentine path
through a simulated macadamia orchard.

This script uses the Nav2 Simple Commander API to send a sequence of goals.

Prerequisites:
1. A fully functional Nav2 stack is running.
2. A map of the environment (with the 'trees' as obstacles) has been created and is
   loaded by the map_server.
3. AMCL or another localization system is running and the robot is well-localized.
"""

class MissionPlanner(Node):
    def __init__(self):
        super().__init__('macadamia_planner')
        self.navigator = BasicNavigator()

    def run_mission(self):
        """The main entry point for the mission."""
        self.get_logger().info("Waiting for Nav2 to be active...")
        # You can use 'amcl' or 'bt_navigator' depending on your Nav2 setup
        self.navigator.waitUntilNav2Active(localizer='amcl')
        self.get_logger().info("Nav2 is active. Starting mission.")

        # --- Define the sequence of waypoints for the serpentine path ---
        # Frame of reference is the 'map' frame.
        # Orientation is in quaternions (x, y, z, w).
        # We use a helper function to convert from Euler angles (yaw) for clarity.

        # Goal 1: Go to the end of the first lane.
        # Position: x=-0.25 (center of lane 1), y=2.5 (past the last tree)
        # Orientation: Facing forward (90 degrees or pi/2 radians)
        q1 = tf_transformations.quaternion_from_euler(0, 0, 1.57) # (roll, pitch, yaw)
        goal1 = self.create_pose_stamped(-0.25, 2.5, q1)

        # Goal 2: Positioned at the end of the second lane, ready to drive back.
        # Nav2 will execute the 90-degree turn, move sideways, and 90-degree turn.
        # Position: x=0.25 (center of lane 2), y=2.5
        # Orientation: Facing backward (-90 degrees or -pi/2 radians)
        q2 = tf_transformations.quaternion_from_euler(0, 0, -1.57)
        goal2 = self.create_pose_stamped(0.25, 2.5, q2)

        # Goal 3: Drive back down the second lane to the starting line.
        # Position: x=0.25, y=0.0
        # Orientation: Still facing backward (-90 degrees or -pi/2 radians)
        goal3 = self.create_pose_stamped(0.25, 0.0, q2) # Use the same orientation as goal2

        # --- The list of goals to execute ---
        mission_goals = [goal1, goal2, goal3]

        # --- Execute the mission ---
        for i, goal in enumerate(mission_goals):
            self.get_logger().info(f"--- Sending Goal {i+1}/{len(mission_goals)} ---")
            self.navigator.goToPose(goal)

            # Wait until the goal is done
            while not self.navigator.isTaskComplete():
                feedback = self.navigator.getFeedback()
                if feedback:
                    # You can print feedback if you want, e.g., distance remaining
                    pass

            # Check the result
            result = self.navigator.getResult()
            if result == TaskResult.SUCCEEDED:
                self.get_logger().info(f"Goal {i+1} succeeded!")
            elif result == TaskResult.CANCELED:
                self.get_logger().warn(f"Goal {i+1} was canceled! Aborting mission.")
                return
            elif result == TaskResult.FAILED:
                self.get_logger().error(f"Goal {i+1} failed! Aborting mission.")
                return
            else:
                self.get_logger().error("Goal has an invalid return status! Aborting mission.")
                return
        
        self.get_logger().info("🎉 Mission Complete! 🎉")


    def create_pose_stamped(self, x, y, quaternion):
        """Helper function to create a PoseStamped message."""
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        pose.pose.orientation.x = quaternion[0]
        pose.pose.orientation.y = quaternion[1]
        pose.pose.orientation.z = quaternion[2]
        pose.pose.orientation.w = quaternion[3]
        return pose

def main(args=None):
    rclpy.init(args=args)

    # You might need to install this library: pip install transforms3d
    # Or for ROS2 Humble: sudo apt install ros-humble-tf-transformations
    # It's often included with a desktop install.
    
    mission_planner = MissionPlanner()
    mission_planner.run_mission()

    # Shutdown
    mission_planner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
