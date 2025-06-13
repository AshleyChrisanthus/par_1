#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import tf_transformations
import math

class MissionPlanner(Node):
    def __init__(self):
        super().__init__('macadamia_planner')

        # --- Publishers and Subscribers ---
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_subscriber = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.scan_subscriber = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)

        # --- Robot State Variables ---
        self.current_pose = None
        self.current_yaw = 0.0
        self.obstacle_detected = False

        # --- Mission Plan ---
        # Define your waypoints based on your drawing (P1, P2, P3, etc.)
        # These are (x, y) coordinates from the odometry frame.
        self.waypoints = [
            (1.0, 0.0),   # P1: Go forward 2 meters
            (1.0, 0.5),   # P2: Go left 1 meter
            (0.0, 0.5)    # P3: Go back to the line of the starting X
            # Add more waypoints for your full path
        ]
        self.current_waypoint_index = 0
        
        # --- Control Parameters ---
        self.distance_threshold = 0.15  # How close to get to a waypoint (meters)
        self.max_linear_speed = 0.15   # m/s
        self.max_angular_speed = 0.5  # rad/s
        
        # --- Safety Parameters ---
        self.obstacle_distance_threshold = 0.1 # meters

        # --- Main loop ---
        self.timer = self.create_timer(0.1, self.run_mission_step) # Run at 10 Hz

    def odom_callback(self, msg: Odometry):
        """Updates the robot's current position and orientation."""
        self.current_pose = msg.pose.pose
        orientation_q = self.current_pose.orientation
        _, _, self.current_yaw = tf_transformations.euler_from_quaternion(
            [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])

    def scan_callback(self, msg: LaserScan):
        """Checks for obstacles directly in front of the robot."""
        # This is a simple check. We look at a 60-degree arc in front.
        # The exact range of indices depends on your Lidar's field of view.
        # For a 360-degree Lidar, the front is around index 0.
        num_ranges = len(msg.ranges)
        front_arc_degrees = 60
        arc_indices = int((front_arc_degrees / 360.0) * num_ranges)
        
        # Check indices at the beginning and end of the array (for 360 Lidar)
        front_ranges = msg.ranges[:arc_indices//2] + msg.ranges[-arc_indices//2:]
        
        # Filter out 'inf' or 0 values
        valid_ranges = [r for r in front_ranges if r > 0.01]
        
        if valid_ranges and min(valid_ranges) < self.obstacle_distance_threshold:
            self.obstacle_detected = True
        else:
            self.obstacle_detected = False

    def run_mission_step(self):
        """The main logic loop that controls the robot's state."""
        if self.current_pose is None:
            self.get_logger().info("Waiting for odometry data...")
            return

        # --- SAFETY FIRST: Check for obstacles ---
        if self.obstacle_detected:
            self.get_logger().warn("Obstacle detected! Stopping.")
            self.stop_robot()
            return

        # --- Check if the mission is complete ---
        if self.current_waypoint_index >= len(self.waypoints):
            self.get_logger().info("🎉 Mission Complete! 🎉")
            self.stop_robot()
            self.timer.cancel() # Stop the timer loop
            return

        # --- Navigate to the current waypoint ---
        target_x, target_y = self.waypoints[self.current_waypoint_index]
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y

        distance_to_target = math.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)

        # --- If close enough, move to the next waypoint ---
        if distance_to_target < self.distance_threshold:
            self.get_logger().info(f"Waypoint {self.current_waypoint_index} reached!")
            self.current_waypoint_index += 1
            self.stop_robot() # Stop briefly before heading to the next goal
            return

        # --- If not there yet, calculate movement command ---
        # This is a simple P-controller (Proportional Controller)
        angle_to_target = math.atan2(target_y - current_y, target_x - current_x)
        angle_error = self.normalize_angle(angle_to_target - self.current_yaw)

        # Control logic
        twist_msg = Twist()
        # If we are not pointing at the goal, turn first.
        if abs(angle_error) > 0.1: # 0.1 radians is about 6 degrees
            twist_msg.angular.z = self.max_angular_speed * (1 if angle_error > 0 else -1)
        # If we are pointing at the goal, drive forward.
        else:
            twist_msg.linear.x = self.max_linear_speed
        
        # Publish the command
        self.cmd_vel_publisher.publish(twist_msg)

    def stop_robot(self):
        """Publishes a zero-velocity Twist message to stop the robot."""
        twist_msg = Twist()
        self.cmd_vel_publisher.publish(twist_msg)

    def normalize_angle(self, angle):
        """Normalize an angle to be between -pi and pi."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle


def main(args=None):
    rclpy.init(args=args)
    mission_controller = MissionPlanner()
    rclpy.spin(mission_controller)
    mission_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
