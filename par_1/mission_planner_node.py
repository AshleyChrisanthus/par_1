#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Quaternion
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import math

class SelfContainedNavigator(Node):
    def __init__(self):
        super().__init__('mission_planner_node')

        # --- Publishers and Subscribers ---
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10)
        self.odom_subscriber = self.create_subscription(
            Odometry, '/rosbot_base_controller/odom', self.odom_callback, odom_qos)
        self.scan_subscriber = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)

        # --- Mission Plan: A list of simple actions ---
        self.mission_plan = [
            {'action': 'drive', 'value': 2.2},   # Drive forward 2.2m
            {'action': 'turn', 'value': 90},     # Turn left 90 degrees
            {'action': 'drive', 'value': 0.6},   # Drive forward 0.6m
            {'action': 'turn', 'value': 90},     # Turn left 90 degrees
            {'action': 'drive', 'value': 2.2}    # Drive forward 2.2m
        ]
        self.current_action_index = 0
        self.is_executing_action = False

        # --- Robot State ---
        self.start_pose = None
        self.current_pose = None
        self.current_yaw = 0.0
        self.obstacle_detected = False

        # --- Control & Safety Parameters ---
        self.linear_speed = 0.2  # m/s
        self.angular_speed = 0.4 # rad/s
        self.obstacle_distance = 0.3 # meters
        self.tolerance = 0.05    # meters for distance, radians for angle

        # --- Main loop ---
        self.timer = self.create_timer(0.1, self.run_mission_step)

    def _quaternion_to_yaw(self, q: Quaternion) -> float:
        """
        Converts a geometry_msgs/Quaternion to a yaw angle in radians.
        This is the math that replaces the external libraries.
        """
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

    def odom_callback(self, msg: Odometry):
        self.get_logger().info("Odom callback")
        self.current_pose = msg.pose.pose
        self.current_yaw = self._quaternion_to_yaw(self.current_pose.orientation)

    def scan_callback(self, msg: LaserScan):
        # Simple check in a 40-degree frontal arc
        self.get_logger().info("Scan callback")
        num_ranges = len(msg.ranges)
        front_arc_indices = int((40 / 360.0) * num_ranges)
        front_ranges = msg.ranges[:front_arc_indices//2] + msg.ranges[-front_arc_indices//2:]
        valid_ranges = [r for r in front_ranges if r > 0.01]
        self.obstacle_detected = valid_ranges and min(valid_ranges) < self.obstacle_distance

    def run_mission_step(self):
        if self.current_pose is None:
            self.get_logger().info("Waiting for odometry...")
            return

        if self.obstacle_detected:
            self.get_logger().warn("Obstacle detected! Stopping.")
            self.stop_robot()
            return

        if self.current_action_index >= len(self.mission_plan):
            self.get_logger().info("🎉 Mission Complete! 🎉")
            self.stop_robot()
            self.timer.cancel()
            return
        
        if not self.is_executing_action:
            self.start_new_action()
            return

        # --- Continue executing the current action ---
        action = self.mission_plan[self.current_action_index]
        twist_msg = Twist()

        if action['action'] == 'drive':
            distance_traveled = math.sqrt((self.current_pose.position.x - self.start_pose.position.x)**2 + 
                                          (self.current_pose.position.y - self.start_pose.position.y)**2)
            if distance_traveled < action['value'] - self.tolerance:
                twist_msg.linear.x = self.linear_speed
            else:
                self.action_completed()
        
        elif action['action'] == 'turn':
            start_yaw = self._quaternion_to_yaw(self.start_pose.orientation)
            angle_turned = abs(self.normalize_angle(self.current_yaw - start_yaw))
            target_angle_rad = math.radians(action['value'])
            
            if angle_turned < target_angle_rad - self.tolerance:
                # Turn left for positive degrees, right for negative
                turn_direction = 1.0 if action['value'] > 0 else -1.0
                twist_msg.angular.z = self.angular_speed * turn_direction
            else:
                self.action_completed()

        self.cmd_vel_publisher.publish(twist_msg)

    def start_new_action(self):
        self.is_executing_action = True
        self.start_pose = self.current_pose
        action = self.mission_plan[self.current_action_index]
        self.get_logger().info(f"Starting action {self.current_action_index + 1}: {action['action']} {action['value']}")

    def action_completed(self):
        self.get_logger().info("Action completed.")
        self.stop_robot()
        self.is_executing_action = False
        self.current_action_index += 1
        # Brief pause before starting the next action
        self.timer.reset()

    def stop_robot(self):
        self.cmd_vel_publisher.publish(Twist())

    def normalize_angle(self, angle):
        while angle > math.pi: angle -= 2.0 * math.pi
        while angle < -math.pi: angle += 2.0 * math.pi
        return angle

def main(args=None):
    rclpy.init(args=args)
    navigator = SelfContainedNavigator()
    rclpy.spin(navigator)
    navigator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
