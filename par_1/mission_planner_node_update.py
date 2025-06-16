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
        super().__init__('mission_planner_node_update')

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

        # --- Mission Plans ---
        self.mission_plan = [
            {'action': 'drive', 'value': 2.2},
            {'action': 'turn', 'value': 90},
            {'action': 'drive', 'value': 0.6},
            {'action': 'turn', 'value': 90},
            {'action': 'drive', 'value': 2.2}
        ]
        
        ## NEW: Define the side-step maneuver. Tune these distances!
        sideways_distance = 0.5  # How far to move sideways (meters)
        forward_distance = 0.8   # How far to drive forward to pass the obstacle
        self.avoidance_plan = [
            {'action': 'turn', 'value': -90}, # Turn right
            {'action': 'drive', 'value': sideways_distance},
            {'action': 'turn', 'value': 90},  # Turn back left (to face original direction)
            {'action': 'drive', 'value': forward_distance},
            {'action': 'turn', 'value': 90},  # Turn left
            {'action': 'drive', 'value': sideways_distance},
            {'action': 'turn', 'value': -90} # Turn back right to realign
        ]

        ## NEW: State machine variables
        self.current_state = 'NAVIGATING' # Can be 'NAVIGATING' or 'AVOIDING'
        self.current_mission_action_index = 0
        self.current_avoidance_action_index = 0
        self.is_executing_action = False
        self.remaining_drive_distance = 0.0

        # --- Robot State ---
        self.start_pose = None
        self.current_pose = None
        self.current_yaw = 0.0

        # --- Control & Safety Parameters ---
        self.linear_speed = 0.2
        self.angular_speed = 0.4
        self.obstacle_distance = 0.35 # Slightly increased for more reaction time
        self.tolerance = 0.05

        # --- Main loop ---
        self.timer = self.create_timer(0.1, self.run_mission_step)

    def _quaternion_to_yaw(self, q: Quaternion) -> float:
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def odom_callback(self, msg: Odometry):
        self.current_pose = msg.pose.pose
        self.current_yaw = self._quaternion_to_yaw(self.current_pose.orientation)

    def scan_callback(self, msg: LaserScan):
        num_ranges = len(msg.ranges)
        front_arc_indices = int((40 / 360.0) * num_ranges)
        front_ranges = msg.ranges[:front_arc_indices//2] + msg.ranges[-front_arc_indices//2:]
        valid_ranges = [r for r in front_ranges if r > 0.01 and r < self.obstacle_distance]
        
        ## NEW: State-change logic
        if valid_ranges and self.current_state == 'NAVIGATING':
            # Only trigger avoidance if we are driving forward
            if self.mission_plan[self.current_mission_action_index]['action'] == 'drive':
                self.get_logger().warn("Obstacle detected! Switching to AVOIDING state.")
                self.current_state = 'AVOIDING'
                
                # Calculate and save remaining distance
                distance_traveled = math.sqrt((self.current_pose.position.x - self.start_pose.position.x)**2 + 
                                              (self.current_pose.position.y - self.start_pose.position.y)**2)
                total_distance = self.mission_plan[self.current_mission_action_index]['value']
                self.remaining_drive_distance = total_distance - distance_traveled
                if self.remaining_drive_distance < 0:
                    self.remaining_drive_distance = 0
                
                # Reset action state to start the avoidance maneuver
                self.stop_robot()
                self.is_executing_action = False
                self.current_avoidance_action_index = 0

    def run_mission_step(self):
        if self.current_pose is None:
            self.get_logger().info("Waiting for odometry...")
            return
        
        ## NEW: State-based dispatcher
        if self.current_state == 'NAVIGATING':
            self._handle_navigation_state()
        elif self.current_state == 'AVOIDING':
            self._handle_avoidance_state()

    ## NEW: All original mission logic moved here
    def _handle_navigation_state(self):
        if self.current_mission_action_index >= len(self.mission_plan):
            self.get_logger().info("🎉 Mission Complete! 🎉")
            self.stop_robot()
            self.timer.cancel()
            return
        
        if not self.is_executing_action:
            self.start_new_action(self.mission_plan, self.current_mission_action_index)
            return
            
        self._execute_action(self.mission_plan, self.current_mission_action_index)

    ## NEW: Logic for the avoidance maneuver
    def _handle_avoidance_state(self):
        if self.current_avoidance_action_index >= len(self.avoidance_plan):
            self.get_logger().info("Avoidance maneuver complete. Switching back to NAVIGATING.")
            self.current_state = 'NAVIGATING'
            
            # Update the original plan with the remaining distance
            self.mission_plan[self.current_mission_action_index]['value'] = self.remaining_drive_distance
            
            self.is_executing_action = False # Start the resumed action
            return

        if not self.is_executing_action:
            self.start_new_action(self.avoidance_plan, self.current_avoidance_action_index)
            return

        self._execute_action(self.avoidance_plan, self.current_avoidance_action_index)

    ## NEW: Generalized function to execute an action from any plan
    def _execute_action(self, plan, action_index):
        action = plan[action_index]
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
                turn_direction = 1.0 if action['value'] > 0 else -1.0
                twist_msg.angular.z = self.angular_speed * turn_direction
            else:
                self.action_completed()

        self.cmd_vel_publisher.publish(twist_msg)

    def start_new_action(self, plan, action_index):
        self.is_executing_action = True
        self.start_pose = self.current_pose
        action = plan[action_index]
        self.get_logger().info(f"[{self.current_state}] Starting action: {action['action']} {action['value']}")

    def action_completed(self):
        self.get_logger().info(f"[{self.current_state}] Action completed.")
        self.stop_robot()
        self.is_executing_action = False
        
        ## NEW: Increment the correct index based on state
        if self.current_state == 'NAVIGATING':
            self.current_mission_action_index += 1
        elif self.current_state == 'AVOIDING':
            self.current_avoidance_action_index += 1
        
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
