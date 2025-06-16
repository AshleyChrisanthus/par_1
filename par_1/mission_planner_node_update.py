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
        
        # --- NEW: Your Requested Avoidance Plan ---
        # ** TUNE THESE VALUES BASED ON YOUR OBSTACLE SIZE **
        self.avoidance_plan = [
            {'action': 'turn', 'value': -45}, # 1. Turn right 45 deg
            {'action': 'drive', 'value': 0.5}, # 2. Drive diagonally forward
            {'action': 'turn', 'value': 90},  # 3. Turn left 90 deg (now parallel to original path)
            {'action': 'drive', 'value': 0.8}, # 4. Drive forward past the obstacle
            {'action': 'turn', 'value': -45}  # 5. Turn right 45 deg to be aligned with original path again
        ]

        # --- State Machine ---
        self.current_state = 'NAVIGATING' # 'NAVIGATING' or 'AVOIDING'
        self.active_plan = self.mission_plan
        self.current_action_index = 0
        self.is_executing_action = False
        self.obstacle_detected = False
        self.remaining_drive_distance = 0.0

        # --- Robot State ---
        self.start_pose = None
        self.current_pose = None
        self.current_yaw = 0.0

        # --- Control & Safety Parameters ---
        self.linear_speed = 0.2
        self.angular_speed = 0.4
        self.obstacle_distance = 0.35
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
        self.obstacle_detected = bool(valid_ranges)

    def run_mission_step(self):
        if self.current_pose is None:
            self.get_logger().info("Waiting for odometry...")
            return

        # --- THE FIX: Immediate state transition is the highest priority ---
        # If navigating and an obstacle is seen while driving, switch state and DO NOTHING ELSE THIS TICK.
        if self.current_state == 'NAVIGATING' and self.obstacle_detected and \
           self.is_executing_action and self.active_plan[self.current_action_index]['action'] == 'drive':
            
            self.get_logger().warn("Obstacle detected! Starting avoidance maneuver immediately.")
            self.stop_robot() # Stop first
            
            # Calculate remaining distance
            distance_traveled = math.sqrt((self.current_pose.position.x - self.start_pose.position.x)**2 + 
                                          (self.current_pose.position.y - self.start_pose.position.y)**2)
            self.remaining_drive_distance = max(0, self.active_plan[self.current_action_index]['value'] - distance_traveled)
            
            # Switch to avoidance state
            self.current_state = 'AVOIDING'
            self.active_plan = self.avoidance_plan
            self.current_action_index = 0
            self.is_executing_action = False
            return # Exit the function immediately to prevent any further action this tick

        # --- Normal Execution Logic ---
        if self.current_action_index >= len(self.active_plan):
            self.handle_plan_completion()
            return
        
        if not self.is_executing_action:
            self.start_new_action()
        
        self._execute_action()

    def handle_plan_completion(self):
        """Called when the current plan (mission or avoidance) is finished."""
        if self.current_state == 'AVOIDING':
            self.get_logger().info("Avoidance maneuver complete. Resuming original path.")
            self.current_state = 'NAVIGATING'
            self.active_plan = self.mission_plan
            # The mission index is already at the correct 'drive' action,
            # we just need to update its value with the remaining distance.
            self.active_plan[self.current_action_index]['value'] = self.remaining_drive_distance
            self.is_executing_action = False
        else: # NAVIGATING state
            self.get_logger().info("🎉 Mission Complete! 🎉")
            self.stop_robot()
            self.timer.cancel()

    def _execute_action(self):
        action = self.active_plan[self.current_action_index]
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

    def start_new_action(self):
        self.is_executing_action = True
        self.start_pose = self.current_pose
        action = self.active_plan[self.current_action_index]
        self.get_logger().info(f"[{self.current_state}] Starting: {action['action']} for {action['value']}")

    def action_completed(self):
        self.get_logger().info(f"[{self.current_state}] Action complete.")
        self.stop_robot()
        self.is_executing_action = False
        self.current_action_index += 1
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
