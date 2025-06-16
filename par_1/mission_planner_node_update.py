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

        # --- Mission Plans ---
        self.mission_plan = [
            {'action': 'drive', 'value': 2.2},
            {'action': 'turn', 'value': 90},
            {'action': 'drive', 'value': 0.6},
            {'action': 'turn', 'value': 90},
            {'action': 'drive', 'value': 2.2}
        ]
        
        # --- Your Requested Avoidance Plan ---
        self.avoidance_plan = [
            {'action': 'turn', 'value': -30}, # 1. Turn right 45 deg
            {'action': 'drive', 'value': 0.5}, # 2. Drive diagonally forward
            {'action': 'turn', 'value': 90},  # 3. Turn left 90 deg (now parallel)
            {'action': 'drive', 'value': 0.8}, # 4. Drive forward past obstacle
            {'action': 'turn', 'value': -30}  # 5. Turn right 45 deg to realign
        ]

        # --- State Machine ---
        self.current_state = 'NAVIGATING'
        self.active_plan = self.mission_plan
        self.current_action_index = 0
        self.is_executing_action = False
        self.obstacle_detected = False
        self.remaining_drive_distance = 0.0

        # --- Robot State ---
        self.start_pose = None
        self.current_pose = None
        self.current_yaw = 0.0

        # --- THE FIX: Corrected Control & Safety Parameters ---
        self.linear_speed = 0.15  # SLOWED DOWN for less momentum
        self.angular_speed = 0.4
        self.obstacle_distance = 0.5 # INCREASED for more braking room
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
            return

        # --- Immediate state transition is the highest priority ---
        if self.current_state == 'NAVIGATING' and self.obstacle_detected and \
           self.is_executing_action and self.active_plan[self.current_action_index]['action'] == 'drive':
            
            self.get_logger().warn(f"Obstacle detected at {min([r for r in self.scan_callback.last_scan.ranges if r > 0.01]):.2f}m! Starting avoidance.")
            self.stop_robot()
            
            distance_traveled = math.sqrt((self.current_pose.position.x - self.start_pose.position.x)**2 + 
                                          (self.current_pose.position.y - self.start_pose.position.y)**2)
            self.remaining_drive_distance = max(0, self.active_plan[self.current_action_index]['value'] - distance_traveled)
            
            self.current_state = 'AVOIDING'
            self.active_plan = self.avoidance_plan
            self.current_action_index = 0
            self.is_executing_action = False
            return

        # --- Normal Execution Logic ---
        if self.current_action_index >= len(self.active_plan):
            self.handle_plan_completion()
            return
        
        if not self.is_executing_action:
            self.start_new_action()
        
        self._execute_action()

    def handle_plan_completion(self):
        if self.current_state == 'AVOIDING':
            self.get_logger().info("Avoidance complete. Resuming original path.")
            self.current_state = 'NAVIGATING'
            self.active_plan = self.mission_plan
            self.active_plan[self.current_action_index]['value'] = self.remaining_drive_distance
            self.is_executing_action = False
        else:
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

# Add a simple helper to the scan callback to store the last scan for logging
def scan_callback_wrapper(self, msg):
    self.scan_callback.last_scan = msg
    self.scan_callback(msg)

SelfContainedNavigator.scan_callback.last_scan = None
SelfContainedNavigator.scan_callback = scan_callback_wrapper.__get__(SelfContainedNavigator, SelfContainedNavigator)


def main(args=None):
    rclpy.init(args=args)
    navigator = SelfContainedNavigator()
    rclpy.spin(navigator)
    navigator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
