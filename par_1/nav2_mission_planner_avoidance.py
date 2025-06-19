#!/usr/bin/env python3

import math
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped, Quaternion
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus


class Nav2MissionPlannerAvoidance(Node):
    def __init__(self):
        super().__init__('nav2_mission_planner_avoidance')

        # --- Waypoints: (x, y, yaw_degrees) ---
        self.waypoints: List[Tuple[float, float, float]] = [
            (1.2, 0.0,   0),  # P1
            (1.2, 0.6,  90),  # P2
            (0.0, 0.6, 90),   # P3
            (0.0, 1.2, -90),  # P4 
            (1.2, 1.2, -90),  # P5
            (0.0, 0.0, -90),  # P6 (HOME) 
        ]

        self.nav_action = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.current_idx = 0
        self.is_sending = False
        self.scan_timer = None # MODIFIED: Variable to hold our scan timer

        self.timer = self.create_timer(1.0, self._tick)

    def _pose_from_xyyaw(self, x: float, y: float, yaw_deg: float) -> PoseStamped:
        yaw_rad = math.radians(yaw_deg)
        q = Quaternion()
        q.z = math.sin(yaw_rad / 2.0)
        q.w = math.cos(yaw_rad / 2.0)

        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation = q
        return pose

    def _tick(self):
        if not self.nav_action.server_is_ready():
            self.get_logger().info('Waiting for Nav2 action server...')
            return
        if not self.is_sending:
            self._send_next_goal()

    def _send_next_goal(self):
        if self.current_idx >= len(self.waypoints):
            self.get_logger().info('🎉 Mission complete – all waypoints reached!')
            self.destroy_timer(self.timer)
            return

        x, y, yaw = self.waypoints[self.current_idx]
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = self._pose_from_xyyaw(x, y, yaw)

        self.get_logger().info(
            f'🚩 Sending waypoint {self.current_idx + 1}/{len(self.waypoints)}: '
            f'({x:.2f}, {y:.2f}, {yaw:.1f}°)')
        self.is_sending = True
        self._send_future = self.nav_action.send_goal_async(
            goal_msg,
            feedback_callback=self._feedback_cb)
        self._send_future.add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('❌ Goal rejected by Nav2')
            self.is_sending = False
            return

        self.get_logger().info('✔️ Goal accepted, navigating...')
        self._result_future = goal_handle.get_result_async()
        self._result_future.add_done_callback(self._result_cb)

    def _feedback_cb(self, feedback_msg):
        fb = feedback_msg.feedback
        self.get_logger().debug(
            f'Distance to goal: {fb.distance_remaining:.2f} m')

    # --- MODIFIED: This function now triggers the scanning phase ---
    def _result_cb(self, future):
        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('✅ Waypoint reached successfully.')
            # If there are more waypoints, start scanning. Otherwise, we are done.
            if self.current_idx < len(self.waypoints) - 1:
                self._start_scan()
            else:
                self.current_idx += 1 # Increment to trigger mission complete
                self.is_sending = False
        else:
            self.get_logger().warn(f'⚠️ Goal ended with status {status}. Not scanning, trying next waypoint.')
            self.current_idx += 1
            self.is_sending = False

    # --- NEW: Function to start the timed pause for scanning ---
    def _start_scan(self):
        self.is_sending = True # Prevent _tick from sending another goal
        self.get_logger().info('------------------------------------')
        self.get_logger().info('⏸️  PAUSING for 5 seconds to scan for obstacles...')
        self.get_logger().info('------------------------------------')
        if self.scan_timer is not None:
            self.scan_timer.cancel()
        # Create a one-shot timer that calls _scan_complete_cb after 5s
        self.scan_timer = self.create_timer(5.0, self._scan_complete_cb)

    # --- NEW: Callback function for when the scan timer finishes ---
    def _scan_complete_cb(self):
        self.get_logger().info('------------------------------------')
        self.get_logger().info('▶️  Scan complete. Resuming mission.')
        self.get_logger().info('------------------------------------')
        # Destroy the timer so it doesn't run again
        if self.scan_timer:
            self.destroy_timer(self.scan_timer)
            self.scan_timer = None
        
        self.current_idx += 1
        self.is_sending = False # Allow _tick to send the next goal


def main(args=None):
    rclpy.init(args=args)
    node = Nav2MissionPlannerAvoidance()
    try:
        rclpy.spin(node)
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
