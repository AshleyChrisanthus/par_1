#!/usr/bin/env python3
"""
Nav2‑based mission planner.

▪ Publishes no direct /cmd_vel; instead, it sends NavigateToPose goals.
▪ Waypoints are listed as [x, y, yaw_deg] in map coordinates.
▪ Relies on Nav2 (BT Navigator, planner, controller, costmaps) for obstacle‑aware motion.
"""

import math
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped, Quaternion
from nav2_msgs.action import NavigateToPose
from tf_transformations import quaternion_from_euler   # pip install tf_transformations

class Nav2MissionPlanner(Node):
    def __init__(self):
        super().__init__('nav2_mission_planner')

        # --- CONFIGURABLE WAYPOINT LIST (x, y, yaw°) --------------------------
        self.declare_parameter('waypoints',
            [[2.2, 0.0,   0],
             [2.2, 0.6,  90],
             [0.0, 0.6, 180]])   # Example: rectangle
        raw_points: List[List[float]] = self.get_parameter('waypoints').get_parameter_value().double_array_value
        # rclpy flattens arrays of arrays; reconstruct tuples of three
        self.waypoints = [tuple(raw_points[i:i+3])           # type: ignore[arg-type]
                          for i in range(0, len(raw_points), 3)]

        # --- ACTION CLIENT ----------------------------------------------------
        self.nav_action = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.current_idx = 0
        self.is_sending = False

        # Poll until Nav2 action server is ready, then start
        self.timer = self.create_timer(1.0, self._tick)

    # -------------------------------------------------------------------------
    # Helper: build a PoseStamped from (x, y, yaw°)
    def _pose_from_xyyaw(self, x: float, y: float, yaw_deg: float) -> PoseStamped:
        q = quaternion_from_euler(0.0, 0.0, math.radians(yaw_deg))
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        return pose

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Timer tick – drives the sequence
    def _tick(self):
        if not self.nav_action.server_is_ready():
            self.get_logger().info('Waiting for Nav2 action server...')
            return
        if not self.is_sending:
            self._send_next_goal()

    # -------------------------------------------------------------------------
    # Action callbacks
    def _goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal rejected by Nav2')
            self.is_sending = False
            return

        self.get_logger().info('Goal accepted, waiting for result...')
        self._result_future = goal_handle.get_result_async()
        self._result_future.add_done_callback(self._result_cb)

    def _feedback_cb(self, feedback_msg):
        fb = feedback_msg.feedback
        self.get_logger().debug(
            f'Current distance to goal: {fb.distance_remaining:.2f} m')

    def _result_cb(self, future):
        result = future.result().result
        if result.error_code == 0:
            self.get_logger().info('Goal reached ✔︎')
        else:
            self.get_logger().warn(f'Goal failed with error code {result.error_code}')
        self.current_idx += 1
        self.is_sending = False   # triggers next goal on next timer tick

# -----------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = Nav2MissionPlanner()
    try:
        rclpy.spin(node)
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
