#!/usr/bin/env python3

import math
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped, Quaternion
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from std_msgs.msg import Header


class Nav2MissionPlanner(Node):
    def __init__(self):
        super().__init__('nav2_mission_planner_home')

        # --- Waypoints: (x, y, yaw_degrees) ---
        # THIS LIST IS NOW COMPLETE AND CORRECT
        self.waypoints: List[Tuple[float, float, float]] = [
            (1.2, 0.0,   0),  # P1 <-- THIS WAYPOINT IS NOW INCLUDED
            (1.2, 0.6,  90),  # P2
            (0.0, 0.6, 90),   # P3
            (0.0, 1.2, -90),  # P4 
            (1.2, 1.2, -90),  # P5
        ]

        self.nav_action = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.current_idx = 0
        self.is_sending = False
        self.is_returning_home = False
        self.timer = self.create_timer(1.0, self._tick)
        self.return_timer = None # Initialize the return timer variable

    def _pose_from_xyyaw(self, x: float, y: float, yaw_deg: float) -> PoseStamped:
        yaw_rad = math.radians(yaw_deg)
        q = Quaternion(z=math.sin(yaw_rad / 2.0), w=math.cos(yaw_rad / 2.0))

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
            if not self.is_returning_home:
                self._start_return_home_sequence()
            return

        x, y, yaw = self.waypoints[self.current_idx]
        goal_msg = NavigateToPose.Goal(pose=self._pose_from_xyyaw(x, y, yaw))

        self.get_logger().info(
            f'🚩 Sending waypoint {self.current_idx + 1}/{len(self.waypoints)}: '
            f'({x:.2f}, {y:.2f}, {yaw:.1f}°)')
        self.is_sending = True
        self._send_future = self.nav_action.send_goal_async(goal_msg, feedback_callback=self._feedback_cb)
        self._send_future.add_done_callback(self._goal_response_cb)

    def _start_return_home_sequence(self):
        self.get_logger().info('✅ Main mission complete. Waiting 30 seconds before returning home...')
        self.is_returning_home = True
        self.timer.cancel()
        
        # Create a regular timer without 'oneshot'. It will be cancelled in its callback.
        self.return_timer = self.create_timer(30.0, self._send_home_goal)
        
    def _send_home_goal(self):
        # The first thing we do is cancel the timer so it never runs again.
        if self.return_timer is not None:
            self.return_timer.cancel()

        self.get_logger().info('⏰ Wait finished. Sending robot home to (0,0)...')
        home_pose = self._pose_from_xyyaw(0.0, 0.0, 0.0)
        goal_msg = NavigateToPose.Goal(pose=home_pose)

        self.is_sending = True
        self._send_future = self.nav_action.send_goal_async(goal_msg, feedback_callback=self._feedback_cb)
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
        self.get_logger().debug(f'Distance to goal: {fb.distance_remaining:.2f} m')

    def _result_cb(self, future):
        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('✅ Goal reached successfully')
        else:
            self.get_logger().warn(f'⚠️ Goal ended with status {status}')

        if self.is_returning_home:
            self.get_logger().info('🎉🎉 Robot has returned home. Mission fully complete! 🎉🎉')
            self.destroy_node()
        else:
            self.current_idx += 1
            self.is_sending = False


def main(args=None):
    rclpy.init(args=args)
    node = Nav2MissionPlanner()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        # A slightly safer shutdown sequence
        if rclpy.ok() and 'node' in locals() and not node.executor.is_shutdown():
             node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
