#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import math

class GoToPose(Node):
    def __init__(self):
        super().__init__('navigation_goal')

        self.publisher_ = self.create_publisher(PoseStamped, 'goal_pose', 10)
        self.subscription = self.create_subscription(
            Odometry,
            # '/rosbot_base_controller/odom', # rosbot 2
            '/odometry/filtered',
            self.odom_callback,
            10
        )

        # Goals: move from (2.2, 0.0) facing forward to (2.2, 1.0) facing left
        self.points = [
            {'x': 2.2, 'y': 0.0, 'z': 0.0, 'ox': 0.0, 'oy': 0.0, 'oz': 0.0, 'ow': 1.0},           # yaw = 0°
            {'x': 2.2, 'y': 1.0, 'z': 0.0, 'ox': 0.0, 'oy': 0.0, 'oz': 0.7071, 'ow': 0.7071}      # yaw = 90°
        ]

        self.current_point_index = 0
        self.goal_sent = False
        self.goal_pose = PoseStamped()
        self.goal_pose.header.frame_id = 'map'

        self.publish_next_goal()

    def publish_next_goal(self):
        if self.current_point_index >= len(self.points):
            self.get_logger().info("✅ Reached final goal. Stopping.")
            return

        p = self.points[self.current_point_index]

        self.goal_pose.pose.position.x = p['x']
        self.goal_pose.pose.position.y = p['y']
        self.goal_pose.pose.position.z = p['z']
        self.goal_pose.pose.orientation.x = p['ox']
        self.goal_pose.pose.orientation.y = p['oy']
        self.goal_pose.pose.orientation.z = p['oz']
        self.goal_pose.pose.orientation.w = p['ow']
        self.goal_pose.header.stamp = self.get_clock().now().to_msg()

        self.publisher_.publish(self.goal_pose)
        self.goal_sent = True

        self.get_logger().info(
            f'📍 Sent Goal {self.current_point_index + 1}: x={p["x"]:.2f}, y={p["y"]:.2f}'
        )

    def odom_callback(self, msg):
        
        if not self.goal_sent or self.current_point_index >= len(self.points):
            return

        current_pose = msg.pose.pose
        goal = self.points[self.current_point_index]

        dx = current_pose.position.x - goal['x']
        dy = current_pose.position.y - goal['y']
        distance = math.sqrt(dx * dx + dy * dy)

        self.get_logger().info(
            f'🔍 Checking goal {self.current_point_index + 1}: distance = {distance:.3f}'
        )

        if distance < 0.25:  # Increase threshold slightly if needed
            self.get_logger().info(f'✅ Reached Goal {self.current_point_index + 1}')
            self.current_point_index += 1
            self.goal_sent = False
            self.publish_next_goal()

def main(args=None):
    rclpy.init(args=args)
    navigation_goal = GoToPose()
    rclpy.spin(navigation_goal)
    navigation_goal.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
