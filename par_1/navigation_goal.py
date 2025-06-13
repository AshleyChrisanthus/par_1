#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

class GoToPose(Node):
    def __init__(self):
        super().__init__('navigation_goal')
        self.publisher_ = self.create_publisher(PoseStamped, 'goal_pose', 10)
        self.timer = self.create_timer(10.0, self.publish_goal)

        # Points to move to in order:
        self.points = [
            # {'x': 0.0, 'y': 0.0, 'z': 0.0,
            #  'ox': 0.0, 'oy': 0.0, 'oz': 0.0, 'ow': 1.0},  # start, yaw=0°
            {'x': 2.2, 'y': 0.0, 'z': 0.0,
             'ox': 0.0, 'oy': 0.0, 'oz': 0.0, 'ow': 1.0},  # forward 1.2m
            # {'x': 2.2, 'y': 0.6, 'z': 0.0,
            #  'ox': 0.0, 'oy': 0.0, 'oz': 0.7071, 'ow': 0.7071},  # left turn + forward 0.4m (yaw=90°)
            # {'x': 0.0, 'y': 0.6, 'z': 0.0,
            #  'ox': 0.0, 'oy': 0.0, 'oz': 1.0, 'ow': 0.0}  # left turn + forward 1.2m (yaw=180°)
        ]
        self.current_point_index = 0
        self.goal_pose = PoseStamped()
        self.goal_pose.header.frame_id = 'map'

    def publish_goal(self):
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

        self.get_logger().info(
            f'Publishing Navigation Goal: x={p["x"]:.2f}, y={p["y"]:.2f}'
        )

        # Move to next point only if not at last
        if self.current_point_index < len(self.points) - 1:
            self.current_point_index += 1
        # else do nothing; stay on last point

def main(args=None):
    rclpy.init(args=args)
    navigation_goal = GoToPose()
    rclpy.spin(navigation_goal)
    navigation_goal.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
