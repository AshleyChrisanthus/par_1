#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

class GoToPose(Node):
    def _init_(self):
        super()._init_('navigation_goal')
        self.publisher_ = self.create_publisher(PoseStamped, 'goal_pose', 10)
        self.timer = self.create_timer(10.0, self.publish_goal)
        self.goal_pose = PoseStamped()

        self.points = [
            {'x': 1.0, 'y': 1.0, 'z': 0.0, 'ox': 0.0, 'oy': 0.0, 'oz': 0.0, 'ow': 1.0},
            {'x': 1.5, 'y': 1.0,  'z': 0.0, 'ox': 0.0, 'oy': 0.0, 'oz': 0.0, 'ow': 1.0},
            {'x': -1.0,'y': 0.5,  'z': 0.0, 'ox': 0.0, 'oy': 0.0, 'oz': 0.0, 'ow': 1.0}
        ]
        self.current_point_index = 0

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
        self.get_logger().info('Publishing Navigation Goal: x=%f, y=%f' % (p['x'], p['y']))

        self.current_point_index = (self.current_point_index + 1) % len(self.points)

def main(args=None):
    rclpy.init(args=args)
    navigation_goal_publisher = GoToPose()
    rclpy.spin(navigation_goal_publisher)
    navigation_goal_publisher.destroy_node()
    rclpy.shutdown()

if _name_ == '_main_':
    main()
