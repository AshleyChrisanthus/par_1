#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import math

def quaternion_from_yaw(yaw):
    """Convert a yaw angle (in radians) into a quaternion."""
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)
    return (0.0, 0.0, qz, qw)

class GoToPose(Node):
    def __init__(self):
        super().__init__('navigation_goal')
        self.publisher_ = self.create_publisher(PoseStamped, 'goal_pose', 10)
        self.timer = self.create_timer(10.0, self.publish_goal)
        self.goal_pose = PoseStamped()

        # Define the path poses:
        # Each pose is a dict with x, y, z, and orientation as quaternion (x,y,z,w)
        self.points = []

        # Start pose at (0,0), facing +x (yaw=0)
        x, y = 0.0, 0.0
        yaw = 0.0
        self.points.append(self.make_pose(x, y, yaw))

        # Move forward 1.2m along +x (yaw=0)
        x += 1.2
        self.points.append(self.make_pose(x, y, yaw))

        # Turn left 90°, yaw=pi/2, position stays the same
        yaw += math.pi / 2

        # Move forward 0.4m along +y
        y += 0.4
        self.points.append(self.make_pose(x, y, yaw))

        # Turn left 90° again, now facing -x (yaw=pi)
        yaw += math.pi / 2

        # Normalize yaw to [-pi, pi]
        yaw = (yaw + math.pi) % (2 * math.pi) - math.pi

        # Move forward 1.2m along -x
        x -= 1.2
        self.points.append(self.make_pose(x, y, yaw))

        self.current_point_index = 1  # start publishing from first move (skip initial pose)

        self.goal_pose.header.frame_id = 'map'

    def make_pose(self, x, y, yaw):
        qx, qy, qz, qw = quaternion_from_yaw(yaw)
        return {'x': x, 'y': y, 'z': 0.0, 'ox': qx, 'oy': qy, 'oz': qz, 'ow': qw}

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
        self.get_logger().info('Publishing Navigation Goal: x=%.2f, y=%.2f, yaw=%.2f deg' % 
                               (p['x'], p['y'], math.degrees(math.atan2(2*(p['ow'] * p['oz']), 1-2*(p['oz']**2)))))

        self.current_point_index += 1
        if self.current_point_index >= len(self.points):
            self.current_point_index = 0  # loop back if you want continuous cycle

def main(args=None):
    rclpy.init(args=args)
    navigation_goal = GoToPose()
    rclpy.spin(navigation_goal)
    navigation_goal.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
