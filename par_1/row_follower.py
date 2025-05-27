import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
import numpy as np
import math

class RowFollower(Node):
    def __init__(self):
        super().__init__('row_follower')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

    def lidar_callback(self, msg):
        # Convert ranges to cartesian coordinates
        angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        ranges = np.array(msg.ranges)
        mask = (ranges > msg.range_min) & (ranges < msg.range_max)
        x = ranges[mask] * np.cos(angles[mask])
        y = ranges[mask] * np.sin(angles[mask])

        # Detect two rows (very basic filtering based on y position)
        left_side = y[y > 0.1]
        right_side = y[y < -0.1]

        if len(left_side) > 0 and len(right_side) > 0:
            center_y = (np.mean(left_side) + np.mean(right_side)) / 2
            self.publish_goal(x=0.5, y=center_y)  # move 0.5m forward in x

    def publish_goal(self, x, y):
        goal = PoseStamped()
        goal.header.frame_id = 'base_link'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = float(x)
        goal.pose.position.y = float(y)
        goal.pose.orientation.w = 1.0  # facing forward
        self.goal_pub.publish(goal)
        self.get_logger().info(f'Published goal: ({x:.2f}, {y:.2f})')

def main(args=None):
    rclpy.init(args=args)
    node = RowFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
