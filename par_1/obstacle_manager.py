#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np
# This is a helper library for converting to and from PointCloud2 messages
import point_cloud2

class ObstacleManager(Node):
    def __init__(self):
        super().__init__('obstacle_manager')
        
        # This list will store the (x, y, z) coordinates of unique obstacles
        self.obstacle_points = []
        
        # Subscribe to the ball positions published by your detector
        self.ball_sub = self.create_subscription(
            PointStamped,
            '/ball_pos',
            self._ball_pos_callback,
            10)
            
        # Publisher for the PointCloud2 message that Nav2 will listen to
        self.obstacle_pub = self.create_publisher(PointCloud2, '/dynamic_obstacles', 10)
        
        self.get_logger().info('Obstacle Manager has started.')
        self.get_logger().info('Listening for /ball_pos and publishing to /dynamic_obstacles.')

    def _ball_pos_callback(self, msg: PointStamped):
        """Callback for when a new ball position is detected."""
        new_point = (msg.point.x, msg.point.y, msg.point.z)
        
        # Check if this obstacle is a duplicate of an existing one
        is_duplicate = False
        for existing_point in self.obstacle_points:
            distance = np.linalg.norm(np.array(new_point) - np.array(existing_point))
            # If the new point is very close to an existing one, consider it a duplicate
            if distance < 0.3: # 30cm threshold
                is_duplicate = True
                break
        
        if not is_duplicate:
            self.get_logger().info(f'New unique obstacle detected at: ({new_point[0]:.2f}, {new_point[1]:.2f})')
            self.obstacle_points.append(new_point)
            self._publish_point_cloud()
        
    def _publish_point_cloud(self):
        """Converts the list of obstacle points to a PointCloud2 message and publishes it."""
        # Define the header for the PointCloud2 message
        header = Header(
            stamp=self.get_clock().now().to_msg(),
            frame_id='map' # The obstacles are in the map frame
        )
        
        # Define the fields (structure) of each point
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        
        # Use the helper library to create the PointCloud2 message
        cloud_msg = point_cloud2.create_cloud(header, fields, self.obstacle_points)
        
        # Publish the message
        self.obstacle_pub.publish(cloud_msg)
        self.get_logger().info(f'Published {len(self.obstacle_points)} obstacles to costmap.')

def main(args=None):
    rclpy.init(args=args)
    node = ObstacleManager()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
