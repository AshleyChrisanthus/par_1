#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Point32
from sensor_msgs.msg import PointCloud  # <-- Use the simpler PointCloud
from std_msgs.msg import Header
import numpy as np

class ObstacleManager(Node):
    def __init__(self):
        super().__init__('obstacle_manager')
        
        # This list will store the (x, y, z) coordinates of unique obstacles
        self.obstacle_points = []
        
        self.ball_sub = self.create_subscription(
            PointStamped,
            '/ball_pos',
            self._ball_pos_callback,
            10)
            
        # The publisher now sends the simpler PointCloud message
        self.obstacle_pub = self.create_publisher(PointCloud, '/dynamic_obstacles', 10)
        
        self.get_logger().info('Obstacle Manager has started.')
        self.get_logger().info('Listening for /ball_pos and publishing to /dynamic_obstacles (using PointCloud type).')

    def _ball_pos_callback(self, msg: PointStamped):
        """Callback for when a new ball position is detected."""
        # We store the original PointStamped to keep the full precision
        
        is_duplicate = False
        for existing_point_stamped in self.obstacle_points:
            dx = msg.point.x - existing_point_stamped.point.x
            dy = msg.point.y - existing_point_stamped.point.y
            dz = msg.point.z - existing_point_stamped.point.z
            distance = np.sqrt(dx*dx + dy*dy + dz*dz)
            
            if distance < 0.3: # 30cm threshold
                is_duplicate = True
                break
        
        if not is_duplicate:
            self.get_logger().info(f'New unique obstacle detected at: ({msg.point.x:.2f}, {msg.point.y:.2f})')
            self.obstacle_points.append(msg)
            self._publish_obstacles()
        
    def _publish_obstacles(self):
        """Converts the list of obstacle points to a PointCloud message and publishes it."""
        
        # Create the PointCloud message
        cloud_msg = PointCloud()
        cloud_msg.header = Header(
            stamp=self.get_clock().now().to_msg(),
            frame_id='map' # The obstacles are in the map frame
        )
        
        # Populate the points list
        # Note: PointCloud uses Point32, which uses 32-bit floats. This is fine.
        cloud_msg.points = [
            Point32(x=p.point.x, y=p.point.y, z=p.point.z) for p in self.obstacle_points
        ]
        
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
