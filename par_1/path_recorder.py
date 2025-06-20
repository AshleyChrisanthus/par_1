#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import tf2_ros
import math

class PathRecorder(Node):
    """
    Node that records the robot's path and publishes it continuously.
    """
    def __init__(self):
        super().__init__('path_recorder')
        
        # Configure the path message
        self.path = Path()
        self.path.header.frame_id = 'map'
        self.recorded_poses = []
        self.last_position = None
        
        # Set up TF2 listener
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Set up path publisher (will publish continuously)
        self.path_publisher = self.create_publisher(Path, '/path_explore', 10)
        
        # Create timer for tracking (1 Hz)
        self.timer = self.create_timer(1.0, self.track_position)
        
        self.get_logger().info('Path recorder initialized')
        self.get_logger().info('Recording and publishing path continuously to /path_explore')
    
    def track_position(self):
        """
        Check the robot's current position and add it to the path if it has moved.
        """
        try:
            # Get the latest transform
            transform = self.tf_buffer.lookup_transform(
                'map',               # Target frame
                'base_link',         # Source frame
                rclpy.time.Time(seconds=0),  # Get latest available transform
                timeout=rclpy.duration.Duration(seconds=1.0)  # Timeout
            )
            
            # Current time for the pose timestamp
            current_time = self.get_clock().now()
            
            # Extract position
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            
            # Check if position has changed enough to record
            record_position = False
            if self.last_position is None:
                # First position
                record_position = True
            else:
                # Calculate distance from last position
                dx = x - self.last_position[0]
                dy = y - self.last_position[1]
                distance = math.sqrt(dx*dx + dy*dy)
                
                # Record if moved at least 0.5 meters
                if distance >= 0.5:
                    record_position = True
            
            if record_position:
                # Create pose message
                pose = PoseStamped()
                pose.header.frame_id = 'map'
                pose.header.stamp = current_time.to_msg()
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.position.z = transform.transform.translation.z
                pose.pose.orientation = transform.transform.rotation
                
                # Add to path
                self.recorded_poses.append(pose)
                self.last_position = (x, y)
                
                # Update path
                self.path.header.stamp = current_time.to_msg()
                self.path.poses = self.recorded_poses
                
                self.get_logger().info(f'Recorded position: ({x:.2f}, {y:.2f}), Total points: {len(self.recorded_poses)}')
                
                # Publish the updated path immediately
                self.path_publisher.publish(self.path)
                self.get_logger().info(f'Published updated path with {len(self.recorded_poses)} points')
            
        except tf2_ros.TransformException as ex:
            self.get_logger().warning(f'Could not transform: {ex}')
    
    def get_recorded_path(self):
        """
        Return the recorded path.
        """
        return self.recorded_poses

def main(args=None):
    rclpy.init(args=args)
    
    node = PathRecorder()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
