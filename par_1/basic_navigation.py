#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import numpy as np
import time

class BasicNavigation(Node):
    """
    Proximity-based navigation that only avoids trees when they are within 10cm.
    Uses LiDAR data to determine actual distance to obstacles.
    """
    
    def __init__(self):
        super().__init__('basic_navigation')
        
        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/navigation_status', 10)
        
        # Subscribers
        self.tree_sub = self.create_subscription(
            String,
            '/detected_trees',
            self.tree_callback,
            10)
        
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)
        
        # Navigation parameters
        self.normal_speed = 0.3          # Normal forward speed
        self.slow_speed = 0.15           # Speed when trees detected but not close
        self.emergency_turn_speed = 0.8  # Fast turn when very close
        
        # Proximity thresholds
        self.danger_distance = 0.10      # 10cm - EMERGENCY AVOIDANCE
        self.caution_distance = 0.25     # 25cm - Slow down
        self.detection_distance = 0.50   # 50cm - Be aware but keep moving
        
        # LiDAR scanning parameters
        self.scan_data = None
        self.front_scan_angles = 60      # Degrees to scan in front (±30°)
        
        # State management
        self.avoiding_obstacle = False
        self.avoidance_start_time = 0
        self.avoidance_duration = 1.5    # Shorter avoidance time
        
        # Detection tracking
        self.trees_detected = False
        self.closest_obstacle_distance = float('inf')
        self.last_tree_detection_time = 0
        
        # Timer for regular movement
        self.movement_timer = self.create_timer(0.1, self.navigation_loop)
        
        self.get_logger().info('Proximity-Based Navigation initialized!')
        self.get_logger().info(f'⚠️ DANGER zone: {self.danger_distance*100:.0f}cm (Tree Avoidance)')
        self.get_logger().info(f'CAUTION zone: {self.caution_distance*100:.0f}cm (slow down)')
        self.get_logger().info(f'DETECTION zone: {self.detection_distance*100:.0f}cm (aware but moving)')
        self.publish_status("Proximity navigation ready - 10cm Tree Avoidance")
        
    def publish_status(self, message):
        """Publish navigation status."""
        status_msg = String()
        status_msg.data = message
        self.status_pub.publish(status_msg)
        
    def scan_callback(self, msg):
        """Store LiDAR scan data for proximity analysis."""
        self.scan_data = msg
        
        # Analyze front area for closest obstacles
        self.analyze_front_proximity()
        
    def analyze_front_proximity(self):
        """Analyze the front area to find closest obstacles."""
        if self.scan_data is None:
            return
            
        # Calculate front scan indices (±30° from front)
        total_angles = len(self.scan_data.ranges)
        angle_increment = (self.scan_data.angle_max - self.scan_data.angle_min) / total_angles
        front_half_span = int((np.radians(self.front_scan_angles/2)) / angle_increment)
        
        # Get center index (front of robot)
        center_index = total_angles // 2
        start_index = max(0, center_index - front_half_span)
        end_index = min(total_angles, center_index + front_half_span)
        
        # Find minimum distance in front area
        front_ranges = self.scan_data.ranges[start_index:end_index]
        valid_ranges = [r for r in front_ranges if np.isfinite(r) and r > self.scan_data.range_min]
        
        if valid_ranges:
            self.closest_obstacle_distance = min(valid_ranges)
        else:
            self.closest_obstacle_distance = float('inf')
    
    def tree_callback(self, msg):
        """Handle tree detection messages."""
        current_time = time.time()
        self.last_tree_detection_time = current_time
        
        if "trees_detected:" in msg.data:
            self.trees_detected = True
            num_trees = int(msg.data.split(':')[1])
            
            # Log tree detection with proximity info
            self.get_logger().info(f'Trees detected: {num_trees} | Closest obstacle: {self.closest_obstacle_distance*100:.1f}cm')
        
    def navigation_loop(self):
        """Main navigation decision loop."""
        current_time = time.time()
        
        # Check if we have recent tree detections
        time_since_detection = current_time - self.last_tree_detection_time
        if time_since_detection > 1.0:  # No trees detected in last 1 second
            self.trees_detected = False
        
        # Make navigation decision based on proximity
        self.make_navigation_decision()
        
    def make_navigation_decision(self):
        """Make navigation decision based on proximity zones."""
        current_time = time.time()
        
        # Check if we're in Tree Avoidance mode
        if self.avoiding_obstacle:
            if current_time - self.avoidance_start_time < self.avoidance_duration:
                self.execute_emergency_avoidance()
                return
            else:
                # Emergency avoidance complete
                self.avoiding_obstacle = False
                self.get_logger().info('Tree avoidance completed')
                self.publish_status("Tree avoidance completed")
        
        # Check proximity zones
        distance = self.closest_obstacle_distance
        
        if distance <= self.danger_distance:
            # EMERGENCY ZONE - 10cm or less
            if not self.avoiding_obstacle:
                self.start_emergency_avoidance()
            
        elif distance <= self.caution_distance and self.trees_detected:
            # CAUTION ZONE - 25cm or less with trees detected
            self.move_with_caution()
            
        elif distance <= self.detection_distance and self.trees_detected:
            # DETECTION ZONE - 50cm or less with trees detected  
            self.move_with_awareness()
            
        else:
            # SAFE ZONE - Normal movement
            self.move_normally()
    
    def start_emergency_avoidance(self):
        """Start Tree Avoidance maneuver."""
        self.avoiding_obstacle = True
        self.avoidance_start_time = time.time()
        
        self.get_logger().warn(f'Tree Avoidance! Obstacle at {self.closest_obstacle_distance*100:.1f}cm')
        self.publish_status(f"Tree Avoidance - {self.closest_obstacle_distance*100:.1f}cm")
        
    def execute_emergency_avoidance(self):
        """Execute Tree Avoidance maneuver."""
        twist = Twist()
        twist.linear.x = -0.05  # Slight backward movement
        twist.angular.z = self.emergency_turn_speed  # Fast turn
        self.cmd_pub.publish(twist)
        
    def move_with_caution(self):
        """Move slowly when in caution zone."""
        twist = Twist()
        twist.linear.x = self.slow_speed
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)
        
        self.publish_status(f"CAUTION - Trees at {self.closest_obstacle_distance*100:.1f}cm - moving slowly")
        
    def move_with_awareness(self):
        """Move normally but be aware of trees."""
        twist = Twist()
        twist.linear.x = self.normal_speed * 0.8  # Slightly reduced speed
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)
        
        self.publish_status(f"AWARE - Trees at {self.closest_obstacle_distance*100:.1f}cm - ready to react")
        
    def move_normally(self):
        """Normal forward movement."""
        twist = Twist()
        twist.linear.x = self.normal_speed
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)
        
        if self.closest_obstacle_distance < float('inf'):
            self.publish_status(f"NORMAL - Clear path, closest obstacle {self.closest_obstacle_distance*100:.1f}cm")
        else:
            self.publish_status("NORMAL - Path clear")

def main(args=None):
    rclpy.init(args=args)
    
    try:
        navigator = BasicNavigation()
        navigator.get_logger().info('Starting proximity-based navigation...')
        # navigator.get_logger().info('Robot will get close to trees and only avoid at 10cm!')
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        navigator.get_logger().info('Proximity navigation shutting down...')
    finally:
        # Stop the robot
        if 'navigator' in locals():
            stop_twist = Twist()
            navigator.cmd_pub.publish(stop_twist)
            navigator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
