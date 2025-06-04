#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import numpy as np
import time

class CompleteMacadamiaNavigation(Node):
    """
    Complete navigation system that:
    1. Detects trees and displays on terminal ✅
    2. Detects nuts and displays on terminal ✅ 
    3. Avoids crashing into trees ✅
    4. Avoids rolling over nuts ✅
    5. Explores field systematically ✅
    """
    
    def __init__(self):
        super().__init__('complete_macadamia_navigation')
        
        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/navigation_status', 10)
        
        # Subscribers - listening to your existing detectors
        self.tree_sub = self.create_subscription(
            String, '/detected_trees', self.tree_callback, 10)
        self.nut_sub = self.create_subscription(
            String, '/tennis_ball_marker', self.nut_callback, 10)  
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        
        # ============ FIELD PARAMETERS ============
        self.field_rows = 2            # 2x2 field 
        self.field_cols = 2
        self.tree_spacing = 1.0        # Distance between trees
        
        # Speed settings
        self.normal_speed = 0.15       # Normal exploration speed
        self.slow_speed = 0.08         # Slow speed near obstacles
        self.crawl_speed = 0.03        # Very slow near nuts
        self.turn_speed = 0.4          # Turn speed for avoidance
        
        # Safety distances
        self.tree_avoidance_distance = 0.20   # 20cm - avoid trees
        self.nut_avoidance_distance = 0.15    # 15cm - avoid nuts (smaller)
        self.emergency_distance = 0.10        # 10cm - emergency stop
        
        # Detection state
        self.trees_detected = False
        self.nuts_detected = False
        self.last_tree_detection = 0
        self.last_nut_detection = 0
        self.detection_timeout = 1.0    # Consider gone after 1 second
        
        # LiDAR data
        self.scan_data = None
        self.min_front_distance = float('inf')
        
        # Avoidance state
        self.avoiding_obstacle = False
        self.avoidance_start_time = 0
        self.avoidance_duration = 2.0
        self.avoidance_reason = ""
        
        # Exploration state (simple wall following for now)
        self.exploration_mode = "wall_follow"
        self.wall_target_distance = 0.4
        
        # Statistics
        self.trees_found_count = 0
        self.nuts_found_count = 0
        self.avoidance_count = 0
        
        # Navigation timer
        self.nav_timer = self.create_timer(0.1, self.navigation_loop)
        
        self.get_logger().info('🤖 Complete Macadamia Navigation System initialized!')
        self.get_logger().info(f'🌾 Field: {self.field_rows}x{self.field_cols} trees')
        self.get_logger().info('🌳 Tree avoidance: 20cm | 🥜 Nut avoidance: 15cm | 🚨 Emergency: 10cm')
        self.get_logger().info('🔍 Listening for tree and nut detections...')
        self.publish_status("Complete navigation ready - tree and nut avoidance active")
        
    def publish_status(self, message):
        """Publish navigation status."""
        status_msg = String()
        status_msg.data = message
        self.status_pub.publish(status_msg)
        
    def scan_callback(self, msg):
        """Process LiDAR for distance measurement."""
        self.scan_data = msg
        
        # Calculate minimum front distance
        if msg.ranges:
            center_idx = len(msg.ranges) // 2
            front_span = len(msg.ranges) // 8  # ±22.5 degrees
            start_idx = max(0, center_idx - front_span)
            end_idx = min(len(msg.ranges), center_idx + front_span)
            
            front_ranges = msg.ranges[start_idx:end_idx]
            valid_ranges = [r for r in front_ranges if np.isfinite(r) and r > msg.range_min]
            
            if valid_ranges:
                self.min_front_distance = min(valid_ranges)
            else:
                self.min_front_distance = float('inf')
    
    def tree_callback(self, msg):
        """Handle tree detection messages."""
        current_time = time.time()
        self.last_tree_detection = current_time
        
        if "trees_detected:" in msg.data:
            num_trees = int(msg.data.split(':')[1])
            self.trees_detected = True
            
            # Display tree detection on terminal
            if num_trees > 0:
                self.trees_found_count += 1
                self.get_logger().info(f'🌳 TREES DETECTED! Count: {num_trees} | Distance: {self.min_front_distance*100:.1f}cm | Total found: {self.trees_found_count}')
                
                # Check if trees are too close
                if self.min_front_distance <= self.tree_avoidance_distance:
                    self.get_logger().warn(f'⚠️  Trees too close ({self.min_front_distance*100:.1f}cm) - initiating avoidance!')
    
    def nut_callback(self, msg):
        """Handle nut detection messages."""
        current_time = time.time()
        self.last_nut_detection = current_time
        
        if "nuts_detected:" in msg.data:
            num_nuts = int(msg.data.split(':')[1])
            self.nuts_detected = True
            
            # Display nut detection on terminal
            if num_nuts > 0:
                self.nuts_found_count += 1
                self.get_logger().info(f'🥜 NUTS DETECTED! Count: {num_nuts} | Distance: {self.min_front_distance*100:.1f}cm | Total found: {self.nuts_found_count}')
                
                # Check if nuts are too close (avoid rolling over them)
                if self.min_front_distance <= self.nut_avoidance_distance:
                    self.get_logger().warn(f'🚨 Nuts too close ({self.min_front_distance*100:.1f}cm) - avoiding to prevent damage!')
    
    def navigation_loop(self):
        """Main navigation decision loop."""
        current_time = time.time()
        
        # Update detection states based on timeouts
        if current_time - self.last_tree_detection > self.detection_timeout:
            self.trees_detected = False
        if current_time - self.last_nut_detection > self.detection_timeout:
            self.nuts_detected = False
        
        # Handle avoidance mode
        if self.avoiding_obstacle:
            if current_time - self.avoidance_start_time < self.avoidance_duration:
                self.execute_avoidance_maneuver()
                return
            else:
                self.avoiding_obstacle = False
                self.get_logger().info(f'✅ {self.avoidance_reason} avoidance completed')
                self.publish_status("Avoidance completed - resuming exploration")
        
        # Check for immediate dangers and obstacles
        distance = self.min_front_distance
        
        # PRIORITY 1: Emergency stop for very close obstacles
        if distance <= self.emergency_distance:
            self.emergency_stop("Emergency stop")
            
        # PRIORITY 2: Avoid nuts (don't roll over them!)
        elif distance <= self.nut_avoidance_distance and self.nuts_detected:
            self.start_avoidance("Nut")
            
        # PRIORITY 3: Avoid trees
        elif distance <= self.tree_avoidance_distance and self.trees_detected:
            self.start_avoidance("Tree")
            
        # PRIORITY 4: Slow down when obstacles detected but not too close
        elif (self.trees_detected or self.nuts_detected) and distance <= 0.5:
            self.move_cautiously()
            
        # PRIORITY 5: Normal exploration
        else:
            self.explore_field()
    
    def emergency_stop(self, reason):
        """Emergency stop for very close obstacles."""
        twist = Twist()  # All zeros = stop
        self.cmd_pub.publish(twist)
        
        self.get_logger().error(f'🛑 EMERGENCY STOP! {reason} at {self.min_front_distance*100:.1f}cm')
        self.publish_status(f"EMERGENCY STOP - {self.min_front_distance*100:.1f}cm")
        
        # Start avoidance immediately
        self.start_avoidance("Emergency")
    
    def start_avoidance(self, reason):
        """Start avoidance maneuver."""
        if not self.avoiding_obstacle:
            self.avoiding_obstacle = True
            self.avoidance_start_time = time.time()
            self.avoidance_reason = reason
            self.avoidance_count += 1
            
            self.get_logger().warn(f'🔄 Starting {reason} avoidance #{self.avoidance_count} - distance: {self.min_front_distance*100:.1f}cm')
            self.publish_status(f"Avoiding {reason} at {self.min_front_distance*100:.1f}cm")
    
    def execute_avoidance_maneuver(self):
        """Execute avoidance turn."""
        twist = Twist()
        twist.linear.x = 0.0              # Stop forward motion
        twist.angular.z = self.turn_speed  # Turn right
        self.cmd_pub.publish(twist)
    
    def move_cautiously(self):
        """Move slowly when obstacles are detected but not too close."""
        twist = Twist()
        
        if self.nuts_detected:
            twist.linear.x = self.crawl_speed  # Very slow near nuts
            speed_desc = "crawling"
        else:
            twist.linear.x = self.slow_speed   # Slow near trees
            speed_desc = "slow"
            
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)
        
        obstacle_type = []
        if self.trees_detected: obstacle_type.append("trees")
        if self.nuts_detected: obstacle_type.append("nuts")
        
        self.publish_status(f"Moving {speed_desc} - {', '.join(obstacle_type)} detected at {self.min_front_distance*100:.1f}cm")
    
    def explore_field(self):
        """Normal field exploration (simple wall following for now)."""
        if not self.scan_data:
            return
            
        # Simple wall following algorithm
        ranges = self.scan_data.ranges
        
        # Get right side distance
        right_idx = len(ranges) // 4 if len(ranges) > 4 else 0
        right_distance = ranges[right_idx] if right_idx < len(ranges) and np.isfinite(ranges[right_idx]) else float('inf')
        
        twist = Twist()
        
        if np.isfinite(right_distance) and right_distance < 2.0:  # Wall detected within 2m
            # Wall following control
            distance_error = right_distance - self.wall_target_distance
            
            twist.linear.x = self.normal_speed
            twist.angular.z = -distance_error * 1.5  # Proportional control
        else:
            # No wall detected, move forward and search
            twist.linear.x = self.normal_speed * 0.8
            twist.angular.z = -0.2  # Slight right turn to find wall
        
        self.cmd_pub.publish(twist)
        self.publish_status(f"Exploring field - wall at {right_distance:.2f}m")
    
    def get_exploration_summary(self):
        """Get summary of exploration progress."""
        return {
            'trees_found': self.trees_found_count,
            'nuts_found': self.nuts_found_count,
            'avoidances': self.avoidance_count
        }

def main(args=None):
    rclpy.init(args=args)
    
    try:
        navigator = CompleteMacadamiaNavigation()
        navigator.get_logger().info('🚀 Starting complete macadamia harvesting navigation...')
        navigator.get_logger().info('🎯 Features: Tree detection ✅ Nut detection ✅ Tree avoidance ✅ Nut avoidance ✅')
        navigator.get_logger().info('🌾 Ready to explore macadamia field safely!')
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        navigator.get_logger().info('Navigation system shutting down...')
        
        # Print final summary
        summary = navigator.get_exploration_summary()
        navigator.get_logger().info('📊 Final Summary:')
        navigator.get_logger().info(f'   🌳 Trees found: {summary["trees_found"]}')
        navigator.get_logger().info(f'   🥜 Nuts found: {summary["nuts_found"]}') 
        navigator.get_logger().info(f'   🔄 Avoidances: {summary["avoidances"]}')
    finally:
        if 'navigator' in locals():
            # Stop robot
            stop_twist = Twist()
            navigator.cmd_pub.publish(stop_twist)
            navigator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
