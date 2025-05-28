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
    Safe proximity-based navigation with reduced speed and immediate obstacle detection.
    Prioritizes safety over speed to prevent crashes.
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
        
        # MUCH SLOWER navigation parameters for safety
        self.normal_speed = 0.15         # Reduced from 0.3 to 0.15
        self.slow_speed = 0.08           # Very slow when trees detected
        self.crawl_speed = 0.03          # Crawling speed when very close
        self.turn_speed = 0.4            # Turn speed for avoidance
        
        # More conservative distance thresholds
        self.emergency_distance = 0.15   # 15cm - STOP AND TURN
        self.danger_distance = 0.25      # 25cm - CRAWL FORWARD
        self.caution_distance = 0.40     # 40cm - SLOW FORWARD
        
        # LiDAR scanning
        self.scan_data = None
        self.closest_front_distance = float('inf')
        
        # Tree detection state
        self.trees_currently_detected = False
        self.last_tree_message_time = 0
        self.tree_detection_timeout = 0.5  # Consider trees gone after 0.5 seconds
        
        # Avoidance state
        self.in_avoidance_mode = False
        self.avoidance_start_time = 0
        self.avoidance_duration = 2.0
        
        # Safety counters
        self.emergency_stops = 0
        self.close_calls = 0
        
        # High-frequency navigation timer (10Hz for responsiveness)
        self.navigation_timer = self.create_timer(0.1, self.navigation_loop)
        
        self.get_logger().info('Safe Proximity Navigation initialized!')
        # self.get_logger().info('Slow speed: Normal=15cm/s, Slow=8cm/s, Crawl=3cm/s')
        # self.get_logger().info('Emergency: 15cm | DANGER: 25cm | CAUTION: 40cm')
        self.publish_status("Safe navigation ready - slow and steady")
        
    def publish_status(self, message):
        """Publish navigation status."""
        status_msg = String()
        status_msg.data = message
        self.status_pub.publish(status_msg)
        
    def scan_callback(self, msg):
        """Process LiDAR scan for immediate obstacle detection."""
        self.scan_data = msg
        self.update_front_distance()
        
        # IMMEDIATE safety check - if something very close, stop now!
        if self.closest_front_distance <= self.emergency_distance:
            self.emergency_stop()
    
    def update_front_distance(self):
        """Update closest obstacle distance in front of robot."""
        if self.scan_data is None:
            return
            
        # Check front 90 degrees (±45°) for obstacles
        ranges = self.scan_data.ranges
        total_ranges = len(ranges)
        
        # Calculate front indices (wider scan for safety)
        center_idx = total_ranges // 2
        front_span = total_ranges // 4  # ±45 degrees
        start_idx = max(0, center_idx - front_span)
        end_idx = min(total_ranges, center_idx + front_span)
        
        # Find minimum valid distance in front area
        front_ranges = ranges[start_idx:end_idx]
        valid_distances = []
        
        for distance in front_ranges:
            if (np.isfinite(distance) and 
                self.scan_data.range_min <= distance <= self.scan_data.range_max):
                valid_distances.append(distance)
        
        if valid_distances:
            self.closest_front_distance = min(valid_distances)
        else:
            self.closest_front_distance = float('inf')
    
    def tree_callback(self, msg):
        """Handle tree detection messages."""
        current_time = time.time()
        self.last_tree_message_time = current_time
        
        if "trees_detected:" in msg.data:
            num_trees = int(msg.data.split(':')[1])
            self.trees_currently_detected = True
            
            self.get_logger().info(f'TREES: {num_trees} detected | Front distance: {self.closest_front_distance*100:.1f}cm')
            
            # If trees detected and very close, immediate avoidance
            if self.closest_front_distance <= self.emergency_distance:
                self.start_emergency_avoidance()
    
    def emergency_stop(self):
        """Immediate emergency stop."""
        twist = Twist()  # All zeros = stop
        self.cmd_pub.publish(twist)
        
        self.emergency_stops += 1
        self.get_logger().warn(f'Emergency Stopping #{self.emergency_stops}! Obstacle at {self.closest_front_distance*100:.1f}cm')
        
        # Start avoidance immediately
        if not self.in_avoidance_mode:
            self.start_emergency_avoidance()
    
    def start_emergency_avoidance(self):
        """Start emergency avoidance maneuver."""
        self.in_avoidance_mode = True
        self.avoidance_start_time = time.time()
        self.close_calls += 1
        
        self.get_logger().warn(f'Emergency Avoidance #{self.close_calls}! Distance: {self.closest_front_distance*100:.1f}cm')
        self.publish_status(f"Emergency Avoidance - {self.closest_front_distance*100:.1f}cm")
    
    def navigation_loop(self):
        """Main navigation loop - runs at 10Hz for responsiveness."""
        current_time = time.time()
        
        # Check if tree detection messages are still coming
        time_since_tree_msg = current_time - self.last_tree_message_time
        if time_since_tree_msg > self.tree_detection_timeout:
            self.trees_currently_detected = False
        
        # Make navigation decision
        self.make_safe_navigation_decision()
        
    def make_safe_navigation_decision(self):
        """Make navigation decision prioritizing safety."""
        current_time = time.time()
        distance = self.closest_front_distance
        
        # Handle avoidance mode
        if self.in_avoidance_mode:
            if current_time - self.avoidance_start_time < self.avoidance_duration:
                self.execute_avoidance_turn()
                return
            else:
                # Check if it's safe to resume
                if distance > self.caution_distance:
                    self.in_avoidance_mode = False
                    self.get_logger().info('Avoidance completed - resuming forward motion')
                    self.publish_status("Avoidance completed")
                else:
                    # Still too close, extend avoidance
                    self.avoidance_start_time = current_time
                    self.get_logger().warn('Still too close, extending avoidance')
                    return
        
        # Normal navigation based on distance and tree detection
        if distance <= self.emergency_distance:
            # EMERGENCY: Stop and avoid
            self.emergency_stop()
            
        elif distance <= self.danger_distance:
            # DANGER: Crawl forward very slowly
            self.crawl_forward()
            
        elif distance <= self.caution_distance and self.trees_currently_detected:
            # CAUTION: Slow forward with trees nearby
            self.slow_forward()
            
        elif self.trees_currently_detected:
            # AWARE: Trees detected but not too close
            self.cautious_forward()
            
        else:
            # NORMAL: Path seems clear
            self.normal_forward()
    
    def execute_avoidance_turn(self):
        """Execute avoidance turn maneuver."""
        twist = Twist()
        twist.linear.x = 0.0  # Stop forward motion
        twist.angular.z = self.turn_speed  # Turn right
        self.cmd_pub.publish(twist)
        
    def crawl_forward(self):
        """Move forward very slowly."""
        twist = Twist()
        twist.linear.x = self.crawl_speed
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)
        
        self.publish_status(f"CRAWLING - {self.closest_front_distance*100:.1f}cm ahead")
        
    def slow_forward(self):
        """Move forward slowly."""
        twist = Twist()
        twist.linear.x = self.slow_speed
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)
        
        self.publish_status(f"SLOW - Trees detected, {self.closest_front_distance*100:.1f}cm ahead")
        
    def cautious_forward(self):
        """Move forward cautiously."""
        twist = Twist()
        twist.linear.x = self.normal_speed * 0.6  # 60% of normal speed
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)
        
        self.publish_status(f"CAUTIOUS - Trees nearby, {self.closest_front_distance*100:.1f}cm clear")
        
    def normal_forward(self):
        """Normal forward movement."""
        twist = Twist()
        twist.linear.x = self.normal_speed
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)
        
        if self.closest_front_distance < float('inf'):
            self.publish_status(f"NORMAL - Path clear, {self.closest_front_distance*100:.1f}cm ahead")
        else:
            self.publish_status("NORMAL - Clear path ahead")

def main(args=None):
    rclpy.init(args=args)
    
    try:
        navigator = BasicNavigation()
        navigator.get_logger().info('Starting SAFE proximity navigation...')
        # navigator.get_logger().info('Using SLOW speeds to prevent crashes')
        # navigator.get_logger().info('Safety first - better slow than crashed!')
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        navigator.get_logger().info('Safe navigation shutting down...')
    finally:
        if 'navigator' in locals():
            # Ensure robot stops
            stop_twist = Twist()
            navigator.cmd_pub.publish(stop_twist)
            navigator.get_logger().info(f'Session stats: {navigator.emergency_stops} emergency stops, {navigator.close_calls}')
            navigator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
