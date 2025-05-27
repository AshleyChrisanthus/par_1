#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
from geometry_msgs.msg import Twist
import time

class BasicNavigationController(Node):
    """
    Basic navigation controller that avoids trees detected by the tree detector.
    Listens to tree detection data and steers the robot away from obstacles.
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
        
        # Navigation parameters
        self.normal_speed = 0.25       # Normal forward speed (m/s)
        self.slow_speed = 0.1          # Speed when trees detected but not dangerous
        self.turn_speed = 0.6          # Angular speed for avoidance turns (rad/s)
        
        # Image parameters (assuming standard camera resolution)
        self.image_width = 640
        self.image_center = self.image_width // 2  # Center of image = 320 pixels
        self.danger_zone_width = 200   # Width of danger zone in pixels (center ±100)
        self.danger_zone_left = self.image_center - self.danger_zone_width // 2  # 220
        self.danger_zone_right = self.image_center + self.danger_zone_width // 2  # 420
        
        # Tree size thresholds (in pixels)
        self.close_tree_size = 70      # Tree width/height > 70px = very close
        self.medium_tree_size = 40     # Tree width/height > 40px = moderately close
        
        # State management
        self.avoiding_obstacle = False
        self.avoidance_start_time = 0
        self.avoidance_duration = 2.5  # Seconds to turn when avoiding
        
        # Statistics
        self.avoidance_count = 0
        self.last_tree_detection_time = 0
        
        # Timer for regular forward movement when no trees detected
        self.movement_timer = self.create_timer(0.1, self.regular_movement_check)
        
        self.get_logger().info('Basic Navigation Controller initialized!')
        self.get_logger().info('Listening for tree detections and ready to navigate...')
        self.get_logger().info(f'Danger zone: pixels {self.danger_zone_left} to {self.danger_zone_right} (center {self.image_center})')
        self.publish_status("Navigation ready - moving forward until trees detected")
        
    def publish_status(self, message):
        """Publish navigation status."""
        status_msg = String()
        status_msg.data = message
        self.status_pub.publish(status_msg)
        
    def tree_callback(self, msg):
        """Handle tree detection messages from tree detector."""
        # Parse the message - format should be "trees_detected:N" 
        # But your tree_detector publishes position data, so let's handle both
        
        current_time = time.time()
        self.last_tree_detection_time = current_time
        
        # Simple parsing - if we get any tree message, trees are detected
        if "tree" in msg.data.lower():
            self.get_logger().info(f'Received tree detection: {msg.data}')
            
            # For now, assume any tree detection means we should be cautious
            # In the future, you could parse the actual positions from the message
            self.handle_tree_detection(msg.data)
        
    def handle_tree_detection(self, tree_data):
        """Process tree detection and decide on navigation action."""
        
        # Simple logic: if trees are detected, start avoidance
        if not self.avoiding_obstacle:
            self.get_logger().warn('🚨 TREES DETECTED - Starting avoidance maneuver!')
            self.start_avoidance_maneuver()
        
    def start_avoidance_maneuver(self):
        """Begin obstacle avoidance procedure."""
        self.avoiding_obstacle = True
        self.avoidance_start_time = time.time()
        self.avoidance_count += 1
        
        self.get_logger().info(f'🔄 AVOIDANCE #{self.avoidance_count} - Turning right to avoid trees')
        self.publish_status(f"AVOIDING TREES - Maneuver #{self.avoidance_count}")
        
        # Immediately start turning
        self.execute_avoidance_turn()
        
    def execute_avoidance_turn(self):
        """Execute the avoidance turn."""
        twist = Twist()
        twist.linear.x = 0.05          # Very slow forward motion while turning
        twist.angular.z = self.turn_speed  # Turn right (positive = left, negative = right)
        self.cmd_pub.publish(twist)
        
    def regular_movement_check(self):
        """Regular timer callback to handle movement when not actively avoiding."""
        current_time = time.time()
        
        # Check if we're in avoidance mode
        if self.avoiding_obstacle:
            # Continue avoidance for the specified duration
            if current_time - self.avoidance_start_time < self.avoidance_duration:
                self.execute_avoidance_turn()
                return
            else:
                # Avoidance complete
                self.avoiding_obstacle = False
                self.get_logger().info('✅ Avoidance maneuver completed - resuming forward motion')
                self.publish_status("Avoidance completed - moving forward")
        
        # Normal forward movement
        # Check if we've received tree detections recently
        time_since_last_detection = current_time - self.last_tree_detection_time
        
        if time_since_last_detection < 1.0:  # Tree detected within last 1 second
            # Move slowly when trees are nearby
            twist = Twist()
            twist.linear.x = self.slow_speed
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)
            self.publish_status("Trees nearby - moving slowly")
            
        elif time_since_last_detection < 3.0:  # Tree detected within last 3 seconds
            # Move at normal speed but stay alert
            twist = Twist()
            twist.linear.x = self.normal_speed
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)
            self.publish_status("Recent tree detection - normal speed")
            
        else:  # No recent tree detections
            # Full speed ahead
            twist = Twist()
            twist.linear.x = self.normal_speed
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)
            self.publish_status("Path clear - full speed forward")

def main(args=None):
    rclpy.init(args=args)
    
    try:
        navigator = BasicNavigationController()
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        navigator.get_logger().info('Navigation controller shutting down...')
    finally:
        # Stop the robot
        if 'navigator' in locals():
            stop_twist = Twist()
            navigator.cmd_pub.publish(stop_twist)
            navigator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()