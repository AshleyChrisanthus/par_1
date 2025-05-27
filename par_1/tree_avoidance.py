#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Empty
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

class TreeAvoidanceController(Node):
    """
    Combined tree detection and avoidance controller.
    Detects brown cylinders (trees) and steers the robot away from them.
    """
    
    def __init__(self):
        super().__init__('tree_avoidance_controller')
        
        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/robot_status', 10)
        self.debug_img_pub = self.create_publisher(Image, '/tree_avoidance/debug_image', 10)
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/oak/rgb/image_raw',
            self.image_callback,
            10)
        
        self.start_sub = self.create_subscription(
            Empty, 
            '/trigger_start', 
            self.start_callback, 
            10)
        
        self.stop_sub = self.create_subscription(
            Empty, 
            '/trigger_stop', 
            self.stop_callback, 
            10)
        
        # OpenCV bridge
        self.bridge = CvBridge()
        
        # Tree detection parameters
        self.brown_hsv_lower = np.array([8, 50, 20])
        self.brown_hsv_upper = np.array([20, 255, 200])
        self.min_tree_area = 800
        self.max_tree_area = 8000
        self.min_aspect_ratio = 1.5
        self.max_aspect_ratio = 4.0
        
        # Movement parameters
        self.normal_speed = 0.3        # Normal forward speed
        self.turn_speed = 0.5          # Turn speed when avoiding
        self.safe_distance_pixels = 150 # How close is "too close" in pixels
        
        # State variables
        self.started = False
        self.avoiding_tree = False
        self.avoidance_start_time = 0
        self.avoidance_duration = 2.0  # Seconds to turn when avoiding
        
        # Image dimensions (will be set when first image arrives)
        self.image_width = 640
        self.image_height = 480
        self.image_center_x = 320
        
        # Detection tracking
        self.detection_count = 0
        self.frame_count = 0
        
        self.get_logger().info('Tree Avoidance Controller initialized!')
        self.get_logger().info('Send /trigger_start to begin autonomous navigation')
        self.publish_status("Waiting for start signal")
        
    def publish_status(self, message):
        """Publish status message."""
        msg = String()
        msg.data = message
        self.status_pub.publish(msg)
        
    def start_callback(self, msg):
        """Start autonomous navigation."""
        self.started = True
        self.get_logger().info("Starting autonomous tree avoidance navigation!")
        self.publish_status("Navigation started - avoiding trees")
        
    def stop_callback(self, msg):
        """Stop autonomous navigation."""
        self.started = False
        self.avoiding_tree = False
        self.stop_robot()
        self.get_logger().info("Stopped autonomous navigation")
        self.publish_status("Navigation stopped")
        
    def stop_robot(self):
        """Send stop command to robot."""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)
        
    def image_callback(self, msg):
        """Process camera image for tree detection and avoidance."""
        if not self.started:
            return
            
        try:
            self.frame_count += 1
            
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Update image dimensions
            self.image_height, self.image_width = cv_image.shape[:2]
            self.image_center_x = self.image_width // 2
            
            # Detect trees
            trees = self.detect_trees(cv_image)
            
            # Make navigation decision based on detections
            self.navigate_with_tree_avoidance(trees)
            
            # Publish debug image
            debug_img = self.create_debug_image(cv_image, trees)
            self.debug_img_pub.publish(
                self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8'))
                
        except Exception as e:
            self.get_logger().error(f'Error in image processing: {str(e)}')
            
    def detect_trees(self, image):
        """Detect brown cylindrical trees in image."""
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create brown mask
        brown_mask = cv2.inRange(hsv, self.brown_hsv_lower, self.brown_hsv_upper)
        
        # Clean up mask
        kernel = np.ones((3, 3), np.uint8)
        brown_mask = cv2.morphologyEx(brown_mask, cv2.MORPH_OPEN, kernel)
        brown_mask = cv2.morphologyEx(brown_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(brown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        trees = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if self.min_tree_area <= area <= self.max_tree_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / w if w > 0 else 0
                
                if self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio:
                    # Calculate how close this tree is (larger = closer)
                    tree_size = max(w, h)
                    trees.append((x, y, w, h, tree_size))
        
        return trees
    
    def navigate_with_tree_avoidance(self, trees):
        """Main navigation logic with tree avoidance."""
        current_time = time.time()
        
        # Check if we're currently in avoidance mode
        if self.avoiding_tree:
            # Continue avoidance maneuver for specified duration
            if current_time - self.avoidance_start_time < self.avoidance_duration:
                # Keep turning
                twist = Twist()
                twist.linear.x = 0.1  # Slow forward while turning
                twist.angular.z = self.turn_speed  # Turn right
                self.cmd_pub.publish(twist)
                self.publish_status("Avoiding tree - turning right")
                return
            else:
                # Avoidance complete
                self.avoiding_tree = False
                self.get_logger().info("Tree avoidance maneuver complete")
        
        # Analyze detected trees for threats
        dangerous_trees = []
        for tree in trees:
            x, y, w, h, size = tree
            tree_center_x = x + w // 2
            
            # Check if tree is in our path (center portion of image)
            path_width = self.image_width // 3  # Central third of image
            path_left = self.image_center_x - path_width // 2
            path_right = self.image_center_x + path_width // 2
            
            # Check if tree is in our path AND close enough to be dangerous
            if (path_left <= tree_center_x <= path_right and 
                size > self.safe_distance_pixels):
                dangerous_trees.append(tree)
        
        # Decision making
        if dangerous_trees:
            # TREE DETECTED IN PATH - AVOID!
            self.detection_count += 1
            self.get_logger().warn('🌳 TREE DETECTED! Initiating avoidance maneuver!')
            
            # Start avoidance
            self.avoiding_tree = True
            self.avoidance_start_time = current_time
            
            # Immediately start turning
            twist = Twist()
            twist.linear.x = 0.1
            twist.angular.z = self.turn_speed  # Turn right
            self.cmd_pub.publish(twist)
            
            self.publish_status(f"AVOIDING TREE #{self.detection_count}")
            
        else:
            # Path is clear - move forward normally
            twist = Twist()
            twist.linear.x = self.normal_speed
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)
            
            if trees:
                self.publish_status(f"Trees visible but not in path - continuing forward")
            else:
                self.publish_status("Path clear - moving forward")
    
    def create_debug_image(self, original_image, trees):
        """Create debug visualization."""
        debug_img = original_image.copy()
        
        # Draw path zone (where we check for dangerous trees)
        path_width = self.image_width // 3
        path_left = self.image_center_x - path_width // 2
        path_right = self.image_center_x + path_width // 2
        
        # Draw path zone as semi-transparent overlay
        overlay = debug_img.copy()
        cv2.rectangle(overlay, (path_left, 0), (path_right, self.image_height), 
                     (0, 255, 255), -1)  # Yellow zone
        cv2.addWeighted(overlay, 0.2, debug_img, 0.8, 0, debug_img)
        
        # Draw detected trees
        for tree in trees:
            x, y, w, h, size = tree
            tree_center_x = x + w // 2
            
            # Color code based on danger level
            if (path_left <= tree_center_x <= path_right and 
                size > self.safe_distance_pixels):
                color = (0, 0, 255)  # Red for dangerous trees
                label = "DANGER!"
            elif path_left <= tree_center_x <= path_right:
                color = (0, 165, 255)  # Orange for trees in path but far
                label = "In Path"
            else:
                color = (0, 255, 0)  # Green for safe trees
                label = "Safe"
            
            # Draw bounding box
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 2)
            
            # Draw center point
            cv2.circle(debug_img, (tree_center_x, y + h // 2), 5, color, -1)
            
            # Add label
            cv2.putText(debug_img, f"TREE - {label}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add status text
        status_text = []
        if self.avoiding_tree:
            status_text.append("STATUS: AVOIDING TREE")
        elif trees:
            status_text.append(f"STATUS: {len(trees)} TREES DETECTED")
        else:
            status_text.append("STATUS: PATH CLEAR")
        
        status_text.append(f"Frame: {self.frame_count}")
        status_text.append(f"Avoidances: {self.detection_count}")
        
        for i, text in enumerate(status_text):
            cv2.putText(debug_img, text, (10, 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return debug_img

def main(args=None):
    rclpy.init(args=args)
    
    try:
        controller = TreeAvoidanceController()
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        if 'controller' in locals():
            controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()