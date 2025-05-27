#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np

class TreeDetector(Node):
    """
    Node responsible for detecting trees represented by brown cylinders.
    
    Detects brown cylindrical objects (like water bottles) that represent trees
    and logs "tree detected" to the terminal.
    """
    
    def __init__(self):
        super().__init__('tree_detector')
        
        # Create subscribers and publishers
        self.image_sub = self.create_subscription(
            Image,
            '/oak/rgb/image_raw',
            self.image_callback,
            10)
        
        self.tree_pub = self.create_publisher(
            String,
            '/detected_trees',
            10)
            
        self.debug_img_pub = self.create_publisher(
            Image,
            '/tree_detection/debug_image',
            10)
        
        # Create OpenCV bridge
        self.bridge = CvBridge()
        
        # Parameters for brown cylinder detection
        self.declare_parameter('brown_hsv_lower', [8, 50, 20])     # Lower brown range in HSV
        self.declare_parameter('brown_hsv_upper', [20, 255, 200])  # Upper brown range in HSV
        self.declare_parameter('min_tree_area', 800)               # Minimum area for cylinder detection
        self.declare_parameter('max_tree_area', 8000)              # Maximum area for cylinder detection
        self.declare_parameter('min_aspect_ratio', 1.5)            # Height/width ratio for cylinders
        self.declare_parameter('max_aspect_ratio', 4.0)            # Max height/width ratio
        self.declare_parameter('debug_view', True)
        
        # Get parameters
        self.brown_hsv_lower = np.array(self.get_parameter('brown_hsv_lower').value)
        self.brown_hsv_upper = np.array(self.get_parameter('brown_hsv_upper').value)
        self.min_tree_area = self.get_parameter('min_tree_area').value
        self.max_tree_area = self.get_parameter('max_tree_area').value
        self.min_aspect_ratio = self.get_parameter('min_aspect_ratio').value
        self.max_aspect_ratio = self.get_parameter('max_aspect_ratio').value
        self.debug_view = self.get_parameter('debug_view').value
        
        # Detection tracking
        self.detection_count = 0
        self.frame_count = 0
        self.last_detection_frame = 0
        
        self.get_logger().info('Tree detector initialized! Looking for brown cylinders...')
        
    def image_callback(self, msg):
        """Process incoming image and detect brown cylindrical trees."""
        try:
            self.frame_count += 1
            
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Detect trees in the image
            trees = self.detect_trees(cv_image)
            
            # If trees detected, log to terminal and publish
            if trees:
                # Avoid spam - only log if it's been a few frames since last detection
                if self.frame_count - self.last_detection_frame > 10:
                    self.detection_count += 1
                    self.get_logger().info('TREE DETECTED!')
                    self.get_logger().info(f'Found {len(trees)} tree(s). Total detections: {self.detection_count}')
                    
                    # Log details of each detected tree
                    for i, (x, y, w, h) in enumerate(trees):
                        aspect_ratio = h / w if w > 0 else 0
                        area = w * h
                        self.get_logger().info(f'  Tree #{i+1}: Position ({int(x)}, {int(y)}), Size: {w}x{h}px, Aspect Ratio: {aspect_ratio:.2f}, Area: {area}px²')
                    
                    self.last_detection_frame = self.frame_count
                
                # Publish tree detection message
                tree_msg = String()
                tree_msg.data = f"trees_detected:{len(trees)}"
                self.tree_pub.publish(tree_msg)
            
            # Every 60 frames, log status if no recent detections
            elif self.frame_count % 60 == 0:
                self.get_logger().info('Scanning for brown cylinder trees... No trees detected in recent frames')
            
            # Publish debug view if enabled  
            if self.debug_view:
                debug_img = self.create_debug_image(cv_image, trees)
                self.debug_img_pub.publish(
                    self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8'))
                
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')
            
    def detect_trees(self, image):
        """
        Detect brown cylindrical trees in the given image.
        
        Args:
            image: OpenCV BGR image
            
        Returns:
            List of tuples (x, y, width, height) for each detected tree
        """
        # Convert to HSV color space for better brown color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for brown colors
        brown_mask = cv2.inRange(hsv, self.brown_hsv_lower, self.brown_hsv_upper)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        brown_mask = cv2.morphologyEx(brown_mask, cv2.MORPH_OPEN, kernel)
        brown_mask = cv2.morphologyEx(brown_mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply slight gaussian blur to reduce noise
        brown_mask = cv2.GaussianBlur(brown_mask, (3, 3), 0)
        
        # Find contours of brown objects
        contours, _ = cv2.findContours(brown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        trees = []
        
        for contour in contours:
            # Calculate contour area
            area = cv2.contourArea(contour)
            
            # Filter by area (must be reasonable size for a cylinder/bottle)
            if self.min_tree_area <= area <= self.max_tree_area:
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate aspect ratio (height/width)
                aspect_ratio = h / w if w > 0 else 0
                
                # Check if aspect ratio fits a cylinder (taller than wide)
                if self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio:
                    
                    # Additional check: contour should be somewhat rectangular
                    # (cylinders appear as rectangles from the side)
                    contour_area = cv2.contourArea(contour)
                    bounding_area = w * h
                    fill_ratio = contour_area / bounding_area if bounding_area > 0 else 0
                    
                    # Cylinder should fill most of its bounding rectangle
                    if fill_ratio > 0.6:  # At least 60% filled
                        trees.append((x, y, w, h))
        
        return trees
    
    def create_debug_image(self, original_image, trees):
        """Create debug visualization showing detected trees."""
        debug_img = original_image.copy()
        
        # Draw detected trees
        for (x, y, w, h) in trees:
            # Draw bounding rectangle in bright green
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw center point
            center_x, center_y = x + w//2, y + h//2
            cv2.circle(debug_img, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # Add label
            aspect_ratio = h / w if w > 0 else 0
            label = f"TREE (AR:{aspect_ratio:.1f})"
            cv2.putText(debug_img, label, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add frame info
        cv2.putText(debug_img, f"Frame: {self.frame_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(debug_img, f"Trees Detected: {len(trees)}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(debug_img, f"Total Detections: {self.detection_count}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return debug_img

def main(args=None):
    rclpy.init(args=args)
    tree_detector = TreeDetector()
    
    try:
        rclpy.spin(tree_detector)
    except KeyboardInterrupt:
        tree_detector.get_logger().info('Tree detector shutting down...')
    finally:
        tree_detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()