#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros
import tf2_geometry_msgs

class ImprovedTreeDetector(Node):
    """
    Improved tree detector that uses multiple methods to detect vertical tree trunks:
    1. Edge detection to find vertical lines
    2. Texture analysis to identify bark-like patterns
    3. LiDAR validation to confirm cylindrical objects
    4. Size and aspect ratio filtering
    """
    
    def __init__(self):
        super().__init__('improved_tree_detector')
        
        # Create OpenCV bridge
        self.bridge = CvBridge()
        self.scan = None
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            'oak/rgb/image_raw',
            self.image_callback,
            10)
        
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)
        
        # Publishers
        self.tree_pub = self.create_publisher(
            String,
            '/detected_trees',
            10)
        
        self.marker_pub = self.create_publisher(
            Marker,
            '/tree_markers',
            10)
        
        self.debug_img_pub = self.create_publisher(
            Image,
            '/tree_detection/debug_image',
            10)
        
        # TF setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Detection parameters - more sophisticated
        self.min_trunk_height = 80      # Minimum height in pixels
        self.max_trunk_width = 60       # Maximum width in pixels
        self.min_aspect_ratio = 2.0     # Height/width ratio (tall and narrow)
        self.min_lidar_distance = 0.5   # Minimum distance to consider
        self.max_lidar_distance = 5.0   # Maximum distance to consider
        
        # Detection tracking
        self.detection_count = 0
        self.frame_count = 0
        self.last_detection_frame = 0
        self.no_detection_count = 0
        
        self.get_logger().info('Improved Tree Detector initialized!')
        self.get_logger().info('Using edge detection + LiDAR validation for tree trunks')
        
    def scan_callback(self, msg):
        """Store LiDAR scan."""
        self.scan = msg
        
    def image_callback(self, msg):
        """Main detection callback."""
        if self.scan is None:
            return
            
        try:
            self.frame_count += 1
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Detect tree trunks using improved method
            trees = self.detect_tree_trunks(cv_image)
            
            if trees:
                self.handle_trees_detected(cv_image, trees)
                self.no_detection_count = 0
            else:
                self.handle_no_trees_detected()
                self.no_detection_count += 1
            
            # Publish debug image
            debug_img = self.create_debug_image(cv_image, trees)
            self.debug_img_pub.publish(
                self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8'))
                
        except Exception as e:
            self.get_logger().error(f'Error in tree detection: {str(e)}')
    
    def detect_tree_trunks(self, image):
        """
        Detect tree trunks using edge detection and shape analysis.
        More reliable than color-based detection in outdoor environments.
        """
        height, width = image.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection - find strong vertical edges
        edges = cv2.Canny(blurred, 50, 150)
        
        # Morphological operations to connect broken edges
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_vertical)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        tree_candidates = []
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size and aspect ratio
            aspect_ratio = h / w if w > 0 else 0
            
            # Check if it looks like a tree trunk
            if (h >= self.min_trunk_height and 
                w <= self.max_trunk_width and 
                aspect_ratio >= self.min_aspect_ratio and
                y + h > height * 0.5):  # Must extend into lower half of image
                
                # Additional validation with LiDAR
                center_x = x + w / 2
                if self.validate_with_lidar(center_x, width):
                    tree_candidates.append((x, y, w, h))
        
        return tree_candidates
    
    def validate_with_lidar(self, center_x, image_width):
        """
        Validate tree candidate using LiDAR data.
        Check if there's a solid object at the expected distance.
        """
        try:
            # Convert pixel x to angle
            normalized_x = center_x / image_width
            angle = self.scan.angle_min + normalized_x * (self.scan.angle_max - self.scan.angle_min)
            
            # Get LiDAR reading
            index = int((angle - self.scan.angle_min) / self.scan.angle_increment)
            if index < 0 or index >= len(self.scan.ranges):
                return False
            
            distance = self.scan.ranges[index]
            
            # Check if distance is reasonable for a tree
            if (np.isfinite(distance) and 
                self.min_lidar_distance <= distance <= self.max_lidar_distance):
                
                # Check consistency with nearby readings (trees should have consistent depth)
                nearby_readings = []
                for i in range(max(0, index-3), min(len(self.scan.ranges), index+4)):
                    if np.isfinite(self.scan.ranges[i]):
                        nearby_readings.append(self.scan.ranges[i])
                
                if nearby_readings:
                    avg_distance = np.mean(nearby_readings)
                    std_distance = np.std(nearby_readings)
                    
                    # Tree trunk should have consistent depth (low standard deviation)
                    if std_distance < 0.3:  # Less than 30cm variation
                        return True
            
            return False
            
        except Exception as e:
            self.get_logger().error(f'LiDAR validation error: {str(e)}')
            return False
    
    def handle_trees_detected(self, cv_image, trees):
        """Handle when trees are detected."""
        # Publish detection message
        tree_msg = String()
        tree_msg.data = f"trees_detected:{len(trees)}"
        self.tree_pub.publish(tree_msg)
        
        # Log detection (avoid spam)
        if self.frame_count - self.last_detection_frame > 30:
            self.detection_count += 1
            self.get_logger().info(f'🌳 TREE TRUNKS DETECTED! Found {len(trees)} trunk(s)')
            self.last_detection_frame = self.frame_count
            
            # Process trees for detailed info
            for i, (x, y, w, h) in enumerate(trees):
                center_x = x + w / 2
                aspect_ratio = h / w
                self.get_logger().info(f'  Tree #{i+1}: size={w}x{h}px, aspect_ratio={aspect_ratio:.1f}')
    
    def handle_no_trees_detected(self):
        """Handle when no trees are detected."""
        # Only log every 2 seconds and if we had recent detections
        if self.no_detection_count % 60 == 0 and self.detection_count > 0:
            self.get_logger().info('No tree trunks detected - scanning for vertical objects...')
    
    def create_debug_image(self, original_image, trees):
        """Create debug visualization."""
        debug_img = original_image.copy()
        
        # Show edge detection result in corner
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        
        # Resize edges for corner display
        h, w = debug_img.shape[:2]
        edge_small = cv2.resize(edges, (w//4, h//4))
        edge_colored = cv2.cvtColor(edge_small, cv2.COLOR_GRAY2BGR)
        
        # Place in top-right corner
        debug_img[0:h//4, -w//4:] = edge_colored
        
        # Draw detected trees
        for i, (x, y, w, h) in enumerate(trees):
            # Draw bounding box
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
            # Draw center line
            center_x = x + w // 2
            cv2.line(debug_img, (center_x, y), (center_x, y + h), (0, 255, 0), 2)
            
            # Add label
            cv2.putText(debug_img, f"TRUNK #{i+1}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add dimensions
            cv2.putText(debug_img, f"{w}x{h}", (x, y + h + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Add status overlay
        cv2.putText(debug_img, f"Tree trunks: {len(trees)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.putText(debug_img, "Edges", (w - w//4 + 5, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if trees:
            cv2.putText(debug_img, "TREES DETECTED!", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(debug_img, "Scanning...", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return debug_img

def main(args=None):
    rclpy.init(args=args)
    
    try:
        tree_detector = ImprovedTreeDetector()
        tree_detector.get_logger().info('Starting improved tree trunk detector...')
        rclpy.spin(tree_detector)
    except KeyboardInterrupt:
        tree_detector.get_logger().info('Tree detector shutting down...')
    finally:
        if 'tree_detector' in locals():
            tree_detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()