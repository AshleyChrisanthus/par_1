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

class TreeDetector(Node):
    """
    Enhanced tree detector that detects brown cylinders and provides both:
    - Pixel coordinates in camera image
    - Real-world coordinates using LiDAR data
    - Map coordinates using TF transforms
    """
    
    def __init__(self):
        super().__init__('tree_detector')
        
        # Create OpenCV bridge
        self.bridge = CvBridge()
        self.scan = None
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/oak/rgb/image_raw',  # Change this to match your camera topic
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
            
        self.debug_img_pub = self.create_publisher(
            Image,
            '/tree_detection/debug_image',
            10)
        
        # Marker publisher for visualization in RViz
        self.marker_pub = self.create_publisher(
            Marker,
            '/tree_markers',
            10)
        
        # TF setup for coordinate transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Tree detection parameters
        self.brown_hsv_lower = np.array([8, 50, 20])
        self.brown_hsv_upper = np.array([20, 255, 200])
        self.min_tree_area = 800
        self.max_tree_area = 8000
        self.min_aspect_ratio = 1.5
        self.max_aspect_ratio = 4.0
        
        # Detection tracking
        self.detection_count = 0
        self.frame_count = 0
        self.last_detection_frame = 0
        self.detected_trees_map = []  # Store map coordinates of detected trees
        
        self.get_logger().info('Tree Detector initialized!')
        self.get_logger().info('Detecting brown cylinders with world coordinates...')
        
    def scan_callback(self, msg):
        """Store the latest LiDAR scan for distance measurements."""
        self.scan = msg
        
    def image_callback(self, msg):
        """Process incoming image and detect trees with world coordinates."""
        if self.scan is None:
            self.get_logger().warn('No LiDAR scan data available yet')
            return
            
        try:
            self.frame_count += 1
            
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Detect trees in the image
            trees = self.detect_trees(cv_image)
            
            # Process each detected tree to get world coordinates
            if trees:
                self.process_tree_detections(cv_image, trees)
            
            # Every 60 frames, log status if no recent detections
            elif self.frame_count % 60 == 0:
                self.get_logger().info('Scanning for brown cylinder trees... No trees detected')
            
            # Publish debug view
            debug_img = self.create_debug_image(cv_image, trees)
            self.debug_img_pub.publish(
                self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8'))
                
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')
            
    def detect_trees(self, image):
        """Detect brown cylindrical trees in the given image."""
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for brown colors
        brown_mask = cv2.inRange(hsv, self.brown_hsv_lower, self.brown_hsv_upper)
        
        # Clean up the mask
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
                    contour_area = cv2.contourArea(contour)
                    bounding_area = w * h
                    fill_ratio = contour_area / bounding_area if bounding_area > 0 else 0
                    
                    if fill_ratio > 0.6:
                        trees.append((x, y, w, h))
        
        return trees
    
    def process_tree_detections(self, cv_image, trees):
        """Process detected trees to get world coordinates."""
        # Avoid spam - only process if it's been a few frames since last detection
        if self.frame_count - self.last_detection_frame > 10:
            self.detection_count += 1
            self.get_logger().info('🌳 TREE DETECTED!')
            self.get_logger().info(f'Found {len(trees)} tree(s). Total detections: {self.detection_count}')
            
            self.last_detection_frame = self.frame_count
        
        # Process each tree to get world coordinates
        for i, (x, y, w, h) in enumerate(trees):
            # Calculate center of tree in image
            center_x = x + w / 2
            center_y = y + h / 2
            
            # Get world coordinates for this tree
            world_coords = self.get_world_coordinates(center_x, cv_image.shape[1])
            
            if world_coords:
                lidar_x, lidar_y, map_x, map_y = world_coords
                
                # Log tree information
                aspect_ratio = h / w if w > 0 else 0
                area = w * h
                self.get_logger().info(f'  Tree #{i+1}:')
                self.get_logger().info(f'    Pixel coords: ({int(center_x)}, {int(center_y)})')
                self.get_logger().info(f'    Size: {w}x{h}px, Aspect Ratio: {aspect_ratio:.2f}, Area: {area}px²')
                self.get_logger().info(f'    LiDAR coords: ({lidar_x:.2f}, {lidar_y:.2f}) meters')
                self.get_logger().info(f'    Map coords: ({map_x:.2f}, {map_y:.2f}) meters')
                
                # Create and publish marker for RViz visualization
                self.publish_tree_marker(map_x, map_y, i)
                
                # Store tree location (avoid duplicates)
                new_tree = (map_x, map_y)
                if not self.is_duplicate_tree(new_tree):
                    self.detected_trees_map.append(new_tree)
        
        # Publish simple tree detection message for navigation
        tree_msg = String()
        tree_msg.data = f"trees_detected:{len(trees)}"
        self.tree_pub.publish(tree_msg)
    
    def get_world_coordinates(self, center_x, image_width):
        """
        Convert pixel coordinates to world coordinates using LiDAR data.
        
        Args:
            center_x: X pixel coordinate of tree center
            image_width: Width of the image in pixels
            
        Returns:
            Tuple of (lidar_x, lidar_y, map_x, map_y) or None if failed
        """
        try:
            # Normalize horizontal center_x to [0,1]
            normalized_x = center_x / image_width
            
            # Compute angle from LiDAR scan parameters
            angle = self.scan.angle_min + normalized_x * (self.scan.angle_max - self.scan.angle_min)
            
            # Get LiDAR range index
            index = int((angle - self.scan.angle_min) / self.scan.angle_increment)
            if index < 0 or index >= len(self.scan.ranges):
                self.get_logger().warn(f'LiDAR index {index} out of range')
                return None
            
            # Get distance from LiDAR
            distance = self.scan.ranges[index]
            if not np.isfinite(distance) or not (self.scan.range_min <= distance <= self.scan.range_max):
                self.get_logger().warn(f'Invalid LiDAR distance: {distance}')
                return None
            
            # Calculate coordinates in LiDAR frame
            lidar_x = distance * np.cos(angle)
            lidar_y = distance * np.sin(angle)
            
            # Build point in LiDAR frame
            point_lidar = PointStamped()
            point_lidar.header.frame_id = self.scan.header.frame_id
            point_lidar.header.stamp = self.scan.header.stamp
            point_lidar.point.x = lidar_x
            point_lidar.point.y = lidar_y
            point_lidar.point.z = 0.0
            
            # Transform to map frame
            map_coords = self.transform_to_map(point_lidar)
            if map_coords:
                map_x, map_y = map_coords
                return (lidar_x, lidar_y, map_x, map_y)
            else:
                return (lidar_x, lidar_y, None, None)
                
        except Exception as e:
            self.get_logger().error(f'Error getting world coordinates: {str(e)}')
            return None
    
    def transform_to_map(self, point_lidar):
        """Transform point from LiDAR frame to map frame."""
        try:
            timeout = rclpy.duration.Duration(seconds=0.1)
            if self.tf_buffer.can_transform('map', point_lidar.header.frame_id, rclpy.time.Time(), timeout=timeout):
                # Transform to map frame
                point_map = tf2_geometry_msgs.do_transform_point(
                    point_lidar,
                    self.tf_buffer.lookup_transform('map', point_lidar.header.frame_id, rclpy.time.Time(), timeout=timeout)
                )
                return (point_map.point.x, point_map.point.y)
            else:
                self.get_logger().warn('TF transform to map not ready')
                return None
        except Exception as e:
            self.get_logger().error(f'TF transform exception: {str(e)}')
            return None
    
    def is_duplicate_tree(self, new_tree, threshold=0.5):
        """Check if this tree is too close to an already detected tree."""
        new_x, new_y = new_tree
        for existing_x, existing_y in self.detected_trees_map:
            distance = np.sqrt((new_x - existing_x)**2 + (new_y - existing_y)**2)
            if distance < threshold:  # Trees closer than 0.5m are considered duplicates
                return True
        return False
    
    def publish_tree_marker(self, map_x, map_y, tree_id):
        """Publish a marker for RViz visualization."""
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'detected_trees'
        marker.id = tree_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        
        # Set position
        marker.pose.position.x = map_x
        marker.pose.position.y = map_y
        marker.pose.position.z = 0.5  # Half-height of cylinder
        marker.pose.orientation.w = 1.0
        
        # Set size (cylinder representing tree)
        marker.scale.x = 0.2  # Diameter
        marker.scale.y = 0.2  # Diameter  
        marker.scale.z = 1.0  # Height
        
        # Set color (brown)
        marker.color.r = 0.6
        marker.color.g = 0.3
        marker.color.b = 0.1
        marker.color.a = 0.8
        
        self.marker_pub.publish(marker)
    
    def create_debug_image(self, original_image, trees):
        """Create debug visualization."""
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
        
        cv2.putText(debug_img, f"Trees: {len(trees)}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(debug_img, f"Total Detected: {len(self.detected_trees_map)}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return debug_img

def main(args=None):
    rclpy.init(args=args)
    
    try:
        tree_detector = TreeDetector()
        rclpy.spin(tree_detector)
    except KeyboardInterrupt:
        tree_detector.get_logger().info('Tree detector shutting down...')
    finally:
        if 'tree_detector' in locals():
            tree_detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
