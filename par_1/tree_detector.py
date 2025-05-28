#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros
import tf2_geometry_msgs

class TreeDetector(Node):
    """
    Final complete tree detector that:
    - Continuously detects ALL brown cylinder trees
    - Shows "no trees detected" messages  
    - Displays markers in RViz (like tennis ball detector)
    - Provides pixel, LiDAR, and map coordinates
    - Fast processing for real-time navigation
    """
    
    def __init__(self):
        super().__init__('tree_detector')
        
        # Create OpenCV bridge
        self.bridge = CvBridge()
        self.scan = None
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            'oak/rgb/image_raw',  # Change to your camera topic if needed
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
        
        # Use single Marker publisher like tennis ball detector (not MarkerArray)
        self.marker_pub = self.create_publisher(
            Marker,
            '/tree_markers',
            10)
        
        self.debug_img_pub = self.create_publisher(
            Image,
            '/tree_detection/debug_image',
            10)
        
        # TF setup - same as tennis ball detector
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Fast detection parameters (like tennis ball detector)
        self.brown_hsv_lower = np.array([8, 50, 20])   # Brown color range
        self.brown_hsv_upper = np.array([20, 255, 200])
        self.min_area = 200  # Same as tennis ball detector
        
        # Detection tracking
        self.detection_count = 0
        self.frame_count = 0
        self.last_detection_frame = 0
        self.last_no_detection_log = 0
        self.detected_trees_positions = []  # Store all detected tree positions
        
        self.get_logger().info('Final Tree Detector initialized!')
        self.get_logger().info('Detecting brown cylinder trees with full coordinate tracking')
        self.get_logger().info('Markers will appear in RViz on topic: /tree_markers')
        
    def scan_callback(self, msg):
        """Store LiDAR scan."""
        self.scan = msg
        
    def image_callback(self, msg):
        """Main detection callback - processes every frame."""
        if self.scan is None:
            self.get_logger().warn('Waiting for LiDAR scan data...')
            return
            
        try:
            self.frame_count += 1
            
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Detect all trees using fast method
            trees = self.detect_all_trees_fast(cv_image)
            
            if trees:
                # TREES DETECTED
                self.handle_trees_detected(cv_image, trees)
            else:
                # NO TREES DETECTED
                self.handle_no_trees_detected()
            
            # Always publish debug image
            debug_img = self.create_debug_image(cv_image, trees)
            self.debug_img_pub.publish(
                self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8'))
                
        except Exception as e:
            self.get_logger().error(f'Error in tree detection: {str(e)}')
    
    def detect_all_trees_fast(self, image):
        """
        Fast detection of ALL brown cylinder trees.
        Uses same approach as tennis ball detector but for multiple objects.
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create brown color mask
        mask = cv2.inRange(hsv, self.brown_hsv_lower, self.brown_hsv_upper)
        
        # Minimal morphology operation (same as tennis ball detector)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        # Filter contours by area (same simple check as tennis ball)
        trees = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area:  # Simple area filter like tennis ball
                x, y, w, h = cv2.boundingRect(contour)
                trees.append((x, y, w, h, area))
        
        return trees
    
    def handle_trees_detected(self, cv_image, trees):
        """Handle when trees are detected."""
        # Publish detection message for navigation (every frame for real-time response)
        tree_msg = String()
        tree_msg.data = f"trees_detected:{len(trees)}"
        self.tree_pub.publish(tree_msg)
        
        # Log detection info (but avoid spam - every 20 frames)
        if self.frame_count - self.last_detection_frame > 20:
            self.detection_count += 1
            self.get_logger().info(f'TREES DETECTED! Found {len(trees)} tree(s) | Total detection events: {self.detection_count}')
            self.last_detection_frame = self.frame_count
            
            # Process each tree for detailed info and markers
            self.process_all_trees_detailed(cv_image, trees)
    
    def handle_no_trees_detected(self):
        """Handle when no trees are detected."""
        # Log "no trees detected" message every 60 frames (about 2 seconds)
        if self.frame_count - self.last_no_detection_log > 60:
            self.get_logger().info('No trees detected - scanning for brown cylinders...')
            self.last_no_detection_log = self.frame_count
    
    def process_all_trees_detailed(self, cv_image, trees):
        """Process all detected trees for detailed coordinates and markers."""
        for i, (x, y, w, h, area) in enumerate(trees):
            # Calculate tree center in image
            center_x = x + w / 2
            center_y = y + h / 2
            
            # Get world coordinates using same method as tennis ball detector
            world_coords = self.get_world_coordinates(center_x, cv_image.shape[1])
            
            if world_coords:
                lidar_x, lidar_y, map_x, map_y = world_coords
                
                # Log detailed tree information
                aspect_ratio = h / w if w > 0 else 0
                self.get_logger().info(f'    Tree #{i+1}:')
                self.get_logger().info(f'    Pixel coordinates: ({int(center_x)}, {int(center_y)})')
                self.get_logger().info(f'    Size: {w}x{h}px | Aspect ratio: {aspect_ratio:.2f} | Area: {area}px²')
                self.get_logger().info(f'    LiDAR coordinates: ({lidar_x:.2f}, {lidar_y:.2f}) meters')
                self.get_logger().info(f'    Map coordinates: ({map_x:.2f}, {map_y:.2f}) meters')
                
                # Publish RViz marker (same approach as tennis ball detector)
                self.publish_tree_marker(map_x, map_y, i)
                
                # Store tree position
                tree_position = (map_x, map_y)
                if not self.is_duplicate_detection(tree_position):
                    self.detected_trees_positions.append(tree_position)
    
    def get_world_coordinates(self, center_x, image_width):
        """
        Convert pixel coordinates to world coordinates.
        Uses exact same method as tennis ball detector.
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
            
            distance = self.scan.ranges[index]
            if not np.isfinite(distance) or not (self.scan.range_min <= distance <= self.scan.range_max):
                self.get_logger().warn(f'Invalid LiDAR distance: {distance:.2f}')
                return None
            
            # Build point in LiDAR frame (same as tennis ball detector)
            point_lidar = PointStamped()
            point_lidar.header.frame_id = self.scan.header.frame_id
            point_lidar.header.stamp = self.scan.header.stamp
            point_lidar.point.x = distance * np.cos(angle)
            point_lidar.point.y = distance * np.sin(angle)
            point_lidar.point.z = 0.0
            
            # Transform to map frame (same method as tennis ball detector)
            map_coords = self.transform_to_map(point_lidar)
            if map_coords:
                map_x, map_y = map_coords
                return (point_lidar.point.x, point_lidar.point.y, map_x, map_y)
            else:
                return (point_lidar.point.x, point_lidar.point.y, None, None)
                
        except Exception as e:
            self.get_logger().error(f'Error getting world coordinates: {str(e)}')
            return None
    
    def transform_to_map(self, point_lidar):
        """Transform point to map frame - same as tennis ball detector."""
        try:
            # Use proper timeout handling (same as tennis ball)
            timeout = rclpy.duration.Duration(seconds=0.1)
            if self.tf_buffer.can_transform('map', point_lidar.header.frame_id, rclpy.time.Time(), timeout=timeout):
                # Use tf2_geometry_msgs for proper transformation (same as tennis ball)
                point_map = tf2_geometry_msgs.do_transform_point(
                    point_lidar,
                    self.tf_buffer.lookup_transform('map', point_lidar.header.frame_id, rclpy.time.Time(), timeout=timeout)
                )
                return (point_map.point.x, point_map.point.y)
            else:
                self.get_logger().warn('TF transform not ready')
                return None
        except Exception as e:
            self.get_logger().error(f'TF exception: {e}')
            return None
    
    def publish_tree_marker(self, map_x, map_y, tree_id):
        """
        Publish marker for RViz visualization.
        Uses same approach as tennis ball detector but for trees.
        """
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'detected_trees'
        marker.id = tree_id
        marker.type = Marker.CYLINDER  # Cylinder for tree 
        marker.action = Marker.ADD
        
        # Set position in map frame
        marker.pose.position.x = map_x
        marker.pose.position.y = map_y
        marker.pose.position.z = 0.5  # Half height of cylinder
        marker.pose.orientation.w = 1.0
        
        # Set scale (tree dimensions)
        marker.scale.x = 0.25  # 25cm diameter
        marker.scale.y = 0.25  # 25cm diameter
        marker.scale.z = 1.0   # 1m height
        
        # Set brown color (vs green for tennis ball)
        marker.color.r = 0.6
        marker.color.g = 0.3
        marker.color.b = 0.1
        marker.color.a = 0.9
        
        # Publish marker (same as tennis ball detector)
        self.marker_pub.publish(marker)
        self.get_logger().info(f'Published tree marker #{tree_id} at map coordinates ({map_x:.2f}, {map_y:.2f})')
    
    def is_duplicate_detection(self, new_position, threshold=0.5):
        """Check if tree position is too close to already detected trees."""
        new_x, new_y = new_position
        for existing_x, existing_y in self.detected_trees_positions:
            distance = np.sqrt((new_x - existing_x)**2 + (new_y - existing_y)**2)
            if distance < threshold:  # Trees within 50cm considered duplicates
                return True
        return False
    
    def create_debug_image(self, original_image, trees):
        """Create debug visualization showing all detected trees."""
        debug_img = original_image.copy()
        
        # Draw all detected trees
        for i, (x, y, w, h, area) in enumerate(trees):
            # Use different colors for each tree
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
            color = colors[i % len(colors)]
            
            # Draw bounding rectangle
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 3)
            
            # Draw center point
            center_x, center_y = x + w//2, y + h//2
            cv2.circle(debug_img, (center_x, center_y), 8, color, -1)
            
            # Add tree label
            label = f"TREE #{i+1}"
            cv2.putText(debug_img, label, (x, y - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Add area information
            cv2.putText(debug_img, f"{area}px²", (x, y + h + 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add status information overlay
        cv2.putText(debug_img, f"🌳 Trees in view: {len(trees)}", (10, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.putText(debug_img, f"Frame: {self.frame_count}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(debug_img, f"Detection events: {self.detection_count}", (10, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(debug_img, f"Total trees found: {len(self.detected_trees_positions)}", (10, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add detection status
        if trees:
            cv2.putText(debug_img, "TREES DETECTED!", (10, 170), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        else:
            cv2.putText(debug_img, "Scanning for trees...", (10, 170), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return debug_img

def main(args=None):
    rclpy.init(args=args)
    
    try:
        tree_detector = TreeDetector()
        tree_detector.get_logger().info('Starting final tree detector...')
        # tree_detector.get_logger().info('To view markers: Open RViz → Add Marker display → Set topic to /tree_markers')
        rclpy.spin(tree_detector)
    except KeyboardInterrupt:
        tree_detector.get_logger().info('Final tree detector shutting down...')
    finally:
        if 'tree_detector' in locals():
            tree_detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
