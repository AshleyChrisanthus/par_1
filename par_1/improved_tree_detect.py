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
from collections import deque
import time

class TreeTracker:
    """Individual tree tracker with persistent ID"""
    def __init__(self, tree_id, detection):
        self.id = tree_id
        self.position = (detection['center_x'], detection['center_y'])
        self.last_seen_frame = 0
        self.detection_count = 1
        self.total_detections = 1
        self.is_stable = False
        self.map_position = None
        self.distance = detection.get('distance', None)
        
    def update(self, detection, frame_num):
        """Update tracker with new detection"""
        self.position = (detection['center_x'], detection['center_y'])
        self.last_seen_frame = frame_num
        self.detection_count += 1
        self.total_detections += 1
        self.distance = detection.get('distance', None)
        
        # Consider stable after 3 detections (like original)
        if self.detection_count >= 3:
            self.is_stable = True
    
    def is_match(self, detection, max_distance=80):
        """Check if detection matches this tracker"""
        dist = np.sqrt((self.position[0] - detection['center_x'])**2 + 
                      (self.position[1] - detection['center_y'])**2)
        return dist <= max_distance
    
    def is_active(self, current_frame, max_frames_missing=10):
        """Check if tracker is still active (recently seen)"""
        return (current_frame - self.last_seen_frame) <= max_frames_missing

class ImprovedTreeDetector(Node):
    """
    Improved tree detector - based on original working approach with intelligent background filtering
    Maintains original detection sensitivity while filtering real-world background
    """
    
    def __init__(self):
        super().__init__('improved_tree_detect')
        
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
        self.tree_pub = self.create_publisher(String, '/detected_trees', 10)
        self.marker_array_pub = self.create_publisher(MarkerArray, '/tree_markers', 10)
        self.debug_img_pub = self.create_publisher(Image, '/tree_detection/debug_image', 10)
        self.mask_img_pub = self.create_publisher(Image, '/tree_detection/mask_image', 10)
        self.background_mask_pub = self.create_publisher(Image, '/tree_detection/background_mask', 10)
        
        # TF setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # ORIGINAL DETECTION PARAMETERS (proven to work)
        self.brown_hsv_lower = np.array([8, 40, 20])
        self.brown_hsv_upper = np.array([25, 255, 200])
        
        # Original size filtering (good for tree detection)
        self.min_area = 250
        self.max_area = 8000
        self.min_aspect_ratio = 0.4
        self.max_aspect_ratio = 4.0
        
        # Original clustering
        self.cluster_distance = 100
        
        # SMART BACKGROUND FILTERING (new addition)
        # These help distinguish trees from background without being too restrictive
        self.background_removal_enabled = True
        self.edge_detection_threshold = 50
        self.texture_analysis_enabled = True
        
        # Tree tracking
        self.tree_trackers = {}
        self.next_tree_id = 1
        self.frame_count = 0
        self.last_log_frame = 0
        
        # Background pattern detection
        self.prev_frames = deque(maxlen=3)  # Store last 3 frames for motion analysis
        
        # Original LiDAR validation (permissive)
        self.expected_tree_width_meters = 0.15
        self.size_tolerance = 0.6  # Original tolerance
        
        self.get_logger().info('Smart Tree Detector initialized!')
        # self.get_logger().info('🌳 Based on original working approach')
        # self.get_logger().info('🧠 Added intelligent background filtering for real-world environments')
        # self.get_logger().info('⚠️  Safety first: Detects trees at ALL distances')
        
    def scan_callback(self, msg):
        """Store LiDAR scan."""
        self.scan = msg
        
    def image_callback(self, msg):
        """Main detection callback with smart background filtering"""
        if self.scan is None:
            return
            
        try:
            self.frame_count += 1
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Store frame for background analysis
            self.prev_frames.append(cv_image.copy())
            
            # Smart detection with background filtering
            current_detections = self.detect_trees_smart(cv_image)
            
            # Update tree trackers
            self.update_tree_trackers(current_detections)
            
            # Clean up old trackers
            self.cleanup_inactive_trackers()
            
            # Publish results
            self.publish_detection_results()
            
            # Log state (every 30 frames)
            if self.frame_count - self.last_log_frame >= 30:
                self.log_current_state()
                self.last_log_frame = self.frame_count
            
            # Publish markers
            self.publish_tree_markers(cv_image)
            
            # Debug visualization
            debug_img = self.create_debug_image(cv_image, current_detections)
            self.debug_img_pub.publish(self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8'))
                
        except Exception as e:
            self.get_logger().error(f'Error in smart tree detection: {str(e)}')
    
    def detect_trees_smart(self, image):
        """
        Smart detection: Original approach + intelligent background filtering
        Maintains detection sensitivity while filtering background objects
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Step 1: Original color detection (proven to work)
        color_mask = cv2.inRange(hsv, self.brown_hsv_lower, self.brown_hsv_upper)
        
        # Step 2: Smart background filtering
        if self.background_removal_enabled:
            background_filter_mask = self.create_background_filter(image, hsv)
            
            # Combine color detection with background filtering
            # Only remove detections that are clearly background
            mask = cv2.bitwise_and(color_mask, background_filter_mask)
            
            # Publish background filter for debugging
            self.background_mask_pub.publish(self.bridge.cv2_to_imgmsg(background_filter_mask, encoding='mono8'))
        else:
            mask = color_mask
        
        # Step 3: Original morphological operations (light touch)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Publish mask for debugging
        self.mask_img_pub.publish(self.bridge.cv2_to_imgmsg(mask, encoding='mono8'))
        
        # Step 4: Find contours (original approach)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        # Step 5: Original filtering and clustering
        valid_detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area <= area <= self.max_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / w if w > 0 else 0
                
                if self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio:
                    center_x = x + w / 2
                    center_y = y + h / 2
                    
                    # Get distance
                    distance = self.get_lidar_distance(center_x, image.shape[1])
                    
                    detection = {
                        'center_x': center_x,
                        'center_y': center_y,
                        'bbox': (x, y, w, h),
                        'area': area,
                        'distance': distance,
                        'aspect_ratio': aspect_ratio
                    }
                    
                    # Original size validation (permissive)
                    if self.validate_detection_size(detection):
                        # Additional smart filtering check
                        if self.is_likely_tree(detection, image):
                            valid_detections.append(detection)
        
        # Step 6: Original clustering
        clustered_detections = self.cluster_detections(valid_detections)
        
        return clustered_detections
    
    def create_background_filter(self, image, hsv):
        """
        Create a filter to remove background objects while keeping trees
        Returns a mask where 255 = keep, 0 = remove
        """
        height, width = image.shape[:2]
        filter_mask = np.ones((height, width), dtype=np.uint8) * 255
        
        # Filter 1: Remove very large horizontal regions (walls, floors)
        # Large horizontal brown areas are likely walls/furniture, not trees
        brown_mask = cv2.inRange(hsv, self.brown_hsv_lower, self.brown_hsv_upper)
        
        # Find large horizontal structures
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 10))
        horizontal_structures = cv2.morphologyEx(brown_mask, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Remove these from our filter
        filter_mask = cv2.bitwise_and(filter_mask, cv2.bitwise_not(horizontal_structures))
        
        # Filter 2: Remove regions at image edges (likely background)
        # Create edge mask - remove detections too close to image borders
        edge_buffer = 20
        filter_mask[:edge_buffer, :] = 0      # Top edge
        filter_mask[-edge_buffer:, :] = 0     # Bottom edge  
        filter_mask[:, :edge_buffer] = 0      # Left edge
        filter_mask[:, -edge_buffer:] = 0     # Right edge
        
        # Filter 3: Motion-based filtering (if we have previous frames)
        if len(self.prev_frames) >= 2:
            # Objects that don't move between frames are more likely background
            motion_mask = self.detect_motion_regions(self.prev_frames[-2], self.prev_frames[-1])
            
            # Slight preference for moving objects (but don't completely eliminate static)
            static_regions = cv2.bitwise_not(motion_mask)
            static_eroded = cv2.erode(static_regions, np.ones((10, 10), np.uint8), iterations=1)
            filter_mask = cv2.bitwise_and(filter_mask, cv2.bitwise_not(static_eroded))
        
        # Filter 4: Texture analysis - trees have more texture than walls
        if self.texture_analysis_enabled:
            texture_mask = self.analyze_texture(image)
            filter_mask = cv2.bitwise_and(filter_mask, texture_mask)
        
        return filter_mask
    
    def detect_motion_regions(self, prev_frame, curr_frame):
        """Detect regions with motion between frames"""
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        diff = cv2.absdiff(prev_gray, curr_gray)
        
        # Threshold to get motion mask
        _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Clean up noise
        kernel = np.ones((5, 5), np.uint8)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        
        return motion_mask
    
    def analyze_texture(self, image):
        """Analyze texture to distinguish trees from smooth surfaces"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate Laplacian (edge detection)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_abs = np.absolute(laplacian)
        
        # Areas with more edges/texture are more likely to be trees
        _, texture_mask = cv2.threshold(laplacian_abs, 20, 255, cv2.THRESH_BINARY)
        
        # Dilate to be more inclusive
        kernel = np.ones((7, 7), np.uint8)
        texture_mask = cv2.dilate(texture_mask.astype(np.uint8), kernel, iterations=1)
        
        return texture_mask
    
    def is_likely_tree(self, detection, image):
        """
        Additional check to see if detection is likely a tree vs background object
        This is permissive - only rejects obvious non-trees
        """
        x, y, w, h = detection['bbox']
        roi = image[y:y+h, x:x+w]
        
        if roi.size == 0:
            return True  # Benefit of doubt
        
        # Check 1: Color consistency
        # Trees should have relatively consistent brown color throughout
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        brown_pixels = cv2.inRange(hsv_roi, self.brown_hsv_lower, self.brown_hsv_upper)
        brown_ratio = np.sum(brown_pixels > 0) / (w * h)
        
        # If less than 30% brown pixels, might be false positive
        if brown_ratio < 0.3:
            return False
        
        # Check 2: Shape regularity
        # Trees should have somewhat regular shapes, not very irregular blobs
        contours, _ = cv2.findContours(brown_pixels, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(largest_contour)
            
            # Solidity = contour area / hull area
            contour_area = cv2.contourArea(largest_contour)
            hull_area = cv2.contourArea(hull)
            
            if hull_area > 0:
                solidity = contour_area / hull_area
                # Very irregular shapes (solidity < 0.3) are suspicious
                if solidity < 0.3:
                    return False
        
        # If passes basic checks, consider it likely a tree
        return True
    
    # Original helper methods (proven to work)
    def cluster_detections(self, detections):
        """Original clustering logic"""
        if not detections:
            return []
        
        clustered = []
        used = [False] * len(detections)
        
        for i, detection in enumerate(detections):
            if used[i]:
                continue
                
            cluster = [detection]
            used[i] = True
            
            for j, other_detection in enumerate(detections):
                if used[j]:
                    continue
                    
                distance = np.sqrt((detection['center_x'] - other_detection['center_x'])**2 + 
                                 (detection['center_y'] - other_detection['center_y'])**2)
                
                if distance <= self.cluster_distance:
                    cluster.append(other_detection)
                    used[j] = True
            
            # Merge cluster
            merged_detection = self.merge_cluster(cluster)
            clustered.append(merged_detection)
        
        return clustered
    
    def merge_cluster(self, cluster):
        """Original cluster merging"""
        if len(cluster) == 1:
            cluster[0]['cluster_size'] = 1
            return cluster[0]
        
        min_x = min([det['bbox'][0] for det in cluster])
        min_y = min([det['bbox'][1] for det in cluster])
        max_x = max([det['bbox'][0] + det['bbox'][2] for det in cluster])
        max_y = max([det['bbox'][1] + det['bbox'][3] for det in cluster])
        
        merged_w = max_x - min_x
        merged_h = max_y - min_y
        merged_area = sum([det['area'] for det in cluster])
        avg_distance = np.mean([det['distance'] for det in cluster if det['distance']])
        
        return {
            'center_x': min_x + merged_w / 2,
            'center_y': min_y + merged_h / 2,
            'bbox': (min_x, min_y, merged_w, merged_h),
            'area': merged_area,
            'distance': avg_distance,
            'cluster_size': len(cluster)
        }
    
    def validate_detection_size(self, detection):
        """Original size validation (permissive)"""
        if detection['distance'] is None or detection['distance'] <= 0:
            return True  # Accept if no distance data
        
        expected_pixel_width = self.calculate_expected_pixel_size(detection['distance'])
        actual_pixel_width = detection['bbox'][2]
        
        if expected_pixel_width <= 0:
            return True
        
        size_ratio = actual_pixel_width / expected_pixel_width
        return (1 - self.size_tolerance) <= size_ratio <= (1 + self.size_tolerance)
    
    def update_tree_trackers(self, current_detections):
        """Update tree trackers - original logic"""
        for tracker in self.tree_trackers.values():
            tracker.detection_count = 0
        
        unmatched_detections = []
        
        for detection in current_detections:
            matched = False
            best_match = None
            best_distance = float('inf')
            
            for tracker in self.tree_trackers.values():
                if tracker.is_match(detection):
                    dist = np.sqrt((tracker.position[0] - detection['center_x'])**2 + 
                                  (tracker.position[1] - detection['center_y'])**2)
                    if dist < best_distance:
                        best_distance = dist
                        best_match = tracker
            
            if best_match:
                best_match.update(detection, self.frame_count)
                matched = True
            else:
                unmatched_detections.append(detection)
        
        # Create new trackers for unmatched detections
        for detection in unmatched_detections:
            new_tracker = TreeTracker(self.next_tree_id, detection)
            new_tracker.last_seen_frame = self.frame_count
            self.tree_trackers[self.next_tree_id] = new_tracker
            
            distance_info = f" at {detection['distance']:.2f}m" if detection['distance'] else ""
            self.get_logger().info(f'🆕 NEW TREE DETECTED! Tree #{self.next_tree_id}{distance_info}')
            self.next_tree_id += 1
    
    def cleanup_inactive_trackers(self):
        """Remove inactive trackers - original logic"""
        inactive_ids = []
        
        for tree_id, tracker in self.tree_trackers.items():
            if not tracker.is_active(self.frame_count, max_frames_missing=15):
                inactive_ids.append(tree_id)
        
        for tree_id in inactive_ids:
            self.get_logger().info(f'❌ Tree #{tree_id} lost from view')
            del self.tree_trackers[tree_id]
    
    def publish_detection_results(self):
        """Publish detection results"""
        active_trees = [t for t in self.tree_trackers.values() if t.is_active(self.frame_count)]
        
        tree_msg = String()
        tree_msg.data = f"trees_detected:{len(active_trees)}"
        self.tree_pub.publish(tree_msg)
    
    def log_current_state(self):
        """Log current state"""
        active_trees = [t for t in self.tree_trackers.values() if t.is_active(self.frame_count)]
        
        if active_trees:
            self.get_logger().info(f'🌳 ACTIVE TREES: {len(active_trees)} total')
            
            for tracker in sorted(active_trees, key=lambda x: x.id):
                status = "STABLE" if tracker.is_stable else "TRACKING"
                distance_info = f"dist:{tracker.distance:.2f}m" if tracker.distance else "dist:unknown"
                
                self.get_logger().info(f'  Tree #{tracker.id}: {status} | {distance_info} | detections:{tracker.total_detections}')
        else:
            self.get_logger().info('No trees currently detected - scanning...')
    
    def publish_tree_markers(self, cv_image):
        """Publish markers for tracked trees"""
        marker_array = MarkerArray()
        
        for tracker in self.tree_trackers.values():
            if tracker.is_active(self.frame_count):
                world_coords = self.get_world_coordinates(tracker.position[0], cv_image.shape[1])
                
                if world_coords and world_coords[2] is not None:
                    lidar_x, lidar_y, map_x, map_y = world_coords
                    tracker.map_position = (map_x, map_y)
                    
                    marker = Marker()
                    marker.header.frame_id = 'map'
                    marker.header.stamp = self.get_clock().now().to_msg()
                    marker.ns = 'smart_trees'
                    marker.id = tracker.id
                    marker.type = Marker.CYLINDER
                    marker.action = Marker.ADD
                    
                    marker.pose.position.x = map_x
                    marker.pose.position.y = map_y
                    marker.pose.position.z = 0.5
                    marker.pose.orientation.w = 1.0
                    
                    marker.scale.x = 0.25
                    marker.scale.y = 0.25
                    marker.scale.z = 1.0
                    
                    # Color based on stability
                    if tracker.is_stable:
                        marker.color.r = 0.6
                        marker.color.g = 0.3
                        marker.color.b = 0.1
                        marker.color.a = 0.9
                    else:
                        marker.color.r = 1.0
                        marker.color.g = 0.5
                        marker.color.b = 0.0
                        marker.color.a = 0.7
                    
                    marker_array.markers.append(marker)
        
        self.marker_array_pub.publish(marker_array)
    
    def create_debug_image(self, original_image, current_detections):
        """Create debug image"""
        debug_img = original_image.copy()
        
        # Draw current detections
        for detection in current_detections:
            x, y, w, h = detection['bbox']
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(debug_img, "CURRENT", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw tracked trees
        for tracker in self.tree_trackers.values():
            if tracker.is_active(self.frame_count):
                x, y = int(tracker.position[0] - 50), int(tracker.position[1] - 50)
                w, h = 100, 100
                
                color = (0, 255, 0) if tracker.is_stable else (0, 165, 255)
                
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 3)
                cv2.circle(debug_img, (int(tracker.position[0]), int(tracker.position[1])), 10, color, -1)
                
                label = f"TREE #{tracker.id}"
                status = " (STABLE)" if tracker.is_stable else " (NEW)"
                cv2.putText(debug_img, label + status, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                if tracker.distance:
                    cv2.putText(debug_img, f"{tracker.distance:.1f}m", (x, y + h + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Status overlay
        active_count = len([t for t in self.tree_trackers.values() if t.is_active(self.frame_count)])
        stable_count = len([t for t in self.tree_trackers.values() if t.is_active(self.frame_count) and t.is_stable])
        
        cv2.putText(debug_img, f"🌳 Trees: {active_count} | Stable: {stable_count} | Current: {len(current_detections)}", 
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return debug_img
    
    # Original utility methods
    def get_lidar_distance(self, center_x, image_width):
        """Get LiDAR distance for a pixel coordinate"""
        if self.scan is None:
            return None
            
        try:
            normalized_x = center_x / image_width
            angle = self.scan.angle_min + normalized_x * (self.scan.angle_max - self.scan.angle_min)
            index = int((angle - self.scan.angle_min) / self.scan.angle_increment)
            
            if 0 <= index < len(self.scan.ranges):
                distance = self.scan.ranges[index]
                if np.isfinite(distance) and self.scan.range_min <= distance <= self.scan.range_max:
                    return distance
        except Exception:
            pass
        return None
    
    def calculate_expected_pixel_size(self, distance):
        """Calculate expected pixel size of tree at given distance"""
        camera_fov = np.radians(60)
        image_width_pixels = 640
        angular_size = 2 * np.arctan(self.expected_tree_width_meters / (2 * distance))
        pixels_per_radian = image_width_pixels / camera_fov
        expected_pixels = angular_size * pixels_per_radian
        return max(expected_pixels, 10)
    
    def get_world_coordinates(self, center_x, image_width):
        """Convert pixel coordinates to world coordinates"""
        try:
            normalized_x = center_x / image_width
            angle = self.scan.angle_min + normalized_x * (self.scan.angle_max - self.scan.angle_min)
            index = int((angle - self.scan.angle_min) / self.scan.angle_increment)
            
            if index < 0 or index >= len(self.scan.ranges):
                return None
            
            distance = self.scan.ranges[index]
            if not np.isfinite(distance) or not (self.scan.range_min <= distance <= self.scan.range_max):
                return None
            
            point_lidar = PointStamped()
            point_lidar.header.frame_id = self.scan.header.frame_id
            point_lidar.header.stamp = self.scan.header.stamp
            point_lidar.point.x = distance * np.cos(angle)
            point_lidar.point.y = distance * np.sin(angle)
            point_lidar.point.z = 0.0
            
            map_coords = self.transform_to_map(point_lidar)
            if map_coords:
                map_x, map_y = map_coords
                return (point_lidar.point.x, point_lidar.point.y, map_x, map_y)
            else:
                return (point_lidar.point.x, point_lidar.point.y, None, None)
                
        except Exception as e:
            return None
    
    def transform_to_map(self, point_lidar):
        """Transform point to map frame"""
        try:
            timeout = rclpy.duration.Duration(seconds=0.1)
            if self.tf_buffer.can_transform('map', point_lidar.header.frame_id, rclpy.time.Time(), timeout=timeout):
                point_map = tf2_geometry_msgs.do_transform_point(
                    point_lidar,
                    self.tf_buffer.lookup_transform('map', point_lidar.header.frame_id, rclpy.time.Time(), timeout=timeout)
                )
                return (point_map.point.x, point_map.point.y)
            else:
                return None
        except Exception as e:
            return None

def main(args=None):
    rclpy.init(args=args)
    
    try:
        tree_detector = ImprovedTreeDetector()
        tree_detector.get_logger().info('🚀 Starting tree detector...')
        # tree_detector.get_logger().info('🌳 Original detection sensitivity + intelligent background filtering')
        # tree_detector.get_logger().info('⚠️  Detects trees at ALL distances for navigation safety')
        # tree_detector.get_logger().info('🧠 Smart filters: motion analysis, texture analysis, edge detection')
        rclpy.spin(tree_detector)
    except KeyboardInterrupt:
        tree_detector.get_logger().info('Improved tree detector shutting down...')
    finally:
        if 'tree_detector' in locals():
            tree_detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
