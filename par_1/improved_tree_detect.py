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
        self.confidence_score = detection.get('confidence', 0.5)
        
    def update(self, detection, frame_num):
        """Update tracker with new detection"""
        self.position = (detection['center_x'], detection['center_y'])
        self.last_seen_frame = frame_num
        self.detection_count += 1
        self.total_detections += 1
        self.distance = detection.get('distance', None)
        self.confidence_score = detection.get('confidence', 0.5)
        
        # Consider stable after 5 consistent detections (more restrictive)
        if self.detection_count >= 5:
            self.is_stable = True
    
    def is_match(self, detection, max_distance=60):  # Reduced from 80
        """Check if detection matches this tracker"""
        dist = np.sqrt((self.position[0] - detection['center_x'])**2 + 
                      (self.position[1] - detection['center_y'])**2)
        return dist <= max_distance
    
    def is_active(self, current_frame, max_frames_missing=8):  # More strict
        """Check if tracker is still active (recently seen)"""
        return (current_frame - self.last_seen_frame) <= max_frames_missing

class ImprovedTreeDetector(Node):
    """
    Restrictive tree detector specifically tuned for brown cylinders in real-world environments
    Much more selective to avoid false positives from background objects
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
        self.filtered_img_pub = self.create_publisher(Image, '/tree_detection/filtered_image', 10)
        
        # TF setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # RESTRICTIVE DETECTION PARAMETERS FOR REAL WORLD
        # More specific brown color range for cylinders (not furniture/walls)
        self.brown_hsv_lower = np.array([10, 60, 30])   # More restrictive saturation/value
        self.brown_hsv_upper = np.array([20, 200, 150]) # Avoid very bright/dark browns
        
        # Much more restrictive size filtering
        self.min_area = 400          # Larger minimum (reject small brown spots)
        self.max_area = 3000         # Smaller maximum (reject walls/furniture)
        self.min_aspect_ratio = 0.8  # More vertical (cylinders are tall)
        self.max_aspect_ratio = 2.5  # Not too thin
        
        # Geometric validation for cylinders
        self.min_width = 15          # Minimum pixel width
        self.max_width = 120         # Maximum pixel width
        self.min_height = 25         # Minimum pixel height
        self.max_height = 200        # Maximum pixel height
        
        # NO distance constraints for safety - robot must detect ALL trees
        # Will use distance as confidence weighting factor instead of hard limit
        
        # Clustering parameters (more strict)
        self.cluster_distance = 50   # Reduced clustering distance
        self.min_separation = 0.4    # Minimum 40cm between different trees
        
        # Confidence scoring parameters (safety-first approach)
        self.min_confidence = 0.4    # Lower threshold - detect more potential trees for safety
        
        # Tree tracking
        self.tree_trackers = {}
        self.next_tree_id = 1
        self.frame_count = 0
        self.last_log_frame = 0
        
        # LiDAR validation (strict)
        self.expected_tree_width_meters = 0.12  # Expected tree diameter (12cm)
        self.size_tolerance = 0.4  # Only 40% tolerance (much more strict)
        
        self.get_logger().info('Restrictive Tree Detector initialized!')
        self.get_logger().info('🎯 Tuned for brown cylinders in real-world environments')
        self.get_logger().info('🚫 Strict shape filtering to avoid background false positives')
        self.get_logger().info('⚠️  SAFETY FIRST: Detects trees at ALL distances to prevent collisions')
        
    def scan_callback(self, msg):
        """Store LiDAR scan."""
        self.scan = msg
        
    def image_callback(self, msg):
        """Main detection callback with restrictive filtering"""
        if self.scan is None:
            return
            
        try:
            self.frame_count += 1
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # RESTRICTIVE detection pipeline
            current_detections = self.detect_trees_restrictive(cv_image)
            
            # Update tree trackers with only high-confidence detections
            self.update_tree_trackers(current_detections)
            
            # Clean up trackers more aggressively
            self.cleanup_inactive_trackers()
            
            # Publish results
            self.publish_detection_results()
            
            # Log state less frequently to reduce spam
            if self.frame_count - self.last_log_frame >= 60:  # Every 2 seconds
                self.log_current_state()
                self.last_log_frame = self.frame_count
            
            # Publish markers
            self.publish_tree_markers(cv_image)
            
            # Debug visualization
            debug_img = self.create_debug_image(cv_image, current_detections)
            self.debug_img_pub.publish(self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8'))
                
        except Exception as e:
            self.get_logger().error(f'Error in restrictive tree detection: {str(e)}')
    
    def detect_trees_restrictive(self, image):
        """
        RESTRICTIVE detection specifically for brown cylinders
        Multiple filtering stages to eliminate false positives
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Stage 1: Color filtering (more restrictive)
        mask = cv2.inRange(hsv, self.brown_hsv_lower, self.brown_hsv_upper)
        
        # Stage 2: Aggressive noise removal
        # Remove small noise first
        kernel_small = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
        
        # Fill gaps in cylinders
        kernel_large = np.ones((8, 8), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=1)
        
        # Final noise removal
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # Publish mask for debugging
        self.mask_img_pub.publish(self.bridge.cv2_to_imgmsg(mask, encoding='mono8'))
        
        # Stage 3: Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        # Stage 4: Geometric filtering for cylinders
        candidate_detections = []
        
        for contour in contours:
            detection = self.evaluate_contour_as_tree(contour, image.shape[1])
            if detection:
                candidate_detections.append(detection)
        
        # Stage 5: Spatial filtering (remove trees too close together)
        final_detections = self.filter_by_separation(validated_detections)
        
        return final_detections
    
    def evaluate_contour_as_tree(self, contour, image_width):
        """
        Comprehensive evaluation of whether a contour represents a tree
        Returns detection dict if valid, None if rejected
        """
        # Basic geometric properties
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        
        # Stage 1: Size filtering
        if not (self.min_area <= area <= self.max_area):
            return None
        
        if not (self.min_width <= w <= self.max_width):
            return None
            
        if not (self.min_height <= h <= self.max_height):
            return None
        
        # Stage 2: Aspect ratio (cylinders should be vertical-ish)
        aspect_ratio = h / w if w > 0 else 0
        if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
            return None
        
        # Stage 3: Shape analysis
        # Cylinders should have relatively regular shapes
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return None
            
        # Circularity measure (4π*area/perimeter²) - cylinders viewed from side should be somewhat rectangular
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Extent (area/bounding_box_area) - should be reasonably filled
        bbox_area = w * h
        extent = area / bbox_area if bbox_area > 0 else 0
        
        # Stage 4: Get distance (no hard constraints for safety)
        center_x = x + w / 2
        center_y = y + h / 2
        distance = self.get_lidar_distance(center_x, image_width)
        
        # Distance is allowed to be None or any value - safety first!
        
        # Stage 5: Size-distance consistency (when distance available)
        size_ratio = 1.0  # Default assumption if no distance
        if distance and distance > 0:
            expected_pixel_width = self.calculate_expected_pixel_size(distance)
            if expected_pixel_width > 0:
                size_ratio = w / expected_pixel_width
                # Only reject if size is extremely inconsistent (very permissive)
                if size_ratio < 0.2 or size_ratio > 5.0:  # Much wider tolerance
                    return None
        
        # Stage 6: Calculate confidence score (distance influences confidence, not rejection)
        confidence = self.calculate_confidence_score({
            'aspect_ratio': aspect_ratio,
            'extent': extent,
            'circularity': circularity,
            'size_ratio': size_ratio,
            'distance': distance,
            'area': area
        })
        
        if confidence < self.min_confidence:
            return None
        
        # All tests passed - create detection
        return {
            'center_x': center_x,
            'center_y': center_y,
            'bbox': (x, y, w, h),
            'area': area,
            'distance': distance,
            'aspect_ratio': aspect_ratio,
            'extent': extent,
            'circularity': circularity,
            'size_ratio': size_ratio,
            'confidence': confidence
        }
    
    def calculate_confidence_score(self, metrics):
        """
        Calculate confidence score - distance influences confidence but doesn't reject
        Safety first: ALL potential trees are detected, confidence varies by likelihood
        """
        confidence = 0.0
        
        # Aspect ratio score (prefer vertical rectangles)
        ideal_aspect = 1.5
        aspect_score = 1.0 - abs(metrics['aspect_ratio'] - ideal_aspect) / ideal_aspect
        confidence += max(0, aspect_score) * 0.3
        
        # Extent score (prefer well-filled bounding boxes)
        extent_score = metrics['extent']
        confidence += extent_score * 0.2
        
        # Size consistency score (when distance available)
        size_score = 1.0 - abs(1.0 - metrics['size_ratio']) / 2.0  # More permissive
        confidence += max(0, size_score) * 0.2
        
        # Distance score - influences confidence but doesn't reject
        distance = metrics.get('distance', None)
        if distance and distance > 0:
            if 0.5 <= distance <= 3.0:
                distance_score = 1.0  # Ideal range
            elif 0.1 <= distance <= 0.5 or 3.0 <= distance <= 6.0:
                distance_score = 0.7  # Less ideal but acceptable
            else:
                distance_score = 0.4  # Low confidence but still detected
        else:
            distance_score = 0.5  # Unknown distance gets neutral score
        
        confidence += distance_score * 0.3
        
        return min(1.0, confidence)
    
    def filter_by_separation(self, detections):
        """Remove trees that are too close together (likely duplicates)"""
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence (keep higher confidence detections)
        sorted_detections = sorted(detections, key=lambda d: d['confidence'], reverse=True)
        
        filtered = []
        
        for detection in sorted_detections:
            # Check if this detection is too close to any accepted detection
            too_close = False
            
            for accepted in filtered:
                # Calculate real-world distance between detections
                if detection['distance'] and accepted['distance']:
                    # Approximate world-space separation using distance and pixel separation
                    pixel_sep = np.sqrt((detection['center_x'] - accepted['center_x'])**2 + 
                                      (detection['center_y'] - accepted['center_y'])**2)
                    
                    # Rough conversion to world distance (this is approximate)
                    avg_distance = (detection['distance'] + accepted['distance']) / 2
                    approx_world_sep = (pixel_sep / 640) * (avg_distance * np.tan(np.radians(30)))
                    
                    if approx_world_sep < self.min_separation:
                        too_close = True
                        break
            
            if not too_close:
                filtered.append(detection)
        
        return filtered
    
    def update_tree_trackers(self, current_detections):
        """Update tree trackers - only accept high-confidence detections"""
        
        # Reset detection counts for this frame
        for tracker in self.tree_trackers.values():
            tracker.detection_count = 0
        
        # Try to match detections to existing trackers
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
        
        # Create new trackers for high confidence detections (safety-conscious threshold)
        for detection in unmatched_detections:
            if detection['confidence'] >= 0.5:  # Lowered for safety - detect more potential trees
                new_tracker = TreeTracker(self.next_tree_id, detection)
                new_tracker.last_seen_frame = self.frame_count
                self.tree_trackers[self.next_tree_id] = new_tracker
                
                distance_info = f" at {detection['distance']:.2f}m" if detection['distance'] else " (distance unknown)"
                self.get_logger().info(f'🆕 TREE DETECTED! Tree #{self.next_tree_id}{distance_info} (confidence: {detection["confidence"]:.2f})')
                self.next_tree_id += 1
            else:
                self.get_logger().debug(f'Low confidence detection rejected: {detection["confidence"]:.2f}')
    
    def cleanup_inactive_trackers(self):
        """Remove trackers more aggressively"""
        inactive_ids = []
        
        for tree_id, tracker in self.tree_trackers.items():
            if not tracker.is_active(self.frame_count, max_frames_missing=5):  # Very strict
                inactive_ids.append(tree_id)
        
        for tree_id in inactive_ids:
            self.get_logger().info(f'❌ Tree #{tree_id} lost (strict timeout)')
            del self.tree_trackers[tree_id]
    
    def publish_detection_results(self):
        """Publish detection results"""
        active_trees = [t for t in self.tree_trackers.values() if t.is_active(self.frame_count)]
        
        tree_msg = String()
        tree_msg.data = f"trees_detected:{len(active_trees)}"
        self.tree_pub.publish(tree_msg)
    
    def log_current_state(self):
        """Log current state less frequently"""
        active_trees = [t for t in self.tree_trackers.values() if t.is_active(self.frame_count)]
        
        if active_trees:
            self.get_logger().info(f'🌳 CONFIRMED TREES: {len(active_trees)} total')
            
            for tracker in sorted(active_trees, key=lambda x: x.id):
                status = "STABLE" if tracker.is_stable else "TRACKING"
                confidence_info = f"conf:{tracker.confidence_score:.2f}"
                distance_info = f"dist:{tracker.distance:.2f}m" if tracker.distance else "dist:unknown"
                
                self.get_logger().info(f'  Tree #{tracker.id}: {status} | {confidence_info} | {distance_info} | detections:{tracker.total_detections}')
        else:
            self.get_logger().info('No confirmed trees - scanning...')
    
    def publish_tree_markers(self, cv_image):
        """Publish markers for confirmed trees only"""
        marker_array = MarkerArray()
        
        for tracker in self.tree_trackers.values():
            if tracker.is_active(self.frame_count) and tracker.is_stable:  # Only stable trees get markers
                world_coords = self.get_world_coordinates(tracker.position[0], cv_image.shape[1])
                
                if world_coords and world_coords[2] is not None:
                    lidar_x, lidar_y, map_x, map_y = world_coords
                    tracker.map_position = (map_x, map_y)
                    
                    marker = Marker()
                    marker.header.frame_id = 'map'
                    marker.header.stamp = self.get_clock().now().to_msg()
                    marker.ns = 'confirmed_trees'
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
                    
                    # Green for confirmed trees
                    marker.color.r = 0.0
                    marker.color.g = 0.8
                    marker.color.b = 0.0
                    marker.color.a = 0.9
                    
                    marker_array.markers.append(marker)
        
        self.marker_array_pub.publish(marker_array)
    
    def create_debug_image(self, original_image, current_detections):
        """Create debug image showing only high-confidence detections"""
        debug_img = original_image.copy()
        
        # Draw current high-confidence detections
        for detection in current_detections:
            x, y, w, h = detection['bbox']
            confidence = detection['confidence']
            
            # Color based on confidence
            if confidence >= 0.8:
                color = (0, 255, 0)  # Green for high confidence
            elif confidence >= 0.6:
                color = (0, 255, 255)  # Yellow for medium confidence
            else:
                color = (0, 0, 255)  # Red for low confidence
            
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(debug_img, f"CONF:{confidence:.2f}", (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw confirmed tracked trees
        for tracker in self.tree_trackers.values():
            if tracker.is_active(self.frame_count):
                x, y = int(tracker.position[0] - 40), int(tracker.position[1] - 40)
                w, h = 80, 80
                
                color = (0, 255, 0) if tracker.is_stable else (0, 165, 255)
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 3)
                cv2.circle(debug_img, (int(tracker.position[0]), int(tracker.position[1])), 8, color, -1)
                
                label = f"TREE #{tracker.id}"
                cv2.putText(debug_img, label, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Status overlay
        active_count = len([t for t in self.tree_trackers.values() if t.is_active(self.frame_count)])
        stable_count = len([t for t in self.tree_trackers.values() if t.is_active(self.frame_count) and t.is_stable])
        
        cv2.putText(debug_img, f"🌳 Confirmed Trees: {active_count} | Stable: {stable_count}", 
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(debug_img, f"High-conf detections: {len([d for d in current_detections if d['confidence'] >= 0.8])}", 
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return debug_img
    
    # Keep utility methods
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
        return max(expected_pixels, 8)
    
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
        tree_detector.get_logger().info('🚀 Starting IMPROVED tree detector for real-world environments...')
        # tree_detector.get_logger().info('🎯 Strict shape filtering to eliminate background false positives')
        # tree_detector.get_logger().info('⚠️  SAFETY FIRST: Detects trees at ALL distances to prevent collisions')
        # tree_detector.get_logger().info('🌳 All potential cylindrical trees detected, filtered by confidence')
        rclpy.spin(tree_detector)
    except KeyboardInterrupt:
        tree_detector.get_logger().info('Improved tree detector shutting down...')
    finally:
        if 'tree_detector' in locals():
            tree_detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
