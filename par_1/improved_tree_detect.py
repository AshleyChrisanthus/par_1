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

class RobustTreeTracker:
    """Robust tree tracker with better persistence"""
    def __init__(self, tree_id, detection):
        self.id = tree_id
        self.position = (detection['center_x'], detection['center_y'])
        self.last_seen_frame = 0
        self.consecutive_detections = 1
        self.total_detections = 1
        self.is_stable = False
        self.map_position = None
        self.distance = detection.get('distance', None)
        self.missed_frames = 0
        
    def update(self, detection, frame_num):
        """Update tracker with new detection"""
        self.position = (detection['center_x'], detection['center_y'])
        self.last_seen_frame = frame_num
        self.consecutive_detections += 1
        self.total_detections += 1
        self.distance = detection.get('distance', None)
        self.missed_frames = 0  # Reset missed frames
        
        # More strict stability requirement
        if self.consecutive_detections >= 5:
            self.is_stable = True
    
    def miss_frame(self, frame_num):
        """Called when tracker is not matched this frame"""
        self.missed_frames += 1
        self.consecutive_detections = 0  # Reset consecutive count
        if self.missed_frames > 5:  # Only lose stable status after missing many frames
            self.is_stable = False
    
    def is_match(self, detection, max_distance=60):  # Tighter matching
        """Check if detection matches this tracker"""
        dist = np.sqrt((self.position[0] - detection['center_x'])**2 + 
                      (self.position[1] - detection['center_y'])**2)
        return dist <= max_distance
    
    def is_active(self, current_frame, max_frames_missing=8):  # Faster timeout
        """Check if tracker is still active"""
        return self.missed_frames <= max_frames_missing

class ImprovedTreeDetector(Node):
    """
    Simple, robust tree detector focused on accuracy over fancy features
    Back to basics with better tracking logic
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
        
        # TF setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # SIMPLE, PROVEN DETECTION PARAMETERS
        self.brown_hsv_lower = np.array([8, 50, 30])   # Slightly more restrictive
        self.brown_hsv_upper = np.array([22, 255, 180])
        
        # Reasonable size constraints
        self.min_area = 300
        self.max_area = 5000
        self.min_aspect_ratio = 0.5
        self.max_aspect_ratio = 3.5
        
        # Simple clustering
        self.cluster_distance = 80
        
        # Tree tracking with much better logic
        self.tree_trackers = {}
        self.next_tree_id = 1
        self.frame_count = 0
        self.last_log_frame = 0
        
        # Minimum separation between trees (in meters, not pixels)
        self.min_tree_separation_meters = 0.3
        
        # LiDAR validation
        self.expected_tree_width_meters = 0.15
        self.size_tolerance = 0.7
        
        self.get_logger().info('Tree Detector initialized!')
        # self.get_logger().info('🎯 Back to basics - simple and reliable')
        # self.get_logger().info('🔒 Robust tracking to prevent ID jumping')
        
    def scan_callback(self, msg):
        """Store LiDAR scan."""
        self.scan = msg
        
    def image_callback(self, msg):
        """Main detection callback - simple and robust"""
        if self.scan is None:
            return
            
        try:
            self.frame_count += 1
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Simple detection
            current_detections = self.detect_trees_simple(cv_image)
            
            # Robust tracking
            self.update_tree_trackers_robust(current_detections)
            
            # Cleanup
            self.cleanup_inactive_trackers()
            
            # Publish
            self.publish_detection_results()
            
            # Log every 1 second (30 frames)
            if self.frame_count - self.last_log_frame >= 30:
                self.log_current_state()
                self.last_log_frame = self.frame_count
            
            # Visualize
            self.publish_tree_markers(cv_image)
            debug_img = self.create_debug_image(cv_image, current_detections)
            self.debug_img_pub.publish(self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8'))
                
        except Exception as e:
            self.get_logger().error(f'Error in tree detection: {str(e)}')
    
    def detect_trees_simple(self, image):
        """
        Simple, reliable detection - no fancy background filtering
        Focus on getting the basic detection right
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Basic color filtering
        mask = cv2.inRange(hsv, self.brown_hsv_lower, self.brown_hsv_upper)
        
        # Simple noise removal
        kernel = np.ones((4, 4), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Remove tiny noise
        kernel_small = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # Publish mask for debugging
        self.mask_img_pub.publish(self.bridge.cv2_to_imgmsg(mask, encoding='mono8'))
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        # Simple filtering
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
                    
                    # Only keep detections with valid distance and reasonable size
                    if distance and 0.2 <= distance <= 6.0:  # Reasonable range
                        # Simple size check
                        expected_size = self.calculate_expected_pixel_size(distance)
                        if expected_size > 0:
                            size_ratio = w / expected_size
                            if 0.3 <= size_ratio <= 3.0:  # Very permissive
                                detection = {
                                    'center_x': center_x,
                                    'center_y': center_y,
                                    'bbox': (x, y, w, h),
                                    'area': area,
                                    'distance': distance,
                                    'aspect_ratio': aspect_ratio
                                }
                                valid_detections.append(detection)
        
        # Simple clustering
        clustered = self.cluster_detections_simple(valid_detections)
        
        # Remove trees that are too close together in real world
        filtered = self.filter_by_real_world_separation(clustered)
        
        return filtered
    
    def cluster_detections_simple(self, detections):
        """Simple clustering without over-complication"""
        if not detections:
            return []
        
        clustered = []
        used = [False] * len(detections)
        
        for i, detection in enumerate(detections):
            if used[i]:
                continue
                
            cluster = [detection]
            used[i] = True
            
            # Find nearby detections
            for j, other in enumerate(detections):
                if used[j]:
                    continue
                    
                pixel_dist = np.sqrt((detection['center_x'] - other['center_x'])**2 + 
                                   (detection['center_y'] - other['center_y'])**2)
                
                if pixel_dist <= self.cluster_distance:
                    cluster.append(other)
                    used[j] = True
            
            # Merge cluster
            if len(cluster) == 1:
                clustered.append(cluster[0])
            else:
                # Simple merge - take detection closest to center of cluster
                avg_x = np.mean([d['center_x'] for d in cluster])
                avg_y = np.mean([d['center_y'] for d in cluster])
                
                best_detection = min(cluster, key=lambda d: 
                    np.sqrt((d['center_x'] - avg_x)**2 + (d['center_y'] - avg_y)**2))
                
                best_detection['cluster_size'] = len(cluster)
                clustered.append(best_detection)
        
        return clustered
    
    def filter_by_real_world_separation(self, detections):
        """Remove detections that are too close in real world coordinates"""
        if len(detections) <= 1:
            return detections
        
        # Sort by distance (prefer closer detections)
        sorted_detections = sorted(detections, key=lambda d: d.get('distance', 999))
        
        filtered = []
        
        for detection in sorted_detections:
            # Check if this detection is too close to any accepted detection
            too_close = False
            
            for accepted in filtered:
                real_distance = self.calculate_real_world_distance(detection, accepted)
                if real_distance < self.min_tree_separation_meters:
                    too_close = True
                    break
            
            if not too_close:
                filtered.append(detection)
        
        return filtered
    
    def calculate_real_world_distance(self, det1, det2):
        """Calculate approximate real-world distance between two detections"""
        if not det1.get('distance') or not det2.get('distance'):
            return 999  # Unknown, assume far
        
        # Convert pixel positions to approximate world coordinates
        angle1 = self.pixel_to_angle(det1['center_x'])
        angle2 = self.pixel_to_angle(det2['center_x'])
        
        x1 = det1['distance'] * np.cos(angle1)
        y1 = det1['distance'] * np.sin(angle1)
        x2 = det2['distance'] * np.cos(angle2)
        y2 = det2['distance'] * np.sin(angle2)
        
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    def pixel_to_angle(self, pixel_x):
        """Convert pixel x coordinate to angle"""
        if self.scan is None:
            return 0
        
        normalized_x = pixel_x / 640  # Assuming 640px width
        angle = self.scan.angle_min + normalized_x * (self.scan.angle_max - self.scan.angle_min)
        return angle
    
    def update_tree_trackers_robust(self, current_detections):
        """
        Much more robust tracking logic
        Prevents trees from constantly appearing/disappearing
        """
        # Mark all trackers as missed this frame initially
        for tracker in self.tree_trackers.values():
            tracker.miss_frame(self.frame_count)
        
        # Try to match each detection to existing trackers
        unmatched_detections = []
        
        for detection in current_detections:
            matched = False
            best_match = None
            best_distance = float('inf')
            
            # Find best matching tracker
            for tracker in self.tree_trackers.values():
                if tracker.is_match(detection):
                    dist = np.sqrt((tracker.position[0] - detection['center_x'])**2 + 
                                  (tracker.position[1] - detection['center_y'])**2)
                    if dist < best_distance:
                        best_distance = dist
                        best_match = tracker
            
            if best_match:
                # Update existing tracker
                best_match.update(detection, self.frame_count)
                matched = True
            else:
                # No match found
                unmatched_detections.append(detection)
        
        # Only create new trackers if we have very few existing trees
        # This prevents ID explosion
        active_count = len([t for t in self.tree_trackers.values() if t.is_active(self.frame_count)])
        
        for detection in unmatched_detections:
            # Be more conservative about creating new trees
            if active_count < 5:  # Max 5 trees at once (reasonable for your setup)
                # Check if this detection is far enough from existing trees
                too_close_to_existing = False
                for tracker in self.tree_trackers.values():
                    if tracker.is_active(self.frame_count):
                        pixel_dist = np.sqrt((tracker.position[0] - detection['center_x'])**2 + 
                                           (tracker.position[1] - detection['center_y'])**2)
                        if pixel_dist < 100:  # Don't create new tree too close to existing
                            too_close_to_existing = True
                            break
                
                if not too_close_to_existing:
                    new_tracker = RobustTreeTracker(self.next_tree_id, detection)
                    new_tracker.last_seen_frame = self.frame_count
                    self.tree_trackers[self.next_tree_id] = new_tracker
                    
                    distance_info = f" at {detection['distance']:.2f}m" if detection['distance'] else ""
                    self.get_logger().info(f'🆕 NEW TREE DETECTED! Tree #{self.next_tree_id}{distance_info}')
                    self.next_tree_id += 1
                    active_count += 1
    
    def cleanup_inactive_trackers(self):
        """Remove truly inactive trackers"""
        inactive_ids = []
        
        for tree_id, tracker in self.tree_trackers.items():
            if not tracker.is_active(self.frame_count):
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
                missed_info = f"missed:{tracker.missed_frames}"
                
                self.get_logger().info(f'  Tree #{tracker.id}: {status} | {distance_info} | {missed_info} | total:{tracker.total_detections}')
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
                    
                    marker = Marker()
                    marker.header.frame_id = 'map'
                    marker.header.stamp = self.get_clock().now().to_msg()
                    marker.ns = 'robust_trees'
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
                        marker.color.r = 0.0
                        marker.color.g = 0.8
                        marker.color.b = 0.0
                        marker.color.a = 0.9
                    else:
                        marker.color.r = 1.0
                        marker.color.g = 0.6
                        marker.color.b = 0.0
                        marker.color.a = 0.7
                    
                    marker_array.markers.append(marker)
        
        self.marker_array_pub.publish(marker_array)
    
    def create_debug_image(self, original_image, current_detections):
        """Create debug image"""
        debug_img = original_image.copy()
        
        # Draw current detections in yellow
        for detection in current_detections:
            x, y, w, h = detection['bbox']
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            
            distance_text = f"{detection['distance']:.1f}m" if detection['distance'] else "?m"
            cv2.putText(debug_img, distance_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw tracked trees
        for tracker in self.tree_trackers.values():
            if tracker.is_active(self.frame_count):
                x, y = int(tracker.position[0] - 40), int(tracker.position[1] - 40)
                w, h = 80, 80
                
                color = (0, 255, 0) if tracker.is_stable else (0, 165, 255)
                
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 3)
                cv2.circle(debug_img, (int(tracker.position[0]), int(tracker.position[1])), 8, color, -1)
                
                # Label
                label = f"T#{tracker.id}"
                status = "STABLE" if tracker.is_stable else f"M{tracker.missed_frames}"
                cv2.putText(debug_img, f"{label} {status}", (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Status
        active_count = len([t for t in self.tree_trackers.values() if t.is_active(self.frame_count)])
        stable_count = len([t for t in self.tree_trackers.values() if t.is_active(self.frame_count) and t.is_stable])
        
        cv2.putText(debug_img, f"Trees: {active_count} | Stable: {stable_count} | Current: {len(current_detections)}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return debug_img
    
    # Utility methods
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
        tree_detector.get_logger().info('🚀 Starting SIMPLE ROBUST tree detector...')
        # tree_detector.get_logger().info('🎯 Back to basics - reliable detection with stable tracking')
        # tree_detector.get_logger().info('🔒 Prevents ID jumping and false detection spam')
        rclpy.spin(tree_detector)
    except KeyboardInterrupt:
        tree_detector.get_logger().info('Simple robust tree detector shutting down...')
    finally:
        if 'tree_detector' in locals():
            tree_detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
