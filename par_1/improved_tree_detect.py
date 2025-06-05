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
        
        # Consider stable after 3 detections
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
    Real-time tree detector with persistent tracking and immediate detection display
    Shows both immediate detections and maintains persistent tree IDs
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
        
        # Detection parameters (made less restrictive for real-time detection)
        self.brown_hsv_lower = np.array([8, 40, 20])
        self.brown_hsv_upper = np.array([25, 255, 200])
        
        # More permissive size filtering for immediate detection
        self.min_area = 250
        self.max_area = 8000
        self.min_aspect_ratio = 0.4
        self.max_aspect_ratio = 4.0
        
        # Clustering parameters
        self.cluster_distance = 100
        
        # Tree tracking
        self.tree_trackers = {}  # Dict of TreeTracker objects
        self.next_tree_id = 1
        self.frame_count = 0
        self.last_log_frame = 0
        
        # LiDAR validation (more permissive)
        self.expected_tree_width_meters = 0.15
        self.size_tolerance = 0.8  # More permissive (80% tolerance)
        
        self.get_logger().info('Real-time Tree Detector initialized!')
        self.get_logger().info('🌳 Shows immediate detections with persistent Tree IDs')
        
    def scan_callback(self, msg):
        """Store LiDAR scan."""
        self.scan = msg
        
    def image_callback(self, msg):
        """Main detection callback - now with real-time tracking"""
        if self.scan is None:
            return
            
        try:
            self.frame_count += 1
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Detect trees with improved pipeline but less restrictive filtering
            current_detections = self.detect_trees_realtime(cv_image)
            
            # Update tree trackers with current detections
            self.update_tree_trackers(current_detections)
            
            # Clean up old trackers
            self.cleanup_inactive_trackers()
            
            # Publish results
            self.publish_detection_results()
            
            # Enhanced logging (every 30 frames = ~1 second)
            if self.frame_count - self.last_log_frame >= 30:
                self.log_current_state()
                self.last_log_frame = self.frame_count
            
            # Publish markers for all tracked trees
            self.publish_tree_markers(cv_image)
            
            # Debug visualization
            debug_img = self.create_realtime_debug_image(cv_image, current_detections)
            self.debug_img_pub.publish(self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8'))
                
        except Exception as e:
            self.get_logger().error(f'Error in real-time tree detection: {str(e)}')
    
    def detect_trees_realtime(self, image):
        """
        Real-time detection with immediate results (less filtering)
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask
        mask = cv2.inRange(hsv, self.brown_hsv_lower, self.brown_hsv_upper)
        
        # Light morphological operations for speed
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Publish mask for debugging
        self.mask_img_pub.publish(self.bridge.cv2_to_imgmsg(mask, encoding='mono8'))
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        # Filter and cluster contours
        valid_detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area <= area <= self.max_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / w if w > 0 else 0
                
                if self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio:
                    center_x = x + w / 2
                    center_y = y + h / 2
                    
                    # Get distance for this detection
                    distance = self.get_lidar_distance(center_x, image.shape[1])
                    
                    detection = {
                        'center_x': center_x,
                        'center_y': center_y,
                        'bbox': (x, y, w, h),
                        'area': area,
                        'distance': distance,
                        'aspect_ratio': aspect_ratio
                    }
                    
                    # Light size validation (more permissive)
                    if self.validate_detection_size(detection):
                        valid_detections.append(detection)
        
        # Cluster nearby detections
        clustered_detections = self.cluster_detections(valid_detections)
        
        return clustered_detections
    
    def cluster_detections(self, detections):
        """Cluster nearby detections - same logic but simpler"""
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
        """Merge cluster of detections"""
        if len(cluster) == 1:
            cluster[0]['cluster_size'] = 1
            return cluster[0]
        
        # Calculate merged bounding box
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
        """Light size validation - more permissive"""
        if detection['distance'] is None or detection['distance'] <= 0:
            return True  # Accept if no distance data
        
        expected_pixel_width = self.calculate_expected_pixel_size(detection['distance'])
        actual_pixel_width = detection['bbox'][2]
        
        if expected_pixel_width <= 0:
            return True
        
        size_ratio = actual_pixel_width / expected_pixel_width
        return (1 - self.size_tolerance) <= size_ratio <= (1 + self.size_tolerance)
    
    def update_tree_trackers(self, current_detections):
        """
        Update tree trackers with current detections
        This maintains persistent Tree IDs
        """
        # Mark all trackers as not updated this frame
        for tracker in self.tree_trackers.values():
            tracker.detection_count = 0  # Reset for this frame
        
        # Try to match each detection to existing trackers
        unmatched_detections = []
        
        for detection in current_detections:
            matched = False
            
            # Find best matching tracker
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
                # Update existing tracker
                best_match.update(detection, self.frame_count)
                matched = True
            else:
                # No match found, add to unmatched
                unmatched_detections.append(detection)
        
        # Create new trackers for unmatched detections
        for detection in unmatched_detections:
            new_tracker = TreeTracker(self.next_tree_id, detection)
            new_tracker.last_seen_frame = self.frame_count
            self.tree_trackers[self.next_tree_id] = new_tracker
            
            self.get_logger().info(f'🆕 NEW TREE DETECTED! Assigned Tree #{self.next_tree_id}')
            self.next_tree_id += 1
    
    def cleanup_inactive_trackers(self):
        """Remove trackers that haven't been seen recently"""
        inactive_ids = []
        
        for tree_id, tracker in self.tree_trackers.items():
            if not tracker.is_active(self.frame_count, max_frames_missing=15):  # 0.5 seconds
                inactive_ids.append(tree_id)
        
        for tree_id in inactive_ids:
            self.get_logger().info(f'❌ Tree #{tree_id} lost from view')
            del self.tree_trackers[tree_id]
    
    def publish_detection_results(self):
        """Publish detection results for navigation"""
        active_trees = [t for t in self.tree_trackers.values() if t.is_active(self.frame_count)]
        
        tree_msg = String()
        tree_msg.data = f"trees_detected:{len(active_trees)}"
        self.tree_pub.publish(tree_msg)
    
    def log_current_state(self):
        """Log current detection state with persistent IDs"""
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
        """Publish markers for all tracked trees"""
        marker_array = MarkerArray()
        
        for tracker in self.tree_trackers.values():
            if tracker.is_active(self.frame_count):
                # Get world coordinates
                world_coords = self.get_world_coordinates(tracker.position[0], cv_image.shape[1])
                
                if world_coords and world_coords[2] is not None:
                    lidar_x, lidar_y, map_x, map_y = world_coords
                    tracker.map_position = (map_x, map_y)
                    
                    # Create marker
                    marker = Marker()
                    marker.header.frame_id = 'map'
                    marker.header.stamp = self.get_clock().now().to_msg()
                    marker.ns = 'persistent_trees'
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
                        # Stable trees are brown
                        marker.color.r = 0.6
                        marker.color.g = 0.3
                        marker.color.b = 0.1
                        marker.color.a = 0.9
                    else:
                        # New trees are orange
                        marker.color.r = 1.0
                        marker.color.g = 0.5
                        marker.color.b = 0.0
                        marker.color.a = 0.7
                    
                    marker_array.markers.append(marker)
        
        self.marker_array_pub.publish(marker_array)
    
    def create_realtime_debug_image(self, original_image, current_detections):
        """Create debug image showing real-time tracking"""
        debug_img = original_image.copy()
        
        # Draw current detections
        for detection in current_detections:
            x, y, w, h = detection['bbox']
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Yellow for current
            cv2.putText(debug_img, "CURRENT", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw tracked trees with persistent IDs
        for tracker in self.tree_trackers.values():
            if tracker.is_active(self.frame_count):
                x, y = int(tracker.position[0] - 50), int(tracker.position[1] - 50)
                w, h = 100, 100  # Approximate box for visualization
                
                # Color based on stability
                color = (0, 255, 0) if tracker.is_stable else (0, 165, 255)  # Green=stable, Orange=new
                
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 3)
                cv2.circle(debug_img, (int(tracker.position[0]), int(tracker.position[1])), 10, color, -1)
                
                # Label with persistent ID
                label = f"TREE #{tracker.id}"
                status = " (STABLE)" if tracker.is_stable else " (NEW)"
                cv2.putText(debug_img, label + status, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Technical info
                tech_info = f"detections:{tracker.total_detections}"
                if tracker.distance:
                    tech_info += f" {tracker.distance:.1f}m"
                cv2.putText(debug_img, tech_info, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Status overlay
        active_count = len([t for t in self.tree_trackers.values() if t.is_active(self.frame_count)])
        stable_count = len([t for t in self.tree_trackers.values() if t.is_active(self.frame_count) and t.is_stable])
        
        cv2.putText(debug_img, f"🌳 Active Trees: {active_count} | Stable: {stable_count}", 
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.putText(debug_img, f"Frame: {self.frame_count} | Current detections: {len(current_detections)}", 
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return debug_img
    
    # Keep utility methods from previous version
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
        tree_detector.get_logger().info('🚀 Starting Improved tree detector with persistent tracking...')
        # tree_detector.get_logger().info('🌳 Trees get persistent IDs: Tree #1, Tree #2, etc.')
        # tree_detector.get_logger().info('🆕 New trees appear immediately, stable after 3 detections')
        rclpy.spin(tree_detector)
    except KeyboardInterrupt:
        tree_detector.get_logger().info('Improved tree detector shutting down...')
    finally:
        if 'tree_detector' in locals():
            tree_detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
