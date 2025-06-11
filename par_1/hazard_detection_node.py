#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from visualization_msgs.msg import Marker # Using single Marker publisher
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros
import tf2_geometry_msgs
import math # For aspect ratio calculations, etc.

class TreeDetectorNode(Node): # Renamed class for clarity
    def __init__(self):
        super().__init__('tree_detector_node') # Updated node name
        
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
        self.tree_detection_pub = self.create_publisher( # Renamed for clarity
            String,
            '/detected_trees_status', # More descriptive topic
            10)
        
        self.tree_marker_pub = self.create_publisher( # Renamed for clarity
            Marker,
            '/tree_markers_viz', # More descriptive topic for RViz markers
            10) # Queue size of 10 is fine for markers
        
        self.debug_img_pub = self.create_publisher(
            Image,
            '/tree_detection/debug_image',
            10)
        
        # TF setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # --- Detection Parameters ---
        # HSV Color Range for Brown Cylinders
        # These values are CRITICAL and need tuning for your specific cylinders and lighting.
        # Use a color picker tool in HSV space while looking at your camera feed.
        self.brown_hsv_lower = np.array([5, 80, 30])   # Example: Hue, Saturation, Value
        self.brown_hsv_upper = np.array([25, 255, 200]) # Example

        # Contour Filtering Parameters
        self.min_contour_area = 500     # Minimum pixel area to be considered a potential tree
                                        # Adjust based on how large trees appear at relevant distances
        self.max_contour_area = 50000   # Maximum pixel area (to ignore very large blobs if any)

        # Aspect Ratio (Height / Width) - Cylinders are taller than they are wide
        self.min_aspect_ratio = 1.5     # e.g., height must be at least 1.5x width
        self.max_aspect_ratio = 8.0     # e.g., not excessively skinny (might be noise or pole)

        # Solidity (Contour Area / Convex Hull Area) - Should be high for regular shapes
        self.min_solidity = 0.80        # Closer to 1.0 means more convex and less irregular

        # Extent (Contour Area / Bounding Box Area) - Also for shape regularity
        self.min_extent = 0.60

        # Detection tracking
        self.total_detection_events = 0 # Counts frames where trees are found
        self.frame_count = 0
        self.last_detection_frame = 0
        self.last_no_detection_log_time = self.get_clock().now()
        self.detected_trees_map_positions = [] # Store map coordinates of confirmed trees to avoid re-publishing markers for the same tree too often

        self.get_logger().info('Enhanced Tree Detector Node initialized!')
        self.get_logger().info('Parameters for HSV, area, aspect ratio, solidity, and extent need careful tuning.')
        
    def scan_callback(self, msg):
        self.scan = msg
        
    def image_callback(self, msg):
        if self.scan is None:
            # self.get_logger().warn('Waiting for LiDAR scan data...', throttle_duration_sec=5)
            return
            
        self.frame_count += 1
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # 1. HSV Color Masking
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            color_mask = cv2.inRange(hsv_image, self.brown_hsv_lower, self.brown_hsv_upper)
            
            # 2. Morphological Operations (to clean up the mask)
            # MORPH_OPEN: Erosion followed by Dilation (removes small noise)
            kernel_open = np.ones((5, 5), np.uint8) 
            mask_opened = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
            # MORPH_CLOSE: Dilation followed by Erosion (fills small holes in objects)
            kernel_close = np.ones((7, 7), np.uint8)
            mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel_close, iterations=1)
            
            # 3. Find Contours on the cleaned mask
            contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detected_trees_in_frame = [] # Store data of trees found in this specific frame

            if contours:
                for contour in contours:
                    # --- Apply Contour Filters ---
                    area = cv2.contourArea(contour)

                    # Area Filter
                    if not (self.min_contour_area < area < self.max_contour_area):
                        continue

                    x_bbox, y_bbox, w_bbox, h_bbox = cv2.boundingRect(contour)

                    # Aspect Ratio Filter
                    aspect_ratio = float(h_bbox) / w_bbox if w_bbox > 0 else 0
                    if not (self.min_aspect_ratio < aspect_ratio < self.max_aspect_ratio):
                        continue
                    
                    # Solidity Filter
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = float(area) / hull_area if hull_area > 0 else 0
                    if solidity < self.min_solidity:
                        continue

                    # Extent Filter
                    # extent = float(area) / (w_bbox * h_bbox) if (w_bbox * h_bbox) > 0 else 0
                    # if extent < self.min_extent: # Can be redundant if solidity is good
                    #     continue

                    # If all filters pass, consider it a potential tree
                    center_x_pixel = x_bbox + w_bbox / 2
                    center_y_pixel = y_bbox + h_bbox / 2
                    
                    # Add bounding box and center for drawing on debug image
                    detected_trees_in_frame.append({
                        'bbox': (x_bbox, y_bbox, w_bbox, h_bbox),
                        'center_pixel': (center_x_pixel, center_y_pixel),
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'solidity': solidity
                    })

            # --- Process and Publish Detections ---
            if detected_trees_in_frame:
                self.handle_trees_detected(cv_image, detected_trees_in_frame)
            else:
                self.handle_no_trees_detected()
            
            # Always publish debug image
            debug_img = self.create_debug_image(cv_image, detected_trees_in_frame, mask_closed) # Pass cleaned mask too
            self.debug_img_pub.publish(self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8'))
                
        except Exception as e:
            self.get_logger().error(f'Error in image_callback: {str(e)}')
    
    # def handle_trees_detected(self, cv_image, trees_data):
    #     num_detected = len(trees_data)
    #     status_msg_str = f"trees_detected:{num_detected}"
        
    #     status_msg = String()
    #     status_msg.data = status_msg_str
    #     self.tree_detection_pub.publish(status_msg)
        
    #     # if self.frame_count - self.last_detection_frame > 30: # Log detailed info less frequently
    #     if num_detected > 0 and (self.frame_count - self.last_detection_frame > 15 or self.total_detection_events == 0) : # Log more often if trees are seen
    #         self.total_detection_events += 1
    #         self.get_logger().info(f'TREES DETECTED! Found {num_detected} tree(s) | Event #{self.total_detection_events}')
    #         self.last_detection_frame = self.frame_count
            
    #         for i, tree_info in enumerate(trees_data):
    #             center_x_pixel = tree_info['center_pixel'][0]
    #             self.get_logger().info(f"  Tree #{i+1} at pixel X: {center_x_pixel:.0f}, Area: {tree_info['area']:.0f}, AR: {tree_info['aspect_ratio']:.2f}, Solidity: {tree_info['solidity']:.2f}")
                
    #             # Get world coordinates and publish markers
    #             world_coords = self.get_world_coordinates(center_x_pixel, cv_image.shape[1]) # image_width
    #             if world_coords:
    #                 lidar_x, lidar_y, map_x, map_y = world_coords
    #                 self.get_logger().info(f"    Lidar Coords: ({lidar_x:.2f}, {lidar_y:.2f}), Map Coords: ({map_x:.2f}, {map_y:.2f})")
                    
    #                 # Publish RViz marker only if it's a "new" tree in map frame
    #                 if not self.is_duplicate_map_tree((map_x, map_y)):
    #                     self.publish_tree_marker_in_map(map_x, map_y, len(self.detected_trees_map_positions))
    #                     self.detected_trees_map_positions.append((map_x, map_y))

    def handle_trees_detected(self, cv_image, trees_data):
        num_detected = len(trees_data)
        status_msg_str = f"trees_detected:{num_detected}"
        
        status_msg = String()
        status_msg.data = status_msg_str
        self.tree_detection_pub.publish(status_msg)
        
        # Log detailed info less frequently, or every time if num_detected > 0
        if num_detected > 0 and (self.frame_count - self.last_detection_frame > 15 or self.total_detection_events == 0) : # Log more often if trees are seen
            self.total_detection_events += 1
            self.get_logger().info(f'TREES DETECTED! Found {num_detected} tree(s) | Event #{self.total_detection_events}')
            self.last_detection_frame = self.frame_count
            
            for i, tree_info in enumerate(trees_data):
                center_x_pixel = tree_info['center_pixel'][0]
                
                # Basic info before world_coords
                basic_info_str = (f"  Tree #{i+1} at PxX: {center_x_pixel:.0f}, "
                                  f"Area: {tree_info['area']:.0f}, "
                                  f"AR: {tree_info['aspect_ratio']:.2f}, "
                                  f"Solidity: {tree_info['solidity']:.2f}")

                world_coords_result = self.get_world_coordinates(center_x_pixel, cv_image.shape[1]) # image_width
                
                if world_coords_result:
                    # Unpack 5 values now
                    lidar_x, lidar_y, map_x, map_y, raw_dist = world_coords_result 
                    
                    # Prepare strings for lidar and map coordinates, handling None for map
                    lidar_str = f"({lidar_x:.2f},{lidar_y:.2f})" # lidar_x, lidar_y should always be valid if raw_dist is
                    map_str = f"({map_x:.2f},{map_y:.2f})" if map_x is not None and map_y is not None else "(Map N/A)"
                    
                    # Log all info including the raw distance
                    self.get_logger().info(f"{basic_info_str}, Dist: {raw_dist:.2f}m, LidarXY:{lidar_str}, MapXY:{map_str}")
                    
                    if map_x is not None and map_y is not None: # Check before using for marker
                        if not self.is_duplicate_map_tree((map_x, map_y)):
                            self.publish_tree_marker_in_map(map_x, map_y, len(self.detected_trees_map_positions))
                            self.detected_trees_map_positions.append((map_x, map_y))
                else:
                    self.get_logger().info(f"{basic_info_str}, Dist: N/A (World coords failed)")
        elif num_detected == 0: # Ensure this branch from the original logic is covered
            self.handle_no_trees_detected() # Or call the no_trees_detected method if appropriate

    def handle_no_trees_detected(self):
        status_msg = String()
        status_msg.data = "trees_detected:0"
        self.tree_detection_pub.publish(status_msg)
        
        current_time = self.get_clock().now()
        if (current_time - self.last_no_detection_log_time).nanoseconds / 1e9 > 5.0: # Log every 5 seconds
            self.get_logger().info('No trees detected in current view - scanning...')
            self.last_no_detection_log_time = current_time
    
    # def get_world_coordinates(self, center_x_pixel, image_width):
    #     if self.scan is None: return None
    #     try:
    #         normalized_x = center_x_pixel / image_width
    #         angle = self.scan.angle_min + normalized_x * (self.scan.angle_max - self.scan.angle_min)
            
    #         index = int((angle - self.scan.angle_min) / self.scan.angle_increment)
    #         if not (0 <= index < len(self.scan.ranges)):
    #             self.get_logger().warn(f'LiDAR index {index} out of range ({len(self.scan.ranges)}) for pixel X {center_x_pixel}')
    #             return None
            
    #         distance = self.scan.ranges[index]
    #         if not (self.scan.range_min <= distance <= self.scan.range_max and np.isfinite(distance)):
    #             # self.get_logger().warn(f'Invalid LiDAR distance: {distance:.2f} at index {index}')
    #             return None
            
    #         point_lidar_frame = PointStamped()
    #         point_lidar_frame.header.frame_id = self.scan.header.frame_id
    #         point_lidar_frame.header.stamp = self.scan.header.stamp # Use scan time for TF
    #         point_lidar_frame.point.x = distance * math.cos(angle)
    #         point_lidar_frame.point.y = distance * math.sin(angle)
    #         point_lidar_frame.point.z = 0.0 # Assume trees are on the ground plane for this
            
    #         map_coords = self.transform_to_map_frame(point_lidar_frame)
    #         if map_coords:
    #             map_x, map_y = map_coords
    #             return (point_lidar_frame.point.x, point_lidar_frame.point.y, map_x, map_y)
    #         else: # Could not transform
    #             return (point_lidar_frame.point.x, point_lidar_frame.point.y, None, None)
                
    #     except Exception as e:
    #         self.get_logger().error(f'Error in get_world_coordinates: {str(e)}')
    #         return None
    
    def get_world_coordinates(self, center_x_pixel, image_width):
        if self.scan is None:
            self.get_logger().warn("Scan data missing in get_world_coordinates.", throttle_duration_sec=5)
            return None # Return None if scan is missing
        if image_width <= 0:
            self.get_logger().error(f"Invalid image_width: {image_width}")
            return None # Return None for invalid image width

        try:
            center_x_pixel = np.clip(center_x_pixel, 0, image_width -1)
            normalized_x = center_x_pixel / float(image_width) 
            angle_span = self.scan.angle_max - self.scan.angle_min
            angle = self.scan.angle_min + normalized_x * angle_span
            index = int((angle - self.scan.angle_min) / self.scan.angle_increment)
            
            if not (0 <= index < len(self.scan.ranges)):
                self.get_logger().warn(f'LiDAR index {index} out of range ({len(self.scan.ranges)}). Angle: {math.degrees(angle):.1f}deg, NormX: {normalized_x:.2f}, PxX: {center_x_pixel}')
                return None # Return None if index is out of range
            
            # This is the raw distance from LiDAR for the calculated angle
            raw_lidar_distance = self.scan.ranges[index] 

            if not (self.scan.range_min <= raw_lidar_distance <= self.scan.range_max and np.isfinite(raw_lidar_distance)):
                # self.get_logger().debug(f'Invalid LiDAR distance: {raw_lidar_distance:.2f} at index {index} for angle {math.degrees(angle):.1f}deg.')
                return None # Return None if distance is invalid
            
            # If we reach here, raw_lidar_distance is valid.
            # self.get_logger().info(f"Valid LiDAR point: PxX:{center_x_pixel}, NormX:{normalized_x:.3f}, Angle:{math.degrees(angle):.1f}deg, Index:{index}, Dist:{raw_lidar_distance:.2f}m")

            point_lidar_frame = PointStamped()
            point_lidar_frame.header.frame_id = self.scan.header.frame_id
            point_lidar_frame.header.stamp = self.scan.header.stamp 
            point_lidar_frame.point.x = raw_lidar_distance * math.cos(angle)
            point_lidar_frame.point.y = raw_lidar_distance * math.sin(angle)
            point_lidar_frame.point.z = 0.0 
            
            map_coords = self.transform_to_map_frame(point_lidar_frame)

            if map_coords:
                map_x, map_y = map_coords
                # Return all values including the raw_lidar_distance
                return (point_lidar_frame.point.x, point_lidar_frame.point.y, map_x, map_y, raw_lidar_distance)
            else: 
                # Still return raw_lidar_distance even if map transform fails
                return (point_lidar_frame.point.x, point_lidar_frame.point.y, None, None, raw_lidar_distance)
                
        except Exception as e:
            self.get_logger().error(f'Error in get_world_coordinates: {type(e).__name__} - {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            return None # Return None on exception
    

    def transform_to_map_frame(self, point_stamped_msg):
        try:
            # Use a reasonable timeout for can_transform and lookup_transform
            timeout_duration = rclpy.duration.Duration(seconds=0.1)
            
            if self.tf_buffer.can_transform('map', point_stamped_msg.header.frame_id, point_stamped_msg.header.stamp, timeout=timeout_duration):
                point_map_frame = self.tf_buffer.lookup_transform(
                    'map', 
                    point_stamped_msg.header.frame_id, 
                    point_stamped_msg.header.stamp, # Use the timestamp from the source data
                    timeout=timeout_duration)
                
                # Apply the transform using tf2_geometry_msgs
                transformed_point = tf2_geometry_msgs.do_transform_point(point_stamped_msg, point_map_frame)
                return (transformed_point.point.x, transformed_point.point.y)
            else:
                self.get_logger().warn(f'TF transform from {point_stamped_msg.header.frame_id} to map not available.', throttle_duration_sec=5)
                return None
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException, tf2_ros.TransformException) as e:
            self.get_logger().error(f'TF exception in transform_to_map_frame: {e}', throttle_duration_sec=5)
            return None

    def is_duplicate_map_tree(self, new_map_pos, threshold=0.5): # 50cm threshold
        """Checks if a new tree detection in map coordinates is close to an existing one."""
        new_x, new_y = new_map_pos
        for old_x, old_y in self.detected_trees_map_positions:
            if math.hypot(new_x - old_x, new_y - old_y) < threshold:
                return True
        return False

    def publish_tree_marker_in_map(self, map_x, map_y, marker_id):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'detected_tree_cylinders'
        marker.id = marker_id # Unique ID for each new tree marker
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        
        marker.pose.position.x = map_x
        marker.pose.position.y = map_y
        marker.pose.position.z = 0.5  # Assuming cylinder base is on ground, marker center at 0.5m height
        marker.pose.orientation.w = 1.0 # Default orientation (upright)
        
        marker.scale.x = 0.20  # Diameter of cylinder (adjust based on your tree size)
        marker.scale.y = 0.20
        marker.scale.z = 1.0   # Height of cylinder
        
        marker.color.r = 0.6  # Brownish color
        marker.color.g = 0.3
        marker.color.b = 0.1
        marker.color.a = 0.8  # Slightly transparent

        # Lifetime (optional, 0 means forever until deleted or replaced by ID)
        # marker.lifetime = rclpy.duration.Duration(seconds=30).to_msg()

        self.tree_marker_pub.publish(marker)
        # self.get_logger().info(f'Published tree marker #{marker_id} at map ({map_x:.2f}, {map_y:.2f})')
    
    def create_debug_image(self, original_image, detected_trees_data, processed_mask):
        debug_img = original_image.copy()
        
        # Overlay the binary mask (optional, can be noisy)
        # mask_viz = cv2.cvtColor(processed_mask, cv2.COLOR_GRAY2BGR)
        # debug_img = cv2.addWeighted(debug_img, 0.7, mask_viz, 0.3, 0)

        for i, tree_info in enumerate(detected_trees_data):
            x, y, w, h = tree_info['bbox']
            # Use different colors for each tree in this frame
            color = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)][i % 6]
            
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 2)
            
            label = f"T{i+1} AR:{tree_info['aspect_ratio']:.1f} S:{tree_info['solidity']:.2f}"
            cv2.putText(debug_img, label, (x, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        cv2.putText(debug_img, f"Frame: {self.frame_count}", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(debug_img, f"Trees in view: {len(detected_trees_data)}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if detected_trees_data else (255,0,0), 1)
        
        return debug_img

def main(args=None):
    rclpy.init(args=args)
    try:
        tree_detector = TreeDetectorNode()
        rclpy.spin(tree_detector)
    except KeyboardInterrupt:
        tree_detector.get_logger().info('Tree Detector Node shutting down...')
    finally:
        if 'tree_detector' in locals() and rclpy.ok():
            tree_detector.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
