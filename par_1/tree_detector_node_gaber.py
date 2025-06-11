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
import math

class TreeDetectorNode(Node):
    def __init__(self):
        super().__init__('tree_detector_node_gaber')
        
        # ... (all your existing __init__ code for subscribers, publishers, TF, etc.) ...
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

        self.bridge = CvBridge()
        self.scan = None
        
        # Subscribers, Publishers, TF setup (as before)
        self.image_sub = self.create_subscription(Image, 'oak/rgb/image_raw', self.image_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.tree_detection_pub = self.create_publisher(String, '/detected_trees_status', 10)
        self.tree_marker_pub = self.create_publisher(Marker, '/tree_markers_viz', 10)
        self.debug_img_pub = self.create_publisher(Image, '/tree_detection/debug_image', 10)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # HSV Color Range for Brown Cylinders
        self.brown_hsv_lower = np.array([5, 60, 20])   # Tuned slightly from before based on bark image
        self.brown_hsv_upper = np.array([30, 200, 150]) # Value might be lower for dark bark

        # Contour Filtering Parameters
        self.min_contour_area = 500    
        self.max_contour_area = 70000  # Increased slightly
        self.min_aspect_ratio = 1.5    
        self.max_aspect_ratio = 10.0 # Increased slightly for potentially thinner views due to pattern
        self.min_solidity = 0.75       # Slightly relaxed for potentially more complex outline with bark
        self.min_extent = 0.55         # Slightly relaxed

        # --- Gabor Filter Parameters for Vertical Bark Pattern ---
        self.gabor_ksize = (31, 31)     # Kernel size (odd numbers) - size of the filter
        self.gabor_sigma = 4.0          # Standard deviation of the Gaussian envelope
        self.gabor_theta = np.pi / 2    # Orientation: pi/2 for vertical patterns (0 for horizontal)
        self.gabor_lambda = 10.0        # Wavelength of the sinusoidal factor (controls "thickness" of stripes it detects)
        self.gabor_gamma = 0.5          # Spatial aspect ratio (ellipticity of the support of Gabor function)
        self.gabor_psi = 0              # Phase offset
        # Threshold for Gabor response to confirm pattern
        self.min_gabor_response_mean = 25.0 # This is a CRITICAL tuning parameter

        # Create Gabor kernel once
        self.gabor_kernel = cv2.getGaborKernel(
            self.gabor_ksize, self.gabor_sigma, self.gabor_theta,
            self.gabor_lambda, self.gabor_gamma, self.gabor_psi, ktype=cv2.CV_32F
        )
        # Normalize kernel for better visualization/consistent response (optional but good practice)
        # self.gabor_kernel /= 1.5 * self.gabor_kernel.sum()


        # Detection tracking (as before)
        self.total_detection_events = 0 
        self.frame_count = 0
        self.last_detection_frame = 0
        self.last_no_detection_log_time = self.get_clock().now()
        self.detected_trees_map_positions = []

        self.get_logger().info('Enhanced Tree Detector Node with Pattern Verification initialized!')
        self.get_logger().info('Gabor filter parameters and response threshold need careful tuning.')

    # ... (scan_callback, get_world_coordinates, transform_to_map_frame, is_duplicate_map_tree, publish_tree_marker_in_map as before) ...
    def scan_callback(self, msg):
        self.scan = msg
        
    def get_world_coordinates(self, center_x_pixel, image_width): # Copied from previous for completeness
        if self.scan is None: return None
        try:
            normalized_x = center_x_pixel / image_width
            angle = self.scan.angle_min + normalized_x * (self.scan.angle_max - self.scan.angle_min)
            
            index = int((angle - self.scan.angle_min) / self.scan.angle_increment)
            if not (0 <= index < len(self.scan.ranges)):
                self.get_logger().warn(f'LiDAR index {index} out of range ({len(self.scan.ranges)}) for pixel X {center_x_pixel}')
                return None
            
            distance = self.scan.ranges[index]
            if not (self.scan.range_min <= distance <= self.scan.range_max and np.isfinite(distance)):
                return None
            
            point_lidar_frame = PointStamped()
            point_lidar_frame.header.frame_id = self.scan.header.frame_id
            point_lidar_frame.header.stamp = self.scan.header.stamp 
            point_lidar_frame.point.x = distance * math.cos(angle)
            point_lidar_frame.point.y = distance * math.sin(angle)
            point_lidar_frame.point.z = 0.0 
            
            map_coords = self.transform_to_map_frame(point_lidar_frame)
            if map_coords:
                map_x, map_y = map_coords
                return (point_lidar_frame.point.x, point_lidar_frame.point.y, map_x, map_y)
            else: 
                return (point_lidar_frame.point.x, point_lidar_frame.point.y, None, None)
                
        except Exception as e:
            self.get_logger().error(f'Error in get_world_coordinates: {str(e)}')
            return None
    
    def transform_to_map_frame(self, point_stamped_msg): # Copied from previous for completeness
        try:
            timeout_duration = rclpy.duration.Duration(seconds=0.1)
            if self.tf_buffer.can_transform('map', point_stamped_msg.header.frame_id, point_stamped_msg.header.stamp, timeout=timeout_duration):
                point_map_frame = self.tf_buffer.lookup_transform(
                    'map', 
                    point_stamped_msg.header.frame_id, 
                    point_stamped_msg.header.stamp, 
                    timeout=timeout_duration)
                transformed_point = tf2_geometry_msgs.do_transform_point(point_stamped_msg, point_map_frame)
                return (transformed_point.point.x, transformed_point.point.y)
            else:
                self.get_logger().warn(f'TF transform from {point_stamped_msg.header.frame_id} to map not available.', throttle_duration_sec=5)
                return None
        except Exception as e: # More generic exception catch for TF
            self.get_logger().error(f'TF exception in transform_to_map_frame: {type(e).__name__} - {e}', throttle_duration_sec=5)
            return None

    def is_duplicate_map_tree(self, new_map_pos, threshold=0.5): # Copied
        new_x, new_y = new_map_pos
        for old_x, old_y in self.detected_trees_map_positions:
            if math.hypot(new_x - old_x, new_y - old_y) < threshold:
                return True
        return False

    def publish_tree_marker_in_map(self, map_x, map_y, marker_id): # Copied
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'detected_tree_cylinders'
        marker.id = marker_id 
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = map_x; marker.pose.position.y = map_y; marker.pose.position.z = 0.5 
        marker.pose.orientation.w = 1.0 
        marker.scale.x = 0.20; marker.scale.y = 0.20; marker.scale.z = 1.0   
        marker.color.r = 0.6; marker.color.g = 0.3; marker.color.b = 0.1; marker.color.a = 0.8 
        self.tree_marker_pub.publish(marker)

    def verify_tree_pattern(self, roi_image):
        """
        Verifies if the ROI contains the expected vertical bark pattern using a Gabor filter.
        Returns: True if pattern is likely present, False otherwise.
        """
        if roi_image is None or roi_image.size == 0 or roi_image.shape[0] < self.gabor_ksize[0] or roi_image.shape[1] < self.gabor_ksize[1]:
            # self.get_logger().debug("ROI too small for Gabor filter.")
            return False # ROI too small

        gray_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gabor filter
        gabor_response = cv2.filter2D(gray_roi, cv2.CV_32F, self.gabor_kernel)
        
        # We are interested in strong responses, so take absolute values
        abs_gabor_response = np.abs(gabor_response)
        
        # Calculate the mean of the Gabor response.
        # High mean indicates strong presence of features matching the filter.
        mean_response = np.mean(abs_gabor_response)
        
        # Store for debugging
        setattr(self, 'last_gabor_response_img', gabor_response) # Store raw for visualization
        setattr(self, 'last_gabor_mean_response', mean_response)

        if mean_response > self.min_gabor_response_mean:
            # self.get_logger().debug(f"Pattern VERIFIED. Gabor mean: {mean_response:.2f}")
            return True
        else:
            # self.get_logger().debug(f"Pattern FAILED. Gabor mean: {mean_response:.2f} (Threshold: {self.min_gabor_response_mean})")
            return False

    def image_callback(self, msg):
        if self.scan is None:
            return
            
        self.frame_count += 1
        setattr(self, 'last_gabor_response_img', None) # Clear last Gabor response for debug image
        setattr(self, 'last_gabor_mean_response', 0.0)

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            color_mask = cv2.inRange(hsv_image, self.brown_hsv_lower, self.brown_hsv_upper)
            
            kernel_open = np.ones((5, 5), np.uint8) 
            mask_opened = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
            kernel_close = np.ones((7, 7), np.uint8)
            mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel_close, iterations=1)
            
            contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detected_trees_in_frame = [] 

            if contours:
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if not (self.min_contour_area < area < self.max_contour_area):
                        continue

                    x_bbox, y_bbox, w_bbox, h_bbox = cv2.boundingRect(contour)
                    aspect_ratio = float(h_bbox) / w_bbox if w_bbox > 0 else 0
                    if not (self.min_aspect_ratio < aspect_ratio < self.max_aspect_ratio):
                        continue
                    
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = float(area) / hull_area if hull_area > 0 else 0
                    if solidity < self.min_solidity:
                        continue

                    # --- New: Pattern Verification Step ---
                    # Extract the ROI from the original BGR image
                    roi = cv_image[y_bbox : y_bbox + h_bbox, x_bbox : x_bbox + w_bbox]
                    
                    if not self.verify_tree_pattern(roi):
                        # self.get_logger().debug("Contour failed pattern verification.")
                        continue # Skip this contour if pattern doesn't match
                    # --- End of Pattern Verification ---

                    # If all filters pass, including pattern:
                    center_x_pixel = x_bbox + w_bbox / 2.0
                    center_y_pixel = y_bbox + h_bbox / 2.0
                    
                    detected_trees_in_frame.append({
                        'bbox': (x_bbox, y_bbox, w_bbox, h_bbox),
                        'center_pixel': (center_x_pixel, center_y_pixel),
                        'area': area, 'aspect_ratio': aspect_ratio, 'solidity': solidity,
                        'gabor_mean': getattr(self, 'last_gabor_mean_response', 0.0) # Store for debug
                    })

            if detected_trees_in_frame:
                self.handle_trees_detected(cv_image, detected_trees_in_frame) # (Pass cv_image for get_world_coordinates)
            else:
                self.handle_no_trees_detected()
            
            debug_img = self.create_debug_image(cv_image, detected_trees_in_frame, mask_closed)
            self.debug_img_pub.publish(self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8'))
                
        except Exception as e:
            self.get_logger().error(f'Error in image_callback: {type(e).__name__} - {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())


    def handle_trees_detected(self, cv_image, trees_data): # Added cv_image param
        # ... (same as your previous handle_trees_detected, ensure it uses cv_image.shape[1] for get_world_coordinates) ...
        num_detected = len(trees_data)
        status_msg_str = f"trees_detected:{num_detected}"
        status_msg = String(); status_msg.data = status_msg_str
        self.tree_detection_pub.publish(status_msg)
        
        if self.frame_count - self.last_detection_frame > 30: 
            self.total_detection_events += 1
            self.get_logger().info(f'TREES DETECTED! Found {num_detected} tree(s) | Event #{self.total_detection_events}')
            self.last_detection_frame = self.frame_count
            
            for i, tree_info in enumerate(trees_data):
                center_x_pixel = tree_info['center_pixel'][0]
                self.get_logger().info(f"  Tree #{i+1} PxX:{center_x_pixel:.0f} A:{tree_info['area']:.0f} AR:{tree_info['aspect_ratio']:.2f} S:{tree_info['solidity']:.2f} GaborM:{tree_info.get('gabor_mean', 0.0):.1f}")
                
                world_coords = self.get_world_coordinates(center_x_pixel, cv_image.shape[1]) 
                if world_coords:
                    lidar_x, lidar_y, map_x, map_y = world_coords
                    self.get_logger().info(f"    Lidar:({lidar_x:.2f},{lidar_y:.2f}), Map:({map_x:.2f},{map_y:.2f})")
                    if map_x is not None and map_y is not None: # Check if map coords are valid
                        if not self.is_duplicate_map_tree((map_x, map_y)):
                            self.publish_tree_marker_in_map(map_x, map_y, len(self.detected_trees_map_positions))
                            self.detected_trees_map_positions.append((map_x, map_y))
    
    def handle_no_trees_detected(self): # Same as before
        status_msg = String(); status_msg.data = "trees_detected:0"
        self.tree_detection_pub.publish(status_msg)
        current_time = self.get_clock().now()
        if (current_time - self.last_no_detection_log_time).nanoseconds / 1e9 > 5.0:
            self.get_logger().info('No trees detected in current view - scanning...')
            self.last_no_detection_log_time = current_time
    
    def create_debug_image(self, original_image, detected_trees_data, processed_mask):
        debug_img = original_image.copy()
        
        # Optionally show Gabor response for the *first detected tree's ROI* for tuning
        # This is a bit hacky for a general debug image, but useful for tuning Gabor.
        gabor_response_to_show = getattr(self, 'last_gabor_response_img', None)
        if gabor_response_to_show is not None and len(detected_trees_data) > 0:
            first_tree_bbox = detected_trees_data[0]['bbox']
            x,y,w,h = first_tree_bbox
            if gabor_response_to_show.shape[0] == h and gabor_response_to_show.shape[1] == w: # Ensure it matches
                # Normalize for visualization
                vis_gabor = cv2.normalize(np.abs(gabor_response_to_show), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                vis_gabor_bgr = cv2.cvtColor(vis_gabor, cv2.COLOR_GRAY2BGR)
                debug_img[y:y+h, x:x+w] = cv2.addWeighted(debug_img[y:y+h, x:x+w], 0.5, vis_gabor_bgr, 0.5, 0)


        for i, tree_info in enumerate(detected_trees_data):
            x, y, w, h = tree_info['bbox']
            color = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)][i % 6]
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 2)
            label = f"T{i+1} G:{tree_info.get('gabor_mean',0.0):.0f}" # Show Gabor mean
            cv2.putText(debug_img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        cv2.putText(debug_img, f"F:{self.frame_count}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
        cv2.putText(debug_img, f"Trees:{len(detected_trees_data)}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0) if detected_trees_data else (255,0,0),1)
        
        return debug_img

# main function (same as before)
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
