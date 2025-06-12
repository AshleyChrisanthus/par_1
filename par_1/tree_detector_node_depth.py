#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo # Added CameraInfo
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import tf2_ros
import tf2_geometry_msgs # For do_transform_point
import math

# For synchronizing messages
from message_filters import Subscriber, ApproximateTimeSynchronizer

class TreeDetectorNode(Node):
    def __init__(self):
        super().__init__('tree_detector_node_depth') # New node name

        self.bridge = CvBridge() # Bridge for color images
        self.depth_bridge = CvBridge() # Bridge for depth images

        self.fx = None; self.fy = None; self.cx = None; self.cy = None
        self.camera_optical_frame_id = "camera_link" # Default, will be overridden by CameraInfo

        # --- Subscribers using message_filters ---
        self.color_image_sub = Subscriber(self, Image, '/oak/rgb/image_raw')
        
        # ATTEMPT 1: Try to get depth aligned with /oak/rgb/image_raw by subscribing to its base name
        # We hope image_transport uses the /oak/rgb/image_raw/compressedDepth plugin for this.
        self.depth_image_sub = Subscriber(self, Image, '/oak/rgb/image_raw') 
        # Note: If this doesn't work, the next attempt would be to subscribe to '/oak/stereo/image_raw'
        # and use '/oak/stereo/camera_info', which complicates alignment with RGB detections.

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/oak/rgb/camera_info', # Camera info for the RGB camera
            self.camera_info_callback,
            rclpy.qos.QoSProfile(depth=1, reliability=rclpy.qos.ReliabilityPolicy.RELIABLE, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL) # For latched CameraInfo
        )

        self.time_synchronizer = ApproximateTimeSynchronizer(
            [self.color_image_sub, self.depth_image_sub],
            queue_size=20, # Adjusted queue size
            slop=0.15      # Adjusted slop
        )
        self.time_synchronizer.registerCallback(self.synchronized_images_callback)

        self.tree_detection_pub = self.create_publisher(String, '/detected_trees_status', 10)
        self.tree_marker_pub = self.create_publisher(Marker, '/tree_markers_viz', 10)
        self.debug_img_pub = self.create_publisher(Image, '/tree_detection/debug_image', 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.brown_hsv_lower = np.array([0, 29, 0])      # From your HSV tuning
        self.brown_hsv_upper = np.array([179, 255, 62])  # From your HSV tuning
        self.min_contour_area = 1000
        self.max_contour_area = 80000
        self.min_aspect_ratio = 1.2
        self.max_aspect_ratio = 10.0
        self.min_solidity = 0.70
        self.kernel_open_size = (5,5)
        self.kernel_close_size = (15,15) 
        self.morph_open_iterations = 1
        self.morph_close_iterations = 2
        self.depth_window_size = 5 

        self.total_detection_events = 0
        self.frame_count = 0 
        self.last_detection_log_time = self.get_clock().now()
        self.detected_trees_map_positions = []

        self.get_logger().info(f"{self.get_name()} Initialized!")
        self.get_logger().info("Attempting to get depth via image_transport from base topic of /oak/rgb/image_raw (expecting /compressedDepth).")
        self.get_logger().info("Waiting for camera_info and synchronized images...")

    def camera_info_callback(self, msg):
        if self.fx is None: 
            self.fx = msg.k[0]; self.fy = msg.k[4]
            self.cx = msg.k[2]; self.cy = msg.k[5]
            self.camera_optical_frame_id = msg.header.frame_id
            self.get_logger().info(f"RGB Camera intrinsics received from '{msg.header.frame_id}': fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}")
            # self.camera_info_sub.destroy() # Optional

    def get_distance_from_depth_pixel(self, x_pixel, y_pixel, depth_image):
        if depth_image is None: return None
        h, w = depth_image.shape[:2]
        x_pixel, y_pixel = int(round(x_pixel)), int(round(y_pixel))

        if not (0 <= y_pixel < h and 0 <= x_pixel < w):
            self.get_logger().warn(f"Pixel ({x_pixel},{y_pixel}) out of depth bounds ({w}x{h})", throttle_duration_sec=5)
            return None

        half_win = self.depth_window_size // 2
        y_start, y_end = max(0, y_pixel - half_win), min(h, y_pixel + half_win + 1)
        x_start, x_end = max(0, x_pixel - half_win), min(w, x_pixel + half_win + 1)
        depth_patch = depth_image[y_start:y_end, x_start:x_end]

        distance_meters = None
        if depth_image.dtype == np.uint16: # Typically mm
            valid_depths = depth_patch[depth_patch > 0] 
            if valid_depths.size > self.depth_window_size: # Require some valid points in window
                median_depth_mm = np.median(valid_depths)
                distance_meters = float(median_depth_mm) / 1000.0
        elif depth_image.dtype == np.float32: # Typically meters
            valid_depths = depth_patch[np.isfinite(depth_patch) & (depth_patch > 0.01)] # Add small min dist
            if valid_depths.size > self.depth_window_size: # Require some valid points
                distance_meters = float(np.median(valid_depths))
        else:
            self.get_logger().error(f"Unsupported depth image dtype: {depth_image.dtype}", throttle_duration_sec=10)
            return None

        MAX_TREE_DEPTH_DISTANCE = 7.0 # Increased slightly
        if distance_meters is not None and (distance_meters <= 0.01 or distance_meters > MAX_TREE_DEPTH_DISTANCE) :
            return None
        return distance_meters

    def project_pixel_to_3d_camera_frame(self, x_pixel, y_pixel, distance_m):
        if self.fx is None: return None
        if distance_m is None or distance_m <= 0: return None
        z_cam = distance_m
        x_cam = (x_pixel - self.cx) * z_cam / self.fx
        y_cam = (y_pixel - self.cy) * z_cam / self.fy
        return (x_cam, y_cam, z_cam)

    def synchronized_images_callback(self, color_msg, depth_msg_from_filter):
        self.frame_count += 1
        
        self.get_logger().debug(f"--- Frame {self.frame_count} Sync Callback ---")
        # Log topic anme only once or a few times to avoid spam
        if self.frame_count < 5 or self.frame_count % 100 == 0 :
            color_topic = color_msg._connection_header['topic'] if hasattr(color_msg, '_connection_header') and color_msg._connection_header else 'N/A'
            depth_topic = depth_msg_from_filter._connection_header['topic'] if hasattr(depth_msg_from_filter, '_connection_header') and depth_msg_from_filter._connection_header else 'N/A'
            self.get_logger().info(f"Color Topic (filter): {color_topic}, Encoding: {color_msg.encoding}")
            self.get_logger().info(f"Depth Topic (filter): {depth_topic}, Encoding: {depth_msg_from_filter.encoding}, Dtype via cv_bridge will show actual pixel type")
            self.get_logger().info(f"Depth Header Frame ID: {depth_msg_from_filter.header.frame_id}, Stamp Diff (color-depth): {(color_msg.header.stamp.sec - depth_msg_from_filter.header.stamp.sec) + (color_msg.header.stamp.nanosec - depth_msg_from_filter.header.stamp.nanosec)/1e9:.4f}s")

        if self.fx is None:
            self.get_logger().warn("Waiting for camera intrinsics...", throttle_duration_sec=5)
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            
            # Critical Check: Is depth_msg_from_filter actually depth data?
            # Common depth encodings are '16UC1' (mm), '32FC1' (m).
            # If it's 'bgr8' or 'rgb8', then image_transport gave us the color image again for depth.
            if "bgr" in depth_msg_from_filter.encoding.lower() or \
               "rgb" in depth_msg_from_filter.encoding.lower() or \
               depth_msg_from_filter.encoding == color_msg.encoding :
                self.get_logger().error(f"Depth message encoding '{depth_msg_from_filter.encoding}' is same as color or a color format! Check image_transport setup or OAK-D depth publishing. Subscribing to same base topic for color and depth might be problematic if not handled correctly by driver's image_transport plugins for depth variants.", throttle_duration_sec=10)
                # Publish color image as debug and return to avoid crashing on wrong depth data
                self.debug_img_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8'))
                return

            depth_image = self.depth_bridge.imgmsg_to_cv2(depth_msg_from_filter, desired_encoding="passthrough")
            if self.frame_count < 5 or self.frame_count % 100 == 0 :
                 self.get_logger().info(f"Depth CV image from bridge: shape={depth_image.shape}, dtype={depth_image.dtype}")


        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge error in sync_callback: {e}')
            return
        except Exception as e:
            self.get_logger().error(f'Generic error processing images in sync_callback: {type(e).__name__} - {e}')
            return

        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv_image, self.brown_hsv_lower, self.brown_hsv_upper)
        kernel_open = np.ones(self.kernel_open_size, np.uint8)
        mask_opened = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel_open, iterations=self.morph_open_iterations)
        kernel_close = np.ones(self.kernel_close_size, np.uint8)
        mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel_close, iterations=self.morph_close_iterations)
        contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_trees_in_frame = []
        if contours:
            for contour in contours:
                area = cv2.contourArea(contour)
                if not (self.min_contour_area < area < self.max_contour_area): continue
                x_bbox, y_bbox, w_bbox, h_bbox = cv2.boundingRect(contour)
                aspect_ratio = float(h_bbox) / w_bbox if w_bbox > 0 else 0
                if not (self.min_aspect_ratio < aspect_ratio < self.max_aspect_ratio): continue
                hull = cv2.convexHull(contour); hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0
                if solidity < self.min_solidity: continue

                center_x_pixel = x_bbox + w_bbox / 2.0
                center_y_pixel = y_bbox + h_bbox / 2.0
                direct_depth_distance = self.get_distance_from_depth_pixel(center_x_pixel, center_y_pixel, depth_image)

                if direct_depth_distance is not None:
                    detected_trees_in_frame.append({
                        'bbox': (x_bbox, y_bbox, w_bbox, h_bbox),
                        'center_pixel': (center_x_pixel, center_y_pixel),
                        'area': area, 'aspect_ratio': aspect_ratio, 'solidity': solidity,
                        'depth_distance': direct_depth_distance
                    })

        if detected_trees_in_frame:
            self.handle_trees_detected_with_depth(detected_trees_in_frame, color_msg.header.stamp)
        
        # Pass cv_image for drawing, detected_trees_in_frame for boxes, mask_closed for mask overlay
        debug_img_to_publish = self.create_debug_image(cv_image, detected_trees_in_frame, mask_closed) 
        try:
            self.debug_img_pub.publish(self.bridge.cv2_to_imgmsg(debug_img_to_publish, encoding='bgr8'))
        except CvBridgeError as e:
            self.get_logger().error(f"Error publishing debug image: {e}")
        except Exception as e_pub:
             self.get_logger().error(f"Unexpected error publishing debug image: {type(e_pub).__name__} - {e_pub}")


    def handle_trees_detected_with_depth(self, trees_data, image_stamp):
        num_detected = len(trees_data)
        status_msg = String(); status_msg.data = f"trees_detected:{num_detected}"
        self.tree_detection_pub.publish(status_msg)

        current_time = self.get_clock().now()
        # Log more frequently if trees are seen or if it's the first few detections
        if num_detected > 0 and ((current_time - self.last_detection_log_time).nanoseconds / 1e9 > 1.0 or self.total_detection_events < 5) :
            self.total_detection_events += 1
            log_msg_header = f'TREES DETECTED ({num_detected}) Evt#{self.total_detection_events}'
            self.get_logger().info(log_msg_header)
            self.last_detection_log_time = current_time

            for i, tree_info in enumerate(trees_data):
                px_x, px_y = tree_info['center_pixel']; depth_dist = tree_info['depth_distance']
                basic_info_str = (f"  T#{i+1} Px:({px_x:.0f},{px_y:.0f}) A:{tree_info['area']:.0f} "
                                  f"AR:{tree_info['aspect_ratio']:.1f} S:{tree_info['solidity']:.2f} D:{depth_dist:.2f}m")
                point_3d_cam = self.project_pixel_to_3d_camera_frame(px_x, px_y, depth_dist)

                if point_3d_cam and self.camera_optical_frame_id:
                    x_cam, y_cam, z_cam = point_3d_cam
                    point_cam_stamped = PointStamped()
                    point_cam_stamped.header.frame_id = self.camera_optical_frame_id
                    point_cam_stamped.header.stamp = image_stamp 
                    point_cam_stamped.point.x = x_cam; point_cam_stamped.point.y = y_cam; point_cam_stamped.point.z = z_cam
                    map_coords_tuple = self.transform_to_map_frame(point_cam_stamped)

                    if map_coords_tuple:
                        map_x, map_y = map_coords_tuple; map_str = f"({map_x:.2f},{map_y:.2f})"
                        self.get_logger().info(f"{basic_info_str}, CamXYZ:({x_cam:.2f},{y_cam:.2f},{z_cam:.2f}), MapXY:{map_str}")
                        if not self.is_duplicate_map_tree((map_x, map_y)):
                            self.publish_tree_marker_in_map(map_x, map_y, len(self.detected_trees_map_positions))
                            self.detected_trees_map_positions.append((map_x, map_y))
                    else:
                        self.get_logger().info(f"{basic_info_str}, CamXYZ:({x_cam:.2f},{y_cam:.2f},{z_cam:.2f}), MapXY:(TF Fail)")
                else:
                    self.get_logger().info(f"{basic_info_str}, CamXYZ:(Projection Fail or No Intrinsics)")

    def transform_to_map_frame(self, point_stamped_msg):
        try:
            timeout_duration = rclpy.duration.Duration(seconds=0.05) # Reduced timeout
            # Check if target frame exists and if source frame exists in TF buffer
            if not self.tf_buffer.can_transform('map', point_stamped_msg.header.frame_id, point_stamped_msg.header.stamp, timeout=timeout_duration):
                # self.get_logger().warn(f'TF: Cannot transform from {point_stamped_msg.header.frame_id} to map (can_transform=false). Buffer time: {self.tf_buffer.get_latest_common_time("map", point_stamped_msg.header.frame_id)}', throttle_duration_sec=5)
                return None
            
            transform_stamped = self.tf_buffer.lookup_transform(
                'map', point_stamped_msg.header.frame_id,
                point_stamped_msg.header.stamp, timeout=timeout_duration)
            transformed_point_stamped = tf2_geometry_msgs.do_transform_point(point_stamped_msg, transform_stamped)
            return (transformed_point_stamped.point.x, transformed_point_stamped.point.y)
        except tf2_ros.TransformException as e: # Catch specific TF exceptions
            # self.get_logger().warn(f'TF exception transforming {point_stamped_msg.header.frame_id} to map: {type(e).__name__} - {e}', throttle_duration_sec=5)
            return None
        except Exception as e_gen: # Catch any other unexpected errors
            self.get_logger().error(f'Unexpected error in transform_to_map_frame: {type(e_gen).__name__} - {e_gen}', throttle_duration_sec=5)
            return None


    def is_duplicate_map_tree(self, new_map_pos, threshold=0.5): # Same
        new_x, new_y = new_map_pos
        for old_x, old_y in self.detected_trees_map_positions:
            if math.hypot(new_x - old_x, new_y - old_y) < threshold: return True
        return False

    def publish_tree_marker_in_map(self, map_x, map_y, marker_id): # Same
        marker = Marker(); marker.header.frame_id = 'map'; marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'detected_tree_cylinders'; marker.id = marker_id; marker.type = Marker.CYLINDER
        marker.action = Marker.ADD; marker.pose.position.x = map_x; marker.pose.position.y = map_y
        marker.pose.position.z = 0.5; marker.pose.orientation.w = 1.0
        marker.scale.x = 0.20; marker.scale.y = 0.20; marker.scale.z = 1.0
        marker.color.r = 0.6; marker.color.g = 0.3; marker.color.b = 0.1; marker.color.a = 0.8
        self.tree_marker_pub.publish(marker)

    def create_debug_image(self, original_image, detected_trees_data, processed_mask):
        # This is the same full version from my previous response.
        debug_img = original_image.copy()
        if processed_mask is not None:
            mask_viz_bgr = cv2.cvtColor(processed_mask, cv2.COLOR_GRAY2BGR)
            debug_img = cv2.addWeighted(debug_img, 0.7, mask_viz_bgr, 0.3, 0)

        for i, tree_info in enumerate(detected_trees_data):
            x, y, w, h = tree_info['bbox']
            box_colors = [(0,255,0), (255,0,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]
            color = box_colors[i % len(box_colors)]
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 2)
            
            depth_dist_label = f"D:{tree_info['depth_distance']:.2f}m" if 'depth_distance' in tree_info else ""
            label_text = f"T{i+1} AR:{tree_info['aspect_ratio']:.1f} S:{tree_info['solidity']:.2f} {depth_dist_label}"
            cv2.putText(debug_img, label_text, (x, y - 7), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)
        
        cv2.putText(debug_img, f"F:{self.frame_count}",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(230,230,230),1,cv2.LINE_AA)
        status_text_color = (0,255,0) if detected_trees_data else (0,0,255)
        cv2.putText(debug_img, f"Trees:{len(detected_trees_data)}",(10,45),cv2.FONT_HERSHEY_SIMPLEX,0.6,status_text_color,1,cv2.LINE_AA)
        return debug_img

def main(args=None):
    rclpy.init(args=args)
    try:
        tree_detector = TreeDetectorNode()
        rclpy.spin(tree_detector)
    except KeyboardInterrupt:
        tree_detector.get_logger().info(f"{tree_detector.get_name()} shutting down...")
    finally:
        if 'tree_detector' in locals() and rclpy.ok():
            tree_detector.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
