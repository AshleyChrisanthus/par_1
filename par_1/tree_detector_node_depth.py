#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import tf2_ros
import tf2_geometry_msgs 
import math

from message_filters import Subscriber, ApproximateTimeSynchronizer

class TreeDetectorNode(Node):
    def __init__(self):
        super().__init__('tree_detector_node_depth') # New node name for this test

        self.bridge = CvBridge()
        self.depth_bridge = CvBridge()

        self.fx = None; self.fy = None; self.cx = None; self.cy = None
        # This will store the optical frame ID of the RGB camera, used for projection
        self.rgb_camera_optical_frame_id = "rgb_camera_optical_frame" # Default, will be updated

        # --- Subscribers using message_filters ---
        self.color_image_sub = Subscriber(self, Image, '/oak/rgb/image_raw') # For color detection
        
        # ATTEMPT 2: Try to get depth from the stereo camera's depth output
        # We subscribe to its base name, hoping image_transport uses '/oak/stereo/image_raw/compressedDepth'
        self.depth_image_sub = Subscriber(self, Image, '/oak/stereo/image_raw') 

        # CameraInfo for the RGB camera (since detections are on RGB image)
        self.rgb_camera_info_sub = self.create_subscription(
            CameraInfo,
            '/oak/rgb/camera_info', 
            self.rgb_camera_info_callback, 
            rclpy.qos.QoSProfile(depth=1, reliability=rclpy.qos.ReliabilityPolicy.RELIABLE, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL)
        )

        self.time_synchronizer = ApproximateTimeSynchronizer(
            [self.color_image_sub, self.depth_image_sub],
            queue_size=20, 
            slop=0.2 # Increased slop a bit more, can be tuned down later
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
        self.max_contour_area = 80000 # Increased from 50k
        self.min_aspect_ratio = 1.2   # Relaxed from 1.5
        self.max_aspect_ratio = 10.0  # Increased from 8.0
        self.min_solidity = 0.65      # Relaxed from 0.70 or 0.80

        self.kernel_open_size = (5,5)
        self.kernel_close_size = (15,15) # Start larger
        self.morph_open_iterations = 1
        self.morph_close_iterations = 3 # Increased iterations for closing

        self.depth_window_size = 5 
        self.MAX_TREE_DEPTH_DISTANCE = 6.0 # meters, trees further than this are ignored by depth

        self.total_detection_events = 0
        self.frame_count = 0 
        self.last_detection_log_time = self.get_clock().now()
        self.detected_trees_map_positions = []

        self.get_logger().info(f"{self.get_name()} Initialized!")
        self.get_logger().info("Attempting to get depth via image_transport from /oak/stereo/image_raw (expecting /compressedDepth).")
        self.get_logger().info("Waiting for RGB camera_info and synchronized images...")

    def rgb_camera_info_callback(self, msg): # For RGB camera's intrinsics
        if self.fx is None: 
            self.fx = msg.k[0]; self.fy = msg.k[4]
            self.cx = msg.k[2]; self.cy = msg.k[5]
            self.rgb_camera_optical_frame_id = msg.header.frame_id # Store RGB optical frame
            self.get_logger().info(f"RGB Camera intrinsics received from '{self.rgb_camera_optical_frame_id}': fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}")

    def get_distance_from_depth_pixel(self, x_pixel, y_pixel, depth_image):
        if depth_image is None: return None
        h_depth, w_depth = depth_image.shape[:2]
        x_pixel_int, y_pixel_int = int(round(x_pixel)), int(round(y_pixel))

        # IMPORTANT: If depth_image has different resolution than color_image,
        # x_pixel, y_pixel (from color image) need to be scaled to depth_image coordinates.
        # This assumes for now they are roughly aligned or same resolution.
        # If not, you'd do:
        # x_depth_px = int(round(x_pixel * (w_depth / w_color_image)))
        # y_depth_px = int(round(y_pixel * (h_depth / h_color_image)))
        # For now, using x_pixel_int, y_pixel_int directly and checking bounds.

        if not (0 <= y_pixel_int < h_depth and 0 <= x_pixel_int < w_depth):
            self.get_logger().warn(f"Pixel ({x_pixel_int},{y_pixel_int}) for depth lookup out of depth image bounds ({w_depth}x{h_depth}). Original color px: ({x_pixel:.1f},{y_pixel:.1f})", throttle_duration_sec=5)
            return None

        half_win = self.depth_window_size // 2
        y_start, y_end = max(0, y_pixel_int - half_win), min(h_depth, y_pixel_int + half_win + 1)
        x_start, x_end = max(0, x_pixel_int - half_win), min(w_depth, x_pixel_int + half_win + 1)
        depth_patch = depth_image[y_start:y_end, x_start:x_end]

        distance_meters = None
        if not depth_patch.size > 0: # Empty patch
            return None

        if depth_image.dtype == np.uint16: # Typically mm
            valid_depths = depth_patch[depth_patch > 0] 
            if valid_depths.size > self.depth_window_size // 2 : # Require at least a few valid points
                median_depth_mm = np.median(valid_depths)
                distance_meters = float(median_depth_mm) / 1000.0
        elif depth_image.dtype == np.float32: # Typically meters
            valid_depths = depth_patch[np.isfinite(depth_patch) & (depth_patch > 0.01)]
            if valid_depths.size > self.depth_window_size // 2:
                distance_meters = float(np.median(valid_depths))
        else:
            self.get_logger().error(f"Unsupported depth image dtype: {depth_image.dtype}", throttle_duration_sec=10)
            return None

        if distance_meters is not None and (distance_meters <= 0.01 or distance_meters > self.MAX_TREE_DEPTH_DISTANCE) :
            # self.get_logger().debug(f"Depth distance {distance_meters:.2f}m out of range or too far.", throttle_duration_sec=2)
            return None
        return distance_meters

    def project_pixel_to_3d_camera_frame(self, x_pixel, y_pixel, distance_m):
        if self.fx is None: return None # Using RGB camera intrinsics (self.fx, self.cx etc.)
        if distance_m is None or distance_m <= 0: return None
        z_cam = distance_m
        x_cam = (x_pixel - self.cx) * z_cam / self.fx
        y_cam = (y_pixel - self.cy) * z_cam / self.fy
        return (x_cam, y_cam, z_cam)

    def synchronized_images_callback(self, color_msg_from_filter, depth_msg_from_filter):
        self.frame_count += 1
        
        if self.frame_count < 5 or self.frame_count % 60 == 0 : # Log less frequently after startup
            self.get_logger().info(f"--- Frame {self.frame_count} Sync Callback ---")
            color_topic = color_msg_from_filter._connection_header['topic'] if hasattr(color_msg_from_filter, '_connection_header') and color_msg_from_filter._connection_header else 'N/A'
            depth_topic = depth_msg_from_filter._connection_header['topic'] if hasattr(depth_msg_from_filter, '_connection_header') and depth_msg_from_filter._connection_header else 'N/A'
            self.get_logger().info(f"Color Topic (filter): {color_topic}, Encoding: {color_msg_from_filter.encoding}")
            self.get_logger().info(f"Depth Topic (filter): {depth_topic}, Encoding: {depth_msg_from_filter.encoding}") # This is encoding from sensor_msgs/Image
            self.get_logger().info(f"Depth Header Frame ID: {depth_msg_from_filter.header.frame_id}, Stamp Diff (color-depth): {(color_msg_from_filter.header.stamp.sec - depth_msg_from_filter.header.stamp.sec) + (color_msg_from_filter.header.stamp.nanosec - depth_msg_from_filter.header.stamp.nanosec)/1e9:.4f}s")

        if self.fx is None: # Using RGB camera intrinsics
            self.get_logger().warn("Waiting for RGB camera intrinsics...", throttle_duration_sec=5)
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(color_msg_from_filter, desired_encoding='bgr8')
            
            # CRITICAL CHECK for depth message encoding after image_transport processing
            if "bgr" in depth_msg_from_filter.encoding.lower() or \
               "rgb" in depth_msg_from_filter.encoding.lower() or \
               "yuv" in depth_msg_from_filter.encoding.lower() or \
               "mono8" == depth_msg_from_filter.encoding.lower() and "depth" not in depth_topic.lower() : # mono8 could be intensity or misidentified depth
                self.get_logger().error(f"Depth message encoding '{depth_msg_from_filter.encoding}' from topic '{depth_topic}' "
                                        f"is a color/intensity format! Expecting '16UC1' or '32FC1' from image_transport. "
                                        f"Check OAK-D depth publishing and image_transport plugins.", throttle_duration_sec=10)
                self.debug_img_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')) # Publish color as debug
                return

            depth_image_cv = self.depth_bridge.imgmsg_to_cv2(depth_msg_from_filter, desired_encoding="passthrough")
            if self.frame_count < 5 or self.frame_count % 60 == 0 :
                 self.get_logger().info(f"Depth CV image from bridge: shape={depth_image_cv.shape}, dtype={depth_image_cv.dtype}")
                 # Log a sample depth value (e.g., center pixel)
                 h_d, w_d = depth_image_cv.shape[:2]
                 center_depth_val = depth_image_cv[h_d//2, w_d//2]
                 self.get_logger().info(f"Sample center depth value: {center_depth_val} (type: {depth_image_cv.dtype})")


        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge error in sync_callback: {e}')
            return
        except Exception as e_main:
            self.get_logger().error(f'Generic error processing images in sync_callback: {type(e_main).__name__} - {e_main}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            return

        # --- Visual Detection Logic ---
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
                # Additional check: ignore very wide or very short bounding boxes that are unlikely to be trees
                if w_bbox > h_bbox * 2 or h_bbox > w_bbox * 12 : continue # Filter out extreme aspect ratios early
                
                aspect_ratio = float(h_bbox) / w_bbox if w_bbox > 0 else 0
                if not (self.min_aspect_ratio < aspect_ratio < self.max_aspect_ratio): continue
                
                hull = cv2.convexHull(contour); hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0
                if solidity < self.min_solidity: continue

                center_x_pixel = x_bbox + w_bbox / 2.0
                center_y_pixel = y_bbox + h_bbox / 2.0
                
                # Pass the cv_image dimensions for potential scaling if depth image has different resolution
                # For now, get_distance_from_depth_pixel handles this internally if logic is added
                direct_depth_distance = self.get_distance_from_depth_pixel(center_x_pixel, center_y_pixel, depth_image_cv)


                if direct_depth_distance is not None:
                    detected_trees_in_frame.append({
                        'bbox': (x_bbox, y_bbox, w_bbox, h_bbox),
                        'center_pixel': (center_x_pixel, center_y_pixel),
                        'area': area, 'aspect_ratio': aspect_ratio, 'solidity': solidity,
                        'depth_distance': direct_depth_distance
                    })

        if detected_trees_in_frame:
            self.handle_trees_detected_with_depth(detected_trees_in_frame, color_msg_from_filter.header.stamp) # Pass color_msg stamp
        
        debug_img_to_publish = self.create_debug_image(cv_image, detected_trees_in_frame, mask_closed) 
        try:
            self.debug_img_pub.publish(self.bridge.cv2_to_imgmsg(debug_img_to_publish, encoding='bgr8'))
        except CvBridgeError as e_bridge_debug:
             self.get_logger().error(f"CvBridge error publishing debug image: {e_bridge_debug}")
        except Exception as e_pub:
             self.get_logger().error(f"Unexpected error publishing debug image: {type(e_pub).__name__} - {e_pub}")


    def handle_trees_detected_with_depth(self, trees_data, image_stamp):
        # (This function is largely the same as the previous version, ensuring it uses
        # self.rgb_camera_optical_frame_id when creating PointStamped for TF)
        num_detected = len(trees_data)
        status_msg = String(); status_msg.data = f"trees_detected:{num_detected}"
        self.tree_detection_pub.publish(status_msg)

        current_time = self.get_clock().now()
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

                if point_3d_cam and self.rgb_camera_optical_frame_id: # Ensure we use RGB optical frame ID
                    x_cam, y_cam, z_cam = point_3d_cam
                    point_cam_stamped = PointStamped()
                    point_cam_stamped.header.frame_id = self.rgb_camera_optical_frame_id # Use RGB frame
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
                    self.get_logger().info(f"{basic_info_str}, CamXYZ:(Projection Fail or No RGB Intrinsics/Frame ID)")

    def transform_to_map_frame(self, point_stamped_msg): # Mostly same
        try:
            timeout_duration = rclpy.duration.Duration(seconds=0.05) 
            if not self.tf_buffer.can_transform('map', point_stamped_msg.header.frame_id, point_stamped_msg.header.stamp, timeout=timeout_duration):
                self.get_logger().warn(f'TF: Cannot transform from {point_stamped_msg.header.frame_id} to map (can_transform=false).', throttle_duration_sec=5)
                return None
            
            transform_stamped = self.tf_buffer.lookup_transform('map', point_stamped_msg.header.frame_id, point_stamped_msg.header.stamp, timeout=timeout_duration)
            transformed_point_stamped = tf2_geometry_msgs.do_transform_point(point_stamped_msg, transform_stamped)
            return (transformed_point_stamped.point.x, transformed_point_stamped.point.y)
        except tf2_ros.TransformException as e: 
            self.get_logger().warn(f'TF exception transforming {point_stamped_msg.header.frame_id} to map: {type(e).__name__} - {e}', throttle_duration_sec=5)
            return None
        except Exception as e_gen: 
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
        marker.pose.position.z = 0.5; marker.pose.orientation.w = 1.0 # Make z=0.5 for cylinder center if height is 1.0
        marker.scale.x = 0.20; marker.scale.y = 0.20; marker.scale.z = 1.0 # Example size
        marker.color.r = 0.6; marker.color.g = 0.3; marker.color.b = 0.1; marker.color.a = 0.8
        self.tree_marker_pub.publish(marker)

    def create_debug_image(self, original_image, detected_trees_data, processed_mask): # Same as previous full example
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
        tree_detector.get_logger().info(f"{tree_detector.get_name()} shutting down...") # Use get_name()
    finally:
        if 'tree_detector' in locals() and rclpy.ok():
            tree_detector.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
