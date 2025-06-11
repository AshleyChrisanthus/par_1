# File: par_1/orb_tree_detector_node.py # (or your actual filename)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory
from glob import glob # For finding reference image files

# --- ORB Feature Matching Constants ---
MIN_MATCH_COUNT_DEFAULT = 15
LOWES_RATIO_TEST_THRESHOLD_DEFAULT = 0.7 # Start a bit stricter
# REFERENCE_IMAGE_FILENAME_PATTERN_DEFAULT = 'reference_cylinder_view*.jpg' # Pattern for multiple images
REFERENCE_IMAGE_FOLDER_DEFAULT = 'config' # Subfolder in share directory

class OrbTreeDetectorNode(Node):
    def __init__(self):
        super().__init__('orb_tree_detector_node')

        # Declare parameters
        # self.declare_parameter('reference_image_filename_pattern', REFERENCE_IMAGE_FILENAME_PATTERN_DEFAULT)
        self.declare_parameter('reference_image_folder', REFERENCE_IMAGE_FOLDER_DEFAULT)
        self.declare_parameter('reference_image_prefix', 'tree_ref_image') # e.g. 'reference_cylinder_view' for 'reference_cylinder_view1.jpg'

        self.declare_parameter('min_match_count', MIN_MATCH_COUNT_DEFAULT)
        self.declare_parameter('lowes_ratio', LOWES_RATIO_TEST_THRESHOLD_DEFAULT)
        self.declare_parameter('camera_topic', '/oak/rgb/image_raw')
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('orb_nfeatures', 2000)
        self.declare_parameter('orb_score_type', 'FAST') # Options: "FAST", "HARRIS"
        self.declare_parameter('clahe_clip_limit', 2.0)
        self.declare_parameter('clahe_tile_grid_size', 8)


        # Get parameters
        # reference_image_pattern = self.get_parameter('reference_image_filename_pattern').get_parameter_value().string_value
        reference_image_folder = self.get_parameter('reference_image_folder').get_parameter_value().string_value
        self.reference_image_prefix = self.get_parameter('reference_image_prefix').get_parameter_value().string_value

        self.min_match_count = self.get_parameter('min_match_count').get_parameter_value().integer_value
        self.lowes_ratio = self.get_parameter('lowes_ratio').get_parameter_value().double_value
        camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self.publish_debug_image = self.get_parameter('publish_debug_image').get_parameter_value().bool_value
        orb_nfeatures = self.get_parameter('orb_nfeatures').get_parameter_value().integer_value
        orb_score_type_str = self.get_parameter('orb_score_type').get_parameter_value().string_value
        clahe_clip_limit = self.get_parameter('clahe_clip_limit').get_parameter_value().double_value
        clahe_tile_grid_size = self.get_parameter('clahe_tile_grid_size').get_parameter_value().integer_value


        self.bridge = CvBridge()
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(clahe_tile_grid_size, clahe_tile_grid_size))

        # --- Load Reference Images and Precompute Features ---
        self.reference_data = [] # List to hold (path, gray_img, kps, des) for each ref image
        package_share_directory = get_package_share_directory('par_1')
        config_path = os.path.join(package_share_directory, reference_image_folder)

        # Find reference images matching the pattern
        # search_pattern = os.path.join(config_path, reference_image_pattern)
        # reference_image_paths = glob(search_pattern)
        reference_image_paths = sorted(glob(os.path.join(config_path, f"{self.reference_image_prefix}*.jpg"))) # More specific

        if not reference_image_paths:
            self.get_logger().error(f"No reference images found in '{config_path}' matching prefix '{self.reference_image_prefix}*.jpg'")
            self.get_logger().error("Make sure they are installed via data_files in setup.py and filenames match.")
            rclpy.shutdown()
            return

        # Initialize ORB detector
        orb_score_type = cv2.ORB_FAST_SCORE if orb_score_type_str.upper() == "FAST" else cv2.ORB_HARRIS_SCORE
        self.orb = cv2.ORB_create(nfeatures=orb_nfeatures, scoreType=orb_score_type)

        for ref_path in reference_image_paths:
            ref_img_bgr = cv2.imread(ref_path)
            if ref_img_bgr is None:
                self.get_logger().warn(f"Could not load reference image from '{ref_path}'. Skipping.")
                continue
            
            ref_img_gray = cv2.cvtColor(ref_img_bgr, cv2.COLOR_BGR2GRAY)
            ref_img_gray_enhanced = self.clahe.apply(ref_img_gray) # Apply CLAHE

            try:
                ref_kps, ref_des = self.orb.detectAndCompute(ref_img_gray_enhanced, None)
                if ref_des is None or len(ref_kps) == 0:
                    self.get_logger().warn(f"No descriptors found in reference image '{ref_path}'. Skipping.")
                    continue
                self.reference_data.append({
                    'path': ref_path,
                    'gray_img': ref_img_gray_enhanced, # Store enhanced for potential drawing
                    'bgr_img': ref_img_bgr, # Store original bgr for drawing matches
                    'kps': ref_kps,
                    'des': ref_des,
                    'h': ref_img_gray_enhanced.shape[0],
                    'w': ref_img_gray_enhanced.shape[1]
                })
                self.get_logger().info(f"Loaded reference '{os.path.basename(ref_path)}' with {len(ref_kps)} keypoints.")
            except cv2.error as e:
                self.get_logger().error(f"Error in ORB detectAndCompute for reference image '{ref_path}': {e}. Skipping.")
        
        if not self.reference_data:
            self.get_logger().error("No valid reference images were loaded. Shutting down.")
            rclpy.shutdown()
            return

        # Initialize Brute-Force Matcher
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Subscribers and Publishers
        self.subscription = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            10)
        
        if self.publish_debug_image:
            self.debug_image_publisher = self.create_publisher(Image, 'orb_detections/debug_image', 10)
        
        self.get_logger().info(f"ORB Tree Detector Node started. Listening on {camera_topic}.")
        self.get_logger().info(f"Loaded {len(self.reference_data)} reference images.")
        self.get_logger().info(f"Min matches: {self.min_match_count}, Lowe's ratio: {self.lowes_ratio}")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert ROS Image to OpenCV: {e}")
            return

        frame_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        frame_gray_enhanced = self.clahe.apply(frame_gray) # Apply CLAHE to current frame

        best_match_info = {
            'score': -1, # e.g., number of inliers, or just a flag
            'outline_points': None,
            'processed_image': cv_image.copy(), # Start with a copy of original for drawing
            'ref_idx': -1,
            'homography_matrix': None
        }

        try:
            kps_frame, des_frame = self.orb.detectAndCompute(frame_gray_enhanced, None)
            if des_frame is None or len(des_frame) < 2: # Need at least 2 for knnMatch
                if self.publish_debug_image:
                    self._publish_debug_image(cv_image, msg.header) # Publish original if no features in frame
                return
        except cv2.error as e:
            self.get_logger().warn(f"ORB detectAndCompute error on frame: {e}")
            if self.publish_debug_image:
                self._publish_debug_image(cv_image, msg.header)
            return


        for idx, ref_data_item in enumerate(self.reference_data):
            ref_des = ref_data_item['des']
            ref_kps = ref_data_item['kps']
            ref_h = ref_data_item['h']
            ref_w = ref_data_item['w']

            if ref_des is None: # Should have been caught at init, but double check
                continue

            try:
                matches = self.bf_matcher.knnMatch(ref_des, des_frame, k=2)
            except cv2.error as e:
                self.get_logger().warn(f"Error during BFMatcher.knnMatch for ref '{os.path.basename(ref_data_item['path'])}': {e}")
                continue
            
            good_matches = []
            if matches: # Ensure matches is not None or empty
                for m_pair in matches:
                    if len(m_pair) == 2: # Ensure knnMatch returned two neighbors
                        m, n = m_pair
                        if m.distance < self.lowes_ratio * n.distance:
                            good_matches.append(m)
            
            if len(good_matches) >= self.min_match_count:
                src_pts = np.float32([ref_kps[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kps_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                try:
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if M is not None:
                        matchesMask = mask.ravel().tolist()
                        num_inliers = np.sum(matchesMask)

                        # --- Prioritize match with more inliers ---
                        if num_inliers > best_match_info['score']:
                            pts_ref_corners = np.float32([[0, 0], [0, ref_h - 1], [ref_w - 1, ref_h - 1], [ref_w - 1, 0]]).reshape(-1, 1, 2)
                            dst_corners = cv2.perspectiveTransform(pts_ref_corners, M)
                            detected_outline_points = np.int32(dst_corners)

                            # --- Add Sanity Checks for the detected polygon (Aspect Ratio, Area, Convexity) ---
                            brect = cv2.boundingRect(detected_outline_points)
                            _, _, w_box, h_box = brect
                            aspect_ratio = w_box / float(h_box) if h_box > 0 else 0
                            area = cv2.contourArea(detected_outline_points)

                            MIN_AREA = 500  # Tune these!
                            MAX_AREA = 100000 # Increased for potentially larger detections
                            MIN_ASPECT_RATIO = 0.1 # More tolerant
                            MAX_ASPECT_RATIO = 5.0 # More tolerant

                            is_valid_shape = (MIN_AREA < area < MAX_AREA and \
                                            MIN_ASPECT_RATIO < aspect_ratio < MAX_ASPECT_RATIO and \
                                            cv2.isContourConvex(detected_outline_points))

                            if is_valid_shape:
                                best_match_info['score'] = num_inliers
                                best_match_info['outline_points'] = detected_outline_points
                                best_match_info['ref_idx'] = idx
                                best_match_info['homography_matrix'] = M
                                best_match_info['good_matches_for_best'] = good_matches # For drawing later
                                best_match_info['kps_frame_for_best'] = kps_frame
                                best_match_info['matchesMask_for_best'] = matchesMask


                except cv2.error as e:
                    self.get_logger().warn(f"Error in findHomography for ref '{os.path.basename(ref_data_item['path'])}': {e}")
                    # continue to next reference image

        # --- After checking all reference images, process the best match ---
        output_image_for_debug = cv_image.copy() # Start fresh with original camera image
        if best_match_info['outline_points'] is not None:
            self.get_logger().info(f"Cylinder detected with reference '{os.path.basename(self.reference_data[best_match_info['ref_idx']]['path'])}' ({best_match_info['score']} inliers).")
            
            # Draw on the output_image_for_debug
            cv2.polylines(output_image_for_debug, [best_match_info['outline_points']], True, (0, 255, 0), 3, cv2.LINE_AA)
            text_pos_x = best_match_info['outline_points'][0][0][0]
            text_pos_y = best_match_info['outline_points'][0][0][1] - 10
            cv2.putText(output_image_for_debug, f"Cylinder (ORB Ref {best_match_info['ref_idx']+1})", (text_pos_x, text_pos_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
            # Optionally, draw the matches for the best one (can be noisy)
            # best_ref_data = self.reference_data[best_match_info['ref_idx']]
            # output_image_for_debug = cv2.drawMatches(best_ref_data['bgr_img'], best_ref_data['kps'],
            #                                       output_image_for_debug, best_match_info['kps_frame_for_best'],
            #                                       best_match_info['good_matches_for_best'], None,
            #                                       matchesMask=best_match_info['matchesMask_for_best'],
            #                                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


            # Here you would publish actual detection data if you have a custom message
            # e.g., publish best_match_info['outline_points'], or its centroid, etc.
        else:
            if kps_frame is not None : # Only log if we actually processed frame features
                 self.get_logger().debug("No robust cylinder detection across all reference images.")


        if self.publish_debug_image:
            self._publish_debug_image(output_image_for_debug, msg.header)

    def _publish_debug_image(self, image_to_publish, header):
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(image_to_publish, encoding='bgr8')
            debug_msg.header = header # Keep the original timestamp and frame_id
            self.debug_image_publisher.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to convert/publish debug image: {e}")


def main(args=None):
    rclpy.init(args=args)
    orb_tree_detector_node = OrbTreeDetectorNode()
    if rclpy.ok(): # Check if init was successful before spinning
        try:
            rclpy.spin(orb_tree_detector_node)
        except KeyboardInterrupt:
            orb_tree_detector_node.get_logger().info('Keyboard interrupt, shutting down.')
        except Exception as e:
            orb_tree_detector_node.get_logger().error(f"Unhandled exception in spin: {e}")
        finally:
            if orb_tree_detector_node and rclpy.ok() and orb_tree_detector_node.handle: # Check if node is valid
                 orb_tree_detector_node.destroy_node()
            if rclpy.ok():
                 rclpy.shutdown()

if __name__ == '__main__':
    main()
