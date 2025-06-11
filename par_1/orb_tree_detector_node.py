# File: par_1/orb_tree_detector.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory

# --- ORB Feature Matching Constants ---
# These can be turned into ROS Parameters for more flexibility
MIN_MATCH_COUNT_DEFAULT = 15
LOWES_RATIO_TEST_THRESHOLD_DEFAULT = 0.7
REFERENCE_IMAGE_FILENAME_DEFAULT = 'reference_cylinder.jpg' # Name of your reference image

class OrbTreeDetectorNode(Node):
    def __init__(self):
        super().__init__('orb_tree_detector_node')

        # Declare parameters
        self.declare_parameter('reference_image_filename', REFERENCE_IMAGE_FILENAME_DEFAULT)
        self.declare_parameter('min_match_count', MIN_MATCH_COUNT_DEFAULT)
        self.declare_parameter('lowes_ratio', LOWES_RATIO_TEST_THRESHOLD_DEFAULT)
        self.declare_parameter('camera_topic', '/oak/rgb/image_raw') # Default camera topic
        self.declare_parameter('publish_debug_image', True)

        # Get parameters
        self.reference_image_filename = self.get_parameter('reference_image_filename').get_parameter_value().string_value
        self.min_match_count = self.get_parameter('min_match_count').get_parameter_value().integer_value
        self.lowes_ratio = self.get_parameter('lowes_ratio').get_parameter_value().double_value
        camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self.publish_debug_image = self.get_parameter('publish_debug_image').get_parameter_value().bool_value

        self.bridge = CvBridge()

        # --- Load Reference Image and Precompute Features ---
        package_share_directory = get_package_share_directory('par_1')
        # Assume reference image is in a 'config' or 'resource' folder within your package's share dir
        # For example: <your_workspace>/install/par_1/share/par_1/config/reference_cylinder.jpg
        self.reference_image_path = os.path.join(package_share_directory, 'config', self.reference_image_filename)

        if not os.path.exists(self.reference_image_path):
            self.get_logger().error(f"Reference image not found at '{self.reference_image_path}'")
            self.get_logger().error("Make sure it's installed via data_files in setup.py (e.g., in a 'config' folder).")
            rclpy.shutdown()
            return

        self.ref_img_bgr = cv2.imread(self.reference_image_path)
        if self.ref_img_bgr is None:
            self.get_logger().error(f"Could not load reference image from '{self.reference_image_path}'")
            rclpy.shutdown()
            return
        self.ref_img_gray = cv2.cvtColor(self.ref_img_bgr, cv2.COLOR_BGR2GRAY)

        # Initialize ORB detector
        self.orb = cv2.ORB_create(nfeatures=2000, scoreType=cv2.ORB_FAST_SCORE)
        try:
            self.ref_kps, self.ref_des = self.orb.detectAndCompute(self.ref_img_gray, None)
            if self.ref_des is None or len(self.ref_kps) == 0:
                self.get_logger().error(f"No descriptors found in reference image '{self.reference_image_path}'. Is it too plain?")
                rclpy.shutdown()
                return
            self.get_logger().info(f"Found {len(self.ref_kps)} keypoints in reference image.")
        except cv2.error as e:
            self.get_logger().error(f"Error in ORB detectAndCompute for reference image: {e}")
            rclpy.shutdown()
            return

        # Initialize Brute-Force Matcher
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Subscribers and Publishers
        self.subscription = self.create_subscription(
            Image,
            camera_topic, # Use the parameter for the topic name
            self.image_callback,
            10)
        self.subscription  # prevent unused variable warning

        if self.publish_debug_image:
            self.debug_image_publisher = self.create_publisher(Image, 'orb_detections/debug_image', 10)
        
        # You might want to publish actual detection results (e.g., bounding box, status)
        # self.detection_publisher = self.create_publisher(YourCustomDetectionMsg, 'orb_detections/detections', 10)

        self.get_logger().info(f"ORB Tree Detector Node started. Listening on {camera_topic}.")
        self.get_logger().info(f"Using reference image: {self.reference_image_path}")
        self.get_logger().info(f"Min matches: {self.min_match_count}, Lowe's ratio: {self.lowes_ratio}")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert ROS Image to OpenCV: {e}")
            return

        processed_image, detected_outline_points = self._detect_object_orb(cv_image)

        if self.publish_debug_image:
            try:
                debug_msg = self.bridge.cv2_to_imgmsg(processed_image, encoding='bgr8')
                debug_msg.header = msg.header # Keep the original timestamp and frame_id
                self.debug_image_publisher.publish(debug_msg)
            except Exception as e:
                self.get_logger().error(f"Failed to convert OpenCV image to ROS Image for publishing: {e}")
        
        if detected_outline_points is not None:
            self.get_logger().info("Cylinder detected by ORB!")
            # Here you would publish actual detection data if you have a custom message
            # For example, calculate centroid, bounding box, etc. from detected_outline_points
            # and publish it on self.detection_publisher

    def _detect_object_orb(self, frame):
        """
        Internal method to perform ORB detection on a given frame.
        Returns the processed frame (with drawings) and detected outline points (or None).
        """
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_outline_points = None # To store [[x1,y1], [x2,y2]...] for the bounding polygon

        try:
            kps_frame, des_frame = self.orb.detectAndCompute(frame_gray, None)
        except cv2.error as e:
            self.get_logger().warn(f"ORB detectAndCompute error on frame: {e}")
            return frame, None

        if des_frame is None or len(des_frame) < 2 or self.ref_des is None:
            return frame, None

        try:
            matches = self.bf_matcher.knnMatch(self.ref_des, des_frame, k=2)
        except cv2.error as e:
             # This can happen if ref_des or des_frame have an unexpected format or are empty
            self.get_logger().warn(f"Error during BFMatcher.knnMatch: {e}")
            return frame, None


        good_matches = []
        if matches:
            for m_pair in matches:
                if len(m_pair) == 2:
                    m, n = m_pair
                    if m.distance < self.lowes_ratio * n.distance:
                        good_matches.append(m)
        
        frame_with_detection = frame.copy() # Work on a copy for drawing

        if len(good_matches) >= self.min_match_count:
            src_pts = np.float32([self.ref_kps[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kps_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            try:
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if M is not None:
                    matchesMask = mask.ravel().tolist()
                    h, w = self.ref_img_gray.shape
                    pts_ref_corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                    dst_corners = cv2.perspectiveTransform(pts_ref_corners, M)
                    
                    detected_outline_points = np.int32(dst_corners) # Store these points
                    cv2.polylines(frame_with_detection, [detected_outline_points], True, (0, 255, 0), 3, cv2.LINE_AA)
                    cv2.putText(frame_with_detection, "Cylinder (ORB)", (detected_outline_points[0][0][0], detected_outline_points[0][0][1] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                    # Option to draw good matches (can be noisy for published image)
                    # frame_with_detection = cv2.drawMatches(self.ref_img_gray, self.ref_kps, frame_with_detection, kps_frame,
                    #                                       good_matches, None, matchesMask=matchesMask,
                    #                                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                else:
                    matchesMask = None # Homography not found
            except cv2.error as e:
                self.get_logger().warn(f"Error in findHomography: {e}")
                matchesMask = None
        else:
            matchesMask = None
            # self.get_logger().debug(f"Not enough good matches: {len(good_matches)}/{self.min_match_count}")

        return frame_with_detection, detected_outline_points


def main(args=None):
    rclpy.init(args=args)
    orb_tree_detector_node = OrbTreeDetectorNode()
    try:
        rclpy.spin(orb_tree_detector_node)
    except KeyboardInterrupt:
        pass
    finally:
        orb_tree_detector_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
