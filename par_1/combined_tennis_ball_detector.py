#!/usr/bin/env python3
import rclpy
import rclpy.time
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.duration import Duration

from visualization_msgs.msg import Marker
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from geometry_msgs.msg import PointStamped, Point, Pose, Vector3
from std_msgs.msg import ColorRGBA, Empty
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import time

import tf2_ros
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs

# =====================================================================================
# --- TUNING PARAMETERS ---
# =====================================================================================

# --- Ball Detection Parameters ---
GREEN_LOWER = np.array([25, 100, 100])  # Green tennis balls
GREEN_UPPER = np.array([40, 255, 255])
MIN_BALL_AREA = 100

# --- NEW: Parameters for robust mapping ---
ASSOCIATION_THRESHOLD_METERS = 0.3  # Distance threshold for considering balls as the same
# 1. Time to wait after startup before adding permanent markers
INITIALIZATION_DELAY_SEC = 5.0
# 2. Valid depth range for a detection to be considered
MIN_DETECTION_RANGE_M = 0.3
MAX_DETECTION_RANGE_M = 3.0

# =====================================================================================

class CombinedTennisBallDetector(Node):
    def __init__(self):
        super().__init__('combined_tennis_ball_detector')

        # Robustly handle sim time parameter
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', True)
        sim_time_is_enabled = self.get_parameter('use_sim_time').get_parameter_value().bool_value
        log_msg = 'Node is using simulation time.' if sim_time_is_enabled else 'Node is using system (wall) time.'
        self.get_logger().info(log_msg)

        # NEW: Record the startup time
        self.start_time = self.get_clock().now()
        
        self.queue_size = 30
        self.bridge = CvBridge()

        # Cached data
        self.latest_depth_image = None
        self.camera_intrinsics = None
        self.intrinsics_received = False
        
        # Persistent ball tracking
        self.known_balls = []
        self.next_marker_id = 0
        self.detection_count = 0
        self.frame_count = 0

        # TF setup (using the robust setup from horizontal_cylinder_detector)
        self.get_logger().info('Setting TF buffer duration to 10.0 seconds.')
        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.source_frame = 'oak_camera_rgb_camera_optical_frame'
        self.target_frame = 'map'
        self.get_logger().info(f'Source frame: {self.source_frame}, Target frame: {self.target_frame}')

        # Subscribers
        self.image_sub = self.create_subscription(
            CompressedImage, '/oak/rgb/image_rect/compressed', self.image_callback, self.queue_size)
        self.depth_sub = self.create_subscription(
            Image, '/oak/stereo/image_raw', self.depth_callback, self.queue_size)
        self.info_sub = self.create_subscription(
            CameraInfo, '/oak/rgb/camera_info', self.info_callback, self.queue_size)
        
        # Publishers
        self.marker_publisher = self.create_publisher(Marker, '~/permanent_tennis_balls', self.queue_size)
        self.home_trigger_pub = self.create_publisher(Empty, '/trigger_home', 10)
        self.debug_image_pub = self.create_publisher(Image, '~/debug_image', 10)
        
        self.get_logger().info('Combined Tennis Ball Detector (Persistent Mapping) has started.')
        self.get_logger().info('Looking for GREEN tennis balls with persistent world mapping.')

    def depth_callback(self, msg):
        try: 
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e: 
            self.get_logger().error(f'Failed to convert depth image: {e}')

    def info_callback(self, msg):
        if not self.intrinsics_received:
            self.camera_intrinsics = msg
            self.intrinsics_received = True
            self.destroy_subscription(self.info_sub)
            self.get_logger().info('Camera intrinsics received.')

    def image_callback(self, msg):
        if self.latest_depth_image is None or not self.intrinsics_received: 
            return
        
        try: 
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e: 
            self.get_logger().error(f'Failed to convert color image: {e}')
            return

        self.frame_count += 1
        
        # Check if image and depth dimensions match
        if cv_image.shape[:2] != self.latest_depth_image.shape[:2]:
            return

        # Detect all green tennis balls using the table_tennis_detector logic
        detected_balls = self.detect_all_green_tennis_balls(cv_image)
        if not detected_balls:
            return

        # Get 3D positions for all detected balls
        all_ball_data = []
        for ball_dict in detected_balls:
            ball_data_with_depth = self.get_position_in_camera_frame(ball_dict['contour'], msg.header.stamp)
            if ball_data_with_depth:
                all_ball_data.append(ball_data_with_depth)
        
        if not all_ball_data:
            return

        # Process each detected ball for persistent mapping
        for ball_data in all_ball_data:
            point_in_camera = ball_data['point_camera']
            depth_m = point_in_camera.point.z

            # NEW: Enforce valid detection range (from horizontal_cylinder_detector logic)
            if not (MIN_DETECTION_RANGE_M <= depth_m <= MAX_DETECTION_RANGE_M):
                continue  # Skip this ball, it's outside our desired range

            # Transform to map frame using the robust transformation logic
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.target_frame, 
                    self.source_frame, 
                    rclpy.time.Time()
                )
                newly_detected_point = tf2_geometry_msgs.do_transform_point(point_in_camera, transform).point
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                self.get_logger().warn(f'Could not transform new detection: {e}', throttle_duration_sec=1.0)
                continue

            # Associate with known balls using the robust mapping logic
            is_new_ball = True
            for known_point in self.known_balls:
                dx, dy = newly_detected_point.x - known_point.x, newly_detected_point.y - known_point.y
                distance = math.sqrt(dx*dx + dy*dy)
                if distance < ASSOCIATION_THRESHOLD_METERS:
                    is_new_ball = False
                    break

            if is_new_ball:
                # NEW: Check if the initial stabilization period has passed
                elapsed_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
                if elapsed_time > INITIALIZATION_DELAY_SEC:
                    self.detection_count += 1
                    self.get_logger().info(f"*** DISCOVERED A NEW TENNIS BALL #{self.detection_count}! Total count: {len(self.known_balls) + 1} ***")
                    self.get_logger().info(f'Ball position in map: x={newly_detected_point.x:.2f}, y={newly_detected_point.y:.2f}, z={newly_detected_point.z:.2f}')
                    self.known_balls.append(newly_detected_point)
                    self.publish_permanent_marker(newly_detected_point, self.next_marker_id)
                    self.next_marker_id += 1
                else:
                    self.get_logger().warn("Ignoring new ball during initial stabilization period.", throttle_duration_sec=1.0)

        # Publish debug image showing the nearest ball
        if all_ball_data:
            nearest_ball = min(all_ball_data, key=lambda b: b['point_camera'].point.z)
            self.publish_debug_image(cv_image, nearest_ball)

    def detect_all_green_tennis_balls(self, cv_image):
        """Tennis ball detection logic from table_tennis_detector"""
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []

        detected_balls = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_BALL_AREA:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            detected_balls.append({'contour': contour, 'area': area, 'bbox': (x, y, w, h)})
        
        return detected_balls

    def get_position_in_camera_frame(self, contour, header_stamp):
        """3D position calculation from table_tennis_detector"""
        M = cv2.moments(contour)
        if M["m00"] == 0: 
            return None
        
        pixel_x = int(M["m10"] / M["m00"])
        pixel_y = int(M["m01"] / M["m00"])

        depth_mm = self.latest_depth_image[pixel_y, pixel_x]
        if depth_mm == 0:
            return None
        
        depth_m = float(depth_mm) / 1000.0
        fx = self.camera_intrinsics.k[0]
        fy = self.camera_intrinsics.k[4]
        cx = self.camera_intrinsics.k[2]
        cy = self.camera_intrinsics.k[5]

        x_cam = (pixel_x - cx) * depth_m / fx
        y_cam = (pixel_y - cy) * depth_m / fy
        z_cam = depth_m

        point_in_camera_frame = PointStamped()
        point_in_camera_frame.header.stamp = header_stamp
        point_in_camera_frame.header.frame_id = self.source_frame
        point_in_camera_frame.point = Point(x=x_cam, y=y_cam, z=z_cam)
        
        return {'contour': contour, 'point_camera': point_in_camera_frame}

    def publish_permanent_marker(self, position: Point, marker_id: int):
        """Persistent marker publishing from horizontal_cylinder_detector"""
        marker = Marker()
        marker.header.frame_id = self.target_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "permanent_tennis_balls"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = position.x
        marker.pose.position.y = position.y
        marker.pose.position.z = position.z
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.15
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.9
        marker.lifetime = Duration(seconds=0).to_msg()  # Permanent marker
        self.marker_publisher.publish(marker)
        self.get_logger().info(f"Published permanent tennis ball marker with ID {marker_id}.")

    def publish_debug_image(self, cv_image, ball_data):
        """Debug visualization from table_tennis_detector"""
        debug_image = cv_image.copy()
        x, y, w, h = cv2.boundingRect(ball_data['contour'])
        depth_m = ball_data['point_camera'].point.z
        label = f"Ball @ {depth_m:.2f}m"
        cv2.rectangle(debug_image, (x, y), (x + w, y + h), (36, 255, 12), 2)
        cv2.putText(debug_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
        
        # Add overall stats
        stats_text = f"Known Balls: {len(self.known_balls)} | Frame: {self.frame_count}"
        cv2.putText(debug_image, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        try:
            self.debug_image_pub.publish(self.bridge.cv2_to_imgmsg(debug_image, 'bgr8'))
        except Exception as e:
            self.get_logger().error(f'Failed to publish debug image: {e}')


def main(args=None):
    rclpy.init(args=args)
    detector_node = CombinedTennisBallDetector()
    try: 
        rclpy.spin(detector_node)
    except KeyboardInterrupt: 
        pass
    finally:
        detector_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
