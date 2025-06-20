#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from std_msgs.msg import Empty
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PointStamped, Point
import tf2_ros
import tf2_geometry_msgs
import cv2
from cv_bridge import CvBridge
import numpy as np

class NewTennisBallDetector(Node):
    def __init__(self):
        super().__init__('new_table_tennis_detector')

        self.bridge = CvBridge()
        self.latest_depth_image = None
        self.camera_intrinsics = None

        # --- TF SETUP (Increase buffer duration to handle potential delays) ---
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- SUBSCRIBERS ---
        self.image_sub = self.create_subscription(
            CompressedImage, '/oak/rgb/image_rect/compressed', self.image_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/oak/stereo/image_raw', self.depth_callback, 10)
        self.info_sub = self.create_subscription(
            CameraInfo, '/oak/rgb/camera_info', self.info_callback, 10)

        # --- PUBLISHERS ---
        self.marker_pub = self.create_publisher(Marker, '/tennis_ball_marker', 10)
        self.home_trigger_pub = self.create_publisher(Empty, '/trigger_home', 10)
        self.debug_image_pub = self.create_publisher(Image, '~/debug_image', 10)

        # --- DETECTION TRACKING ---
        self.detected_points = []
        self.detection_count = 0
        self.frame_count = 0

        # --- COLOR RANGE (Green tennis balls) ---
        self.green_lower = np.array([25, 100, 100])
        self.green_upper = np.array([40, 255, 255])

        self.get_logger().info('Tennis Ball Detector (Depth Camera, Corrected) initialized!')
        self.get_logger().info('Looking for GREEN tennis balls.')

    def info_callback(self, msg):
        if self.camera_intrinsics is None:
            self.camera_intrinsics = msg
            self.get_logger().info('Camera intrinsics received.')
            self.destroy_subscription(self.info_sub)

    def depth_callback(self, msg):
        self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def image_callback(self, msg):
        if self.latest_depth_image is None or self.camera_intrinsics is None:
            return

        self.frame_count += 1
        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
        if cv_image.shape[:2] != self.latest_depth_image.shape[:2]:
            return

        # This part remains the same, it finds all potential balls in the current view
        detected_balls = self.detect_all_green_tennis_balls(cv_image)
        if not detected_balls:
            return

        all_ball_data = []
        for ball_dict in detected_balls:
            ball_data_with_depth = self.get_position_in_camera_frame(ball_dict['contour'], msg.header.stamp)
            if ball_data_with_depth:
                all_ball_data.append(ball_data_with_depth)
        
        if not all_ball_data:
            return

        # --- MODIFICATION START: Process ALL detected balls, not just the nearest ---
        
        # Create a copy of the image to draw on for debugging
        debug_image = cv_image.copy()
        new_balls_found_in_frame = []

        # Loop through every ball we found in the frame
        for ball_data in all_ball_data:
            point_in_camera_frame = ball_data['point_camera']
            source_frame = point_in_camera_frame.header.frame_id
            target_frame = 'map'
            
            try:
                transform_stamped = self.tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    rclpy.time.Time(),
                    timeout=Duration(seconds=0.2)
                )
                point_in_map_frame = tf2_geometry_msgs.do_transform_point(point_in_camera_frame, transform_stamped)

                # Check if this ball is a new, unique detection
                if self.is_new_ball(point_in_map_frame):
                    self.detection_count += 1
                    
                    # --- MODIFICATION: Add terminal output with Z coordinate ---
                    camera_depth_z = point_in_camera_frame.point.z
                    self.get_logger().info(f"--- NEW BALL #{self.detection_count} DETECTED! ---")
                    self.get_logger().info(f'  - Map Coords: x={point_in_map_frame.point.x:.2f}, y={point_in_map_frame.point.y:.2f}')
                    self.get_logger().info(f'  - Camera Depth (Z): {camera_depth_z:.2f} meters')
                    
                    self.detected_points.append(point_in_map_frame)
                    self.publish_marker(point_in_map_frame, self.detection_count)
                    
                    # Keep track of new balls found in this specific frame for the debug image
                    new_balls_found_in_frame.append(ball_data)

            except tf2_ros.TransformException as e:
                self.get_logger().warn(f"Could not transform point: {e}", throttle_duration_sec=5.0)
                self.get_logger().warn(f"  - Target Frame: '{target_frame}'", throttle_duration_sec=5.0)
                self.get_logger().warn(f"  - Source Frame: '{source_frame}'", throttle_duration_sec=5.0)
                self.get_logger().warn(f"  - Image Timestamp: {point_in_camera_frame.header.stamp.sec}.{point_in_camera_frame.header.stamp.nanosec}", throttle_duration_sec=5.0)
                all_frames = self.tf_buffer.all_frames_as_string()
                self.get_logger().warn(f"  - Frames available in TF buffer:\n{all_frames}", throttle_duration_sec=5.0)

        # After checking all balls, publish one debug image with all new detections from this frame
        if new_balls_found_in_frame:
            self.publish_debug_image(debug_image, new_balls_found_in_frame)
        
        # --- MODIFICATION END ---

    def detect_all_green_tennis_balls(self, cv_image):
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []

        detected_balls = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            detected_balls.append({'contour': contour, 'area': area, 'bbox': (x, y, w, h)})
        
        # Sort by area (optional, but can be useful)
        detected_balls.sort(key=lambda b: b['area'], reverse=True)
        return detected_balls

    def get_position_in_camera_frame(self, contour, header_stamp):
        M = cv2.moments(contour)
        if M["m00"] == 0: return None
        pixel_x = int(M["m10"] / M["m00"]); pixel_y = int(M["m01"] / M["m00"])

        depth_mm = self.latest_depth_image[pixel_y, pixel_x]
        if depth_mm == 0:
            return None
        
        depth_m = float(depth_mm) / 1000.0
        fx = self.camera_intrinsics.k[0]; fy = self.camera_intrinsics.k[4]
        cx = self.camera_intrinsics.k[2]; cy = self.camera_intrinsics.k[5]

        x_cam = (pixel_x - cx) * depth_m / fx; y_cam = (pixel_y - cy) * depth_m / fy
        z_cam = depth_m

        point_in_camera_frame = PointStamped()
        point_in_camera_frame.header.stamp = header_stamp
        point_in_camera_frame.header.frame_id = 'oak_camera_rgb_camera_optical_frame'
        point_in_camera_frame.point = Point(x=x_cam, y=y_cam, z=z_cam)
        
        return {'contour': contour, 'point_camera': point_in_camera_frame}

    def is_new_ball(self, new_point_stamped, threshold=0.3):
        # Increased threshold slightly to be more robust against minor noise
        for existing_point_stamped in self.detected_points:
            distance = np.hypot(new_point_stamped.point.x - existing_point_stamped.point.x,
                                new_point_stamped.point.y - existing_point_stamped.point.y)
            if distance < threshold:
                return False
        return True

    def publish_marker(self, point_map, marker_id):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'tennis_balls'; marker.id = marker_id
        marker.type = Marker.SPHERE; marker.action = Marker.ADD
        marker.pose.position = point_map.point
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.15
        marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 0.0; marker.color.a = 0.9
        self.marker_pub.publish(marker)

    # --- MODIFICATION: Updated to draw multiple balls on a single image ---
    def publish_debug_image(self, debug_image, new_balls_data):
        """
        Draws bounding boxes for all newly detected balls onto a single debug image.
        
        :param debug_image: The CV2 image to draw on.
        :param new_balls_data: A list of ball_data dictionaries for new balls found in the frame.
        """
        for ball_data in new_balls_data:
            x, y, w, h = cv2.boundingRect(ball_data['contour'])
            depth_m = ball_data['point_camera'].point.z
            label = f"New @ {depth_m:.2f}m"
            cv2.rectangle(debug_image, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv2.putText(debug_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
        
        self.debug_image_pub.publish(self.bridge.cv2_to_imgmsg(debug_image, 'bgr8'))

def main(args=None):
    rclpy.init(args=args)
    detector = NewTennisBallDetector()
    try: rclpy.spin(detector)
    except KeyboardInterrupt: pass
    finally: detector.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()
