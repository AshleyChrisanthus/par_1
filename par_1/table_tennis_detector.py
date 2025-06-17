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

class TennisBallDetector(Node):
    def __init__(self):
        super().__init__('table_tennis_detector')

        self.bridge = CvBridge()
        self.latest_depth_image = None
        self.camera_intrinsics = None

        # --- TF SETUP (From working cylinder detector) ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- SUBSCRIBERS (Using Depth Camera, NOT LiDAR) ---
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
        self.detected_points = []  # Store detected ball positions in the 'map' frame
        self.detection_count = 0
        self.frame_count = 0

        # --- COLOR RANGE (From original tennis_ball_detector.py) ---
        # Green tennis balls (macadamia nuts) - bright yellow-green
        self.green_lower = np.array([25, 100, 100])
        self.green_upper = np.array([40, 255, 255])

        self.get_logger().info('Tennis Ball Detector (Depth Camera, Corrected) initialized!')
        self.get_logger().info('Looking for GREEN tennis balls.')

    def info_callback(self, msg):
        """Store camera intrinsics and stop subscribing."""
        if self.camera_intrinsics is None:
            self.camera_intrinsics = msg
            self.get_logger().info('Camera intrinsics received.')
            self.destroy_subscription(self.info_sub)

    def depth_callback(self, msg):
        """Store latest depth image."""
        self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def image_callback(self, msg):
        """Main image processing callback using depth data."""
        if self.latest_depth_image is None or self.camera_intrinsics is None:
            return

        self.frame_count += 1
        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
        if cv_image.shape[:2] != self.latest_depth_image.shape[:2]:
            self.get_logger().warn('Color and depth image dimensions do not match.', throttle_duration_sec=5)
            return

        # --- 1. DETECTION (Using the original, requested detection logic) ---
        detected_balls = self.detect_all_green_tennis_balls(cv_image)
        if not detected_balls:
            return

        # --- 2. GET 3D POSITIONS & FIND NEAREST ---
        all_ball_data = []
        for ball_dict in detected_balls:
            # Use the contour from the dictionary returned by the detection function
            ball_data_with_depth = self.get_position_in_camera_frame(ball_dict['contour'], msg.header.stamp)
            if ball_data_with_depth:
                all_ball_data.append(ball_data_with_depth)
        
        if not all_ball_data:
            return

        # Find the nearest ball based on its depth (Z-distance in camera frame)
        nearest_ball = min(all_ball_data, key=lambda b: b['point_camera'].point.z)
        
        # --- 3. TRANSFORM & PUBLISH (With the robust TF check) ---
        point_in_camera_frame = nearest_ball['point_camera']
        source_frame = point_in_camera_frame.header.frame_id
        target_frame = 'map'
        
        # --- THE FIX: Wait for the transform to be available ---
        try:
            # Check if the transform is possible before requesting it.
            if not self.tf_buffer.can_transform(target_frame, source_frame, rclpy.time.Time(), timeout=Duration(seconds=0.1)):
                self.get_logger().warn(f"Waiting for transform from '{source_frame}' to '{target_frame}'...", throttle_duration_sec=1.0)
                return
            
            # Now that we know it exists, perform the transform.
            point_in_map_frame = self.tf_buffer.transform(point_in_camera_frame, target_frame)
            
            # --- 4. PROCESS THE FINAL POINT ---
            if self.is_new_ball(point_in_map_frame):
                self.detection_count += 1
                self.get_logger().info(f"--- NEW BALL #{self.detection_count} DETECTED! ---")
                self.get_logger().info(f'Published nearest ball at map coords: x={point_in_map_frame.point.x:.2f}, y={point_in_map_frame.point.y:.2f}')
                
                self.detected_points.append(point_in_map_frame)
                self.publish_marker(point_in_map_frame, self.detection_count)
            else:
                self.get_logger().info(f"Duplicate ball detected near map coords: x={point_in_map_frame.point.x:.2f}, y={point_in_map_frame.point.y:.2f}. Ignoring.", throttle_duration_sec=2.0)

            self.publish_debug_image(cv_image, nearest_ball)

        except tf2_ros.TransformException as e:
            self.get_logger().warn(f"Could not transform point: {e}")

    def detect_all_green_tennis_balls(self, cv_image):
        """Detect ALL green tennis balls. This logic is from the original file."""
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        # Only use MORPH_OPEN as in the original file
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []

        detected_balls = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Filter very small blobs
                continue
            # Return the dictionary structure the old code used, we only need the contour but this is faithful
            x, y, w, h = cv2.boundingRect(contour)
            detected_balls.append({'contour': contour, 'area': area, 'bbox': (x, y, w, h)})
        
        return detected_balls

    def get_position_in_camera_frame(self, contour, header_stamp):
        """Calculates a ball's 3D position in the camera's optical frame."""
        M = cv2.moments(contour)
        if M["m00"] == 0: return None
        pixel_x = int(M["m10"] / M["m00"])
        pixel_y = int(M["m01"] / M["m00"])

        depth_mm = self.latest_depth_image[pixel_y, pixel_x]
        if depth_mm == 0:
            return None
        
        depth_m = float(depth_mm) / 1000.0
        fx = self.camera_intrinsics.k[0]; fy = self.camera_intrinsics.k[4]
        cx = self.camera_intrinsics.k[2]; cy = self.camera_intrinsics.k[5]

        x_cam = (pixel_x - cx) * depth_m / fx
        y_cam = (pixel_y - cy) * depth_m / fy
        z_cam = depth_m

        point_in_camera_frame = PointStamped()
        point_in_camera_frame.header.stamp = header_stamp
        point_in_camera_frame.header.frame_id = 'oak_rgb_camera_optical_frame'
        point_in_camera_frame.point = Point(x=x_cam, y=y_cam, z=z_cam)
        
        return {'contour': contour, 'point_camera': point_in_camera_frame}

    def is_new_ball(self, new_point_stamped, threshold=0.3):
        """Check if this is a new ball using distance in the 'map' frame."""
        for existing_point_stamped in self.detected_points:
            distance = np.hypot(new_point_stamped.point.x - existing_point_stamped.point.x,
                                new_point_stamped.point.y - existing_point_stamped.point.y)
            if distance < threshold:
                return False
        return True

    def publish_marker(self, point_map, marker_id):
        """Publish a GREEN marker for RViz visualization."""
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'tennis_balls'
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = point_map.point
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.15
        # Green color for the marker
        marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 0.0; marker.color.a = 0.9
        self.marker_pub.publish(marker)

    def publish_debug_image(self, cv_image, ball_data):
        """Draw detection info on an image and publish it."""
        debug_image = cv_image.copy()
        x, y, w, h = cv2.boundingRect(ball_data['contour'])
        depth_m = ball_data['point_camera'].point.z
        label = f"Nearest @ {depth_m:.2f}m"
        cv2.rectangle(debug_image, (x, y), (x + w, y + h), (36, 255, 12), 2)
        cv2.putText(debug_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
        self.debug_image_pub.publish(self.bridge.cv2_to_imgmsg(debug_image, 'bgr8'))

def main(args=None):
    rclpy.init(args=args)
    detector = TennisBallDetector()
    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        pass
    finally:
        detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
