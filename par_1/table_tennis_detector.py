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

        # --- TF SETUP (Same as cylinder detector) ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- SUBSCRIBERS (Using Depth Camera, NOT LiDAR) ---
        # Subscribing to the rectified, compressed image for efficiency and alignment with depth
        self.image_sub = self.create_subscription(
            CompressedImage, '/oak/rgb/image_rect/compressed', self.image_callback, 10)
        # Subscribing to the raw stereo depth image
        self.depth_sub = self.create_subscription(
            Image, '/oak/stereo/image_raw', self.depth_callback, 10)
        # Subscribing to camera info to get intrinsics (fx, fy, cx, cy)
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

        # --- COLOR RANGES ---
        # Using white for table tennis balls, as requested
        # H can be anything, S is very low, V is high.
        self.ball_lower = np.array([0, 0, 200])
        self.ball_upper = np.array([180, 30, 255])
        
        # Original green for tennis balls (macadamia nuts) - kept here for reference
        # self.ball_lower = np.array([25, 100, 100])
        # self.ball_upper = np.array([40, 255, 255])

        self.get_logger().info('Tennis Ball Detector (Depth Camera Version) initialized!')
        self.get_logger().info('Looking for WHITE table-tennis balls.')

    def info_callback(self, msg):
        """Store camera intrinsics and stop subscribing."""
        if self.camera_intrinsics is None:
            self.camera_intrinsics = msg
            self.get_logger().info('Camera intrinsics received.')
            # Unsubscribe after receiving the info once, as it's static
            self.destroy_subscription(self.info_sub)

    def depth_callback(self, msg):
        """Store latest depth image."""
        # The depth image is '16UC1' (16-bit unsigned, 1 channel)
        self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def image_callback(self, msg):
        """Main image processing callback using depth data."""
        # --- GUARD CLAUSES: Wait for necessary data ---
        if self.latest_depth_image is None or self.camera_intrinsics is None:
            self.get_logger().warn('Waiting for depth image and/or camera intrinsics...', throttle_duration_sec=2)
            return

        self.frame_count += 1
        
        try:
            # --- 1. IMAGE PREPARATION & DETECTION ---
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
            
            # Ensure depth and color images have the same dimensions
            if cv_image.shape[:2] != self.latest_depth_image.shape[:2]:
                self.get_logger().warn('Color and depth image dimensions do not match. Skipping frame.')
                return

            detected_balls = self.detect_all_balls(cv_image)
            
            if not detected_balls:
                if self.frame_count % 120 == 0:  # Log occasionally
                    self.get_logger().info('Scanning for white balls...')
                return
            
            # --- 2. PROCESS DETECTED BALLS ---
            self.get_logger().info(f'Found {len(detected_balls)} potential ball(s) in frame #{self.frame_count}')
            
            all_ball_data = []
            for ball_contour in detected_balls:
                # --- 3. CALCULATE 3D POSITION IN CAMERA FRAME ---
                ball_data = self.get_position_in_camera_frame(ball_contour)
                if ball_data:
                    all_ball_data.append(ball_data)
            
            if not all_ball_data:
                self.get_logger().info('No balls with valid depth found.')
                return

            # --- 4. TRANSFORM & PUBLISH THE NEAREST BALL ---
            # Find the nearest ball based on its depth (Z-distance in camera frame)
            nearest_ball = min(all_ball_data, key=lambda b: b['point_camera'].point.z)
            
            point_in_camera_frame = nearest_ball['point_camera']
            target_frame = 'map'
            
            # Use the robust transform method from the cylinder detector
            try:
                # Transform the point from the camera's frame to the map's frame
                point_in_map_frame = self.tf_buffer.transform(point_in_camera_frame, target_frame, timeout=Duration(seconds=0.2))
                
                # Check if this ball is a new detection or a duplicate
                if self.is_new_ball(point_in_map_frame):
                    self.detection_count += 1
                    self.get_logger().info(f"--- NEW BALL #{self.detection_count} DETECTED! ---")
                    self.get_logger().info(f'Published nearest ball at map coords: x={point_in_map_frame.point.x:.2f}, y={point_in_map_frame.point.y:.2f}')
                    
                    self.detected_points.append(point_in_map_frame)
                    self.publish_marker(point_in_map_frame)

                    # Optional: trigger a return-to-home sequence
                    # self.home_trigger_pub.publish(Empty())
                else:
                    self.get_logger().info(f"Duplicate ball detected at map coords: x={point_in_map_frame.point.x:.2f}, y={point_in_map_frame.point.y:.2f}. Ignoring.")

                # Publish debug image for the nearest valid ball
                self.publish_debug_image(cv_image, nearest_ball)

            except tf2_ros.TransformException as e:
                self.get_logger().warn(f"Could not transform point from '{point_in_camera_frame.header.frame_id}' to '{target_frame}': {e}")

        except Exception as e:
            self.get_logger().error(f'Error in image_callback: {str(e)}')

    def detect_all_balls(self, cv_image):
        """Detects all balls of the specified color in the frame."""
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.ball_lower, self.ball_upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area to remove noise
        min_area = 100
        return [c for c in contours if cv2.contourArea(c) > min_area]

    def get_position_in_camera_frame(self, contour):
        """Calculates a ball's 3D position in the camera's optical frame."""
        # Get center of the ball in pixel coordinates
        M = cv2.moments(contour)
        if M["m00"] == 0: return None
        pixel_x = int(M["m10"] / M["m00"])
        pixel_y = int(M["m01"] / M["m00"])

        # Get depth value at the center of the ball (in millimeters)
        depth_mm = self.latest_depth_image[pixel_y, pixel_x]

        # Check for invalid depth readings (0 usually means no reading)
        if depth_mm == 0:
            return None
        
        # --- PINHOLE CAMERA MODEL CALCULATION ---
        depth_m = float(depth_mm) / 1000.0
        
        fx = self.camera_intrinsics.k[0]
        fy = self.camera_intrinsics.k[4]
        cx = self.camera_intrinsics.k[2]
        cy = self.camera_intrinsics.k[5]

        # Deproject pixel to 3D point in camera's coordinate frame
        x_cam = (pixel_x - cx) * depth_m / fx
        y_cam = (pixel_y - cy) * depth_m / fy
        z_cam = depth_m

        # Create a PointStamped message for transformation
        point_in_camera_frame = PointStamped()
        point_in_camera_frame.header.stamp = self.get_clock().now().to_msg()
        point_in_camera_frame.header.frame_id = 'oak_rgb_camera_optical_frame' # CRITICAL: This must match your TF tree
        point_in_camera_frame.point = Point(x=x_cam, y=y_cam, z=z_cam)
        
        return {
            'contour': contour,
            'point_camera': point_in_camera_frame
        }

    def is_new_ball(self, new_point_stamped, threshold=0.3):
        """Check if this is a new ball or previously detected, using distance in the 'map' frame."""
        for existing_point_stamped in self.detected_points:
            dx = new_point_stamped.point.x - existing_point_stamped.point.x
            dy = new_point_stamped.point.y - existing_point_stamped.point.y
            distance = np.hypot(dx, dy)
            if distance < threshold:
                return False
        return True

    def publish_marker(self, point_map):
        """Publish marker for RViz visualization."""
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'tennis_balls'
        marker.id = len(self.detected_points)
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = point_map.point
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.1
        marker.color.r = 1.0; marker.color.g = 1.0; marker.color.b = 1.0; marker.color.a = 1.0 # White
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
        # cv2.imshow("Tennis Ball Detection", debug_image) # Optional: display locally
        # cv2.waitKey(1)

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
        # cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
