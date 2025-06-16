#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PointStamped
import tf2_ros
import tf2_geometry_msgs
import cv2
from cv_bridge import CvBridge
import numpy as np

class TennisBallMarker(Node):
    def __init__(self):
        super().__init__('tennis_ball_marker')

        self.bridge = CvBridge()
        self.depth_image = None

        # Default intrinsics (replace with real values from /oak/rgb/camera_info)
        self.fx = 615.0
        self.fy = 615.0
        self.cx = 320.0
        self.cy = 240.0
        self.camera_frame = 'oak_rgb_camera_frame'  # Replace with your actual RGB camera TF frame

        # Subscribers
        self.sub_image = self.create_subscription(Image, '/oak/rgb/image_raw', self.image_callback, 10)
        self.sub_depth = self.create_subscription(Image, '/oak/stereo/image_raw', self.depth_callback, 10)

        # Publishers
        self.marker_pub = self.create_publisher(Marker, '/tennis_ball_marker', 10)
        self.home_trigger_pub = self.create_publisher(Empty, '/trigger_home', 10)

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Detection state
        self.detected_points = []
        self.detection_count = 0
        self.frame_count = 0

        # Color detection
        self.green_lower = np.array([25, 100, 100])
        self.green_upper = np.array([40, 255, 255])

        self.get_logger().info('Tennis Ball Detector using stereo depth initialized!')

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough').astype(np.uint16)
        except Exception as e:
            self.get_logger().error(f'Depth callback error: {e}')

    def image_callback(self, msg):
        if self.depth_image is None:
            self.get_logger().warn('Waiting for depth image...')
            return

        try:
            self.frame_count += 1
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            ball = self.detect_green_tennis_ball(cv_image)
            if ball:
                self.process_macadamia_nut(ball)
            elif self.frame_count % 120 == 0:
                self.get_logger().info('Scanning for green tennis balls...')
        except Exception as e:
            self.get_logger().error(f'Image callback error: {e}')

    def detect_green_tennis_ball(self, cv_image):
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 200:
            return None

        x, y, w, h = cv2.boundingRect(largest)
        return {
            'center_x': x + w / 2,
            'center_y': y + h / 2
        }

    def process_macadamia_nut(self, ball):
        cx = int(ball['center_x'])
        cy = int(ball['center_y'])

        depth = self.depth_image[cy, cx] / 1000.0  # mm to meters
        if depth == 0 or not np.isfinite(depth):
            self.get_logger().warn(f'Invalid depth at pixel ({cx}, {cy})')
            return

        # Project pixel to 3D
        X = (cx - self.cx) * depth / self.fx
        Y = (cy - self.cy) * depth / self.fy
        Z = depth

        point_camera = PointStamped()
        point_camera.header.frame_id = self.camera_frame
        point_camera.header.stamp = self.get_clock().now().to_msg()
        point_camera.point.x = X
        point_camera.point.y = Y
        point_camera.point.z = Z

        self.transform_and_publish(point_camera)

    def transform_and_publish(self, point_camera):
        try:
            timeout = rclpy.duration.Duration(seconds=0.1)
            if self.tf_buffer.can_transform('map', point_camera.header.frame_id, rclpy.time.Time(), timeout=timeout):
                transform = self.tf_buffer.lookup_transform('map', point_camera.header.frame_id, rclpy.time.Time(), timeout=timeout)
                point_map = tf2_geometry_msgs.do_transform_point(point_camera, transform)

                if self.is_new_ball(point_map):
                    self.publish_marker(point_map)
                    self.detected_points.append(point_map)
                    self.get_logger().info(f'Nut detected at map: ({point_map.point.x:.2f}, {point_map.point.y:.2f})')
                    self.home_trigger_pub.publish(Empty())
            else:
                self.get_logger().warn('TF not available for camera->map')
        except Exception as e:
            self.get_logger().error(f'Transform error: {e}')

    def is_new_ball(self, new_point, threshold=0.3):
        for point in self.detected_points:
            dx = new_point.point.x - point.point.x
            dy = new_point.point.y - point.point.y
            if np.hypot(dx, dy) < threshold:
                return False
        return True

    def publish_marker(self, point_map):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'macadamia_nuts'
        marker.id = len(self.detected_points)
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = point_map.point
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.15
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.9

        self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = TennisBallMarker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
