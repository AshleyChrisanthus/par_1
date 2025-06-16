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

        # Update with actual intrinsics if known
        self.fx = 615.0
        self.fy = 615.0
        self.cx = 320.0
        self.cy = 240.0
        self.camera_frame = 'oak_rgb_camera_optical_frame'

        # Subscriptions
        self.create_subscription(Image, '/oak/rgb/image_raw', self.image_callback, 10)
        self.create_subscription(Image, '/oak/stereo/image_raw', self.depth_callback, 10)

        # Publishers
        self.marker_pub = self.create_publisher(Marker, '/tennis_ball_marker', 10)
        self.home_trigger_pub = self.create_publisher(Empty, '/trigger_home', 10)

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.detected_points = []
        self.green_lower = np.array([25, 100, 100])
        self.green_upper = np.array([40, 255, 255])
        self.marker_id = 0

        self.get_logger().info("Tennis Ball Marker node initialized.")

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough').astype(np.uint16)
        except Exception as e:
            self.get_logger().error(f'Depth image error: {e}')

    def image_callback(self, msg):
        if self.depth_image is None:
            self.get_logger().warn('No depth image available yet.')
            return

        try:
            rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return

            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) < 200:
                return

            x, y, w, h = cv2.boundingRect(largest)
            cx = int(x + w / 2)
            cy = int(y + h / 2)

            depth = self.get_depth(cx, cy)
            if depth is None:
                self.get_logger().warn(f'Invalid depth at pixel ({cx}, {cy})')
                return

            X = (cx - self.cx) * depth / self.fx
            Y = (cy - self.cy) * depth / self.fy
            Z = depth

            pt = PointStamped()
            pt.header.frame_id = self.camera_frame
            pt.header.stamp = self.get_clock().now().to_msg()
            pt.point.x = X
            pt.point.y = Y
            pt.point.z = Z

            self.transform_and_publish_marker(pt)

        except Exception as e:
            self.get_logger().error(f'Image callback error: {e}')

    def get_depth(self, cx, cy):
        try:
            depth = self.depth_image[cy, cx]
            if depth > 0:
                return depth / 1000.0
            # fallback: average 3x3 kernel
            kernel = self.depth_image[max(cy - 1, 0):cy + 2, max(cx - 1, 0):cx + 2]
            valid = kernel[kernel > 0]
            return np.mean(valid) / 1000.0 if valid.size else None
        except:
            return None

    def transform_and_publish_marker(self, point_camera):
        try:
            timeout = rclpy.duration.Duration(seconds=0.2)
            if self.tf_buffer.can_transform('map', point_camera.header.frame_id, rclpy.time.Time(), timeout=timeout):
                transform = self.tf_buffer.lookup_transform('map', point_camera.header.frame_id, rclpy.time.Time(), timeout=timeout)
                point_map = tf2_geometry_msgs.do_transform_point(point_camera, transform)

                if self.is_new_detection(point_map):
                    self.publish_marker(point_map)
                    self.detected_points.append(point_map)
                    self.home_trigger_pub.publish(Empty())
                    self.get_logger().info(f'Published marker at ({point_map.point.x:.2f}, {point_map.point.y:.2f})')
            else:
                self.get_logger().warn('TF not available: camera → map')
        except Exception as e:
            self.get_logger().error(f'TF transform error: {e}')

    def is_new_detection(self, point, threshold=0.3):
        for p in self.detected_points:
            if np.hypot(p.point.x - point.point.x, p.point.y - point.point.y) < threshold:
                return False
        return True

    def publish_marker(self, point_map):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'tennis_balls'
        marker.id = self.marker_id
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
        self.marker_id += 1

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
