#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Empty
import tf2_ros
import tf2_geometry_msgs
import cv2
from cv_bridge import CvBridge
import numpy as np
import math

class TennisBallMarker(Node):
    def __init__(self):
        super().__init__('tennis_ball_marker')

        self.bridge = CvBridge()
        self.scan_data = None
        self.camera_frame = 'base_link'  # Assuming camera is mounted on base_link
        self.image_width = 640  # default fallback
        self.camera_fov_deg = 87.0  # OAK-D default horizontal FOV

        self.sub_image = self.create_subscription(Image, '/oak/rgb/image_raw', self.image_callback, 10)
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        self.marker_pub = self.create_publisher(Marker, '/tennis_ball_marker', 10)
        self.home_trigger_pub = self.create_publisher(Empty, '/trigger_home', 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.green_lower = np.array([25, 100, 100])
        self.green_upper = np.array([40, 255, 255])
        self.detected_points = []
        self.marker_id = 0

        self.get_logger().info('Tennis Ball LIDAR Marker initialized.')

    def scan_callback(self, msg):
        self.scan_data = msg

    def image_callback(self, msg):
        if self.scan_data is None:
            self.get_logger().warn('Waiting for LiDAR scan...')
            return

        try:
            self.image_width = msg.width
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
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

            point_robot = self.estimate_position_from_lidar(cx)
            if point_robot is not None:
                self.transform_and_publish_marker(point_robot)

        except Exception as e:
            self.get_logger().error(f'Image processing error: {e}')

    def estimate_position_from_lidar(self, cx):
        angle_offset = (cx - self.image_width / 2) / (self.image_width / 2)
        angle_rad = math.radians(angle_offset * (self.camera_fov_deg / 2))

        # Find the index in the scan closest to the angle
        scan = self.scan_data
        angle = angle_rad
        index = int((angle - scan.angle_min) / scan.angle_increment)

        if 0 <= index < len(scan.ranges):
            distance = scan.ranges[index]
            if math.isfinite(distance) and scan.range_min < distance < scan.range_max:
                x = distance * math.cos(angle)
                y = distance * math.sin(angle)

                point = PointStamped()
                point.header.frame_id = self.camera_frame
                point.header.stamp = self.get_clock().now().to_msg()
                point.point.x = x
                point.point.y = y
                point.point.z = 0.0
                return point
            else:
                self.get_logger().warn(f'Invalid range at index {index}')
        else:
            self.get_logger().warn('Scan index out of bounds')

        return None

    def transform_and_publish_marker(self, point_robot):
        try:
            if self.tf_buffer.can_transform('map', point_robot.header.frame_id, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.2)):
                transform = self.tf_buffer.lookup_transform('map', point_robot.header.frame_id, rclpy.time.Time())
                point_map = tf2_geometry_msgs.do_transform_point(point_robot, transform)

                if self.is_new_detection(point_map):
                    self.publish_marker(point_map)
                    self.detected_points.append(point_map)
                    self.home_trigger_pub.publish(Empty())
                    self.get_logger().info(f'Marker at map: ({point_map.point.x:.2f}, {point_map.point.y:.2f})')
            else:
                self.get_logger().warn('TF not available: base_link → map')
        except Exception as e:
            self.get_logger().error(f'TF transform error: {e}')

    def is_new_detection(self, point, threshold=0.3):
        for p in self.detected_points:
            dx = p.point.x - point.point.x
            dy = p.point.y - point.point.y
            if math.hypot(dx, dy) < threshold:
                return False
        return True

    def publish_marker(self, point_map):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'tennis_ball'
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
