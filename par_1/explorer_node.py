#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist, PointStamped
from visualization_msgs.msg import Marker
import tf2_ros
import tf2_geometry_msgs
import cv2
from cv_bridge import CvBridge
import numpy as np

class MacadamiaExplorer(Node):
    def __init__(self):
        super().__init__('macadamia_explorer')
        self.bridge = CvBridge()
        self.scan = None
        self.detected_points = []

        self.sub_image = self.create_subscription(Image, '/oak/rgb/image_raw', self.image_callback, 10)
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.marker_pub = self.create_publisher(Marker, '/nut_markers', 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.green_lower = np.array([25, 100, 100])
        self.green_upper = np.array([40, 255, 255])

        self.row_following_active = True
        self.get_logger().info('Macadamia Explorer Initialized')
        self.create_timer(0.2, self.row_follow_logic)

    def scan_callback(self, msg):
        self.scan = msg

    def image_callback(self, msg):
        if self.scan is None:
            return
        image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 200:
            return
        x, y, w, h = cv2.boundingRect(c)
        cx = x + w // 2

        angle = self.scan.angle_min + (cx / image.shape[1]) * (self.scan.angle_max - self.scan.angle_min)
        index = int((angle - self.scan.angle_min) / self.scan.angle_increment)
        if index < 0 or index >= len(self.scan.ranges):
            return
        dist = self.scan.ranges[index]
        if not np.isfinite(dist):
            return

        if dist > 0.6:
            return  # Skip distant nuts

        lx = dist * np.cos(angle)
        ly = dist * np.sin(angle)
        point_lidar = PointStamped()
        point_lidar.header.frame_id = self.scan.header.frame_id
        point_lidar.header.stamp = self.scan.header.stamp
        point_lidar.point.x = lx
        point_lidar.point.y = ly
        point_lidar.point.z = 0.0

        try:
            tf = self.tf_buffer.lookup_transform('map', point_lidar.header.frame_id, rclpy.time.Time())
            point_map = tf2_geometry_msgs.do_transform_point(point_lidar, tf)
        except Exception as e:
            self.get_logger().warn(f"TF failed: {e}")
            return

        for prev in self.detected_points:
            dx = point_map.point.x - prev.point.x
            dy = point_map.point.y - prev.point.y
            if np.hypot(dx, dy) < 0.3:
                return

        self.detected_points.append(point_map)
        self.publish_marker(point_map)
        self.get_logger().info(f"Nut #{len(self.detected_points)} detected and recorded.")
        self.avoid_obstacle()

    def publish_marker(self, pt):
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'nuts'
        m.id = len(self.detected_points)
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position = pt.point
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.15
        m.color.r = 0.0
        m.color.g = 1.0
        m.color.b = 0.0
        m.color.a = 0.9
        self.marker_pub.publish(m)

    def avoid_obstacle(self):
        self.row_following_active = False
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.5
        self.cmd_vel_pub.publish(twist)
        self.get_logger().info("Avoiding nut: turning left for 1.5 seconds")
        self.create_timer(1.5, self.resume_navigation, callback_group=None)

    def resume_navigation(self):
        self.cmd_vel_pub.publish(Twist())
        self.row_following_active = True
        self.get_logger().info("Resumed row following.")

    def row_follow_logic(self):
        if not self.row_following_active or self.scan is None:
            return

        ranges = np.array(self.scan.ranges)
        mid_idx = len(ranges) // 2
        window = 20
        left_range = np.nanmean(ranges[mid_idx - window - 30:mid_idx - 30])
        right_range = np.nanmean(ranges[mid_idx + 30:mid_idx + 30 + window])

        twist = Twist()
        twist.linear.x = 0.15
        if np.isfinite(left_range) and np.isfinite(right_range):
            diff = right_range - left_range
            twist.angular.z = -diff * 0.5

        self.cmd_vel_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = MacadamiaExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
