#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty
from sensor_msgs.msg import Image, LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PointStamped
import tf2_ros
import tf2_geometry_msgs  # Add this import
import cv2
from cv_bridge import CvBridge
import numpy as np

class TennisBallDetectorOpenCV(Node):
    def __init__(self):
        super().__init__('tennis_ball_detector_opencv')

        self.bridge = CvBridge()
        self.scan = None

        # self.sub_image = self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 10)
        self.sub_image = self.create_subscription(Image, '/oak/rgb/image_raw', self.image_callback, 10)
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        self.marker_pub = self.create_publisher(Marker, '/tennis_ball_marker', 10)
        self.home_trigger_pub = self.create_publisher(Empty, '/trigger_home', 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.seen = False

        self.get_logger().info('OpenCV Tennis Ball Detector initialized.')

    def scan_callback(self, msg):
        self.scan = msg

    def image_callback(self, msg):
        if self.scan is None:
            self.get_logger().warn('No scan data yet')
            return

        # Convert ROS Image to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # HSV color threshold for tennis ball (bright yellow-green)
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([25, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Morphology to reduce noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.get_logger().info('No tennis ball contours found')
            return

        # Largest contour assumed tennis ball
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        if area < 200:  # Filter small blobs
            self.get_logger().info('Contour too small')
            return

        # Compute bounding box and center X
        x, y, w, h = cv2.boundingRect(largest_contour)
        center_x = x + w / 2

        # Normalize horizontal center_x to [0,1]
        normalized_x = center_x / cv_image.shape[1]

        # Compute angle from LiDAR scan parameters
        angle = self.scan.angle_min + normalized_x * (self.scan.angle_max - self.scan.angle_min)

        # Get LiDAR range index
        index = int((angle - self.scan.angle_min) / self.scan.angle_increment)
        if index < 0 or index >= len(self.scan.ranges):
            self.get_logger().warn('LiDAR index out of range')
            return

        distance = self.scan.ranges[index]
        if not np.isfinite(distance) or not (self.scan.range_min <= distance <= self.scan.range_max):
            self.get_logger().warn(f'Invalid LiDAR distance: {distance}')
            return

        # Build point in LiDAR frame
        point_lidar = PointStamped()
        point_lidar.header.frame_id = self.scan.header.frame_id
        point_lidar.header.stamp = self.scan.header.stamp
        point_lidar.point.x = distance * np.cos(angle)
        point_lidar.point.y = distance * np.sin(angle)
        point_lidar.point.z = 0.0

        # Transform to map frame and publish marker
        self.transform_and_publish(point_lidar)

    def transform_and_publish(self, point_lidar):
        try:
            # Use proper timeout handling
            timeout = rclpy.duration.Duration(seconds=0.1)
            if self.tf_buffer.can_transform('map', point_lidar.header.frame_id, rclpy.time.Time(), timeout=timeout):
                # Use tf2_geometry_msgs for proper PointStamped transformation
                point_map = tf2_geometry_msgs.do_transform_point(
                    point_lidar,
                    self.tf_buffer.lookup_transform('map', point_lidar.header.frame_id, rclpy.time.Time(), timeout=timeout)
                )

                if not self.seen:
                    self.publish_marker(point_map)
                    self.seen = True
                    self.get_logger().info(f'Tennis ball detected at ({point_map.point.x:.2f}, {point_map.point.y:.2f})')

                    # Trigger home after detection (example)
                    self.home_trigger_pub.publish(Empty())
            else:
                self.get_logger().warn('TF transform not ready')
        except Exception as e:
            self.get_logger().error(f'TF exception: {e}')

    def publish_marker(self, point_map):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'tennis_ball'
        marker.id = 0
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
        self.get_logger().info('Published tennis ball marker.')

def main(args=None):
    rclpy.init(args=args)
    node = TennisBallDetectorOpenCV()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
