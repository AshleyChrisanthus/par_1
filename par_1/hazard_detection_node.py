#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty
from find_object_2d.msg import ObjectsStamped
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from tf2_geometry_msgs import PointStamped

import tf2_ros
import numpy as np

class HazardDetectionNode(Node):
    def __init__(self):
        super().__init__('hazard_detection_node')

        self.initialize_subscribers()
        self.initialize_publishers()
        self.setup_tf()

        self.scan = None
        self.seen_ids = set()

        self.get_logger().info('Hazard Detection Node is up and running.')

    def initialize_subscribers(self):
        self.create_subscription(ObjectsStamped, '/objectsStamped', self.on_objects_detected, 10)
        self.create_subscription(LaserScan, '/scan', self.on_scan_received, 10)

    def initialize_publishers(self):
        self.hazard_marker_pub = self.create_publisher(Marker, '/hazards', 10)
        self.start_trigger_publisher = self.create_publisher(Empty, '/trigger_start', 10)
        self.home_trigger_publisher = self.create_publisher(Empty, '/trigger_home', 10)  # New publisher for home trigger

    def setup_tf(self):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def on_scan_received(self, scan_msg):
        self.scan = scan_msg

    def on_objects_detected(self, msg):
        if self.scan is None:
            self.get_logger().warn('Scan data missing.')
            return

        if not msg.objects.data:
            self.get_logger().warn('Object data missing.')
            return

        total = len(msg.objects.data) // 12
        self.get_logger().info(f'Found {total} objects to analyze.')

        for i in range(0, len(msg.objects.data), 12):
            object_info = msg.objects.data[i:i + 12]
            self.process_object(object_info)

    def process_object(self, obj_data):
        object_id = int(obj_data[0])
        bbox_x = obj_data[3]
        bbox_width = obj_data[5]

        if object_id == 13:
            self.get_logger().info("Start marker detected.")
            msg = Empty()
            self.start_trigger_publisher.publish(msg)
            return

        image_center_x = bbox_x + bbox_width / 2.0
        normalized_x = image_center_x / 800
        angle = self.scan.angle_min + normalized_x * (self.scan.angle_max - self.scan.angle_min)
        index = int((angle - self.scan.angle_min) / self.scan.angle_increment)

        if 0 <= index < len(self.scan.ranges):
            distance = self.scan.ranges[index]

            if not np.isfinite(distance) or not (self.scan.range_min <= distance <= self.scan.range_max):
                self.get_logger().warn(f'Invalid range for object {object_id}: {distance:.2f}m')
                return

            point = PointStamped()
            point.header.frame_id = self.scan.header.frame_id
            point.header.stamp = self.scan.header.stamp
            point.point.x = distance * np.cos(angle)
            point.point.y = distance * np.sin(angle)
            point.point.z = 0.0

            self.transform_and_publish(object_id, point)

    def transform_and_publish(self, object_id, point_laser):
        try:
            if self.tf_buffer.can_transform('map', point_laser.header.frame_id, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1)):
                point_map = self.tf_buffer.transform(point_laser, 'map', timeout=rclpy.duration.Duration(seconds=0.1))
                self.get_logger().info(f'Object {object_id} transformed to map at ({point_map.point.x:.2f}, {point_map.point.y:.2f})')
                self.publish_marker(object_id, point_map)
            else:
                self.get_logger().warn(f'Transform to map frame not ready for object {object_id}.')
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f'TF error for object {object_id}: {e}')

    def publish_marker(self, obj_id, map_point):
        if obj_id in self.seen_ids:
            self.get_logger().info(f'Marker for object {obj_id} already exists.')
            return

        self.seen_ids.add(obj_id)

        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'hazard_zone'
        marker.id = obj_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position = map_point.point
        marker.pose.orientation.w = 1.0

        marker.scale.x = marker.scale.y = marker.scale.z = 0.2
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.8

        self.hazard_marker_pub.publish(marker)
        self.get_logger().info(f'Marker published for object {obj_id}.')

        # Check if 5 unique hazards have been detected
        if len(self.seen_ids) == 4:
            self.get_logger().info('Five hazard markers detected. Triggering return to home.')
            empty_msg = Empty()
            self.home_trigger_publisher.publish(empty_msg)

def main():
    rclpy.init()
    node = HazardDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
