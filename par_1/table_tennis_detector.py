#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty
from sensor_msgs.msg import Image, LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PointStamped
import tf2_ros
import tf2_geometry_msgs
import cv2
from cv_bridge import CvBridge
import numpy as np

class PingPongBallDetector(Node):
    def __init__(self):
        super().__init__('table_tennis_detector')

        self.bridge = CvBridge()
        self.scan = None

        # Subscribers
        self.sub_image = self.create_subscription(Image, '/oak/rgb/image_raw', self.image_callback, 10)
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # Publishers
        self.marker_pub = self.create_publisher(Marker, '/ping_pong_ball_marker', 10)
        self.home_trigger_pub = self.create_publisher(Empty, '/trigger_home', 10)

        # TF setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Detection tracking
        self.detected_points = []
        self.detection_count = 0
        self.frame_count = 0

        # --- NEW DETECTION PARAMETERS ---
        # These parameters are crucial for filtering and will need to be tuned.
        self.MIN_CONTOUR_AREA = 100    # Minimum pixel area to be considered a potential ball.
        self.MAX_CONTOUR_AREA = 5000   # Maximum pixel area. Prevents detecting the white wall.
        self.MIN_CIRCULARITY = 0.70    # How "circle-like" the contour must be (1.0 is a perfect circle).

        self.get_logger().info('Advanced Ping Pong Ball Detector initialized!')
        self.get_logger().info('Strategy: Multi-stage contour filtering (Color, Size, Circularity).')


    def scan_callback(self, msg):
        """Store LiDAR scan data."""
        self.scan = msg

    def image_callback(self, msg):
        """Main image processing callback."""
        if self.scan is None:
            self.get_logger().warn('Waiting for LiDAR scan data...')
            return

        try:
            self.frame_count += 1
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Detect balls using the new advanced contour filtering method
            detected_balls = self.detect_ball_with_contour_filtering(cv_image)
            
            if detected_balls:
                # This logic now supports finding multiple valid balls in a frame
                for ball_data in detected_balls:
                    self.process_ping_pong_ball(cv_image, ball_data)
            else:
                if self.frame_count % 120 == 0:
                    self.get_logger().info('Scanning for white ping pong balls...')
                    
        except Exception as e:
            self.get_logger().error(f'Error in ball detection: {str(e)}', exc_info=True)

    # --- COMPLETELY REWRITTEN DETECTION FUNCTION ---
    def detect_ball_with_contour_filtering(self, cv_image):
        """
        Detects the ball using a multi-stage filtering process to eliminate false positives
        from reflections, other objects, and background walls.
        """
        # 1. Create masks for colors of interest
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # Mask for the green floor (tuned for typical indoor astroturf)
        green_lower = np.array([30, 40, 40])
        green_upper = np.array([90, 255, 255])
        floor_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        # Mask for white objects. In HSV, white has low Saturation and high Value.
        white_lower = np.array([0, 0, 180])
        white_upper = np.array([180, 70, 255])
        white_mask = cv2.inRange(hsv, white_lower, white_upper)

        # 2. Combine masks: Find things that are WHITE and NOT on the GREEN FLOOR.
        #    The `~` inverts the floor_mask (we want where the floor ISN'T).
        target_mask = cv2.bitwise_and(white_mask, ~floor_mask)

        # 3. Clean the final mask to remove small noise.
        kernel = np.ones((5, 5), np.uint8)
        target_mask = cv2.morphologyEx(target_mask, cv2.MORPH_OPEN, kernel)
        target_mask = cv2.morphologyEx(target_mask, cv2.MORPH_CLOSE, kernel)

        # 4. Find all contours in our clean target mask.
        contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_balls = []
        if not contours:
            return None

        for c in contours:
            # 5. Apply filters to each contour
            
            # Filter 1: Area Filter
            area = cv2.contourArea(c)
            if not (self.MIN_CONTOUR_AREA < area < self.MAX_CONTOUR_AREA):
                continue # Skip this contour, it's too small (noise) or too big (wall)

            # Filter 2: Circularity Filter
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue # Avoid division by zero
            
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity < self.MIN_CIRCULARITY:
                continue # Skip this contour, it's not shaped like a circle
            
            # If a contour passes all filters, it's a valid ball
            # Get the center of the contour
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            
            valid_balls.append({'center_x': center_x, 'center_y': center_y, 'area': area})
        
        return valid_balls if valid_balls else None


    def process_ping_pong_ball(self, cv_image, ball_data):
        """Processes a single detected ball."""
        center_x = ball_data['center_x']
        center_y = ball_data['center_y']
        
        world_coords = self.get_world_coordinates(center_x, cv_image.shape[1])
        
        if world_coords:
            lidar_x, lidar_y, map_x, map_y = world_coords
            
            self.detection_count += 1
            self.get_logger().info('PING PONG BALL DETECTED!')
            self.get_logger().info(f'  Detection #{self.detection_count}:')
            self.get_logger().info(f'  Pixel coordinates: ({int(center_x)}, {int(center_y)})')
            self.get_logger().info(f'  LiDAR coordinates: ({lidar_x:.2f}, {lidar_y:.2f}) meters')
            if map_x is not None:
                self.get_logger().info(f'  Map coordinates: ({map_x:.2f}, {map_y:.2f}) meters')
            
            point_lidar = PointStamped()
            point_lidar.header.frame_id = self.scan.header.frame_id
            point_lidar.header.stamp = self.scan.header.stamp
            point_lidar.point.x = lidar_x
            point_lidar.point.y = lidar_y
            point_lidar.point.z = 0.0

            self.transform_and_publish(point_lidar)

    def get_world_coordinates(self, center_x, image_width):
        """Convert pixel coordinates to world coordinates."""
        try:
            normalized_x = center_x / image_width
            angle = self.scan.angle_min + normalized_x * (self.scan.angle_max - self.scan.angle_min)
            index = int((angle - self.scan.angle_min) / self.scan.angle_increment)
            if not (0 <= index < len(self.scan.ranges)):
                self.get_logger().warn(f'LiDAR index {index} out of range')
                return None
            distance = self.scan.ranges[index]
            if not np.isfinite(distance) or not (self.scan.range_min <= distance <= self.scan.range_max):
                self.get_logger().warn(f'Invalid LiDAR distance: {distance:.2f}')
                return None
            lidar_x = distance * np.cos(angle)
            lidar_y = distance * np.sin(angle)
            point_lidar = PointStamped()
            point_lidar.header.frame_id = self.scan.header.frame_id
            point_lidar.header.stamp = self.scan.header.stamp
            point_lidar.point.x = lidar_x
            point_lidar.point.y = lidar_y
            point_lidar.point.z = 0.0
            map_coords = self.transform_to_map(point_lidar)
            return (lidar_x, lidar_y, map_coords[0], map_coords[1]) if map_coords else (lidar_x, lidar_y, None, None)
        except Exception as e:
            self.get_logger().error(f'Error getting world coordinates: {str(e)}')
            return None

    def transform_to_map(self, point_lidar):
        """Transform point from LiDAR frame to map frame."""
        try:
            timeout = rclpy.duration.Duration(seconds=0.1)
            transform = self.tf_buffer.lookup_transform('map', point_lidar.header.frame_id, rclpy.time.Time(), timeout=timeout)
            point_map = tf2_geometry_msgs.do_transform_point(point_lidar, transform)
            return (point_map.point.x, point_map.point.y)
        except tf2_ros.TransformException as e:
            self.get_logger().warn(f'TF transform not ready to transform to map frame: {e}')
            return None

    def is_new_ball(self, new_point, threshold=0.3):
        """Check if this is a new ball or previously detected. Threshold is in meters."""
        for point in self.detected_points:
            if np.hypot(new_point.point.x - point.point.x, new_point.point.y - point.point.y) < threshold:
                return False
        return True

    def transform_and_publish(self, point_lidar):
        """Transform to map frame and publish marker if new ball."""
        try:
            timeout = rclpy.duration.Duration(seconds=0.1)
            transform = self.tf_buffer.lookup_transform('map', point_lidar.header.frame_id, rclpy.time.Time(), timeout=timeout)
            point_map = tf2_geometry_msgs.do_transform_point(point_lidar, transform)

            if self.is_new_ball(point_map):
                self.publish_marker(point_map)
                self.detected_points.append(point_map)
                self.get_logger().info(f'New ping pong ball confirmed at map ({point_map.point.x:.2f}, {point_map.point.y:.2f})')
                self.get_logger().info(f'Published marker #{len(self.detected_points)} in RViz')
                
                self.home_trigger_pub.publish(Empty())
                self.get_logger().info('Triggered home return sequence')
            else:
                self.get_logger().info('Previously detected ball - ignoring duplicate')
        except tf2_ros.TransformException as e:
            self.get_logger().error(f'TF exception during publish: {e}')

    def publish_marker(self, point_map):
        """Publish marker for RViz visualization."""
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'ping_pong_balls'
        marker.id = len(self.detected_points)
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = point_map.point
        marker.pose.orientation.w = 1.0

        marker.scale.x = marker.scale.y = marker.scale.z = 0.04
        
        marker.color.r = 1.0
        marker.color.g = 0.5
        marker.color.b = 0.0
        marker.color.a = 1.0

        self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    detector = None
    try:
        detector = PingPongBallDetector()
        rclpy.spin(detector)
    except KeyboardInterrupt:
        pass
    finally:
        if detector:
            detector.get_logger().info('Ping pong ball detector shutting down...')
            detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
