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
        self.detected_points = []  # Store detected ball positions in the map frame
        self.detection_count = 0
        self.frame_count = 0

        self.get_logger().info('White Ping Pong Ball Detector initialized!')
        self.get_logger().info('Strategy: Mask out GREEN floor, then find WHITE circles.')

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

            # Detect balls using the hybrid color-mask + shape-detection method
            detected_balls = self.detect_balls(cv_image)
            
            if detected_balls:
                for ball_data in detected_balls:
                    self.process_ping_pong_ball(cv_image, ball_data)
            else:
                if self.frame_count % 120 == 0:  # Log status every ~4 seconds
                    self.get_logger().info('Scanning for white ping pong balls...')
                    
        except Exception as e:
            self.get_logger().error(f'Error in ball detection: {str(e)}')

    def detect_balls(self, cv_image):
        """
        Detects white circular balls by first masking out the green floor to create contrast.
        """
        # 1. Create a mask for the green floor
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # --- IMPORTANT ---
        # These HSV values for green must be tuned for your specific lighting and floor color!
        green_lower = np.array([35, 50, 50])
        green_upper = np.array([85, 255, 255])
        floor_mask = cv2.inRange(hsv, green_lower, green_upper)

        # 2. Invert the mask to get everything that is NOT the floor (balls, walls, etc.)
        non_floor_mask = ~floor_mask

        # 3. Apply the inverted mask to a grayscale version of the image.
        # This makes the floor black, ensuring the ball has contrast all around it.
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        gray_masked = cv2.bitwise_and(gray, gray, mask=non_floor_mask)

        # 4. Blur the masked image to reduce noise for the circle detector
        gray_blurred = cv2.GaussianBlur(gray_masked, (9, 9), 2, 2)

        # 5. Run Hough Circle Transform on the pre-processed, masked image
        # These parameters require tuning for your specific camera setup and distance.
        circles = cv2.HoughCircles(
            gray_blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1.2, 
            minDist=40,       # Min distance between centers of detected circles.
            param1=100,       # Higher threshold for the internal Canny edge detector.
            param2=20,        # Accumulator threshold. Lower -> more (and possibly false) circles.
            minRadius=5,      # Minimum radius of the ball in pixels (when it's far away).
            maxRadius=40      # Maximum radius of the ball in pixels (when it's close).
        )
        
        if circles is None:
            return None

        circles = np.uint16(np.around(circles))
        ball_data_list = []
        for i in circles[0, :]:
            ball_data = {'center_x': i[0], 'center_y': i[1], 'radius': i[2]}
            ball_data_list.append(ball_data)

        return ball_data_list

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
            if not (self.scan.range_min <= distance <= self.scan.range_max):
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

        # Set scale to the real-world size of a ping pong ball (40mm diameter)
        marker.scale.x = marker.scale.y = marker.scale.z = 0.04
        
        # Set color to a distinct bright blue for high visibility in RViz
        marker.color.r = 0.0
        marker.color.g = 0.7
        marker.color.b = 1.0
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
