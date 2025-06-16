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

#+ We can rename the class to be more generic
class BallDetector(Node):
    def __init__(self):
        #+ Update the node name
        super().__init__('table_tennis_detector')

        self.bridge = CvBridge()
        self.scan = None

        # Subscribers
        self.sub_image = self.create_subscription(Image, '/oak/rgb/image_raw', self.image_callback, 10)
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # Publishers
        #+ Update the marker topic name for clarity
        self.marker_pub = self.create_publisher(Marker, '/ping_pong_ball_marker', 10)
        self.home_trigger_pub = self.create_publisher(Empty, '/trigger_home', 10)

        # TF setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Detection tracking
        self.detected_points = []  # Store detected ball positions in the map frame
        self.detection_count = 0
        self.frame_count = 0

        #--- REMOVE ALL COLOR-BASED DETECTION VARIABLES ---
        #- self.green_lower = np.array([25, 100, 100])
        #- self.green_upper = np.array([40, 255, 255])
        #- self.other_color_ranges = [ ... ]

        #+ Update logging messages
        self.get_logger().info('Shape-based Ball Detector initialized!')
        self.get_logger().info('Looking for any circular objects (ping pong balls)')

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
            
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            #+ Detect balls using shape (Hough Circle Transform)
            detected_balls = self.detect_balls(cv_image)
            
            if detected_balls:
                #+ Loop through all detected balls in the frame
                for ball_data in detected_balls:
                    self.process_ball(cv_image, ball_data)
            else:
                # Log scanning status occasionally
                if self.frame_count % 120 == 0:  # Every ~4 seconds
                    self.get_logger().info('Scanning for balls...')
                    
        except Exception as e:
            self.get_logger().error(f'Error in ball detection: {str(e)}')

    #+ NEW DETECTION METHOD USING HOUGH CIRCLES
    def detect_balls(self, cv_image):
        """Detect circular objects (ping pong balls) of any color."""
        # 1. Preprocessing
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        # Apply a Gaussian blur to reduce noise and improve detection
        gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2, 2)

        # 2. Hough Circle Transform
        # These parameters need careful tuning!
        # - dp: Inverse ratio of accumulator resolution. 1 is usually fine.
        # - minDist: Minimum distance between centers of detected circles.
        # - param1: Higher threshold for the Canny edge detector.
        # - param2: Accumulator threshold for circle centers (confidence). Lower value -> more circles.
        # - minRadius, maxRadius: The range of ball radii (in pixels) you expect to see.
        circles = cv2.HoughCircles(
            gray_blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1.2, 
            minDist=30,
            param1=100, 
            param2=25, 
            minRadius=10, 
            maxRadius=60
        )
        
        if circles is None:
            return None

        # Convert the (x, y, r) parameters to integers
        circles = np.uint16(np.around(circles))
        
        ball_data_list = []
        for i in circles[0, :]:
            # Create a dictionary for each detected ball
            ball_data = {
                'center_x': i[0],
                'center_y': i[1],
                'radius': i[2]
            }
            ball_data_list.append(ball_data)

        return ball_data_list

    #- REMOVE OLD COLOR-BASED DETECTION METHODS
    #- def detect_green_tennis_ball(self, cv_image): ...
    #- def detect_other_colored_balls(self, cv_image): ...
    
    #+ RENAME and simplify the processing function
    def process_ball(self, cv_image, ball_data):
        """Process a detected ball."""
        center_x = ball_data['center_x']
        center_y = ball_data['center_y']
        
        # Get world coordinates from the ball's center pixel
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

            # Transform to map frame and publish marker if it's a new ball
            self.transform_and_publish(point_lidar)

    def get_world_coordinates(self, center_x, image_width):
        """Convert pixel coordinates to world coordinates."""
        try:
            normalized_x = center_x / image_width
            angle = self.scan.angle_min + normalized_x * (self.scan.angle_max - self.scan.angle_min)
            index = int((angle - self.scan.angle_min) / self.scan.angle_increment)
            if index < 0 or index >= len(self.scan.ranges):
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
            if map_coords:
                return (lidar_x, lidar_y, map_coords[0], map_coords[1])
            else:
                self.get_logger().warn('Could not get map coordinates, but have LiDAR coordinates.')
                return (lidar_x, lidar_y, None, None)
        except Exception as e:
            self.get_logger().error(f'Error getting world coordinates: {str(e)}')
            return None

    def transform_to_map(self, point_lidar):
        """Transform point from LiDAR frame to map frame."""
        try:
            timeout = rclpy.duration.Duration(seconds=0.1)
            if self.tf_buffer.can_transform('map', point_lidar.header.frame_id, rclpy.time.Time(), timeout=timeout):
                transform = self.tf_buffer.lookup_transform('map', point_lidar.header.frame_id, rclpy.time.Time())
                point_map = tf2_geometry_msgs.do_transform_point(point_lidar, transform)
                return (point_map.point.x, point_map.point.y)
            else:
                self.get_logger().warn('TF transform not ready to transform to map frame')
                return None
        except Exception as e:
            self.get_logger().error(f'TF exception: {e}')
            return None

    def is_new_ball(self, new_point, threshold=0.3):
        """Check if this is a new ball or previously detected. Threshold is in meters."""
        for point in self.detected_points:
            dist = np.hypot(new_point.point.x - point.point.x, new_point.point.y - point.point.y)
            if dist < threshold:
                return False
        return True

    def transform_and_publish(self, point_lidar):
        """Transform to map frame and publish marker if new ball."""
        try:
            timeout = rclpy.duration.Duration(seconds=0.1)
            if self.tf_buffer.can_transform('map', point_lidar.header.frame_id, rclpy.time.Time(), timeout=timeout):
                transform = self.tf_buffer.lookup_transform('map', point_lidar.header.frame_id, rclpy.time.Time())
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
            else:
                self.get_logger().warn('TF transform not ready for publishing')
        except Exception as e:
            self.get_logger().error(f'TF exception during publish: {e}')

    def publish_marker(self, point_map):
        """Publish marker for RViz visualization."""
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        #+ Update namespace
        marker.ns = 'ping_pong_balls'
        marker.id = len(self.detected_points)
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position = point_map.point
        marker.pose.orientation.w = 1.0

        #+ Change marker size and color to something neutral/distinct, like orange
        marker.scale.x = marker.scale.y = marker.scale.z = 0.08 # Ping pong balls are smaller
        marker.color.r = 1.0  # Orange
        marker.color.g = 0.5
        marker.color.b = 0.0
        marker.color.a = 0.9

        self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    
    try:
        detector = BallDetector()
        rclpy.spin(detector)
    except KeyboardInterrupt:
        pass
    finally:
        if 'detector' in locals():
            detector.get_logger().info('Ball detector shutting down...')
            detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
