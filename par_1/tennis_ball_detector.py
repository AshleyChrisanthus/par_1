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

class TennisBallDetector(Node):
    def __init__(self):
        super().__init__('tennis_ball_detector')

        self.bridge = CvBridge()
        self.scan = None

        # Subscribers
        self.sub_image = self.create_subscription(Image, '/oak/rgb/image_raw', self.image_callback, 10)
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # Publishers
        self.marker_pub = self.create_publisher(Marker, '/tennis_ball_marker', 10)
        self.home_trigger_pub = self.create_publisher(Empty, '/trigger_home', 10)

        # TF setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Detection tracking
        self.detected_points = []  # Store detected tennis ball positions
        self.detection_count = 0
        self.frame_count = 0

        # Color ranges for detection
        # Green tennis balls (macadamia nuts) - bright yellow-green
        self.green_lower = np.array([25, 100, 100])
        self.green_upper = np.array([40, 255, 255])
        
        # Other ball colors to reject
        self.other_color_ranges = [
            # Red balls
            ([0, 100, 100], [10, 255, 255], "red"),
            ([170, 100, 100], [180, 255, 255], "red"),
            # Blue balls  
            ([100, 100, 100], [130, 255, 255], "blue"),
            # Yellow balls (different from green-yellow)
            ([15, 100, 100], [25, 255, 255], "yellow"),
            # Orange balls
            ([10, 100, 100], [15, 255, 255], "orange"),
        ]

        self.get_logger().info('Tennis Ball Detector initialized!')
        self.get_logger().info('Looking for green tennis balls (macadamia nuts) only')
        # self.get_logger().info('Will reject other colored balls with "Not a macadamia nut" message')

    def scan_callback(self, msg):
        """Store LiDAR scan data."""
        self.scan = msg

    def image_callback(self, msg):
        """Main image processing callback with enhanced logging."""
        if self.scan is None:
            self.get_logger().warn('Waiting for LiDAR scan data...')
            return

        try:
            self.frame_count += 1
            
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Check for green tennis balls (macadamia nuts)
            green_ball = self.detect_green_tennis_ball(cv_image)
            
            if green_ball:
                # Process the green ball (macadamia nut)
                self.process_macadamia_nut(cv_image, green_ball)
            else:
                # Check for other colored balls to reject
                other_ball = self.detect_other_colored_balls(cv_image)
                if other_ball:
                    color_name = other_ball['color']
                    self.get_logger().info(f'Not a macadamia nut - detected (ignoring)')
                
                # Log scanning status occasionally
                elif self.frame_count % 120 == 0:  # Every ~4 seconds
                    self.get_logger().info('Scanning for green tennis balls (macadamia nuts)...')
                    
        except Exception as e:
            self.get_logger().error(f'Error in tennis ball detection: {str(e)}')

    def detect_green_tennis_ball(self, cv_image):
        """Detect green tennis balls (macadamia nuts)."""
        # Convert to HSV color space
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # Create mask for green tennis balls
        mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        
        # Morphology to reduce noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None

        # Find largest contour (assumed to be the tennis ball)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area < 200:  # Filter small blobs
            return None

        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        center_x = x + w / 2
        center_y = y + h / 2
        
        return {
            'contour': largest_contour,
            'center_x': center_x,
            'center_y': center_y,
            'area': area,
            'bbox': (x, y, w, h)
        }

    def detect_other_colored_balls(self, cv_image):
        """Detect other colored balls to reject them."""
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        for lower, upper, color_name in self.other_color_ranges:
            lower_np = np.array(lower)
            upper_np = np.array(upper)
            
            # Create mask for this color
            mask = cv2.inRange(hsv, lower_np, upper_np)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                if area > 200:  # Same size threshold as green balls
                    return {'color': color_name, 'area': area}
        
        return None

    def process_macadamia_nut(self, cv_image, ball_data):
        """Process detected green tennis ball (macadamia nut) with detailed logging."""
        center_x = ball_data['center_x']
        center_y = ball_data['center_y']
        area = ball_data['area']
        
        # Get world coordinates
        world_coords = self.get_world_coordinates(center_x, cv_image.shape[1])
        
        if world_coords:
            lidar_x, lidar_y, map_x, map_y = world_coords
            
            # Enhanced logging (similar to tree detector)
            self.detection_count += 1
            self.get_logger().info('MACADAMIA NUT DETECTED! (Green tennis ball)')
            self.get_logger().info(f'  Detection #{self.detection_count}:')
            self.get_logger().info(f'  Pixel coordinates: ({int(center_x)}, {int(center_y)})')
            self.get_logger().info(f'  LiDAR coordinates: ({lidar_x:.2f}, {lidar_y:.2f}) meters')
            self.get_logger().info(f'  Map coordinates: ({map_x:.2f}, {map_y:.2f}) meters')
            # self.get_logger().info(f'   Ball area: {area}px²')
            
            # Build point for map transformation
            point_lidar = PointStamped()
            point_lidar.header.frame_id = self.scan.header.frame_id
            point_lidar.header.stamp = self.scan.header.stamp
            point_lidar.point.x = lidar_x
            point_lidar.point.y = lidar_y
            point_lidar.point.z = 0.0

            # Transform to map frame and publish marker if new
            self.transform_and_publish(point_lidar)

    def get_world_coordinates(self, center_x, image_width):
        """Convert pixel coordinates to world coordinates."""
        try:
            # Normalize horizontal center_x to [0,1]
            normalized_x = center_x / image_width

            # Compute angle from LiDAR scan parameters
            angle = self.scan.angle_min + normalized_x * (self.scan.angle_max - self.scan.angle_min)

            # Get LiDAR range index
            index = int((angle - self.scan.angle_min) / self.scan.angle_increment)
            if index < 0 or index >= len(self.scan.ranges):
                self.get_logger().warn(f'LiDAR index {index} out of range')
                return None

            distance = self.scan.ranges[index]
            if not np.isfinite(distance) or not (self.scan.range_min <= distance <= self.scan.range_max):
                self.get_logger().warn(f'Invalid LiDAR distance: {distance:.2f}')
                return None

            # Calculate LiDAR coordinates
            lidar_x = distance * np.cos(angle)
            lidar_y = distance * np.sin(angle)
            
            # Build point in LiDAR frame
            point_lidar = PointStamped()
            point_lidar.header.frame_id = self.scan.header.frame_id
            point_lidar.header.stamp = self.scan.header.stamp
            point_lidar.point.x = lidar_x
            point_lidar.point.y = lidar_y
            point_lidar.point.z = 0.0
            
            # Transform to map coordinates
            map_coords = self.transform_to_map(point_lidar)
            if map_coords:
                map_x, map_y = map_coords
                return (lidar_x, lidar_y, map_x, map_y)
            else:
                return (lidar_x, lidar_y, None, None)
                
        except Exception as e:
            self.get_logger().error(f'Error getting world coordinates: {str(e)}')
            return None

    def transform_to_map(self, point_lidar):
        """Transform point from LiDAR frame to map frame."""
        try:
            timeout = rclpy.duration.Duration(seconds=0.1)
            if self.tf_buffer.can_transform('map', point_lidar.header.frame_id, rclpy.time.Time(), timeout=timeout):
                point_map = tf2_geometry_msgs.do_transform_point(
                    point_lidar,
                    self.tf_buffer.lookup_transform('map', point_lidar.header.frame_id, rclpy.time.Time(), timeout=timeout)
                )
                return (point_map.point.x, point_map.point.y)
            else:
                self.get_logger().warn('TF transform not ready')
                return None
        except Exception as e:
            self.get_logger().error(f'TF exception: {e}')
            return None

    def is_new_ball(self, new_point, threshold=0.3):
        """Check if this is a new ball or previously detected."""
        for point in self.detected_points:
            dx = new_point.point.x - point.point.x
            dy = new_point.point.y - point.point.y
            if np.hypot(dx, dy) < threshold:
                return False
        return True

    def transform_and_publish(self, point_lidar):
        """Transform to map frame and publish marker if new ball."""
        try:
            timeout = rclpy.duration.Duration(seconds=0.1)
            if self.tf_buffer.can_transform('map', point_lidar.header.frame_id, rclpy.time.Time(), timeout=timeout):
                point_map = tf2_geometry_msgs.do_transform_point(
                    point_lidar,
                    self.tf_buffer.lookup_transform('map', point_lidar.header.frame_id, rclpy.time.Time(), timeout=timeout)
                )

                if self.is_new_ball(point_map):
                    self.publish_marker(point_map)
                    self.detected_points.append(point_map)
                    self.get_logger().info(f'New macadamia nut confirmed at map coordinates ({point_map.point.x:.2f}, {point_map.point.y:.2f})')
                    self.get_logger().info(f'Published marker #{len(self.detected_points)} in RViz')
                    
                    # Trigger home after detection
                    self.home_trigger_pub.publish(Empty())
                    self.get_logger().info('Triggered home return sequence')
                else:
                    self.get_logger().info('Previously detected macadamia nut - ignoring duplicate')
            else:
                self.get_logger().warn('TF transform not ready')
        except Exception as e:
            self.get_logger().error(f'TF exception: {e}')

    def publish_marker(self, point_map):
        """Publish marker for RViz visualization."""
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'macadamia_nuts'
        marker.id = len(self.detected_points)  # Unique ID per nut
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position = point_map.point
        marker.pose.orientation.w = 1.0

        # Green sphere for macadamia nuts
        marker.scale.x = marker.scale.y = marker.scale.z = 0.15
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.9

        self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    
    try:
        detector = TennisBallDetector()
        detector.get_logger().info('Starting enhanced tennis ball detector...')
        detector.get_logger().info('GREEN tennis balls will be processed as macadamia nuts')
        detector.get_logger().info('Other colored balls are ignored')
        rclpy.spin(detector)
    except KeyboardInterrupt:
        detector.get_logger().info('Enhanced tennis ball detector shutting down...')
    finally:
        if 'detector' in locals():
            detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
