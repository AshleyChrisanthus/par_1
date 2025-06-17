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

        # NEW (white table‑tennis balls: very low saturation, high value)
        # self.green_lower = np.array([0, 0, 200])     # H can be anything, S≈0, V high
        # self.green_upper = np.array([180, 30, 255])  # allow a little saturation        
        
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

    def scan_callback(self, msg):
        """Store LiDAR scan data."""
        self.scan = msg

    def image_callback(self, msg):
        """Main image processing callback with multi-ball detection."""
        if self.scan is None:
            self.get_logger().warn('Waiting for LiDAR scan data...')
            return

        try:
            self.frame_count += 1
            
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Detect ALL green tennis balls in current frame
            green_balls = self.detect_all_green_tennis_balls(cv_image)
            
            if green_balls:
                self.get_logger().info(f'Found {len(green_balls)} green ball(s) in frame #{self.frame_count}')
                
                # Process each detected ball
                new_balls_found = 0
                for i, ball_data in enumerate(green_balls):
                    self.get_logger().info(f'Processing ball {i+1}/{len(green_balls)}:')
                    if self.process_macadamia_nut(cv_image, ball_data, i+1):
                        new_balls_found += 1
                
                if new_balls_found > 0:
                    self.get_logger().info(f'=== SUMMARY: {new_balls_found} NEW ball(s) added, {len(green_balls) - new_balls_found} duplicate(s) ignored ===')
                else:
                    self.get_logger().info('=== SUMMARY: All balls were previously detected - no new balls added ===')
            else:
                # Check for other colored balls to reject
                other_ball = self.detect_other_colored_balls(cv_image)
                if other_ball:
                    color_name = other_ball['color']
                    self.get_logger().info(f'Not a macadamia nut - detected {color_name} ball (ignoring)')
                
                # Log scanning status occasionally
                elif self.frame_count % 120 == 0:  # Every ~4 seconds
                    self.get_logger().info('Scanning for green tennis balls (macadamia nuts)...')
                    
        except Exception as e:
            self.get_logger().error(f'Error in tennis ball detection: {str(e)}')

    def detect_all_green_tennis_balls(self, cv_image):
        """Detect ALL green tennis balls (macadamia nuts) in the frame."""
        # Convert to HSV color space
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # Create mask for green tennis balls
        mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        
        # Morphology to reduce noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []

        detected_balls = []
        
        # Process ALL contours above minimum area threshold
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < 100:  # Filter very small blobs
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w / 2
            center_y = y + h / 2
            
            ball_data = {
                'contour': contour,
                'center_x': center_x,
                'center_y': center_y,
                'area': area,
                'bbox': (x, y, w, h)
            }
            
            detected_balls.append(ball_data)
        
        # Sort by area (largest first) for consistent processing order
        detected_balls.sort(key=lambda x: x['area'], reverse=True)
        
        return detected_balls

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

    def process_macadamia_nut(self, cv_image, ball_data, ball_number):
        """Process detected green tennis ball (macadamia nut) with detailed logging."""
        center_x = ball_data['center_x']
        center_y = ball_data['center_y']
        area = ball_data['area']
        
        # Get world coordinates
        world_coords = self.get_world_coordinates(center_x, cv_image.shape[1])
        
        if world_coords:
            lidar_x, lidar_y, map_x, map_y = world_coords
            
            # Build point for duplicate checking
            point_lidar = PointStamped()
            point_lidar.header.frame_id = self.scan.header.frame_id
            point_lidar.header.stamp = self.scan.header.stamp
            point_lidar.point.x = lidar_x
            point_lidar.point.y = lidar_y
            point_lidar.point.z = 0.0

            # Transform to map coordinates for duplicate checking
            map_coords_full = self.transform_to_map(point_lidar)
            if map_coords_full:
                point_map = PointStamped()
                point_map.header.frame_id = 'map'
                point_map.header.stamp = self.scan.header.stamp
                point_map.point.x = map_coords_full[0]
                point_map.point.y = map_coords_full[1]
                point_map.point.z = 0.0
                
                # Check if this is a new ball
                if self.is_new_ball(point_map):
                    # This is a NEW ball - log and process it
                    self.detection_count += 1
                    
                    self.get_logger().info(f'  ✓ MACADAMIA NUT #{ball_number} - NEW DETECTION!')
                    self.get_logger().info(f'    Detection ID: #{self.detection_count}')
                    self.get_logger().info(f'    Pixel coords: ({int(center_x)}, {int(center_y)})')
                    self.get_logger().info(f'    LiDAR coords: ({lidar_x:.2f}, {lidar_y:.2f}) meters')
                    self.get_logger().info(f'    Map coords: ({map_coords_full[0]:.2f}, {map_coords_full[1]:.2f}) meters')
                    self.get_logger().info(f'    Ball area: {area}px²')
                    
                    # Add to detected points and publish marker
                    self.detected_points.append(point_map)
                    self.publish_marker(point_map)
                    self.get_logger().info(f'    Published marker #{len(self.detected_points)} in RViz')
                    
                    # Trigger home after detection (optional - might want to disable for multiple balls)
                    # self.home_trigger_pub.publish(Empty())
                    # self.get_logger().info('    Triggered home return sequence')
                    
                    return True  # New ball found
                else:
                    # This is a DUPLICATE ball
                    self.get_logger().info(f'  ✗ Ball #{ball_number} - DUPLICATE (already detected)')
                    self.get_logger().info(f'    Pixel coords: ({int(center_x)}, {int(center_y)})')
                    self.get_logger().info(f'    Map coords: ({map_coords_full[0]:.2f}, {map_coords_full[1]:.2f}) meters')
                    self.get_logger().info(f'    Ignoring duplicate detection')
                    
                    return False  # Duplicate ball
            else:
                self.get_logger().warn(f'  ! Ball #{ball_number} - Could not transform to map coordinates')
                return False
        else:
            self.get_logger().warn(f'  ! Ball #{ball_number} - Could not get world coordinates')
            return False

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
            distance = np.hypot(dx, dy)
            if distance < threshold:
                return False
        return True

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
        detector.get_logger().info('Starting enhanced MULTI-BALL tennis ball detector...')
        detector.get_logger().info('GREEN tennis balls will be processed as macadamia nuts')
        detector.get_logger().info('Multiple balls per frame supported - duplicates will be ignored')
        detector.get_logger().info('Other colored balls are ignored')
        rclpy.spin(detector)
    except KeyboardInterrupt:
        detector.get_logger().info('Enhanced multi-ball tennis ball detector shutting down...')
    finally:
        if 'detector' in locals():
            detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
