#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty, Int32
from sensor_msgs.msg import Image, LaserScan, CameraInfo
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
        self.camera_info = None
        self.camera_matrix = None
        self.dist_coeffs = None

        # Subscribers
        self.sub_image = self.create_subscription(Image, '/oak/rgb/image_raw', self.image_callback, 10)
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.sub_camera_info = self.create_subscription(CameraInfo, '/oak/rgb/camera_info', self.camera_info_callback, 10)

        # Publishers
        self.marker_pub = self.create_publisher(Marker, '/tennis_ball_marker', 10)
        self.home_trigger_pub = self.create_publisher(Empty, '/trigger_home', 10)
        self.count_marker_pub = self.create_publisher(Marker, '/tennis_ball_count_marker', 10)
        self.count_pub = self.create_publisher(Int32, '/tennis_ball_count', 10)

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

        # Camera frame name - adjust if different
        self.camera_frame = 'oak_rgb_camera_optical_frame'
        self.lidar_frame = 'laser'  # Adjust based on your setup

        # Create timer for periodic status updates
        self.status_timer = self.create_timer(5.0, self.publish_status)  # Every 5 seconds

        self.get_logger().info('Tennis Ball Detector initialized!')
        self.get_logger().info('Looking for green tennis balls (macadamia nuts) only')
        
        # Publish initial count marker
        self.publish_count_marker()

    def camera_info_callback(self, msg):
        """Store camera calibration data."""
        self.camera_info = msg
        # Extract camera matrix
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d)
        self.get_logger().info('Camera calibration data received')

    def scan_callback(self, msg):
        """Store LiDAR scan data."""
        self.scan = msg

    def image_callback(self, msg):
        """Main image processing callback with enhanced logging."""
        if self.scan is None:
            self.get_logger().warn('Waiting for LiDAR scan data...')
            return
            
        if self.camera_info is None:
            self.get_logger().warn('Waiting for camera calibration data...')
            return

        try:
            self.frame_count += 1
            
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Check for green tennis balls (macadamia nuts)
            green_ball = self.detect_green_tennis_ball(cv_image)
            
            if green_ball:
                # Process the green ball (macadamia nut)
                self.process_macadamia_nut(cv_image, green_ball, msg.header)
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

    def process_macadamia_nut(self, cv_image, ball_data, image_header):
        """Process detected green tennis ball (macadamia nut) with proper coordinate transformation."""
        center_x = ball_data['center_x']
        center_y = ball_data['center_y']
        area = ball_data['area']
        
        # Get world coordinates using proper transformation
        world_coords = self.get_world_coordinates_v2(center_x, center_y, image_header)
        
        if world_coords:
            lidar_x, lidar_y, map_x, map_y = world_coords
            
            # Enhanced logging
            self.detection_count += 1
            self.get_logger().info('MACADAMIA NUT DETECTED! (Green tennis ball)')
            self.get_logger().info(f'  Detection #{self.detection_count}:')
            self.get_logger().info(f'  Pixel coordinates: ({int(center_x)}, {int(center_y)})')
            self.get_logger().info(f'  LiDAR coordinates: ({lidar_x:.2f}, {lidar_y:.2f}) meters')
            self.get_logger().info(f'  Map coordinates: ({map_x:.2f}, {map_y:.2f}) meters')
            
            # Build point for map transformation
            point_lidar = PointStamped()
            point_lidar.header.frame_id = self.scan.header.frame_id
            point_lidar.header.stamp = self.scan.header.stamp
            point_lidar.point.x = lidar_x
            point_lidar.point.y = lidar_y
            point_lidar.point.z = 0.0

            # Transform to map frame and publish marker if new
            self.transform_and_publish(point_lidar)

    def get_world_coordinates_v2(self, pixel_x, pixel_y, image_header):
        """Convert pixel coordinates to world coordinates using proper camera-lidar transformation."""
        try:
            # Step 1: Get the transform from camera to lidar frame
            timeout = rclpy.duration.Duration(seconds=0.1)
            
            # Try to get camera frame from header first
            camera_frame = image_header.frame_id if image_header.frame_id else self.camera_frame
            
            # Get transform from camera to lidar
            try:
                camera_to_lidar = self.tf_buffer.lookup_transform(
                    self.scan.header.frame_id,  # target frame (lidar)
                    camera_frame,  # source frame (camera)
                    rclpy.time.Time(),
                    timeout=timeout
                )
            except:
                self.get_logger().warn(f'Could not get transform from {camera_frame} to {self.scan.header.frame_id}')
                return None

            # Step 2: Create a ray from camera through the pixel
            # Normalize pixel coordinates using camera intrinsics
            fx = self.camera_matrix[0, 0]
            fy = self.camera_matrix[1, 1]
            cx = self.camera_matrix[0, 2]
            cy = self.camera_matrix[1, 2]
            
            # Ray direction in camera frame (normalized)
            ray_camera_x = (pixel_x - cx) / fx
            ray_camera_y = (pixel_y - cy) / fy
            ray_camera_z = 1.0
            
            # Normalize the ray
            ray_length = np.sqrt(ray_camera_x**2 + ray_camera_y**2 + ray_camera_z**2)
            ray_camera_x /= ray_length
            ray_camera_y /= ray_length
            ray_camera_z /= ray_length
            
            # Create point in camera frame (we'll project this onto the lidar plane)
            point_camera = PointStamped()
            point_camera.header.frame_id = camera_frame
            point_camera.header.stamp = image_header.stamp
            point_camera.point.x = ray_camera_x
            point_camera.point.y = ray_camera_y
            point_camera.point.z = ray_camera_z
            
            # Transform ray to lidar frame
            point_lidar_ray = tf2_geometry_msgs.do_transform_point(point_camera, camera_to_lidar)
            
            # Get camera position in lidar frame
            camera_origin = PointStamped()
            camera_origin.header.frame_id = camera_frame
            camera_origin.header.stamp = image_header.stamp
            camera_origin.point.x = 0.0
            camera_origin.point.y = 0.0
            camera_origin.point.z = 0.0
            camera_origin_lidar = tf2_geometry_msgs.do_transform_point(camera_origin, camera_to_lidar)
            
            # Step 3: Find intersection with lidar scan plane
            # Calculate the angle of the ray in the lidar's horizontal plane
            ray_x = point_lidar_ray.point.x - camera_origin_lidar.point.x
            ray_y = point_lidar_ray.point.y - camera_origin_lidar.point.y
            
            # Calculate angle in lidar frame
            angle = np.arctan2(ray_y, ray_x)
            
            # Get the corresponding lidar reading
            if angle < self.scan.angle_min or angle > self.scan.angle_max:
                self.get_logger().warn(f'Angle {angle:.2f} outside lidar range [{self.scan.angle_min:.2f}, {self.scan.angle_max:.2f}]')
                return None
            
            # Get LiDAR range index
            index = int((angle - self.scan.angle_min) / self.scan.angle_increment)
            if index < 0 or index >= len(self.scan.ranges):
                self.get_logger().warn(f'LiDAR index {index} out of range')
                return None

            distance = self.scan.ranges[index]
            if not np.isfinite(distance) or not (self.scan.range_min <= distance <= self.scan.range_max):
                self.get_logger().warn(f'Invalid LiDAR distance: {distance:.2f}')
                return None

            # Calculate actual position in lidar frame
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
                    self.get_logger().info(f'=== TOTAL TENNIS BALLS DETECTED: {len(self.detected_points)} ===')
                    
                    # Update count display
                    self.publish_count_marker()
                    
                    # Trigger home after detection
                    self.home_trigger_pub.publish(Empty())
                    self.get_logger().info('Triggered home return sequence')
                else:
                    self.get_logger().info('Previously detected macadamia nut - ignoring duplicate')
                    self.get_logger().info(f'Current total: {len(self.detected_points)} tennis balls')
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

    def publish_status(self):
        """Publish periodic status update about detected tennis balls."""
        total_balls = len(self.detected_points)
        if total_balls > 0:
            self.get_logger().info(f'===== DETECTION STATUS =====')
            self.get_logger().info(f'Total Tennis Balls Found: {total_balls}')
            self.get_logger().info(f'Total Detections Processed: {self.detection_count}')
            self.get_logger().info(f'Frames Analyzed: {self.frame_count}')
            
            # List positions
            for i, point in enumerate(self.detected_points):
                self.get_logger().info(f'  Ball #{i+1}: ({point.point.x:.2f}, {point.point.y:.2f})')
            self.get_logger().info(f'===========================')
        else:
            self.get_logger().info(f'No tennis balls detected yet. Frames analyzed: {self.frame_count}')

    def publish_count_marker(self):
        """Publish a text marker showing the total count of tennis balls."""
        # Publish count as Int32 message
        count_msg = Int32()
        count_msg.data = len(self.detected_points)
        self.count_pub.publish(count_msg)
        
        # Publish visual count marker
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'tennis_ball_count'
        marker.id = 9999  # Fixed ID for count marker
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        # Position the text above the robot's typical operating area
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 2.0  # 2 meters high
        marker.pose.orientation.w = 1.0

        # Text content
        marker.text = f"Tennis Balls: {len(self.detected_points)}"

        # Text appearance
        marker.scale.z = 0.5  # Text size
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        self.count_marker_pub.publish(marker)

        # Also publish individual numbered markers for each ball
        for i, point in enumerate(self.detected_points):
            text_marker = Marker()
            text_marker.header.frame_id = 'map'
            text_marker.header.stamp = self.get_clock().now().to_msg()
            text_marker.ns = 'tennis_ball_numbers'
            text_marker.id = i + 1000  # Offset to avoid ID conflicts
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD

            # Position text above the ball
            text_marker.pose.position.x = point.point.x
            text_marker.pose.position.y = point.point.y
            text_marker.pose.position.z = 0.3  # 30cm above ground
            text_marker.pose.orientation.w = 1.0

            # Number text
            text_marker.text = f"#{i+1}"

            # Text appearance
            text_marker.scale.z = 0.15  # Smaller text for individual balls
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 0.0
            text_marker.color.a = 1.0

            self.marker_pub.publish(text_marker)

def main(args=None):
    rclpy.init(args=args)
    
    try:
        detector = TennisBallDetector()
        detector.get_logger().info('Starting enhanced tennis ball detector with counting...')
        detector.get_logger().info('GREEN tennis balls will be processed as macadamia nuts')
        detector.get_logger().info('Other colored balls are ignored')
        detector.get_logger().info('Features:')
        detector.get_logger().info('  - Real-time count display in RViz')
        detector.get_logger().info('  - Individual ball numbering')
        detector.get_logger().info('  - Count published on /tennis_ball_count topic')
        detector.get_logger().info('  - Status updates every 5 seconds')
        rclpy.spin(detector)
    except KeyboardInterrupt:
        detector.get_logger().info('Enhanced tennis ball detector shutting down...')
        detector.get_logger().info(f'Final count: {len(detector.detected_points)} tennis balls detected')
    finally:
        if 'detector' in locals():
            detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
