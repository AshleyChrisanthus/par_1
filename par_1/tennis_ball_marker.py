#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty
from sensor_msgs.msg import Image, LaserScan, CameraInfo
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PointStamped
import tf2_ros
import tf2_geometry_msgs
import cv2
from cv_bridge import CvBridge
import numpy as np
from collections import deque

class TennisBallMarker(Node):
    def __init__(self):
        super().__init__('tennis_ball_marker')

        self.bridge = CvBridge()
        self.scan = None
        self.camera_info = None

        # Declare parameters for easy tuning
        self.declare_parameter('camera_lidar_yaw_offset', 0.0)  # radians
        self.declare_parameter('ball_radius_m', 0.033)  # tennis ball radius
        self.declare_parameter('min_ball_area_pixels', 200)
        self.declare_parameter('detection_threshold_m', 0.3)
        self.declare_parameter('camera_height_m', 0.15)  # camera height above ground
        self.declare_parameter('camera_pitch_rad', 0.0)  # camera pitch angle
        self.declare_parameter('temporal_filter_size', 5)
        self.declare_parameter('lidar_samples', 5)  # number of lidar rays to sample

        # Get parameters
        self.camera_lidar_offset = self.get_parameter('camera_lidar_yaw_offset').value
        self.ball_radius = self.get_parameter('ball_radius_m').value
        self.min_ball_area = self.get_parameter('min_ball_area_pixels').value
        self.detection_threshold = self.get_parameter('detection_threshold_m').value
        self.camera_height = self.get_parameter('camera_height_m').value
        self.camera_pitch = self.get_parameter('camera_pitch_rad').value
        self.filter_size = self.get_parameter('temporal_filter_size').value
        self.lidar_samples = self.get_parameter('lidar_samples').value

        # Subscribers
        self.sub_image = self.create_subscription(Image, '/oak/rgb/image_raw', self.image_callback, 10)
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.sub_camera_info = self.create_subscription(CameraInfo, '/oak/rgb/camera_info', self.camera_info_callback, 10)

        # Publishers
        self.marker_pub = self.create_publisher(Marker, '/tennis_ball_marker', 10)
        self.home_trigger_pub = self.create_publisher(Empty, '/trigger_home', 10)
        self.debug_image_pub = self.create_publisher(Image, '/tennis_ball_debug', 10)

        # TF setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Detection tracking
        self.detected_points = []  # Store detected tennis ball positions
        self.detection_count = 0
        self.frame_count = 0
        
        # Temporal filtering
        self.detection_buffer = deque(maxlen=self.filter_size)

        # Camera intrinsics (will be updated from camera_info)
        self.fx = 554.25  # default focal length x
        self.fy = 554.25  # default focal length y
        self.cx = 320.5   # default principal point x
        self.cy = 240.5   # default principal point y

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

        self.get_logger().info('Enhanced Tennis Ball Detector initialized!')
        self.get_logger().info(f'Camera-LiDAR offset: {self.camera_lidar_offset:.3f} rad')
        self.get_logger().info(f'Using {self.lidar_samples} LiDAR samples per detection')
        self.get_logger().info(f'Temporal filter size: {self.filter_size}')

    def camera_info_callback(self, msg):
        """Update camera intrinsics from camera info."""
        if self.camera_info is None:
            self.camera_info = msg
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.get_logger().info(f'Camera intrinsics updated: fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f}')

    def scan_callback(self, msg):
        """Store LiDAR scan data."""
        self.scan = msg

    def image_callback(self, msg):
        """Main image processing callback with enhanced logging."""
        if self.scan is None:
            if self.frame_count % 30 == 0:
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
                
                # Publish debug image
                self.publish_debug_image(cv_image, green_ball)
            else:
                # Check for other colored balls to reject
                other_ball = self.detect_other_colored_balls(cv_image)
                if other_ball:
                    self.get_logger().info(f'Not a macadamia nut - {other_ball["color"]} ball detected (ignoring)')
                
                # Log scanning status occasionally
                elif self.frame_count % 120 == 0:  # Every ~4 seconds
                    self.get_logger().info('Scanning for green tennis balls (macadamia nuts)...')
                    
        except Exception as e:
            self.get_logger().error(f'Error in tennis ball detection: {str(e)}')

    def detect_green_tennis_ball(self, cv_image):
        """Detect green tennis balls using circle detection."""
        # Convert to HSV color space
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # Create mask for green tennis balls
        mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        
        # Morphology to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Blur for better circle detection
        mask_blurred = cv2.GaussianBlur(mask, (9, 9), 2)
        
        # Try circle detection first
        circles = cv2.HoughCircles(
            mask_blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=100
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            # Use the largest circle
            largest_circle = max(circles[0], key=lambda c: c[2])
            center_x, center_y, radius = largest_circle
            
            # Verify it's in the green region
            if mask[int(center_y), int(center_x)] > 0:
                return {
                    'center_x': float(center_x),
                    'center_y': float(center_y),
                    'radius': float(radius),
                    'area': np.pi * radius * radius,
                    'method': 'circle'
                }
        
        # Fallback to contour detection if circle detection fails
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None

        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area < self.min_ball_area:
            return None

        # Get circular approximation
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        
        # Check circularity
        perimeter = cv2.arcLength(largest_contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        if circularity > 0.7:  # Reasonably circular
            return {
                'center_x': float(x),
                'center_y': float(y),
                'radius': float(radius),
                'area': area,
                'circularity': circularity,
                'method': 'contour'
            }
        
        return None

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
                
                if area > self.min_ball_area:
                    return {'color': color_name, 'area': area}
        
        return None

    def process_macadamia_nut(self, cv_image, ball_data):
        """Process detected green tennis ball with enhanced accuracy."""
        center_x = ball_data['center_x']
        center_y = ball_data['center_y']
        radius = ball_data.get('radius', 20)
        
        # Get world coordinates with multi-point sampling
        world_coords = self.get_world_coordinates_multipoint(
            center_x, center_y, radius, cv_image.shape
        )
        
        if world_coords:
            # Apply temporal filtering
            filtered_coords = self.apply_temporal_filter(world_coords)
            
            if filtered_coords:
                lidar_x, lidar_y, lidar_z, map_x, map_y, map_z = filtered_coords
                
                # Enhanced logging
                self.detection_count += 1
                self.get_logger().info('='*50)
                self.get_logger().info('MACADAMIA NUT DETECTED! (Green tennis ball)')
                self.get_logger().info(f'  Detection #{self.detection_count}:')
                self.get_logger().info(f'  Detection method: {ball_data.get("method", "unknown")}')
                self.get_logger().info(f'  Pixel coordinates: ({int(center_x)}, {int(center_y)})')
                self.get_logger().info(f'  Radius: {radius:.1f} pixels')
                if 'circularity' in ball_data:
                    self.get_logger().info(f'  Circularity: {ball_data["circularity"]:.3f}')
                self.get_logger().info(f'  LiDAR coordinates: ({lidar_x:.3f}, {lidar_y:.3f}, {lidar_z:.3f}) meters')
                self.get_logger().info(f'  Map coordinates: ({map_x:.3f}, {map_y:.3f}, {map_z:.3f}) meters')
                
                # Build point for map transformation
                point_map = PointStamped()
                point_map.header.frame_id = 'map'
                point_map.header.stamp = self.scan.header.stamp
                point_map.point.x = map_x
                point_map.point.y = map_y
                point_map.point.z = map_z

                # Check if new and publish
                if self.is_new_ball(point_map):
                    self.publish_marker(point_map)
                    self.detected_points.append(point_map)
                    self.get_logger().info(f'New macadamia nut confirmed!')
                    self.get_logger().info(f'Published marker #{len(self.detected_points)} in RViz')
                    
                    # Trigger home after detection
                    self.home_trigger_pub.publish(Empty())
                    self.get_logger().info('Triggered home return sequence')
                else:
                    self.get_logger().info('Previously detected macadamia nut - ignoring duplicate')

    def get_world_coordinates_multipoint(self, center_x, center_y, radius, image_shape):
        """Convert pixel coordinates to world coordinates with multi-point sampling."""
        try:
            image_width, image_height = image_shape[1], image_shape[0]
            
            # Convert pixel to camera angle using intrinsics
            angle_from_camera = np.arctan((center_x - self.cx) / self.fx)
            
            # Apply camera-LiDAR calibration offset
            base_angle = angle_from_camera + self.camera_lidar_offset
            
            # Sample multiple LiDAR points around the ball
            valid_measurements = []
            sample_offsets = np.linspace(-0.5, 0.5, self.lidar_samples)
            
            for offset in sample_offsets:
                # Adjust angle based on ball radius
                angle_offset = np.arctan(offset * radius / self.fx)
                sample_angle = base_angle + angle_offset
                
                # Get LiDAR range at this angle
                if self.scan.angle_min <= sample_angle <= self.scan.angle_max:
                    index = int((sample_angle - self.scan.angle_min) / self.scan.angle_increment)
                    
                    if 0 <= index < len(self.scan.ranges):
                        distance = self.scan.ranges[index]
                        
                        if np.isfinite(distance) and self.scan.range_min <= distance <= self.scan.range_max:
                            valid_measurements.append((sample_angle, distance))
            
            if not valid_measurements:
                self.get_logger().warn('No valid LiDAR measurements found')
                return None
            
            # Use the closest valid distance (most likely to be the ball)
            closest_measurement = min(valid_measurements, key=lambda x: x[1])
            angle, distance = closest_measurement
            
            # Account for ball radius - adjust distance to ball center
            adjusted_distance = distance - self.ball_radius
            
            # Calculate LiDAR coordinates
            lidar_x = adjusted_distance * np.cos(angle)
            lidar_y = adjusted_distance * np.sin(angle)
            
            # Estimate height based on camera model
            lidar_z = self.estimate_ball_height(center_y, image_height, adjusted_distance)
            
            # Transform to map frame
            point_lidar = PointStamped()
            point_lidar.header.frame_id = self.scan.header.frame_id
            point_lidar.header.stamp = self.scan.header.stamp
            point_lidar.point.x = lidar_x
            point_lidar.point.y = lidar_y
            point_lidar.point.z = lidar_z
            
            map_coords = self.transform_to_map(point_lidar)
            if map_coords:
                map_x, map_y, map_z = map_coords
                return (lidar_x, lidar_y, lidar_z, map_x, map_y, map_z)
            else:
                return None
                
        except Exception as e:
            self.get_logger().error(f'Error in multipoint world coordinates: {str(e)}')
            return None

    def estimate_ball_height(self, center_y, image_height, distance):
        """Estimate ball height based on camera model."""
        # Normalized vertical position (0 = top, 1 = bottom)
        normalized_y = center_y / image_height
        
        # Camera ray angle in vertical direction
        vertical_angle = np.arctan((center_y - self.cy) / self.fy) + self.camera_pitch
        
        # Estimate ground distance and height
        ground_distance = distance * np.cos(vertical_angle)
        height_offset = distance * np.sin(vertical_angle)
        
        # Ball center height (camera height + offset - ball on ground)
        estimated_height = self.camera_height + height_offset - self.ball_radius
        
        # Clamp to reasonable values (ball should be near ground)
        estimated_height = np.clip(estimated_height, 0, 2 * self.ball_radius)
        
        return estimated_height

    def apply_temporal_filter(self, world_coords):
        """Apply temporal filtering to reduce noise."""
        self.detection_buffer.append(world_coords)
        
        if len(self.detection_buffer) < 3:  # Need at least 3 samples
            return world_coords
        
        # Extract coordinates from buffer
        coords_array = np.array(list(self.detection_buffer))
        
        # Calculate median for each coordinate
        median_coords = np.median(coords_array, axis=0)
        
        # Check if current detection is within reasonable range of median
        current = np.array(world_coords)
        distance = np.linalg.norm(current[:3] - median_coords[:3])  # Use LiDAR coords
        
        if distance < 0.15:  # Within 15cm of median
            # Return filtered coordinates
            return tuple(median_coords)
        else:
            # Detection is too far from median, might be noise
            self.get_logger().warn(f'Detection {distance:.3f}m from median - possible noise')
            return None

    def transform_to_map(self, point_lidar):
        """Transform point from LiDAR frame to map frame."""
        try:
            timeout = rclpy.duration.Duration(seconds=0.1)
            if self.tf_buffer.can_transform('map', point_lidar.header.frame_id, 
                                           rclpy.time.Time(), timeout=timeout):
                transform = self.tf_buffer.lookup_transform(
                    'map', point_lidar.header.frame_id, 
                    rclpy.time.Time(), timeout=timeout
                )
                point_map = tf2_geometry_msgs.do_transform_point(point_lidar, transform)
                return (point_map.point.x, point_map.point.y, point_map.point.z)
            else:
                self.get_logger().warn('TF transform not ready')
                return None
        except Exception as e:
            self.get_logger().error(f'TF exception: {e}')
            return None

    def is_new_ball(self, new_point):
        """Check if this is a new ball or previously detected."""
        for point in self.detected_points:
            dx = new_point.point.x - point.point.x
            dy = new_point.point.y - point.point.y
            dz = new_point.point.z - point.point.z
            distance = np.sqrt(dx*dx + dy*dy + dz*dz)
            
            if distance < self.detection_threshold:
                return False
        return True

    def publish_marker(self, point_map):
        """Publish marker for RViz visualization."""
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'macadamia_nuts'
        marker.id = len(self.detected_points)
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position = point_map.point
        marker.pose.orientation.w = 1.0

        # Green sphere for macadamia nuts
        marker.scale.x = marker.scale.y = marker.scale.z = 2 * self.ball_radius
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.9

        # Add text label
        text_marker = Marker()
        text_marker.header = marker.header
        text_marker.ns = 'macadamia_labels'
        text_marker.id = marker.id
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.pose = marker.pose
        text_marker.pose.position.z += 0.1  # Above the sphere
        text_marker.text = f"Nut #{marker.id + 1}"
        text_marker.scale.z = 0.05
        text_marker.color.r = 1.0
        text_marker.color.g = 1.0
        text_marker.color.b = 1.0
        text_marker.color.a = 1.0

        self.marker_pub.publish(marker)
        self.marker_pub.publish(text_marker)

    def publish_debug_image(self, cv_image, ball_data):
        """Publish debug image showing detection."""
        debug_image = cv_image.copy()
        
        # Draw detection
        center = (int(ball_data['center_x']), int(ball_data['center_y']))
        radius = int(ball_data.get('radius', 20))
        
        # Draw circle
        cv2.circle(debug_image, center, radius, (0, 255, 0), 2)
        cv2.circle(debug_image, center, 3, (0, 255, 0), -1)
        
        # Add text
        text = f"Macadamia ({ball_data.get('method', 'unknown')})"
        cv2.putText(debug_image, text, (center[0] - 50, center[1] - radius - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Publish debug image
        debug_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding='bgr8')
        self.debug_image_pub.publish(debug_msg)

def main(args=None):
    rclpy.init(args=args)
    
    try:
        detector = TennisBallMarker()
        detector.get_logger().info('Starting enhanced tennis ball detector...')
        detector.get_logger().info('GREEN tennis balls will be processed as macadamia nuts')
        detector.get_logger().info('Other colored balls will be ignored')
        rclpy.spin(detector)
    except KeyboardInterrupt:
        detector.get_logger().info('Enhanced tennis ball detector shutting down...')
    finally:
        if 'detector' in locals():
            detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
