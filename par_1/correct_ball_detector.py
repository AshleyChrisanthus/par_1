#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from geometry_msgs.msg import Point, PointStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import Empty
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros
import tf2_geometry_msgs

class HybridTennisBallDetectorNoTransform(Node):
    def __init__(self):
        super().__init__('hybrid_tennis_ball_detector_no_transform')
        
        self.bridge = CvBridge()
        self.latest_depth_image = None
        self.camera_intrinsics = None
        # Commented out transform listener to avoid transform errors
        # self.tf_buffer = tf2_ros.Buffer()
        # self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Detection tracking
        self.detected_points = []  # Store detected table tennis ball positions
        self.detection_count = 0
        self.frame_count = 0
        
        # Color ranges for detection (from second code)
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
        
        # Subscribers
        self.image_sub = self.create_subscription(
            CompressedImage, '/oak/rgb/image_rect/compressed', self.image_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/oak/stereo/image_raw', self.depth_callback, 10)
        self.info_sub = self.create_subscription(
            CameraInfo, '/oak/rgb/camera_info', self.info_callback, 10)
        
        # Publishers
        self.ball_pos_pub = self.create_publisher(PointStamped, '/ball_pos', 10)
        self.marker_pub = self.create_publisher(Marker, '/tennis_ball_marker', 10)
        self.home_trigger_pub = self.create_publisher(Empty, '/trigger_home', 10)
        self.debug_image_pub = self.create_publisher(Image, '~/debug_image', 10)
        
        self.get_logger().info('Hybrid Table Tennis Ball Detector has started.')
        self.get_logger().info('Using detection logic from tennis ball detector + depth distance from original')
        self.get_logger().info('Looking for green tennis balls (macadamia nuts) only')
        self.get_logger().info('Multiple balls per frame supported - duplicates will be ignored')
        self.get_logger().info('TRANSFORM DISABLED - Publishing positions in camera frame only')
        # --- NEW --- Added log message for the distance filter
        self.get_logger().info('Minimum detection distance set to 0.6m')

    def info_callback(self, msg):
        if self.camera_intrinsics is None:
            self.camera_intrinsics = msg
            self.get_logger().info('Camera intrinsics received.')
            self.get_logger().info(f'Camera matrix: fx={msg.k[0]:.2f}, fy={msg.k[4]:.2f}, cx={msg.k[2]:.2f}, cy={msg.k[5]:.2f}')
            self.destroy_subscription(self.info_sub)

    def depth_callback(self, msg):
        self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, 'passthrough')

    def image_callback(self, msg):
        if self.latest_depth_image is None or self.camera_intrinsics is None: 
            return

        try:
            self.frame_count += 1
            
            # Convert compressed image to OpenCV format
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
            if cv_image.shape[:2] != self.latest_depth_image.shape[:2]: 
                return

            # Detect ALL green tennis balls in current frame (using second code's logic)
            green_balls = self.detect_all_green_tennis_balls(cv_image)
            
            if green_balls:
                self.get_logger().info(f'Found {len(green_balls)} green ball(s) in frame #{self.frame_count}')
                
                # Process each detected ball
                new_balls_found = 0
                for i, ball_data in enumerate(green_balls):
                    self.get_logger().info(f'Processing ball {i+1}/{len(green_balls)}:')
                    if self.process_tennis_ball(cv_image, msg, ball_data, i+1):
                        new_balls_found += 1
                
                if new_balls_found > 0:
                    self.get_logger().info(f'=== SUMMARY: {new_balls_found} NEW ball(s) added, {len(green_balls) - new_balls_found} duplicate(s) ignored ===')
                else:
                    self.get_logger().info('=== SUMMARY: All balls were previously detected or too close - no new balls added ===')
                    
                # Create visualization with all detected balls
                self.create_debug_visualization(cv_image, green_balls)
            else:
                # Check for other colored balls to reject (from second code)
                other_ball = self.detect_other_colored_balls(cv_image)
                if other_ball:
                    color_name = other_ball['color']
                    self.get_logger().info(f'Not a macadamia nut - detected {color_name} ball (ignoring)')
                
                # Log scanning status occasionally
                elif self.frame_count % 120 == 0:  # Every ~4 seconds
                    self.get_logger().info('Scanning for green tennis balls (macadamia nuts)...')
                    
                # Publish empty debug image
                self.debug_image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, 'bgr8'))
                
        except Exception as e:
            self.get_logger().error(f'Error in tennis ball detection: {str(e)}')

    def detect_all_green_tennis_balls(self, cv_image):
        """Detect ALL green tennis balls (macadamia nuts) in the frame using second code's logic."""
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
            
            # Get depth information for this ball using original code's approach
            depth_mm = self.get_depth_for_region(x, y, w, h)
            if depth_mm is None or depth_mm == 0:
                continue  # Skip if no valid depth
            
            ball_data = {
                'contour': contour,
                'center_x': center_x,
                'center_y': center_y,
                'area': area,
                'bbox': (x, y, w, h),
                'depth': depth_mm  # Add depth information
            }
            
            detected_balls.append(ball_data)
        
        # Sort by depth (nearest first) for consistent processing order
        detected_balls.sort(key=lambda x: x['depth'])
        
        return detected_balls

    def get_depth_for_region(self, x, y, w, h):
        """Get average depth for a bounding box region."""
        try:
            # Extract depth values from the bounding box region
            depth_region = self.latest_depth_image[y:y+h, x:x+w]
            
            # Filter out invalid depth values (0 or very far)
            valid_depths = depth_region[depth_region > 0]
            
            if valid_depths.size == 0:
                return None
            
            # Return average depth in mm
            return np.mean(valid_depths)
            
        except Exception as e:
            self.get_logger().error(f'Error getting depth for region: {str(e)}')
            return None

    def detect_other_colored_balls(self, cv_image):
        """Detect other colored balls to reject them (from second code)."""
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

    def process_tennis_ball(self, cv_image, msg, ball_data, ball_number):
        """Process detected tennis ball using camera frame 3D calculation only."""
        center_x = ball_data['center_x']
        center_y = ball_data['center_y']
        area = ball_data['area']
        depth_mm = ball_data['depth']
        
        # Calculate 3D position in camera frame only
        point_in_camera_frame = self.calculate_3d_position_camera_frame(ball_data, msg)
        
        if point_in_camera_frame:
            # --- NEW: Add minimum distance check ---
            distance_from_camera = point_in_camera_frame.point.z
            if distance_from_camera < 0.6:
                self.get_logger().info(f'  ! Ball #{ball_number} - IGNORED (too close at {distance_from_camera:.2f}m, min is 0.6m)')
                return False # Stop processing this ball as it's too close

            # Check if this is a new ball (using camera frame coordinates)
            if self.is_new_ball(point_in_camera_frame):
                # This is a NEW ball - log and process it
                self.detection_count += 1
                
                self.get_logger().info(f'  ✓ MACADAMIA NUT #{ball_number} - NEW DETECTION!')
                self.get_logger().info(f'    Detection ID: #{self.detection_count}')
                self.get_logger().info(f'    Pixel coords: ({int(center_x)}, {int(center_y)})')
                
                # Log 3D position in camera frame
                cam_x = point_in_camera_frame.point.x
                cam_y = point_in_camera_frame.point.y
                cam_z = point_in_camera_frame.point.z
                self.get_logger().info(f'    Camera frame coords: X={cam_x:.3f}m, Y={cam_y:.3f}m, Z={cam_z:.3f}m')
                self.get_logger().info(f'    Distance from camera: {cam_z:.3f}m')
                
                self.get_logger().info(f'    Ball area: {int(area)}px²')
                self.get_logger().info(f'    Raw depth: {depth_mm/1000.0:.3f}m')
                
                # Add to detected points and publish
                self.detected_points.append(point_in_camera_frame)
                self.ball_pos_pub.publish(point_in_camera_frame)  # Publish in camera frame
                self.publish_marker(point_in_camera_frame)  # Marker in camera frame
                self.get_logger().info(f'    Published ball position and marker #{len(self.detected_points)} (camera frame)')
                
                return True  # New ball found
            else:
                # This is a DUPLICATE ball
                cam_x = point_in_camera_frame.point.x
                cam_y = point_in_camera_frame.point.y
                cam_z = point_in_camera_frame.point.z
                self.get_logger().info(f'  ✗ Ball #{ball_number} - DUPLICATE (already detected)')
                self.get_logger().info(f'    Pixel coords: ({int(center_x)}, {int(center_y)})')
                self.get_logger().info(f'    Camera frame coords: X={cam_x:.3f}m, Y={cam_y:.3f}m, Z={cam_z:.3f}m')
                self.get_logger().info(f'    Ignoring duplicate detection')
                
                return False  # Duplicate ball
        else:
            self.get_logger().warn(f'  ! Ball #{ball_number} - Could not calculate 3D position')
            return False

    def calculate_3d_position_camera_frame(self, ball_data, msg):
        """Calculate 3D position of the ball in camera frame only (no transform)."""
        # Calculate 3D position using camera intrinsics and depth
        x, y, w, h = ball_data['bbox']
        depth_mm = ball_data['depth']
        fx = self.camera_intrinsics.k[0]
        fy = self.camera_intrinsics.k[4]
        cx = self.camera_intrinsics.k[2]
        cy = self.camera_intrinsics.k[5]
        
        pixel_x = x + w / 2
        pixel_y = y + h / 2
        depth_m = depth_mm / 1000.0
        
        # Calculate 3D coordinates in camera frame
        x_cam = (pixel_x - cx) * depth_m / fx
        y_cam = (pixel_y - cy) * depth_m / fy
        z_cam = depth_m

        # Create point in camera frame
        point_in_camera_frame = PointStamped()
        point_in_camera_frame.header.stamp = msg.header.stamp
        point_in_camera_frame.header.frame_id = 'oak_camera_rgb_camera_optical_frame'  # Camera frame
        point_in_camera_frame.point = Point(x=x_cam, y=y_cam, z=z_cam)

        return point_in_camera_frame

    def is_new_ball(self, new_point, threshold=0.3):
        """Check if this is a new ball or previously detected (using camera frame coordinates)."""
        for point in self.detected_points:
            dx = new_point.point.x - point.point.x
            dy = new_point.point.y - point.point.y
            dz = new_point.point.z - point.point.z
            distance = np.sqrt(dx*dx + dy*dy + dz*dz)  # 3D distance
            if distance < threshold:
                return False
        return True

    def publish_marker(self, point_camera):
        """Publish marker for RViz visualization in camera frame."""
        marker = Marker()
        marker.header.frame_id = 'oak_rgb_camera_optical_frame'  # Camera frame instead of map
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'macadamia_nuts'
        marker.id = len(self.detected_points)  # Unique ID per nut
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position = point_camera.point
        marker.pose.orientation.w = 1.0

        # Green sphere for macadamia nuts
        marker.scale.x = marker.scale.y = marker.scale.z = 0.15
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.9

        self.marker_pub.publish(marker)

    def create_debug_visualization(self, cv_image, detected_balls):
        """Create debug visualization with all detected balls."""
        debug_image = cv_image.copy()
        
        for i, ball_data in enumerate(detected_balls):
            x, y, w, h = ball_data['bbox']
            depth_m = ball_data['depth'] / 1000.0
            
            # Use green color for detected balls
            color = (0, 255, 0)  # Green
            label = f"Ball {i+1} @ {depth_m:.2f}m"
            
            # --- MODIFIED --- Change label color if ball is too close
            if depth_m < 0.6:
                color = (0, 165, 255) # Orange for "too close"
                label += " (ignored)"

            cv2.rectangle(debug_image, (x, y), (x + w, y + h), color, 3)
            cv2.putText(debug_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        self.debug_image_pub.publish(self.bridge.cv2_to_imgmsg(debug_image, 'bgr8'))

def main(args=None):
    rclpy.init(args=args)
    
    try:
        detector = HybridTennisBallDetectorNoTransform()
        detector.get_logger().info('Starting hybrid tennis ball detector...')
        detector.get_logger().info('GREEN tennis balls will be processed as macadamia nuts')
        detector.get_logger().info('TRANSFORM DISABLED - All positions in camera frame')
        detector.get_logger().info('Camera frame: X=right, Y=down, Z=forward (distance from camera)')
        rclpy.spin(detector)
    except KeyboardInterrupt:
        detector.get_logger().info('Hybrid tennis ball detector shutting down...')
    finally:
        if 'detector' in locals():
            detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
