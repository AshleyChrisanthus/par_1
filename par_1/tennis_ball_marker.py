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

class TennisBallMarker(Node):
    def __init__(self):
        super().__init__('tennis_ball_marker')

        self.bridge = CvBridge()
        self.scan = None
        self.depth_image = None
        self.camera_info = None

        # Subscribers
        self.sub_image = self.create_subscription(Image, '/oak/rgb/image_raw', self.image_callback, 10)
        self.sub_depth = self.create_subscription(Image, '/oak/stereo/image_raw', self.depth_callback, 10) # Subscribe to depth image
        self.sub_camera_info = self.create_subscription(CameraInfo, '/oak/rgb/camera_info', self.camera_info_callback, 10) # Subscribe to camera info
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10) # Keep for logging/debugging if needed, but not for direct 3D calcs

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

    def scan_callback(self, msg):
        """Store LiDAR scan data (for general awareness, not for 3D point calc)."""
        self.scan = msg

    def depth_callback(self, msg):
        """Store depth image data."""
        self.depth_image = msg

    def camera_info_callback(self, msg):
        """Store camera intrinsic parameters."""
        self.camera_info = msg
        # Unsubscribe after getting info, assuming it doesn't change
        self.destroy_subscription(self.sub_camera_info) 
        self.get_logger().info('Received camera info.')


    def image_callback(self, msg):
        """Main image processing callback with enhanced logging."""
        # Ensure all necessary data is available
        if self.depth_image is None:
            self.get_logger().warn('Waiting for depth image data...')
            return
        if self.camera_info is None:
            self.get_logger().warn('Waiting for camera info data...')
            return

        try:
            self.frame_count += 1
            
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Convert ROS Depth Image to OpenCV
            # Be careful with depth encoding, typically 16UC1 (ushort) or 32FC1 (float)
            # D435i often uses 16UC1, where depth is in millimeters
            cv_depth_image = self.bridge.imgmsg_to_cv2(self.depth_image, desired_encoding='passthrough')

            # Check for green tennis balls (macadamia nuts)
            green_ball = self.detect_green_tennis_ball(cv_image)
            
            if green_ball:
                # Process the green ball (macadamia nut)
                self.process_macadamia_nut(cv_image, cv_depth_image, green_ball, msg.header.frame_id, msg.header.stamp)
            else:
                # Check for other colored balls to reject
                other_ball = self.detect_other_colored_balls(cv_image)
                if other_ball:
                    color_name = other_ball['color']
                    self.get_logger().info(f'Not a macadamia nut - detected {color_name} (ignoring)')
                
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
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) # Add closing to fill small gaps

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None

        # Find largest contour (assumed to be the tennis ball)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area < 200:  # Filter small blobs
            return None

        # Get bounding box and centroid
        M = cv2.moments(largest_contour)
        if M["m00"] == 0: # Avoid division by zero if moment is zero (e.g., single point contour)
            return None
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        
        x, y, w, h = cv2.boundingRect(largest_contour) # Bounding box for visualization if needed
        
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
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                if area > 200:  # Same size threshold as green balls
                    return {'color': color_name, 'area': area}
        
        return None

    def process_macadamia_nut(self, cv_image, cv_depth_image, ball_data, camera_frame_id, timestamp):
        """Process detected green tennis ball (macadamia nut) with detailed logging."""
        center_x = ball_data['center_x']
        center_y = ball_data['center_y']
        area = ball_data['area']
        
        # Get world coordinates using depth camera
        world_coords = self.get_world_coordinates_from_depth(center_x, center_y, cv_depth_image, camera_frame_id, timestamp)
        
        if world_coords:
            camera_x, camera_y, camera_z, map_x, map_y, map_z = world_coords
            
            self.detection_count += 1
            self.get_logger().info('MACADAMIA NUT DETECTED! (Green tennis ball)')
            self.get_logger().info(f'  Detection #{self.detection_count}:')
            self.get_logger().info(f'  Pixel coordinates: ({int(center_x)}, {int(center_y)})')
            self.get_logger().info(f'  Camera coordinates: (X:{camera_x:.2f}, Y:{camera_y:.2f}, Z:{camera_z:.2f}) meters')
            self.get_logger().info(f'  Map coordinates: (X:{map_x:.2f}, Y:{map_y:.2f}, Z:{map_z:.2f}) meters')
            
            # Build point for map transformation
            point_camera = PointStamped()
            point_camera.header.frame_id = camera_frame_id # This should be the camera's optical frame
            point_camera.header.stamp = timestamp
            point_camera.point.x = camera_x
            point_camera.point.y = camera_y
            point_camera.point.z = camera_z

            # Transform to map frame and publish marker if new
            self.transform_and_publish(point_camera)
        else:
            self.get_logger().warn(f"Could not get 3D coordinates for detected ball at ({int(center_x)}, {int(center_y)})")


    def get_world_coordinates_from_depth(self, u, v, cv_depth_image, camera_frame_id, timestamp):
        """
        Convert pixel (u, v) and depth from depth image to 3D world coordinates
        in the camera's frame, then transform to map frame.
        """
        if self.camera_info is None:
            self.get_logger().error('Camera intrinsic parameters not available for depth calculation.')
            return None

        # Ensure pixel coordinates are within image bounds
        if not (0 <= u < cv_depth_image.shape[1] and 0 <= v < cv_depth_image.shape[0]):
            self.get_logger().warn(f'Pixel ({u}, {v}) out of depth image bounds.')
            return None

        # Get depth value at the pixel
        # Assuming depth image encoding is 16UC1 (unsigned short, millimeters) or 32FC1 (float, meters)
        depth_val = cv_depth_image[v, u]
        
        # Convert depth to meters if it's in millimeters (common for 16UC1)
        # Check the encoding of your /oak/stereo/image_raw topic
        if self.depth_image.encoding == '16UC1': # Example for 16-bit unsigned integer, typically millimeters
            depth_in_meters = float(depth_val) / 1000.0
        elif self.depth_image.encoding == '32FC1': # Example for 32-bit float, typically meters
            depth_in_meters = float(depth_val)
        else:
            self.get_logger().error(f"Unsupported depth image encoding: {self.depth_image.encoding}. Please check.")
            return None

        if not np.isfinite(depth_in_meters) or depth_in_meters <= 0.01: # Filter out invalid or very close depths
            self.get_logger().warn(f'Invalid or zero depth value at ({u}, {v}): {depth_in_meters:.4f} meters.')
            return None

        # Get camera intrinsic parameters
        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]

        # Convert pixel to 3D point in camera frame
        # X, Y, Z in camera frame (Z is forward, X is right, Y is down for optical frame)
        camera_z = depth_in_meters
        camera_x = (u - cx) * camera_z / fx
        camera_y = (v - cy) * camera_z / fy

        # Build PointStamped in camera frame
        point_camera = PointStamped()
        point_camera.header.frame_id = camera_frame_id # This should be the camera's optical frame (e.g., 'oak_rgb_camera_optical_frame')
        point_camera.header.stamp = timestamp
        point_camera.point.x = camera_x
        point_camera.point.y = camera_y
        point_camera.point.z = camera_z

        # Transform to map coordinates
        map_coords = self.transform_to_map(point_camera)
        if map_coords:
            map_x, map_y, map_z = map_coords
            return (camera_x, camera_y, camera_z, map_x, map_y, map_z)
        else:
            return None

    def transform_to_map(self, point_stamped_in_source_frame):
        """Transform PointStamped from its current frame to the map frame."""
        try:
            timeout = rclpy.duration.Duration(seconds=0.5) # Increased timeout
            if self.tf_buffer.can_transform('map', point_stamped_in_source_frame.header.frame_id, point_stamped_in_source_frame.header.stamp, timeout=timeout):
                # Using the timestamp of the incoming message for TF lookup
                # This is crucial for time-synchronous transforms
                transform = self.tf_buffer.lookup_transform(
                    'map',
                    point_stamped_in_source_frame.header.frame_id,
                    point_stamped_in_source_frame.header.stamp, # Use the timestamp of the data
                    timeout=timeout
                )
                point_map = tf2_geometry_msgs.do_transform_point(point_stamped_in_source_frame, transform)
                return (point_map.point.x, point_map.point.y, point_map.point.z)
            else:
                self.get_logger().warn(f"TF transform from '{point_stamped_in_source_frame.header.frame_id}' to 'map' not ready or timed out for stamp {point_stamped_in_source_frame.header.stamp.sec}.{point_stamped_in_source_frame.header.stamp.nanosec}")
                return None
        except (tf2_ros.TransformException, tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f'TF exception during transform to map: {e}')
            return None

    def is_new_ball(self, new_point, threshold=0.3):
        """Check if this is a new ball or previously detected."""
        for point in self.detected_points:
            dx = new_point.point.x - point.point.x
            dy = new_point.point.y - point.point.y
            dz = new_point.point.z - point.point.z # Also consider Z for 3D proximity
            if np.hypot(dx, dy, dz) < threshold: # Use hypot for 3D distance
                return False
        return True

    def transform_and_publish(self, point_camera):
        """Transform to map frame and publish marker if new ball."""
        try:
            timeout = rclpy.duration.Duration(seconds=0.5)
            # Use the exact timestamp of the image/depth frame for TF lookup
            if self.tf_buffer.can_transform('map', point_camera.header.frame_id, point_camera.header.stamp, timeout=timeout):
                transform = self.tf_buffer.lookup_transform(
                    'map',
                    point_camera.header.frame_id,
                    point_camera.header.stamp,
                    timeout=timeout
                )
                point_map = tf2_geometry_msgs.do_transform_point(point_camera, transform)

                if self.is_new_ball(point_map):
                    self.publish_marker(point_map)
                    self.detected_points.append(point_map)
                    self.get_logger().info(f'New macadamia nut confirmed at map coordinates (X:{point_map.point.x:.2f}, Y:{point_map.point.y:.2f}, Z:{point_map.point.z:.2f})')
                    self.get_logger().info(f'Published marker #{len(self.detected_points)} in RViz')
                    
                    # Trigger home after detection
                    self.home_trigger_pub.publish(Empty())
                    self.get_logger().info('Triggered home return sequence')
                else:
                    self.get_logger().info('Previously detected macadamia nut - ignoring duplicate')
            else:
                self.get_logger().warn(f"TF transform from '{point_camera.header.frame_id}' to 'map' not ready or timed out for stamp {point_camera.header.stamp.sec}.{point_camera.header.stamp.nanosec}")
        except (tf2_ros.TransformException, tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f'TF exception in transform_and_publish: {e}')

    def publish_marker(self, point_map):
        """Publish marker for RViz visualization."""
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg() # Use current time for marker stamp, or point_map.header.stamp
        marker.ns = 'macadamia_nuts'
        marker.id = len(self.detected_points)  # Unique ID per nut
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position = point_map.point
        marker.pose.orientation.w = 1.0

        # Green sphere for macadamia nuts
        marker.scale.x = marker.scale.y = marker.scale.z = 0.15 # Approx tennis ball size
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.9

        self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    
    try:
        detector = TennisBallMarker()
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
