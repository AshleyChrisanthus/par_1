#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from geometry_msgs.msg import Point, PointStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import Empty
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros
import tf2_geometry_msgs

# Renamed class to reflect its new transform capabilities
class HybridTennisBallDetector(Node):
    def __init__(self):
        # Renamed node for clarity
        super().__init__('hybrid_tennis_ball_detector_no_transform')
        
        self.bridge = CvBridge()
        self.latest_depth_image = None
        self.camera_intrinsics = None
        
        # --- REQUIREMENT 2: Re-enabling TF2 listener ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.target_frame = 'map'  # The frame we want to publish markers in
        
        # Detection tracking
        # This list will now store detected ball positions in the 'map' frame
        self.detected_points = []
        self.detection_count = 0
        self.frame_count = 0
        
        # --- REQUIREMENT 1: Minimum distance configuration ---
        self.min_detection_distance_m = 0.6
        
        # Color ranges for detection
        self.green_lower = np.array([25, 100, 100])
        self.green_upper = np.array([40, 255, 255])
        
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
        
        self.get_logger().info('Hybrid Tennis Ball Detector has started.')
        self.get_logger().info(f'1. Ignoring balls closer than {self.min_detection_distance_m}m.')
        self.get_logger().info(f'2. Publishing positions and markers in the "{self.target_frame}" frame.')

    def info_callback(self, msg):
        if self.camera_intrinsics is None:
            self.camera_intrinsics = msg
            self.get_logger().info('Camera intrinsics received.')
            self.destroy_subscription(self.info_sub)

    def depth_callback(self, msg):
        self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, 'passthrough')

    def image_callback(self, msg):
        if self.latest_depth_image is None or self.camera_intrinsics is None: 
            return

        try:
            self.frame_count += 1
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
            if cv_image.shape[:2] != self.latest_depth_image.shape[:2]: 
                return

            # Detect green balls that are NOT too close
            green_balls = self.detect_all_green_tennis_balls(cv_image)
            
            if green_balls:
                self.get_logger().info(f'Found {len(green_balls)} potential green ball(s) in frame #{self.frame_count}')
                
                new_balls_found = 0
                # The process_tennis_ball function now handles the transformation
                for i, ball_data in enumerate(green_balls):
                    if self.process_tennis_ball(msg, ball_data, i + 1):
                        new_balls_found += 1
                
                if new_balls_found > 0:
                    self.get_logger().info(f'=== SUMMARY: {new_balls_found} NEW ball(s) added to map ===')
                    
                self.create_debug_visualization(cv_image, green_balls)
            else:
                if self.frame_count % 120 == 0:
                    self.get_logger().info('Scanning for green tennis balls...')
                self.debug_image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, 'bgr8'))
                
        except Exception as e:
            self.get_logger().error(f'Error in image callback: {str(e)}')

    def detect_all_green_tennis_balls(self, cv_image):
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_balls = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            depth_mm = self.get_depth_for_region(x, y, w, h)
            if depth_mm is None or depth_mm == 0:
                continue
            
            # --- REQUIREMENT 1: Filter balls that are too close ---
            depth_m = depth_mm / 1000.0
            if depth_m < self.min_detection_distance_m:
                self.get_logger().debug(f'Ignoring potential ball at {depth_m:.2f}m (too close).')
                continue # Skip this ball
            
            detected_balls.append({
                'bbox': (x, y, w, h),
                'depth': depth_mm
            })
        
        detected_balls.sort(key=lambda x: x['depth'])
        return detected_balls

    def get_depth_for_region(self, x, y, w, h):
        try:
            depth_region = self.latest_depth_image[y:y+h, x:x+w]
            valid_depths = depth_region[depth_region > 0]
            return np.mean(valid_depths) if valid_depths.size > 0 else None
        except Exception:
            return None

    def process_tennis_ball(self, msg, ball_data, ball_number):
        """Calculates 3D position, transforms to map frame, and publishes if new."""
        # Step 1: Calculate 3D position in the camera's frame
        point_in_camera_frame = self.calculate_3d_position_camera_frame(ball_data, msg)
        if not point_in_camera_frame:
            return False

        # --- REQUIREMENT 2 & 3: Transform point to map frame with error handling ---
        try:
            # This function uses the timestamp in point_in_camera_frame.header.stamp
            # to get the transform at the correct time, solving time mismatch issues.
            point_in_map_frame = self.tf_buffer.transform(
                point_in_camera_frame,
                self.target_frame,
                timeout=Duration(seconds=0.2) # Don't wait too long for transform
            )
            self.get_logger().debug(f'Ball #{ball_number}: Successfully transformed to "{self.target_frame}" frame.')
        except (tf2_ros.TransformException) as ex:
            self.get_logger().warn(f'Could not transform point for ball #{ball_number}: {ex}')
            return False

        # Step 2: Check if this is a new ball (using map frame coordinates)
        if self.is_new_ball(point_in_map_frame):
            self.detection_count += 1
            self.get_logger().info(f'  ✓ NEW DETECTION! Total: {self.detection_count}')
            
            map_x = point_in_map_frame.point.x
            map_y = point_in_map_frame.point.y
            self.get_logger().info(f'    Map Coords: X={map_x:.3f}m, Y={map_y:.3f}m')
            
            # Step 3: Add to detected points list (in map frame) and publish
            self.detected_points.append(point_in_map_frame)
            self.ball_pos_pub.publish(point_in_map_frame)
            self.publish_marker(point_in_map_frame)
            return True
        else:
            self.get_logger().debug(f'  ✗ Ball #{ball_number}: DUPLICATE of a known ball. Ignoring.')
            return False

    def calculate_3d_position_camera_frame(self, ball_data, msg):
        x, y, w, h = ball_data['bbox']
        depth_mm = ball_data['depth']
        fx, fy = self.camera_intrinsics.k[0], self.camera_intrinsics.k[4]
        cx, cy = self.camera_intrinsics.k[2], self.camera_intrinsics.k[5]
        
        pixel_x, pixel_y = x + w / 2, y + h / 2
        depth_m = depth_mm / 1000.0
        
        x_cam = (pixel_x - cx) * depth_m / fx
        y_cam = (pixel_y - cy) * depth_m / fy
        
        point_stamped = PointStamped()
        point_stamped.header.stamp = msg.header.stamp
        point_stamped.header.frame_id = 'oak_rgb_camera_optical_frame'
        point_stamped.point = Point(x=x_cam, y=y_cam, z=depth_m)
        return point_stamped

    def is_new_ball(self, new_point_stamped, threshold=0.3):
        """Checks for duplicates in the fixed 'map' frame."""
        for known_point in self.detected_points:
            dist = np.linalg.norm([
                new_point_stamped.point.x - known_point.point.x,
                new_point_stamped.point.y - known_point.point.y,
                new_point_stamped.point.z - known_point.point.z
            ])
            if dist < threshold:
                return False  # It's a duplicate
        return True # It's a new ball

    def publish_marker(self, point_in_map):
        """Publish marker for RViz visualization in the map frame."""
        marker = Marker()
        # --- REQUIREMENT 2: Header frame is now the target map frame ---
        marker.header.frame_id = self.target_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'macadamia_nuts'
        marker.id = self.detection_count # Use unique detection count as ID
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = point_in_map.point
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.15
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = (0.0, 1.0, 0.0, 0.9)
        self.marker_pub.publish(marker)
        self.get_logger().info(f'    Published marker #{marker.id} to RViz.')

    def create_debug_visualization(self, cv_image, detected_balls):
        debug_image = cv_image.copy()
        for i, ball_data in enumerate(detected_balls):
            x, y, w, h = ball_data['bbox']
            depth_m = ball_data['depth'] / 1000.0
            label = f"Ball @ {depth_m:.2f}m"
            cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(debug_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        self.debug_image_pub.publish(self.bridge.cv2_to_imgmsg(debug_image, 'bgr8'))

def main(args=None):
    rclpy.init(args=args)
    detector = None
    try:
        detector = HybridTennisBallDetector()
        rclpy.spin(detector)
    except KeyboardInterrupt:
        pass
    finally:
        if detector:
            detector.get_logger().info('Shutting down hybrid tennis ball detector.')
            detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
