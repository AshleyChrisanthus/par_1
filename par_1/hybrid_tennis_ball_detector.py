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

# Renamed class to reflect its new capability
class HybridTennisBallDetector(Node):
    def __init__(self):
        # Renamed node
        super().__init__('hybrid_tennis_ball_detector')
        
        self.bridge = CvBridge()
        self.latest_depth_image = None
        self.camera_intrinsics = None
        
        # --- MODIFICATION: Re-enabled TF2 listener ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.target_frame = 'map'  # The frame we want to transform our points into
        
        # Detection tracking
        # This list will now store detected ball positions in the 'map' frame
        self.detected_points = []
        self.detection_count = 0
        self.frame_count = 0
        
        self.min_detection_distance_m = 0.6
        
        # Color ranges for detection
        self.green_lower = np.array([25, 100, 100])
        self.green_upper = np.array([40, 255, 255])
        
        # Other ball colors to reject
        self.other_color_ranges = [
            ([0, 100, 100], [10, 255, 255], "red"),
            ([170, 100, 100], [180, 255, 255], "red"),
            ([100, 100, 100], [130, 255, 255], "blue"),
            ([15, 100, 100], [25, 255, 255], "yellow"),
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
        self.get_logger().info(f'Ignoring balls closer than {self.min_detection_distance_m}m')
        self.get_logger().info(f'TRANSFORM ENABLED - Publishing positions and markers in the "{self.target_frame}" frame.')

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

            green_balls = self.detect_all_green_tennis_balls(cv_image)
            
            if green_balls:
                self.get_logger().info(f'Found {len(green_balls)} potential green ball(s) in frame #{self.frame_count}')
                
                new_balls_found = 0
                for i, ball_data in enumerate(green_balls):
                    if self.process_tennis_ball(msg, ball_data, i+1):
                        new_balls_found += 1
                
                if new_balls_found > 0:
                    self.get_logger().info(f'=== SUMMARY: {new_balls_found} NEW ball(s) added, {len(green_balls) - new_balls_found} duplicate(s)/untransformed ignored ===')
                
                self.create_debug_visualization(cv_image, green_balls)
            else:
                if self.frame_count % 120 == 0:
                    self.get_logger().info('Scanning for green tennis balls (macadamia nuts)...')
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
            
            depth_m = depth_mm / 1000.0
            if depth_m < self.min_detection_distance_m:
                continue
            
            detected_balls.append({
                'bbox': (x, y, w, h),
                'depth': depth_mm,
            })
        
        detected_balls.sort(key=lambda x: x['depth'])
        return detected_balls

    def get_depth_for_region(self, x, y, w, h):
        try:
            depth_region = self.latest_depth_image[y:y+h, x:x+w]
            valid_depths = depth_region[depth_region > 0]
            return np.mean(valid_depths) if valid_depths.size > 0 else None
        except Exception as e:
            self.get_logger().error(f'Error getting depth for region: {str(e)}')
            return None

    # --- MODIFICATION: Core logic for transforming and publishing ---
    def process_tennis_ball(self, msg, ball_data, ball_number):
        """Process a single detected ball: transform to map frame and check for duplicates."""
        
        # 1. Calculate 3D position in the camera's frame
        point_in_camera_frame = self.calculate_3d_position_camera_frame(ball_data, msg)
        if not point_in_camera_frame:
            return False

        # 2. Transform the point from the camera frame to the map frame
        try:
            # This is the magic step. It uses the timestamp from the PointStamped message
            # to get the transform at the correct time, handling delays automatically.
            point_in_map_frame = self.tf_buffer.transform(
                point_in_camera_frame,
                self.target_frame,
                timeout=Duration(seconds=0.1)
            )
            self.get_logger().info(f'  Ball #{ball_number}: Successfully transformed to "{self.target_frame}" frame.')

        except (tf2_ros.TransformException, tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as ex:
            self.get_logger().warn(f'  Ball #{ball_number}: Could not transform point from camera to map frame: {ex}')
            return False

        # 3. Check if this is a new ball (using map frame coordinates)
        if self.is_new_ball(point_in_map_frame):
            self.detection_count += 1
            self.get_logger().info(f'  ✓ NEW DETECTION! ID: #{self.detection_count}')
            
            map_x = point_in_map_frame.point.x
            map_y = point_in_map_frame.point.y
            map_z = point_in_map_frame.point.z
            self.get_logger().info(f'    Map frame coords: X={map_x:.3f}m, Y={map_y:.3f}m, Z={map_z:.3f}m')
            
            # 4. Add to detected points list and publish
            self.detected_points.append(point_in_map_frame)
            self.ball_pos_pub.publish(point_in_map_frame)  # Publish in map frame
            self.publish_marker(point_in_map_frame)        # Publish marker in map frame
            self.get_logger().info(f'    Published ball position and marker #{len(self.detected_points)} in "{self.target_frame}" frame.')
            
            return True  # New ball found and processed
        else:
            self.get_logger().info(f'  ✗ Ball #{ball_number}: DUPLICATE of a previously detected ball. Ignoring.')
            return False

    def calculate_3d_position_camera_frame(self, ball_data, msg):
        x, y, w, h = ball_data['bbox']
        depth_mm = ball_data['depth']
        fx = self.camera_intrinsics.k[0]
        fy = self.camera_intrinsics.k[4]
        cx = self.camera_intrinsics.k[2]
        cy = self.camera_intrinsics.k[5]
        
        pixel_x = x + w / 2
        pixel_y = y + h / 2
        depth_m = depth_mm / 1000.0
        
        x_cam = (pixel_x - cx) * depth_m / fx
        y_cam = (pixel_y - cy) * depth_m / fy
        
        point_stamped = PointStamped()
        point_stamped.header.stamp = msg.header.stamp
        point_stamped.header.frame_id = 'oak_rgb_camera_optical_frame'
        point_stamped.point = Point(x=x_cam, y=y_cam, z=depth_m)
        return point_stamped

    def is_new_ball(self, new_point_stamped, threshold=0.3):
        """
        Check if a new point in the map frame is a duplicate of an existing one.
        `new_point_stamped` and `self.detected_points` are both in the map frame.
        """
        for known_point in self.detected_points:
            dx = new_point_stamped.point.x - known_point.point.x
            dy = new_point_stamped.point.y - known_point.point.y
            dz = new_point_stamped.point.z - known_point.point.z
            distance = np.sqrt(dx*dx + dy*dy + dz*dz)
            if distance < threshold:
                return False  # It's a duplicate
        return True # It's a new ball

    def publish_marker(self, point_in_map):
        """Publish marker for RViz visualization in the map frame."""
        marker = Marker()
        # --- MODIFICATION: Header frame is now the target map frame ---
        marker.header.frame_id = self.target_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'macadamia_nuts'
        marker.id = len(self.detected_points)  # Use the count as a unique ID
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # --- MODIFICATION: Position is from the transformed point ---
        marker.pose.position = point_in_map.point
        marker.pose.orientation.w = 1.0

        marker.scale.x = marker.scale.y = marker.scale.z = 0.15
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = (0.0, 1.0, 0.0, 0.9)

        self.marker_pub.publish(marker)

    def create_debug_visualization(self, cv_image, detected_balls):
        debug_image = cv_image.copy()
        for i, ball_data in enumerate(detected_balls):
            x, y, w, h = ball_data['bbox']
            depth_m = ball_data['depth'] / 1000.0
            color = (0, 255, 0)
            label = f"Ball @ {depth_m:.2f}m"
            cv2.rectangle(debug_image, (x, y), (x + w, y + h), color, 3)
            cv2.putText(debug_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        self.debug_image_pub.publish(self.bridge.cv2_to_imgmsg(debug_image, 'bgr8'))

def main(args=None):
    rclpy.init(args=args)
    detector = None
    try:
        detector = HybridTennisBallDetector()
        detector.get_logger().info('Starting hybrid tennis ball detector with map transforms...')
        rclpy.spin(detector)
    except KeyboardInterrupt:
        detector.get_logger().info('Hybrid tennis ball detector shutting down...')
    finally:
        if detector:
            detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
