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

class HybridTennisBallDetector(Node):
    def __init__(self):
        super().__init__('hybrid_tennis_ball_detector')
        
        self.bridge = CvBridge()
        self.latest_depth_image = None
        self.camera_intrinsics = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Detection tracking
        self.detected_points = []
        self.detection_count = 0
        
        # --- NEW: Minimum detection distance configuration ---
        self.min_detection_distance_m = 0.6
        
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
        self.debug_image_pub = self.create_publisher(Image, '~/debug_image', 10)
        
        self.get_logger().info('Hybrid Tennis Ball Detector has started.')
        self.get_logger().info(f'Ignoring any balls detected closer than {self.min_detection_distance_m}m.')

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

        # --- NEW: Get the current time as soon as the message is received ---
        # This will be used to overwrite the bad timestamp from the camera.
        now = self.get_clock().now().to_msg()

        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
            # The detect function now filters by distance automatically
            green_balls = self.detect_all_green_tennis_balls(cv_image)
            
            if green_balls:
                for ball_data in green_balls:
                    # self.process_tennis_ball(msg, ball_data)
                    self.process_tennis_ball(now, ball_data)
                self.create_debug_visualization(cv_image, green_balls)

        except Exception as e:
            self.get_logger().error(f'Error in image_callback: {str(e)}')

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
            
            # --- NEW: Filter objects that are too close ---
            if (depth_mm / 1000.0) < self.min_detection_distance_m:
                self.get_logger().debug(f'Ignoring potential ball at {depth_mm / 1000.0:.2f}m (too close).')
                continue # Skip to the next contour
            
            detected_balls.append({'bbox': (x, y, w, h), 'depth': depth_mm})
        
        detected_balls.sort(key=lambda x: x['depth'])
        return detected_balls

    def get_depth_for_region(self, x, y, w, h):
        try:
            depth_region = self.latest_depth_image[y:y+h, x:x+w]
            valid_depths = depth_region[depth_region > 0]
            return np.mean(valid_depths) if valid_depths.size > 0 else None
        except Exception:
            return None

    def process_tennis_ball(self, timestamp, ball_data):
        # point_in_map_frame = self.calculate_3d_position(ball_data, msg)
        point_in_map_frame = self.calculate_3d_position(timestamp, ball_data)

        if point_in_map_frame and self.is_new_ball(point_in_map_frame):
            self.detection_count += 1
            self.get_logger().info(f'✓ CONFIRMED NEW DETECTION! Total: {self.detection_count}')
            self.get_logger().info(f'  Map Coords: X={point_in_map_frame.point.x:.2f}, Y={point_in_map_frame.point.y:.2f}, Z={point_in_map_frame.point.z:.2f} ')
            
            self.detected_points.append(point_in_map_frame)
            self.ball_pos_pub.publish(point_in_map_frame)
            self.publish_marker(point_in_map_frame)

    # def calculate_3d_position(self, ball_data, msg):
    def calculate_3d_position(self, timestamp, ball_data):
        target_frame = 'map'
        # Check this with: ros2 run tf2_tools view_frames
        source_frame = 'oak_camera_rgb_camera_optical_frame' 
        
        x, y, w, h = ball_data['bbox']
        depth_mm = ball_data['depth']
        fx, fy = self.camera_intrinsics.k[0], self.camera_intrinsics.k[4]
        cx, cy = self.camera_intrinsics.k[2], self.camera_intrinsics.k[5]
        
        pixel_x, pixel_y = x + w / 2, y + h / 2
        depth_m = depth_mm / 1000.0
        x_cam, y_cam, z_cam = (pixel_x - cx) * depth_m / fx, (pixel_y - cy) * depth_m / fy, depth_m

        point_in_camera_frame = PointStamped()
        # point_in_camera_frame.header.stamp = msg.header.stamp
        # point_in_camera_frame.header.stamp = timestamp
        point_in_camera_frame.header.stamp = rclpy.time.Time().to_msg()
        point_in_camera_frame.header.frame_id = source_frame
        point_in_camera_frame.point = Point(x=x_cam, y=y_cam, z=z_cam)

        try:
            point_in_map_frame = self.tf_buffer.transform(
                point_in_camera_frame,
                target_frame,
                timeout=Duration(seconds=0.2)
            )
            return point_in_map_frame
        except tf2_ros.TransformException as ex:
            self.get_logger().warn(f'Could not transform point: {ex}')
            return None

    def is_new_ball(self, new_point, threshold=0.3):
        # for point in self.detected_points:
        #     dist = np.hypot(new_point.point.x - point.point.x, new_point.point.y - point.point.y)
        #     if dist < threshold:
        #         return False
        # return True

        for point in self.detected_points:
            # Calculate the 3D distance between the new point and the known point
            dx = new_point.point.x - point.point.x
            dy = new_point.point.y - point.point.y
            dz = new_point.point.z - point.point.z
            dist = np.sqrt(dx*dx + dy*dy + dz*dz) # Full 3D distance
            
            if dist < threshold:
                return False # It's a duplicate
        return True # It's a new ball
    
    def publish_marker(self, point_map):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'macadamia_nuts'
        marker.id = self.detection_count
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = point_map.point
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.15
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = (0.0, 1.0, 0.0, 0.9)
        self.marker_pub.publish(marker)

    def create_debug_visualization(self, cv_image, detected_balls):
        debug_image = cv_image.copy()
        for ball_data in detected_balls:
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
            detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
