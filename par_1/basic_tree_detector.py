#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np

class BasicTreeDetector(Node):
    """
    VERY SIMPLE tree detector - just count brown cylinders
    Back to basics that worked with white background
    """
    
    def __init__(self):
        super().__init__('basic_tree_detector')
        
        # Create OpenCV bridge
        self.bridge = CvBridge()
        
        # Subscriber to camera
        self.image_sub = self.create_subscription(
            Image,
            'oak/rgb/image_raw',
            self.image_callback,
            10)
        
        # Publisher for tree count
        self.tree_pub = self.create_publisher(String, '/detected_trees', 10)
        
        # SIMPLE detection parameters (what worked before)
        self.brown_hsv_lower = np.array([8, 40, 20])
        self.brown_hsv_upper = np.array([25, 255, 200])
        
        # Simple size filtering
        self.min_area = 250
        self.max_area = 8000
        self.min_aspect_ratio = 0.4
        self.max_aspect_ratio = 4.0
        
        # Simple clustering distance
        self.cluster_distance = 100
        
        self.get_logger().info('Basic Tree Detector initialized - simple and reliable!')
        
    def image_callback(self, msg):
        """Simple detection callback"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Simple detection
            tree_count = self.detect_trees_basic(cv_image)
            
            # Publish count
            tree_msg = String()
            tree_msg.data = f"trees_detected:{tree_count}"
            self.tree_pub.publish(tree_msg)
                
        except Exception as e:
            self.get_logger().error(f'Error in basic tree detection: {str(e)}')
    
    def detect_trees_basic(self, image):
        """
        VERY BASIC detection - just like what worked with white background
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Basic color detection
        mask = cv2.inRange(hsv, self.brown_hsv_lower, self.brown_hsv_upper)
        
        # Simple cleanup
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0
        
        # Simple filtering
        valid_trees = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area <= area <= self.max_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / w if w > 0 else 0
                
                if self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio:
                    center_x = x + w / 2
                    center_y = y + h / 2
                    valid_trees.append({
                        'center_x': center_x,
                        'center_y': center_y,
                        'area': area
                    })
        
        # Simple clustering (remove close duplicates)
        clustered_trees = self.simple_cluster(valid_trees)
        
        return len(clustered_trees)
    
    def simple_cluster(self, detections):
        """Very simple clustering"""
        if not detections:
            return []
        
        clustered = []
        used = [False] * len(detections)
        
        for i, detection in enumerate(detections):
            if used[i]:
                continue
                
            # Mark this detection as used
            used[i] = True
            clustered.append(detection)
            
            # Remove nearby detections
            for j, other in enumerate(detections):
                if used[j]:
                    continue
                    
                distance = np.sqrt((detection['center_x'] - other['center_x'])**2 + 
                                 (detection['center_y'] - other['center_y'])**2)
                
                if distance <= self.cluster_distance:
                    used[j] = True
        
        return clustered

def main(args=None):
    rclpy.init(args=args)
    
    try:
        tree_detector = BasicTreeDetector()
        tree_detector.get_logger().info('🌳 Starting BASIC tree detector - simple and reliable')
        rclpy.spin(tree_detector)
    except KeyboardInterrupt:
        tree_detector.get_logger().info('Basic tree detector shutting down...')
    finally:
        if 'tree_detector' in locals():
            tree_detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
