#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from std_msgs.msg import String
import numpy as np
from cv_bridge import CvBridge
import cv2
from collections import deque
import time

class NoFrontTreeError(Exception):
    """Exception raised when no tree is found in front of the robot"""
    pass

class MacadamiaScanner(Node):
    """
    Scanner class for macadamia field navigation
    Combines LiDAR and vision for robust tree detection
    Adapted for 2x2 brown cylinder setup
    """
    
    def __init__(self):
        super().__init__('macadamia_scanner')
        
        # ROS2 setup
        self.bridge = CvBridge()
        self.scan = LaserScan()
        self.current_image = None
        
        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)
        
        self.image_sub = self.create_subscription(
            Image,
            'oak/rgb/image_raw',
            self.image_callback,
            10)
        
        self.tree_detection_sub = self.create_subscription(
            String,
            '/detected_trees',
            self.tree_detection_callback,
            10)
        
        # Tree detection parameters for macadamia field
        self.distance_between_trees = 1.5  # Approximate spacing in 2x2 grid
        self.max_tree_detection_range = 3.0  # Max range to look for trees
        self.detected_tree_count = 0
        
        # Brown cylinder detection parameters
        self.brown_hsv_lower = np.array([8, 50, 30])
        self.brown_hsv_upper = np.array([22, 255, 180])
        
        # Navigation parameters
        self.side_detection_angles = [88, 89, 90, 91, 92]  # Angles to check for side trees
        self.front_detection_angles = list(range(0, 40)) + list(range(320, 360))  # Front detection
        
        self.get_logger().info('Macadamia Scanner initialized for 2x2 brown cylinder field')
        
        # Give sensors time to initialize
        time.sleep(1)
        
    def scan_callback(self, msg):
        """Store LiDAR scan data"""
        self.scan = msg
        
    def image_callback(self, msg):
        """Store current camera image"""
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')
    
    def tree_detection_callback(self, msg):
        """Receive tree detection results from vision system"""
        try:
            # Parse "trees_detected:X" format
            if "trees_detected:" in msg.data:
                count = int(msg.data.split(':')[1])
                self.detected_tree_count = count
        except Exception as e:
            self.get_logger().debug(f'Error parsing tree detection: {str(e)}')
    
    def get_scan_data(self):
        """Get current LiDAR scan data"""
        if hasattr(self.scan, 'ranges') and len(self.scan.ranges) > 0:
            return list(self.scan.ranges)
        else:
            return [0.0] * 360  # Return zeros if no data
    
    def get_x_y_from_angle_dist(self, angle, dist):
        """Convert polar coordinates to Cartesian (from original code)"""
        x = -np.cos(np.radians(angle)) * dist
        y = np.sin(np.radians(angle)) * dist
        return x, y
    
    def get_generated_data(self):
        """
        Generate synthetic tree line data for navigation
        Adapted for macadamia brown cylinder detection
        """
        try:
            # Get tree positions using combined vision + LiDAR
            front_tree_angle, front_tree_dist, back_tree_angle, back_tree_dist = self.get_tree_angles_dist()
            
            # Calculate virtual tree line (same algorithm as original)
            front_x, front_y = self.get_x_y_from_angle_dist(90 - front_tree_angle, front_tree_dist)
            back_x, back_y = self.get_x_y_from_angle_dist(back_tree_angle - 90, back_tree_dist)
            
            back_y = -back_y  # Adjust back Y coordinate
            
            # Calculate line equation: y = mx + b
            if front_x - back_x == 0:
                m = 0
            else:
                m = ((front_y - back_y) / (front_x - back_x))
            
            b = -m * front_x + front_y
            
            # Generate distance data for angles 70-110°
            list_of_ranges = []
            for index in range(0, 40):
                angle = index + 70
                
                if angle == 90:  # 90 degrees case
                    if m == 0:  # Parallel to tree line
                        x = front_x
                        self.get_logger().debug(f"Parallel case - x={x}, b={b}")
                    else:
                        x = b / m
                        self.get_logger().debug(f"90° case - x={x}")
                    list_of_ranges.append(abs(x))
                else:
                    if m == 0:  # Robot parallel to tree line
                        if angle < 90:
                            distance = abs(front_x / np.sin(np.radians(angle)))
                        else:
                            distance = abs(front_x / np.sin(np.radians(180 - angle)))
                        list_of_ranges.append(distance)
                    else:  # General case
                        x = b / (np.tan(np.radians(90 + angle)) - m)
                        if angle < 90:
                            distance = abs(x / np.sin(np.radians(angle)))
                        else:
                            distance = abs(x / np.sin(np.radians(180 - angle)))
                        list_of_ranges.append(distance)
            
            return list_of_ranges
            
        except NoFrontTreeError:
            self.get_logger().warn("No front tree found for navigation line generation")
            raise
        except Exception as e:
            self.get_logger().error(f"Error generating navigation data: {str(e)}")
            raise NoFrontTreeError()
    
    def get_avg_angle(self, data, from_ang, to_ang, step):
        """
        Find average angle of detected objects (adapted for brown cylinders)
        """
        count = 0
        sum_angles = 0
        tree_found = False
        
        # Enhanced detection combining LiDAR + vision
        for index in range(from_ang, to_ang, step):
            # Boundary check
            if index < 0 or index >= len(data):
                continue
                
            current_range = data[index]
            
            # Check if there's a valid range reading
            if current_range == 0 or np.isinf(current_range) or np.isnan(current_range):
                if tree_found:
                    break  # End of tree detection
                continue
            
            # Check if range is within reasonable tree detection distance
            if current_range > self.max_tree_detection_range:
                if tree_found:
                    break
                continue
            
            # Validate with vision if we have camera data
            if self.current_image is not None and self.is_brown_cylinder_at_angle(index, current_range):
                if not tree_found:
                    tree_found = True
                sum_angles += index
                count += 1
            elif self.current_image is None:
                # Fallback to LiDAR-only detection
                if not tree_found:
                    tree_found = True
                sum_angles += index
                count += 1
        
        # Handle no tree found cases
        if step == 1 and count == 0:  # Searching backward
            return 0
        
        if count == 0:
            raise NoFrontTreeError("No tree found in specified range")
        
        return int(round(sum_angles / count))
    
    def is_brown_cylinder_at_angle(self, angle, distance):
        """
        Check if there's a brown cylinder at the specified angle using vision
        """
        if self.current_image is None:
            return True  # Assume valid if no vision data
        
        try:
            # Convert angle to pixel coordinate
            height, width = self.current_image.shape[:2]
            
            # Map LiDAR angle to camera pixel (approximate)
            # This is a simplified mapping - you might need to calibrate
            if angle <= 180:
                pixel_x = int(width * (1 - angle / 180.0))
            else:
                pixel_x = int(width * (360 - angle) / 180.0)
            
            # Check region around the pixel for brown color
            roi_size = 30
            x_start = max(0, pixel_x - roi_size)
            x_end = min(width, pixel_x + roi_size)
            y_start = height // 3  # Look in middle third of image
            y_end = 2 * height // 3
            
            roi = self.current_image[y_start:y_end, x_start:x_end]
            
            if roi.size == 0:
                return True  # Benefit of doubt
            
            # Check for brown color in ROI
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            brown_mask = cv2.inRange(hsv_roi, self.brown_hsv_lower, self.brown_hsv_upper)
            brown_ratio = np.sum(brown_mask > 0) / brown_mask.size
            
            # If more than 10% of ROI is brown, consider it a cylinder
            return brown_ratio > 0.1
            
        except Exception as e:
            self.get_logger().debug(f"Error in vision validation: {str(e)}")
            return True  # Assume valid on error
    
    def get_tree_angles_dist(self):
        """
        Find angles and distances of trees for navigation line calculation
        Enhanced for brown cylinder detection
        """
        data = self.get_scan_data()
        
        if len(data) == 0:
            raise NoFrontTreeError("No LiDAR data available")
        
        # Clean up data - remove distant objects
        cleaned_data = data.copy()
        for angle in range(len(cleaned_data)):
            if cleaned_data[angle] > self.distance_between_trees * 2.0:
                cleaned_data[angle] = 0
        
        try:
            # Find front tree (ahead-left, 0-70°)
            angle_front = self.get_avg_angle(cleaned_data, 70, 0, -1)
            
            # Find back tree (behind-left, 110-180°)  
            angle_back = self.get_avg_angle(cleaned_data, 110, 180, 1)
            
        except NoFrontTreeError:
            self.get_logger().warn("Could not find trees for navigation")
            raise
        
        # Get distances
        dist_front = cleaned_data[angle_front] if angle_front < len(cleaned_data) else 0
        dist_back = cleaned_data[angle_back] if angle_back < len(cleaned_data) else 0
        
        # Handle edge case: no back tree found (start of row)
        if angle_back == 0 or dist_back == 0:
            angle_back = 180 - angle_front
            dist_back = dist_front
            self.get_logger().info("Mirroring front tree as back tree (start of row)")
        
        # Validation
        if dist_front == 0:
            raise NoFrontTreeError("No valid front tree distance")
        
        self.get_logger().info(f"Trees found - Front: {angle_front}°@{dist_front:.2f}m, Back: {angle_back}°@{dist_back:.2f}m")
        
        return angle_front, dist_front, angle_back, dist_back
    
    def tree_from_side(self):
        """
        Detect if there's a tree directly to the side (for sampling)
        Enhanced with vision + LiDAR for brown cylinders
        """
        data = self.get_scan_data()
        
        if len(data) == 0:
            return False
        
        # Check multiple angles around 90° (directly left)
        side_distances = []
        brown_detections = 0
        
        for angle in self.side_detection_angles:
            if angle < len(data):
                distance = data[angle]
                if 0 < distance < 1.2:  # Tree within 1.2m to the side
                    side_distances.append(distance)
                    
                    # Validate with vision
                    if self.is_brown_cylinder_at_angle(angle, distance):
                        brown_detections += 1
        
        if not side_distances:
            return False
        
        min_distance = min(side_distances)
        
        # Decision logic: close distance + vision confirmation
        if min_distance < 1.0:  # Very close
            if self.current_image is not None:
                # Require vision confirmation for close detections
                return brown_detections >= 2  # At least 2 angles confirm brown
            else:
                # No vision available, trust LiDAR
                return True
        elif min_distance < 1.2:  # Moderately close
            # Require strong vision confirmation
            return brown_detections >= 3
        
        return False
    
    def get_front_obstacles(self):
        """
        Check for obstacles directly in front
        """
        data = self.get_scan_data()
        
        if len(data) == 0:
            return False, 999
        
        front_distances = []
        for angle in self.front_detection_angles:
            if angle < len(data):
                distance = data[angle]
                if 0 < distance < 2.0:  # Check 2m ahead
                    front_distances.append(distance)
        
        if front_distances:
            min_front_distance = min(front_distances)
            obstacle_detected = min_front_distance < 0.5  # Obstacle within 50cm
            return obstacle_detected, min_front_distance
        
        return False, 999
    
    def get_detected_tree_count(self):
        """Get current number of detected trees from vision system"""
        return self.detected_tree_count
