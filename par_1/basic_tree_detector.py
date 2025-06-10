#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np

class RobustHybridDetector(Node):
    """
    Robust tree detector for changing backgrounds
    Strategy: Remove green turf (consistent) + Detect brown cylinders (target)
    Background-independent detection!
    """
    
    def __init__(self):
        super().__init__('basic_tree_detector')
        
        self.bridge = CvBridge()
        
        # Subscribe to camera
        self.image_sub = self.create_subscription(
            Image,
            'oak/rgb/image_raw',
            self.image_callback,
            10)
        
        # Publishers
        self.tree_pub = self.create_publisher(String, '/detected_trees', 10)
        self.debug_pub = self.create_publisher(Image, '/tree_detection/debug_image', 10)
        self.mask_pub = self.create_publisher(Image, '/tree_detection/mask_image', 10)
        
        # Green turf detection (consistent across room)
        self.green_hsv_lower = np.array([35, 30, 30])
        self.green_hsv_upper = np.array([85, 255, 255])
        
        # Brown cylinder detection (multiple ranges for different lighting)
        # Light brown/tan ranges for cardboard
        self.brown_ranges = [
            # Range 1: Light tan/beige
            (np.array([8, 30, 60]), np.array([25, 180, 220])),
            # Range 2: Medium brown  
            (np.array([10, 40, 80]), np.array([30, 200, 255])),
            # Range 3: Dark brown/cardboard
            (np.array([5, 50, 40]), np.array([20, 255, 180]))
        ]
        
        # Adaptive parameters
        self.min_area = 200
        self.max_area = 5000
        self.min_height = 30
        self.min_aspect_ratio = 1.0  # More permissive for different angles
        self.max_aspect_ratio = 6.0
        self.cluster_distance = 70
        
        # Confidence scoring
        self.height_weight = 0.3
        self.area_weight = 0.3
        self.position_weight = 0.2  # Objects higher in image more likely trees
        self.color_weight = 0.2
        
        self.get_logger().info('🎯 Robust Hybrid Detector initialized!')
        # self.get_logger().info('💪 Background-independent detection strategy')
        # self.get_logger().info('🟢 Remove green turf + 🟤 Detect brown cylinders')
        
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Robust detection with multiple strategies
            tree_count, debug_image, mask_image = self.robust_detection(cv_image)
            
            # Publish results
            tree_msg = String()
            tree_msg.data = f"trees_detected:{tree_count}"
            self.tree_pub.publish(tree_msg)
            
            # Publish debug images
            if debug_image is not None:
                debug_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding='bgr8')
                self.debug_pub.publish(debug_msg)
            
            if mask_image is not None:
                mask_msg = self.bridge.cv2_to_imgmsg(mask_image, encoding='mono8')
                self.mask_pub.publish(mask_msg)
            
            if tree_count > 0:
                self.get_logger().info(f'🌳 Detected {tree_count} brown cylinders')
                
        except Exception as e:
            self.get_logger().error(f'Detection error: {str(e)}')
    
    def robust_detection(self, image):
        """
        Multi-stage robust detection:
        1. Remove green turf (noise reduction)
        2. Detect brown cylinders (multiple color ranges)
        3. Combine and score candidates
        """
        height, width = image.shape[:2]
        
        # Stage 1: Create region of interest (remove green turf)
        roi_mask = self.create_roi_mask(image)
        
        # Stage 2: Detect brown objects in multiple ways
        brown_candidates = self.detect_brown_objects(image, roi_mask)
        
        # Stage 3: Score and filter candidates
        scored_candidates = self.score_candidates(brown_candidates, image)
        
        # Stage 4: Final filtering and clustering
        final_trees = self.final_filtering(scored_candidates)
        
        # Stage 5: Create debug visualization
        debug_image = self.create_debug_image(image, final_trees, roi_mask, brown_candidates)
        
        return len(final_trees), debug_image, roi_mask
    
    def create_roi_mask(self, image):
        """
        Create region of interest by removing green turf
        This is consistent regardless of background changes
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Detect green areas (turf)
        green_mask = cv2.inRange(hsv, self.green_hsv_lower, self.green_hsv_upper)
        
        # Dilate green mask to remove grass edges
        kernel = np.ones((7, 7), np.uint8)
        green_mask_dilated = cv2.dilate(green_mask, kernel, iterations=2)
        
        # ROI is everything NOT green
        roi_mask = cv2.bitwise_not(green_mask_dilated)
        
        # Additional filtering: focus on upper 2/3 of image (where trees are)
        height = image.shape[0]
        roi_mask[int(height * 0.8):, :] = 0  # Remove bottom 20% (likely ground)
        
        return roi_mask
    
    def detect_brown_objects(self, image, roi_mask):
        """
        Detect brown objects using multiple color ranges
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        candidates = []
        
        # Try each brown color range
        for i, (lower, upper) in enumerate(self.brown_ranges):
            # Create brown mask
            brown_mask = cv2.inRange(hsv, lower, upper)
            
            # Combine with ROI mask
            combined_mask = cv2.bitwise_and(brown_mask, roi_mask)
            
            # Clean up mask
            kernel = np.ones((3, 3), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process contours
            for contour in contours:
                candidate = self.process_contour(contour, f"brown_range_{i}")
                if candidate is not None:
                    candidates.append(candidate)
        
        # Also try general non-green detection as backup
        backup_candidates = self.detect_non_green_objects(image, roi_mask)
        candidates.extend(backup_candidates)
        
        return candidates
    
    def detect_non_green_objects(self, image, roi_mask):
        """
        Backup detection: Find any non-green objects in ROI
        """
        # Convert to LAB color space for better object separation
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Use A channel (green-red axis)
        a_channel = lab[:, :, 1]
        
        # Threshold to find non-green objects
        _, object_mask = cv2.threshold(a_channel, 132, 255, cv2.THRESH_BINARY)
        
        # Combine with ROI
        combined_mask = cv2.bitwise_and(object_mask, roi_mask)
        
        # Clean up
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for contour in contours:
            candidate = self.process_contour(contour, "non_green")
            if candidate is not None:
                candidates.append(candidate)
        
        return candidates
    
    def process_contour(self, contour, detection_method):
        """
        Process a contour and return candidate if it meets basic criteria
        """
        area = cv2.contourArea(contour)
        
        # Basic size filtering
        if area < self.min_area or area > self.max_area:
            return None
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Basic height filtering
        if h < self.min_height:
            return None
        
        # Basic aspect ratio filtering
        aspect_ratio = h / w if w > 0 else 0
        if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
            return None
        
        # Calculate center
        center_x = x + w / 2
        center_y = y + h / 2
        
        return {
            'center_x': center_x,
            'center_y': center_y,
            'width': w,
            'height': h,
            'area': area,
            'aspect_ratio': aspect_ratio,
            'method': detection_method,
            'contour': contour
        }
    
    def score_candidates(self, candidates, image):
        """
        Score candidates based on multiple criteria
        """
        if not candidates:
            return []
        
        height, width = image.shape[:2]
        
        for candidate in candidates:
            score = 0.0
            
            # Height score (taller objects more likely to be trees)
            height_score = min(1.0, candidate['height'] / 150.0)
            score += height_score * self.height_weight
            
            # Area score (reasonable size objects)
            optimal_area = 1000
            area_score = 1.0 - abs(candidate['area'] - optimal_area) / optimal_area
            area_score = max(0.0, area_score)
            score += area_score * self.area_weight
            
            # Position score (objects higher in image more likely trees)
            position_score = 1.0 - (candidate['center_y'] / height)
            score += position_score * self.position_weight
            
            # Aspect ratio score (cylindrical objects)
            optimal_ratio = 2.5
            ratio_score = 1.0 - abs(candidate['aspect_ratio'] - optimal_ratio) / optimal_ratio
            ratio_score = max(0.0, ratio_score)
            score += ratio_score * self.color_weight
            
            candidate['confidence'] = score
        
        # Return candidates with confidence > threshold
        return [c for c in candidates if c['confidence'] > 0.3]
    
    def final_filtering(self, candidates):
        """
        Final clustering and filtering
        """
        if not candidates:
            return []
        
        # Sort by confidence
        candidates.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Cluster nearby detections
        clustered = []
        used = [False] * len(candidates)
        
        for i, candidate in enumerate(candidates):
            if used[i]:
                continue
            
            # Start new cluster with highest confidence detection
            cluster = [candidate]
            used[i] = True
            
            # Find nearby detections
            for j, other in enumerate(candidates):
                if used[j]:
                    continue
                
                distance = np.sqrt((candidate['center_x'] - other['center_x'])**2 + 
                                 (candidate['center_y'] - other['center_y'])**2)
                
                if distance <= self.cluster_distance:
                    cluster.append(other)
                    used[j] = True
            
            # Take best detection from cluster
            best_detection = max(cluster, key=lambda x: x['confidence'])
            clustered.append(best_detection)
        
        return clustered
    
    def create_debug_image(self, original, final_trees, roi_mask, all_candidates):
        """
        Create comprehensive debug visualization
        """
        debug = original.copy()
        
        # Show ROI (green mask removal)
        roi_overlay = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
        roi_overlay[:, :, 1] = roi_overlay[:, :, 1] // 3  # Dim the green channel
        debug = cv2.addWeighted(debug, 0.8, roi_overlay, 0.2, 0)
        
        # Draw all candidates (small blue circles)
        for candidate in all_candidates:
            center = (int(candidate['center_x']), int(candidate['center_y']))
            cv2.circle(debug, center, 3, (255, 100, 0), -1)
        
        # Draw final detections (large green circles + rectangles)
        for i, tree in enumerate(final_trees):
            center = (int(tree['center_x']), int(tree['center_y']))
            
            # Draw center
            cv2.circle(debug, center, 8, (0, 255, 0), -1)
            
            # Draw bounding rectangle
            x = int(tree['center_x'] - tree['width']/2)
            y = int(tree['center_y'] - tree['height']/2)
            w = int(tree['width'])
            h = int(tree['height'])
            cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add label with confidence
            label = f"Tree {i+1} ({tree['confidence']:.2f})"
            cv2.putText(debug, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add title and stats
        cv2.putText(debug, 'Robust Hybrid Detection', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(debug, f'Trees: {len(final_trees)} | Candidates: {len(all_candidates)}', 
                   (10, debug.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return debug

def main(args=None):
    rclpy.init(args=args)
    
    try:
        detector = RobustHybridDetector()
        detector.get_logger().info('🚀 Starting Robust Hybrid Tree Detection')
        #detector.get_logger().info('🎯 Works with ANY background - only needs green turf + brown cylinders!')
        #detector.get_logger().info('📺 Debug: ros2 run rqt_image_view rqt_image_view /tree_detection/debug_image')
        rclpy.spin(detector)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
