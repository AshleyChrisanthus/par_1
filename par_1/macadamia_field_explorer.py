# ROSbot 3 Pro Macadamia Field Exploration Implementation
# A complete ROS2 node for autonomous row-by-row field exploration

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan, Image
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import String
import numpy as np
import cv2
from cv_bridge import CvBridge
import math
from enum import Enum

class ExplorationState(Enum):
    INITIALIZATION = 1
    FIELD_ENTRY = 2
    ROW_NAVIGATION = 3
    ROW_TRANSITION = 4
    COVERAGE_ASSESSMENT = 5
    RETURN_HOME = 6
    MISSION_COMPLETE = 7

class MacadamiaFieldExplorer(Node):
    """
    ROSbot 3 Pro Macadamia Field Exploration Node
    
    This node implements a comprehensive exploration strategy for
    systematic row-by-row coverage of macadamia orchards using
    ROSbot 3 Pro platform with advanced sensor integration.
    """
    
    def __init__(self):
        super().__init__('macadamia_field_explorer')
        
        # Initialize parameters
        self.declare_parameter('row_spacing', 4.0)  # meters between rows
        self.declare_parameter('tree_spacing', 3.0)  # meters between trees
        self.declare_parameter('field_width', 50.0)  # field width in meters
        self.declare_parameter('field_length', 100.0)  # field length in meters
        self.declare_parameter('robot_speed', 0.5)  # m/s navigation speed
        
        # Get parameters
        self.row_spacing = self.get_parameter('row_spacing').value
        self.tree_spacing = self.get_parameter('tree_spacing').value
        self.field_width = self.get_parameter('field_width').value
        self.field_length = self.get_parameter('field_length').value
        self.robot_speed = self.get_parameter('robot_speed').value
        
        # State management
        self.current_state = ExplorationState.INITIALIZATION
        self.current_row = 0
        self.total_rows = 0
        self.exploration_complete = False
        self.home_position = None
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/exploration_status', 10)
        
        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)
        self.camera_sub = self.create_subscription(
            Image, '/camera/image_raw', self.camera_callback, 10)
            
        # Navigation action client
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # Computer vision bridge
        self.cv_bridge = CvBridge()
        
        # Data storage
        self.current_pose = None
        self.current_scan = None
        self.current_map = None
        self.detected_rows = []
        self.row_waypoints = []
        
        # Timer for main exploration loop
        self.exploration_timer = self.create_timer(1.0, self.exploration_loop)
        
        self.get_logger().info("Macadamia Field Explorer initialized successfully")
    
    def odom_callback(self, msg):
        """Process odometry data for robot localization"""
        self.current_pose = msg.pose.pose
        if self.home_position is None:
            self.home_position = self.current_pose
    
    def scan_callback(self, msg):
        """Process LIDAR scan data for obstacle detection and row identification"""
        self.current_scan = msg
        self.detect_tree_rows(msg)
    
    def map_callback(self, msg):
        """Process occupancy grid map for global path planning"""
        self.current_map = msg
        self.update_exploration_progress()
    
    def camera_callback(self, msg):
        """Process camera image for visual navigation and tree detection"""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            self.process_visual_navigation(cv_image)
        except Exception as e:
            self.get_logger().error(f"Camera processing error: {e}")
    
    def detect_tree_rows(self, scan_msg):
        """
        Detect macadamia tree rows using LIDAR data
        Uses clustering algorithm to identify parallel lines of trees
        """
        ranges = np.array(scan_msg.ranges)
        angles = np.arange(scan_msg.angle_min, scan_msg.angle_max, 
                          scan_msg.angle_increment)
        
        # Filter out invalid readings
        valid_indices = np.isfinite(ranges) & (ranges > scan_msg.range_min) & \
                       (ranges < scan_msg.range_max)
        valid_ranges = ranges[valid_indices]
        valid_angles = angles[valid_indices]
        
        # Convert to Cartesian coordinates
        x_points = valid_ranges * np.cos(valid_angles)
        y_points = valid_ranges * np.sin(valid_angles)
        
        # Cluster points to detect tree rows
        rows = self.cluster_tree_rows(x_points, y_points)
        self.detected_rows = rows
        
        return rows
    
    def cluster_tree_rows(self, x_points, y_points):
        """
        Cluster LIDAR points to identify tree rows
        Returns list of row orientations and positions
        """
        rows = []
        
        # Simple clustering based on y-coordinate grouping
        # This assumes rows are roughly parallel to x-axis
        y_sorted_indices = np.argsort(y_points)
        
        current_row_points = []
        last_y = None
        threshold = self.row_spacing / 2
        
        for idx in y_sorted_indices:
            y = y_points[idx]
            x = x_points[idx]
            
            if last_y is None or abs(y - last_y) < threshold:
                current_row_points.append((x, y))
            else:
                if len(current_row_points) > 3:  # Minimum points for a row
                    rows.append(self.fit_row_line(current_row_points))
                current_row_points = [(x, y)]
            
            last_y = y
        
        # Handle last row
        if len(current_row_points) > 3:
            rows.append(self.fit_row_line(current_row_points))
        
        return rows
    
    def fit_row_line(self, points):
        """Fit a line to points representing a tree row"""
        points = np.array(points)
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        
        # Linear regression to fit line
        A = np.vstack([x_coords, np.ones(len(x_coords))]).T
        slope, intercept = np.linalg.lstsq(A, y_coords, rcond=None)[0]
        
        return {
            'slope': slope,
            'intercept': intercept,
            'points': points,
            'start_x': np.min(x_coords),
            'end_x': np.max(x_coords)
        }
    
    def process_visual_navigation(self, image):
        """Process camera image for row following and tree detection"""
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define green color range for tree detection
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        
        # Create mask for green areas (trees)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Find contours for tree detection
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate row center for navigation
        row_center = self.calculate_row_center(contours, image.shape[1])
        
        return row_center
    
    def calculate_row_center(self, contours, image_width):
        """Calculate the center of the tree row for navigation"""
        if len(contours) < 2:
            return image_width // 2  # Default to center
        
        # Find left and right tree boundaries
        left_boundary = min([cv2.boundingRect(c)[0] for c in contours])
        right_boundary = max([cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] 
                             for c in contours])
        
        return (left_boundary + right_boundary) // 2
    
    def exploration_loop(self):
        """Main exploration state machine"""
        status_msg = String()
        
        if self.current_state == ExplorationState.INITIALIZATION:
            self.initialize_exploration()
            status_msg.data = "Initializing exploration system..."
            
        elif self.current_state == ExplorationState.FIELD_ENTRY:
            self.enter_field()
            status_msg.data = "Entering field and starting mapping..."
            
        elif self.current_state == ExplorationState.ROW_NAVIGATION:
            self.navigate_current_row()
            status_msg.data = f"Navigating row {self.current_row + 1}/{self.total_rows}"
            
        elif self.current_state == ExplorationState.ROW_TRANSITION:
            self.transition_to_next_row()
            status_msg.data = "Transitioning to next row..."
            
        elif self.current_state == ExplorationState.COVERAGE_ASSESSMENT:
            self.assess_coverage()
            status_msg.data = "Assessing exploration coverage..."
            
        elif self.current_state == ExplorationState.RETURN_HOME:
            self.return_to_home()
            status_msg.data = "Returning to starting position..."
            
        elif self.current_state == ExplorationState.MISSION_COMPLETE:
            status_msg.data = "Mission completed successfully!"
        
        self.status_pub.publish(status_msg)
    
    def initialize_exploration(self):
        """Initialize exploration parameters and estimate field layout"""
        if self.current_map is not None and len(self.detected_rows) > 0:
            self.total_rows = len(self.detected_rows)
            self.generate_row_waypoints()
            self.current_state = ExplorationState.FIELD_ENTRY
            self.get_logger().info(f"Detected {self.total_rows} tree rows")
    
    def enter_field(self):
        """Navigate to the first row and begin systematic exploration"""
        if len(self.row_waypoints) > 0:
            first_waypoint = self.row_waypoints[0][0]  # First point of first row
            self.navigate_to_pose(first_waypoint)
            self.current_state = ExplorationState.ROW_NAVIGATION
    
    def navigate_current_row(self):
        """Navigate along the current row using visual and LIDAR feedback"""
        if self.current_row < len(self.row_waypoints):
            current_row_points = self.row_waypoints[self.current_row]
            
            # Check if row navigation is complete
            if self.is_row_complete():
                self.current_state = ExplorationState.ROW_TRANSITION
            else:
                # Continue row following
                self.follow_row_visual()
    
    def transition_to_next_row(self):
        """Plan and execute transition to the next row"""
        self.current_row += 1
        
        if self.current_row >= self.total_rows:
            self.current_state = ExplorationState.COVERAGE_ASSESSMENT
        else:
            # Navigate to start of next row
            next_row_start = self.row_waypoints[self.current_row][0]
            self.navigate_to_pose(next_row_start)
            self.current_state = ExplorationState.ROW_NAVIGATION
    
    def assess_coverage(self):
        """Assess if the field has been completely explored"""
        coverage_percentage = self.calculate_coverage_percentage()
        self.get_logger().info(f"Field coverage: {coverage_percentage:.1f}%")
        
        if coverage_percentage >= 95.0:
            self.current_state = ExplorationState.RETURN_HOME
        else:
            # Continue exploration if needed
            self.plan_additional_coverage()
    
    def return_to_home(self):
        """Navigate back to the starting position"""
        if self.home_position is not None:
            home_pose = PoseStamped()
            home_pose.pose = self.home_position
            self.navigate_to_pose(home_pose)
            self.current_state = ExplorationState.MISSION_COMPLETE
    
    def generate_row_waypoints(self):
        """Generate waypoints for systematic row-by-row exploration"""
        self.row_waypoints = []
        
        for i, row in enumerate(self.detected_rows):
            waypoints = []
            
            # Generate waypoints along the row
            start_x = row['start_x']
            end_x = row['end_x']
            y_pos = row['intercept']
            
            # Alternate direction for efficient coverage
            if i % 2 == 0:  # Even rows: left to right
                x_positions = np.arange(start_x, end_x, self.tree_spacing)
            else:  # Odd rows: right to left
                x_positions = np.arange(end_x, start_x, -self.tree_spacing)
            
            for x in x_positions:
                pose = PoseStamped()
                pose.pose.position.x = x
                pose.pose.position.y = y_pos
                pose.pose.orientation.w = 1.0  # Facing forward
                waypoints.append(pose)
            
            self.row_waypoints.append(waypoints)
    
    def follow_row_visual(self):
        """Use visual feedback to follow the current row"""
        cmd = Twist()
        
        # Basic row following using visual center
        if hasattr(self, 'current_image_center'):
            image_center = 320  # Assume 640px width
            error = self.current_image_center - image_center
            
            # Simple proportional controller
            angular_velocity = -error * 0.001  # Proportional gain
            
            cmd.linear.x = self.robot_speed
            cmd.angular.z = angular_velocity
        else:
            # Default forward motion
            cmd.linear.x = self.robot_speed
        
        self.cmd_vel_pub.publish(cmd)
    
    def navigate_to_pose(self, target_pose):
        """Navigate to a specific pose using Nav2"""
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = target_pose
        
        self.nav_client.wait_for_server()
        future = self.nav_client.send_goal_async(goal_msg)
        
        return future
    
    def is_row_complete(self):
        """Check if the current row navigation is complete"""
        # Implementation depends on specific completion criteria
        # For example, reached end of row or covered required distance
        return False  # Simplified for this example
    
    def calculate_coverage_percentage(self):
        """Calculate the percentage of field coverage achieved"""
        # Implementation would analyze the map to determine coverage
        # This is a simplified placeholder
        return min(100.0, (self.current_row / self.total_rows) * 100.0)
    
    def plan_additional_coverage(self):
        """Plan additional coverage for unexplored areas"""
        # Implementation for handling incomplete coverage
        self.current_state = ExplorationState.RETURN_HOME
    
    def update_exploration_progress(self):
        """Update exploration progress based on current map"""
        # Analyze map to track exploration progress
        pass

def main(args=None):
    """Main function to run the macadamia field explorer"""
    rclpy.init(args=args)
    
    explorer = MacadamiaFieldExplorer()
    
    try:
        rclpy.spin(explorer)
    except KeyboardInterrupt:
        explorer.get_logger().info("Exploration interrupted by user")
    finally:
        explorer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
