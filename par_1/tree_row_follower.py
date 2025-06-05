#!/usr/bin/env python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import String, Empty
import math
import numpy as np # For easier array manipulation
from tf2_ros import TransformListener, Buffer
import rclpy.duration

class TreeRowFollowerNav(Node):
    def __init__(self):
        super().__init__('tree_row_follower')

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/snc_status', 10)
        self.path_pub = self.create_publisher(Path, '/path_explore', 10)

        # Subscribers
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.start_sub = self.create_subscription(Empty, '/trigger_start', self.start_callback, 10)
        self.teleop_sub = self.create_subscription(Empty, '/trigger_teleop', self.teleop_callback, 10)
        self.home_sub = self.create_subscription(Empty, '/trigger_home', self.home_callback, 10)

        # --- Parameters for Tree Following ---
        self.setpoint_dist_to_tree = 0.8  # Desired distance to the side of a tree
        self.follow_side = 'right'      # 'left' or 'right'

        # PID for distance control
        self.Kp_dist = 1.5
        self.Ki_dist = 0.01
        self.Kd_dist = 0.2
        self.integral_dist = 0.0
        self.prev_error_dist = 0.0

        # P-controller for alignment (angle to tree)
        self.Kp_angle = 0.8 # Proportional gain for steering towards tree's desired bearing
        # Target bearing: 0 rad is front. Negative for right, positive for left.
        self.target_bearing_rad = -math.pi / 2.1 if self.follow_side == 'right' else math.pi / 2.1

        self.prev_time = self.get_clock().now()

        # Tree Detection Parameters
        self.min_tree_detection_dist = 0.2  # Min distance to consider an object a tree
        self.max_tree_detection_dist = 2.0  # Max distance
        self.min_tree_points = 3            # Min number of consecutive scan points to be a tree
        self.max_tree_points = 20           # Max number of points (related to tree width & distance)
        # FOV for tree detection (relative to robot's front)
        # e.g. -60 to +60 degrees if following right, or -30 to 90 deg
        self.detection_angle_min_rad = -math.pi / 2.5 # More forward-looking
        self.detection_angle_max_rad = math.pi / 2.5

        # Path publishing
        self.path_msg = Path()
        self.path_msg.header.frame_id = 'map' # Or 'odom' if map is not available
        self.last_pose = None
        self.pose_publish_threshold = 0.1

        self.started = False
        self.teleop_mode = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.current_scan_angle_increment = 0.0 # Will be updated from scan msg

        self.get_logger().info(f"TreeRowFollowerNav node started. Following {self.follow_side} side.")
        self.publish_status("Waiting for start signal")

    def publish_status(self, text):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)

    def start_callback(self, msg):
        self.get_logger().info("Start trigger received.......")
        self.started = True
        self.teleop_mode = False # Ensure teleop is off
        self.publish_status("Tree row exploration started.")
        # Reset PID
        self.integral_dist = 0.0
        self.prev_error_dist = 0.0
        self.prev_time = self.get_clock().now()


    def home_callback(self, msg):
        self.get_logger().info("Home trigger received. Stopping exploration.")
        self.started = False
        self.publish_status("Exploration stopped. Awaiting return-home procedures.")
        stop_twist = Twist()
        self.cmd_pub.publish(stop_twist)
        self.get_logger().info("Published stop command to /cmd_vel")

    def teleop_callback(self, msg):
        self.get_logger().warning("Teleop mode activated.")
        self.teleop_mode = True
        self.started = False # Stop autonomous behavior
        self.publish_status("Teleop mode active.")
        stop_twist = Twist() # Stop any current movement
        self.cmd_pub.publish(stop_twist)

    def detect_trees(self, ranges_full, angle_min_scan, angle_increment):
        """
        Detects potential tree candidates from laser scan data.
        Returns a list of tuples: (distance_to_center, angle_to_center_rad, num_points)
        Angle is relative to robot's front (0 rad).
        """
        self.current_scan_angle_increment = angle_increment # Store for other uses if needed
        candidates = []
        num_ranges = len(ranges_full)

        # Determine scan indices for our detection FOV
        # Angle relative to robot's front (0 rad)
        # scan_angle = angle_min_scan + index * angle_increment
        # index = (scan_angle - angle_min_scan) / angle_increment

        start_idx_fov = int(max(0, (self.detection_angle_min_rad - angle_min_scan) / angle_increment))
        end_idx_fov = int(min(num_ranges -1, (self.detection_angle_max_rad - angle_min_scan) / angle_increment))

        if start_idx_fov >= end_idx_fov :
            self.get_logger().warn("Detection FOV is invalid, check angle_min/max_rad.")
            return []

        # Use a slice for easier indexing within the FOV
        ranges_fov = ranges_full[start_idx_fov : end_idx_fov + 1]
        
        current_segment = []
        for i, r_dist in enumerate(ranges_fov):
            if self.min_tree_detection_dist < r_dist < self.max_tree_detection_dist:
                current_segment.append(i) # Store index within ranges_fov
            else:
                if self.min_tree_points <= len(current_segment) <= self.max_tree_points:
                    # Valid segment found
                    # Calculate center point of the segment in terms of original scan indices
                    segment_original_indices = [idx + start_idx_fov for idx in current_segment]
                    center_original_idx = segment_original_indices[len(segment_original_indices) // 2]
                    
                    dist_to_center = ranges_full[center_original_idx]
                    angle_to_center_rad = angle_min_scan + center_original_idx * angle_increment
                    
                    candidates.append((dist_to_center, angle_to_center_rad, len(current_segment)))
                current_segment = []
        
        # Check last segment
        if self.min_tree_points <= len(current_segment) <= self.max_tree_points:
            segment_original_indices = [idx + start_idx_fov for idx in current_segment]
            center_original_idx = segment_original_indices[len(segment_original_indices) // 2]
            dist_to_center = ranges_full[center_original_idx]
            angle_to_center_rad = angle_min_scan + center_original_idx * angle_increment
            candidates.append((dist_to_center, angle_to_center_rad, len(current_segment)))
            
        return candidates

    def select_target_tree(self, tree_candidates):
        """
        Selects the best tree to follow from the candidates.
        Returns (distance, angle_rad) of the target tree, or None.
        """
        if not tree_candidates:
            return None

        best_tree = None
        min_score = float('inf')

        for dist, angle_rad, _num_points in tree_candidates:
            # Prioritize trees on the desired follow_side
            # And closer to the target bearing
            if self.follow_side == 'right':
                if angle_rad > 0: # Tree is to the left, less desirable
                    angle_penalty = abs(angle_rad) * 2.0 # Heavier penalty
                else: # Tree is to the right or front
                    angle_penalty = 0.0
            else: # follow_side == 'left'
                if angle_rad < 0: # Tree is to the right, less desirable
                    angle_penalty = abs(angle_rad) * 2.0
                else: # Tree is to the left or front
                    angle_penalty = 0.0
            
            # Score: combination of distance error from setpoint and angular error from target bearing
            # Lower score is better
            score = abs(dist - self.setpoint_dist_to_tree) + \
                    abs(angle_rad - self.target_bearing_rad) * 0.5 + \
                    angle_penalty # Weight angular deviation less than distance deviation

            if score < min_score:
                min_score = score
                best_tree = (dist, angle_rad)
        
        return best_tree

    def detect_front_obstacle(self, ranges, angle_min, angle_increment):
        """Detects obstacles directly in front."""
        num_ranges = len(ranges)
        # Front region: e.g., -15 to +15 degrees
        front_angle_range_rad = math.radians(15)
        
        center_idx = int((-angle_min) / angle_increment) # Index roughly corresponding to 0 rad
        half_width_pts = int(front_angle_range_rad / angle_increment / 2)
        
        start_idx = max(0, center_idx - half_width_pts)
        end_idx = min(num_ranges - 1, center_idx + half_width_pts)

        if start_idx >= end_idx: return float('inf') # Should not happen with valid scan

        min_front_dist = float('inf')
        for i in range(start_idx, end_idx + 1):
            if not math.isinf(ranges[i]) and not math.isnan(ranges[i]):
                min_front_dist = min(min_front_dist, ranges[i])
        
        return min_front_dist


    def scan_callback(self, msg):
        if not self.started or self.teleop_mode:
            return

        ranges = list(msg.ranges) # Make it a mutable list
        # Replace inf/nan with a large number for easier processing, but before obstacle detection
        # For tree detection, we filter inf/nan within the function.
        # For front obstacle, we handle it there.

        # 1. Detect front obstacle
        front_distance = self.detect_front_obstacle(ranges, msg.angle_min, msg.angle_increment)
        if front_distance < 0.4: # Obstacle too close in front
            self.publish_status("Front obstacle! Turning.")
            self.get_logger().warn(f"Front obstacle at {front_distance:.2f}m. Turning.")
            twist = Twist()
            # Turn away from the follow_side to potentially find a path around
            twist.angular.z = -0.4 if self.follow_side == 'right' else 0.4
            self.cmd_pub.publish(twist)
            return

        # 2. Detect potential trees
        # angle_min and angle_increment come from the LaserScan message
        tree_candidates = self.detect_trees(ranges, msg.angle_min, msg.angle_increment)

        if not tree_candidates:
            self.publish_status("No trees detected. Searching...")
            self.get_logger().info("No trees detected. Searching...")
            twist = Twist()
            twist.linear.x = 0.05 # Slow crawl
            # Slow turn towards the side we expect trees
            twist.angular.z = -0.2 if self.follow_side == 'right' else 0.2
            self.cmd_pub.publish(twist)
            return

        # 3. Select the target tree
        target_tree = self.select_target_tree(tree_candidates)

        if target_tree is None:
            self.publish_status("Could not select a target tree. Searching...")
            self.get_logger().info("Could not select a target tree. Searching...")
            twist = Twist()
            twist.linear.x = 0.05
            twist.angular.z = -0.2 if self.follow_side == 'right' else 0.2
            self.cmd_pub.publish(twist)
            return

        dist_to_target_tree, angle_to_target_tree_rad = target_tree
        self.publish_status(f"Targeting tree at {dist_to_target_tree:.2f}m, {math.degrees(angle_to_target_tree_rad):.1f}deg")
        # self.get_logger().info(f"Target tree: dist={dist_to_target_tree:.2f}, angle={math.degrees(angle_to_target_tree_rad):.1f}deg")


        # 4. PID and P-control for movement
        now = self.get_clock().now()
        dt = (now - self.prev_time).nanoseconds / 1e9
        if dt <= 0: # Avoid division by zero if time hasn't passed
            self.get_logger().warn("dt is zero or negative, skipping PID step.")
            return 
            
        # PID for distance control
        error_dist = self.setpoint_dist_to_tree - dist_to_target_tree
        self.integral_dist += error_dist * dt
        self.integral_dist = max(min(self.integral_dist, 1.0), -1.0) # Anti-windup for integral
        derivative_dist = (error_dist - self.prev_error_dist) / dt

        pid_angular_z_dist = self.Kp_dist * error_dist + \
                             self.Ki_dist * self.integral_dist + \
                             self.Kd_dist * derivative_dist

        # P-control for alignment (steering to keep tree at target_bearing_rad)
        error_angle = self.target_bearing_rad - angle_to_target_tree_rad
        p_angular_z_align = self.Kp_angle * error_angle
        
        # Combine controls
        # If following right, pid_angular_z_dist should be positive to turn left (away from tree) if too close
        # and negative to turn right (towards tree) if too far.
        # Angle error is positive if tree is too far "left" of target bearing.
        # For right side following, target bearing is negative. If tree angle is less negative (or positive),
        # error_angle (target_bearing - tree_angle) will be more negative, so p_angular_z_align turns right.
        # This seems correct.

        final_angular_z = pid_angular_z_dist + p_angular_z_align
        
        twist = Twist()
        twist.linear.x = 0.12  # Constant moderate speed when following
        twist.angular.z = np.clip(final_angular_z, -0.7, 0.7) # Limit angular velocity

        self.cmd_pub.publish(twist)

        # self.get_logger().info(f"Cmd: lin.x={twist.linear.x:.2f}, ang.z={twist.angular.z:.2f} | "
        #                        f"ErrDist: {error_dist:.2f}, ErrAng: {math.degrees(error_angle):.1f} | "
        #                        f"PID_dist: {pid_angular_z_dist:.2f}, P_align: {p_angular_z_align:.2f}")

        self.prev_error_dist = error_dist
        self.prev_time = now

        self.update_pose_in_map()


    # --- Pose and Path methods (largely unchanged from WallFollower) ---
    def get_current_pose(self):
        try:
            now = rclpy.time.Time()
            # Ensure 'base_link' exists and 'odom' or 'map' is appropriate
            transform = self.tf_buffer.lookup_transform('odom', 'base_link', now, timeout=rclpy.duration.Duration(seconds=0.1))
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            z = transform.transform.translation.z
            orientation = transform.transform.rotation
            return x, y, z, orientation
        except Exception as e:
            self.get_logger().debug(f"TF lookup failed: odom to base_link: {e}")
            return None

    def has_moved_enough(self, x, y):
        if not self.last_pose:
            return True
        dx = x - self.last_pose.pose.position.x
        dy = y - self.last_pose.pose.position.y
        return math.hypot(dx, dy) >= self.pose_publish_threshold

    def publish_new_pose(self, x, y, z, orientation):
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = self.path_msg.header.frame_id # Use 'odom' or 'map'
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation = orientation

        self.path_msg.header.stamp = pose.header.stamp
        self.path_msg.poses.append(pose)
        self.path_pub.publish(self.path_msg)
        self.last_pose = pose

    def update_pose_in_map(self):
        current_pose_data = self.get_current_pose()
        if current_pose_data:
            x, y, z, orientation = current_pose_data
            if self.has_moved_enough(x, y):
                self.publish_new_pose(x, y, z, orientation)


def main(args=None):
    rclpy.init(args=args)
    node = TreeRowFollowerNav()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Publish a zero velocity command before shutting down
        stop_twist = Twist()
        node.cmd_pub.publish(stop_twist)
        node.get_logger().info("Published zero velocity before shutdown.")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()