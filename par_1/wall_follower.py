#!/usr/bin/env python  
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import String, Empty
import math
from tf2_ros import TransformListener, Buffer
import rclpy.duration


class WallFollowerNav(Node):
    def __init__(self):
        super().__init__('wall_follower')

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/snc_status', 10)
        self.path_pub = self.create_publisher(Path, '/path_explore', 10)

        # Subscribers
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.start_sub = self.create_subscription(Empty, '/trigger_start', self.start_callback, 10)
        self.teleop_sub = self.create_subscription(Empty, '/trigger_teleop', self.teleop_callback, 10)
        self.home_sub = self.create_subscription(Empty, '/trigger_home', self.home_callback, 10)


        self.setpoint = 0.5 
        self.follow_side = 'right'

        # PID  
        self.Kp = 1.7
        self.Ki = 0.0
        self.Kd = 0.3
            
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = self.get_clock().now()

        self.path_msg = Path()
        self.path_msg.header.frame_id = 'map'
        self.last_pose = None
        self.pose_publish_threshold = 0.1  

        self.started = False
        self.teleop_mode = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.get_logger().info("WallFollowerNav node started")
        self.publish_status("Waiting for start signal")

    def publish_status(self, text):
        """Publishes current system status to /snc_status."""
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)

   
    def start_callback(self, msg):
        """Callback to handle start trigger."""
        self.get_logger().info("Start trigger received.......")
        self.started = True
        self.publish_status("exploration started.")

    def home_callback(self, msg):
        self.get_logger().info("Home trigger received. Stopping exploration.")
        self.started = False
        self.publish_status("Exploration stopped. Awaiting return-home procedures.")

        # set cmd vel to 0
        stop_twist = Twist()
        self.cmd_pub.publish(stop_twist)
        self.get_logger().info("Published stop command to /cmd_vel")

    def teleop_callback(self, msg):
        """Callback to activate teleop mode"""
        self.get_logger().warning("teleop mode activated.")
        self.teleop_mode = True
        self.publish_status("teleop started")

    # ----------------------------
    def calculate_map_ranges(self, center_ratio, width_ratio):
        """Calculates map ranges, makes it a left right center range"""
        total_ranges = self.num_ranges
        center_index = int(center_ratio * total_ranges)
        width = int(width_ratio * total_ranges)

        start_idx = (center_index - width // 2) % total_ranges
        end_idx = (center_index + width // 2) % total_ranges

        if start_idx <= end_idx:
            return [(start_idx, end_idx)]
        else:
            return [(start_idx, total_ranges - 1), (0, end_idx)]

    def update_scan_regions(self):
        """Defines scan regions"""
        self.scan_regions = {
            'front': self.calculate_map_ranges(0.0, 0.1),
            'right': self.calculate_map_ranges(0.75, 0.05),
            'left': self.calculate_map_ranges(0.25, 0.05),
        }

    def compute_min_distance_in_region(self, ranges, index_ranges):
        """Returns the minimum distance of a region"""
        valid_values = (
            r
            for start, end in index_ranges
            for r in (ranges[start:end + 1] if start <= end else ranges[start:] + ranges[:end + 1])
            if not math.isinf(r) and not math.isnan(r)
            )
        return min(valid_values, default=float('inf'))


    def detect_and_avoid_wall(self, front_distance):
        """Detects obstacles in front and commands avoidance rotation if necessary."""
        if front_distance < 0.4:
            self.publish_status("wall in front, avoid start")
            #self.get_logger().warning("Rotating to avoid collision.")
            twist = Twist()
            twist.angular.z = 0.5 if self.follow_side == 'right' else -0.5
            self.cmd_pub.publish(twist)
            return True
        return False

    def perform_pid_wall_following(self, side_distance):
        """Applies PID control to maintain distance from the wall."""
        now = self.get_clock().now()
        dt = (now - self.prev_time).nanoseconds / 1e9

        error = self.setpoint - side_distance
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0

        angular_z = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        if self.follow_side == 'left':
            angular_z *= -1

        twist = Twist()
        twist.linear.x = 0.2
        twist.angular.z = max(min(angular_z, 0.6), -0.6)
        self.cmd_pub.publish(twist)

        self.publish_status("wall following distance calc")
        #self.get_logger().info(f"[PID Error]: {error:.2f}")
        self.prev_error = error
        self.prev_time = now

    def scan_callback(self, msg):
        if not self.started or self.teleop_mode:
            return

        ranges = list(msg.ranges)
        self.num_ranges = len(ranges)

        self.update_scan_regions()

        side_distance = self.compute_min_distance_in_region(ranges, self.scan_regions[self.follow_side])
        front_distance = self.compute_min_distance_in_region(ranges, self.scan_regions['front'])

        if self.detect_and_avoid_wall(front_distance):
            return

        self.perform_pid_wall_following(side_distance)

        self.update_pose_in_map()

   
    def get_current_pose(self):
        """Retrieves current robot pose relative to the map"""
        now = rclpy.time.Time()
        transform = self.tf_buffer.lookup_transform('odom','base_link',now,timeout=rclpy.duration.Duration(seconds=1.0))
        x = transform.transform.translation.x
        y = transform.transform.translation.y
        z = transform.transform.translation.z
        orientation = transform.transform.rotation

        return x, y, z, orientation

    def has_moved_enough(self, x, y):
        """Checks if the robot has moved enough to publish"""
        if not self.last_pose:
            return True
        dx = x - self.last_pose.pose.position.x
        dy = y - self.last_pose.pose.position.y
        dist = math.hypot(dx, dy)

        if dist >= self.pose_publish_threshold:
            return True
        else:
            return False

    def publish_new_pose(self, x, y, z, orientation):
        """Publishes the robot's current pose topic."""
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "map"
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation = orientation

        self.path_msg.header.stamp = pose.header.stamp
        self.path_msg.poses.append(pose)
        self.path_pub.publish(self.path_msg)
        self.last_pose = pose

    def update_pose_in_map(self):
        """Updates the robotpose"""
        try:
            x, y, z, orientation = self.get_current_pose()

            if not self.has_moved_enough(x, y):
                return

            self.publish_new_pose(x, y, z, orientation)

        except Exception as e:
            self.get_logger().warning(f"[TF2] Transform lookup failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = WallFollowerNav()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
