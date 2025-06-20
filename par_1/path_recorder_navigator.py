#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import Path
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from std_msgs.msg import String, Empty
import tf2_ros
import math


class PathRecorderNavigator(Node):
    """
    Combined node that records the robot's path and follows it in reverse when triggered.
    """
    def __init__(self):
        super().__init__('path_recorder_navigator')
        
        # ---- Path Recording Components ----
        
        # Configure the path message
        self.path = Path()
        self.path.header.frame_id = 'map'
        self.recorded_poses = []
        self.last_position = None
        
        # Set up TF2 listener
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Set up path publisher (will publish continuously)
        self.path_publisher = self.create_publisher(Path, '/path_explore', 10)
        
        # Create timer for tracking (1 Hz)
        self.track_timer = self.create_timer(1.0, self.track_position)
        
        # ---- Navigation Components ----
        
        # Initialize waypoints list and navigation status
        self.waypoints = []
        self.current_waypoint_index = 0
        self.is_navigating = False
        
        # Create contingency trigger subscriber
        self.trigger_home_sub = self.create_subscription(
            Empty,
            '/trigger_home',
            self.trigger_home_callback,
            10
        )
        
        # Create Nav2 action client
        self._action_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        
        # Create status publisher
        self.status_publisher = self.create_publisher(
            String, 
            '/snc_status', 
            10
        )
        
        # Create return path publisher
        self.return_path_publisher = self.create_publisher(
            Path, 
            '/path_return', 
            10
        )
        
        # Initialize return path message
        self.return_path = Path()
        self.return_path.header.frame_id = 'map'
        self.return_path.poses = []
        
        self.get_logger().info('Path recorder and navigator initialized')
        self.get_logger().info('Recording path continuously and publishing to /path_explore')
        self.get_logger().info('Manual contingency trigger available on /trigger_home')
        
        # Publish initial status
        self.publish_status("Initialized - Recording path")
    
    # ---- Path Recording Methods ----
    
    def track_position(self):
        """
        Check the robot's current position and add it to the path if it has moved.
        """
        try:
            # Get the latest transform
            transform = self.tf_buffer.lookup_transform(
                'map',               # Target frame
                'base_link',         # Source frame
                rclpy.time.Time(seconds=0),  # Get latest available transform
                timeout=rclpy.duration.Duration(seconds=1.0)  # Timeout
            )
            
            # Current time for the pose timestamp
            current_time = self.get_clock().now()
            
            # Extract position
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            
            # Check if position has changed enough to record
            record_position = False
            if self.last_position is None:
                # First position
                record_position = True
            else:
                # Calculate distance from last position
                dx = x - self.last_position[0]
                dy = y - self.last_position[1]
                distance = math.sqrt(dx*dx + dy*dy)
                
                # Record if moved at least 0.5 meters
                if distance >= 0.8:
                    record_position = True
            
            if record_position:
                # Create pose message
                pose = PoseStamped()
                pose.header.frame_id = 'map'
                pose.header.stamp = current_time.to_msg()
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.position.z = transform.transform.translation.z
                pose.pose.orientation = transform.transform.rotation
                
                # Add to path
                self.recorded_poses.append(pose)
                self.last_position = (x, y)
                
                # Update path
                self.path.header.stamp = current_time.to_msg()
                self.path.poses = self.recorded_poses
                
                self.get_logger().info(f'Recorded position: ({x:.2f}, {y:.2f}), Total points: {len(self.recorded_poses)}')
                
                # Publish the updated path
                self.path_publisher.publish(self.path)
                self.get_logger().debug(f'Published updated path with {len(self.recorded_poses)} points')
            
        except tf2_ros.TransformException as ex:
            self.get_logger().warning(f'Could not transform: {ex}')
    
    def get_recorded_path(self):
        """
        Return the recorded path.
        """
        return self.recorded_poses
    
    # ---- Navigation Methods ----
    
    def create_quaternion_from_yaw(self, yaw):
        """
        Convert a yaw angle to a quaternion orientation.
        """
        # For a 2D robot, we only care about rotation around Z axis (yaw)
        return Quaternion(
            x=0.0,
            y=0.0, 
            z=math.sin(yaw/2.0),
            w=math.cos(yaw/2.0)
        )
    
    def trigger_home_callback(self, msg):
        """
        Callback for when the manual contingency trigger is received.
        Starts the return-to-home process using the recorded path.
        """
        if self.is_navigating:
            self.get_logger().info('Already navigating - ignoring contingency trigger')
            self.publish_status("Already navigating - ignoring contingency trigger")
            return
            
        if len(self.recorded_poses) == 0:
            self.get_logger().warn('No path available for return journey - cannot trigger home')
            self.publish_status("No path available - cannot return home")
            return
            
        self.get_logger().info('Manual contingency trigger received! Starting return journey')
        self.publish_status("Manual trigger received - Starting return journey")
        
        # Process the path directly (no need to pass as argument since we have it internally)
        self.start_return_journey()
    
    def start_return_journey(self):
        """
        Start the return journey using the recorded path.
        """
        # Extract and reverse waypoints
        self.waypoints = list(reversed(self.recorded_poses))
        num_waypoints = len(self.waypoints)
        
        if num_waypoints == 0:
            self.get_logger().warn('Path is empty - nothing to navigate')
            self.publish_status("Path is empty - nothing to navigate")
            return
        
        # Update orientations to face the next waypoint
        self.update_waypoint_orientations()
        
        self.get_logger().info(f'Starting return journey with {num_waypoints} waypoints')
        self.publish_status(f"Starting return journey with {num_waypoints} waypoints")
        
        # Reset navigation state
        self.current_waypoint_index = 0
        self.is_navigating = True
        
        # Clear return path
        self.return_path.poses = []
        
        # Start navigation to first waypoint
        self.navigate_to_next_waypoint()
    
    def update_waypoint_orientations(self):
        """
        Update orientations of waypoints to face toward the next waypoint.
        """
        num_waypoints = len(self.waypoints)
        if num_waypoints <= 1:
            return
        
        # Process all waypoints except the last one
        for i in range(num_waypoints - 1):
            # Current and next waypoint positions
            current_pos = self.waypoints[i].pose.position
            next_pos = self.waypoints[i+1].pose.position
            
            # Calculate direction vector
            dx = next_pos.x - current_pos.x
            dy = next_pos.y - current_pos.y
            
            # Calculate yaw (heading) angle
            yaw = math.atan2(dy, dx)
            
            # Update orientation
            self.waypoints[i].pose.orientation = self.create_quaternion_from_yaw(yaw)
        
        # For last waypoint, use the same orientation as the second-to-last waypoint
        if num_waypoints > 1:
            self.waypoints[num_waypoints-1].pose.orientation = self.waypoints[num_waypoints-2].pose.orientation
    
    def navigate_to_next_waypoint(self):
        """
        Navigate to the next waypoint in the sequence.
        """
        if not self.is_navigating:
            return
        
        if self.current_waypoint_index >= len(self.waypoints):
            self.get_logger().info("Navigation complete! Reached all waypoints.")
            self.publish_status("Navigation complete! Reached all waypoints.")
            self.is_navigating = False
            return
        
        # Get current waypoint
        current_waypoint = self.waypoints[self.current_waypoint_index]
        
        # Create navigation goal
        goal_pose = NavigateToPose.Goal()
        goal_pose.pose = current_waypoint
        
        self.get_logger().info(f'Navigating to waypoint {self.current_waypoint_index + 1}/{len(self.waypoints)}')
        self.publish_status(f"Navigating to waypoint {self.current_waypoint_index + 1}/{len(self.waypoints)}")
        
        # Wait for the action server
        if not self._action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('Navigation action server not available')
            self.publish_status("Error: Navigation server not available")
            self.is_navigating = False
            return
        
        # Send the goal
        self._send_goal_future = self._action_client.send_goal_async(
            goal_pose,
            feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)
    
    def goal_response_callback(self, future):
        """
        Callback for handling the response to a navigation goal request.
        """
        goal_handle = future.result()
        
        if not goal_handle.accepted:
            self.get_logger().warn('Navigation goal was rejected')
            self.publish_status(f"Waypoint {self.current_waypoint_index + 1} rejected")
            
            # Move to the next waypoint
            self.current_waypoint_index += 1
            self.navigate_to_next_waypoint()
            return
        
        self.get_logger().info('Navigation goal accepted')
        
        # Get the result future
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)
    
    def get_result_callback(self, future):
        """
        Callback for handling the result of a navigation goal.
        """
        status = future.result().status
        
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f'Successfully reached waypoint {self.current_waypoint_index + 1}')
            self.publish_status(f"Reached waypoint {self.current_waypoint_index + 1}")
            
            # Add the current waypoint to the return path
            current_waypoint = self.waypoints[self.current_waypoint_index]
            self.return_path.poses.append(current_waypoint)
            self.return_path.header.stamp = self.get_clock().now().to_msg()
            
            # Publish the updated return path
            self.return_path_publisher.publish(self.return_path)
            self.get_logger().info(f'Published return path (now {len(self.return_path.poses)} points)')
            
        else:
            self.get_logger().warn(f'Navigation to waypoint {self.current_waypoint_index + 1} failed with status: {status}')
            self.publish_status(f"Failed waypoint {self.current_waypoint_index + 1}")
        
        # Move to the next waypoint regardless of success or failure
        self.current_waypoint_index += 1
        self.navigate_to_next_waypoint()
    
    def feedback_callback(self, feedback_msg):
        """
        Callback for handling navigation feedback.
        """
        # We could process feedback here if needed
        pass
    
    def publish_status(self, status_text):
        """
        Publish status message.
        """
        msg = String()
        msg.data = status_text
        self.status_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    
    node = PathRecorderNavigator()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
