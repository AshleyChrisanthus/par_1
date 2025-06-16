#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from nav_msgs.msg import Path
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from std_msgs.msg import String, Empty
import math
from geometry_msgs.msg import Quaternion


class ReverseWaypointFollower(Node):
    """
    Node that follows a series of waypoints in reverse order.
    Subscribes to a path and navigates through its waypoints in reverse when triggered.
    """
    def __init__(self):
        super().__init__('reverse_waypoint_follower')
        
        # Initialize waypoints list and navigation status
        self.waypoints = []
        self.current_waypoint_index = 0
        self.is_navigating = False
        
        # Create path subscriber (continuously receiving path updates)
        self.path_sub = self.create_subscription(
            Path,
            '/path_explore',
            self.path_callback,
            10
        )
        
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
        
        # Store the latest received path
        self.latest_path = None
        
        self.get_logger().info('Reverse waypoint follower initialized')
        self.get_logger().info('Continuously receiving path updates from /path_explore')
        self.get_logger().info('Manual contingency trigger available on /trigger_home')
        
        # Publish initial status
        self.publish_status("Initialized - Waiting for trigger")
    
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
        Starts the return-to-home process using the latest path.
        """
        if self.is_navigating:
            self.get_logger().info('Already navigating - ignoring contingency trigger')
            self.publish_status("Already navigating - ignoring contingency trigger")
            return
            
        if self.latest_path is None or len(self.latest_path.poses) == 0:
            self.get_logger().warn('No path available for return journey - cannot trigger home')
            self.publish_status("No path available - cannot return home")
            return
            
        self.get_logger().info('Manual contingency trigger received! Starting return journey')
        self.publish_status("Manual trigger received - Starting return journey")
        
        # Process the latest path
        self.start_return_journey(self.latest_path)
    
    def path_callback(self, msg):
        """
        Callback for receiving path updates.
        Store the path but don't start navigation automatically.
        """
        # Store the latest path
        self.latest_path = msg
        
        path_points = len(msg.poses)
        if path_points > 0:
            self.get_logger().debug(f'Received updated path with {path_points} points')
            
            # Only log significant path updates to avoid spamming
            if path_points % 5 == 0 or path_points == 1:  # Log every 5 points or the first point
                self.get_logger().info(f'Path now has {path_points} points')
    
    def start_return_journey(self, path_msg):
        """
        Start the return journey using the provided path.
        """
        # Extract and reverse waypoints
        self.waypoints = list(reversed(path_msg.poses))
        num_waypoints = len(self.waypoints)
        
        if num_waypoints == 0:
            self.get_logger().warn('Received empty path - nothing to navigate')
            self.publish_status("Received empty path - nothing to navigate")
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
    
    node = ReverseWaypointFollower()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()