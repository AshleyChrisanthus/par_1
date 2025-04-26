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
    def __init__(self):
        super().__init__('path_recorder_navigator')

        # Path recording
        self.path = Path()
        self.path.header.frame_id = 'map'
        self.recorded_poses = []
        self.last_position = None

        # TF2 setup
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Publishers
        self.path_publisher = self.create_publisher(Path, '/path_explore', 10)
        self.return_path_publisher = self.create_publisher(Path, '/path_return', 10)
        self.status_publisher = self.create_publisher(String, '/snc_status', 10)

        # Subscribers
        self.trigger_home_sub = self.create_subscription(Empty, '/trigger_home', self.trigger_home_callback, 10)

        # Action client
        self._action_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

        # Timers
        self.track_timer = self.create_timer(1.0, self.track_position)

        # Navigation state
        self.waypoints = []
        self.current_waypoint_index = 0
        self.is_navigating = False

        # Return path
        self.return_path = Path()
        self.return_path.header.frame_id = 'map'

        self.get_logger().info('PathRecorderNavigator node started.')
        self.publish_status('Initialized - recording path')

    def publish_status(self, text):
        msg = String()
        msg.data = text
        self.status_publisher.publish(msg)

    def track_position(self):
        try:
            # Check if transform is available
            if not self.tf_buffer.can_transform('map', 'base_link', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0)):
                self.get_logger().warning('TF map->base_link not ready yet.')
                return

            transform = self.tf_buffer.lookup_transform(
                'map', 'base_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )

            current_time = self.get_clock().now()
            x = transform.transform.translation.x
            y = transform.transform.translation.y

            record = False
            if self.last_position is None:
                record = True
            else:
                dx = x - self.last_position[0]
                dy = y - self.last_position[1]
                distance = math.hypot(dx, dy)
                if distance >= 0.35:
                    record = True

            if record:
                pose = PoseStamped()
                pose.header.stamp = current_time.to_msg()
                pose.header.frame_id = 'map'
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.position.z = transform.transform.translation.z
                pose.pose.orientation = transform.transform.rotation

                self.recorded_poses.append(pose)
                self.last_position = (x, y)
                self.path.header.stamp = current_time.to_msg()
                self.path.poses = self.recorded_poses

                self.path_publisher.publish(self.path)
                self.get_logger().info(f'Recorded ({x:.2f}, {y:.2f}) — Total points: {len(self.recorded_poses)}')

        except tf2_ros.TransformException as ex:
            self.get_logger().warning(f'TF Exception: {ex}')

    def create_quaternion_from_yaw(self, yaw):
        return Quaternion(
            x=0.0,
            y=0.0,
            z=math.sin(yaw / 2.0),
            w=math.cos(yaw / 2.0)
        )

    def trigger_home_callback(self, msg):
        if self.is_navigating:
            self.get_logger().info('Already navigating, ignoring trigger')
            self.publish_status('Already navigating')
            return

        if not self.recorded_poses:
            self.get_logger().warn('No recorded path to follow')
            self.publish_status('No recorded path to follow')
            return

        self.get_logger().info('Manual trigger received: Starting return journey')
        self.publish_status('Return journey starting')

        self.start_return_journey()

    def start_return_journey(self):
        self.waypoints = list(reversed(self.recorded_poses))
        if not self.waypoints:
            self.get_logger().warn('No waypoints available')
            self.publish_status('Return path empty')
            return

        self.update_waypoint_orientations()

        self.current_waypoint_index = 0
        self.is_navigating = True
        self.return_path.poses = []

        self.navigate_to_next_waypoint()

    def update_waypoint_orientations(self):
        n = len(self.waypoints)
        if n <= 1:
            return

        for i in range(n - 1):
            dx = self.waypoints[i+1].pose.position.x - self.waypoints[i].pose.position.x
            dy = self.waypoints[i+1].pose.position.y - self.waypoints[i].pose.position.y
            yaw = math.atan2(dy, dx)
            self.waypoints[i].pose.orientation = self.create_quaternion_from_yaw(yaw)

        self.waypoints[-1].pose.orientation = self.waypoints[-2].pose.orientation

    def navigate_to_next_waypoint(self):
        if not self.is_navigating:
            return

        if self.current_waypoint_index >= len(self.waypoints):
            self.get_logger().info('Return journey complete.')
            self.publish_status('Return journey complete.')
            self.is_navigating = False
            return

        waypoint = self.waypoints[self.current_waypoint_index]

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = waypoint

        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('NavigateToPose server not available.')
            self.publish_status('Nav2 server unavailable')
            self.is_navigating = False
            return

        self._send_goal_future = self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().warn(f'Waypoint {self.current_waypoint_index + 1} rejected')
            self.current_waypoint_index += 1
            self.navigate_to_next_waypoint()
            return

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        status = future.result().status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f'Reached waypoint {self.current_waypoint_index + 1}')
            waypoint = self.waypoints[self.current_waypoint_index]
            self.return_path.poses.append(waypoint)
            self.return_path.header.stamp = self.get_clock().now().to_msg()
            self.return_path_publisher.publish(self.return_path)
        else:
            self.get_logger().warn(f'Failed to reach waypoint {self.current_waypoint_index + 1}, status={status}')

        self.current_waypoint_index += 1
        self.navigate_to_next_waypoint()

    def feedback_callback(self, feedback_msg):
        pass


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
