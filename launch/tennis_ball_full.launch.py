from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Start tennis_ball_detector (replacing hazard_detection_node)
        Node(
            package='par_1',
            executable='tennis_ball_detector',
            name='tennis_ball_detector',
            output='screen'
        ),

        # Start path_recorder_navigator
        Node(
            package='par_1',
            executable='path_recorder_navigator',
            name='path_recorder_navigator',
            output='screen'
        ),

        # Start wall_follower
        Node(
            package='par_1',
            executable='wall_follower',
            name='wall_follower',
            output='screen'
        ),
    ])
