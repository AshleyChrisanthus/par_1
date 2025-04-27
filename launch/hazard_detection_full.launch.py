from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import os

from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    aiil_rosbot_demo_dir = get_package_share_directory('aiil_rosbot_demo')

    return LaunchDescription([

        # Start find_object_2d_robot first
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(aiil_rosbot_demo_dir, 'launch', 'find_object_2d_robot.launch.py')
            ),
            launch_arguments={'gui': 'false'}.items()
        ),

        # Then start hazard_detection_node
        Node(
            package='par_1',
            executable='hazard_detection_node',
            name='hazard_detector',
            output='screen'
        ),

        # Then start path_recorder_navigator
        Node(
            package='par_1',
            executable='path_recorder_navigator',
            name='path_recorder_navigator',
            output='screen'
        ),

        # Finally start wall_follower
        Node(
            package='par_1',
            executable='wall_follower',
            name='wall_follower',
            output='screen'
        ),
    ])
