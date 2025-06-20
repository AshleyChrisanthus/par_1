from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node

def generate_launch_description():
    """
    Generates a launch description to start the project nodes in a timed sequence.
    1. Starts the ball detector.
    2. Waits 10 seconds.
    3. Starts the cylinder detector.
    4. Waits another 10 seconds.
    5. Starts the mission planner/navigator.
    """

    # --- 1. Define the Table Tennis Ball Detector Node ---
    # The 'executable' name comes from your setup.py entry_points
    ball_detector_node = Node(
        package='par_1',
        executable='new_table_tennis_detector',  # CORRECT: Use the name from setup.py
        name='new_table_tennis_detector',
        output='screen'
    )

    # --- 2. Define the Horizontal Cylinder Detector Node ---
    # The 'executable' name comes from your setup.py entry_points
    cylinder_detector_node = Node(
        package='par_1',
        executable='cylinder_detector',      # CORRECT: Use the name from setup.py
        name='horizontal_cylinder_detector_stabilized_fix',
        output='screen'
    )

    # --- 3. Define the Navigation Mission Planner Node ---
    # The 'executable' name comes from your setup.py entry_points
    navigation_node = Node(
        package='par_1',
        executable='nav2_mission_planner',  # CORRECT: Use the name from setup.py
        name='nav2_mission_planner',
        output='screen'
    )


    # --- Assemble the Launch Description with Timed Actions ---
    return LaunchDescription([
        ball_detector_node,
        TimerAction(
            period=10.0,
            actions=[cylinder_detector_node]
        ),
        TimerAction(
            period=20.0,
            actions=[navigation_node]
        )
    ])
