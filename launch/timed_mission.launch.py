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
    # This node will start immediately at t=0 seconds.
    # Node name from its code: 'new_table_tennis_detector'
    ball_detector_node = Node(
        package='par_1',  # Assuming 'par_1' is your package name from the old file
        executable='new_table_tennis_detector.py',
        name='new_table_tennis_detector',
        output='screen'
    )

    # --- 2. Define the Horizontal Cylinder Detector Node ---
    # This node will be started after a delay.
    cylinder_detector_node = Node(
        package='par_1',
        executable='horizontal_cylinder_detector.py',      # timed_mission
        name='horizontal_cylinder_detector_stabilized_fix',
        output='screen'
    )

    # --- 3. Define the Navigation Mission Planner Node ---
    # Node name from its code: 'nav2_mission_planner'
    navigation_node = Node(
        package='par_1',
        executable='nav2_mission_planner.py',
        name='nav2_mission_planner',
        output='screen'
    )


    # --- Assemble the Launch Description with Timed Actions ---
    return LaunchDescription([
        # Action 1: Start the ball detector immediately.
        ball_detector_node,

        # Action 2: Use a TimerAction to delay the start of the cylinder detector.
        # This will execute after 10 seconds have passed.
        TimerAction(
            period=10.0,
            actions=[cylinder_detector_node]
        ),

        # Action 3: Use another TimerAction for the final navigation node.
        # The period is cumulative from the launch start time (10s + 10s = 20s).
        TimerAction(
            period=20.0,
            actions=[navigation_node]
        )
    ])
