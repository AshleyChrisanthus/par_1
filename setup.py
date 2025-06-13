from setuptools import setup
import os
from glob import glob

package_name = 'par_1'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.jpg'))),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='',
    maintainer_email='',
    description='C',
    license='TODO: License declaration',
    # tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tennis_ball_detector = par_1.tennis_ball_detector:main',
            'tree_detector = par_1.tree_detector:main',
            'basic_navigation = par_1.basic_navigation:main',
            'tree_following_controller = par_1.tree_following_controller:main',
            'complete_macadamia_navigation = par_1.complete_macadamia_navigation:main',
            'improved_tree_detector = par_1.improved_tree_detector:main',
            'macadamia_field_explorer = par_1.macadamia_field_explorer:main',
            'improved_tree_detect = par_1.improved_tree_detect:main',
            'tree_row_follower = par_1.tree_row_follower:main',
            'tree_detector_node = par_1.tree_detector_node:main',
            # 'orb_tree_detector_node = par_1.orb_tree_detector_node:main',
            # 'tree_detector_node_gaber = par_1.tree_detector_node_gaber:main',
            # 'tree_detector_node_depth = par_1.tree_detector_node_depth:main',
            'macadamia_planner = par_1.macadamia_planner:main',
            'navigation_goal = par_1.navigation_goal:main',
            'explorer_node = par_1.explorer_node:main',
            'mission_planner_node = par_1.mission_planner_node:main',
        ],
    },
)
