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
            'hazard_detection_node = par_1.hazard_detection_node:main',
            'tennis_ball_detector = par_1.tennis_ball_detector:main',
            'wall_follower = par_1.wall_follower:main',
            'path_recorder_navigator = par_1.path_rec_nav:main',
        ],
    },
)
