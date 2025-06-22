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
            'nav2_mission_planner = par_1.nav2_mission_planner:main',
            'new_table_tennis_detector = par_1.new_table_tennis_detector:main',
            'cylinder_detector = par_1.horizontal_cylinder_detector:main',
        ],
    },
)
