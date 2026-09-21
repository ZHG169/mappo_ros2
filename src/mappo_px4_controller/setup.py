from glob import glob
import os

from setuptools import find_packages, setup


package_name = 'mappo_px4_controller'


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        ('share/' + package_name, ['package.xml', 'README.md']),
        (
            os.path.join('share', package_name, 'config'),
            glob('config/*.yaml'),
        ),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py'),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ncrl',
    maintainer_email='ncrl@ncrl.local',
    description=(
        'ROS 2 controller bridging MAPPO core state and decisions to PX4.'
    ),
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            (
                'mappo_px4_controller = '
                'mappo_px4_controller.integrated_node:main'
            ),
        ],
    },
)
