"""Launch the MAPPO PX4 controller with its default parameter file."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory('mappo_px4_controller')
    default_params = os.path.join(package_share, 'config', 'mappo_params.yaml')

    params_file = LaunchConfiguration('params_file')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='Path to the MAPPO PX4 controller parameter file.',
        ),
        Node(
            package='mappo_px4_controller',
            executable='mappo_px4_controller',
            name='mappo_px4_controller',
            output='screen',
            parameters=[params_file],
        ),
    ])
