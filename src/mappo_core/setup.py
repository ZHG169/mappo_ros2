from setuptools import find_packages, setup

package_name = 'mappo_core'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/weights', ['weights/mappo_pxlimit_v1_final.pt']),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='ncrl',
    maintainer_email='ncrl@ncrl.local',
    description='ROS-independent MAPPO environment, observation, decision, inference, and tracking modules.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        ],
    },
)
