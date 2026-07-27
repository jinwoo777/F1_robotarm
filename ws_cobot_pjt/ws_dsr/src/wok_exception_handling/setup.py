import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'wok_exception_handling'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rokey',
    maintainer_email='onlyho12@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'safety_monitor = wok_exception_handling.safety_monitor:main',
            'recovery_manager = wok_exception_handling.recovery_manager:main',
            'estop_button_io = wok_exception_handling.estop_button_io:main',
            'estop_button_io_integrate = wok_exception_handling.estop_button_io_integrate:main',
            'shake_test_node = wok_exception_handling.shake_test_node:main',
            'force_feedback_bridge = wok_exception_handling.force_feedback_bridge:main',
            'robot_command_bridge = wok_exception_handling.robot_command_bridge:main',
            'robot_state_watchdog = wok_exception_handling.robot_state_watchdog:main',
        ],
    },
)
