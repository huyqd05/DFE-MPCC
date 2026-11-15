#!/bin/bash
set -e

source /opt/ros/jazzy/setup.bash

# Copy dfempcc_node.py vào package
cp /mnt/c/Users/LENOVO/Desktop/mujoco/dfempcc_node.py ~/amr_ws/src/amr_slosh_bridge/amr_slosh_bridge/dfempcc_node.py

# Cập nhật setup.py đăng ký dfempcc_node
cat << 'SETUP_EOF' > ~/amr_ws/src/amr_slosh_bridge/setup.py
from setuptools import find_packages, setup

package_name = 'amr_slosh_bridge'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lenovo',
    maintainer_email='lenovo@todo.todo',
    description='Cầu nối mô phỏng MuJoCo và ROS 2 cho xe AMR chở chất lỏng sóng sánh',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bridge_node = amr_slosh_bridge.bridge_node:main',
            'dfempcc_node = amr_slosh_bridge.dfempcc_node:main',
        ],
    },
)
SETUP_EOF

cd ~/amr_ws
colcon build --symlink-install
echo ">>> BIÊN DỊCH THÀNH CÔNG DFE-MPCC ROS 2 NODE!"
