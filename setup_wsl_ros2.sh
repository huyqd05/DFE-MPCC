#!/bin/bash
set -e

echo "=========================================================="
echo "KHỞI TẠO VÀ TÍCH HỢP HỆ THỐNG ROS 2 CHO DỰ ÁN AMR-SLOSH"
echo "=========================================================="

# 1. Cấu hình ~/.bashrc
echo "[1/5] Cấu hình môi trường trong ~/.bashrc..."
grep -q 'source /opt/ros/jazzy/setup.bash' ~/.bashrc || echo 'source /opt/ros/jazzy/setup.bash' >> ~/.bashrc

PROJECT_DIR="/mnt/c/Users/LENOVO/Desktop/mujoco"
if ! grep -q "$PROJECT_DIR" ~/.bashrc; then
    echo "export PYTHONPATH=\$PYTHONPATH:$PROJECT_DIR:$PROJECT_DIR/controller:$PROJECT_DIR/solver:$PROJECT_DIR/planning" >> ~/.bashrc
fi

# 2. Tạo cấu trúc thư mục workspace trong Linux filesystem
echo "[2/5] Khởi tạo workspace ~/amr_ws..."
mkdir -p ~/amr_ws/src
cd ~/amr_ws/src

# 3. Tạo package amr_slosh_bridge nếu chưa có
if [ ! -d "amr_slosh_bridge" ]; then
    echo "[3/5] Tạo ROS 2 package amr_slosh_bridge..."
    source /opt/ros/jazzy/setup.bash
    ros2 pkg create --build-type ament_python amr_slosh_bridge \
        --dependencies rclpy sensor_msgs nav_msgs geometry_msgs tf2_ros
fi

# Tạo thư mục launch và config
mkdir -p ~/amr_ws/src/amr_slosh_bridge/launch
mkdir -p ~/amr_ws/src/amr_slosh_bridge/config

# 4. Viết mã nguồn node bridge
echo "[4/5] Cài đặt mã nguồn bridge_node.py..."
cat << 'PY_EOF' > ~/amr_ws/src/amr_slosh_bridge/amr_slosh_bridge/bridge_node.py
import os
import sys
import numpy as np

# Nạp các thư mục của dự án vào sys.path
PROJECT_DIR = "/mnt/c/Users/LENOVO/Desktop/mujoco"
for p in [PROJECT_DIR, f"{PROJECT_DIR}/controller", f"{PROJECT_DIR}/solver", f"{PROJECT_DIR}/planning"]:
    if p not in sys.path:
        sys.path.insert(0, p)

import rclpy
from rclpy.node import Node
import tf2_ros
from geometry_msgs.msg import TransformStamped, Twist, Vector3Stamped
from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry
import mujoco

from obstacle_injector import get_combined_xml_string


class MujocoRos2Bridge(Node):
    def __init__(self):
        super().__init__('mujoco_ros2_bridge')
        
        # 1. Nạp mô hình kết hợp (Bản đồ + Xe AMR slosh3.xml)
        self.get_logger().info("Đang nạp mô hình kết hợp world.xml + slosh3.xml...")
        xml_str = get_combined_xml_string("world.xml")
        self.model = mujoco.MjModel.from_xml_string(xml_str)
        self.data = mujoco.MjData(self.model)
        
        # Đặt trạng thái ban đầu: đặt xe tại vị trí Start (-2.2, -2.2)
        self.data.qpos[0] = -2.2
        self.data.qpos[1] = -2.2
        self.data.qpos[2] = 0.0
        mujoco.mj_forward(self.model, self.data)
        
        # 2. Khởi tạo Publishers
        self.scan_pub = self.create_publisher(LaserScan, '/scan', 10)
        self.imu_pub = self.create_publisher(Imu, '/imu/data', 50)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 50)
        self.slosh_pub = self.create_publisher(Vector3Stamped, '/slosh/telemetry', 50)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # 3. Subscriber nhận vận tốc thân xe từ Nav2 / Controller
        self.cmd_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.v_cmd_body = np.zeros(3)  # [v_bx, v_by, w_bz]
        
        # Ma trận quán tính và hệ số điều khiển vận tốc cấp thấp
        self.M_full = np.zeros((self.model.nv, self.model.nv))
        self.kp_v = np.array([45.0, 45.0, 18.0])
        self.kd_v = np.array([12.0, 12.0, 5.0])
        self.v_cmd_world_filtered = np.zeros(3)
        
        # Thông số LiDAR
        self.n_beams = 180
        self.lidar_range_max = 5.0
        self.lidar_angles = np.linspace(-np.pi, np.pi, self.n_beams, endpoint=False)
        
        # 4. Timer vòng lặp vật lý 100 Hz (dt = 0.01 s)
        self.physics_dt = 0.005 # dt gốc của mô hình
        self.timer = self.create_timer(0.01, self.physics_step)
        
        self.get_logger().info(">>> Cầu nối MuJoCo - ROS 2 đã sẵn sàng! Xuất bản /scan, /imu/data, /odom...")

    def cmd_vel_callback(self, msg: Twist):
        # Lệnh vận tốc trong hệ quy chiếu thân xe (Body frame)
        self.v_cmd_body[0] = msg.linear.x
        self.v_cmd_body[1] = msg.linear.y
        self.v_cmd_body[2] = msg.angular.z

    def physics_step(self):
        # 1. Chuyển đổi vận tốc lệnh từ Body sang World frame
        theta = self.data.qpos[2]
        c_th, s_th = np.cos(theta), np.sin(theta)
        R_b2w = np.array([
            [c_th, -s_th, 0.0],
            [s_th,  c_th, 0.0],
            [0.0,   0.0,  1.0]
        ])
        v_cmd_world_target = R_b2w @ self.v_cmd_body
        
        # Lọc thông thấp vận tốc để chống giật
        alpha = 0.25
        self.v_cmd_world_filtered = (1.0 - alpha) * self.v_cmd_world_filtered + alpha * v_cmd_world_target
        
        # 2. Điều khiển bám vận tốc cấp thấp (Low-level Velocity Controller)
        mujoco.mj_fullM(self.model, self.data, self.M_full)
        v_now = self.data.qvel[:3]
        err_v = self.v_cmd_world_filtered - v_now
        
        # Bù quán tính và bù cản ma sát
        fric_ff = np.zeros(3)
        fric_ff[0:2] = 0.6 * np.tanh(v_now[0:2] / 0.02)
        fric_ff[2] = 0.25 * np.tanh(v_now[2] / 0.02)
        F_bias = self.data.qfrc_bias[:3].copy()
        
        F_cmd = self.kp_v * err_v + fric_ff + F_bias
        F_cmd[0:2] = np.clip(F_cmd[0:2], -30.0, 30.0)
        F_cmd[2] = np.clip(F_cmd[2], -10.0, 10.0)
        
        self.data.ctrl[:3] = F_cmd
        
        # 3. Tích phân 2 bước MuJoCo (2 x 0.005 s = 0.01 s)
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_step(self.model, self.data)
        
        # 4. Xuất bản các thông điệp ROS 2
        now = self.get_clock().now().to_msg()
        x, y, th = float(self.data.qpos[0]), float(self.data.qpos[1]), float(self.data.qpos[2])
        vx, vy, w_z = float(self.data.qvel[0]), float(self.data.qvel[1]), float(self.data.qvel[2])

        # A. Xuất bản TF: odom -> base_link
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = 0.05
        t.transform.rotation.z = float(np.sin(th / 2.0))
        t.transform.rotation.w = float(np.cos(th / 2.0))
        self.tf_broadcaster.sendTransform(t)

        # B. Xuất bản TF: base_link -> laser_link
        t_laser = TransformStamped()
        t_laser.header.stamp = now
        t_laser.header.frame_id = 'base_link'
        t_laser.child_frame_id = 'laser_link'
        t_laser.transform.translation.x = 0.0
        t_laser.transform.translation.y = 0.0
        t_laser.transform.translation.z = 0.20
        t_laser.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t_laser)

        # C. Xuất bản Odometry (/odom)
        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.orientation.z = float(np.sin(th / 2.0))
        odom.pose.pose.orientation.w = float(np.cos(th / 2.0))
        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.angular.z = w_z
        self.odom_pub.publish(odom)

        # D. Xuất bản Cảm biến IMU (/imu/data)
        imu = Imu()
        imu.header.stamp = now
        imu.header.frame_id = 'base_link'
        # Gia tốc thân xe trong hệ quy chiếu bệ xe
        a_bx = c_th * self.data.qacc[0] + s_th * self.data.qacc[1]
        a_by = -s_th * self.data.qacc[0] + c_th * self.data.qacc[1]
        imu.linear_acceleration.x = float(a_bx)
        imu.linear_acceleration.y = float(a_by)
        imu.linear_acceleration.z = 9.81 + float(self.data.qacc[2])
        imu.angular_velocity.z = w_z
        self.imu_pub.publish(imu)

        # E. Xuất bản trạng thái sóng sánh (/slosh/telemetry)
        if self.model.nq >= 5:
            slosh_msg = Vector3Stamped()
            slosh_msg.header.stamp = now
            slosh_msg.header.frame_id = 'container'
            slosh_msg.vector.x = float(self.data.qpos[3]) # phi_x (rad)
            slosh_msg.vector.y = float(self.data.qpos[4]) # phi_y (rad)
            slosh_msg.vector.z = float(self.data.qvel[3]) # dphi_x (rad/s)
            self.slosh_pub.publish(slosh_msg)

        # F. Xuất bản 2D LiDAR Ray-cast (/scan)
        scan = LaserScan()
        scan.header.stamp = now
        scan.header.frame_id = 'laser_link'
        scan.angle_min = -np.pi
        scan.angle_max = np.pi
        scan.angle_increment = 2.0 * np.pi / float(self.n_beams)
        scan.time_increment = 0.0
        scan.scan_time = 0.01
        scan.range_min = 0.10
        scan.range_max = self.lidar_range_max
        
        p_lidar = np.array([x, y, 0.20])
        ranges = []
        geomid = np.zeros(1, dtype=np.int32)
        for ang in self.lidar_angles:
            ray_dir = np.array([np.cos(th + ang), np.sin(th + ang), 0.0])
            d_ray = mujoco.mj_ray(self.model, self.data, p_lidar, ray_dir, None, 1, -1, geomid)
            if d_ray > 0.0 and d_ray <= self.lidar_range_max:
                ranges.append(float(d_ray))
            else:
                ranges.append(float('inf'))
                
        scan.ranges = ranges
        self.scan_pub.publish(scan)


def main(args=None):
    rclpy.init(args=args)
    node = MujocoRos2Bridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
PY_EOF

# 5. Cập nhật setup.py và build package
echo "[5/5] Cập nhật setup.py và biên dịch với colcon..."
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
        ],
    },
)
SETUP_EOF

cd ~/amr_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install

echo "=========================================================="
echo ">>> TÍCH HỢP HOÀN TẤT THÀNH CÔNG 100%!"
echo "Package amr_slosh_bridge đã được biên dịch sẵn sàng."
echo "=========================================================="
