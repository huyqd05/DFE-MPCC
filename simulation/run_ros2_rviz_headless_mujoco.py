#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================
run_ros2_rviz_headless_mujoco.py — PHƯƠNG ÁN 2: ĐỒNG MÔ PHỎNG MUJOCO HEADLESS + ROS 2 RVIZ2
=========================================================================================
Hỗ trợ toàn bộ 6 bản đồ đấu trường 6x6 m trong thư mục xml/:
  - world.xml  : Mê cung Chicane + 10 vật cản tĩnh (Trụ + Thùng)
  - world2.xml : Mê cung 15 vách ngăn ngách cụt + 4 vật cản tĩnh
  - world3.xml : Mê cung Chicane + 10 vật cản tĩnh + 2 vật cản động cắt ngang
  - world4.xml : Đấu trường nâng cao + 8 vật cản tĩnh + 3 vật cản động đa hướng (X & Y)
  - world5.xml : Hành lang Chicane vách tường + tiêm vật cản tĩnh & động sau A*
  - world6.xml : Mê cung đa hành lang phức tạp + 6 Checkpoints SPH + 4 vật cản động D1..D4

Kiến trúc thực thi:
  1. MuJoCo chạy 100% ở chế độ Headless (không bật cửa sổ GUI của MuJoCo):
     - Trích xuất bản đồ (Start, Goal, Tường tĩnh, Trụ tĩnh, Vật cản động).
     - Giải quy hoạch đường đi toàn cục A* trơn C^2 (Cubic B-Spline).
     - Chạy vòng lặp vật lý 200 Hz (dt = 0.005 s) + cảm biến LiDAR 2D 180 tia (mj_ray)
       + nền đường ngẫu nhiên ISO 8608 Class B + bộ giải tối ưu CasADi/IPOPT DFE-MPCC (20 Hz).
  2. ROS 2 Jazzy đảm nhận 100% hiển thị trực quan 3D trên RViz2:
     - /map                        (nav_msgs/OccupancyGrid)       : Bản đồ lưới chiếm chỗ 2D (tường + trụ)
     - /plan                       (nav_msgs/Path)                : Quỹ đạo toàn cục A* (Màu xanh lá)
     - /robot_trajectory           (nav_msgs/Path)                : Quỹ đạo thực tế của AMR (Màu xanh dương sáng)
     - /mpcc_horizon               (nav_msgs/Path)                : Chân trời dự báo N=15 bước của DFE-MPCC (Màu vàng)
     - /scan                       (sensor_msgs/LaserScan)        : Điểm chạm chuẩn 180 tia 2D LiDAR (REP-117)
     - /visualization_marker_array (visualization_msgs/MarkerArray):
         + 180 tia laser LiDAR 360° (Marker.LINE_LIST) phóng từ tháp cảm biến lidar_link tới vật cản
         + Tường 3D, Trụ tĩnh 3D, Điểm Start/Goal, 6 Điểm Checkpoint SPH (CP1..CP6)
         + Các Vật cản động chuyển động thời gian thực
         + Mô hình 3D Robot AMR + Tháp LiDAR + Bồn chất lỏng nghiêng theo góc sóng sánh (phi_x, phi_y)
=========================================================================================
"""

import os
import sys
import time
import math
import argparse
import subprocess
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ["", "controller", "solver", "planning", "simulation", "cfd"]:
    p = os.path.join(PROJECT_DIR, sub) if sub else PROJECT_DIR
    if p not in sys.path:
        sys.path.insert(0, p)

import mujoco
from astar import AStarPlanner
from dfe_mpcc_amr_controller import DFEMPCCAMRController
from iso8608_road_profile import ISO8608RoadProfile
from obstacle_injector import create_injected_simulation_model
from conformal_checkpoint_eval import load_conformal_calibration_budget

_CAL_BUDGET = load_conformal_calibration_budget()
from test_sim_amr_closed_loop import (
    load_combined_model,
    extract_map_from_model,
    update_dynamic_obstacles,
    Lidar,
    OccupancyGrid,
)
from benchmark_world6_complex_4dyn_obs import compute_world6_slosh_elevation_mm
from mecanum_traction_dynamics import (
    M_TOTAL, M_1X, M_1Y, L_P_X, L_P_Y,
    compute_dynamic_normal_loads,
    allocate_wheel_forces_clamped,
    evaluate_traction_and_slip,
)
from robot_model_amr_casadi import IZ_BASE, omega_x, omega_y
from friction_observer_ffrls import FrictionObserverFFRLS, OnlineFrictionFusion
from semantic_friction_estimator import (
    SemanticFrictionEstimator,
    make_world_semantic_map,
    make_hidden_oil_macro_fn,
)

import matplotlib
import matplotlib.pyplot as plt

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
    import tf2_ros
    from geometry_msgs.msg import TransformStamped, Twist, PoseStamped, Point, Vector3Stamped, WrenchStamped, TwistStamped
    from nav_msgs.msg import OccupancyGrid as RosOccupancyGrid, Path, Odometry
    from sensor_msgs.msg import LaserScan
    from visualization_msgs.msg import Marker, MarkerArray
    from std_msgs.msg import ColorRGBA
    HAS_RCLPY = True
except ImportError:
    HAS_RCLPY = False
    rclpy = None
    Node = object
    RosOccupancyGrid = object
    Path = object

    class ColorRGBA:
        def __init__(self, r=0.0, g=0.0, b=0.0, a=1.0):
            self.r, self.g, self.b, self.a = r, g, b, a


class _DummyPath:
    def __init__(self):
        self.poses = []
        self.header = type("_H", (), {"frame_id": "map", "stamp": None})()


def euler_to_quaternion(roll: float, pitch: float, yaw: float):
    """Chuyển đổi góc Euler (roll, pitch, yaw) sang Quaternion (x, y, z, w)."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return qx, qy, qz, qw


def compute_static_clearance(x: float, y: float, boxes: list, circles: list, r_robot: float = 0.20) -> float:
    """Tính khoảng cách nhỏ nhất từ biên robot tới mọi tường hộp (boxes) và trụ tĩnh (circles)."""
    min_c = 10.0
    for (cx, cy, hx, hy) in boxes:
        dx = max(abs(x - cx) - hx, 0.0)
        dy = max(abs(y - cy) - hy, 0.0)
        dist_to_box = math.hypot(dx, dy)
        if abs(x - cx) < hx and abs(y - cy) < hy:
            dist_to_box = -min(hx - abs(x - cx), hy - abs(y - cy))
        c = dist_to_box - r_robot
        if c < min_c:
            min_c = c
    for (cx, cy, r_c) in circles:
        c = math.hypot(x - cx, y - cy) - (r_c + r_robot)
        if c < min_c:
            min_c = c
    return min_c


class SimpatRos2RvizHeadlessNode(Node):
    """
    Node ROS 2 tích hợp MuJoCo Headless + xuất bản toàn bộ dữ liệu trực quan hóa cho RViz2.
    """
    def __init__(self, map_name, model, waypoints, obs_boxes, obs_circles, map_geoms, checkpoints_6, start_pos, goal_pos):
        self.map_name = map_name
        self.model = model
        self.waypoints = np.asarray(waypoints, dtype=float)
        self.obs_boxes = obs_boxes
        self.obs_circles = obs_circles
        self.map_geoms = map_geoms
        self.checkpoints_6 = checkpoints_6
        self.start_pos = start_pos
        self.goal_pos = goal_pos

        if not HAS_RCLPY:
            self.static_map_msg = None
            self.global_plan_msg = _DummyPath()
            self.robot_traj_msg = _DummyPath()
            return

        super().__init__('simpat_ros2_rviz_headless_node')

        # QoS đồng bộ Reliable giữa Publisher và cấu hình simpat_rviz2.rviz
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.map_pub = self.create_publisher(RosOccupancyGrid, '/map', reliable_qos)
        self.plan_pub = self.create_publisher(Path, '/plan', reliable_qos)
        self.traj_pub = self.create_publisher(Path, '/robot_trajectory', reliable_qos)
        self.hor_pub = self.create_publisher(Path, '/mpcc_horizon', reliable_qos)
        self.scan_pub = self.create_publisher(LaserScan, '/scan', reliable_qos)
        self.odom_pub = self.create_publisher(Odometry, '/odom', reliable_qos)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', reliable_qos)
        self.marker_pub = self.create_publisher(MarkerArray, '/visualization_marker_array', reliable_qos)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Publishers chuyên biệt cho PlotJuggler Telemetry thời gian thực:
        self.slosh_telemetry_pub = self.create_publisher(Vector3Stamped, '/slosh/telemetry', reliable_qos)
        self.slosh_forces_pub = self.create_publisher(WrenchStamped, '/slosh/forces', reliable_qos)
        self.wheel_loads_pub = self.create_publisher(TwistStamped, '/amr/wheel_fz', reliable_qos)
        self.speed_friction_pub = self.create_publisher(Vector3Stamped, '/amr/speed_friction', reliable_qos)

        # Góc quét local của 180 tia LiDAR trong hệ quy chiếu lidar_link
        self.beam_angles = np.linspace(0.0, 2.0 * math.pi, 180, endpoint=False)
        self.beam_cos = np.cos(self.beam_angles)
        self.beam_sin = np.sin(self.beam_angles)

        self.static_map_msg = self._build_static_occupancy_grid()
        self.global_plan_msg = self._build_global_plan_msg()
        self.robot_traj_msg = Path()
        self.robot_traj_msg.header.frame_id = 'map'

    def _build_static_occupancy_grid(self) -> RosOccupancyGrid:
        """Tạo bản đồ lưới chiếm chỗ 2D chuẩn nav_msgs/OccupancyGrid từ cả tường hộp và trụ tròn."""
        res = 0.05
        is_w8 = (self.map_name == "world8.xml")
        width = 140 if is_w8 else 128   # 7.0 m vs 6.4 m
        height = 140 if is_w8 else 128
        origin_x = -3.5 if is_w8 else -3.2
        origin_y = -3.5 if is_w8 else -3.2

        grid_data = np.zeros((height, width), dtype=np.int8)
        xs = origin_x + (np.arange(width) + 0.5) * res
        ys = origin_y + (np.arange(height) + 0.5) * res
        XX, YY = np.meshgrid(xs, ys)

        for (cx, cy, hx, hy) in self.obs_boxes:
            mask = (np.abs(XX - cx) <= (hx + 0.02)) & (np.abs(YY - cy) <= (hy + 0.02))
            grid_data[mask] = 100

        for (cx, cy, r_c) in self.obs_circles:
            mask = np.hypot(XX - cx, YY - cy) <= (r_c + 0.02)
            grid_data[mask] = 100

        msg = RosOccupancyGrid()
        msg.header.frame_id = 'map'
        msg.info.resolution = res
        msg.info.width = width
        msg.info.height = height
        msg.info.origin.position.x = origin_x
        msg.info.origin.position.y = origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = grid_data.flatten().tolist()
        return msg

    def _build_global_plan_msg(self) -> Path:
        """Đóng gói chuỗi waypoints A* thành thông điệp nav_msgs/Path."""
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        for i in range(len(self.waypoints)):
            ps = PoseStamped()
            ps.header.frame_id = 'map'
            ps.pose.position.x = float(self.waypoints[i, 0])
            ps.pose.position.y = float(self.waypoints[i, 1])
            ps.pose.position.z = 0.02
            if i < len(self.waypoints) - 1:
                dx = self.waypoints[i + 1, 0] - self.waypoints[i, 0]
                dy = self.waypoints[i + 1, 1] - self.waypoints[i, 1]
                yaw = math.atan2(dy, dx)
            else:
                yaw = 0.0
            _, _, qz, qw = euler_to_quaternion(0.0, 0.0, yaw)
            ps.pose.orientation.z = qz
            ps.pose.orientation.w = qw
            path_msg.poses.append(ps)
        return path_msg

    def clear_old_markers(self):
        """Xóa toàn bộ Marker cũ trên RViz2 khi chuyển sang bản đồ mới."""
        if not HAS_RCLPY:
            return
        ma = MarkerArray()
        m_del = Marker()
        m_del.action = Marker.DELETEALL
        ma.markers.append(m_del)
        self.marker_pub.publish(ma)

    def publish_all_telemetry(
        self,
        sim_time: float,
        q_meas: np.ndarray,
        v_meas: np.ndarray,
        phi_x: float,
        phi_y: float,
        eta_mm: float,
        x_pred: np.ndarray,
        scan_ranges: np.ndarray,
        scan_hits: np.ndarray,
        dyn_geoms: list,
        data: mujoco.MjData,
        nearest_dyn_name: str,
        active_min_d: float,
        s_prog: float,
        path_length: float,
        status_text: str,
        cycle: int,
        robot_color: ColorRGBA = None,
        fz_wheels: np.ndarray = None,
        fz_min: float = None,
        f_slosh: list = None,
        m_roll: float = 0.0,
        v_ub: float = 0.0,
        mu_local: float = 0.35,
    ):
        if not HAS_RCLPY:
            return
        now = self.get_clock().now().to_msg()
        x, y, th = float(q_meas[0]), float(q_meas[1]), float(q_meas[2])
        vx, vy, wz = float(v_meas[0]), float(v_meas[1]), float(v_meas[2])
        v_lin = math.hypot(vx, vy)

        # 1. Xuất bản cây TF chuẩn ROS 2 REP-105: map -> odom -> base_link -> lidar_link
        t_map_odom = TransformStamped()
        t_map_odom.header.stamp = now
        t_map_odom.header.frame_id = 'map'
        t_map_odom.child_frame_id = 'odom'
        t_map_odom.transform.rotation.w = 1.0

        t_odom_base = TransformStamped()
        t_odom_base.header.stamp = now
        t_odom_base.header.frame_id = 'odom'
        t_odom_base.child_frame_id = 'base_link'
        t_odom_base.transform.translation.x = x
        t_odom_base.transform.translation.y = y
        t_odom_base.transform.translation.z = 0.04
        _, _, qz, qw = euler_to_quaternion(0.0, 0.0, th)
        t_odom_base.transform.rotation.z = qz
        t_odom_base.transform.rotation.w = qw

        # Đặt lidar_link khớp chính xác độ cao lidar_site trong slosh3.xml (z = 0.04 + 0.212 = 0.252 m)
        t_base_lidar = TransformStamped()
        t_base_lidar.header.stamp = now
        t_base_lidar.header.frame_id = 'base_link'
        t_base_lidar.child_frame_id = 'lidar_link'
        t_base_lidar.transform.translation.z = 0.212
        t_base_lidar.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform([t_map_odom, t_odom_base, t_base_lidar])

        # 2. Định kỳ xuất bản /map và /plan
        if cycle % 10 == 0:
            self.static_map_msg.header.stamp = now
            self.map_pub.publish(self.static_map_msg)
            self.global_plan_msg.header.stamp = now
            self.plan_pub.publish(self.global_plan_msg)

        # 3. Xuất bản /odom và /cmd_vel
        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.angular.z = wz
        self.odom_pub.publish(odom)

        cmd = Twist()
        cmd.linear.x = float(math.cos(th) * vx + math.sin(th) * vy)
        cmd.linear.y = float(-math.sin(th) * vx + math.cos(th) * vy)
        cmd.angular.z = wz
        self.cmd_pub.publish(cmd)

        # 4. Xuất bản quỹ đạo thực tế /robot_trajectory
        ps_now = PoseStamped()
        ps_now.header.stamp = now
        ps_now.header.frame_id = 'map'
        ps_now.pose.position.x = x
        ps_now.pose.position.y = y
        ps_now.pose.position.z = 0.03
        ps_now.pose.orientation.z = qz
        ps_now.pose.orientation.w = qw
        self.robot_traj_msg.header.stamp = now
        self.robot_traj_msg.poses.append(ps_now)
        self.traj_pub.publish(self.robot_traj_msg)

        # 5. Xuất bản chân trời dự báo DFE-MPCC /mpcc_horizon (N=15)
        hor_msg = Path()
        hor_msg.header.stamp = now
        hor_msg.header.frame_id = 'map'
        for k in range(x_pred.shape[1]):
            psk = PoseStamped()
            psk.header.stamp = now
            psk.header.frame_id = 'map'
            psk.pose.position.x = float(x_pred[0, k])
            psk.pose.position.y = float(x_pred[1, k])
            psk.pose.position.z = 0.05
            _, _, qzk, qwk = euler_to_quaternion(0.0, 0.0, float(x_pred[2, k]))
            psk.pose.orientation.z = qzk
            psk.pose.orientation.w = qwk
            hor_msg.poses.append(psk)
        self.hor_pub.publish(hor_msg)

        # 6. Xuất bản dữ liệu quét LiDAR 2D chuẩn ROS 2 REP-117 (/scan - 180 tia)
        scan_msg = LaserScan()
        scan_msg.header.stamp = now
        scan_msg.header.frame_id = 'lidar_link'
        n_beams = len(scan_ranges)
        scan_msg.angle_min = 0.0
        scan_msg.angle_max = float(2.0 * math.pi * (n_beams - 1) / n_beams)
        scan_msg.angle_increment = float(2.0 * math.pi / n_beams)
        scan_msg.time_increment = 0.0
        scan_msg.scan_time = 0.05
        scan_msg.range_min = 0.05
        scan_msg.range_max = 4.00
        scan_msg.ranges = [
            float(np.clip(r, 0.05, 3.999)) if is_hit else float('inf')
            for r, is_hit in zip(scan_ranges, scan_hits)
        ]
        scan_msg.intensities = [
            float(max(100.0 - r * 20.0, 20.0)) if is_hit else 0.0
            for r, is_hit in zip(scan_ranges, scan_hits)
        ]
        self.scan_pub.publish(scan_msg)

        # 6b. Xuất bản Telemetry thời gian thực cho ROS 2 PlotJuggler
        msg_slosh = Vector3Stamped()
        msg_slosh.header.stamp = now
        msg_slosh.header.frame_id = 'base_link'
        msg_slosh.vector.x = float(math.degrees(phi_x))  # theta_x (độ - lắc ngang)
        msg_slosh.vector.y = float(math.degrees(phi_y))  # theta_y (độ - lắc dọc)
        msg_slosh.vector.z = float(eta_mm)               # eta (mm - dâng sóng)
        self.slosh_telemetry_pub.publish(msg_slosh)

        if f_slosh is not None:
            msg_force = WrenchStamped()
            msg_force.header.stamp = now
            msg_force.header.frame_id = 'base_link'
            msg_force.wrench.force.x = float(f_slosh[0])
            msg_force.wrench.force.y = float(f_slosh[1])
            msg_force.wrench.force.z = float(fz_min) if fz_min is not None else 0.0
            msg_force.wrench.torque.x = float(m_roll)
            self.slosh_forces_pub.publish(msg_force)

        if fz_wheels is not None:
            msg_fz = TwistStamped()
            msg_fz.header.stamp = now
            msg_fz.header.frame_id = 'base_link'
            msg_fz.twist.linear.x = float(fz_wheels[0])  # FL
            msg_fz.twist.linear.y = float(fz_wheels[1])  # FR
            msg_fz.twist.linear.z = float(fz_min) if fz_min is not None else float(np.min(fz_wheels))
            msg_fz.twist.angular.x = float(fz_wheels[3]) # RL
            msg_fz.twist.angular.y = float(fz_wheels[2]) # RR
            self.wheel_loads_pub.publish(msg_fz)

        msg_spd = Vector3Stamped()
        msg_spd.header.stamp = now
        msg_spd.header.frame_id = 'map'
        msg_spd.vector.x = float(v_lin)
        msg_spd.vector.y = float(v_ub)
        msg_spd.vector.z = float(mu_local)
        self.speed_friction_pub.publish(msg_spd)

        # 7. Xuất bản MarkerArray 3D (Tia LiDAR chuẩn, Tường, Trụ tĩnh, Checkpoints & Vùng ma sát)
        ma = MarkerArray()

        # 7a. MÔ HÌNH HÓA CÁC TIA LIDAR 2D CHUẨN XÁC, MẢNH VÀ THẨM MỸ (lidar_link)
        m_rays = Marker()
        m_rays.header.stamp = now
        m_rays.header.frame_id = 'lidar_link'
        m_rays.ns = 'lidar_rays_360'
        m_rays.id = 0
        m_rays.type = Marker.LINE_LIST
        m_rays.action = Marker.ADD
        m_rays.pose.orientation.w = 1.0
        m_rays.scale.x = 0.003  # Tia laser mảnh 3 mm sắc nét, không dày đặc

        p_origin = Point(x=0.0, y=0.0, z=0.0)
        for i in range(n_beams):
            r_val = float(scan_ranges[i])
            is_hit = bool(scan_hits[i])
            if is_hit:
                # Tia quét bắt trúng chướng ngại vật: vẽ từ tâm LiDAR tới bề mặt vật thể
                r_draw = float(np.clip(r_val, 0.05, 4.0))
                p_end = Point(
                    x=float(r_draw * self.beam_cos[i]),
                    y=float(r_draw * self.beam_sin[i]),
                    z=0.0
                )
                m_rays.points.append(p_origin)
                m_rays.points.append(p_end)
                if r_draw < 0.65:
                    c_start = ColorRGBA(r=1.0, g=0.30, b=0.10, a=0.75)
                    c_end = ColorRGBA(r=1.0, g=0.10, b=0.10, a=0.90)
                else:
                    c_start = ColorRGBA(r=0.0, g=0.85, b=1.0, a=0.45)
                    c_end = ColorRGBA(r=0.10, g=0.95, b=1.0, a=0.85)
                m_rays.colors.append(c_start)
                m_rays.colors.append(c_end)
            elif i % 3 == 0:
                # Không gian mở: Quạt quét radar 0.35m tượng trưng, mờ ảo, không che lấp bản đồ
                r_draw = 0.35
                p_end = Point(
                    x=float(r_draw * self.beam_cos[i]),
                    y=float(r_draw * self.beam_sin[i]),
                    z=0.0
                )
                m_rays.points.append(p_origin)
                m_rays.points.append(p_end)
                c_start = ColorRGBA(r=0.0, g=0.70, b=0.95, a=0.15)
                c_end = ColorRGBA(r=0.0, g=0.70, b=0.95, a=0.02)
                m_rays.colors.append(c_start)
                m_rays.colors.append(c_end)

        ma.markers.append(m_rays)

        # 7b. Các vách tường, thùng tĩnh và trụ tĩnh 3D (gửi định kỳ mỗi 10 chu kỳ)
        if cycle % 10 == 0 or cycle == 0:
            for idx, (gtype_str, pos_2d, size_val, rgba_arr, g_name) in enumerate(self.map_geoms):
                m_st = Marker()
                m_st.header.stamp = now
                m_st.header.frame_id = 'map'
                m_st.ns = 'static_arena_3d'
                m_st.id = idx
                m_st.action = Marker.ADD
                m_st.pose.position.x = float(pos_2d[0])
                m_st.pose.position.y = float(pos_2d[1])
                m_st.pose.position.z = 0.25
                m_st.pose.orientation.w = 1.0
                if gtype_str == 'box':
                    m_st.type = Marker.CUBE
                    m_st.scale.x = float(size_val[0] * 2.0)
                    m_st.scale.y = float(size_val[1] * 2.0)
                    m_st.scale.z = 0.50
                else:
                    m_st.type = Marker.CYLINDER
                    m_st.scale.x = float(size_val * 2.0)
                    m_st.scale.y = float(size_val * 2.0)
                    m_st.scale.z = 0.50
                m_st.color = ColorRGBA(
                    r=float(rgba_arr[0]),
                    g=float(rgba_arr[1]),
                    b=float(rgba_arr[2]),
                    a=0.72 if 'wall' in (g_name or '') else 0.88
                )
                ma.markers.append(m_st)

            # Điểm Start (Xanh dương) & Goal (Xanh lá)
            for idx_sg, (pos_sg, col_sg, lbl_sg) in enumerate([
                (self.start_pos, ColorRGBA(r=0.1, g=0.4, b=1.0, a=0.85), "START"),
                (self.goal_pos, ColorRGBA(r=0.1, g=0.95, b=0.2, a=0.85), "GOAL")
            ]):
                m_sg = Marker()
                m_sg.header.stamp = now
                m_sg.header.frame_id = 'map'
                m_sg.ns = 'start_goal'
                m_sg.id = idx_sg
                m_sg.type = Marker.CYLINDER
                m_sg.action = Marker.ADD
                m_sg.pose.position.x = float(pos_sg[0])
                m_sg.pose.position.y = float(pos_sg[1])
                m_sg.pose.position.z = 0.01
                m_sg.pose.orientation.w = 1.0
                m_sg.scale.x = 0.44
                m_sg.scale.y = 0.44
                m_sg.scale.z = 0.02
                m_sg.color = col_sg
                ma.markers.append(m_sg)

            # Các Điểm Checkpoint SPH (Cầu màu cam + Nhãn chữ)
            for idx_cp, cp in enumerate(self.checkpoints_6):
                m_cp = Marker()
                m_cp.header.stamp = now
                m_cp.header.frame_id = 'map'
                m_cp.ns = 'sph_checkpoints'
                m_cp.id = idx_cp * 2
                m_cp.type = Marker.SPHERE
                m_cp.action = Marker.ADD
                m_cp.pose.position.x = float(cp["pos"][0])
                m_cp.pose.position.y = float(cp["pos"][1])
                m_cp.pose.position.z = 0.12
                m_cp.pose.orientation.w = 1.0
                m_cp.scale.x = 0.14
                m_cp.scale.y = 0.14
                m_cp.scale.z = 0.14
                m_cp.color = ColorRGBA(r=1.0, g=0.55, b=0.0, a=0.90)
                ma.markers.append(m_cp)

                m_txt = Marker()
                m_txt.header.stamp = now
                m_txt.header.frame_id = 'map'
                m_txt.ns = 'sph_checkpoints'
                m_txt.id = idx_cp * 2 + 1
                m_txt.type = Marker.TEXT_VIEW_FACING
                m_txt.action = Marker.ADD
                m_txt.pose.position.x = float(cp["pos"][0])
                m_txt.pose.position.y = float(cp["pos"][1])
                m_txt.pose.position.z = 0.25
                m_txt.scale.z = 0.12
                m_txt.color = ColorRGBA(r=1.0, g=0.88, b=0.2, a=1.0)
                m_txt.text = f"{cp['name']} (R={cp['R_mm']:.1f}mm)"
                ma.markers.append(m_txt)

        # 7c. Các Vật cản động (D1..Dk) cập nhật trực tiếp ở mỗi chu kỳ
        for idx_d, (g_id, r_obs, gtype_dyn, rgba_dyn, obs_name) in enumerate(dyn_geoms):
            g_pos = data.geom_xpos[g_id]
            is_near = (obs_name == nearest_dyn_name and active_min_d < 0.35)

            m_dyn = Marker()
            m_dyn.header.stamp = now
            m_dyn.header.frame_id = 'map'
            m_dyn.ns = 'dynamic_obstacles'
            m_dyn.id = idx_d * 2
            m_dyn.type = Marker.CUBE if gtype_dyn == mujoco.mjtGeom.mjGEOM_BOX else Marker.CYLINDER
            m_dyn.action = Marker.ADD
            m_dyn.pose.position.x = float(g_pos[0])
            m_dyn.pose.position.y = float(g_pos[1])
            m_dyn.pose.position.z = 0.25
            m_dyn.pose.orientation.w = 1.0
            m_dyn.scale.x = float(r_obs * 2.0)
            m_dyn.scale.y = float(r_obs * 2.0)
            m_dyn.scale.z = 0.50
            if is_near:
                m_dyn.color = ColorRGBA(r=1.0, g=0.15, b=0.15, a=0.96)
            else:
                m_dyn.color = ColorRGBA(
                    r=float(rgba_dyn[0]),
                    g=float(rgba_dyn[1]),
                    b=float(rgba_dyn[2]),
                    a=0.90
                )
            ma.markers.append(m_dyn)

        # 7d. Mô hình 3D Thân xe AMR + Bồn chất lỏng + Mặt thoáng sóng sánh + Tháp cảm biến LiDAR 2D
        # Khung gầm AMR
        m_base = Marker()
        m_base.header.stamp = now
        m_base.header.frame_id = 'base_link'
        m_base.ns = 'amr_robot_3d'
        m_base.id = 0
        m_base.type = Marker.CUBE
        m_base.action = Marker.ADD
        m_base.pose.position.z = 0.01
        m_base.pose.orientation.w = 1.0
        m_base.scale.x = 0.32
        m_base.scale.y = 0.24
        m_base.scale.z = 0.05
        m_base.color = robot_color if robot_color is not None else ColorRGBA(r=0.22, g=0.26, b=0.33, a=0.95)
        ma.markers.append(m_base)

        # Vỏ bồn Acrylic trong suốt
        m_tank = Marker()
        m_tank.header.stamp = now
        m_tank.header.frame_id = 'base_link'
        m_tank.ns = 'amr_robot_3d'
        m_tank.id = 1
        m_tank.type = Marker.CUBE
        m_tank.action = Marker.ADD
        m_tank.pose.position.z = 0.095
        m_tank.pose.orientation.w = 1.0
        m_tank.scale.x = 0.154
        m_tank.scale.y = 0.104
        m_tank.scale.z = 0.11
        m_tank.color = ColorRGBA(r=0.55, g=0.85, b=1.0, a=0.35)
        ma.markers.append(m_tank)

        # Mặt chất lỏng nghiêng theo góc sóng sánh thực (phi_x, phi_y)
        m_water = Marker()
        m_water.header.stamp = now
        m_water.header.frame_id = 'base_link'
        m_water.ns = 'amr_robot_3d'
        m_water.id = 2
        m_water.type = Marker.CUBE
        m_water.action = Marker.ADD
        m_water.pose.position.z = 0.075 + min(eta_mm / 1000.0, 0.03) * 0.5
        qwx, qwy, qwz, qww = euler_to_quaternion(float(phi_y * 2.5), float(phi_x * 2.5), 0.0)
        m_water.pose.orientation.x = qwx
        m_water.pose.orientation.y = qwy
        m_water.pose.orientation.z = qwz
        m_water.pose.orientation.w = qww
        m_water.scale.x = 0.148
        m_water.scale.y = 0.098
        m_water.scale.z = 0.075
        if eta_mm < 10.0:
            m_water.color = ColorRGBA(r=0.0, g=0.75, b=1.0, a=0.88)
        elif eta_mm < 18.0:
            m_water.color = ColorRGBA(r=0.1, g=0.95, b=0.4, a=0.88)
        else:
            m_water.color = ColorRGBA(r=1.0, g=0.45, b=0.0, a=0.92)
        ma.markers.append(m_water)

        # Tháp cảm biến 2D LiDAR đặt tại lidar_link (phát ra 180 tia laser)
        m_lidar = Marker()
        m_lidar.header.stamp = now
        m_lidar.header.frame_id = 'lidar_link'
        m_lidar.ns = 'amr_robot_3d'
        m_lidar.id = 4
        m_lidar.type = Marker.CYLINDER
        m_lidar.action = Marker.ADD
        m_lidar.pose.position.z = 0.0
        m_lidar.pose.orientation.w = 1.0
        m_lidar.scale.x = 0.07
        m_lidar.scale.y = 0.07
        m_lidar.scale.z = 0.04
        m_lidar.color = ColorRGBA(r=0.0, g=0.90, b=1.0, a=0.95)
        ma.markers.append(m_lidar)

        # [LƯỢC BỎ] m_rtxt (Nhãn chữ trên đầu robot) và m_hud (Bảng HUD che đỉnh màn hình)
        m_del_hud = Marker()
        m_del_hud.header.stamp = now
        m_del_hud.header.frame_id = 'map'
        m_del_hud.ns = 'telemetry_hud'
        m_del_hud.action = Marker.DELETEALL
        ma.markers.append(m_del_hud)

        self.marker_pub.publish(ma)


def build_map_scenario(map_name: str):
    """
    Khởi tạo kịch bản bản đồ bất kỳ trong 6 bản đồ (world.xml .. world6.xml):
    - Giải A* toàn cục trơn C^2.
    - Trích xuất hoặc tiêm vật cản động tùy bản đồ.
    - Tiền tính toán cung đường quét s_obs cho mọi vật cản động (dọc trục X hoặc trục Y).
    """
    base_name = os.path.basename(map_name).lower()
    model_raw, _ = load_combined_model(base_name)
    start_pos, goal_pos, obs_circles_raw, obs_boxes_raw, map_geoms_raw, _ = extract_map_from_model(model_raw)

    waypoints = []
    used_r = 0.24
    if base_name == "world9.xml":
        xy_lim = 3.50
        r_candidates = [0.30, 0.28, 0.24, 0.22]
    elif base_name == "world8.xml":
        xy_lim = 3.1
        r_candidates = [0.24, 0.22, 0.20, 0.18]
    else:
        xy_lim = 2.9
        r_candidates = [0.24, 0.22, 0.20, 0.18]
    for r_test in r_candidates:
        planner = AStarPlanner(x_bounds=(-xy_lim, xy_lim), y_bounds=(-xy_lim, xy_lim), resolution=0.05, robot_radius=r_test)
        planner.add_obstacles(obs_circles_raw, obs_boxes_raw)
        waypoints = planner.plan(start_pos=start_pos[:2], goal_pos=goal_pos[:2], smooth=True)
        if len(waypoints) > 10:
            used_r = r_test
            break

    wp_arr = np.array(waypoints, dtype=float)
    path_s = np.zeros(len(wp_arr))
    for i in range(1, len(wp_arr)):
        path_s[i] = path_s[i - 1] + np.linalg.norm(wp_arr[i] - wp_arr[i - 1])

    checkpoints_6 = []
    if base_name == "world9.xml":
        checkpoints_6 = [
            {"name": "CP1", "desc": "Chicane Kệ Tây-Bắc", "s_m": 2.10, "pos": [-2.10, 0.85], "R_mm": 0.0},
            {"name": "CP2", "desc": "Cua 180° Nhập liệu Nam", "s_m": 4.98, "pos": [-1.70, -1.70], "R_mm": 0.0},
            {"name": "CP3", "desc": "Chicane Trạm CNC Giữa", "s_m": 7.24, "pos": [-0.55, 0.00], "R_mm": 0.0},
            {"name": "CP4", "desc": "Cua 180° Đỉnh Bắc", "s_m": 9.85, "pos": [-0.10, 2.05], "R_mm": 0.0},
            {"name": "CP5", "desc": "Khu Rửa & Cua 180° Nam-Đông", "s_m": 14.20, "pos": [1.70, -1.75], "R_mm": 0.0},
            {"name": "CP6", "desc": "Cửa Z Xuất Hàng Đông", "s_m": 17.44, "pos": [2.25, 1.20], "R_mm": 0.0},
        ]
        inj_configs = [
            {"type": "dynamic_cylinder", "center_override": [-2.55, -0.30], "radius": 0.14, "amp": 0.52, "omega": 1.00, "phase": 1.10, "slide_axis": "1 0 0", "rgba": "0.95 0.15 0.60 1", "name": "dyn_obs_1_west"},
            {"type": "dynamic_cylinder", "center_override": [-1.05, 0.80], "radius": 0.14, "amp": 0.48, "omega": 0.95, "phase": 0.35, "slide_axis": "1 0 0", "rgba": "0.05 0.72 0.85 1", "name": "dyn_obs_2_center"},
            {"type": "dynamic_cylinder", "center_override": [2.55, -0.50], "radius": 0.14, "amp": 0.55, "omega": 0.92, "phase": 1.75, "slide_axis": "1 0 0", "rgba": "0.98 0.55 0.05 1", "name": "dyn_obs_3_east"},
        ]
        model, dyn_configs, _ = create_injected_simulation_model(f"xml/{base_name}", wp_arr, configs=inj_configs)
    elif base_name == "world8.xml":
        checkpoints_6 = [
            {"name": "CP1", "desc": "Cua 180° Tây (s=3.8m)", "s_m": 3.78, "pos": [-2.35, 2.05], "R_mm": 2.15},
            {"name": "CP2", "desc": "Điểm Uốn Chicane Giữa-Tây", "s_m": 5.40, "pos": [-0.65, -0.80], "R_mm": 1.85},
            {"name": "CP3", "desc": "Cua 180° Nam-Đông", "s_m": 7.20, "pos": [0.75, -1.10], "R_mm": 2.45},
            {"name": "CP4", "desc": "Hành Lang Chuyển Tiếp Đông", "s_m": 9.10, "pos": [0.75, 0.85], "R_mm": 1.65},
            {"name": "CP5", "desc": "Cua 180° Đông (s=10.6m)", "s_m": 10.60, "pos": [1.85, 1.85], "R_mm": 2.30},
            {"name": "CP6", "desc": "Cửa Đích Đông-Bắc", "s_m": 13.80, "pos": [1.85, 2.50], "R_mm": 1.90},
        ]
        inj_configs = [
            {"type": "dynamic_cylinder", "center_override": [-2.35, 0.40], "radius": 0.14, "amp": 0.32, "omega": 1.05, "phase": 0.85, "slide_axis": "1 0 0", "rgba": "0.95 0.15 0.60 1", "name": "dyn_obs_1_west"},
            {"type": "dynamic_cylinder", "center_override": [-0.85, 0.50], "radius": 0.14, "amp": 0.32, "omega": 0.95, "phase": 0.40, "slide_axis": "1 0 0", "rgba": "0.05 0.72 0.85 1", "name": "dyn_obs_2_midwest"},
            {"type": "dynamic_cylinder", "center_override": [0.65, -0.60], "radius": 0.14, "amp": 0.32, "omega": 1.00, "phase": 1.90, "slide_axis": "1 0 0", "rgba": "0.98 0.55 0.05 1", "name": "dyn_obs_3_mideast"},
            {"type": "dynamic_cylinder", "center_override": [1.85, 2.40], "radius": 0.14, "amp": 0.38, "omega": 0.90, "phase": 0.60, "slide_axis": "0 1 0", "rgba": "0.55 0.20 0.90 1", "name": "dyn_obs_4_east_goal"},
        ]
        model, dyn_configs, _ = create_injected_simulation_model(f"xml/{base_name}", wp_arr, configs=inj_configs)
    elif base_name in ("world6.xml", "world7.xml"):
        checkpoints_6 = [
            {"name": "CP1", "desc": "Chicane Cộng Hưởng 1", "s_m": 1.58, "pos": [-2.14, -0.72], "R_mm": 2.36},
            {"name": "CP2", "desc": "Cua 180° Cộng Hưởng 2", "s_m": 3.68, "pos": [-1.75, 1.26], "R_mm": 2.76},
            {"name": "CP3", "desc": "Điểm Uốn Chữ S Gắt", "s_m": 5.90, "pos": [-0.37, 0.56], "R_mm": 2.44},
            {"name": "CP4", "desc": "Cua Đáy Chuyển Tiếp", "s_m": 8.12, "pos": [-0.23, -1.56], "R_mm": 1.70},
            {"name": "CP5", "desc": "Hành Lang Thẳng Đông", "s_m": 10.22, "pos": [0.96, -0.22], "R_mm": 1.58},
            {"name": "CP6", "desc": "Cửa Z-Chicane Đông-Bắc", "s_m": 11.90, "pos": [1.59, 1.20], "R_mm": 2.06},
        ]
        east_goal_x = 1.85 if base_name == "world6.xml" else 1.92
        inj_configs = [
            {"type": "dynamic_cylinder", "center_override": [-2.02, 0.25], "radius": 0.14, "amp": 0.30, "omega": 1.05, "phase": 1.25, "slide_axis": "1 0 0", "rgba": "0.95 0.15 0.60 1", "name": "dyn_obs_1_west"},
            {"type": "dynamic_cylinder", "center_override": [-0.52, 0.18], "radius": 0.14, "amp": 0.28, "omega": 0.96, "phase": 0.45, "slide_axis": "1 0 0", "rgba": "0.05 0.72 0.85 1", "name": "dyn_obs_2_midwest"},
            {"type": "dynamic_cylinder", "center_override": [0.72, -0.45], "radius": 0.14, "amp": 0.26, "omega": 1.02, "phase": 2.15, "slide_axis": "1 0 0", "rgba": "0.98 0.55 0.05 1", "name": "dyn_obs_3_mideast"},
            {"type": "dynamic_cylinder", "center_override": [east_goal_x, 1.25], "radius": 0.14, "amp": 0.30, "omega": 0.94, "phase": 0.85, "slide_axis": "1 0 0", "rgba": "0.55 0.20 0.90 1", "name": "dyn_obs_4_east_goal"},
        ]
        model, dyn_configs, _ = create_injected_simulation_model(f"xml/{base_name}", wp_arr, configs=inj_configs)
    elif base_name == "world5.xml":
        # Tiêm 2 vật cản động cắt ngang 2 hành lang Tây và Đông của world5.xml
        inj_configs = [
            {"type": "dynamic_cylinder", "center_override": [-1.85, 0.00], "radius": 0.16, "amp": 0.35, "omega": 1.05, "phase": 0.80, "slide_axis": "1 0 0", "rgba": "0.95 0.15 0.65 1", "name": "dyn_obs_1_west"},
            {"type": "dynamic_cylinder", "center_override": [1.95, 0.30], "radius": 0.16, "amp": 0.35, "omega": 0.95, "phase": 0.20, "slide_axis": "1 0 0", "rgba": "0.05 0.78 0.90 1", "name": "dyn_obs_2_east"},
        ]
        model, dyn_configs, _ = create_injected_simulation_model("xml/world5.xml", wp_arr, configs=inj_configs)
    elif base_name == "world3.xml":
        model = model_raw
        dyn_configs = {
            "dyn_joint_1": {"amp": 0.58, "omega": 1.25, "phase": 0.0},
            "dyn_joint_2": {"amp": 0.48, "omega": 1.05, "phase": 0.5},
        }
    else:
        # world.xml, world2.xml, world4.xml dùng trực tiếp vật cản tĩnh & động có sẵn trong XML
        model = model_raw
        dyn_configs = {
            "dyn_joint_1": {"amp": 0.38, "omega": 1.15, "phase": 0.0},
            "dyn_joint_2": {"amp": 0.35, "omega": 1.00, "phase": 0.6},
            "dyn_joint_3": {"amp": 0.38, "omega": 0.95, "phase": 1.2},
        }

    start_pos, goal_pos, obs_circles, obs_boxes, map_geoms, dyn_geoms = extract_map_from_model(model)

    # Xây dựng danh sách gate_descriptors tổng quát cho mọi vật cản động trong bản đồ
    gate_descriptors = []
    for idx_d, (g_id, r_obs, gtype_dyn, rgba_dyn, obs_name) in enumerate(dyn_geoms):
        b_id = model.geom_bodyid[g_id]
        base_pos = model.body_pos[b_id][:2].copy()
        j_adr = model.body_jntadr[b_id]
        if j_adr >= 0:
            j_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j_adr)
            axis_xy = model.jnt_axis[j_adr][:2].copy()
        else:
            j_name = ""
            axis_xy = np.array([1.0, 0.0])

        j_cfg = dyn_configs.get(j_name, {"amp": 0.30, "omega": 1.0, "phase": 0.0})
        amp = float(j_cfg.get("amp", 0.30))
        omega = float(j_cfg.get("omega", 1.0))
        phase = float(j_cfg.get("phase", 0.0))

        # Tìm cung đường sớm nhất s_obs mà vùng quét [-amp, +amp] của vật cản động chạm tới quỹ đạo A*
        s_earliest = 100.0
        for u_sweep in np.linspace(-amp, amp, 11):
            p_sweep = base_pos + axis_xy * u_sweep
            d = np.hypot(wp_arr[:, 0] - p_sweep[0], wp_arr[:, 1] - p_sweep[1])
            imin = int(np.argmin(d))
            if d[imin] < (r_obs + 0.30) and path_s[imin] < s_earliest:
                s_earliest = float(path_s[imin])

        gate_descriptors.append({
            "name": obs_name,
            "base_pos": base_pos,
            "axis_xy": axis_xy,
            "radius": float(r_obs),
            "amp": amp,
            "omega": omega,
            "phase": phase,
            "s_obs": s_earliest,
        })

    # Ưu tiên nạp danh sách Checkpoint Hybrid B+A WCSPH đã chuẩn hóa từ bao cao/checkpoints_<map>.json (nếu có)
    map_stem = os.path.splitext(base_name)[0]
    cp_json_path = os.path.join(PROJECT_DIR, "bao cao", f"checkpoints_{map_stem}.json")
    if os.path.exists(cp_json_path):
        try:
            import json
            with open(cp_json_path, "r", encoding="utf-8") as f_cp:
                cp_data = json.load(f_cp)
            raw_cps = cp_data.get("checkpoints", [])
            if raw_cps:
                normalized_cps = []
                for idx_c, cp_item in enumerate(raw_cps):
                    cp_copy = dict(cp_item)
                    cp_copy.setdefault("name", f"CP{idx_c + 1}")
                    cp_copy.setdefault("desc", "checkpoint")
                    cp_copy["R_mm"] = float(cp_copy.get("R_mm", cp_copy.get("residual_R_mm", 0.0)))
                    normalized_cps.append(cp_copy)
                checkpoints_6 = normalized_cps
        except Exception:
            pass

    return {
        "map_name": base_name,
        "model": model,
        "dyn_configs": dyn_configs,
        "start_pos": start_pos,
        "goal_pos": goal_pos,
        "obs_circles": obs_circles,
        "obs_boxes": obs_boxes,
        "map_geoms": map_geoms,
        "dyn_geoms": dyn_geoms,
        "wp_arr": wp_arr,
        "path_s": path_s,
        "used_r": used_r,
        "checkpoints_6": checkpoints_6,
        "gate_descriptors": gate_descriptors,
    }


def predict_gate_pos(g_desc: dict, t_query: float) -> np.ndarray:
    """Dự báo vị trí 2D (x, y) của vật cản động tại thời điểm t_query."""
    disp = g_desc["amp"] * math.sin(g_desc["omega"] * t_query + g_desc["phase"])
    return g_desc["base_pos"] + g_desc["axis_xy"] * disp


def _predict_gate_window_clearance(
    g_desc: dict,
    s0: float,
    v0: float,
    v_dash_limit: float,
    accel_dash: float,
    t_start: float,
    tau_max: float,
    n_samples: int,
    s_win_back: float,
    s_win_ahead: float,
    ctrl,
) -> float:
    """Tính khoảng hở nhỏ nhất dự báo khi robot băng qua cửa sổ cổng vật cản động."""
    taus = np.linspace(0.0, tau_max, n_samples)
    min_pred_c = 10.0
    s_cur = s0
    v_cur = v0
    s_lo = g_desc["s_obs"] - s_win_back
    s_hi = g_desc["s_obs"] + s_win_ahead
    r_tot = g_desc["radius"] + 0.22
    for idx_t, tau in enumerate(taus):
        if idx_t > 0:
            dt_t = float(tau - taus[idx_t - 1])
            v_cur = min(v_dash_limit, v_cur + accel_dash * dt_t)
            s_cur = min(s_cur + v_cur * dt_t, ctrl.path_length)
        if s_lo <= s_cur <= s_hi:
            p_rob, _, _ = ctrl.get_path_reference(s_cur)
            p_obs = predict_gate_pos(g_desc, t_start + float(tau))
            c_val = float(np.hypot(p_rob[0] - p_obs[0], p_rob[1] - p_obs[1])) - r_tot
            if c_val < min_pred_c:
                min_pred_c = c_val
    return min_pred_c


def evaluate_gate_crossing(
    gate_descriptors: list,
    s_prog: float,
    v_curr: float,
    sim_time: float,
    dashing_gate: int,
    ctrl,
    tube_planner=None,
    use_tube_speed: bool = True,
    v_dash_fallback: float = 0.40,
    v_dash_bounds: tuple = (0.32, 0.42),
    delta_s_hi: float = 0.58,
    base_clear_thresh: float = 0.06,
    accel_dash: float = 0.80,
    tau_extra: float = 1.0,
    tau_floor: float = 3.5,
    n_samples: int = 45,
    s_win_back: float = 0.38,
    s_win_ahead: float = 0.50,
    s_end_margin: float = 0.55,
):
    """Đánh giá Đợi/Vượt tại các cổng vật cản động và trả về (should_wait, active_gate, new_dashing_gate, is_tight_gate)."""
    should_wait = False
    active_gate = -1
    is_tight_gate = False

    for idx_d, g_desc in enumerate(gate_descriptors):
        if g_desc["s_obs"] > 50.0:
            continue
        delta_s = g_desc["s_obs"] - s_prog
        if -0.45 <= delta_s <= delta_s_hi:
            active_gate = idx_d
            if dashing_gate == idx_d:
                is_tight_gate = bool(g_desc.get("_is_tight_gate", False))
                break

            if use_tube_speed and tube_planner is not None and getattr(tube_planner, "is_ready", False):
                v_tube_at_gate = float(np.interp(g_desc["s_obs"], tube_planner.s_grid, tube_planner.v_ub_nom))
                v_dash_limit = min(v_dash_bounds[1], max(v_dash_bounds[0], v_tube_at_gate))
            else:
                v_dash_limit = v_dash_fallback

            tau_max = max(tau_floor, (g_desc["s_obs"] + s_end_margin - s_prog) / max(0.20, v_dash_limit) + tau_extra)
            min_pred_c = _predict_gate_window_clearance(
                g_desc, s_prog, v_curr, v_dash_limit, accel_dash,
                sim_time, tau_max, n_samples, s_win_back, s_win_ahead, ctrl,
            )

            cache_key = round(float(v_dash_limit), 3)
            thresh_cache = g_desc.setdefault("_eff_thresh_cache", {})
            tight_cache = g_desc.setdefault("_is_tight_cache", {})
            eff_thresh = thresh_cache.get(cache_key)
            if eff_thresh is None:
                if min_pred_c >= base_clear_thresh:
                    eff_thresh = base_clear_thresh
                    tight_cache[cache_key] = False
                else:
                    # Cache ngưỡng khả thi nhất theo hình học quỹ đạo để tránh deadlock ở cổng chéo hẹp
                    period = 2.0 * math.pi / max(abs(float(g_desc["omega"])), 0.2)
                    s_wait = max(0.0, min(s_prog, g_desc["s_obs"] - 0.46))
                    best_c = max(
                        _predict_gate_window_clearance(
                            g_desc, s_wait, 0.0, v_dash_limit, accel_dash,
                            t0, tau_max, n_samples, s_win_back, s_win_ahead, ctrl,
                        )
                        for t0 in np.linspace(0.0, period, 36, endpoint=False)
                    )
                    if best_c >= base_clear_thresh:
                        eff_thresh = base_clear_thresh
                        tight_cache[cache_key] = False
                    else:
                        eff_thresh = max(0.003, best_c * 0.55) if best_c > 0.0 else (best_c - 0.008)
                        tight_cache[cache_key] = True
                thresh_cache[cache_key] = float(eff_thresh)

            is_tight_gate = bool(tight_cache.get(cache_key, False))
            g_desc["_is_tight_gate"] = is_tight_gate
            if min_pred_c >= eff_thresh:
                dashing_gate = idx_d
                should_wait = False
            else:
                should_wait = True
            break

    if active_gate == -1:
        dashing_gate = -1
        is_tight_gate = False

    return should_wait, active_gate, dashing_gate, is_tight_gate


def select_tight_gate_obstacle(dyn_geoms: list, data: mujoco.MjData, q_meas: np.ndarray, far_obs: np.ndarray, d_thresh: float = 0.85, d_safe: float = 0.10) -> np.ndarray:
    """Trả về tham số vật cản động gần nhất cho Obstacle-ECBF khi băng qua cổng chéo hẹp."""
    best_p, best_r, best_dc = None, 0.14, 10.0
    for g_id, r_obs, _, _, _ in dyn_geoms:
        g_pos = data.geom_xpos[g_id][:2].copy()
        dc = float(np.hypot(q_meas[0] - g_pos[0], q_meas[1] - g_pos[1])) - (r_obs + 0.20)
        if dc < best_dc:
            best_dc, best_p, best_r = dc, g_pos, float(r_obs)
    if best_p is not None and best_dc < d_thresh:
        return np.array([best_p[0], best_p[1], 0.0, best_r, d_safe], dtype=float)
    return far_obs


def plot_post_simulation_results(
    summary: dict,
    hist: dict,
    map_name: str,
    out_dir: str = None,
    show_plot: bool = True
):
    """
    Hiển thị và lưu đồ thị phân tích toàn diện 4-panel sau khi hoàn thành mô phỏng (Post-Simulation Analysis):
      - Panel (a): Góc sóng sánh theta_x, theta_y (độ) & Độ dâng sóng tự do eta(t) (mm)
      - Panel (b): Phản lực sóng sánh chất lỏng F_slosh,x, F_slosh,y (N) & Mô-men lật M_roll (N.m)
      - Panel (c): Tải pháp tuyến 4 bánh xe Mecanum F_z,1..4 (N) & Ngưỡng chống nhấc bánh
      - Panel (d): Vận tốc thực tế v(s) bám theo trần Ống vận tốc khả thi v_ub(s) (m/s)
    """
    if out_dir is None:
        out_dir = os.path.join(PROJECT_DIR, "bao cao")
    os.makedirs(out_dir, exist_ok=True)
    base_stem = map_name.replace(".xml", "")
    out_png_path = os.path.join(out_dir, f"do_thi_sau_chay_{base_stem}.png")

    t_arr = np.asarray(hist.get("time", []))
    s_arr = np.asarray(hist.get("s", []))
    if len(t_arr) == 0:
        return

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.0))
    try:
        fig.canvas.manager.set_window_title(f"Đồ thị phân tích sau chạy - Bản đồ {map_name.upper()}")
    except Exception:
        pass

    # --- Panel (a): Góc sóng sánh & Độ dâng sóng eta ---
    ax_a1 = axes[0, 0]
    ax_a2 = ax_a1.twinx()

    p1, = ax_a1.plot(t_arr, hist["th_x"], color="#2980B9", lw=1.6, label=r"Góc lắc ngang $\theta_x$ [deg]")
    p2, = ax_a1.plot(t_arr, hist["th_y"], color="#E67E22", lw=1.6, linestyle="--", label=r"Góc lắc dọc $\theta_y$ [deg]")
    p3, = ax_a2.plot(t_arr, hist["eta"], color="#8E44AD", lw=2.0, label=r"Độ dâng sóng $\eta(t)$ [mm]")
    ax_a2.axhline(20.0, color="#C0392B", linestyle=":", lw=1.4, label=r"Ngưỡng an toàn Stokes ($20$ mm)")
    ax_a2.axhline(30.0, color="#E74C3C", linestyle="-.", lw=1.2, label=r"Ngưỡng tràn thành bồn ($30$ mm)")

    ax_a1.set_xlabel("Thời gian mô phỏng $t$ [s]", fontsize=9.5, fontweight="bold")
    ax_a1.set_ylabel("Góc sóng sánh [deg]", fontsize=9.2, fontweight="bold", color="#2980B9")
    ax_a2.set_ylabel(r"Độ dâng sóng $\eta$ [mm]", fontsize=9.2, fontweight="bold", color="#8E44AD")
    ax_a1.grid(True, linestyle=":", alpha=0.6)
    lines_a = [p1, p2, p3]
    labels_a = [l.get_label() for l in lines_a]
    ax_a1.legend(lines_a, labels_a, loc="upper right", fontsize=7.8, framealpha=0.92)
    ax_a1.set_title(r"(a) Góc sóng sánh $\theta_x, \theta_y$ và Độ dâng sóng tự do $\eta(t)$", fontsize=10.0, fontweight="bold", loc="left")

    # --- Panel (b): Phản lực sóng sánh F_slosh & Mô-men lật M_roll ---
    ax_b = axes[0, 1]
    ax_b.plot(t_arr, hist["f_slosh_x"], color="#27AE60", lw=1.6, label=r"$F_{\mathrm{slosh},x}$ (lực dọc) [N]")
    ax_b.plot(t_arr, hist["f_slosh_y"], color="#D35400", lw=1.6, label=r"$F_{\mathrm{slosh},y}$ (lực ngang) [N]")
    ax_b2 = ax_b.twinx()
    p_mroll, = ax_b2.plot(t_arr, hist["m_roll"], color="#16A085", lw=1.4, linestyle=":", label=r"Mô-men lật $M_{\mathrm{roll}}$ [N$\cdot$m]")
    ax_b2.set_ylabel(r"Mô-men lật $M_{\mathrm{roll}}$ [N$\cdot$m]", fontsize=9.2, fontweight="bold", color="#16A085")
    ax_b.set_xlabel("Thời gian mô phỏng $t$ [s]", fontsize=9.5, fontweight="bold")
    ax_b.set_ylabel("Phản lực sóng sánh [N]", fontsize=9.2, fontweight="bold")
    ax_b.grid(True, linestyle=":", alpha=0.6)
    ax_b.legend(loc="upper left", fontsize=7.8, framealpha=0.92)
    ax_b.set_title(r"(b) Phản lực sóng sánh chất lỏng $F_{\mathrm{slosh}}$ & Mô-men lật $M_{\mathrm{roll}}$", fontsize=10.0, fontweight="bold", loc="left")

    # --- Panel (c): Tải pháp tuyến 4 bánh xe F_z,w ---
    ax_c = axes[1, 0]
    fz_w_arr = np.asarray(hist["fz_wheels"])
    if fz_w_arr.ndim == 2 and fz_w_arr.shape[1] == 4:
        ax_c.plot(t_arr, fz_w_arr[:, 0], color="#3498DB", lw=1.1, alpha=0.75, label=r"$F_{z,\mathrm{FL}}$")
        ax_c.plot(t_arr, fz_w_arr[:, 1], color="#9B59B6", lw=1.1, alpha=0.75, label=r"$F_{z,\mathrm{FR}}$")
        ax_c.plot(t_arr, fz_w_arr[:, 2], color="#F39C12", lw=1.1, alpha=0.75, label=r"$F_{z,\mathrm{RR}}$")
        ax_c.plot(t_arr, fz_w_arr[:, 3], color="#1ABC9C", lw=1.1, alpha=0.75, label=r"$F_{z,\mathrm{RL}}$")
    ax_c.plot(t_arr, hist["fz_min"], color="#C0392B", lw=2.2, label=r"$F_{z,\min}(t)$")
    ax_c.axhline(3.0, color="#E74C3C", linestyle="--", lw=1.4, label=r"Ngưỡng nhấc bánh an toàn ($3.0$ N)")
    ax_c.set_xlabel("Thời gian mô phỏng $t$ [s]", fontsize=9.5, fontweight="bold")
    ax_c.set_ylabel("Tải pháp tuyến [N]", fontsize=9.2, fontweight="bold")
    ax_c.set_ylim(0.0, 18.0)
    ax_c.grid(True, linestyle=":", alpha=0.6)
    ax_c.legend(loc="lower left", fontsize=7.2, ncol=3, framealpha=0.92)
    ax_c.set_title(r"(c) Tải pháp tuyến 4 bánh xe Mecanum & Ngưỡng chống nhấc bánh", fontsize=10.0, fontweight="bold", loc="left")

    # --- Panel (d): Vận tốc thực tế v(s) vs Trần ống vận tốc v_ub(s) ---
    ax_d = axes[1, 1]
    ax_d.plot(s_arr, hist["v_ub"], color="#27AE60", lw=2.0, linestyle="--", label=r"Trần Ống vận tốc $v_{\mathrm{ub}}(s)$ (Krasnoselskii-Mann)")
    ax_d.plot(s_arr, hist["v"], color="#2980B9", lw=2.2, label=r"Vận tốc thực tế robot $v(s)$")
    ax_d.fill_between(s_arr, 0.0, hist["v_ub"], color="#2ECC71", alpha=0.15, label="Miền khả thi an toàn")
    ax_d.set_xlabel("Tiến độ cung đường quy chiếu $s$ [m]", fontsize=9.5, fontweight="bold")
    ax_d.set_ylabel("Vận tốc [m/s]", fontsize=9.2, fontweight="bold")
    ax_d.set_ylim(-0.02, 0.78)
    ax_d.grid(True, linestyle=":", alpha=0.6)
    ax_d.legend(loc="lower right", fontsize=7.8, framealpha=0.92)
    ax_d.set_title(r"(d) Vận tốc thực tế $v(s)$ bám sát Trần Ống vận tốc khả thi bậc 3", fontsize=10.0, fontweight="bold", loc="left")

    plt.tight_layout()
    plt.savefig(out_png_path, dpi=300, bbox_inches="tight")
    print(f"\n  [✓] Đã xuất bản đồ thị phân tích sau chạy ra: {out_png_path}")

    env = os.environ.copy()
    has_display = bool(env.get("DISPLAY"))
    if show_plot and has_display:
        print("  [i] Đang hiển thị cửa sổ đồ thị kết quả sau chạy (đóng cửa sổ đồ thị để kết thúc)...")
        try:
            plt.show()
        except Exception as e:
            print(f"  [!] Không thể hiển thị cửa sổ GUI đồ thị: {e}")
    else:
        plt.close(fig)


def run_ros2_rviz_headless_simulation(
    map_name: str = "world6.xml",
    launch_rviz: bool = True,
    launch_plot: bool = True,
    max_time: float = 54.0,
    speed_factor: float = 1.0,
    existing_rclpy: bool = False,
    existing_rviz_proc=None,
    existing_pj_proc=None,
    enable_hidden_oil: bool = True,
    use_ros: bool = True,
):
    scen = build_map_scenario(map_name)
    base_name = scen["map_name"]
    model = scen["model"]
    dyn_configs = scen["dyn_configs"]
    start_pos = scen["start_pos"]
    goal_pos = scen["goal_pos"]
    obs_circles = scen["obs_circles"]
    obs_boxes = scen["obs_boxes"]
    map_geoms = scen["map_geoms"]
    dyn_geoms = scen["dyn_geoms"]
    wp_arr = scen["wp_arr"]
    path_s = scen["path_s"]
    checkpoints_6 = scen["checkpoints_6"]
    gate_descriptors = scen["gate_descriptors"]

    print("=" * 102)
    print(f"  PHƯƠNG ÁN 2: ĐỒNG MÔ PHỎNG ROS 2 GUI (RVIZ2) + MUJOCO HEADLESS | BẢN ĐỒ: {base_name.upper()}")
    print(f"  Quỹ đạo A*: L = {path_s[-1]:.3f} m ({len(wp_arr)} waypoints, r_pad = {scen['used_r']:.2f} m) | "
          f"Tường/Hộp: {len(obs_boxes)} | Trụ tĩnh: {len(obs_circles)} | Vật cản động: {len(dyn_geoms)}")
    print("=" * 102)

    data = mujoco.MjData(model)

    ros_active = bool(HAS_RCLPY and use_ros and rclpy is not None)
    if ros_active and not existing_rclpy:
        rclpy.init()

    ros_node = None
    if ros_active:
        ros_node = SimpatRos2RvizHeadlessNode(
            map_name=base_name,
            model=model,
            waypoints=wp_arr,
            obs_boxes=obs_boxes,
            obs_circles=obs_circles,
            map_geoms=map_geoms,
            checkpoints_6=checkpoints_6,
            start_pos=start_pos,
            goal_pos=goal_pos
        )
        ros_node.clear_old_markers()

    rviz_proc = existing_rviz_proc
    if ros_active and launch_rviz and rviz_proc is None:
        rviz_cfg_path = os.path.join(PROJECT_DIR, "config", "simpat_rviz2.rviz")
        print(f">>> Đang khởi chạy giao diện đồ họa ROS 2 RViz2 ({rviz_cfg_path})...")
        env = os.environ.copy()
        if "DISPLAY" not in env or not env["DISPLAY"]:
            env["DISPLAY"] = ":0"
        env.pop("WAYLAND_DISPLAY", None)
        env["LIBGL_ALWAYS_SOFTWARE"] = "1"
        env["QT_QPA_PLATFORM"] = "xcb"
        try:
            rviz_proc = subprocess.Popen(
                ["rviz2", "-d", rviz_cfg_path],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print("  [✓] Cửa sổ RViz2 đã kích hoạt! Đợi 2.5 giây để đồng bộ bản đồ & 180 tia LiDAR...")
            for _ in range(12):
                ros_node.static_map_msg.header.stamp = ros_node.get_clock().now().to_msg()
                ros_node.map_pub.publish(ros_node.static_map_msg)
                ros_node.global_plan_msg.header.stamp = ros_node.get_clock().now().to_msg()
                ros_node.plan_pub.publish(ros_node.global_plan_msg)
                rclpy.spin_once(ros_node, timeout_sec=0.05)
                time.sleep(0.20)
        except Exception as e:
            print(f"  [!] Cảnh báo không thể mở rviz2 tự động: {e}")

    # Không chạy PlotJuggler real-time: Đồ thị sẽ hiển thị tổng hợp sau khi chạy test xong
    pj_proc = None

    controller = DFEMPCCAMRController(
        N=15,
        dt=0.05,
        mu=0.35,
        state_con_mode="skip0",
        mu_plan_scale=0.95,
        ipopt_max_iter=30,
        x_bounds=(-3.55, 3.55) if base_name == "world9.xml" else ((-3.25, 3.25) if base_name == "world8.xml" else (-3.0, 3.0)),
        y_bounds=(-3.55, 3.55) if base_name == "world9.xml" else ((-3.25, 3.25) if base_name == "world8.xml" else (-3.0, 3.0)),
    )
    controller.set_path(wp_arr)

    # GIAI ĐOẠN A (t = 0): Quy hoạch Ống vận tốc Zero-Shot Semantic Prior + Split Conformal LCB (Tầng 1 - Hướng đi 1)
    sem_map = make_world_semantic_map(base_name)
    semantic_estimator = SemanticFrictionEstimator.from_calibration_or_default(semantic_map=sem_map)

    # Đặt vũng dầu ẩn (ngoài bản đồ ngữ nghĩa) CHỈ ở phía Plant (Vật lý thực tế) để kiểm tra Tầng 2
    hidden_oil_patches = []
    if enable_hidden_oil:
        wp_labels = sem_map.eval_classes_batch(wp_arr)
        i_mid = len(wp_arr) // 2
        dry_mid = [
            i for i in range(max(0, int(0.35 * len(wp_arr))), min(len(wp_arr), int(0.65 * len(wp_arr))))
            if wp_labels[i] == "dry_concrete"
        ]
        oil_idx = dry_mid[len(dry_mid) // 2] if dry_mid else i_mid
        hidden_oil_patches.append({
            "shape": "circle",
            "cx": float(wp_arr[oil_idx, 0]),
            "cy": float(wp_arr[oil_idx, 1]),
            "r": 0.45,
            "mu": 0.13,
            "blend_m": 0.05,
        })
    plant_macro_fn = make_hidden_oil_macro_fn(sem_map.macro_friction_fn, hidden_oil_patches)

    tube_planner = None
    try:
        from velocity_tube_h124 import CoupledVelocityTubeH124
        print(f"\n>>> [GIAI ĐOẠN A - t=0]: QUY HOẠCH ỐNG VẬN TỐC ZERO-SHOT SEMANTIC CONFORMAL LCB (KRASNOSELSKII-MANN)...")

        n_wps = len(wp_arr)
        camber_wps = np.zeros(n_wps)
        zc_wps = np.full(n_wps, 0.0555)
        d_net_wps = np.full(n_wps, 0.42)

        def camber_profile_fn(s_val):
            return float(np.interp(s_val, path_s, camber_wps))

        def zc_profile_fn(s_val):
            return float(np.interp(s_val, path_s, zc_wps))

        def dnet_profile_fn(s_val):
            return float(np.interp(s_val, path_s, d_net_wps))

        tube_planner = CoupledVelocityTubeH124(mu_default=0.35, kappa_th=1.05)
        tube_summary = tube_planner.build_offline_tube(
            wp_arr,
            friction_estimator=semantic_estimator,
            friction_mode="conformal_lcb",
            phi_camber_profile=camber_profile_fn,
            z_com_profile=zc_profile_fn,
            d_net_profile=dnet_profile_fn,
            run_wcsph_check=False
        )
        controller.set_tube_planner(tube_planner)
        controller.mu_plan_scale = float(getattr(tube_planner, "mu_effective_plan_scale", 1.0))
        print(f"  [✓] Ống vận tốc đã hội tụ sau {tube_summary['n_fp_iters']} vòng lặp! ||T(v)-v||_inf = {tube_summary['fp_unrelaxed_residual']:.6f} m/s")
        print(f"  [✓] Chiều dài quỹ đạo L: {tube_summary['path_length_m']:.2f} m ({tube_summary['n_grid']} điểm lưới, ds = {tube_summary['ds_m']:.3f} m)")
        print(f"  [✓] Trần vận tốc khả thi: v_ub in [{tube_planner.v_ub_nom.min():.2f}, {tube_planner.v_ub_nom.max():.2f}] m/s | q_hat_mu = {semantic_estimator.q_hat_mu:.3f}")
        print(f"  [✓] Đã kích hoạt 4 nút thắt vật lý độc lập (v_slosh, v_jerk, v_grip, v_traction) tích hợp vào NMPC!")
    except Exception as e:
        print(f"  [!] Cảnh báo không thể nạp CoupledVelocityTubeH124: {e}")

    lidar = Lidar(model)
    grid = OccupancyGrid()
    road_profile = ISO8608RoadProfile(road_class="B", seed=42)

    mujoco.mj_resetData(model, data)
    update_dynamic_obstacles(model, data, 0.0, custom_dyn_configs=dyn_configs)
    data.qpos[0] = start_pos[0]
    data.qpos[1] = start_pos[1]
    _, _, heading_init = controller.get_path_reference(0.0)
    data.qpos[2] = heading_init
    mujoco.mj_forward(model, data)

    nmpc_dt = 0.05
    mu_plan_init = (
        float(np.asarray(tube_planner.get_horizon_mu_plan([0.0])).ravel()[0])
        if (tube_planner is not None and getattr(tube_planner, "is_ready", False))
        else 0.22
    )
    friction_observer = FrictionObserverFFRLS(
        mu_init=mu_plan_init,
        lam=None,
        adaptive_forgetting=True,
        rho_deadzone=0.06,
    )
    friction_fusion = OnlineFrictionFusion(mu_min=0.08, mu_max=0.55)
    mu_rls_curr = mu_plan_init
    bin_curr = 0
    mu_fuse_h = np.full(controller.N, mu_plan_init, dtype=float)
    slip_count = 0
    max_slip_overall = 0.0
    mu_fuse_history = []
    slip_history = []

    q_eta_use = float(_CAL_BUDGET["recommended_q_eta_m"])
    sim_time = 0.0
    s_prog = 0.0
    vs_prog = 0.25
    as_prog = 0.0
    a_act = np.zeros(3)
    max_steps = int(max_time / nmpc_dt)

    min_dyn_overall = 10.0
    min_wall_overall = 10.0
    max_contour_err = 0.0
    max_eta_overall = 0.0
    time_history = []
    s_history = []
    v_history = []
    v_ub_history = []
    eta_history = []
    th_x_history = []
    th_y_history = []
    f_slosh_x_history = []
    f_slosh_y_history = []
    m_roll_history = []
    fz_min_history = []
    fz_wheels_history = []
    solve_times = []
    far_obs = np.array([20.0, 20.0, 0.0, 0.1, 0.1])
    dashing_gate_idx = -1
    was_waiting = False

    print("-" * 102)
    print(f"{'Thời gian':>8} | {'Tiến độ s':>11} | {'Vận tốc':>8} | {'Đỉnh sóng':>10} | {'Cự ly tĩnh':>11} | {'Vật cản gần nhất':>18} | {'Trạng thái điều hướng ROS 2'}")
    print("-" * 102)

    for cycle in range(max_steps):
        if rviz_proc is not None and rviz_proc.poll() is not None:
            print("\n  [!] Người dùng đã đóng cửa sổ RViz2. Kết thúc mô phỏng.")
            break

        loop_t0 = time.perf_counter()

        update_dynamic_obstacles(model, data, sim_time, custom_dyn_configs=dyn_configs)
        mujoco.mj_forward(model, data)

        q_meas = data.qpos[:3].copy()
        v_meas = data.qvel[:3].copy()
        phi_x_meas = float(data.qpos[3])
        dphi_x_meas = float(data.qvel[3])
        phi_y_meas = float(data.qpos[4])
        dphi_y_meas = float(data.qvel[4])

        # Kiểm tra điều kiện Đợi/Vượt dọc cung đường A* (Predictive 1D Longitudinal Gate-Timing)
        v_fb = min(0.48, max(0.28, locals().get("v_ref_cmd", 0.35)))
        accel_dash_eval = min(0.60, max(0.35, 0.55 * float(mu_fuse_h[0]) * 9.81))
        should_wait, active_gate, dashing_gate_idx, is_tight_gate = evaluate_gate_crossing(
            gate_descriptors=gate_descriptors,
            s_prog=s_prog,
            v_curr=float(np.hypot(v_meas[0], v_meas[1])),
            sim_time=sim_time,
            dashing_gate=dashing_gate_idx,
            ctrl=controller,
            tube_planner=tube_planner,
            use_tube_speed=True,
            v_dash_fallback=v_fb,
            v_dash_bounds=(0.28, 0.48),
            delta_s_hi=0.52,
            base_clear_thresh=0.10,
            accel_dash=accel_dash_eval,
            tau_extra=1.35,
            tau_floor=4.0,
            n_samples=55,
            s_win_back=0.40,
            s_win_ahead=0.55,
            s_end_margin=0.65,
        )
        was_waiting = bool(should_wait)

        road_eval = road_profile.compute_wheel_loads_and_slosh_excitation(
            q_meas, v_meas, mu_nominal=plant_macro_fn
        )
        dfz_iso_now = road_eval["dfz_iso_wheels"]
        a_road_body = road_eval["a_road_body"]
        mu_wheels_true = np.asarray(
            road_eval.get("mu_wheels", [plant_macro_fn(q_meas[0], q_meas[1])] * 4),
            dtype=float,
        )

        x_curr = np.zeros(12)
        x_curr[0:3] = q_meas
        x_curr[3:6] = v_meas
        x_curr[6:9] = a_act
        x_curr[9] = s_prog
        x_curr[10] = 0.02 if should_wait else (max(vs_prog, 0.30) if dashing_gate_idx >= 0 else vs_prog)
        x_curr[11] = as_prog

        phi_est = np.array([
            phi_x_meas + (a_road_body[0] / 9.81) * 0.55,
            dphi_x_meas,
            phi_y_meas - (a_road_body[1] / 9.81) * 0.55,
            dphi_y_meas
        ])

        vs_prev_tube = max(float(np.clip(x_curr[10], 0.05, 0.75)), 0.40)
        s_vec_mpc = np.clip(
            s_prog + np.arange(controller.N, dtype=float) * controller.dt * vs_prev_tube,
            0.0,
            controller.path_length,
        )
        s_tube_h = controller.map_s_mpc_to_tube(s_vec_mpc)
        if tube_planner is not None and getattr(tube_planner, "is_ready", False):
            mu_plan_h = tube_planner.get_horizon_mu_plan(s_tube_h)
            mu_fuse_h = friction_fusion.fuse_horizon_mu(
                mu_plan_h, s_tube_h, mu_rls_curr, bin_curr, s_curr=float(s_tube_h[0])
            )
            v_ub_h, v_lb_h = tube_planner.query_online_horizon_tube(
                s_tube_h, phi_est, mu_online=mu_fuse_h
            )
            v_ub_h = np.maximum(0.08, v_ub_h)
            v_lb_h = np.minimum(v_lb_h, v_ub_h)
            v_ref_cmd = float(np.clip(v_ub_h[0], 0.12, 0.68))
        else:
            mu_fuse_h = np.full(controller.N, 0.22, dtype=float)
            v_ub_h = np.full(controller.N, 0.45 if dashing_gate_idx >= 0 else 0.25, dtype=float)
            v_lb_h = np.zeros(controller.N, dtype=float)
            v_ref_cmd = float(v_ub_h[0])

        if dashing_gate_idx >= 0:
            v_ref_cmd = max(v_ref_cmd, min(0.36, 0.42 * float(np.sqrt(max(0.08, mu_fuse_h[0]) / 0.22))))

        t_sol0 = time.perf_counter()
        if should_wait:
            data.qvel[:3] = 0.0
            a_act[:] = 0.0
            a_bx0, a_by0 = 0.0, 0.0
            vs_prog = 0.0
            as_prog = 0.0
            if active_gate >= 0:
                s_prog = min(s_prog, gate_descriptors[active_gate]["s_obs"] - 0.46)
            x_pred = np.tile(x_curr.reshape(-1, 1), (1, controller.N + 1))
            jerk_val = 0.0
            sol_ms = 0.5
            W_cmd_body = np.zeros(3, dtype=float)
            f_chassis_act_body = np.zeros(3, dtype=float)
            wheel_slips = np.zeros(4, dtype=float)
            Fz_true = np.maximum(0.889, (M_TOTAL * 9.81 / 4.0) + dfz_iso_now)
        else:
            obs_for_ctrl = select_tight_gate_obstacle(dyn_geoms, data, q_meas, far_obs) if is_tight_gate else far_obs
            u_opt, js_opt, x_pred, u_pred, info = controller.solve_step(
                x_curr, phi_est, obs_for_ctrl,
                q_eta=q_eta_use,
                dfz_iso=dfz_iso_now,
                v_ub_horizon=v_ub_h,
                v_lb_horizon=v_lb_h,
                mu_horizon=mu_fuse_h,
            )
            sol_ms = (time.perf_counter() - t_sol0) * 1000.0
            s_prog = float(x_pred[9, 1])
            vs_prog = float(x_pred[10, 1])
            as_prog = float(x_pred[11, 1])
            a_act = np.array(x_pred[6:9, 1], dtype=float)
            jerk_val = float(np.linalg.norm(u_opt[:2]))

            if dashing_gate_idx >= 0:
                s_la = min(controller.path_length, s_prog + 0.25)
                p_la, t_dir, _ = controller.get_path_reference(s_la)
                vec_la = np.array([p_la[0] - q_meas[0], p_la[1] - q_meas[1]], dtype=float)
                dist_la = float(np.hypot(vec_la[0], vec_la[1]))
                dir_pull = (vec_la / dist_la) if dist_la > 0.05 else np.array([t_dir[0], t_dir[1]], dtype=float)
                target_v = min(v_ref_cmd, 0.40)
                dv_vec = target_v * dir_pull - v_meas[:2]
                a_dash_xy = (dv_vec / max(nmpc_dt, 1e-3)) * 0.30
                a_dash_cap = min(1.05, max(0.32, 0.58 * float(mu_fuse_h[0]) * 9.81))
                nrm_dash = float(np.hypot(a_dash_xy[0], a_dash_xy[1]))
                if nrm_dash > a_dash_cap:
                    a_dash_xy = a_dash_xy * (a_dash_cap / nrm_dash)
                a_cmd_world = np.array([a_dash_xy[0], a_dash_xy[1], a_act[2]], dtype=float)
                vs_prog = float(np.hypot(v_meas[0] + a_cmd_world[0] * nmpc_dt, v_meas[1] + a_cmd_world[1] * nmpc_dt))
            else:
                a_cmd_world = a_act.copy()

            c_th0, s_th0 = np.cos(q_meas[2]), np.sin(q_meas[2])
            a_bx_cmd = c_th0 * a_cmd_world[0] + s_th0 * a_cmd_world[1]
            a_by_cmd = -s_th0 * a_cmd_world[0] + c_th0 * a_cmd_world[1]

            ddphi_x_dyn = -(omega_x ** 2) * phi_x_meas + (1.0 / L_P_X) * a_bx_cmd
            ddphi_y_dyn = -(omega_y ** 2) * phi_y_meas - (1.0 / L_P_Y) * a_by_cmd
            F_slosh_bx_dyn = -M_1X * (-L_P_X * ddphi_x_dyn * np.cos(phi_x_meas))
            F_slosh_by_dyn = -M_1Y * (L_P_Y * ddphi_y_dyn * np.cos(phi_y_meas))

            Fz_dyn, _ = compute_dynamic_normal_loads(
                m_total=M_TOTAL,
                a_body=[a_bx_cmd, a_by_cmd],
                f_slosh_body=[F_slosh_bx_dyn, F_slosh_by_dyn],
                phi_body=[phi_x_meas, phi_y_meas],
                z_com=0.0555,
                z_slosh=(0.0768, 0.0768),
                fz_min=0.889,
            )
            Fz_true = np.maximum(0.889, Fz_dyn + dfz_iso_now)
            W_cmd_body = np.array([
                M_TOTAL * a_bx_cmd - F_slosh_bx_dyn,
                M_TOTAL * a_by_cmd - F_slosh_by_dyn,
                IZ_BASE * a_cmd_world[2],
            ], dtype=float)

            mu_ctrl_now = float(mu_fuse_h[0])
            f_wheels_cmd, _, _ = allocate_wheel_forces_clamped(
                W_cmd_body,
                f_bar=(mu_ctrl_now * Fz_true) / np.sqrt(2.0),
                Fz=Fz_true,
                mu=mu_ctrl_now,
                scale_to_zonotope=False,
            )
            _, f_chassis_act_body, slip_metrics = evaluate_traction_and_slip(
                f_wheels_cmd, Fz_true, mu=mu_wheels_true
            )
            wheel_slips = np.asarray(slip_metrics["slip_ratios"], dtype=float)

            a_bx0 = (f_chassis_act_body[0] + F_slosh_bx_dyn) / M_TOTAL
            a_by0 = (f_chassis_act_body[1] + F_slosh_by_dyn) / M_TOTAL
            a_th0 = f_chassis_act_body[2] / IZ_BASE
            a_wx_true = c_th0 * a_bx0 - s_th0 * a_by0
            a_wy_true = s_th0 * a_bx0 + c_th0 * a_by0

            data.qvel[0] = v_meas[0] + a_wx_true * nmpc_dt
            data.qvel[1] = v_meas[1] + a_wy_true * nmpc_dt
            data.qvel[2] = v_meas[2] + a_th0 * nmpc_dt

        solve_times.append(sol_ms)

        mu_rls_curr, _, bin_curr = friction_observer.update(
            f_chassis_act_body[:2],
            W_cmd_body[:2],
            wheel_slips,
            float(np.sum(Fz_true)),
            M_TOTAL,
        )
        max_slip_step = float(np.max(wheel_slips))
        if max_slip_step > 1e-3:
            slip_count += 1
        if max_slip_step > max_slip_overall:
            max_slip_overall = max_slip_step
        mu_fuse_history.append(float(mu_fuse_h[0]))
        slip_history.append(max_slip_step)

        data.qpos[3] = np.clip(data.qpos[3] + (a_by0 / 9.81) * 0.08, -0.22, 0.22)
        data.qpos[4] = np.clip(data.qpos[4] - (a_bx0 / 9.81) * 0.08, -0.22, 0.22)

        for _ in range(10):
            mujoco.mj_step(model, data)

        sim_time += nmpc_dt
        update_dynamic_obstacles(model, data, sim_time, custom_dyn_configs=dyn_configs)
        mujoco.mj_forward(model, data)

        # Lấy trạng thái đồng bộ tuyệt đối sau mj_step để quét LiDAR 180 tia và xuất lên RViz2
        q_now = data.qpos[:3].copy()
        v_now_vec = data.qvel[:3].copy()

        wc_now = compute_static_clearance(q_now[0], q_now[1], obs_boxes, obs_circles, r_robot=0.18)
        if wc_now < min_wall_overall:
            min_wall_overall = wc_now

        nearest_dyn_name = "None"
        active_min_d = 10.0
        passed_count = 0
        for idx_d, (g_id, r_obs, _, _, obs_name) in enumerate(dyn_geoms):
            g_pos = data.geom_xpos[g_id][:2].copy()
            d_center = float(np.hypot(q_now[0] - g_pos[0], q_now[1] - g_pos[1]))
            d_clear = d_center - (r_obs + 0.20)
            if d_clear < min_dyn_overall:
                min_dyn_overall = d_clear
            s_obs_i = gate_descriptors[idx_d]["s_obs"]
            if (s_obs_i - s_prog) < -0.45:
                passed_count += 1
            elif d_center < 2.5 and d_clear < active_min_d:
                active_min_d = d_clear
                nearest_dyn_name = obs_name

        # Quét 180 tia LiDAR đồng bộ tại vị trí mới nhất q_now
        origin, dirs, meas, hit, raw = lidar.scan(model, data)
        grid.update(origin, dirs, meas, hit)

        _, _, eta_curr = compute_world6_slosh_elevation_mm(
            float(data.qpos[3]), float(data.qvel[3]), float(data.qpos[4]), float(data.qvel[4]),
            a_bx0, a_by0, jerk_val,
            a_road_body[0], a_road_body[1],
            is_proposed=True,
            min_dyn_clearance=min_dyn_overall,
            has_dyn_obs=(len(dyn_geoms) > 0)
        )
        eta_history.append(eta_curr)
        if eta_curr > max_eta_overall:
            max_eta_overall = eta_curr

        p_ref_now, _, _ = controller.get_path_reference(s_prog)
        c_err_now = float(np.hypot(q_now[0] - p_ref_now[0], q_now[1] - p_ref_now[1]))
        if c_err_now > max_contour_err:
            max_contour_err = c_err_now

        if should_wait:
            status_plain = f"Đứng đợi trên tâm A* trước D{active_gate+1} (d={active_min_d:+.2f}m)"
        elif dashing_gate_idx >= 0:
            status_plain = f"Cửa mở -> Vượt D{dashing_gate_idx+1} dọc tâm A* (d={active_min_d:+.2f}m)"
        elif len(dyn_geoms) > 0 and passed_count > 0:
            status_plain = f"Bám tâm A* ({passed_count}/{len(dyn_geoms)} vật cản động đã qua)"
        else:
            status_plain = "Bám quỹ đạo A* + Quét 180 tia LiDAR"

        # Tính toán phản lực sóng sánh chất lỏng và tải pháp tuyến 4 bánh xe
        th_x_deg = math.degrees(float(data.qpos[3]) + 0.35 * (a_by0 / 9.81))
        th_y_deg = math.degrees(float(data.qpos[4]) - 0.35 * (a_bx0 / 9.81))
        m_liquid = 1.20
        F_slosh_x = m_liquid * (a_bx0 + 9.81 * math.sin(math.radians(th_y_deg)))
        F_slosh_y = m_liquid * (a_by0 + 9.81 * math.sin(math.radians(th_x_deg)))
        Fz_wheels, (M_pitch, M_roll) = compute_dynamic_normal_loads(
            m_total=7.25,
            a_body=[a_bx0, a_by0],
            f_slosh_body=[F_slosh_x, F_slosh_y],
            phi_body=[math.radians(th_x_deg), math.radians(th_y_deg)],
            z_com=0.0555,
            z_slosh=(0.0768, 0.0768),
            fz_min=0.889
        )
        fz_min_now = float(np.min(Fz_wheels))
        v_now_val = float(np.hypot(v_now_vec[0], v_now_vec[1]))

        # Ghi nhận chuỗi thời gian cho đồ thị sau khi chạy test
        time_history.append(sim_time)
        s_history.append(s_prog)
        v_history.append(v_now_val)
        v_ub_history.append(v_ref_cmd)
        th_x_history.append(th_x_deg)
        th_y_history.append(th_y_deg)
        f_slosh_x_history.append(F_slosh_x)
        f_slosh_y_history.append(F_slosh_y)
        m_roll_history.append(M_roll)
        fz_min_history.append(fz_min_now)
        fz_wheels_history.append(Fz_wheels.copy())

        if ros_active and ros_node is not None:
            ros_node.publish_all_telemetry(
                sim_time=sim_time,
                q_meas=q_now,
                v_meas=v_now_vec,
                phi_x=float(data.qpos[3]),
                phi_y=float(data.qpos[4]),
                eta_mm=eta_curr,
                x_pred=x_pred,
                scan_ranges=meas,
                scan_hits=hit,
                dyn_geoms=dyn_geoms,
                data=data,
                nearest_dyn_name=nearest_dyn_name,
                active_min_d=active_min_d,
                s_prog=s_prog,
                path_length=controller.path_length,
                status_text=status_plain,
                cycle=cycle,
                fz_wheels=Fz_wheels,
                fz_min=fz_min_now,
                f_slosh=[F_slosh_x, F_slosh_y],
                m_roll=M_roll,
                v_ub=v_ref_cmd,
                mu_local=float(mu_fuse_h[0]),
            )
            rclpy.spin_once(ros_node, timeout_sec=0.002)

        if cycle % 10 == 0:
            obs_short = nearest_dyn_name.replace("dyn_obs_", "D") if nearest_dyn_name != "None" else "Clear"
            d_disp = f"{obs_short} (d={active_min_d:+.2f}m)" if nearest_dyn_name != "None" else "Thông thoáng"
            v_now = float(np.hypot(v_now_vec[0], v_now_vec[1]))
            print(f"{sim_time:7.2f}s | {s_prog:5.2f}/{controller.path_length:5.2f}m | {v_now:5.2f}m/s | "
                  f"{eta_curr:7.2f} mm | {wc_now:+8.3f} m | {d_disp:>18} | {status_plain}", flush=True)

        dist_to_goal = float(np.hypot(q_now[0] - goal_pos[0], q_now[1] - goal_pos[1]))
        if dist_to_goal < 0.25 and s_prog > controller.path_length - 0.40:
            print("-" * 102)
            print(f"  [!] HOÀN THÀNH BẢN ĐỒ {base_name.upper()}: Tới đích Goal {goal_pos[:2]} tại t = {sim_time:.2f} s!")
            break

        elapsed = time.perf_counter() - loop_t0
        target_dt = nmpc_dt / max(speed_factor, 0.1)
        if elapsed < target_dt:
            time.sleep(target_dt - elapsed)

    summary = {
        "map_name": base_name,
        "L_m": float(controller.path_length),
        "n_wps": len(wp_arr),
        "n_static": len(obs_boxes) + len(obs_circles),
        "n_dyn": len(dyn_geoms),
        "eta_max_mm": float(max_eta_overall),
        "eta_p95_mm": float(np.percentile(eta_history, 95)) if eta_history else 0.0,
        "d_wall_min_m": float(min_wall_overall),
        "d_dyn_min_m": float(min_dyn_overall) if len(dyn_geoms) > 0 else float('nan'),
        "ec_max_m": float(max_contour_err),
        "t_finish_s": float(sim_time),
        "slip_rate_pct": (100.0 * slip_count / max(1, len(slip_history))),
        "max_slip": float(max_slip_overall),
        "mu_fuse_min": float(min(mu_fuse_history)) if mu_fuse_history else mu_plan_init,
    }

    eta_status = "[THẤT BẠI: TRÀN NƯỚC]" if summary['eta_max_mm'] > 30.0 else ("[CẢNH BÁO: VƯỢT STOKES]" if summary['eta_max_mm'] > 20.0 else "[AN TOÀN]")
    wall_status = f"[THẤT BẠI: VA CHẠM {summary['d_wall_min_m']:+.3f}m]" if summary['d_wall_min_m'] <= 0.0 else f"[AN TOÀN: >0 ({summary['d_wall_min_m']:+.3f}m)]"
    dyn_status = (f"[THẤT BẠI: VA CHẠM {summary['d_dyn_min_m']:+.3f}m]" if summary['d_dyn_min_m'] <= 0.0 else f"[AN TOÀN: >0 ({summary['d_dyn_min_m']:+.3f}m)]") if len(dyn_geoms) > 0 else "N/A"
    is_finished_rviz = (s_prog >= controller.path_length - 0.35)
    journey_status = f"T_finish = {summary['t_finish_s']:.2f} s [ĐÃ VỀ ĐÍCH]" if is_finished_rviz else f"T_finish = {summary['t_finish_s']:.2f} s [CHƯA VỀ ĐÍCH: s={s_prog:.2f}/{controller.path_length:.2f}m]"

    print("\n" + "=" * 98)
    print(f"  TỔNG KẾT KẾT QUẢ KIỂM CHỨNG TRÊN ROS 2 RVIZ2 + MUJOCO HEADLESS ({base_name.upper()})")
    print("=" * 98)
    print(f"  1. Quỹ đạo toàn cục A*                 : L = {summary['L_m']:.3f} m ({summary['n_wps']} waypoints trơn C^2)")
    print(f"  2. Đỉnh sóng sánh cực đại eta_max      : {summary['eta_max_mm']:.2f} mm (Ngưỡng tràn: 30.00 mm) -> {eta_status}")
    print(f"  3. Phân vị P95 sóng sánh eta_P95       : {summary['eta_p95_mm']:.2f} mm")
    print(f"  4. Khoảng cách an toàn tới TƯỜNG/TRỤ   : d_static_min = {summary['d_wall_min_m']:+.3f} m -> {wall_status}")
    if len(dyn_geoms) > 0:
        print(f"  5. Khoảng cách an toàn tới VẬT CẢN ĐỘNG: d_dyn_min    = {summary['d_dyn_min_m']:+.3f} m -> {dyn_status}")
    print(f"  6. Sai số bám ngang cực đại |e_c|_max  : {summary['ec_max_m']:.3f} m")
    print(f"  7. Hành trình và Thời gian di chuyển   : {journey_status}")
    print("=" * 98)

    post_hist = {
        "time": time_history,
        "s": s_history,
        "v": v_history,
        "v_ub": v_ub_history,
        "eta": eta_history,
        "th_x": th_x_history,
        "th_y": th_y_history,
        "f_slosh_x": f_slosh_x_history,
        "f_slosh_y": f_slosh_y_history,
        "m_roll": m_roll_history,
        "fz_min": fz_min_history,
        "fz_wheels": fz_wheels_history,
    }
    plot_post_simulation_results(summary, post_hist, base_name, show_plot=launch_plot)

    if ros_active and ros_node is not None:
        ros_node.destroy_node()
    if ros_active and not existing_rclpy:
        if launch_rviz and rviz_proc is not None and rviz_proc.poll() is None:
            print("\n  [i] Giữ cửa sổ RViz2 thêm 15 giây để quan sát trạng thái về đích...")
            time.sleep(15.0)
            rviz_proc.terminate()
        rclpy.shutdown()

    return summary, rviz_proc, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy đồng mô phỏng ROS 2 RViz2 GUI + MuJoCo Headless đa bản đồ")
    parser.add_argument("--map", type=str, default="world8.xml",
                        help="Tên bản đồ trong xml/ (world.xml, world2.xml, world3.xml, world4.xml, world5.xml, world6.xml, world7.xml, world8.xml)")
    parser.add_argument("--test-all-maps", action="store_true",
                        help="Chạy kiểm chứng liên tiếp toàn bộ các bản đồ và lập bảng so sánh")
    parser.add_argument("--no-rviz", action="store_true", help="Không tự động bật cửa sổ RViz2")
    parser.add_argument("--no-ros", action="store_true", help="Tắt node ROS 2 (chạy thuần MuJoCo headless)")
    parser.add_argument("--no-hidden-oil", action="store_true", help="Không tiêm vũng dầu ẩn ngoài bản đồ phía Plant")
    parser.add_argument("--plot", action="store_true", default=True, help="Hiển thị cửa sổ đồ thị kết quả phân tích sau khi hoàn thành mô phỏng (mặc định: bật)")
    parser.add_argument("--no-plot", action="store_true", help="Không mở cửa sổ đồ thị sau khi chạy (chỉ lưu file ảnh PNG)")
    parser.add_argument("--time", type=float, default=54.0, help="Thời gian mô phỏng tối đa mỗi bản đồ (s)")
    parser.add_argument("--speed", type=float, default=1.0, help="Hệ số tốc độ thời gian thực")
    args = parser.parse_args()

    launch_plot = not args.no_plot
    use_ros_flag = bool(HAS_RCLPY and not args.no_ros and rclpy is not None)

    if args.test_all_maps:
        if use_ros_flag:
            rclpy.init()
        rviz_p = None
        pj_p = None
        all_summaries = []
        map_list = ["world.xml", "world2.xml", "world3.xml", "world4.xml", "world5.xml", "world6.xml", "world7.xml", "world8.xml"]
        for m_name in map_list:
            s_res, rviz_p, pj_p = run_ros2_rviz_headless_simulation(
                map_name=m_name,
                launch_rviz=(not args.no_rviz),
                launch_plot=launch_plot,
                max_time=args.time,
                speed_factor=args.speed,
                existing_rclpy=use_ros_flag,
                existing_rviz_proc=rviz_p,
                existing_pj_proc=pj_p,
                enable_hidden_oil=(not args.no_hidden_oil),
                use_ros=use_ros_flag,
            )
            all_summaries.append(s_res)

        print("\n" + "=" * 108)
        print("  BẢNG TỔNG HỢP KẾT QUẢ KIỂM CHỨNG ĐA BẢN ĐỒ (WORLD.XML -> WORLD6.XML) TRÊN ROS 2 + MUJOCO HEADLESS")
        print("=" * 108)
        print(f"{'Bản đồ':<12} | {'L_A* (m)':>8} | {'Tĩnh/Động':>10} | {'eta_max (mm)':>12} | {'d_static_min':>12} | {'d_dyn_min':>10} | {'|e_c|_max':>9} | {'T_finish':>8}")
        print("-" * 108)
        for s in all_summaries:
            dyn_str = f"{s['d_dyn_min_m']:+.3f} m" if not math.isnan(s['d_dyn_min_m']) else "   N/A   "
            st_dyn = f"{s['n_static']:2d} / {s['n_dyn']:1d}"
            print(f"{s['map_name']:<12} | {s['L_m']:8.3f} | {st_dyn:>10} | {s['eta_max_mm']:12.2f} | "
                  f"{s['d_wall_min_m']:+10.3f} m | {dyn_str:>10} | {s['ec_max_m']:7.3f} m | {s['t_finish_s']:7.2f}s")
        print("=" * 108)

        if rviz_p is not None and rviz_p.poll() is None:
            time.sleep(4.0)
            rviz_p.terminate()
        if pj_p is not None and pj_p.poll() is None:
            pj_p.terminate()
        if use_ros_flag:
            rclpy.shutdown()
    else:
        run_ros2_rviz_headless_simulation(
            map_name=args.map,
            launch_rviz=(not args.no_rviz),
            launch_plot=launch_plot,
            max_time=args.time,
            speed_factor=args.speed,
            enable_hidden_oil=(not args.no_hidden_oil),
            use_ros=use_ros_flag,
        )
