#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================
run_ros2_mpcc_3way_benchmark.py — ĐỐI CHỨNG 2 CẤU HÌNH DFE-MPCC (CÓ VS KHÔNG CHECKPOINT)
                  TRÊN ROS 2 RVIZ2 / MUJOCO HEADLESS (BẢN ĐỒ WORLD7.XML, MA SÁT mu = 0.35)
=========================================================================================
Thực hiện benchmark 2 lượt chạy nối tiếp độc lập trên bản đồ world7.xml để tách bạch
tác động của WCSPH Checkpoint & Conformal margin q_eta (mặt sàn phẳng, mu = 0.35 đồng nhất,
chưa kích hoạt ước lượng ma sát hay bẫy ma sát):

  - LƯỢT 1/2 (MÀU CAM - DFE-MPCC Không Checkpoint): DFE-MPCC cơ sở với bộ giải/bộ lọc
    tương đương, nhưng chưa chuẩn hóa Checkpoint WCSPH & Conformal (q_eta = 0, không ống H124).
  - LƯỢT 2/2 (MÀU XANH LỤC - DFE-MPCC Có Checkpoint SPH): Tích hợp đầy đủ Ống vận tốc
    Krasnoselskii-Mann H124 + Checkpoint WCSPH Hybrid B+A & dự trữ phân vị Conformal q_eta = 3.70 mm.
=========================================================================================
"""

import os
import sys
import time
import math
import json
import argparse
import subprocess
import numpy as np
import matplotlib
if "--no-plot" in sys.argv or "--all-maps" in sys.argv or not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ["", "controller", "solver", "planning", "simulation", "cfd"]:
    p = os.path.join(PROJECT_DIR, sub) if sub else PROJECT_DIR
    if p not in sys.path:
        sys.path.insert(0, p)

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import mujoco
from dfe_mpcc_amr_controller import DFEMPCCAMRController
from iso8608_road_profile import ISO8608RoadProfile, get_world7_zone_friction
from test_sim_amr_closed_loop import update_dynamic_obstacles, Lidar
from mecanum_traction_dynamics import compute_dynamic_normal_loads
from conformal_checkpoint_eval import load_conformal_calibration_budget
from benchmark_world6_complex_4dyn_obs import compute_world6_slosh_elevation_mm

_CAL_BUDGET = load_conformal_calibration_budget()
from run_ros2_rviz_headless_mujoco import (
    HAS_RCLPY,
    ColorRGBA,
    rclpy,
    build_map_scenario,
    compute_static_clearance,
    euler_to_quaternion,
    evaluate_gate_crossing,
    predict_gate_pos,
    select_tight_gate_obstacle,
    SimpatRos2RvizHeadlessNode,
)

if HAS_RCLPY:
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import Point, PoseStamped
    from nav_msgs.msg import Path


VARIANT_META = {
    "no_checkpoint": {
        "title": "DFE-MPCC (Không Checkpoint SPH)",
        "short_name": "DFE-MPCC (Không Checkpoint)",
        "color_hex": "#E67E22",
        "color_rgba": ColorRGBA(r=0.95, g=0.55, b=0.10, a=0.95),
        "tag": "LƯỢT 1/2",
        "desc": "DFE-MPCC cơ sở (q_eta = 0, không ống vận tốc H124 & không chuẩn hóa WCSPH Checkpoint)",
    },
    "full": {
        "title": "DFE-MPCC (Có Checkpoint SPH)",
        "short_name": "DFE-MPCC (Có Checkpoint SPH)",
        "color_hex": "#117864",
        "color_rgba": ColorRGBA(r=0.08, g=0.82, b=0.48, a=0.95),
        "tag": "LƯỢT 2/2",
        "desc": "Tích hợp Ống vận tốc Krasnoselskii-Mann H124 + Checkpoint WCSPH Hybrid B+A / Conformal (mu = 0.35)",
    },
}


def evaluate_2way_slosh_and_fz_physics(
    dphi_x: float,
    dphi_y: float,
    phi_x: float,
    phi_y: float,
    a_bx: float,
    a_by: float,
    jerk_mag: float,
    a_road_x: float,
    a_road_y: float,
    z_road_mm: float,
    dfz_iso_wheels: np.ndarray,
    solver_info: dict = None,
):
    """
    Tính toán đồng nhất độ dâng sóng mặt thoáng eta (mm), tải pháp tuyến cực tiểu F_z,min (N)
    từ động lực học bồn chất lỏng thực + phản lực 4 bánh Mecanum + mấp mô ISO 8608,
    không phụ thuộc nhãn biến thể.
    """
    _, _, eta_mm = compute_world6_slosh_elevation_mm(
        phi_x, dphi_x, phi_y, dphi_y,
        a_bx, a_by, jerk_mag,
        a_road_bx=a_road_x, a_road_by=a_road_y,
        z_road_mm=z_road_mm,
    )

    m_liquid = 1.20
    F_slosh_x = m_liquid * (a_bx + 9.81 * math.sin(phi_y))
    F_slosh_y = m_liquid * (a_by + 9.81 * math.sin(phi_x))
    Fz_wheels, _ = compute_dynamic_normal_loads(
        m_total=7.25,
        a_body=[a_bx, a_by],
        f_slosh_body=[F_slosh_x, F_slosh_y],
        phi_body=[phi_x, phi_y],
        z_com=0.0555,
        z_slosh=(0.0768, 0.0768),
        fz_min=0.0,
    )
    fz_true_wheels = np.maximum(0.0, Fz_wheels + np.asarray(dfz_iso_wheels, dtype=float))
    fz_min_N = float(np.min(fz_true_wheels))

    slack_norm = 0.0
    if solver_info:
        slack_norm = float(max(solver_info.get("slack_zono_max", 0.0), solver_info.get("slack_tube", 0.0)))

    return float(eta_mm), float(fz_min_N), float(slack_norm)


def get_zone_name_and_macro_mu(x: float, y: float) -> tuple:
    """Trả về tên mặt sàn và hệ số ma sát đồng nhất mu = 0.35 trên world7.xml."""
    return "Sàn Phẳng Chuẩn (Uniform)", get_world7_zone_friction(x, y)


def publish_single_variant_markers(
    ros_node: SimpatRos2RvizHeadlessNode,
    sim_time: float,
    variant_key: str,
    run_idx: int,
    total_runs: int,
    st: dict,
    dyn_geoms: list,
    data_ref,
    scan_ranges: np.ndarray,
    scan_hits: np.ndarray,
    path_length: float,
    cycle: int,
    baseline_traj_xy: list = None,
):
    """
    Xuất bản duy nhất robot đang chạy trong lượt hiện tại lên ROS 2 RViz2, kèm:
      - 180 tia LiDAR độc lập quét chướng ngại vật thời gian thực.
      - Chân trời dự báo N=15 bước và vệt quỹ đạo màu đặc trưng của biến thể.
      - 6 Điểm Checkpoint WCSPH, vùng Sàn Dầu Trơn mu=0.12, Sàn Sơn Mài mu=0.18 (mặt phẳng, không dốc).
      - Bảng HUD thời gian thực nêu rõ tiến độ lượt chạy và trạng thái an toàn vật lý.
    """
    v_info = VARIANT_META[variant_key]
    col_rgba = v_info["color_rgba"]

    # 1. Đồng bộ các topic chuẩn telemetry của robot hiện tại
    status_label = f"LƯỢT {run_idx}/{total_runs}: {v_info['short_name']}"
    ros_node.publish_all_telemetry(
        sim_time=sim_time,
        q_meas=st["q_now"],
        v_meas=st["v_now"],
        phi_x=st["phi_x"],
        phi_y=st["phi_y"],
        eta_mm=st["eta_mm"],
        x_pred=st["x_pred"],
        scan_ranges=scan_ranges,
        scan_hits=scan_hits,
        dyn_geoms=dyn_geoms,
        data=data_ref,
        nearest_dyn_name=st["nearest_dyn"],
        active_min_d=st["active_min_d"],
        s_prog=st["s_prog"],
        path_length=path_length,
        status_text=status_label,
        cycle=cycle,
        robot_color=col_rgba,
        fz_wheels=st.get("fz_wheels"),
        fz_min=st["fz_min_N"],
        f_slosh=st.get("f_slosh"),
        m_roll=st.get("m_roll", 0.0),
        v_ub=st.get("v_ub", 0.0),
        mu_local=st.get("mu_local", 0.35),
    )

    if not HAS_RCLPY:
        return

    now = ros_node.get_clock().now().to_msg()
    ma = MarkerArray()

    # 2. Xóa bỏ các marker cũ/thừa thãi để tránh nhiễu loạn trên RViz
    m_del_body = Marker()
    m_del_body.header.stamp = now
    m_del_body.header.frame_id = "map"
    m_del_body.ns = "active_robot_body"
    m_del_body.action = Marker.DELETEALL
    ma.markers.append(m_del_body)

    m_del_hud = Marker()
    m_del_hud.header.stamp = now
    m_del_hud.header.frame_id = "map"
    m_del_hud.ns = "telemetry_hud_single"
    m_del_hud.action = Marker.DELETEALL
    ma.markers.append(m_del_hud)

    m_del_zones = Marker()
    m_del_zones.header.stamp = now
    m_del_zones.header.frame_id = "map"
    m_del_zones.ns = "special_zones_benchmark"
    m_del_zones.action = Marker.DELETEALL
    ma.markers.append(m_del_zones)

    # 3. Vẽ vệt quỹ đạo thực tế (LINE_STRIP) của robot đang chạy
    if len(st["traj_xy"]) >= 2:
        m_trail = Marker()
        m_trail.header.stamp = now
        m_trail.header.frame_id = "map"
        m_trail.ns = "active_robot_trail"
        m_trail.id = 10
        m_trail.type = Marker.LINE_STRIP
        m_trail.action = Marker.ADD
        m_trail.pose.orientation.w = 1.0
        m_trail.scale.x = 0.038
        m_trail.color = col_rgba
        step_idx = max(1, len(st["traj_xy"]) // 260)
        for pt_xy in st["traj_xy"][::step_idx]:
            m_trail.points.append(Point(x=float(pt_xy[0]), y=float(pt_xy[1]), z=0.022))
        ma.markers.append(m_trail)

    # 4. Ở lượt Có Checkpoint (Lượt 2/2), vẽ chồng vệt Lượt 1 (Không Checkpoint) màu cam mờ để so sánh trực tiếp
    if baseline_traj_xy and len(baseline_traj_xy) >= 2 and variant_key == "full":
        m_base = Marker()
        m_base.header.stamp = now
        m_base.header.frame_id = "map"
        m_base.ns = "baseline_comparison_trail"
        m_base.id = 11
        m_base.type = Marker.LINE_STRIP
        m_base.action = Marker.ADD
        m_base.pose.orientation.w = 1.0
        m_base.scale.x = 0.024
        m_base.color = ColorRGBA(r=0.95, g=0.55, b=0.10, a=0.55)
        step_b = max(1, len(baseline_traj_xy) // 260)
        for pt_xy in baseline_traj_xy[::step_b]:
            m_base.points.append(Point(x=float(pt_xy[0]), y=float(pt_xy[1]), z=0.014))
        ma.markers.append(m_base)

    ros_node.marker_pub.publish(ma)


def publish_intermission_banner(
    ros_node: SimpatRos2RvizHeadlessNode,
    completed_key: str,
    run_idx: int,
    total_runs: int,
    next_key: str,
    res: dict,
    pause_sec: float = 2.5,
):
    """Hiển thị bảng tổng kết ngắn gọn của lượt vừa chạy và thông báo chuẩn bị lượt tiếp theo."""
    if not HAS_RCLPY:
        return
    c_meta = VARIANT_META[completed_key]
    n_meta = VARIANT_META[next_key]
    now = ros_node.get_clock().now().to_msg()

    ma = MarkerArray()
    m_inter = Marker()
    m_inter.header.stamp = now
    m_inter.header.frame_id = "map"
    m_inter.ns = "intermission_hud"
    m_inter.id = 200
    m_inter.type = Marker.TEXT_VIEW_FACING
    m_inter.action = Marker.ADD
    m_inter.pose.position.x = 0.0
    m_inter.pose.position.y = 0.0
    m_inter.pose.position.z = 1.25
    m_inter.scale.z = 0.22
    m_inter.color = ColorRGBA(r=1.0, g=0.92, b=0.20, a=1.0)

    fin_ok = res.get("finished", False)
    min_d = float(res.get("min_dyn", 1.0))
    min_w = float(res.get("min_wall", 1.0))
    if min_d <= 0.0:
        col_eval = f"THẤT BẠI (VA CHẠM ĐỘNG: d={min_d:.2f}m)"
    elif min_w <= 0.0:
        col_eval = f"THẤT BẠI (VA CHẠM TƯỜNG: d={min_w:.2f}m)"
    elif not fin_ok:
        col_eval = f"CHƯA VỀ ĐÍCH (s={res.get('s_prog', 0.0):.2f}/{res.get('path_length', 13.08):.2f}m)"
    else:
        col_eval = "HOÀN THÀNH AN TOÀN TOÀN TUYẾN"

    eta_eval = "THẤT BẠI (TRÀN NƯỚC)" if res["max_eta"] > 30.0 else (
        "CẢNH BÁO (VƯỢT STOKES)" if res["max_eta"] > 20.0 else "ĐẠT CHUẨN AN TOÀN"
    )
    fz_eval = "THẤT BẠI (NHẤC BÁNH)" if res["min_fz"] < 3.0 else "BÁM ĐƯỜNG AN TOÀN"

    m_inter.text = (
        f"===========================================================\n"
        f"  KẾT THÚC {c_meta['tag']}: {c_meta['short_name']} (T = {res['t_finish_s']:.2f} s)\n"
        f"  Kết quả: [{col_eval}]\n"
        f"  max eta = {res['max_eta']:.2f} mm [{eta_eval}] | Fz,min = {res['min_fz']:.2f} N [{fz_eval}]\n"
        f"  Tỷ lệ kích hoạt bù Slack: {100.0 * res['slack_steps'] / max(1, res['total_steps']):.1f}%\n"
        f"-----------------------------------------------------------\n"
        f"  CHUẨN BỊ {n_meta['tag']}: {n_meta['short_name']}...\n"
        f"==========================================================="
    )
    ma.markers.append(m_inter)
    ros_node.marker_pub.publish(ma)

    t0 = time.time()
    while time.time() - t0 < pause_sec:
        rclpy.spin_once(ros_node, timeout_sec=0.05)
        time.sleep(0.05)

    # Xóa banner nghỉ
    m_del = Marker()
    m_del.header.stamp = ros_node.get_clock().now().to_msg()
    m_del.header.frame_id = "map"
    m_del.ns = "intermission_hud"
    m_del.id = 200
    m_del.action = Marker.DELETE
    ma_del = MarkerArray()
    ma_del.markers.append(m_del)
    ros_node.marker_pub.publish(ma_del)


def run_single_variant_simulation(
    variant_key: str,
    run_idx: int,
    total_runs: int,
    scen: dict,
    ros_node: SimpatRos2RvizHeadlessNode,
    lidar: Lidar,
    max_time: float = 54.0,
    speed_factor: float = 1.0,
    rviz_proc = None,
    live_plotter = None,
    tube_planner = None,
    baseline_traj_xy: list = None,
) -> dict:
    """
    Thực thi độc lập 1 lượt chạy hoàn chỉnh cho biến thể variant_key trên bản đồ scen["map_name"]:
      1. Khởi tạo controller và đặt lại robot ở start_pos.
      2. Làm sạch vệt quỹ đạo cũ trên RViz2.
      3. Lặp 20 Hz NMPC + 200 Hz MuJoCo Headless SiL + Quét 180 tia LiDAR độc lập.
      4. Tích hợp mặt sàn đồng nhất mu = 0.35 + mấp mô hình học ISO 8608 Class B (Seed 32).
      5. Ghi nhận đầy đủ chuỗi thời gian quỹ đạo, eta(s), Fz(s), v(s), slack.
    """
    v_info = VARIANT_META[variant_key]
    model = scen["model"]
    wp_arr = scen["wp_arr"]
    checkpoints_6 = scen["checkpoints_6"]
    gate_descriptors = scen["gate_descriptors"]
    dyn_geoms = scen["dyn_geoms"]
    obs_boxes = scen["obs_boxes"]
    obs_circles = scen["obs_circles"]
    start_pos = scen["start_pos"]
    goal_pos = scen["goal_pos"]

    print("\n" + "=" * 108)
    print(f"  BẮT ĐẦU {v_info['tag']} [{scen['map_name'].upper()}]: {v_info['title'].upper()}")
    print(f"  Đặc tính: {v_info['desc']}")
    print(f"  Mặt sàn: Phẳng hoàn toàn, ma sát đồng nhất mu = 0.35 + mấp mô hình học 2D ISO 8608 Class B")
    print("=" * 108)

    # 1. Khởi tạo Controller DFE-MPCC đồng đẳng (h_lim_m=0.020 m = ngưỡng Stokes để P_q_eta siết ECBF & T2)
    xy_bnd = 3.25 if "world8" in scen["map_name"].lower() else 3.0
    ctrl = DFEMPCCAMRController(
        N=15,
        dt=0.05,
        mu=0.35,
        state_con_mode="skip0",
        mu_plan_scale=0.95,
        ipopt_max_iter=60,
        h_lim_m=0.020,
        x_bounds=(-xy_bnd, xy_bnd),
        y_bounds=(-xy_bnd, xy_bnd),
    )
    ctrl.set_path(wp_arr)
    # Biến thể Có Checkpoint: Nhúng Ống vận tốc khả thi bậc 3 vào bộ giải NLP trực tuyến
    if variant_key == "full" and tube_planner is not None and getattr(tube_planner, "is_ready", False):
        ctrl.set_tube_planner(tube_planner)

    # 2. Đặt lại đối tượng vật lý MuJoCo Data độc lập
    d_mj = mujoco.MjData(model)
    mujoco.mj_resetData(model, d_mj)
    update_dynamic_obstacles(model, d_mj, 0.0, custom_dyn_configs=scen["dyn_configs"])
    d_mj.qpos[0] = start_pos[0]
    d_mj.qpos[1] = start_pos[1]
    _, _, h_init = ctrl.get_path_reference(0.0)
    d_mj.qpos[2] = h_init
    mujoco.mj_forward(model, d_mj)

    # 3. Làm sạch vệt quỹ đạo cũ trong node ROS 2
    ros_node.robot_traj_msg.poses.clear()

    # 4. Bộ sinh địa hình mặt đường ngẫu nhiên ISO 8608 Class B (Seed 32)
    road = ISO8608RoadProfile(road_class="B", seed=32)

    init_vs = 0.25

    st = {
        "ctrl": ctrl,
        "data": d_mj,
        "road": road,
        "s_prog": 0.0,
        "vs_prog": init_vs,
        "as_prog": 0.0,
        "a_act": np.zeros(3),
        "dashing_gate": -1,
        "finished": False,
        "t_finish_s": max_time,
        "q_now": d_mj.qpos[:3].copy(),
        "v_now": np.zeros(3),
        "phi_x": 0.0,
        "phi_y": 0.0,
        "eta_mm": 0.0,
        "fz_min_N": 16.0,
        "max_eta": 0.0,
        "min_fz": 16.0,
        "max_ec": 0.0,
        "min_wall": 10.0,
        "min_dyn": 10.0,
        "slack_steps": 0,
        "total_steps": 0,
        "max_slack": 0.0,
        "split_mu_max": 0.0,
        "mu_local": 0.35,
        "split_mu": 0.0,
        "solve_times": [],
        "traj_xy": [start_pos[:2].copy()],
        "s_hist": [],
        "eta_hist": [],
        "fz_hist": [],
        "v_hist": [],
        "th_x_hist": [],
        "th_y_hist": [],
        "f_slosh_x_hist": [],
        "f_slosh_y_hist": [],
        "m_roll_hist": [],
        "fz_wheels_hist": [],
        "x_pred": np.zeros((12, 16)),
        "nearest_dyn": "None",
        "active_min_d": 10.0,
    }

    nmpc_dt = 0.05
    max_steps = int(max_time / nmpc_dt)
    sim_time = 0.0
    far_obs = np.array([20.0, 20.0, 0.0, 0.1, 0.1])

    print(f"{'Thời gian':>8} | {'Tiến độ s':>10} | {'Vận tốc v':>10} | {'eta (mm)':>10} | {'Fz,min (N)':>11} | {'Mặt sàn & mu':<28} | {'Trạng thái':<20}")
    print("-" * 108)

    for cycle in range(max_steps):
        if rviz_proc is not None and rviz_proc.poll() is not None:
            print("\n  [!] Người dùng đã đóng cửa sổ RViz2. Kết thúc lượt chạy.")
            break

        loop_t0 = time.perf_counter()

        # Cập nhật chuyển động 4 vật cản động theo thời gian thực
        update_dynamic_obstacles(model, d_mj, sim_time, custom_dyn_configs=scen["dyn_configs"])
        mujoco.mj_forward(model, d_mj)

        if st["finished"]:
            break

        st["total_steps"] += 1
        q_meas = d_mj.qpos[:3].copy()
        v_meas = d_mj.qvel[:3].copy()
        phi_x_meas = float(d_mj.qpos[3])
        dphi_x_meas = float(d_mj.qvel[3])
        phi_y_meas = float(d_mj.qpos[4])
        dphi_y_meas = float(d_mj.qvel[4])

        # Đánh giá cổng vật cản động (Gate Crossing) dùng helper dùng chung
        should_wait, active_gate, dashing_new, is_tight_gate = evaluate_gate_crossing(
            gate_descriptors=gate_descriptors,
            s_prog=st["s_prog"],
            v_curr=float(np.hypot(v_meas[0], v_meas[1])),
            sim_time=sim_time,
            dashing_gate=st["dashing_gate"],
            ctrl=ctrl,
            tube_planner=tube_planner,
            use_tube_speed=(variant_key == "full"),
            v_dash_fallback=0.40,
            v_dash_bounds=(0.32, 0.42),
            delta_s_hi=0.58,
            base_clear_thresh=0.06,
            accel_dash=0.80,
            tau_extra=1.0,
            tau_floor=3.5,
            n_samples=45,
            s_win_back=0.38,
            s_win_ahead=0.50,
            s_end_margin=0.55,
        )
        st["dashing_gate"] = dashing_new

        d_cp_min = min([abs(st["s_prog"] - cp["s_m"]) for cp in checkpoints_6]) if checkpoints_6 else 2.0
        cp_factor = math.exp(-0.5 * (d_cp_min / 0.42) ** 2)

        # Đánh giá kích động hình học mặt đường ISO 8608 Class B với ma sát đồng nhất mu = 0.35
        road_eval = road.compute_wheel_loads_and_slosh_excitation(
            q_meas, v_meas,
            mu_nominal=0.35,
            friction_variation_scale=0.0,
        )
        dfz_iso_now = road_eval["dfz_iso_wheels"]
        a_road_body = road_eval["a_road_body"]
        z_road_mean_mm = float(np.mean(np.abs(road_eval["z_wheels_mm"])))
        mu_local = 0.35
        split_mu = 0.0

        st["mu_local"] = mu_local
        st["split_mu"] = split_mu
        st["split_mu_max"] = 0.0

        # Bộ ước lượng trạng thái 3D & ma sát đồng nhất giữa hai cấu hình
        phi_est = np.array([
            phi_x_meas + (a_road_body[0] / 9.81) * 0.55,
            dphi_x_meas,
            phi_y_meas - (a_road_body[1] / 9.81) * 0.55,
            dphi_y_meas,
        ])
        mu_solve = 0.35

        # Chỉ khác biệt ở việc kích hoạt Checkpoint WCSPH (lượng tử q_eta + Ống vận tốc H124)
        if variant_key == "no_checkpoint":
            q_eta_use = 0.0
            v_ref_cmd = 0.65
            v_ub_h = np.full((1, ctrl.N), 0.65)
            v_lb_h = np.zeros((1, ctrl.N))
        else:
            q_eta_use = float(_CAL_BUDGET["recommended_q_eta_m"])
            if tube_planner is not None and getattr(tube_planner, "is_ready", False):
                s_preview = np.clip(st["s_prog"] + np.arange(ctrl.N) * nmpc_dt * max(st["vs_prog"], 0.35), 0.0, ctrl.path_length)
                v_ub_nom_q = np.interp(s_preview, tube_planner.s_grid, tube_planner.v_ub_nom)
                v_ub_online, _ = tube_planner.query_online_horizon_tube(
                    s_horizon=s_preview,
                    phi_est=phi_est,
                    mu_online=0.35,
                )
                v_ub_eff = np.minimum(v_ub_nom_q, v_ub_online)
                v_ub_h = np.clip(v_ub_eff, 0.25, 0.65).reshape(1, ctrl.N)
                v_lb_h = np.zeros((1, ctrl.N))
                v_ref_cmd = float(v_ub_h[0, 0])
            else:
                v_ref_cmd = 0.46 if cp_factor > 0.60 else 0.58
                v_ub_h = np.full((1, ctrl.N), v_ref_cmd)
                v_lb_h = np.zeros((1, ctrl.N))

            if st["dashing_gate"] >= 0:
                v_ref_cmd = max(v_ref_cmd, 0.36)

        x_curr = np.zeros(12)
        x_curr[0:3] = q_meas
        x_curr[3:6] = v_meas
        x_curr[6:9] = st["a_act"]
        x_curr[9] = st["s_prog"]
        x_curr[10] = 0.02 if should_wait else (max(st["vs_prog"], 0.30) if st["dashing_gate"] >= 0 else st["vs_prog"])
        x_curr[11] = st["as_prog"]

        t_sol0 = time.perf_counter()
        sol_info = {}
        if should_wait:
            d_mj.qvel[:3] = 0.0
            st["a_act"][:] = 0.0
            st["vs_prog"] = 0.0
            st["as_prog"] = 0.0
            if active_gate >= 0:
                st["s_prog"] = min(st["s_prog"], gate_descriptors[active_gate]["s_obs"] - 0.46)
            x_pred = np.tile(x_curr.reshape(-1, 1), (1, ctrl.N + 1))
            jerk_val = 0.0
            sol_ms = 0.5
        else:
            obs_for_ctrl = select_tight_gate_obstacle(dyn_geoms, d_mj, q_meas, far_obs) if is_tight_gate else far_obs
            u_opt, _, x_pred, _, sol_info = ctrl.solve_step(
                x_curr, phi_est, obs_for_ctrl,
                mu_floor=mu_solve,
                q_eta=q_eta_use,
                dfz_iso=dfz_iso_now,
                v_ub_horizon=v_ub_h,
                v_lb_horizon=v_lb_h,
            )
            sol_ms = (time.perf_counter() - t_sol0) * 1000.0
            st["s_prog"] = float(x_pred[9, 1])
            st["vs_prog"] = float(x_pred[10, 1])
            st["as_prog"] = float(x_pred[11, 1])
            st["a_act"] = np.array(x_pred[6:9, 1])
            jerk_val = float(np.linalg.norm(u_opt[:2]))

            if st["dashing_gate"] >= 0:
                _, t_dir, _ = ctrl.get_path_reference(st["s_prog"])
                target_v = min(v_ref_cmd, 0.40)
                d_mj.qvel[0] = v_meas[0] + (target_v * t_dir[0] - v_meas[0]) * 0.45
                d_mj.qvel[1] = v_meas[1] + (target_v * t_dir[1] - v_meas[1]) * 0.45
                d_mj.qvel[2] = v_meas[2] + st["a_act"][2] * nmpc_dt
                st["vs_prog"] = float(np.hypot(d_mj.qvel[0], d_mj.qvel[1]))
            else:
                d_mj.qvel[0] = v_meas[0] + st["a_act"][0] * nmpc_dt
                d_mj.qvel[1] = v_meas[1] + st["a_act"][1] * nmpc_dt
                d_mj.qvel[2] = v_meas[2] + st["a_act"][2] * nmpc_dt

        st["solve_times"].append(sol_ms)
        st["x_pred"] = x_pred

        c_th0, s_th0 = np.cos(q_meas[2]), np.sin(q_meas[2])
        a_bx0 = c_th0 * st["a_act"][0] + s_th0 * st["a_act"][1]
        a_by0 = -s_th0 * st["a_act"][0] + c_th0 * st["a_act"][1]

        d_mj.qpos[3] = np.clip(d_mj.qpos[3] + ((a_by0 + a_road_body[1]) / 9.81) * 0.08, -0.25, 0.25)
        d_mj.qpos[4] = np.clip(d_mj.qpos[4] - ((a_bx0 + a_road_body[0]) / 9.81) * 0.08, -0.25, 0.25)

        for _ in range(10):
            mujoco.mj_step(model, d_mj)

        q_now = d_mj.qpos[:3].copy()
        v_now = d_mj.qvel[:3].copy()

        # Tọa độ thực tế thuần túy từ động học mô phỏng MuJoCo
        q_vis = q_now.copy()

        st["q_now"] = q_vis
        st["v_now"] = v_now
        st["phi_x"] = float(d_mj.qpos[3])
        st["phi_y"] = float(d_mj.qpos[4])
        dphi_x_now = float(d_mj.qvel[3])
        dphi_y_now = float(d_mj.qvel[4])
        st["traj_xy"].append(q_vis[:2].copy())

        wc_now = compute_static_clearance(q_vis[0], q_vis[1], obs_boxes, obs_circles, r_robot=0.18)
        if wc_now < st["min_wall"]:
            st["min_wall"] = wc_now

        nearest_dyn_name = "None"
        active_min_d = 10.0
        for idx_d, (g_id, r_obs, _, _, obs_name) in enumerate(dyn_geoms):
            g_pos = d_mj.geom_xpos[g_id][:2].copy()
            d_clear = float(np.hypot(q_vis[0] - g_pos[0], q_vis[1] - g_pos[1])) - (r_obs + 0.20)
            if d_clear < st["min_dyn"]:
                st["min_dyn"] = d_clear
            if d_clear < active_min_d:
                active_min_d = d_clear
                nearest_dyn_name = obs_name

        st["nearest_dyn"] = nearest_dyn_name
        st["active_min_d"] = active_min_d

        v_lin_now = min(0.70, float(np.hypot(v_now[0], v_now[1])))
        eta_mm, fz_min_N, slack_norm = evaluate_2way_slosh_and_fz_physics(
            dphi_x=dphi_x_now,
            dphi_y=dphi_y_now,
            phi_x=st["phi_x"],
            phi_y=st["phi_y"],
            a_bx=a_bx0,
            a_by=a_by0,
            jerk_mag=jerk_val,
            a_road_x=a_road_body[0],
            a_road_y=a_road_body[1],
            z_road_mm=z_road_mean_mm,
            dfz_iso_wheels=dfz_iso_now,
            solver_info=sol_info,
        )

        st["eta_mm"] = eta_mm
        st["fz_min_N"] = fz_min_N
        if eta_mm > st["max_eta"]:
            st["max_eta"] = eta_mm
        if fz_min_N < st["min_fz"]:
            st["min_fz"] = fz_min_N
        if slack_norm > 1e-4:
            st["slack_steps"] += 1
            if slack_norm > st["max_slack"]:
                st["max_slack"] = slack_norm

        d_to_wp = np.hypot(wp_arr[:, 0] - q_vis[0], wp_arr[:, 1] - q_vis[1])
        ec_now = float(np.min(d_to_wp))
        if ec_now > st["max_ec"]:
            st["max_ec"] = ec_now

        st["s_hist"].append(st["s_prog"])
        st["eta_hist"].append(eta_mm)
        st["fz_hist"].append(fz_min_N)
        st["v_hist"].append(v_lin_now)

        dist_to_goal = float(np.hypot(q_now[0] - goal_pos[0], q_now[1] - goal_pos[1]))
        if dist_to_goal < 0.25 and st["s_prog"] > ctrl.path_length - 0.40:
            st["finished"] = True
            st["t_finish_s"] = sim_time + nmpc_dt
            print(f"\n  >>> [{v_info['short_name']}] ĐÃ VỀ ĐÍCH THÀNH CÔNG TẠI t = {st['t_finish_s']:.2f} s!")

        sim_time += nmpc_dt

        # Quét 180 tia LiDAR độc lập của robot này
        _, _, meas, hit, _ = lidar.scan(model, d_mj)

        # Tính toán phản lực sóng sánh và tải bánh xe cho ROS 2 PlotJuggler Telemetry
        th_x_deg = math.degrees(float(st["phi_x"]) + 0.35 * (a_by0 / 9.81))
        th_y_deg = math.degrees(float(st["phi_y"]) - 0.35 * (a_bx0 / 9.81))
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
        st["fz_wheels"] = Fz_wheels
        st["f_slosh"] = [F_slosh_x, F_slosh_y]
        st["m_roll"] = M_roll
        st["v_ub"] = v_ref_cmd
        st["mu_local"] = mu_local
        st["th_x_hist"].append(th_x_deg)
        st["th_y_hist"].append(th_y_deg)
        st["f_slosh_x_hist"].append(F_slosh_x)
        st["f_slosh_y_hist"].append(F_slosh_y)
        st["m_roll_hist"].append(M_roll)
        st["fz_wheels_hist"].append(Fz_wheels.copy())

        if HAS_RCLPY and rclpy is not None:
            publish_single_variant_markers(
                ros_node=ros_node,
                sim_time=sim_time,
                variant_key=variant_key,
                run_idx=run_idx,
                total_runs=total_runs,
                st=st,
                dyn_geoms=dyn_geoms,
                data_ref=d_mj,
                scan_ranges=meas,
                scan_hits=hit,
                path_length=ctrl.path_length,
                cycle=cycle,
                baseline_traj_xy=baseline_traj_xy,
            )
            rclpy.spin_once(ros_node, timeout_sec=0.002)

        if cycle % 10 == 0:
            z_n, _ = get_zone_name_and_macro_mu(q_vis[0], q_vis[1])
            cond_str = (
                "TRÀN NƯỚC!" if eta_mm > 30.0 else (
                    "VƯỢT STOKES" if eta_mm > 20.0 else (
                        "NHẤC BÁNH!" if fz_min_N < 3.0 else "Ổn định"
                    )
                )
            )
            print(
                f"{sim_time:7.2f}s | {st['s_prog']:8.2f} m | {v_lin_now:8.2f}m/s | "
                f"{eta_mm:8.2f}mm | {fz_min_N:9.2f} N | {z_n:<22} (mu={mu_local:.2f}) | {cond_str:<18}",
                flush=True,
            )

        if HAS_RCLPY and speed_factor <= 2.0:
            elapsed = time.perf_counter() - loop_t0
            target_dt = nmpc_dt / max(speed_factor, 0.1)
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

    print("-" * 108)
    p95_eta = float(np.percentile(st["eta_hist"], 95)) if st["eta_hist"] else 0.0
    slk_pct = 100.0 * st["slack_steps"] / max(1, st["total_steps"])
    print(f"  TỔNG KẾT {v_info['tag']}: {v_info['title']}")
    print(f"    - Thời gian về đích: T_finish = {st['t_finish_s']:.2f} s")
    print(f"    - Sóng sánh cực đại : max_eta = {st['max_eta']:.2f} mm | P95 = {p95_eta:.2f} mm")
    print(f"    - Lực pháp tuyến min: min_Fz  = {st['min_fz']:.2f} N (Ngưỡng chống nhấc bánh >= 3.0 N)")
    print(f"    - Tỷ lệ kích hoạt bù: Slack   = {slk_pct:.1f}% ({st['slack_steps']}/{st['total_steps']} bước)")
    print(f"    - Sai số bám viền max: |ec|   = {st['max_ec']:.3f} m")
    print(f"    - Chênh ma sát Split-Mu max: Delta_mu = {st['split_mu_max']:.3f}")
    print("=" * 108)

    res_clean = {
        "variant_key": variant_key,
        "traj_xy": [np.asarray(pt).tolist() for pt in st["traj_xy"]],
        "s_hist": list(st["s_hist"]),
        "eta_hist": list(st["eta_hist"]),
        "fz_hist": list(st["fz_hist"]),
        "v_hist": list(st["v_hist"]),
        "th_x_hist": list(st.get("th_x_hist", [])),
        "th_y_hist": list(st.get("th_y_hist", [])),
        "f_slosh_x_hist": list(st.get("f_slosh_x_hist", [])),
        "f_slosh_y_hist": list(st.get("f_slosh_y_hist", [])),
        "m_roll_hist": list(st.get("m_roll_hist", [])),
        "max_eta": float(st["max_eta"]),
        "min_fz": float(st["min_fz"]),
        "min_dyn": float(st["min_dyn"]),
        "min_wall": float(st["min_wall"]),
        "finished": bool(st.get("finished", False)),
        "s_prog": float(st["s_prog"]),
        "path_length": float(ctrl.path_length),
        "max_ec": float(st["max_ec"]),
        "slack_steps": int(st["slack_steps"]),
        "total_steps": int(st["total_steps"]),
        "t_finish_s": float(st["t_finish_s"]),
        "split_mu_max": float(st["split_mu_max"]),
        "solve_times": list(st["solve_times"]),
    }
    return res_clean


def generate_3way_mpcc_comparison_figure(
    scen: dict,
    records: dict,
    out_png_path: str,
    show_plot: bool = True,
    tube_planner=None,
):
    """
    Xuất đồ thị 4-panel so sánh trực quan 2 cấu hình DFE-MPCC trên bản đồ world7.xml
    (mặt sàn phẳng hoàn toàn, hệ số ma sát đồng nhất mu = 0.35 + mấp mô hình học ISO 8608 Class B):
      - Panel (a): Bản đồ 2D world7.xml + 6 Checkpoint WCSPH + quỹ đạo 2 cấu hình
      - Panel (b): Hồ sơ độ dâng sóng mặt thoáng eta(s) qua 6 Checkpoint
      - Panel (c): Hồ sơ tải pháp tuyến bánh xe nhỏ nhất F_z,min(s) so với ngưỡng nhấc bánh (3.00 N)
      - Panel (d): Hồ sơ vận tốc dọc v(s): Ống vận tốc H124 + 6 Checkpoint vs. Không Checkpoint
    """
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(13.8, 11.2))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.10, 1.38], height_ratios=[1.0, 1.0, 1.0], hspace=0.36, wspace=0.22)

    ax_map = fig.add_subplot(gs[:, 0])
    ax_eta = fig.add_subplot(gs[0, 1])
    ax_fz = fig.add_subplot(gs[1, 1], sharex=ax_eta)
    ax_vel = fig.add_subplot(gs[2, 1], sharex=ax_eta)

    # --- Panel (a): Bản đồ 2D world7.xml và 2 quỹ đạo ---
    for (cx, cy, hx, hy) in scen["obs_boxes"]:
        rect = plt.Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy, facecolor="#5D6D7E", edgecolor="#2C3E50", lw=1.0, zorder=3)
        ax_map.add_patch(rect)

    wp = scen["wp_arr"]
    ax_map.plot(wp[:, 0], wp[:, 1], ":", color="#ABB2B9", lw=1.5, label="Tâm quy hoạch A* ($C^2$ B-Spline)", zorder=3)

    n_cp = len(scen["checkpoints_6"])
    plot_keys = ["no_checkpoint", "full"]
    colors = {"no_checkpoint": "#E67E22", "full": "#117864"}
    labels = {
        "no_checkpoint": r"DFE-MPCC (Không Checkpoint, $q_\eta = 0$)",
        "full": rf"DFE-MPCC (Có {n_cp} Checkpoint SPH, $q_\eta = 3.70$ mm)",
    }
    lstyles = {"no_checkpoint": "--", "full": "-"}
    lwidths = {"no_checkpoint": 2.0, "full": 2.4}

    for k in plot_keys:
        if k in records and len(records[k]["traj_xy"]) > 1:
            tr = np.asarray(records[k]["traj_xy"])
            ax_map.plot(tr[:, 0], tr[:, 1], lstyles[k], color=colors[k], lw=lwidths[k], label=labels[k], zorder=5 if k == "full" else 4)

    map_offsets = [(4, 5), (-22, 6), (5, -12), (-24, -12)]
    for idx_cp, cp in enumerate(scen["checkpoints_6"]):
        cpx, cpy = cp["pos"]
        ox, oy = map_offsets[idx_cp % len(map_offsets)]
        ax_map.plot(cpx, cpy, "o", color="#F1C40F", markeredgecolor="#1C2833", markersize=7.0, zorder=7)
        ax_map.annotate(
            cp["name"], xy=(cpx, cpy), xytext=(ox, oy), textcoords="offset points",
            fontsize=7.6, fontweight="bold", color="#1C2833",
            bbox=dict(boxstyle="round,pad=0.16", fc="#FEF9E7", ec="#B7950B", alpha=0.92, lw=0.7), zorder=8
        )

    for g_idx, g_d in enumerate(scen["gate_descriptors"]):
        bp = g_d["base_pos"]
        ax_map.plot(bp[0], bp[1], "D", color="#C0392B", markersize=6.8, zorder=7)
        ax_map.annotate(f"D{g_idx+1}", xy=(bp[0], bp[1]), xytext=(-10, -13), textcoords="offset points", fontsize=7.8, fontweight="bold", color="#922B21")

    s0 = scen["start_pos"]
    g0 = scen["goal_pos"]
    base_map = scen.get("map_name", "world7.xml")
    lim_xy = 3.40 if "world8" in base_map.lower() else 3.15
    ax_map.plot(s0[0], s0[1], "s", color="#2980B9", markersize=9, label=f"Start ({s0[0]:+.2f}, {s0[1]:+.2f})", zorder=8)
    ax_map.plot(g0[0], g0[1], "*", color="#27AE60", markersize=12, label=f"Goal ({g0[0]:+.2f}, {g0[1]:+.2f})", zorder=8)
    ax_map.set_xlim(-lim_xy, lim_xy)
    ax_map.set_ylim(-lim_xy, lim_xy)
    ax_map.set_aspect("equal")
    ax_map.grid(True, linestyle=":", alpha=0.5)
    ax_map.set_xlabel("Tọa độ X [m]", fontsize=9.5, fontweight="bold")
    ax_map.set_ylabel("Tọa độ Y [m]", fontsize=9.5, fontweight="bold")
    ax_map.set_title(rf"(a) Bản đồ {base_map} (Sàn phẳng, $\mu = 0.35$ đồng nhất) & Quỹ đạo", fontsize=9.8, fontweight="bold")
    ax_map.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=7.2, framealpha=0.95)

    # --- Panel (b): Độ dâng sóng sánh mặt thoáng eta(s) ---
    max_eta_plot = 32.0
    for k in plot_keys:
        if k in records and len(records[k]["s_hist"]) > 0:
            s_arr = np.asarray(records[k]["s_hist"])
            eta_arr = np.asarray(records[k]["eta_hist"])
            max_eta_plot = max(max_eta_plot, float(np.max(eta_arr)) + 3.5)
            ax_eta.plot(s_arr, eta_arr, lstyles[k], color=colors[k], lw=lwidths[k], label=f"{labels[k]} (max={np.max(eta_arr):.2f} mm)")

    ax_eta.axhline(30.00, color="#922B21", linestyle="-", lw=1.5, label="Ngưỡng tràn thành bể $h_{\\mathrm{free}} = 30.00$ mm")
    eta_bnd_cl = float(_CAL_BUDGET["eta_bound_cl_mm"])
    ax_eta.axhline(eta_bnd_cl, color="#8E44AD", linestyle="--", lw=1.35, label=f"Bao trên Conformal $\\eta_{{\\mathrm{{bound}}}}^{{\\mathrm{{CL}}}} = {eta_bnd_cl:.2f}$ mm")
    ax_eta.axhline(20.00, color="#566573", linestyle=":", lw=1.1, label="Ngưỡng sóng tuyến tính Stokes (20.00 mm)")
    for idx_cp, cp in enumerate(scen["checkpoints_6"]):
        ax_eta.axvline(cp["s_m"], color="#BDC3C7", linestyle=":", lw=0.9)
        y_frac = 0.94 - 0.08 * (idx_cp % 2)
        ax_eta.text(cp["s_m"], max_eta_plot * y_frac, cp["name"], ha="center", va="top", fontsize=7.2, fontweight="bold", color="#34495E")

    ax_eta.set_ylabel("Độ dâng sóng $\\eta(s)$ [mm]", fontsize=9.2, fontweight="bold")
    ax_eta.set_ylim(0.0, max_eta_plot)
    ax_eta.grid(True, linestyle=":", alpha=0.6)
    ax_eta.legend(loc="upper right", fontsize=6.6, ncol=2, framealpha=0.94)
    ax_eta.set_title(rf"(b) Đáp ứng độ dâng sóng mặt thoáng $\eta(s)$ qua {n_cp} Checkpoint WCSPH ($\mu = 0.35$)", fontsize=9.5, fontweight="bold", loc="left")

    # --- Panel (c): Tải pháp tuyến bánh nhỏ nhất F_z,min(s) ---
    min_fz_plot = 2.0
    max_fz_plot = 16.5
    for k in plot_keys:
        if k in records and len(records[k]["s_hist"]) > 0:
            s_arr = np.asarray(records[k]["s_hist"])
            fz_arr = np.asarray(records[k]["fz_hist"])
            min_fz_plot = min(min_fz_plot, float(np.min(fz_arr)) - 1.0)
            max_fz_plot = max(max_fz_plot, float(np.max(fz_arr)) + 1.0)
            ax_fz.plot(s_arr, fz_arr, lstyles[k], color=colors[k], lw=lwidths[k], label=f"{labels[k]} (min={np.min(fz_arr):.2f} N)")

    ax_fz.axhline(3.00, color="#C0392B", linestyle="-", lw=1.5, label="Ngưỡng nhấc bánh nguy hiểm $F_{z,\\min}^{\\mathrm{safe}} = 3.00$ N")
    ax_fz.axhline(3.50, color="#D35400", linestyle="--", lw=1.25, label="Ngưỡng quy hoạch bảo toàn $F_{z,\\min}^{\\mathrm{plan}} = 3.50$ N")
    for cp in scen["checkpoints_6"]:
        ax_fz.axvline(cp["s_m"], color="#BDC3C7", linestyle=":", lw=0.9)

    ax_fz.set_ylabel("Tải pháp tuyến $F_{z,\\min}$ [N]", fontsize=9.2, fontweight="bold")
    ax_fz.set_ylim(max(0.0, min_fz_plot), max_fz_plot)
    ax_fz.grid(True, linestyle=":", alpha=0.6)
    ax_fz.legend(loc="lower left", fontsize=6.8, ncol=2, framealpha=0.93)
    ax_fz.set_title(r"(c) Tải pháp tuyến cực tiểu $F_{z,\min}(s)$ ($\mu = 0.35$ đồng nhất + ISO 8608 Class B)", fontsize=9.4, fontweight="bold", loc="left")

    # --- Panel (d): Hồ sơ vận tốc thực tế v(s) và thời gian hoàn thành ---
    for k in plot_keys:
        if k in records and len(records[k]["s_hist"]) > 0:
            s_arr = np.asarray(records[k]["s_hist"])
            v_arr = np.asarray(records[k]["v_hist"])
            t_fin = records[k]["t_finish_s"]
            ax_vel.plot(s_arr, v_arr, lstyles[k], color=colors[k], lw=lwidths[k], label=f"{labels[k]} ($T_{{\\mathrm{{finish}}}} = {t_fin:.2f}$ s)")

    if (
        tube_planner is not None
        and getattr(tube_planner, "is_ready", False)
        and hasattr(tube_planner, "s_grid")
        and hasattr(tube_planner, "v_ub_nom")
    ):
        ax_vel.plot(
            tube_planner.s_grid,
            np.clip(tube_planner.v_ub_nom, 0.25, 0.65),
            ":",
            color="#2C3E50",
            lw=1.4,
            label=r"Trần ống vận tốc KM+CP $v_{\mathrm{ub}}(s)$",
        )

    for cp in scen["checkpoints_6"]:
        ax_vel.axvline(cp["s_m"], color="#BDC3C7", linestyle=":", lw=0.9)

    ax_vel.set_xlabel("Chiều dài cung đường quy chiếu $s$ [m]", fontsize=9.5, fontweight="bold")
    ax_vel.set_ylabel("Vận tốc $v(s)$ [m/s]", fontsize=9.2, fontweight="bold")
    ax_vel.set_ylim(-0.02, 0.78)
    ax_vel.grid(True, linestyle=":", alpha=0.6)
    ax_vel.legend(loc="lower right", fontsize=6.8, framealpha=0.93)
    ax_vel.set_title(f"(d) Hồ sơ vận tốc $v(s)$: Ống KM + {n_cp} Checkpoint WCSPH vs. Không Checkpoint", fontsize=9.4, fontweight="bold", loc="left")

    fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.08, hspace=0.36, wspace=0.22)
    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    plt.savefig(out_png_path, dpi=300, bbox_inches="tight")
    print(f"  [✓] Đã xuất đồ thị đối chứng 4-panel ra: {out_png_path}")

    env = os.environ.copy()
    has_display = bool(env.get("DISPLAY"))
    if show_plot and has_display:
        print("  [i] Đang hiển thị cửa sổ đồ thị đối chứng 4-panel sau chạy (đóng cửa sổ đồ thị để kết thúc)...")
        try:
            plt.show()
        except Exception as e:
            print(f"  [!] Không thể hiển thị cửa sổ GUI đồ thị: {e}")
    else:
        plt.close(fig)


def _normalize_map_list(map_arg: str, all_maps: bool = False) -> list:
    """Chuẩn hóa danh sách bản đồ từ --map hoặc --all-maps (mặc định 3 map: world6, world7, world8)."""
    default_three = ["world6.xml", "world7.xml", "world8.xml"]
    if all_maps:
        return default_three
    raw = (map_arg or "world7.xml").strip()
    if raw.lower() in ("all", "3maps", "all3"):
        return default_three
    items = []
    for tok in raw.split(","):
        t = tok.strip().lower()
        if not t:
            continue
        if not t.endswith(".xml"):
            t = f"{t}.xml"
        items.append(t)
    return items or ["world7.xml"]


def _resolve_map_artifacts(base_name: str) -> dict:
    """
    Trả về đường dẫn file artifact theo từng bản đồ:
      - Giữ nguyên benchmark_world7_records.json / hinh_so_sanh_3_mpcc_world7.png cho world7.xml.
      - Dùng ros2_benchmark_<map>_records.json cho world6/world8 để không ghi đè dữ liệu benchmark offline.
    """
    map_stem = os.path.splitext(os.path.basename(base_name).lower())[0]
    out_dir = os.path.join(PROJECT_DIR, "bao cao")
    cp_json_path = os.path.join(out_dir, f"checkpoints_{map_stem}.json")
    if map_stem == "world7":
        cache_path = os.path.join(out_dir, "benchmark_world7_records.json")
        out_png = os.path.join(out_dir, "hinh_so_sanh_3_mpcc_world7.png")
    else:
        cache_path = os.path.join(out_dir, f"ros2_benchmark_{map_stem}_records.json")
        out_png = os.path.join(out_dir, f"hinh_so_sanh_ros2_mpcc_{map_stem}.png")
    return {
        "map_stem": map_stem,
        "cp_json_path": cp_json_path,
        "cache_path": cache_path,
        "out_png": out_png,
    }


def _build_wslg_rviz_env() -> dict:
    """Cấu hình biến môi trường X11/OpenGL ổn định cho RViz2 trên WSL2 / WSLg."""
    env = os.environ.copy()
    if not env.get("DISPLAY"):
        env["DISPLAY"] = ":0"
    env.pop("WAYLAND_DISPLAY", None)
    env["LIBGL_ALWAYS_SOFTWARE"] = "1"
    env["QT_QPA_PLATFORM"] = "xcb"
    return env


def _load_or_build_tube_and_checkpoints(
    scen: dict,
    ros_node,
    cp_json_path: str,
    recompute_wcsph: bool = False,
):
    """
    Nạp Checkpoint WCSPH Hybrid B+A từ file JSON có sẵn (<0.1 s để RViz2 chạy tức thì)
    hoặc tính lại 200-step WCSPH khi chưa có cache / người dùng truyền --recompute-wcsph.
    Đồng thời dựng CoupledVelocityTubeH124(mu_default=0.35, kappa_th=1.05) cho lượt Có Checkpoint.
    """
    wp_arr = scen["wp_arr"]
    path_s = scen["path_s"]
    base_name = scen["map_name"]

    has_cached_cp = False
    cached_cps = []
    if not recompute_wcsph and os.path.exists(cp_json_path):
        try:
            with open(cp_json_path, "r", encoding="utf-8") as f_cp:
                cp_data = json.load(f_cp)
            raw_cps = cp_data.get("checkpoints", [])
            if raw_cps:
                for idx_c, cp_item in enumerate(raw_cps):
                    cp_copy = dict(cp_item)
                    cp_copy.setdefault("name", f"CP{idx_c + 1}")
                    cp_copy.setdefault("desc", "checkpoint")
                    cp_copy["R_mm"] = float(cp_copy.get("R_mm", cp_copy.get("residual_R_mm", 0.0)))
                    cached_cps.append(cp_copy)
                has_cached_cp = True
        except Exception:
            cached_cps = []
            has_cached_cp = False

    run_wcsph_now = bool(recompute_wcsph or not has_cached_cp)
    tube_planner = None
    try:
        from velocity_tube_h124 import CoupledVelocityTubeH124
        mode_tag = "TÍNH LẠI WCSPH HYBRID B+A" if run_wcsph_now else "NẠP NHANH CACHE CHECKPOINT + ỐNG H124"
        print(f"\n>>> [GIAI ĐOẠN A - t=0]: {mode_tag} ({base_name}, MU=0.35)...")
        n_wps = len(wp_arr)
        mu_wps = np.full(n_wps, 0.35, dtype=float)
        camber_wps = np.zeros(n_wps, dtype=float)
        zc_wps = np.full(n_wps, 0.0555, dtype=float)
        d_net_wps = np.full(n_wps, 0.42, dtype=float)

        tube_planner = CoupledVelocityTubeH124(mu_default=0.35, kappa_th=1.05)
        tube_summary = tube_planner.build_offline_tube(
            wp_arr,
            mu_profile=lambda s_val: float(np.interp(s_val, path_s, mu_wps)),
            phi_camber_profile=lambda s_val: float(np.interp(s_val, path_s, camber_wps)),
            z_com_profile=lambda s_val: float(np.interp(s_val, path_s, zc_wps)),
            d_net_profile=lambda s_val: float(np.interp(s_val, path_s, d_net_wps)),
            run_wcsph_check=run_wcsph_now,
        )
        print(f"  [✓] Ống vận tốc đã hội tụ sau {tube_summary['n_fp_iters']} vòng lặp! Phần dư ||T(v)-v||_inf = {tube_summary['fp_unrelaxed_residual']:.6f} m/s")
        print(f"  [✓] Chiều dài quỹ đạo chuẩn L: {tube_summary['path_length_m']:.2f} m ({tube_summary['n_grid']} điểm lưới, ds = {tube_summary['ds_m']:.3f} m)")
        print(f"  [✓] Biên trần vận tốc khả thi: v_ub in [{tube_planner.v_ub_nom.min():.2f}, {tube_planner.v_ub_nom.max():.2f}] m/s")

        if has_cached_cp and not run_wcsph_now:
            cp_list = cached_cps
            print(f"  [✓] Đã nạp {len(cp_list)} Checkpoint WCSPH từ cache: {os.path.basename(cp_json_path)} (bỏ qua 400s chạy lặp lại; dùng --recompute-wcsph để tính lại).")
        else:
            scale_s = float(path_s[-1]) / max(float(tube_planner.s_grid[-1]), 1e-6)
            cp_list = []
            for cp in tube_planner.checkpoints:
                idx_p = int(cp["idx_peak"])
                s_mapped = float(cp["s_peak"]) * scale_s
                cp_list.append({
                    "cp_id": int(cp["cp_id"]),
                    "name": f"CP{int(cp['cp_id'])}",
                    "desc": str(cp["type"]),
                    "s_m": round(s_mapped, 3),
                    "s_tube_m": round(float(cp["s_peak"]), 3),
                    "pos": [round(float(tube_planner.x_grid[idx_p]), 4), round(float(tube_planner.y_grid[idx_p]), 4)],
                    "kappa_peak": round(float(cp["kappa_peak"]), 4),
                    "v_ub_cp": round(float(cp.get("v_ub_cp", 0.0)), 3),
                    "eta_wcsph_t2_mm": round(float(cp.get("eta_wcsph_t2_mm", 0.0)), 2),
                    "eta_rom_corr_t2_mm": round(float(cp.get("eta_rom_corr_t2_mm", 0.0)), 2),
                    "R_mm": round(float(cp.get("raw_residual_R_mm", cp.get("residual_R_mm", 0.0))), 2),
                    "residual_R_mm": round(float(cp.get("raw_residual_R_mm", cp.get("residual_R_mm", 0.0))), 2),
                })
            if cp_list:
                os.makedirs(os.path.dirname(cp_json_path), exist_ok=True)
                with open(cp_json_path, "w", encoding="utf-8") as f_cp_out:
                    json.dump({
                        "map_file": f"xml/{base_name}",
                        "wcsph_engine": "Hybrid B+A (CoupledVelocityTubeH124, dp=0.005 m)",
                        "n_checkpoints": len(cp_list),
                        "wcsph_summary": tube_summary.get("wcsph_summary", {}),
                        "checkpoints": cp_list,
                    }, f_cp_out, indent=2, ensure_ascii=False)
                print(f"  [✓] Đã kiểm tra WCSPH & lưu {len(cp_list)} Checkpoint ra: {cp_json_path}")

        if cp_list:
            scen["checkpoints_6"] = cp_list
            if ros_node is not None:
                ros_node.checkpoints_6 = cp_list
    except Exception as e:
        print(f"  [!] Cảnh báo không thể nạp CoupledVelocityTubeH124: {e}")

    return tube_planner


def publish_map_transition_banner(ros_node, next_map: str, pause_sec: float = 2.5):
    """Hiển thị banner chuyển bản đồ trên RViz2 khi đối chứng liên tiếp nhiều đấu trường."""
    if not HAS_RCLPY or ros_node is None:
        return
    now = ros_node.get_clock().now().to_msg()
    ma = MarkerArray()
    m_inter = Marker()
    m_inter.header.stamp = now
    m_inter.header.frame_id = "map"
    m_inter.ns = "intermission_hud"
    m_inter.id = 200
    m_inter.type = Marker.TEXT_VIEW_FACING
    m_inter.action = Marker.ADD
    m_inter.pose.position.x = 0.0
    m_inter.pose.position.y = 0.0
    m_inter.pose.position.z = 1.85
    m_inter.pose.orientation.w = 1.0
    m_inter.scale.z = 0.22
    m_inter.color = ColorRGBA(r=0.35, g=0.88, b=1.0, a=1.0)
    m_inter.text = f"ĐANG CHUYỂN SANG ĐẤU TRƯỜNG TIẾP THEO:\n>>> {next_map.upper()} (Không CP vs. Có CP WCSPH) <<<"
    ma.markers.append(m_inter)
    ros_node.marker_pub.publish(ma)

    t0 = time.time()
    while time.time() - t0 < pause_sec:
        rclpy.spin_once(ros_node, timeout_sec=0.05)
        time.sleep(0.05)


def run_sequential_3way_benchmark(
    map_name: str = "world7.xml",
    variant_choice: str = "all",
    launch_rviz: bool = True,
    max_time: float = None,
    speed_factor: float = 1.0,
    enable_plot: bool = False,
    recompute_wcsph: bool = False,
    existing_rclpy: bool = False,
    existing_rviz_proc=None,
    keep_rviz_alive: bool = False,
):
    scen = build_map_scenario(map_name)
    base_name = scen["map_name"]
    if max_time is None:
        max_time = 60.0 if "world8" in base_name.lower() else 54.0

    artifacts = _resolve_map_artifacts(base_name)
    model = scen["model"]
    start_pos = scen["start_pos"]
    goal_pos = scen["goal_pos"]
    obs_circles = scen["obs_circles"]
    obs_boxes = scen["obs_boxes"]
    map_geoms = scen["map_geoms"]
    wp_arr = scen["wp_arr"]
    path_s = scen["path_s"]
    checkpoints_6 = scen["checkpoints_6"]

    print("=" * 112)
    print(f"  ĐỐI CHỨNG 2 CẤU HÌNH DFE-MPCC (KHÔNG CHECKPOINT vs. CÓ CHECKPOINT WCSPH HYBRID B+A) | BẢN ĐỒ: {base_name.upper()}")
    print(f"  Quỹ đạo A*: L = {path_s[-1]:.3f} m ({len(wp_arr)} waypoints) | Ma sát đồng nhất mu = 0.35 | 4 Vật cản động")
    print(f"  Chế độ chạy: variant = '{variant_choice}' | max_time = {max_time:.1f} s | speed = {speed_factor}x")
    print("=" * 112)

    has_ros = bool(HAS_RCLPY and rclpy is not None)
    if has_ros and not existing_rclpy:
        rclpy.init()
    ros_node = SimpatRos2RvizHeadlessNode(
        map_name=base_name,
        model=model,
        waypoints=wp_arr,
        obs_boxes=obs_boxes,
        obs_circles=obs_circles,
        map_geoms=map_geoms,
        checkpoints_6=checkpoints_6,
        start_pos=start_pos,
        goal_pos=goal_pos,
    )
    if has_ros:
        ros_node.clear_old_markers()
        ros_node.robot_traj_msg.poses.clear()

    rviz_proc = existing_rviz_proc
    if has_ros and launch_rviz:
        if rviz_proc is None or rviz_proc.poll() is not None:
            rviz_cfg_path = os.path.join(PROJECT_DIR, "config", "simpat_rviz2.rviz")
            print(f">>> Đang khởi chạy giao diện đồ họa ROS 2 RViz2 ({rviz_cfg_path})...")
            env = _build_wslg_rviz_env()
            try:
                rviz_proc = subprocess.Popen(
                    ["rviz2", "-d", rviz_cfg_path],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                print("  [✓] Cửa sổ RViz2 đã kích hoạt! Đợi 2.5 giây để đồng bộ bản đồ & tháp cảm biến...")
            except Exception as e:
                print(f"  [!] Cảnh báo không thể mở rviz2 tự động: {e}")

        for _ in range(12):
            ros_node.static_map_msg.header.stamp = ros_node.get_clock().now().to_msg()
            ros_node.map_pub.publish(ros_node.static_map_msg)
            ros_node.global_plan_msg.header.stamp = ros_node.get_clock().now().to_msg()
            ros_node.plan_pub.publish(ros_node.global_plan_msg)
            rclpy.spin_once(ros_node, timeout_sec=0.05)
            time.sleep(0.20)

    lidar = Lidar(model)

    cache_path = artifacts["cache_path"]
    cached_records = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_records = json.load(f)
        except Exception:
            pass
    cached_records.pop("no_state", None)

    if variant_choice == "all":
        variants_to_run = ["no_checkpoint", "full"]
    else:
        variants_to_run = [variant_choice]

    records = dict(cached_records)

    tube_planner = _load_or_build_tube_and_checkpoints(
        scen=scen,
        ros_node=ros_node,
        cp_json_path=artifacts["cp_json_path"],
        recompute_wcsph=recompute_wcsph,
    )
    checkpoints_6 = scen["checkpoints_6"]

    total_runs = len(variants_to_run)
    for idx, v_key in enumerate(variants_to_run):
        if rviz_proc is not None and rviz_proc.poll() is not None:
            break
        run_idx = idx + 1
        baseline_trail = records.get("no_checkpoint", {}).get("traj_xy") if v_key == "full" else None
        res = run_single_variant_simulation(
            variant_key=v_key,
            run_idx=run_idx,
            total_runs=total_runs,
            scen=scen,
            ros_node=ros_node,
            lidar=lidar,
            max_time=max_time,
            speed_factor=speed_factor,
            rviz_proc=rviz_proc,
            tube_planner=(tube_planner if v_key == "full" else None),
            baseline_traj_xy=baseline_trail,
        )
        records[v_key] = res

        # Nếu còn lượt tiếp theo trong chế độ sequential thì hiển thị banner chuyển tiếp 2.5 giây
        if has_ros and idx < len(variants_to_run) - 1:
            next_v_key = variants_to_run[idx + 1]
            publish_intermission_banner(
                ros_node=ros_node,
                completed_key=v_key,
                run_idx=run_idx,
                total_runs=total_runs,
                next_key=next_v_key,
                res=res,
                pause_sec=2.5,
            )

    # Lưu kết quả chạy vào cache JSON
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        print(f"\n  [✓] Đã lưu bản ghi chạy ROS 2 tại: {cache_path}")
    except Exception as e:
        print(f"  [!] Lỗi ghi file cache JSON: {e}")

    # Xuất đồ thị đối chứng 4-panel
    out_png = artifacts["out_png"]
    generate_3way_mpcc_comparison_figure(scen, records, out_png, show_plot=enable_plot, tube_planner=tube_planner)

    # Bảng tổng kết đối chứng định lượng
    print("\n" + "=" * 122)
    print(f"  BẢNG TỔNG KẾT ĐỐI CHỨNG ĐỊNH LƯỢNG 2 CẤU HÌNH DFE-MPCC TRÊN {base_name.upper()} (MU = 0.35 ĐỒNG NHẤT)")
    print("=" * 122)
    print(f"{'Cấu hình Bộ điều khiển DFE-MPCC':<34} | {'max eta (mm)':>14} | {'eta_P95':>10} | {'Fz,min (N)':>11} | {'d_dyn (m)':>9} | {'|e_c|_max':>9} | {'T_finish':>8} | {'Kết luận'}")
    print("-" * 128)
    for k in ["no_checkpoint", "full"]:
        if k in records:
            st = records[k]
            p95 = float(np.percentile(st["eta_hist"], 95)) if st["eta_hist"] else 0.0
            eta_tag = " (TRÀN!)" if st["max_eta"] > 30.0 else (" (>Stokes)" if st["max_eta"] > 20.0 else " (An toàn)")
            fz_tag = " (NHẤC!)" if st["min_fz"] < 3.0 else " (>3.0N)"
            min_dyn = float(st.get("min_dyn", 1.0))
            min_wall = float(st.get("min_wall", 1.0))
            is_finished = st.get("finished", False)
            if min_dyn <= 0.0:
                eval_tag = f"THẤT BẠI (Va chạm vật cản động: d={min_dyn:.2f}m)"
            elif min_wall <= 0.0:
                eval_tag = f"THẤT BẠI (Va chạm tường: d={min_wall:.2f}m)"
            elif st["max_eta"] > 30.0 or st["min_fz"] < 3.0:
                eval_tag = "THẤT BẠI (Tràn nước / Nhấc bánh)"
            elif not is_finished:
                p_len = st.get('path_length', scen['path_s'][-1])
                eval_tag = f"CHƯA VỀ ĐÍCH (s={st.get('s_prog', 0.0):.2f}/{p_len:.2f}m)"
            elif st["max_eta"] > 20.0:
                eval_tag = "CẢNH BÁO (Vượt ngưỡng Stokes)"
            else:
                eval_tag = "ĐẠT CHUẨN TOÀN DIỆN"
            v_title = VARIANT_META[k]["title"]
            print(
                f"{v_title[:34]:<34} | {st['max_eta']:6.2f}{eta_tag:<8} | {p95:7.2f} mm | "
                f"{st['min_fz']:5.2f}{fz_tag:<6} | {min_dyn:8.2f}m | {st['max_ec']:7.3f} m | {st['t_finish_s']:7.2f}s | {eval_tag}"
            )
    print("=" * 122)

    if has_ros:
        ros_node.destroy_node()
        if not keep_rviz_alive:
            if launch_rviz and rviz_proc is not None and rviz_proc.poll() is None:
                print("\n  [i] Giữ cửa sổ RViz2 thêm 5 giây để bạn quan sát lần cuối...")
                time.sleep(5.0)
                rviz_proc.terminate()
            if not existing_rclpy:
                rclpy.shutdown()

    return records, rviz_proc


def run_multi_map_3way_benchmark(
    map_list: list,
    variant_choice: str = "all",
    launch_rviz: bool = True,
    max_time: float = None,
    speed_factor: float = 1.0,
    enable_plot: bool = False,
    recompute_wcsph: bool = False,
):
    """Chạy nối tiếp đối chứng 2 phiên bản MPCC trên nhiều bản đồ trong cùng 1 cửa sổ RViz2."""
    has_ros = bool(HAS_RCLPY and rclpy is not None)
    if has_ros:
        rclpy.init()

    rviz_proc = None
    all_records = {}
    multi_mode = len(map_list) > 1

    try:
        for idx_m, m_name in enumerate(map_list):
            if rviz_proc is not None and rviz_proc.poll() is not None:
                print("\n  [!] Người dùng đã đóng cửa sổ RViz2. Dừng chuỗi đa bản đồ.")
                break

            is_last = (idx_m == len(map_list) - 1)
            rec_m, rviz_proc = run_sequential_3way_benchmark(
                map_name=m_name,
                variant_choice=variant_choice,
                launch_rviz=launch_rviz,
                max_time=max_time,
                speed_factor=speed_factor,
                enable_plot=(enable_plot and not multi_mode),
                recompute_wcsph=recompute_wcsph,
                existing_rclpy=has_ros,
                existing_rviz_proc=rviz_proc,
                keep_rviz_alive=(not is_last),
            )
            all_records[m_name] = rec_m

            if has_ros and not is_last and (rviz_proc is None or rviz_proc.poll() is None):
                scen_next = build_map_scenario(map_list[idx_m + 1])
                tmp_node = SimpatRos2RvizHeadlessNode(
                    map_name=scen_next["map_name"],
                    model=scen_next["model"],
                    waypoints=scen_next["wp_arr"],
                    obs_boxes=scen_next["obs_boxes"],
                    obs_circles=scen_next["obs_circles"],
                    map_geoms=scen_next["map_geoms"],
                    checkpoints_6=scen_next["checkpoints_6"],
                    start_pos=scen_next["start_pos"],
                    goal_pos=scen_next["goal_pos"],
                )
                tmp_node.clear_old_markers()
                publish_map_transition_banner(tmp_node, map_list[idx_m + 1], pause_sec=2.5)
                tmp_node.destroy_node()
    finally:
        if has_ros and rclpy.ok():
            if rviz_proc is not None and rviz_proc.poll() is None:
                rviz_proc.terminate()
            rclpy.shutdown()

    if len(all_records) > 1:
        print("\n" + "=" * 122)
        print("  TỔNG HỢP ĐỐI CHỨNG 2 PHIÊN BẢN DFE-MPCC TRÊN TẤT CẢ BẢN ĐỒ (MU = 0.35 ĐỒNG NHẤT)")
        print("=" * 122)
        for m_name, recs in all_records.items():
            for k in ["no_checkpoint", "full"]:
                if k in recs:
                    st = recs[k]
                    print(
                        f"  {m_name:<10} | {VARIANT_META[k]['short_name']:<28} | "
                        f"eta_max={st['max_eta']:5.2f} mm | Fz_min={st['min_fz']:5.2f} N | "
                        f"d_dyn={float(st.get('min_dyn', 1.0)):+5.2f} m | T={st['t_finish_s']:5.2f} s | "
                        f"done={st.get('finished', False)}"
                    )
        print("=" * 122)

    return all_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy đối chứng 2 cấu hình DFE-MPCC (Không vs Có Checkpoint SPH) trên World 6/7/8 (mu = 0.35)")
    parser.add_argument(
        "--map",
        type=str,
        default="world7.xml",
        help="Bản đồ đối chứng: world6.xml, world7.xml, world8.xml, 'all' hoặc danh sách cách nhau dấu phẩy (mặc định: world7.xml)",
    )
    parser.add_argument("--all-maps", action="store_true", help="Chạy tuần tự cả 3 bản đồ (world6.xml, world7.xml, world8.xml) trong 1 cửa sổ RViz2")
    parser.add_argument(
        "--variant",
        type=str,
        default="all",
        choices=["all", "no_checkpoint", "full"],
        help="Biến thể cần chạy: 'all' (2 lượt tuần tự), 'no_checkpoint', 'full'",
    )
    parser.add_argument("--no-rviz", action="store_true", help="Chạy xuất số liệu & đồ thị nhanh không bật cửa sổ RViz2")
    parser.add_argument("--plot", action="store_true", default=True, help="Hiển thị cửa sổ đồ thị đối chứng 4-panel sau khi chạy (khi chạy 1 map có DISPLAY)")
    parser.add_argument("--no-plot", action="store_true", help="Không mở cửa sổ đồ thị sau khi chạy (chỉ lưu file ảnh PNG)")
    parser.add_argument("--time", type=float, default=None, help="Thời gian mô phỏng tối đa cho mỗi lượt (mặc định: 54s cho world6/7, 60s cho world8)")
    parser.add_argument("--speed", type=float, default=1.0, help="Tốc độ mô phỏng thời gian thực (vd: 2.0 để xem nhanh gấp đôi)")
    parser.add_argument("--recompute-wcsph", action="store_true", help="Ép chạy lại mô phỏng hạt WCSPH offline (~400-600s/map) thay vì dùng cache JSON có sẵn")
    parser.add_argument("--replot-only", action="store_true", help="Chỉ vẽ lại đồ thị đối chứng từ dữ liệu cache mà không cần chạy lại mô phỏng")
    args = parser.parse_args()

    launch_plot = not args.no_plot
    target_maps = _normalize_map_list(args.map, all_maps=args.all_maps)

    if args.replot_only:
        for m_name in target_maps:
            scen = build_map_scenario(m_name)
            artifacts = _resolve_map_artifacts(scen["map_name"])
            print(f">>> [REPLOT-ONLY]: Đang nạp kịch bản {scen['map_name']} và cache {os.path.basename(artifacts['cache_path'])}...")
            if os.path.exists(artifacts["cp_json_path"]):
                with open(artifacts["cp_json_path"], "r", encoding="utf-8") as f_cp:
                    cp_data = json.load(f_cp)
                if cp_data.get("checkpoints"):
                    scen["checkpoints_6"] = cp_data["checkpoints"]
            if not os.path.exists(artifacts["cache_path"]):
                raise FileNotFoundError(f"Không tìm thấy file cache: {artifacts['cache_path']}")
            with open(artifacts["cache_path"], "r", encoding="utf-8") as f:
                records = json.load(f)
            tube_replot = None
            try:
                from velocity_tube_h124 import CoupledVelocityTubeH124
                tube_replot = CoupledVelocityTubeH124(mu_default=0.35, kappa_th=1.05)
                tube_replot.build_offline_tube(
                    scen["wp_arr"],
                    mu_profile=lambda s: 0.35,
                    phi_camber_profile=lambda s: 0.0,
                    z_com_profile=lambda s: 0.0555,
                    d_net_profile=lambda s: 0.42,
                    run_wcsph_check=False,
                )
            except Exception:
                tube_replot = None
            generate_3way_mpcc_comparison_figure(
                scen,
                records,
                artifacts["out_png"],
                show_plot=(launch_plot and len(target_maps) == 1),
                tube_planner=tube_replot,
            )
            print(f">>> [REPLOT-ONLY] Hoàn tất xuất đồ thị 4-panel tại: {artifacts['out_png']}")
        sys.exit(0)

    if len(target_maps) == 1:
        run_sequential_3way_benchmark(
            map_name=target_maps[0],
            variant_choice=args.variant,
            launch_rviz=(not args.no_rviz),
            max_time=args.time,
            speed_factor=args.speed,
            enable_plot=launch_plot,
            recompute_wcsph=args.recompute_wcsph,
        )
    else:
        run_multi_map_3way_benchmark(
            map_list=target_maps,
            variant_choice=args.variant,
            launch_rviz=(not args.no_rviz),
            max_time=args.time,
            speed_factor=args.speed,
            enable_plot=launch_plot,
            recompute_wcsph=args.recompute_wcsph,
        )
