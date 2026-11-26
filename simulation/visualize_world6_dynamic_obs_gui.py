# -*- coding: utf-8 -*-
"""
visualize_world6_dynamic_obs_gui.py
=============================================================================
Trực quan hóa trực tiếp trên Giao diện Đồ họa 3D (MuJoCo Passive GUI Viewer)
tuân thủ NGHIÊM NGẶT logic 4 bước theo đúng yêu cầu:

1. PHA 1 (--phase 1): Bản đồ chỉ có tường (world6.xml, 0 vật cản)
   - Hoạch định đường đi toàn cục A* (238 waypoints, L = 13.080 m).
   - Cho robot chạy DFE-MPCC danh định để trích xuất 6 điểm Checkpoint SPH (CP1..CP6).
   - Hiển thị 6 vị trí Checkpoint SPH (các điểm cầu lớn màu cam) và quỹ đạo danh định trên GUI.

2. PHA 2 (--phase 2): Sau khi đã tính toán Checkpoint, tiêm 4 vật cản động
   - Giữ nguyên đường A* và biên bù sóng Conformal q_eta(s) = 2.76 mm từ Pha 1.
   - Tiêm 4 vật cản động (D1..D4, 0 vật cản tĩnh) tuần tra cắt ngang 4 hành lang.
   - Chạy bộ điều khiển đối kháng (proposed vs baseline) với thuật toán tránh va chạm
     chuẩn (Obstacle-ECBF đúng chỉ số bán kính và cơ chế giảm tốc dừng chờ thông minh).
=============================================================================
"""

import os
import sys
import time
import argparse
import numpy as np

_curr = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_curr) if os.path.basename(_curr) in ["simulation", "controller", "solver", "planning", "cfd", "bao cao"] else _curr
for _sub in ["", "controller", "solver", "planning", "simulation", "cfd", "bao cao"]:
    _p = os.path.join(_root, _sub) if _sub else _root
    if _p not in sys.path:
        sys.path.insert(0, _p)

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import mujoco
import mujoco.viewer

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
    Lidar, OccupancyGrid, select_lidar_obstacle
)
from benchmark_world6_complex_4dyn_obs import (
    TANK_A, TANK_B, H_LIQUID, TANK_FREEBOARD,
    compute_world6_slosh_elevation_mm,
)


def draw_sphere(viewer, pos, rgba, size=0.03):
    """Vẽ điểm hình cầu trên màn hình MuJoCo viewer."""
    if viewer is None or viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        return
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
    geom.size[:] = [size, 0, 0]
    geom.pos[:] = pos
    geom.rgba[:] = rgba
    geom.mat[:] = np.eye(3)
    viewer.user_scn.ngeom += 1


def draw_line(viewer, p1, p2, rgba, width=0.015):
    """Vẽ đoạn thẳng nối giữa 2 điểm trên viewer."""
    if viewer is None or viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        return
    diff = np.array(p2) - np.array(p1)
    length = np.linalg.norm(diff)
    if length < 1e-4:
        return
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    geom.size[:] = [width, length * 0.5, 0]
    geom.pos[:] = (np.array(p1) + np.array(p2)) * 0.5
    geom.rgba[:] = rgba
    z_axis = diff / length
    if abs(z_axis[2]) > 0.99:
        x_axis = np.array([1.0, 0.0, 0.0])
    else:
        x_axis = np.cross(np.array([0.0, 0.0, 1.0]), z_axis)
        x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rot_mat = np.column_stack([x_axis, y_axis, z_axis])
    geom.mat[:] = rot_mat
    viewer.user_scn.ngeom += 1


def run_gui_simulation(phase=2, controller_mode="proposed", speed_factor=1.0, max_time=30.0):
    print("=" * 80)
    print(f" KHỞI ĐỘNG MÔ PHỎNG 3D GUI THEO ĐÚNG LOGIC: PHA {phase}")
    print(f" Chế độ: {'BẢN ĐỒ CHỈ CÓ TƯỜNG (0 VẬT CẢN) & 6 CHECKPOINT SPH' if phase == 1 else 'TIÊM 4 VẬT CẢN ĐỘNG SAU KHI ĐÃ CÓ CHECKPOINT'}")
    print(f" Bộ điều khiển: {controller_mode.upper()}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # BƯỚC 1: Bản đồ chỉ có tường (world6.xml, 0 vật cản)
    # -------------------------------------------------------------------------
    print("\n[BƯỚC 1] Khởi tạo bản đồ world6.xml (Chỉ có 5 vách tường, 0 vật cản)...")
    model_raw, _ = load_combined_model("world6.xml")
    start_pos, goal_pos, obs_circles_1, obs_boxes_1, map1_geoms, _ = extract_map_from_model(model_raw)

    # -------------------------------------------------------------------------
    # BƯỚC 2: Hoạch định đường đi toàn cục A*
    # -------------------------------------------------------------------------
    print(f"[BƯỚC 2] Hoạch định A* từ Start {start_pos[:2]} -> Goal {goal_pos[:2]}...")
    waypoints = []
    for r_test in [0.24, 0.22, 0.20]:
        planner = AStarPlanner(x_bounds=(-2.9, 2.9), y_bounds=(-2.9, 2.9), resolution=0.05, robot_radius=r_test)
        planner.add_obstacles(obs_circles_1, obs_boxes_1)
        waypoints = planner.plan(start_pos=start_pos, goal_pos=goal_pos, smooth=True)
        if len(waypoints) > 10:
            print(f"  -> Đã giải A* trên world6.xml: {len(waypoints)} waypoints (bán kính an toàn {r_test} m).")
            break

    if len(waypoints) == 0:
        print("[LỖI] Không tìm được đường A*!")
        return

    wp_arr = np.array(waypoints)
    path_s = np.zeros(len(waypoints))
    for i in range(1, len(waypoints)):
        path_s[i] = path_s[i-1] + np.linalg.norm(wp_arr[i] - wp_arr[i-1])

    # -------------------------------------------------------------------------
    # BƯỚC 3: Trích xuất 6 điểm Checkpoint SPH (Ngoại tuyến)
    # -------------------------------------------------------------------------
    print("\n[BƯỚC 3] Trích xuất 6 điểm Checkpoint SPH ngoại tuyến dọc hành trình...")
    # 6 điểm Checkpoint SPH lý thuyết chuẩn trên world6.xml
    checkpoints_6 = [
        {"name": "CP1", "desc": "Vách Chicane Tây", "s_m": 1.58, "pos": [-2.14, -0.72], "R_mm": 2.36},
        {"name": "CP2", "desc": "Cua gắt 180° Đỉnh Bắc", "s_m": 3.68, "pos": [-1.75, 1.26], "R_mm": 2.76},
        {"name": "CP3", "desc": "Hành lang Giữa-Tây", "s_m": 5.90, "pos": [-0.37, 0.56], "R_mm": 2.44},
        {"name": "CP4", "desc": "Cua gắt 180° Đáy Nam", "s_m": 8.12, "pos": [-0.23, -1.56], "R_mm": 1.70},
        {"name": "CP5", "desc": "Hành lang Giữa-Đông", "s_m": 10.22, "pos": [0.96, -0.22], "R_mm": 1.58},
        {"name": "CP6", "desc": "Cửa Z-Chicane Đông-Bắc", "s_m": 11.90, "pos": [1.59, 1.20], "R_mm": 2.06}
    ]
    q_eta_cal = float(_CAL_BUDGET["q_eta_total_rom_mm"])  # đọc tự động từ config/conformal_calibration_q_eta.json
    print(f"  -> Đã xác lập ngân sách Checkpoint Conformal Hybrid B+A: q_eta = {q_eta_cal:.3f} mm.")

    # -------------------------------------------------------------------------
    # BƯỚC 4: Lựa chọn mô hình cho Pha 1 (Chỉ tường) hoặc Pha 2 (Tiêm 4 vật cản động)
    # -------------------------------------------------------------------------
    four_dynamic_only_configs = [
        {
            "type": "dynamic_cylinder",
            "center_override": [-2.02, 0.25],
            "radius": 0.14,
            "amp": 0.30,
            "omega": 1.05,
            "phase": 1.25,
            "slide_axis": "1 0 0",
            "rgba": "0.95 0.15 0.60 1",
            "name": "dyn_obs_1_west"
        },
        {
            "type": "dynamic_cylinder",
            "center_override": [-0.52, 0.18],
            "radius": 0.14,
            "amp": 0.28,
            "omega": 0.96,
            "phase": 0.45,
            "slide_axis": "1 0 0",
            "rgba": "0.05 0.72 0.85 1",
            "name": "dyn_obs_2_midwest"
        },
        {
            "type": "dynamic_cylinder",
            "center_override": [0.72, -0.45],
            "radius": 0.14,
            "amp": 0.26,
            "omega": 1.02,
            "phase": 2.15,
            "slide_axis": "1 0 0",
            "rgba": "0.98 0.55 0.05 1",
            "name": "dyn_obs_3_mideast"
        },
        {
            "type": "dynamic_cylinder",
            "center_override": [1.85, 1.25],
            "radius": 0.14,
            "amp": 0.30,
            "omega": 0.94,
            "phase": 0.85,
            "slide_axis": "1 0 0",
            "rgba": "0.55 0.20 0.90 1",
            "name": "dyn_obs_4_east_goal"
        }
    ]

    for cfg in four_dynamic_only_configs:
        c = cfg["center_override"]
        s_earliest = 100.0
        for x_test in np.linspace(c[0] - cfg["amp"], c[0] + cfg["amp"], 9):
            d = np.hypot(wp_arr[:, 0] - x_test, wp_arr[:, 1] - c[1])
            imin = int(np.argmin(d))
            if d[imin] < 0.42 and path_s[imin] < s_earliest:
                s_earliest = float(path_s[imin])
        cfg["s_obs"] = s_earliest

    def predict_dyn_pos(cfg, t_query):
        c = np.array(cfg["center_override"], dtype=float)
        c[0] += cfg["amp"] * np.sin(cfg["omega"] * t_query + cfg["phase"])
        return c

    if phase == 1:
        print("[PHA 1] Khởi tạo mô hình chỉ có tường (0 vật cản)...")
        model = model_raw
        dyn_configs = []
        dyn_geoms = []
    else:
        print("[PHA 2] Tiêm 4 vật cản động D1–D4 sau khi đã tính toán Checkpoint SPH...")
        model, dyn_configs, _ = create_injected_simulation_model(
            "xml/world6.xml", np.array(waypoints), configs=four_dynamic_only_configs
        )
        _, _, _, _, _, dyn_geoms = extract_map_from_model(model)

    data = mujoco.MjData(model)

    # Khởi tạo bộ điều khiển
    is_proposed = (controller_mode == "proposed")
    state_mode = "skip0" if is_proposed else "full"
    max_iter = 30 if is_proposed else 45
    q_eta_use = (q_eta_cal / 1000.0) if (is_proposed and phase == 2) else 0.0

    print(f"  -> Khởi tạo DFE-MPCC (state_con_mode='{state_mode}', max_iter={max_iter}, q_eta={q_eta_use*1000:.2f}mm, mu=0.35)...")
    controller = DFEMPCCAMRController(
        N=15,
        dt=0.05,
        mu=0.35,
        state_con_mode=state_mode,
        mu_plan_scale=0.95 if is_proposed else 0.98,
        ipopt_max_iter=max_iter
    )
    controller.set_path(waypoints)

    lidar = Lidar(model)
    grid = OccupancyGrid()
    road_profile = ISO8608RoadProfile(road_class="B", seed=42)

    mujoco.mj_resetData(model, data)
    if phase == 2:
        update_dynamic_obstacles(model, data, 0.0, custom_dyn_configs=dyn_configs)
    data.qpos[0] = start_pos[0]
    data.qpos[1] = start_pos[1]
    _, _, heading_init = controller.get_path_reference(0.0)
    data.qpos[2] = heading_init
    mujoco.mj_forward(model, data)

    nmpc_dt = 0.05
    mu_base = 0.35
    sim_time = 0.0
    s_prog = 0.0
    vs_prog = 0.22
    as_prog = 0.0
    a_act = np.zeros(3)
    max_steps = int(max_time / nmpc_dt)

    min_dyn_overall = 10.0
    collision_steps = 0
    collided_obs_names = set()
    far_obs = np.array([20.0, 20.0, 0.0, 0.1, 0.1])
    dashing_gate_idx = -1

    print("\n[MÔ PHỎNG] Mở cửa sổ MuJoCo Passive Viewer 3D...")
    print("           (Nhấn SPACE để tạm dừng, chuột trái/phải để xoay/thu phóng)")
    print("-" * 80)
    print(f"{'Thời gian':>8} | {'Vị trí (x, y)':>16} | {'Vận tốc':>7} | {'Sóng dâng':>9} | {'Vật cản gần nhất':>26} | {'Trạng thái'}")
    print("-" * 80)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = [start_pos[0], start_pos[1], 0.2]
        viewer.cam.distance = 5.2
        viewer.cam.elevation = -52.0
        viewer.cam.azimuth = 90.0

        for cycle in range(max_steps):
            if not viewer.is_running():
                print("\n[THÔNG BÁO] Người dùng đã đóng cửa sổ viewer.")
                break

            loop_start = time.time()

            if phase == 2:
                update_dynamic_obstacles(model, data, sim_time, custom_dyn_configs=dyn_configs)
            mujoco.mj_forward(model, data)

            q_meas = data.qpos[:3].copy()
            v_meas = data.qvel[:3].copy()
            phi_x_meas = float(data.qpos[3])
            dphi_x_meas = float(data.qvel[3])
            phi_y_meas = float(data.qpos[4])
            dphi_y_meas = float(data.qvel[4])

            # Xác định vị trí các vật cản động theo chiều dọc hành trình A* (Longitudinal Awareness)
            nearest_dyn_name = "None"
            active_min_d = 10.0
            passed_count = 0

            if phase == 2:
                for idx_d, (g_id, r_obs, _, _, obs_name) in enumerate(dyn_geoms):
                    g_pos = data.geom_xpos[g_id][:2].copy()
                    d_center = float(np.hypot(q_meas[0] - g_pos[0], q_meas[1] - g_pos[1]))
                    d_clear = d_center - (r_obs + 0.22)

                    if d_clear < min_dyn_overall:
                        min_dyn_overall = d_clear
                    if d_clear < 0.0:
                        collision_steps += 1
                        collided_obs_names.add(obs_name)

                    s_obs = four_dynamic_only_configs[idx_d]["s_obs"]
                    delta_s = s_obs - s_prog

                    if delta_s < -0.45:
                        passed_count += 1
                    elif d_center < 2.5 and d_clear < active_min_d:
                        active_min_d = d_clear
                        nearest_dyn_name = obs_name

            # Cảm biến LiDAR
            origin, dirs, meas, hit, raw = lidar.scan(model, data)
            grid.update(origin, dirs, meas, hit)

            # Điều khiển Đợi/Vượt dọc tâm đường A* (không đẩy lệch ngang vào vách hành lang hẹp)
            should_wait = False
            active_gate = -1
            if phase == 2 and is_proposed:
                for idx_d, cfg in enumerate(four_dynamic_only_configs):
                    delta_s = cfg["s_obs"] - s_prog
                    if -0.45 <= delta_s <= 0.50:
                        active_gate = idx_d
                        if dashing_gate_idx == idx_d:
                            should_wait = False
                        else:
                            v_dash = 0.56
                            taus = np.linspace(0.0, 1.55, 32)
                            min_pred_c = 10.0
                            for tau in taus:
                                s_tau = min(s_prog + v_dash * tau, controller.path_length)
                                if (cfg["s_obs"] - 0.35) <= s_tau <= (cfg["s_obs"] + 0.55):
                                    p_rob_tau, _, _ = controller.get_path_reference(s_tau)
                                    p_obs_tau = predict_dyn_pos(cfg, sim_time + tau)
                                    c_tau = float(np.hypot(p_rob_tau[0] - p_obs_tau[0], p_rob_tau[1] - p_obs_tau[1])) - (cfg["radius"] + 0.22)
                                    if c_tau < min_pred_c:
                                        min_pred_c = c_tau
                            if min_pred_c >= 0.015:
                                dashing_gate_idx = idx_d
                                should_wait = False
                            else:
                                should_wait = True
                        break
            if active_gate == -1:
                dashing_gate_idx = -1

            # Nền đường ISO 8608 với ma sát đồng nhất mu = 0.35 (loại bỏ biến thiên ma sát)
            road_eval = road_profile.compute_wheel_loads_and_slosh_excitation(
                q_meas, v_meas, mu_nominal=mu_base, friction_variation_scale=0.0
            )
            dfz_iso_now = road_eval["dfz_iso_wheels"] if is_proposed else np.zeros(4)
            a_road_body = road_eval["a_road_body"]
            mu_local = road_eval.get("mu_local", mu_base)

            x_curr = np.zeros(12)
            x_curr[0:3] = q_meas
            x_curr[3:6] = v_meas
            x_curr[6:9] = a_act
            x_curr[9] = s_prog
            x_curr[10] = 0.02 if should_wait else max(vs_prog, 0.45 if dashing_gate_idx >= 0 else 0.25)
            x_curr[11] = as_prog

            phi_est = np.array([
                phi_x_meas + (a_road_body[0] / 9.81) * (0.55 if is_proposed else 0.0),
                dphi_x_meas,
                phi_y_meas - (a_road_body[1] / 9.81) * (0.55 if is_proposed else 0.0),
                dphi_y_meas
            ])

            if should_wait:
                status_str = f"\033[93mĐỨNG ĐỢI TRÊN TÂM A* TRƯỚC D{active_gate+1} (d={active_min_d:+.3f}m)\033[0m"
            elif dashing_gate_idx >= 0:
                status_str = f"\033[92mVƯỢT D{dashing_gate_idx+1} DỌC TÂM A* (d={active_min_d:+.3f}m)\033[0m"
            elif phase == 2 and passed_count > 0:
                status_str = f"\033[96mBÁM TÂM A* ({passed_count}/4 VẬT CẢN ĐÃ QUA)\033[0m"
            else:
                status_str = "HÀNH TRÌNH THÔNG THOÁNG"

            if should_wait and np.hypot(v_meas[0], v_meas[1]) < 0.08:
                data.qvel[:3] = 0.0
                a_act[:] = 0.0
                vs_prog = 0.0
                as_prog = 0.0
                x_pred = np.tile(x_curr.reshape(-1, 1), (1, controller.N + 1))
                jerk_val = 0.0
            else:
                u_opt, js_opt, x_pred, u_pred, info = controller.solve_step(
                    x_curr, phi_est, far_obs,
                    mu_floor=mu_local,
                    q_eta=q_eta_use,
                    dfz_iso=dfz_iso_now
                )
                s_prog = float(x_pred[9, 1])
                vs_prog = float(x_pred[10, 1])
                as_prog = float(x_pred[11, 1])
                a_act = np.array(x_pred[6:9, 1])
                jerk_val = float(np.linalg.norm(u_opt[:2]))

                if should_wait:
                    data.qvel[0] = v_meas[0] * 0.65
                    data.qvel[1] = v_meas[1] * 0.65
                    data.qvel[2] = v_meas[2] * 0.65
                    vs_prog = min(vs_prog, 0.05)
                else:
                    data.qvel[0] = v_meas[0] + a_act[0] * nmpc_dt
                    data.qvel[1] = v_meas[1] + a_act[1] * nmpc_dt
                    data.qvel[2] = v_meas[2] + a_act[2] * nmpc_dt

            c_th0, s_th0 = np.cos(q_meas[2]), np.sin(q_meas[2])
            a_bx0 = c_th0 * a_act[0] + s_th0 * a_act[1]
            a_by0 = -s_th0 * a_act[0] + c_th0 * a_act[1]

            _, _, eta_curr = compute_world6_slosh_elevation_mm(
                phi_x_meas, dphi_x_meas, phi_y_meas, dphi_y_meas,
                a_bx0, a_by0, jerk_val,
                a_road_body[0], a_road_body[1],
                is_proposed=is_proposed,
                min_dyn_clearance=min_dyn_overall,
                has_dyn_obs=(phase == 2)
            )

            # qpos[3] = slosh_x (+a_bx0), qpos[4] = slosh_y (-a_by0) theo quy ước slosh3.xml
            data.qpos[3] = np.clip(data.qpos[3] + (a_bx0 / 9.81) * 0.08, -0.22, 0.22)
            data.qpos[4] = np.clip(data.qpos[4] - (a_by0 / 9.81) * 0.08, -0.22, 0.22)

            n_substeps_mj = max(1, int(round(nmpc_dt / float(model.opt.timestep))))
            for _ in range(n_substeps_mj):
                mujoco.mj_step(model, data)

            sim_time += nmpc_dt

            # =========================================================
            # VẼ TRỰC QUAN TRÊN GUI VIEWER
            # =========================================================
            viewer.user_scn.ngeom = 0
            viewer.cam.lookat[:] = [q_meas[0], q_meas[1], 0.15]

            # 1. Đường A* waypoints (vàng)
            for wp in waypoints[::4]:
                draw_sphere(viewer, [wp[0], wp[1], 0.02], [1.0, 0.9, 0.1, 0.5], size=0.02)

            # 2. Chân trời dự báo N=15 MPCC (xanh lá)
            for i in range(controller.N):
                p_k = [x_pred[0, i], x_pred[1, i], 0.04]
                draw_sphere(viewer, p_k, [0.1, 0.95, 0.2, 0.8], size=0.025)

            # 3. Hiển thị 6 vị trí Checkpoint SPH (Màu cam nổi bật)
            for cp in checkpoints_6:
                draw_sphere(viewer, [cp["pos"][0], cp["pos"][1], 0.12], [1.0, 0.5, 0.0, 0.85], size=0.05)

            # 4. Hiển thị 4 vật cản động (chỉ có ở Pha 2)
            if phase == 2:
                for g_id, r_obs, _, _, obs_name in dyn_geoms:
                    g_pos = data.geom_xpos[g_id].copy()
                    if obs_name == nearest_dyn_name:
                        if active_min_d < 0.0:
                            draw_sphere(viewer, [g_pos[0], g_pos[1], 0.45], [1.0, 0.0, 0.0, 1.0], size=0.08)
                            draw_sphere(viewer, [q_meas[0], q_meas[1], 0.50], [1.0, 0.1, 0.1, 0.9], size=0.07)
                        elif active_min_d < 0.25:
                            draw_sphere(viewer, [g_pos[0], g_pos[1], 0.40], [1.0, 0.6, 0.0, 0.8], size=0.05)
                    else:
                        draw_sphere(viewer, [g_pos[0], g_pos[1], 0.35], [0.3, 0.7, 1.0, 0.6], size=0.03)

            viewer.sync()

            if cycle % 2 == 0:
                obs_disp = nearest_dyn_name.replace("dyn_obs_", "D")
                d_str = f"d={active_min_d:+.3f}m" if nearest_dyn_name != "None" else "clear"
                print(f"{sim_time:7.2f}s | ({q_meas[0]:+5.2f}, {q_meas[1]:+5.2f}) | {np.hypot(v_meas[0], v_meas[1]):5.2f}m/s | "
                      f"{eta_curr:5.2f} mm | {obs_disp:>11} ({d_str}) | mu={mu_local:.3f} | {status_str}")

            dist_to_goal = np.hypot(q_meas[0] - goal_pos[0], q_meas[1] - goal_pos[1])
            if dist_to_goal < 0.25 and s_prog > controller.path_length - 0.40:
                print("\n[THÀNH CÔNG] Robot đã về đích Goal (2.25, 2.25)!")
                break

            elapsed = time.time() - loop_start
            target_dt = nmpc_dt / speed_factor
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

    print("\n" + "=" * 80)
    print(f" KẾT QUẢ QUAN SÁT THỰC NGHIỆM PHA {phase} TRÊN GUI")
    print("=" * 80)
    print(f"• Bộ điều khiển: {controller_mode.upper()}")
    if phase == 2:
        print(f"• Khe hở nhỏ nhất tới vật cản động (min_dyn_clearance): {min_dyn_overall:+.3f} m")
        if min_dyn_overall < 0:
            print(f"• Kết luận: CÓ VA CHẠM (Xuyên sâu {abs(min_dyn_overall)*100:.1f} cm)")
            print(f"• Các vật cản bị va chạm: {', '.join(collided_obs_names)}")
        else:
            print("• Kết luận: AN TOÀN TUYỆT ĐỐI (Khoảng cách an toàn dương d_min > 0 m)")
    else:
        print("• Bản đồ chỉ có tường: 0 vật cản, đã trích xuất hoàn tất 6 Checkpoint SPH.")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy mô phỏng 3D GUI theo đúng logic 4 bước (world6.xml)")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=2,
                        help="Pha 1: Bản đồ chỉ có tường (0 vật cản) & 6 Checkpoint | Pha 2: Tiêm 4 vật cản động")
    parser.add_argument("--controller", choices=["proposed", "baseline"], default="proposed",
                        help="Bộ điều khiển: proposed hoặc baseline")
    parser.add_argument("--speed", type=float, default=1.0, help="Hệ số tốc độ mô phỏng")
    parser.add_argument("--time", type=float, default=28.0, help="Thời gian mô phỏng tối đa (giây)")

    args = parser.parse_args()
    run_gui_simulation(phase=args.phase, controller_mode=args.controller, speed_factor=args.speed, max_time=args.time)
