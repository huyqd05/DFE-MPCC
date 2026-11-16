# -*- coding: utf-8 -*-
"""
benchmark_world6_complex_4dyn_obs.py
=============================================================================
BENCHMARK ĐỐI CHỨNG 2 BIẾN THỂ DFE-MPCC (CÓ VS KHÔNG CHECKPOINT SPH)
TRÊN BẢN ĐỒ MÊ CUNG TƯỜNG PHỨC TẠP (world6.xml, MA SÁT ĐỒNG NHẤT mu = 0.35):

1. PHA 1 (Bản đồ tường phức tạp world6.xml - 5 vách tường mê cung, 0 vật cản):
   - Giải A* chung cho cả 2 bộ điều khiển (Start (-2.25, -2.25) -> Goal (2.25, 2.25)).
   - Quy hoạch ống vận tốc CoupledVelocityTubeH124 và chuẩn hóa các Checkpoint bằng động cơ
     WCSPH Hybrid B+A (dp = 0.005 m, save_checkpoint/load_checkpoint + quét pha T1/T2).
   - Chạy đối chứng 2 cấu hình đồng nhất bộ giải/bộ lọc/ma sát mu = 0.35:
     + DFE-MPCC (Không Checkpoint): q_eta = 0, không dùng Ống vận tốc Checkpoint.
     + DFE-MPCC (Có Checkpoint SPH): tích hợp q_eta = 3.70 mm + Ống vận tốc H124.

2. PHA 2 (Tiêm 4 Vật Cản Động sau A*, 0 Vật Cản Tĩnh - world6.xml + D1..D4):
   - Giữ nguyên đường A* và bộ Checkpoint WCSPH Hybrid B+A từ Pha 1, ma sát đồng nhất mu = 0.35.
   - Tiêm 4 vật cản động (D1..D4) tuần tra cắt ngang 4 hành lang.
=============================================================================
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
from astar import AStarPlanner
from dfe_mpcc_amr_controller import DFEMPCCAMRController
from velocity_tube_h124 import CoupledVelocityTubeH124
from iso8608_road_profile import ISO8608RoadProfile
from obstacle_injector import create_injected_simulation_model
from conformal_checkpoint_eval import load_conformal_calibration_budget

_CAL_BUDGET = load_conformal_calibration_budget()
from mecanum_nullspace_allocation import allocate_chassis_wrench
from mecanum_traction_dynamics import (
    compute_dynamic_normal_loads,
    evaluate_traction_and_slip
)
from robot_model_amr_casadi import (
    TANK_A, TANK_B, TANK_H, H_LIQUID,
    M_TOTAL, m_1x, lx, omega_x, m_1y, ly, omega_y
)
from test_sim_amr_closed_loop import (
    extract_map_from_model,
    load_combined_model,
    update_dynamic_obstacles,
    Lidar, OccupancyGrid, select_lidar_obstacle
)
from benchmark_dfempcc_t1t2_vs_sph_iso import TANK_FREEBOARD


def compute_world6_slosh_elevation_mm(
    phi_x, dphi_x, phi_y, dphi_y,
    a_bx, a_by, jerk_norm,
    a_road_bx=0.0, a_road_by=0.0,
    z_road_mm=0.0, is_proposed=False,
    min_dyn_clearance=10.0, has_dyn_obs=False
):
    """
    TÍNH TOÁN ĐỘ DÂNG SÓNG SÁNH THỰC TẾ (mm) THEO MÔ HÌNH VẬT LÝ ĐỒNG NHẤT:
    Áp dụng cùng một công thức vật lý cho cả hai bộ điều khiển:
    - Góc con lắc tương đương Housner (phi_x, phi_y)
    - Gia tốc quán tính thân xe (a_bx, a_by) và mấp mô nền đường ISO 8608
    - Hiệu ứng phi tuyến Stokes bậc 2
    """
    eta_1x = 0.5 * TANK_A * abs(np.tan(phi_x)) + 0.0085 * abs(dphi_x) * (0.5 * TANK_A)
    eta_1y = 0.5 * TANK_B * abs(np.tan(phi_y)) + 0.0085 * abs(dphi_y) * (0.5 * TANK_B)

    a_eff_x = abs(a_bx) + 0.85 * abs(a_road_bx)
    a_eff_y = abs(a_by) + 0.85 * abs(a_road_by)

    eta_dyn_x = 0.5 * TANK_A * (a_eff_x / 9.81) * 0.95
    eta_dyn_y = 0.5 * TANK_B * (a_eff_y / 9.81) * 0.95

    jerk_wave_x = 0.0010 * jerk_norm + 0.0005 * z_road_mm
    jerk_wave_y = 0.0010 * jerk_norm + 0.0005 * z_road_mm

    eta_base_x = max(eta_1x * 0.65, eta_dyn_x) + jerk_wave_x
    eta_base_y = max(eta_1y * 0.65, eta_dyn_y) + jerk_wave_y

    k1_h_x = np.pi * H_LIQUID / TANK_A
    k1_h_y = np.pi * H_LIQUID / TANK_B
    coth_x = 1.0 / np.tanh(k1_h_x)
    coth_y = 1.0 / np.tanh(k1_h_y)

    stokes_x = (np.pi * (eta_base_x ** 2) / (2.0 * TANK_A)) * coth_x * 1.55
    stokes_y = (np.pi * (eta_base_y ** 2) / (2.0 * TANK_B)) * coth_y * 1.55

    eta_x_mm = (eta_base_x + stokes_x) * 1000.0
    eta_y_mm = (eta_base_y + stokes_y) * 1000.0
    eta_max_mm = max(eta_x_mm, eta_y_mm)
    return float(eta_x_mm), float(eta_y_mm), float(eta_max_mm)


def build_world6_hybrid_checkpoints(tube_planner, waypoints, path_length_w6):
    """
    Chuyển đổi danh sách Checkpoint chuẩn hóa WCSPH Hybrid B+A (dp = 0.005 m)
    từ CoupledVelocityTubeH124 sang định dạng bản đồ world6.xml (chiếu tỷ lệ s theo quỹ đạo A*).
    """
    checkpoints = []
    s_tube_total = max(float(tube_planner.s_grid[-1]), 1e-6)
    scale_s = float(path_length_w6) / s_tube_total
    for cp in tube_planner.checkpoints:
        idx_p = int(cp["idx_peak"])
        s_m = float(cp["s_peak"]) * scale_s
        x_cp = float(tube_planner.x_grid[idx_p])
        y_cp = float(tube_planner.y_grid[idx_p])
        r_raw = float(cp.get("raw_residual_R_mm", cp.get("residual_R_mm", 0.0)))
        r_deb = float(cp.get("debiased_residual_r_mm", 0.0))
        checkpoints.append({
            "cp_id": int(cp["cp_id"]),
            "name": f"CP{int(cp['cp_id'])}",
            "desc": str(cp["type"]),
            "t_s": 0.0,
            "s_m": round(s_m, 3),
            "s_tube_m": round(float(cp["s_peak"]), 3),
            "pos": [round(x_cp, 4), round(y_cp, 4)],
            "kappa_peak": round(float(cp["kappa_peak"]), 4),
            "v_ub_cp": round(float(cp.get("v_ub_cp", 0.0)), 3),
            "eta_dbep_t2_mm": round(float(cp.get("eta_dbep_t2_mm", 0.0)), 2),
            "eta_rom_corr_t2_mm": round(float(cp.get("eta_rom_corr_t2_mm", 0.0)), 2),
            "eta_wcsph_t2_mm": round(float(cp.get("eta_wcsph_t2_mm", 0.0)), 2),
            "eta_wcsph_cp_mm": round(float(cp.get("eta_wcsph_cp_mm", 0.0)), 2),
            "residual_R_mm": round(r_raw, 2),
            "debiased_residual_r_mm": round(r_deb, 2),
            "save_ram_ms": round(float(cp.get("save_ram_ms", 0.0)), 3),
        })
    return checkpoints


def run_controller_world6(
    model,
    waypoints,
    controller_mode="no_checkpoint",
    checkpoints=None,
    q_eta_cal_mm=float(_CAL_BUDGET["q_eta_total_rom_mm"]),
    dyn_configs=None,
    mu=0.35,
    road_class="B",
    max_time=45.0,
    tube_planner=None,
):
    """
    Chạy mô phỏng vòng kín trên bản đồ phức tạp world6.xml với ma sát đồng nhất mu = 0.35
    và cấu hình solver/quan sát/phân bổ lực đồng đẳng để cô lập đúng tác động của Checkpoint.
    """
    data = mujoco.MjData(model)
    start_pos, goal_pos, obs_circles, obs_boxes, map_geoms, dyn_geoms = extract_map_from_model(model)
    has_dyn_obs = (len(dyn_geoms) > 0)

    dt_sim = model.opt.timestep
    nmpc_dt = 0.05
    substeps = int(round(nmpc_dt / dt_sim))

    is_proposed = (controller_mode in ("dfempcc_sph_iso", "with_checkpoint"))
    ctrl_label = "DFE-MPCC (Có Checkpoint SPH)" if is_proposed else "DFE-MPCC (Không Checkpoint)"

    controller = DFEMPCCAMRController(
        N=15, dt=nmpc_dt, mu=mu,
        state_con_mode="skip0",
        mu_plan_scale=0.95,
        ipopt_max_iter=60,
        h_lim_m=0.020,
    )
    controller.set_path(waypoints)
    if is_proposed and tube_planner is not None and getattr(tube_planner, "is_ready", False):
        controller.set_tube_planner(tube_planner)
    wp_arr = np.array(waypoints)
    path_s = np.zeros(len(waypoints))
    for i in range(1, len(waypoints)):
        path_s[i] = path_s[i-1] + np.linalg.norm(wp_arr[i] - wp_arr[i-1])

    mujoco.mj_resetData(model, data)
    update_dynamic_obstacles(model, data, 0.0, custom_dyn_configs=dyn_configs)
    data.qpos[0] = start_pos[0]
    data.qpos[1] = start_pos[1]
    _, _, heading_init = controller.get_path_reference(0.0)
    data.qpos[2] = heading_init
    mujoco.mj_forward(model, data)

    kp_diag = np.array([120.0, 120.0, 15.0])
    kd_diag = np.array([45.0,  45.0,   2.0])
    M_full = np.zeros((model.nv, model.nv))
    lidar = Lidar(model)
    grid = OccupancyGrid()

    road_profile = ISO8608RoadProfile(
        road_class=road_class,
        seed=42,
        k_contact=4800.0,
        c_contact=135.0,
        max_dfz_ratio=0.42
    )

    log = {
        't': [], 'x': [], 'y': [], 'yaw': [],
        'vx': [], 'vy': [], 'v_lin': [], 'vs': [],
        'ax': [], 'ay': [], 's_prog': [], 'contour_err': [],
        'eta_x': [], 'eta_y': [], 'eta_max': [], 'v_spill_cum': [],
        'jerk_norm': [], 'slip_ratios': [], 'any_slip': [],
        'min_fz': [], 'fz_wheels': [], 'z_road_mm': [],
        'q_eta_applied_mm': [], 'solve_ms': [], 'nlp_failed': [],
        'min_dyn_obs_dist': [], 'dyn_obs_pos': []
    }

    a_act = np.zeros(3)
    s_prog = 0.0
    vs_prog = 0.22
    as_prog = 0.0
    sim_time = 0.0
    v_spill_accum = 0.0
    max_steps = int(max_time / nmpc_dt)
    goal_reached = False

    for cycle in range(max_steps):
        update_dynamic_obstacles(model, data, sim_time, custom_dyn_configs=dyn_configs)
        mujoco.mj_forward(model, data)

        q_meas = data.qpos[:3].copy()
        v_meas = data.qvel[:3].copy()
        phi_x_meas = float(data.qpos[3])
        dphi_x_meas = float(data.qvel[3])
        phi_y_meas = float(data.qpos[4])
        dphi_y_meas = float(data.qvel[4])

        road_eval = road_profile.compute_wheel_loads_and_slosh_excitation(
            q_meas, v_meas, mu_nominal=mu, friction_variation_scale=0.0
        )
        dfz_iso_now = road_eval["dfz_iso_wheels"]
        a_road_body = road_eval["a_road_body"]
        z_road_mean_mm = float(np.mean(np.abs(road_eval["z_wheels_mm"])))

        c_th0, s_th0 = np.cos(q_meas[2]), np.sin(q_meas[2])
        a_bx0 = c_th0 * a_act[0] + s_th0 * a_act[1]
        a_by0 = -s_th0 * a_act[0] + c_th0 * a_act[1]

        dist_to_goal = np.hypot(q_meas[0] - goal_pos[0], q_meas[1] - goal_pos[1])
        if dist_to_goal < 0.20 and s_prog > controller.path_length - 0.40:
            goal_reached = True
            break

        origin, dirs, meas, hit, raw = lidar.scan(model, data)
        grid.update(origin, dirs, meas, hit)
        scan_pts = origin[:2] + dirs[hit, :2] * meas[hit, None]
        occ_pts = grid.occupied_points()
        closest_obs, min_obs_dist = select_lidar_obstacle(scan_pts, occ_pts, q_meas, v_meas, s_prog, controller)

        # Khoảng cách thực tới 4 vật cản động và đưa trực tiếp vật cản động gần nhất vào bộ điều khiển
        # (Khắc phục độ trễ lưới chiếm chỗ 0.5s theo khuyến nghị cuối Mục 8 của Báo cáo A6)
        min_dyn_d = 10.0
        dyn_curr_list = []
        nearest_dyn_pos = None
        nearest_dyn_r = 0.15
        for g_id, r_dyn, _, _, _ in dyn_geoms:
            g_pos = data.geom_xpos[g_id][:2].copy()
            dyn_curr_list.append(g_pos)
            d_center = float(np.hypot(q_meas[0] - g_pos[0], q_meas[1] - g_pos[1]))
            d_clear = d_center - (r_dyn + 0.22)
            if d_clear < min_dyn_d:
                min_dyn_d = d_clear
                nearest_dyn_pos = g_pos
                nearest_dyn_r = r_dyn

        if has_dyn_obs and nearest_dyn_pos is not None and min_dyn_d < 0.85:
            # Đưa trực tiếp tọa độ và bán kính chuẩn của vật cản động vào Obstacle-ECBF:
            # obs_param: [x, y, 0.0, r_obs, d_safe] -> r_total = r_robot + r_obs + d_safe = 0.24 + 0.14 + 0.15 = 0.53 m
            closest_obs = np.array([nearest_dyn_pos[0], nearest_dyn_pos[1], 0.0, nearest_dyn_r, 0.15])
            min_obs_dist = min(min_obs_dist, min_dyn_d + 0.22)

        x_curr = np.zeros(12)
        x_curr[0:3] = q_meas
        x_curr[3:6] = v_meas
        x_curr[6:9] = a_act
        x_curr[9] = s_prog
        x_curr[10] = vs_prog
        x_curr[11] = as_prog

        phi_est_ctrl = np.array([
            phi_x_meas + (a_road_body[0] / 9.81) * 0.55,
            dphi_x_meas,
            phi_y_meas - (a_road_body[1] / 9.81) * 0.55,
            dphi_y_meas
        ])

        # Cơ chế phanh chờ đồng đẳng khi vật cản động chắn hành lang hẹp phía trước:
        if has_dyn_obs and min_dyn_d < 0.55:
            x_curr[10] = min(x_curr[10], 0.12)

        if is_proposed:
            q_eta_apply_m = float(q_eta_cal_mm) / 1000.0
            v_ub_h = None
            v_lb_h = None
            if tube_planner is not None and getattr(tube_planner, "is_ready", False):
                s_preview = np.clip(s_prog + np.arange(controller.N) * nmpc_dt * max(vs_prog, 0.35), 0.0, path_s[-1])
                v_ub_nom_q = np.interp(s_preview, tube_planner.s_grid, tube_planner.v_ub_nom)
                v_ub_online, _ = tube_planner.query_online_horizon_tube(
                    s_horizon=s_preview, phi_est=phi_est_ctrl, mu_online=mu
                )
                v_ub_eff = np.minimum(v_ub_nom_q, v_ub_online)
                v_ub_h = np.clip(v_ub_eff, 0.25, 0.60).reshape(1, controller.N)
                v_lb_h = np.zeros((1, controller.N))

            u_opt, js_opt, x_pred, u_pred, info = controller.solve_step(
                x_curr, phi_est_ctrl, closest_obs,
                mu_floor=mu,
                q_eta=q_eta_apply_m,
                dfz_iso=dfz_iso_now,
                v_ub_horizon=v_ub_h,
                v_lb_horizon=v_lb_h,
            )
        else:
            q_eta_apply_m = 0.0
            u_opt, js_opt, x_pred, u_pred, info = controller.solve_step(
                x_curr, phi_est_ctrl, closest_obs,
                mu_floor=mu,
                q_eta=0.0,
                dfz_iso=dfz_iso_now
            )

        s_prog = float(x_pred[9, 1])
        vs_prog = float(x_pred[10, 1])
        as_prog = float(x_pred[11, 1])
        a_act = np.array(x_pred[6:9, 1])
        jerk_val = float(np.linalg.norm(u_opt[:2]))

        eta_x_mm, eta_y_mm, eta_curr = compute_world6_slosh_elevation_mm(
            phi_x_meas, dphi_x_meas, phi_y_meas, dphi_y_meas,
            a_bx0, a_by0, jerk_val,
            a_road_body[0], a_road_body[1],
            z_road_mm=z_road_mean_mm,
            is_proposed=is_proposed,
            min_dyn_clearance=min_dyn_d,
            has_dyn_obs=has_dyn_obs
        )
        if len(log['eta_max']) > 0:
            eta_x_mm = 0.35 * log['eta_x'][-1] + 0.65 * eta_x_mm
            eta_y_mm = 0.35 * log['eta_y'][-1] + 0.65 * eta_y_mm
            eta_curr = max(eta_x_mm, eta_y_mm)

        if eta_curr > TANK_FREEBOARD:
            delta_ov = eta_curr - TANK_FREEBOARD
            spill_rate_ml = 0.95 * (delta_ov ** 1.15)
            v_spill_accum += spill_rate_ml * nmpc_dt

        p_ref_now, _, _ = controller.get_path_reference(s_prog)
        contour_err_now = float(np.hypot(q_meas[0] - p_ref_now[0], q_meas[1] - p_ref_now[1]))

        q0_cyc = q_meas.copy()
        v0_cyc = v_meas.copy()
        a0_cyc = a_act.copy()
        u0_cyc = u_opt.copy()
        a_act = a0_cyc + nmpc_dt * u0_cyc

        cycle_any_slip = False
        cycle_slip_max = 0.0
        cycle_min_fz = 100.0
        last_fz_wheels = np.full(4, 17.78)

        for sub in range(substeps):
            tau_sub = (sub + 1) * dt_sim
            t_curr = sim_time + dt_sim
            update_dynamic_obstacles(model, data, t_curr, custom_dyn_configs=dyn_configs)

            q_target = q0_cyc + tau_sub * v0_cyc + 0.5 * (tau_sub**2) * a0_cyc + (tau_sub**3 / 6.0) * u0_cyc
            v_target = v0_cyc + tau_sub * a0_cyc + 0.5 * (tau_sub**2) * u0_cyc
            a_target = a0_cyc + tau_sub * u0_cyc

            mujoco.mj_fullM(model, data, M_full)
            F_ff = M_full[:3, :3] @ a_target
            fric_ff = np.zeros(3)
            fric_ff[0:2] = 0.5 * np.tanh(v_target[0:2] / 0.02)
            fric_ff[2] = 0.2 * np.tanh(v_target[2] / 0.02)

            q_now = data.qpos[:3]
            v_now = data.qvel[:3]
            F_bias = data.qfrc_bias[:3].copy()
            err_q = q_target - q_now
            err_q[2] = np.arctan2(np.sin(err_q[2]), np.cos(err_q[2]))
            err_v = v_target - v_now

            F_cmd_world = F_ff + fric_ff + kp_diag * err_q + kd_diag * err_v + F_bias
            th_now = q_now[2]
            c_th, s_th = np.cos(th_now), np.sin(th_now)
            F_cmd_bx = c_th * F_cmd_world[0] + s_th * F_cmd_world[1]
            F_cmd_by = -s_th * F_cmd_world[0] + c_th * F_cmd_world[1]
            Tau_cmd_bz = F_cmd_world[2]
            F_cmd_body = np.array([F_cmd_bx, F_cmd_by, Tau_cmd_bz])

            a_bx = c_th * a_act[0] + s_th * a_act[1]
            a_by = -s_th * a_act[0] + c_th * a_act[1]
            a_body_now = np.array([a_bx, a_by])

            F_react_bx = -m_1x * (a_bx + lx * (-omega_x**2 * phi_x_meas))
            F_react_by = -m_1y * (a_by + ly * (-omega_y**2 * phi_y_meas))
            f_react_body = np.array([F_react_bx, F_react_by])

            Fz_wheels_nom, _ = compute_dynamic_normal_loads(M_TOTAL, a_body_now, f_react_body)
            road_sub = road_profile.compute_wheel_loads_and_slosh_excitation(
                q_now, v_now, mu_nominal=mu, friction_variation_scale=0.0
            )
            Fz_wheels_raw = np.maximum(0.0, Fz_wheels_nom + road_sub["dfz_iso_wheels"])
            Fz_wheels_true = np.maximum(0.5, Fz_wheels_raw)
            last_fz_wheels = Fz_wheels_raw.copy()
            cycle_min_fz = min(cycle_min_fz, float(np.min(Fz_wheels_raw)))

            a_r_bx, a_r_by = road_sub["a_road_body"]
            data.qfrc_applied[3] = m_1x * lx * a_r_bx
            data.qfrc_applied[4] = -m_1y * ly * a_r_by

            f_wheels_cmd, _ = allocate_chassis_wrench(F_cmd_body, Fz_wheels_true, mu=mu, use_nullspace=True)

            f_wheels_act, f_chassis_act_body, slip_metrics = evaluate_traction_and_slip(
                f_wheels_cmd * np.sqrt(2.0), Fz_wheels_true, mu=mu
            )

            if slip_metrics['any_slip']:
                cycle_any_slip = True
            cycle_slip_max = max(cycle_slip_max, float(np.max(slip_metrics['slip_ratios'])))

            F_act_world_x = c_th * f_chassis_act_body[0] - s_th * f_chassis_act_body[1]
            F_act_world_y = s_th * f_chassis_act_body[0] + c_th * f_chassis_act_body[1]
            Tau_act_world_z = f_chassis_act_body[2]
            F_cmd_applied = np.array([F_act_world_x, F_act_world_y, Tau_act_world_z])
            F_cmd_applied[0:2] = np.clip(F_cmd_applied[0:2], -30.0, 30.0)
            F_cmd_applied[2] = np.clip(F_cmd_applied[2], -10.0, 10.0)

            data.ctrl[:3] = F_cmd_applied
            mujoco.mj_step(model, data)
            sim_time += dt_sim

        log['t'].append(sim_time)
        log['x'].append(float(q_meas[0]))
        log['y'].append(float(q_meas[1]))
        log['yaw'].append(float(q_meas[2]))
        log['vx'].append(float(v_meas[0]))
        log['vy'].append(float(v_meas[1]))
        log['v_lin'].append(float(np.hypot(v_meas[0], v_meas[1])))
        log['vs'].append(vs_prog)
        log['ax'].append(float(a_act[0]))
        log['ay'].append(float(a_act[1]))
        log['s_prog'].append(s_prog)
        log['contour_err'].append(contour_err_now)
        log['eta_x'].append(eta_x_mm)
        log['eta_y'].append(eta_y_mm)
        log['eta_max'].append(eta_curr)
        log['v_spill_cum'].append(v_spill_accum)
        log['jerk_norm'].append(jerk_val)
        log['slip_ratios'].append(cycle_slip_max)
        log['any_slip'].append(cycle_any_slip)
        log['min_fz'].append(cycle_min_fz)
        log['fz_wheels'].append(last_fz_wheels)
        log['z_road_mm'].append(z_road_mean_mm)
        log['q_eta_applied_mm'].append(q_eta_apply_m * 1000.0)
        log['solve_ms'].append(info['solve_time_ms'])
        log['nlp_failed'].append(0 if info['status'] == 'optimal' else 1)
        log['min_dyn_obs_dist'].append(min_dyn_d)
        log['dyn_obs_pos'].append(dyn_curr_list)

    eta_arr = np.array(log['eta_max'])
    rms_eta = float(np.sqrt(np.mean(eta_arr ** 2))) if len(eta_arr) > 0 else 0.0
    max_eta = float(np.max(eta_arr)) if len(eta_arr) > 0 else 0.0
    slip_pct = float(np.sum(log['any_slip']) / max(1, len(log['any_slip'])) * 100.0)
    max_slip = float(np.max(log['slip_ratios'])) if len(log['slip_ratios']) > 0 else 0.0
    nlp_fail_pct = float(np.mean(log['nlp_failed']) * 100.0) if len(log['nlp_failed']) > 0 else 0.0
    p95_solve_ms = float(np.percentile(log['solve_ms'], 95)) if len(log['solve_ms']) > 0 else 0.0
    jerk_rms = float(np.sqrt(np.mean(np.array(log['jerk_norm'])**2))) if len(log['jerk_norm']) > 0 else 0.0
    overflow_pct = float(np.mean(eta_arr > TANK_FREEBOARD) * 100.0) if len(eta_arr) > 0 else 0.0

    print(f"  [{ctrl_label}] Goal={goal_reached} | T_fin={sim_time:.2f}s | "
          f"Eta_max={max_eta:.2f}mm | Eta_RMS={rms_eta:.2f}mm | "
          f"Spill={v_spill_accum:.2f}mL | Jerk_RMS={jerk_rms:.3f} | "
          f"Max_Slip={max_slip:.3f} | Solve_P95={p95_solve_ms:.1f}ms | NLP_Fail={nlp_fail_pct:.2f}%")

    log['summary'] = {
        'controller': ctrl_label,
        'goal_reached': bool(goal_reached),
        't_finish_s': round(float(sim_time), 2),
        'eta_max_mm': round(max_eta, 2),
        'eta_rms_mm': round(rms_eta, 2),
        'eta_x_max_mm': round(float(np.max(log['eta_x'])), 2),
        'eta_y_max_mm': round(float(np.max(log['eta_y'])), 2),
        'v_spill_ml': round(float(v_spill_accum), 2),
        'overflow_pct': round(overflow_pct, 2),
        'jerk_rms_m_s3': round(jerk_rms, 3),
        'jerk_max_m_s3': round(float(np.max(log['jerk_norm'])), 2),
        'max_slip_ratio': round(max_slip, 4),
        'slip_time_pct': round(slip_pct, 2),
        'min_fz_N': round(float(np.min(log['min_fz'])), 2),
        'contour_err_max_m': round(float(np.max(log['contour_err'])), 3),
        'mean_solve_ms': round(float(np.mean(log['solve_ms'])), 2),
        'p95_solve_ms': round(p95_solve_ms, 2),
        'nlp_fail_pct': round(nlp_fail_pct, 2),
        'min_dyn_clearance_m': round(float(np.min(log['min_dyn_obs_dist'])), 3) if len(dyn_geoms) > 0 else None
    }
    return log, map_geoms, dyn_geoms, start_pos, goal_pos, controller.path_length, road_profile


def draw_world6_map(ax, map_geoms, dyn_geoms, log_t1t2, log_sph_iso, waypoints, checkpoints, start_pos, goal_pos, title_str, show_dyn=True):
    """
    Vẽ bản đồ mê cung phức tạp world6.xml (5 vách tường + 4 vật cản động, 0 vật cản tĩnh).
    """
    for gtype, pos, size, rgba, name in map_geoms:
        if gtype == 'cylinder':
            circle = plt.Circle((pos[0], pos[1]), size, color=rgba[:3], alpha=0.78, ec='black', lw=0.8, zorder=2)
            ax.add_patch(circle)
        elif gtype == 'box':
            is_wall = ('wall' in (name or '').lower())
            color_use = '#4a5568' if is_wall else rgba[:3]
            alpha_use = 0.90 if is_wall else 0.78
            rect = plt.Rectangle((pos[0] - size[0], pos[1] - size[1]), 2*size[0], 2*size[1],
                                 color=color_use, alpha=alpha_use, ec='#1a202c', lw=0.9, zorder=2)
            ax.add_patch(rect)

    if show_dyn and dyn_geoms and len(dyn_geoms) > 0 and len(log_sph_iso.get('dyn_obs_pos', [])) > 0:
        dyn_pts = np.array(log_sph_iso['dyn_obs_pos'])
        dyn_colors = ['#e83e8c', '#00acc1', '#ff8f00', '#7e57c2']
        for j, (g_id, r_dyn, gtype, rgba, b_name) in enumerate(dyn_geoms):
            c = dyn_colors[j % len(dyn_colors)]
            if j < dyn_pts.shape[1]:
                ax.plot(dyn_pts[:, j, 0], dyn_pts[:, j, 1], color=c, ls='-.', lw=2.2, alpha=0.92, zorder=3,
                        label=f'Vật cản động D{j+1} (Cắt ngang)' if j < 4 else None)
                mid_idx = min(len(dyn_pts) - 1, int(len(dyn_pts) * (0.18 + 0.21 * j)))
                cx, cy = dyn_pts[mid_idx, j, 0], dyn_pts[mid_idx, j, 1]
                ax.add_patch(plt.Circle((cx, cy), r_dyn, color=c, alpha=0.65, ec='black', lw=1.1, zorder=4))
                ax.text(cx, cy + r_dyn + 0.09, f"D{j+1}", ha='center', va='bottom',
                        fontsize=7.5, fontweight='bold', color=c,
                        bbox=dict(boxstyle="round,pad=0.14", fc="white", ec=c, lw=0.8, alpha=0.92))

    wp = np.array(waypoints)
    n_cp = len(checkpoints) if checkpoints else 0
    ax.plot(wp[:, 0], wp[:, 1], 'k--', lw=1.6, alpha=0.75, label='Quỹ đạo A* (Mê cung L=13.08m)', zorder=3)
    ax.plot(log_t1t2['x'], log_t1t2['y'], color='#d62728', lw=2.1,
            label='DFE-MPCC (Không Checkpoint)', zorder=4)
    ax.plot(log_sph_iso['x'], log_sph_iso['y'], color='#0055cc', lw=2.5,
            label=f'DFE-MPCC (Có {n_cp} Checkpoint SPH)', zorder=5)

    if checkpoints:
        cp_x = [cp['pos'][0] for cp in checkpoints]
        cp_y = [cp['pos'][1] for cp in checkpoints]
        ax.scatter(cp_x, cp_y, s=80, c='#ffbf00', edgecolors='#800080', linewidths=1.7, marker='D',
                   zorder=7, label=f'{n_cp} Điểm Checkpoint SPH (Hybrid B+A)')
        offsets = [(0.14, -0.24), (-0.72, 0.30), (-0.78, 0.20), (0.14, -0.28), (-0.75, -0.24), (0.15, 0.22)]
        for i, cp in enumerate(checkpoints):
            px, py = cp['pos']
            ox, oy = offsets[i % len(offsets)]
            ax.annotate(
                f"{cp['name']}\n({cp['s_m']:.1f}m)",
                xy=(px, py), xytext=(px + ox, py + oy),
                fontsize=6.8, fontweight='bold', color='#4a0072',
                bbox=dict(boxstyle="round,pad=0.14", fc="#fff8dc", ec="#800080", lw=0.8, alpha=0.92),
                arrowprops=dict(arrowstyle="->", color="#800080", lw=0.8),
                zorder=8
            )

    ax.plot(start_pos[0], start_pos[1], 'go', markersize=9, label='Start (-2.25, -2.25)', zorder=6)
    ax.plot(goal_pos[0], goal_pos[1], 'r*', markersize=14, label='Goal (2.25, 2.25)', zorder=6)
    ax.set_xlim([-3.15, 3.15])
    ax.set_ylim([-3.15, 3.15])
    ax.set_aspect('equal')
    ax.set_title(title_str, fontsize=10.2, fontweight='bold', color='#002060')
    ax.set_xlabel('Tọa độ X (m)', fontsize=9)
    ax.set_ylabel('Tọa độ Y (m)', fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', fontsize=6.8, framealpha=0.92)


def generate_a6_vs_project_synthesis_diagram(save_path):
    """
    Vẽ Sơ đồ Đối sánh Kỹ thuật: Điểm Vượt Trội của 2 Báo cáo A6 so với Dự án hiện tại
    và Kiến trúc Đóng góp Tích hợp Mới (Hybrid DFE-MPCC + 6 CP SPH + ISO 8608 + A6 Safe Upgrades).
    """
    fig, ax = plt.subplots(figsize=(17.0, 10.5), dpi=250)
    fig.patch.set_facecolor('#ffffff')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 68)
    ax.axis('off')

    ax.text(50, 65.2,
            "ĐÁNH GIÁ ĐỐI SÁNH KỸ THUẬT: 2 BÁO CÁO BỘ ĐIỀU KHIỂN A6 VS. DỰ ÁN HIỆN TẠI & ĐÓNG GÓP TÍCH HỢP MỚI",
            ha='center', va='center', fontsize=13.0, fontweight='bold', color='#002060')
    ax.text(50, 62.7,
            "Chắt lọc các cải tiến từ A6 (U1a, U2, 0.95*mu khi quy hoạch T1) + Chuẩn hóa Hybrid B+A WCSPH Checkpoint & ISO 8608 (mu = 0.35 đồng nhất)",
            ha='center', va='center', fontsize=9.4, style='italic', color='#333333')

    def add_box(x, y, w, h, title, lines, fc="#f0f4f8", ec="#1f4e79", title_color="#002060", lw=1.8):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                      facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 2.0, title, ha='center', va='center',
                fontsize=9.5, fontweight='bold', color=title_color, zorder=3)
        ax.plot([x + 1.2, x + w - 1.2], [y + h - 3.4, y + h - 3.4], color=ec, lw=0.8, alpha=0.6, zorder=3)
        text_body = "\n".join(lines)
        ax.text(x + 1.4, y + h - 4.3, text_body, ha='left', va='top',
                fontsize=8.1, color='#1a1a1a', linespacing=1.30, zorder=3)

    # Cột 1: Điểm vượt trội của 2 Báo cáo A6
    add_box(2.5, 36.0, 44.0, 24.0,
            "1. CÁC ĐIỂM VƯỢT TRỘI CỦA 2 BÁO CÁO BỘ ĐIỀU KHIỂN A6",
            [
                "• [U1a - Bỏ 27 ràng buộc bước k=0 (skip0)]: Xóa hoàn toàn lỗi vô nghiệm",
                "  'Restoration_Failed' khi trạng thái đo x0 vừa chạm biên (NLP fail: 12.7% -> 1.8%).",
                "• [U2 - Giới hạn IPOPT max_iter = 30]: Cắt 50% đuôi thời gian giải P95 (141 -> 71 ms).",
                "• [P1 - EKF thích nghi 6 trạng thái + Bù trễ Load Cell 12.5 ms]: Nhận dạng trực tuyến",
                "  tỷ lệ tần số sigma (<1% sai lệch) & khối lượng kappa (<=3%), giảm 3-7 lần sai số góc",
                "  khi mô hình sai tần số +-20% hoặc lệch tâm tải 3 cm (trượt 30.4% -> 5.2%).",
                "• [Chính sách biên ma sát quy hoạch]: Đưa hệ số an toàn 0.95*mu vào quy hoạch Zonotope T1;",
                "  trong benchmark kiểm chứng hiện tại giữ mu = 0.35 đồng nhất (tạm tắt ước lượng ma sát & Dual-mu).",
                "• [Kiểm định thống kê cặp chuẩn mực]: Wilcoxon + CI 95% bootstrap, McNemar, Holm;",
                "  chẩn đoán trung thực kết quả âm: P1 (nhất là kappa) làm hỏng phanh gấp S5 (va chạm",
                "  tăng 0 -> 70% ở mu=0.3, EKF phân kỳ 37.8° khi |a| lớn làm dồn sai số vào kappa*Ms*a)."
            ],
            fc="#fff8eb", ec="#c57b00", title_color="#7a4900")

    # Cột 2: Điểm mạnh riêng có của Dự án đang làm
    add_box(53.5, 36.0, 44.0, 24.0,
            "2. ĐIỂM MẠNH ĐẶC THÙ CỦA DỰ ÁN ĐANG LÀM (DFE-MPCC + SPH + ISO)",
            [
                "• [Hiệu chuẩn Conformal từ các Checkpoint WCSPH Hybrid B+A (dp = 0.005 m)]:",
                "  Đo trực tiếp phần dư sóng phi tuyến hạt WCSPH (Stokes bậc 2 + va đập thành bồn)",
                "  trên nhánh phanh T2 (mà mô hình con lắc DMTP trong A6 luôn đánh giá thấp 7-29%).",
                "• [Mô hình Địa hình Mặt đường Ngẫu nhiên 2D chuẩn ISO 8608:2016]:",
                "  Tính toán tải pháp tuyến động 4 bánh Fz,w^ISO(t) và kích thích nghiêng mặt đường",
                "  a_road(t) (trong khi A6 chỉ xét mặt sàn phẳng danh định).",
                "• [Ràng buộc Chống nhấc bánh Fz,i >= Fz_min & Phạt L1 Exact Penalty]:",
                "  Đưa trực tiếp biến thiên tải pháp tuyến mặt đường vào 12 mặt biên Zonotope T1.",
                "• [Quy trình kiểm thử 2 Pha (Wall-Only A* -> Post-A* Dynamic Obstacle Injection)]:",
                "  Tách bạch bài toán quy hoạch toàn cục và phản xạ né tránh 4 vật cản động."
            ],
            fc="#e8f4fd", ec="#0055cc", title_color="#003380")

    # Mũi tên thẳng đứng sạch sẽ không cắt qua hộp chữ
    ax.annotate("", xy=(24.5, 28.6), xytext=(24.5, 35.6),
                arrowprops=dict(arrowstyle="-|>,head_length=0.8,head_width=0.5", color="#c57b00", lw=2.2), zorder=4)
    ax.text(24.5, 32.1, "Kế thừa U1a, U2 & Chẩn đoán P1", ha='center', va='center',
            fontsize=8.2, fontweight='bold', color="#7a4900",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#c57b00", lw=1.0), zorder=5)

    ax.annotate("", xy=(75.5, 28.6), xytext=(75.5, 35.6),
                arrowprops=dict(arrowstyle="-|>,head_length=0.8,head_width=0.5", color="#0055cc", lw=2.2), zorder=4)
    ax.text(75.5, 32.1, "Kế thừa WCSPH Hybrid B+A & ISO 8608", ha='center', va='center',
            fontsize=8.2, fontweight='bold', color="#003380",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#0055cc", lw=1.0), zorder=5)

    # Khung dưới: Đóng góp tích hợp mới cho Dự án
    add_box(4.0, 3.5, 92.0, 24.5,
            "3. ĐÓNG GÓP KỸ THUẬT MỚI RÚT RA CHO DỰ ÁN (DFE-MPCC + ỐNG KM H124 + CHECKPOINT WCSPH + ISO 8608)",
            [
                "1) KIẾN TRÚC QUAN SÁT & BIÊN AN TOÀN CONFORMAL (Không phụ thuộc FF-RLS khi đang tách bạch tác động Checkpoint):",
                "   • Giữ hệ lọc góc 4 trạng thái cố định kappa=1 kết hợp biên bù Conformal q_eta = 3.70 mm từ WCSPH Hybrid B+A cấp cho",
                "     Slosh-ECBF và phanh dự phòng T2 (h_lim = 20.0 mm), kiểm chứng sạch trên nền sàn đồng nhất mu = 0.35.",
                "2) BỘ GIẢI NLP THÍCH NGHI CHẾ ĐỘ (U1a 'skip0' + U2 max_iter=60 + TRUYỀN THẲNG ĐIỂM LIDAR THÔ):",
                "   • Áp dụng U1a bỏ 27 ràng buộc hằng số ở k=0 giúp bộ giải IPOPT ổn định và giảm thời gian giải trên world6.xml.",
                "   • Đưa trực tiếp tọa độ 4 vật cản động từ LiDAR thô vào Obstacle-ECBF (khắc phục trễ 0.5 s của lưới chiếm chỗ nêu ở cuối Mục 8 A6)",
                "     với hệ số ma sát quy hoạch 0.95*mu = 0.3325 trên sàn danh định mu = 0.35.",
                "3) BÙ ĐỘ HỤT ĐỈNH PHI TUYẾN DMTP (7-29%) BẰNG HYBRID B+A WCSPH CHECKPOINT & ỐNG VẬN TỐC H124:",
                "   • Mô hình con lắc DMTP thường đánh giá thấp đỉnh sóng T2 so với WCSPH hạt phi tuyến. Việc kết hợp ống vận tốc H124",
                "     và lượng tử Conformal q_eta = 3.70 mm giúp bù chính xác phần dư Stokes mà không làm yếu phanh T2."
            ],
            fc="#eaf7ed", ec="#1e7e34", title_color="#0f5132", lw=2.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=250)
    plt.close()
    print(f"[THÀNH CÔNG] Đã lưu Sơ đồ đánh giá đối sánh A6 vs Dự án tại: {save_path}")


def generate_world6_phase1_plot(log_t1t2, log_sph, waypoints, checkpoints, map_geoms, start_pos, goal_pos, save_path):
    """
    Vẽ biểu đồ 6 khung cho Bản đồ phức tạp world6.xml - PHA 1 (Chỉ có các bức tường mê cung, chưa có vật cản).
    """
    n_cp = len(checkpoints) if checkpoints else 0
    fig = plt.figure(figsize=(19.5, 12.5), dpi=250)
    fig.patch.set_facecolor('#ffffff')
    fig.suptitle(
        "BẢN ĐỒ PHỨC TẠP (world6.xml) - PHA 1 (MÊ CUNG 5 VÁCH TƯỜNG, CHƯA CÓ VẬT CẢN):\n"
        f"Giải A* (L = 13.08 m), Chuẩn hóa {n_cp} Checkpoint WCSPH Hybrid B+A (dp = 0.005 m) & So sánh Mức Độ Sánh Nước",
        fontsize=13.5, fontweight='bold', y=0.985, color='#002060'
    )

    t1 = np.array(log_t1t2['t'])
    t2 = np.array(log_sph['t'])

    ax1 = fig.add_subplot(2, 3, 1)
    draw_world6_map(
        ax1, map_geoms, [], log_t1t2, log_sph, waypoints, checkpoints,
        start_pos, goal_pos,
        f"1. Mê Cung Tường Phức Tạp (world6.xml): A* & {n_cp} Checkpoint SPH",
        show_dyn=False
    )

    ax2 = fig.add_subplot(2, 3, 2)
    if checkpoints:
        cp_names = [cp['name'] for cp in checkpoints]
        x_cp = np.arange(len(checkpoints))
        w_cp = 0.27
        eta_dbep_t2 = [cp['eta_dbep_t2_mm'] for cp in checkpoints]
        eta_rom_t2 = [cp['eta_rom_corr_t2_mm'] for cp in checkpoints]
        eta_sph_t2 = [cp['eta_wcsph_t2_mm'] for cp in checkpoints]
        ax2.bar(x_cp - w_cp, eta_dbep_t2, w_cp, color='#7f8c8d', alpha=0.88, label='DBEP thô T2 (mm)')
        ax2.bar(x_cp, eta_rom_t2, w_cp, color='#2ca02c', alpha=0.88, label='ROM Debiased T2 (mm)')
        ax2.bar(x_cp + w_cp, eta_sph_t2, w_cp, color='#d62728', alpha=0.90, label='WCSPH Hybrid B+A T2 (dp=5mm)')
        for j, (v_s, r_val) in enumerate(zip(eta_sph_t2, [cp['residual_R_mm'] for cp in checkpoints])):
            ax2.text(x_cp[j] + w_cp, v_s + 0.4, f"{v_s:.1f}\n(R={r_val:+.1f})",
                     ha='center', va='bottom', fontsize=6.4, fontweight='bold', color='#800080')
        ax2.set_xticks(x_cp)
        ax2.set_xticklabels(cp_names, fontsize=8.0, fontweight='bold')
    ax2.axhline(TANK_FREEBOARD, color='red', ls='--', lw=1.8, label='Ngưỡng tràn h_free = 30.0 mm')
    ax2.set_title(f'2. Đối chiếu T2 Phanh Gắt Tại {n_cp} Checkpoint WCSPH Hybrid B+A', fontsize=10.2, fontweight='bold', color='#002060')
    ax2.set_xlabel('Điểm Checkpoint hình học trên world6.xml')
    ax2.set_ylabel('Đỉnh sóng phanh T2 (mm)')
    ax2.set_ylim([0, 35])
    ax2.grid(True, axis='y', linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=7.6)

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(t1, log_t1t2['eta_max'], color='#d62728', lw=1.9,
             label=f"Không Checkpoint: Max = {np.max(log_t1t2['eta_max']):.1f} mm")
    ax3.plot(t2, log_sph['eta_max'], color='#0055cc', lw=2.2,
             label=f"Có {n_cp} Checkpoint SPH: Max = {np.max(log_sph['eta_max']):.1f} mm")
    ax3.axhline(TANK_FREEBOARD, color='red', ls='--', lw=1.8, label='Ngưỡng tràn 30.0 mm')
    for idx_cp, cp in enumerate(checkpoints):
        ax3.axvline(cp['t_s'], color='#800080', ls=':', alpha=0.55)
        ax3.text(cp['t_s'], 2.0 + 3.0 * (idx_cp % 2), cp['name'], rotation=90, fontsize=6.6, color='#800080', fontweight='bold')
    ax3.set_title('3. Độ Sánh Nước eta(t) trên Mê Cung Tường (mu = 0.35)', fontsize=10.2, fontweight='bold', color='#002060')
    ax3.set_xlabel('Thời gian (s)')
    ax3.set_ylabel('Độ sánh nước eta(t) (mm)')
    ax3.set_ylim([0, 35])
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(loc='upper right', fontsize=8.0)

    ax4 = fig.add_subplot(2, 3, 4)
    s1 = np.array(log_t1t2['s_prog'])
    s2 = np.array(log_sph['s_prog'])
    ax4.plot(s1, log_t1t2['eta_max'], color='#d62728', lw=1.8, label='Không Checkpoint')
    ax4.plot(s2, log_sph['eta_max'], color='#0055cc', lw=2.2, label=f'Có {n_cp} Checkpoint SPH')
    ax4.plot(s2, log_sph['q_eta_applied_mm'], color='#2ca02c', lw=1.6, ls='--', label='Biên bù Conformal q_eta = 3.70 mm')
    for idx_cp, cp in enumerate(checkpoints):
        eta_at_cp = float(np.interp(cp['s_m'], s2, log_sph['eta_max'])) if len(s2) > 0 else 0.0
        ax4.axvline(cp['s_m'], color='#800080', ls=':', alpha=0.65)
        ax4.plot(cp['s_m'], eta_at_cp, 'D', color='#ffbf00', mec='#800080', ms=5.5)
        ax4.text(cp['s_m'] + 0.05, 25.5 - 2.2 * (idx_cp % 2), cp['name'], fontsize=6.8, color='#800080', fontweight='bold')
    ax4.set_title('4. Độ Sánh Nước Dọc Cung Đường Mê Cung s (L = 13.08 m)', fontsize=10.2, fontweight='bold', color='#002060')
    ax4.set_xlabel('Độ dài cung đường s (m)')
    ax4.set_ylabel('Chiều cao sóng / Biên bù (mm)')
    ax4.set_ylim([0, 32])
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.legend(loc='upper right', fontsize=8.0)

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(t1, log_t1t2['solve_ms'], color='#d62728', lw=1.6, alpha=0.85,
             label=f"Không Checkpoint: P95 = {log_t1t2['summary']['p95_solve_ms']:.1f} ms")
    ax5.plot(t2, log_sph['solve_ms'], color='#0055cc', lw=1.8,
             label=f"Có {n_cp} Checkpoint SPH: P95 = {log_sph['summary']['p95_solve_ms']:.1f} ms")
    ax5.axhline(50.0, color='black', ls='--', lw=1.5, label='Chu kỳ thời gian thực 50 ms (20 Hz)')
    ax5.set_title('5. Thời Gian Giải NLP Mỗi Chu Kỳ (skip0, max_iter=60)', fontsize=10.2, fontweight='bold', color='#002060')
    ax5.set_xlabel('Thời gian (s)')
    ax5.set_ylabel('Thời gian giải IPOPT (ms)')
    ax5.grid(True, linestyle=':', alpha=0.6)
    ax5.legend(loc='upper right', fontsize=8.0)

    ax6 = fig.add_subplot(2, 3, 6)
    metrics_names = ['Đỉnh sóng\nEta_max (mm)', 'Sóng dọc\nEta_x,max (mm)', 'Sóng ngang\nEta_y,max (mm)', 'Sóng hiệu dụng\nEta_RMS (mm)']
    vals_t1t2 = [
        log_t1t2['summary']['eta_max_mm'],
        log_t1t2['summary']['eta_x_max_mm'],
        log_t1t2['summary']['eta_y_max_mm'],
        log_t1t2['summary']['eta_rms_mm']
    ]
    vals_sph = [
        log_sph['summary']['eta_max_mm'],
        log_sph['summary']['eta_x_max_mm'],
        log_sph['summary']['eta_y_max_mm'],
        log_sph['summary']['eta_rms_mm']
    ]
    x_pos = np.arange(len(metrics_names))
    w_bar = 0.34
    b1 = ax6.bar(x_pos - w_bar/2, vals_t1t2, w_bar, color='#d62728', alpha=0.88, label='Không Checkpoint')
    b2 = ax6.bar(x_pos + w_bar/2, vals_sph, w_bar, color='#0055cc', alpha=0.92, label=f'Có {n_cp} Checkpoint SPH')
    ax6.axhline(TANK_FREEBOARD, color='red', ls='--', lw=1.5, label='Ngưỡng tràn 30 mm')
    for rect in b1:
        h = rect.get_height()
        ax6.text(rect.get_x() + rect.get_width()/2, h + 0.5, f"{h:.1f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#a00000')
    for rect in b2:
        h = rect.get_height()
        ax6.text(rect.get_x() + rect.get_width()/2, h + 0.5, f"{h:.1f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#003380')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(metrics_names, fontsize=8.8, fontweight='bold')
    ax6.set_ylabel('Biên độ sóng sánh (mm)')
    ax6.set_ylim([0, 34])
    ax6.set_title('6. Tổng Hợp Sánh Nước Pha 1 (Mê Cung world6.xml Chưa Vật Cản)', fontsize=10.2, fontweight='bold', color='#002060')
    ax6.grid(True, axis='y', linestyle=':', alpha=0.6)
    ax6.legend(loc='upper right', fontsize=8.0)

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    plt.savefig(save_path, dpi=250)
    plt.close()
    print(f"[THÀNH CÔNG] Đã lưu Biểu đồ Pha 1 (Bản đồ phức tạp world6.xml chỉ có tường) tại: {save_path}")


def generate_world6_phase2_4dyn_plot(log_t1t2, log_sph, waypoints, checkpoints, map_geoms, dyn_geoms, start_pos, goal_pos, save_path):
    """
    Vẽ biểu đồ 6 khung cho Bản đồ phức tạp world6.xml - PHA 2 (Tiêm 4 Vật Cản Động, KHÔNG Vật Cản Tĩnh).
    """
    n_cp = len(checkpoints) if checkpoints else 0
    fig = plt.figure(figsize=(19.5, 12.5), dpi=250)
    fig.patch.set_facecolor('#ffffff')
    fig.suptitle(
        "BẢN ĐỒ PHỨC TẠP (world6.xml) - PHA 2 (TIÊM 4 VẬT CẢN ĐỘNG SAU A*, MA SÁT ĐỒNG NHẤT mu = 0.35):\n"
        f"Đối chứng 2 cấu hình: DFE-MPCC (Không Checkpoint) vs. DFE-MPCC (Có {n_cp} Checkpoint SPH)",
        fontsize=13.2, fontweight='bold', y=0.985, color='#002060'
    )

    t1 = np.array(log_t1t2['t'])
    t2 = np.array(log_sph['t'])

    ax1 = fig.add_subplot(2, 3, 1)
    draw_world6_map(
        ax1, map_geoms, dyn_geoms, log_t1t2, log_sph, waypoints, checkpoints,
        start_pos, goal_pos,
        f"1. Mê Cung world6 + 4 Vật Cản Động (D1–D4, mu = 0.35) & {n_cp} CP SPH",
        show_dyn=True
    )

    ax2 = fig.add_subplot(2, 3, 2)
    m1_eta = float(np.max(log_t1t2['eta_max']))
    m2_eta = float(np.max(log_sph['eta_max']))
    ax2.plot(t1, log_t1t2['eta_max'], color='#d62728', lw=2.0,
             label=f"Không Checkpoint: Max = {m1_eta:.2f} mm")
    ax2.plot(t2, log_sph['eta_max'], color='#0055cc', lw=2.3,
             label=f"Có {n_cp} Checkpoint SPH: Max = {m2_eta:.2f} mm")
    ax2.axhline(TANK_FREEBOARD, color='red', ls='--', lw=2.0, label=f'Ngưỡng tràn bồn h_free = {TANK_FREEBOARD:.0f} mm')
    if m1_eta > TANK_FREEBOARD:
        ax2.fill_between(t1, TANK_FREEBOARD, np.maximum(TANK_FREEBOARD, log_t1t2['eta_max']),
                         color='red', alpha=0.28, label=f"Vùng tràn ({log_t1t2['summary']['v_spill_ml']:.2f} mL)")
    for idx_cp, cp in enumerate(checkpoints):
        ax2.axvline(cp['t_s'], color='#800080', ls=':', alpha=0.55)
        ax2.text(cp['t_s'], 2.0 + 3.0 * (idx_cp % 2), cp['name'], rotation=90, fontsize=6.6, color='#800080', fontweight='bold')
    ax2.set_title('2. So sánh Độ Sánh Nước Cực Đại eta(t) Khi Né 4 Vật Cản Động', fontsize=10.2, fontweight='bold', color='#002060')
    ax2.set_xlabel('Thời gian (s)')
    ax2.set_ylabel('Chiều cao sóng sánh eta(t) (mm)')
    ax2.set_ylim([0, max(36.0, max(m1_eta, m2_eta) + 5.0)])
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=7.8)

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(t1, log_t1t2['v_spill_cum'], color='#d62728', lw=2.3,
             label=f"Không Checkpoint: Tràn {log_t1t2['v_spill_cum'][-1]:.2f} mL")
    ax3.plot(t2, log_sph['v_spill_cum'], color='#0055cc', lw=2.6,
             label=f"Có {n_cp} Checkpoint SPH: Tràn {log_sph['v_spill_cum'][-1]:.2f} mL")
    ax3_twin = ax3.twinx()
    ax3_twin.plot(t1, log_t1t2['min_dyn_obs_dist'], color='#ff7f0e', ls=':', lw=1.4, alpha=0.80, label='Khe hở tới 4 VC (Không CP)')
    ax3_twin.plot(t2, log_sph['min_dyn_obs_dist'], color='#2ca02c', ls='--', lw=1.5, alpha=0.85, label='Khe hở tới 4 VC (Có CP SPH)')
    ax3_twin.set_ylabel('Khe hở tới Vật cản động gần nhất (m)', fontsize=8.5, color='#155724')
    ax3_twin.set_ylim([0, 3.0])
    ax3.set_title('3. Thể Tích Nước Tràn V_spill (mL) & Khe Hở An Toàn Tới 4 Vật Cản Động', fontsize=10.2, fontweight='bold', color='#002060')
    ax3.set_xlabel('Thời gian (s)')
    ax3.set_ylabel('Thể tích nước tràn tích lũy (mL)', color='#b22222', fontweight='bold')
    ax3.grid(True, linestyle=':', alpha=0.6)
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=7.6)

    ax4 = fig.add_subplot(2, 3, 4)
    s1 = np.array(log_t1t2['s_prog'])
    s2 = np.array(log_sph['s_prog'])
    ax4.plot(s1, log_t1t2['eta_max'], color='#d62728', lw=1.9, label='Không Checkpoint')
    ax4.plot(s2, log_sph['eta_max'], color='#0055cc', lw=2.2, label=f'Có {n_cp} Checkpoint SPH')
    ax4.axhline(TANK_FREEBOARD, color='red', ls='--', lw=1.6, label='Ngưỡng tràn 30.0 mm')
    for idx_cp, cp in enumerate(checkpoints):
        ax4.axvline(cp['s_m'], color='#800080', ls=':', alpha=0.7)
        ax4.text(cp['s_m'] + 0.05, 31.5 + 1.8 * (idx_cp % 2), cp['name'], fontsize=6.8, color='#800080', fontweight='bold')
    ax4.set_title(f'4. Độ Sánh Nước Theo Cung Đường eta(s) Qua {n_cp} Checkpoint & 4 Vật Cản', fontsize=10.2, fontweight='bold', color='#002060')
    ax4.set_xlabel('Vị trí dọc cung đường A* s (m)')
    ax4.set_ylabel('Độ sánh nước eta(s) (mm)')
    ax4.set_ylim([0, max(36.0, max(m1_eta, m2_eta) + 5.0)])
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.legend(loc='upper left', fontsize=8.0)

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(t1, log_t1t2['jerk_norm'], color='#d62728', lw=1.8,
             label=f"Không Checkpoint: RMS = {log_t1t2['summary']['jerk_rms_m_s3']:.3f} m/s³")
    ax5.plot(t2, log_sph['jerk_norm'], color='#0055cc', lw=2.0,
             label=f"Có {n_cp} Checkpoint SPH: RMS = {log_sph['summary']['jerk_rms_m_s3']:.3f} m/s³")
    ax5.set_title('5. Gia Tốc Giật Bệ Xe ||j(t)|| (m/s³) Khi Lách 4 Vật Cản Động', fontsize=10.2, fontweight='bold', color='#002060')
    ax5.set_xlabel('Thời gian (s)')
    ax5.set_ylabel('Jerk Norm ||j|| (m/s³)')
    ax5.grid(True, linestyle=':', alpha=0.6)
    ax5.legend(loc='upper right', fontsize=8.0)

    ax6 = fig.add_subplot(2, 3, 6)
    min_fz_1 = float(np.min(log_t1t2['min_fz']))
    min_fz_2 = float(np.min(log_sph['min_fz']))
    ax6.plot(t1, log_t1t2['min_fz'], color='#d62728', lw=1.9,
             label=f"Không Checkpoint: Fz,min = {min_fz_1:.2f} N")
    ax6.plot(t2, log_sph['min_fz'], color='#0055cc', lw=2.1,
             label=f"Có {n_cp} Checkpoint SPH: Fz,min = {min_fz_2:.2f} N")
    ax6.axhline(3.0, color='#c0392b', ls='--', lw=1.6, label='Ngưỡng nhấc bánh an toàn = 3.0 N')
    ax6.axhline(3.5, color='#d35400', ls=':', lw=1.4, label='Ngưỡng quy hoạch bảo toàn = 3.5 N')
    ax6.set_title('6. Tải Pháp Tuyến Bánh Xe Cực Tiểu F_z,min(t) (mu = 0.35)', fontsize=10.2, fontweight='bold', color='#002060')
    ax6.set_xlabel('Thời gian (s)')
    ax6.set_ylabel('Tải pháp tuyến F_z,min (N)')
    ax6.set_ylim([0, 19.0])
    ax6.grid(True, linestyle=':', alpha=0.6)
    ax6.legend(loc='lower right', fontsize=8.0)

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    plt.savefig(save_path, dpi=250)
    plt.close()
    print(f"[THÀNH CÔNG] Đã lưu Biểu đồ Pha 2 (world6.xml + 4 Vật cản động) tại: {save_path}")


def generate_world6_combined_dashboard(
    log_m1_t1t2, log_m1_sph, map1_geoms,
    log_m2_t1t2, log_m2_sph, map2_geoms, map2_dyn,
    waypoints, checkpoints, start_pos, goal_pos, save_path
):
    """
    Vẽ Dashboard Tổng Hợp Đặt Cạnh Nhau cho Bản đồ Phức Tạp world6.xml:
    (1A) Mê cung chỉ có tường (Pha 1) | (1B) Mê cung + 4 Vật cản động, 0 Tĩnh (Pha 2) | (1C) So sánh tổng hợp
    (2A) Độ sánh nước Pha 1 | (2B) Độ sánh nước Pha 2 (4 Vật cản động) | (2C) Độ sánh theo cung đường s
    """
    n_cp = len(checkpoints) if checkpoints else 0
    fig = plt.figure(figsize=(20.0, 12.8), dpi=250)
    fig.patch.set_facecolor('#ffffff')
    fig.suptitle(
        "TỔNG HỢP BẢN ĐỒ MÊ CUNG PHỨC TẠP (world6.xml): CHỈ CÓ TƯỜNG VS. 4 VẬT CẢN ĐỘNG (mu = 0.35)\n"
        f"Đối chứng 2 cấu hình đồng đẳng: DFE-MPCC (Không Checkpoint) vs. DFE-MPCC (Có {n_cp} Checkpoint SPH)",
        fontsize=13.8, fontweight='bold', y=0.985, color='#002060'
    )

    ax1 = fig.add_subplot(2, 3, 1)
    draw_world6_map(
        ax1, map1_geoms, [], log_m1_t1t2, log_m1_sph, waypoints, checkpoints,
        start_pos, goal_pos,
        f"1A. Pha 1 (Chỉ có 5 Vách Tường Mê Cung): A* & {n_cp} CP SPH",
        show_dyn=False
    )

    ax2 = fig.add_subplot(2, 3, 2)
    draw_world6_map(
        ax2, map2_geoms, map2_dyn, log_m2_t1t2, log_m2_sph, waypoints, checkpoints,
        start_pos, goal_pos,
        "1B. Pha 2 (Tiêm 4 Vật Cản Động D1–D4, 0 Tĩnh): Quỹ Đạo Né",
        show_dyn=True
    )

    ax3 = fig.add_subplot(2, 3, 3)
    categories = [
        'Pha 1\nEta_max (mm)',
        'Pha 1\nEta_RMS (mm)',
        'Pha 2 (4 VC)\nEta_max (mm)',
        'Pha 2 (4 VC)\nEta_RMS (mm)',
        'Pha 2 (4 VC)\nV_spill (mL)'
    ]
    v_base = [
        log_m1_t1t2['summary']['eta_max_mm'],
        log_m1_t1t2['summary']['eta_rms_mm'],
        log_m2_t1t2['summary']['eta_max_mm'],
        log_m2_t1t2['summary']['eta_rms_mm'],
        log_m2_t1t2['summary']['v_spill_ml']
    ]
    v_prop = [
        log_m1_sph['summary']['eta_max_mm'],
        log_m1_sph['summary']['eta_rms_mm'],
        log_m2_sph['summary']['eta_max_mm'],
        log_m2_sph['summary']['eta_rms_mm'],
        log_m2_sph['summary']['v_spill_ml']
    ]
    x_idx = np.arange(len(categories))
    width = 0.35
    rects1 = ax3.bar(x_idx - width/2, v_base, width, color='#d62728', alpha=0.88, label='DFE-MPCC (Không Checkpoint)')
    rects2 = ax3.bar(x_idx + width/2, v_prop, width, color='#0055cc', alpha=0.92, label=f'DFE-MPCC (Có {n_cp} Checkpoint SPH)')
    ax3.axhline(TANK_FREEBOARD, color='red', ls='--', lw=1.5, label='Ngưỡng tràn 30 mm')
    ymax_c = max(40.0, max(v_base + v_prop) * 1.18)
    for r in rects1:
        h = r.get_height()
        ax3.text(r.get_x() + r.get_width()/2, h + 0.5, f"{h:.2f}" if h < 5 else f"{h:.1f}",
                 ha='center', va='bottom', fontsize=8.2, fontweight='bold', color='#900000')
    for r in rects2:
        h = r.get_height()
        ax3.text(r.get_x() + r.get_width()/2, h + 0.5, f"{h:.2f}" if h < 5 else f"{h:.1f}",
                 ha='center', va='bottom', fontsize=8.2, fontweight='bold', color='#003380')
    ax3.set_xticks(x_idx)
    ax3.set_xticklabels(categories, fontsize=8.1, fontweight='bold')
    ax3.set_ylabel('Biên độ sóng (mm) / Tràn (mL)')
    ax3.set_ylim([0, ymax_c])
    ax3.set_title('1C. Đối Chiếu Định Lượng Sánh Nước Trên Mê Cung world6.xml', fontsize=10.2, fontweight='bold', color='#002060')
    ax3.grid(True, axis='y', linestyle=':', alpha=0.6)
    ax3.legend(loc='upper right', fontsize=8.0)

    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(log_m1_t1t2['t'], log_m1_t1t2['eta_max'], color='#d62728', lw=1.9,
             label=f"Không Checkpoint: Max = {log_m1_t1t2['summary']['eta_max_mm']:.2f} mm")
    ax4.plot(log_m1_sph['t'], log_m1_sph['eta_max'], color='#0055cc', lw=2.2,
             label=f"Có {n_cp} Checkpoint SPH: Max = {log_m1_sph['summary']['eta_max_mm']:.2f} mm")
    ax4.axhline(TANK_FREEBOARD, color='red', ls='--', lw=1.8, label='Ngưỡng tràn 30.0 mm')
    for cp in checkpoints:
        ax4.axvline(cp['t_s'], color='#800080', ls=':', alpha=0.55)
    ax4.set_title('2A. Sánh Nước eta(t) Pha 1 (Mê Cung Chỉ Có Tường, 0 Vật Cản)', fontsize=10.2, fontweight='bold', color='#002060')
    ax4.set_xlabel('Thời gian (s)')
    ax4.set_ylabel('Chiều cao sóng eta(t) (mm)')
    ax4.set_ylim([0, 38])
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.legend(loc='upper right', fontsize=8.0)

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(log_m2_t1t2['t'], log_m2_t1t2['eta_max'], color='#d62728', lw=2.0,
             label=f"Không Checkpoint: Max = {log_m2_t1t2['summary']['eta_max_mm']:.2f} mm")
    ax5.plot(log_m2_sph['t'], log_m2_sph['eta_max'], color='#0055cc', lw=2.3,
             label=f"Có {n_cp} Checkpoint SPH: Max = {log_m2_sph['summary']['eta_max_mm']:.2f} mm")
    ax5.axhline(TANK_FREEBOARD, color='red', ls='--', lw=1.8, label='Ngưỡng tràn 30.0 mm')
    if log_m2_t1t2['summary']['eta_max_mm'] > TANK_FREEBOARD:
        ax5.fill_between(log_m2_t1t2['t'], TANK_FREEBOARD, np.maximum(TANK_FREEBOARD, log_m2_t1t2['eta_max']),
                         color='red', alpha=0.28, label=f"Nước tràn ({log_m2_t1t2['summary']['v_spill_ml']:.2f} mL)")
    for cp in checkpoints:
        ax5.axvline(cp['t_s'], color='#800080', ls=':', alpha=0.55)
    ax5.set_title('2B. Sánh Nước eta(t) Pha 2 (Sau Khi Tiêm 4 Vật Cản Động, mu = 0.35)', fontsize=10.2, fontweight='bold', color='#002060')
    ax5.set_xlabel('Thời gian (s)')
    ax5.set_ylabel('Chiều cao sóng eta(t) (mm)')
    ax5.set_ylim([0, max(36.0, max(log_m2_t1t2['summary']['eta_max_mm'], log_m2_sph['summary']['eta_max_mm']) + 5.0)])
    ax5.grid(True, linestyle=':', alpha=0.6)
    ax5.legend(loc='upper right', fontsize=8.0)

    ax6 = fig.add_subplot(2, 3, 6)
    s1 = np.array(log_m2_t1t2['s_prog'])
    s2 = np.array(log_m2_sph['s_prog'])
    ax6.plot(s1, log_m2_t1t2['eta_max'], color='#d62728', lw=1.9, label='Pha 2 (4 VC Động) - Không Checkpoint')
    ax6.plot(s2, log_m2_sph['eta_max'], color='#0055cc', lw=2.2, label=f'Pha 2 (4 VC Động) - Có {n_cp} Checkpoint SPH')
    ax6.plot(s2, log_m2_sph['q_eta_applied_mm'], color='#2ca02c', ls='--', lw=1.6, label='Biên bù Conformal q_eta = 3.70 mm')
    ax6.axhline(TANK_FREEBOARD, color='red', ls='--', lw=1.6, label='Ngưỡng tràn 30.0 mm')
    for cp in checkpoints:
        ax6.axvline(cp['s_m'], color='#800080', ls=':', alpha=0.65)
    ax6.set_title(f'2C. Sánh Nước Theo Cung Đường s (m) Qua {n_cp} Checkpoint & 4 Vật Cản Động', fontsize=10.2, fontweight='bold', color='#002060')
    ax6.set_xlabel('Độ dài cung đường A* s (m)')
    ax6.set_ylabel('Chiều cao sóng eta(s) (mm)')
    ax6.set_ylim([0, 39])
    ax6.grid(True, linestyle=':', alpha=0.6)
    ax6.legend(loc='upper right', fontsize=7.8)

    fig.subplots_adjust(left=0.05, right=0.96, bottom=0.06, top=0.90, wspace=0.25, hspace=0.28)
    plt.savefig(save_path, dpi=250)
    plt.close()
    print(f"[THÀNH CÔNG] Đã lưu Dashboard Tổng hợp Mê cung phức tạp world6.xml + 4 Vật cản động tại: {save_path}")


def main():
    print("===================================================================================")
    print("BẮT ĐẦU BENCHMARK BẢN ĐỒ MÊ CUNG TƯỜNG PHỨC TẠP (world6.xml) & 4 VẬT CẢN ĐỘNG")
    print("===================================================================================")

    out_dir = os.path.join(_root, "bao cao")
    os.makedirs(out_dir, exist_ok=True)
    mu_floor = 0.35
    road_class = "B"

    # 1. Xuất sơ đồ đánh giá 2 Báo cáo A6 vs Dự án hiện tại
    fig_a6_path = os.path.join(out_dir, "so_sanh_kien_truc_A6_va_DFEMPCC_SPH_ISO.png")
    generate_a6_vs_project_synthesis_diagram(fig_a6_path)

    # 2. PHA 1: Bản đồ phức tạp chỉ có tường (world6.xml - 5 vách tường mê cung, 0 vật cản)
    print("\n>>> [PHA 1 - WORLD6.XML] Khởi tạo Bản đồ mê cung phức tạp chỉ có tường (0 vật cản, mu = 0.35)...")
    model_w6_map1, _ = load_combined_model("world6.xml")
    start_pos, goal_pos, obs_circles_1, obs_boxes_1, map1_geoms, _ = extract_map_from_model(model_w6_map1)

    waypoints = []
    for r_test in [0.24, 0.22, 0.20]:
        planner = AStarPlanner(x_bounds=(-2.9, 2.9), y_bounds=(-2.9, 2.9), resolution=0.05, robot_radius=r_test)
        planner.add_obstacles(obs_circles_1, obs_boxes_1)
        waypoints = planner.plan(start_pos=start_pos, goal_pos=goal_pos, smooth=True)
        if len(waypoints) > 10:
            print(f"  -> Đã giải A* trên world6.xml: {len(waypoints)} waypoints (bán kính an toàn {r_test} m).")
            break

    print("\n>>> [PHA 1.1] Quy hoạch Ống vận tốc H124 & Chuẩn hóa Checkpoint WCSPH Hybrid B+A (mu = 0.35, dp = 0.005 m)...")
    wp_arr = np.array(waypoints, dtype=float)
    path_len_w6 = float(np.sum(np.hypot(np.diff(wp_arr[:, 0]), np.diff(wp_arr[:, 1]))))
    t_sph_offline_start = time.time()
    tube_planner_w6 = CoupledVelocityTubeH124(mu_default=mu_floor, kappa_th=1.05)
    tube_summary_w6 = tube_planner_w6.build_offline_tube(
        wp_arr,
        mu_profile=lambda s: mu_floor,
        phi_camber_profile=lambda s: 0.0,
        z_com_profile=lambda s: 0.0555,
        d_net_profile=lambda s: 0.42,
        run_wcsph_check=True,
    )
    checkpoints_w6 = build_world6_hybrid_checkpoints(tube_planner_w6, waypoints, path_len_w6)
    q_eta_budget_mm = float(_CAL_BUDGET["q_eta_total_rom_mm"])
    q_eta_w6_raw_mm = float(tube_summary_w6.get("wcsph_summary", {}).get("max_raw_residual_eta_mm", q_eta_budget_mm))
    sph_offline_duration_s = float(tube_summary_w6.get("wcsph_summary", {}).get("cal_time_s", time.time() - t_sph_offline_start))
    print(
        f"  -> Hoàn thành chuẩn hóa {len(checkpoints_w6)} Checkpoint WCSPH Hybrid B+A trong: {sph_offline_duration_s:.2f} s "
        f"(NSE = {tube_summary_w6['wcsph_summary']['NSE']:.4f}, RMSE = {tube_summary_w6['wcsph_summary']['RMSE_mm']:.2f} mm) | q_eta áp dụng = {q_eta_budget_mm:.2f} mm."
    )

    print("\n>>> [PHA 1.3] Chạy đối chứng 2 cấu hình DFE-MPCC trên world6.xml (Chỉ có tường, mu = 0.35):")
    log_w6_m1_t1t2, map1_geoms, _, start_pos, goal_pos, _, _ = run_controller_world6(
        model_w6_map1, waypoints,
        controller_mode="no_checkpoint",
        checkpoints=None,
        q_eta_cal_mm=0.0,
        dyn_configs=None,
        mu=mu_floor,
        road_class=road_class,
        max_time=45.0,
        tube_planner=None,
    )
    log_w6_m1_sph, _, _, _, _, _, _ = run_controller_world6(
        model_w6_map1, waypoints,
        controller_mode="with_checkpoint",
        checkpoints=checkpoints_w6,
        q_eta_cal_mm=q_eta_budget_mm,
        dyn_configs=None,
        mu=mu_floor,
        road_class=road_class,
        max_time=45.0,
        tube_planner=tube_planner_w6,
    )
    for cp in checkpoints_w6:
        cp["t_s"] = round(float(np.interp(cp["s_m"], log_w6_m1_sph["s_prog"], log_w6_m1_sph["t"])), 2)

    # Lưu bảng tham số Checkpoint WCSPH Hybrid B+A ra JSON
    ckpt_save_path = os.path.join(out_dir, "checkpoints_world6.json")
    with open(ckpt_save_path, "w", encoding="utf-8") as f_ckpt:
        json.dump({
            "map": "world6.xml",
            "wcsph_engine": "Hybrid B+A (CoupledVelocityTubeH124, dp=0.005 m)",
            "n_checkpoints": len(checkpoints_w6),
            "mu_floor": mu_floor,
            "q_eta_calibrated_mm": round(q_eta_budget_mm, 3),
            "q_eta_branch_raw_mm": round(q_eta_w6_raw_mm, 3),
            "sph_offline_duration_s": round(sph_offline_duration_s, 2),
            "wcsph_summary": tube_summary_w6.get("wcsph_summary", {}),
            "checkpoints": checkpoints_w6,
        }, f_ckpt, indent=2, ensure_ascii=False)

    # 3. PHA 2: Tiêm 4 Vật cản động (KHÔNG CÓ Vật cản tĩnh) cắt ngang trực tiếp tim đường A* tại 4 hành lang
    print("\n>>> [PHA 2 - WORLD6.XML] Tiêm 4 Vật cản động (0 Vật cản tĩnh) cắt ngang trực tiếp đường A* tại 4 hành lang...")
    four_dynamic_only_configs = [
        {
            "type": "dynamic_cylinder",
            "center_override": [-2.02, 0.25],   # Hành lang 1 (Tây): cắt ngang trực tiếp tim đường A* (-1.92, 0.25)
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
            "center_override": [-0.52, 0.18],   # Hành lang 2 (Giữa-Tây): cắt ngang trực tiếp tim đường A* (-0.38, 0.18)
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
            "center_override": [0.72, -0.45],   # Hành lang 3 (Giữa-Đông): cắt ngang trực tiếp tim đường A* (0.85, -0.45)
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
            "center_override": [1.85, 1.25],    # Hành lang 4/5 (Cửa ngõ Đông-Bắc): cắt ngang lối vào Goal (1.68, 1.25)
            "radius": 0.14,
            "amp": 0.30,
            "omega": 0.94,
            "phase": 0.85,
            "slide_axis": "1 0 0",
            "rgba": "0.55 0.20 0.90 1",
            "name": "dyn_obs_4_east_goal"
        }
    ]

    model_w6_map2, dyn_configs_w6, injected_w6_info = create_injected_simulation_model(
        "world6.xml", waypoints, configs=four_dynamic_only_configs
    )
    _, _, _, _, map2_geoms, map2_dyn = extract_map_from_model(model_w6_map2)
    print(f"  -> Đã tiêm {len(injected_w6_info)} vật cản ĐỘNG (0 vật cản tĩnh) vào world6.xml.")

    print("\n>>> [PHA 2.2] Chạy đối chứng 2 cấu hình DFE-MPCC trên world6.xml (4 Vật cản động + mu = 0.35):")
    log_w6_m2_t1t2, _, _, _, _, _, _ = run_controller_world6(
        model_w6_map2, waypoints,
        controller_mode="no_checkpoint",
        checkpoints=None,
        q_eta_cal_mm=0.0,
        dyn_configs=dyn_configs_w6,
        mu=mu_floor,
        road_class=road_class,
        max_time=65.0,
        tube_planner=None,
    )
    log_w6_m2_sph, _, _, _, _, _, _ = run_controller_world6(
        model_w6_map2, waypoints,
        controller_mode="with_checkpoint",
        checkpoints=checkpoints_w6,
        q_eta_cal_mm=q_eta_budget_mm,
        dyn_configs=dyn_configs_w6,
        mu=mu_floor,
        road_class=road_class,
        max_time=65.0,
        tube_planner=tube_planner_w6,
    )

    # 4. Xuất các biểu đồ của Bản đồ phức tạp world6.xml
    checkpoints_w6_p2 = []
    for cp in checkpoints_w6:
        cp_copy = dict(cp)
        cp_copy["t_s"] = round(float(np.interp(cp["s_m"], log_w6_m2_sph["s_prog"], log_w6_m2_sph["t"])), 2)
        checkpoints_w6_p2.append(cp_copy)

    fig_w6_p1 = os.path.join(out_dir, "benchmark_world6_map1_wall_only_6cp.png")
    fig_w6_p2 = os.path.join(out_dir, "benchmark_world6_map2_4dyn_obs_slosh.png")
    fig_w6_comb = os.path.join(out_dir, "tong_hop_world6_phuc_tap_4_vat_can_dong.png")

    generate_world6_phase1_plot(
        log_w6_m1_t1t2, log_w6_m1_sph, waypoints, checkpoints_w6,
        map1_geoms, start_pos, goal_pos, fig_w6_p1
    )
    generate_world6_phase2_4dyn_plot(
        log_w6_m2_t1t2, log_w6_m2_sph, waypoints, checkpoints_w6_p2,
        map2_geoms, map2_dyn, start_pos, goal_pos, fig_w6_p2
    )
    generate_world6_combined_dashboard(
        log_w6_m1_t1t2, log_w6_m1_sph, map1_geoms,
        log_w6_m2_t1t2, log_w6_m2_sph, map2_geoms, map2_dyn,
        waypoints, checkpoints_w6_p2, start_pos, goal_pos, fig_w6_comb
    )

    json_out = os.path.join(out_dir, "benchmark_world6_4dyn_obs_results.json")
    payload = {
        "map_file": "xml/world6.xml",
        "protocol": (
            f"2-Way Controlled Checkpoint Ablation (Uniform mu = 0.35, no friction estimation): "
            f"DFE-MPCC Without Checkpoint vs DFE-MPCC With {len(checkpoints_w6)} Hybrid B+A WCSPH Checkpoints"
        ),
        "wcsph_engine": "Hybrid B+A (CoupledVelocityTubeH124, dp=0.005 m)",
        "n_checkpoints": len(checkpoints_w6),
        "wcsph_summary": tube_summary_w6.get("wcsph_summary", {}),
        "astar_path_length_m": round(path_len_w6, 3),
        "astar_waypoints_count": len(waypoints),
        "mu_floor": mu_floor,
        "friction_mode": "uniform_constant_0.35",
        "iso8608_road_class": road_class,
        "q_eta_calibrated_mm": round(q_eta_budget_mm, 3),
        "dynamic_obstacles_injected": 4,
        "static_obstacles_injected": 0,
        "checkpoints": checkpoints_w6,
        "phase1_wall_only_complex": {
            "dfe_mpcc_no_checkpoint": log_w6_m1_t1t2['summary'],
            "dfe_mpcc_with_checkpoint": log_w6_m1_sph['summary']
        },
        "phase2_4_dynamic_obstacles_only": {
            "dfe_mpcc_no_checkpoint": log_w6_m2_t1t2['summary'],
            "dfe_mpcc_with_checkpoint": log_w6_m2_sph['summary']
        }
    }
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[THÀNH CÔNG] Đã lưu dữ liệu JSON của world6.xml tại: {json_out}")


if __name__ == "__main__":
    main()
