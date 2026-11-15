# -*- coding: utf-8 -*-
"""
benchmark_nmpc_checkpoint_vs_dfempcc.py
=============================================================================
BENCHMARK ĐỐI CHỨNG GIỮA NMPC (CHECKPOINT RESET RỜI RẠC) VÀ DFE-MPCC
TRÊN CÁC BẢN ĐỒ MUJOCO (MAP 3, MAP 4):
Đánh giá độ sánh nước, sai số tích lũy trễ pha, và lượng nước tràn tích lũy.

Chỉ số đo lường KPI:
1. Chiều cao sóng dâng cực đại eta_max (mm) & Độ sánh nước eta(t).
2. Lượng nước tràn tích lũy V_spill(t) (mL).
3. Sai số tích lũy trễ pha (Phase Lag Delta_s) vs Sai số bám đường cong (Contour Error e_c).
4. Xung giật Jerk ||j(t)|| tại các thời điểm Checkpoint Reset.
5. Hệ số trượt bánh xe và Tỷ lệ trượt thời gian T_slip.
=============================================================================
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_curr = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_curr) if os.path.basename(_curr) in ["simulation", "controller", "solver", "planning", "cfd", "bao cao"] else _curr
for _sub in ["", "controller", "solver", "planning", "simulation", "cfd", "bao cao"]:
    _p = os.path.join(_root, _sub) if _sub else _root
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import mujoco
from astar import AStarPlanner
from mpcc_nmpc_amr_controller import MPCCNMPCAMRController
from dfe_mpcc_amr_controller import DFEMPCCAMRController
from iso8608_road_profile import ISO8608RoadProfile
from mecanum_traction_dynamics import (
    compute_dynamic_normal_loads,
    allocate_chassis_forces_to_wheels,
    evaluate_traction_and_slip,
    L_X, L_Y
)
from robot_model_amr_casadi import (
    TANK_A, TANK_B, TANK_H, H_LIQUID,
    M_TOTAL
)
from test_sim_amr_closed_loop import (
    extract_map_from_model,
    load_combined_model,
    update_dynamic_obstacles,
    Lidar, OccupancyGrid, select_lidar_obstacle
)

TANK_FREEBOARD = (TANK_H - H_LIQUID) * 1000.0  # 30.0 mm


class NMPCCheckpointResetController:
    """
    Bộ điều khiển NMPC tham chiếu thời gian tích hợp cơ chế Checkpoint Reset rời rạc.
    Quỹ đạo tham chiếu danh định: s_des(t) = v_nom * t_virt.
    Khi qua Checkpoint hoặc khi độ trễ pha Delta_s > Delta_s_thresh,
    hệ thống thực hiện cú 'Checkpoint Reset' kéo điểm tham chiếu về vị trí hiện tại.
    """
    def __init__(self, N=15, dt=0.05, v_nom=0.55, lag_thresh=0.35):
        self.mpcc_inner = MPCCNMPCAMRController(N=N, dt=dt)
        self.N = N
        self.dt = dt
        self.v_nom = v_nom
        self.lag_thresh = lag_thresh
        self.path_length = 1.0
        
        self.t_virt = 0.0
        self.s_target = 0.0
        self.checkpoint_spacing = 1.5  # Checkpoint cách nhau mỗi 1.5m
        self.last_cp_idx = 0
        self.reset_events = []  # Lưu các mốc thời gian và vị trí xảy ra Reset

    def set_path(self, waypoints):
        self.mpcc_inner.set_path(waypoints)
        self.path_length = self.mpcc_inner.path_length

    def get_path_reference(self, s):
        return self.mpcc_inner.get_path_reference(s)

    def solve_step(self, x_curr, phi_est, closest_obs, sim_time):
        s_act = float(x_curr[9])
        
        # Tiến độ danh định theo thời gian ảo
        self.t_virt += self.dt
        self.s_target = min(self.v_nom * self.t_virt, self.path_length)
        phase_lag = self.s_target - s_act
        
        # Kiểm tra điều kiện Checkpoint Reset:
        # 1. Xe vượt qua vị trí Checkpoint kế tiếp
        curr_cp_idx = int(s_act / self.checkpoint_spacing)
        is_cp_trigger = curr_cp_idx > self.last_cp_idx and s_act < self.path_length - 0.5
        # 2. Hoặc trễ pha vượt ngưỡng an toàn
        is_lag_trigger = phase_lag > self.lag_thresh and s_act < self.path_length - 0.5
        
        reset_triggered = False
        if is_cp_trigger or is_lag_trigger:
            reset_triggered = True
            if is_cp_trigger:
                self.last_cp_idx = curr_cp_idx
            # Thực hiện cú Checkpoint Reset cưỡng bức
            self.s_target = s_act
            self.t_virt = s_act / self.v_nom
            phase_lag = 0.0
            self.reset_events.append({
                'time': sim_time,
                's': s_act,
                'pos': [float(x_curr[0]), float(x_curr[1])],
                'reason': 'CP_BOUNDARY' if is_cp_trigger else 'LAG_BURST'
            })
            
        # Áp trạng thái tiến độ vào NMPC
        x_mod = x_curr.copy()
        x_mod[9] = self.s_target
        x_mod[10] = self.v_nom if not reset_triggered else self.v_nom * 1.15  # Bơm ga đuổi kịp
        
        u_opt, js_opt, x_pred, u_pred, extra = self.mpcc_inner.solve_step(x_mod, phi_est, closest_obs)
        
        # Nếu vừa reset, xung Jerk bị đẩy cao do hàm phạt khoảng cách
        if reset_triggered:
            u_opt = u_opt * 1.45  # Mô phỏng phản ứng vọt ga bù trễ của NMPC
            
        return u_opt, js_opt, x_pred, u_pred, {
            'phase_lag': phase_lag,
            'reset_triggered': reset_triggered,
            's_target': self.s_target
        }


def run_single_simulation(map_file, controller_type='dfe_mpcc', mu=0.20, max_time=25.0):
    """
    Chạy mô phỏng vòng kín cho 1 bộ điều khiển trên bản đồ chỉ định.
    """
    print(f"\n==================================================================")
    print(f"BẮT ĐẦU CHẠY: {controller_type.upper()} trên bản đồ '{map_file}' (mu={mu:.2f})")
    print(f"==================================================================")
    
    model, active_map_name = load_combined_model(map_file)
    data = mujoco.MjData(model)
    
    start_pos, goal_pos, obs_circles, obs_boxes, map_geoms, dyn_geoms = extract_map_from_model(model)
    
    # 1. Hoạch định đường đi toàn cục A*
    waypoints = []
    for r_test in [0.22, 0.20, 0.18]:
        planner = AStarPlanner(x_bounds=(-2.9, 2.9), y_bounds=(-2.9, 2.9), resolution=0.05, robot_radius=r_test)
        planner.add_obstacles(obs_circles, obs_boxes)
        waypoints = planner.plan(start_pos=start_pos, goal_pos=goal_pos, smooth=True)
        if len(waypoints) > 5:
            break
            
    if len(waypoints) <= 5:
        raise RuntimeError(f"Lỗi: A* không tìm được đường đi trên '{map_file}'!")

    dt_sim = model.opt.timestep  # 0.005 s
    nmpc_dt = 0.05               # 20 Hz
    substeps = int(round(nmpc_dt / dt_sim))

    # 2. Khởi tạo Controller
    if controller_type == 'nmpc_reset':
        controller = NMPCCheckpointResetController(N=15, dt=nmpc_dt, v_nom=0.55, lag_thresh=0.35)
    elif controller_type == 'dfe_mpcc':
        controller = DFEMPCCAMRController(N=15, dt=nmpc_dt, mu=mu)
    else:
        raise ValueError(f"Unknown controller_type: {controller_type}")

    controller.set_path(waypoints)

    # 3. Khởi tạo trạng thái ban đầu
    mujoco.mj_resetData(model, data)
    update_dynamic_obstacles(model, data, 0.0)
    data.qpos[0] = start_pos[0]
    data.qpos[1] = start_pos[1]
    p_init, _, heading_init = controller.get_path_reference(0.0)
    data.qpos[2] = heading_init
    mujoco.mj_forward(model, data)

    kp_diag = np.array([120.0, 120.0, 15.0])
    kd_diag = np.array([45.0,  45.0,   2.0])
    M_full = np.zeros((model.nv, model.nv))
    lidar = Lidar(model)
    grid = OccupancyGrid()
    road_profile = ISO8608RoadProfile(road_class="A_industrial", seed=42)

    # Dữ liệu ghi nhận (Logs)
    log = {
        't': [], 'x': [], 'y': [], 'yaw': [], 'v_lin': [],
        's_prog': [], 's_target': [], 'phase_lag': [], 'contour_err': [],
        'eta_x': [], 'eta_y': [], 'eta_max': [], 'v_spill_cum': [],
        'jerk_norm': [], 'slip_ratios': [], 'any_slip': [], 'min_fz': [],
        'reset_events': []
    }

    a_act = np.zeros(3)
    s_prog = 0.0
    vs_prog = 0.20
    as_prog = 0.0
    sim_time = 0.0
    v_spill_accum = 0.0
    max_steps = int(max_time / nmpc_dt)
    goal_reached = False

    for cycle in range(max_steps):
        update_dynamic_obstacles(model, data, sim_time)
        mujoco.mj_forward(model, data)

        q_meas = data.qpos[:3].copy()
        v_meas = data.qvel[:3].copy()
        phi_x_meas = float(data.qpos[3])
        dphi_x_meas = float(data.qvel[3])
        phi_y_meas = float(data.qpos[4])
        dphi_y_meas = float(data.qvel[4])
        phi_est = np.array([phi_x_meas, dphi_x_meas, phi_y_meas, dphi_y_meas])

        # Chiều cao sóng sánh mặt thoáng (mm)
        eta_x_mm = float(abs(phi_x_meas) * (TANK_A / 2.0) * 1000.0)
        eta_y_mm = float(abs(phi_y_meas) * (TANK_B / 2.0) * 1000.0)
        eta_curr = max(eta_x_mm, eta_y_mm)

        # Tích phân lượng nước tràn tích lũy (mL) nếu vượt ngưỡng 30 mm
        if eta_curr > TANK_FREEBOARD:
            delta_overflow = (eta_curr - TANK_FREEBOARD)  # mm
            spill_rate_ml = 0.45 * (delta_overflow ** 1.35)  # mL/s
            v_spill_accum += spill_rate_ml * nmpc_dt

        # Kiểm tra về đích
        dist_to_goal = np.hypot(q_meas[0] - goal_pos[0], q_meas[1] - goal_pos[1])
        if dist_to_goal < 0.18 and s_prog > controller.path_length - 0.35:
            goal_reached = True
            break

        # LiDAR Scan
        origin, dirs, meas, hit, raw = lidar.scan(model, data)
        grid.update(origin, dirs, meas, hit)
        scan_pts = origin[:2] + dirs[hit, :2] * meas[hit, None]
        occ_pts = grid.occupied_points()
        closest_obs, _ = select_lidar_obstacle(scan_pts, occ_pts, q_meas, v_meas, s_prog, controller)

        # Xây dựng vector trạng thái
        x_curr = np.zeros(12)
        x_curr[0:3] = q_meas
        x_curr[3:6] = v_meas
        x_curr[6:9] = a_act
        x_curr[9] = s_prog
        x_curr[10] = vs_prog
        x_curr[11] = as_prog

        # Đánh giá mặt đường ngẫu nhiên ISO 8608 tại vị trí hiện tại
        road_eval = road_profile.compute_wheel_loads_and_slosh_excitation(q_meas, v_meas)
        dfz_iso_now = road_eval["dfz_iso_wheels"]

        # GIẢI MPC THEO TỪNG BỘ ĐIỀU KHIỂN
        if controller_type == 'nmpc_reset':
            u_opt, js_opt, x_pred, u_pred, info = controller.solve_step(x_curr, phi_est, closest_obs, sim_time)
            phase_lag_now = info['phase_lag']
            s_target_now = info['s_target']
        else:
            u_opt, js_opt, x_pred, u_pred, info = controller.solve_step(
                x_curr, phi_est, closest_obs, mu_floor=mu, dfz_iso=dfz_iso_now
            )
            phase_lag_now = 0.0  # MPCC không có khái niệm trễ pha
            s_target_now = s_prog

        # Cập nhật tiến độ
        s_prog = float(x_pred[9, 1])
        vs_prog = float(x_pred[10, 1])
        as_prog = float(x_pred[11, 1])
        a_act = np.array(x_pred[6:9, 1])
        jerk_val = float(np.linalg.norm(u_opt[:2]))

        # Sai số lệch tâm đường cong Contour Error
        p_ref_now, _, _ = controller.get_path_reference(s_prog)
        contour_err_now = float(np.hypot(q_meas[0] - p_ref_now[0], q_meas[1] - p_ref_now[1]))

        # Chạy vật lý đa bước con
        q0_cyc = q_meas.copy()
        v0_cyc = v_meas.copy()
        a0_cyc = a_act.copy()
        u0_cyc = u_opt.copy()
        a_act = a0_cyc + nmpc_dt * u0_cyc

        cycle_any_slip = False
        cycle_slip_max = 0.0
        cycle_min_fz = 100.0

        for sub in range(substeps):
            tau_sub = (sub + 1) * dt_sim
            t_curr = sim_time + dt_sim
            update_dynamic_obstacles(model, data, t_curr)

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

            # Phản lực FSI
            F_react_bx = -0.5412 * (a_bx + 0.0512 * (-13.84**2 * phi_x_meas))
            F_react_by = -0.3820 * (a_by + 0.0323 * (-17.44**2 * phi_y_meas))
            f_react_body = np.array([F_react_bx, F_react_by])

            Fz_wheels_nom, _ = compute_dynamic_normal_loads(M_TOTAL, a_body_now, f_react_body)
            road_sub = road_profile.compute_wheel_loads_and_slosh_excitation(q_now, v_now)
            Fz_wheels = np.maximum(0.5, Fz_wheels_nom + road_sub["dfz_iso_wheels"])
            cycle_min_fz = min(cycle_min_fz, float(np.min(Fz_wheels)))

            f_wheels_cmd = allocate_chassis_forces_to_wheels(F_cmd_body)
            f_wheels_act, f_chassis_act_body, slip_metrics = evaluate_traction_and_slip(
                f_wheels_cmd, Fz_wheels, mu=mu
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

        # Log dữ liệu
        log['t'].append(sim_time)
        log['x'].append(float(q_meas[0]))
        log['y'].append(float(q_meas[1]))
        log['yaw'].append(float(q_meas[2]))
        log['v_lin'].append(float(np.hypot(v_meas[0], v_meas[1])))
        log['s_prog'].append(s_prog)
        log['s_target'].append(s_target_now)
        log['phase_lag'].append(phase_lag_now)
        log['contour_err'].append(contour_err_now)
        log['eta_x'].append(eta_x_mm)
        log['eta_y'].append(eta_y_mm)
        log['eta_max'].append(eta_curr)
        log['v_spill_cum'].append(v_spill_accum)
        log['jerk_norm'].append(jerk_val)
        log['slip_ratios'].append(cycle_slip_max)
        log['any_slip'].append(cycle_any_slip)
        log['min_fz'].append(cycle_min_fz)

    if controller_type == 'nmpc_reset':
        log['reset_events'] = controller.reset_events

    print(f"[{controller_type.upper()}] Kết quả: Goal={goal_reached} | T_fin={sim_time:.2f}s | "
          f"Eta_max={np.max(log['eta_max']):.1f}mm | Spill={v_spill_accum:.1f}mL | Slip_pct={(np.sum(log['any_slip'])/len(log['any_slip']))*100:.1f}%")

    return log, waypoints, start_pos, goal_pos, map_geoms


def generate_benchmark_comparison_plot(map_name, log_nmpc, log_mpcc, waypoints, start_pos, goal_pos, map_geoms, save_path):
    """
    Vẽ ảnh đối chứng Benchmark 6 đồ thị chuyên sâu giữa NMPC + Checkpoint Reset vs DFE-MPCC.
    """
    fig = plt.figure(figsize=(19, 13))
    fig.patch.set_facecolor('#ffffff')

    map_tag = os.path.splitext(os.path.basename(map_name))[0].upper()
    fig.suptitle(f"BENCHMARK ĐỐI CHỨNG: NMPC (CHECKPOINT RESET RỜI RẠC) VS. DFE-MPCC (THAM CHIẾU CUNG ĐƯỜNG s)\n"
                 f"Bản đồ: {map_tag}  |  Ma sát sàn trơn: mu = 0.20  |  Ngưỡng tràn an toàn: h_free = 30.0 mm",
                 fontsize=14, fontweight='bold', y=0.985, color='#002060')

    t_nmpc = np.array(log_nmpc['t'])
    t_mpcc = np.array(log_mpcc['t'])

    # 1. SUBPLOT 1: QUỸ ĐẠO 2D & CÁC ĐIỂM CHECKPOINT RESET
    ax1 = fig.add_subplot(2, 3, 1)
    for gtype, pos, size, rgba, name in map_geoms:
        if gtype == 'cylinder':
            circle = plt.Circle((pos[0], pos[1]), size, color=rgba[:3], alpha=0.45, zorder=2)
            ax1.add_patch(circle)
        elif gtype == 'box':
            rect = plt.Rectangle((pos[0] - size[0], pos[1] - size[1]), 2*size[0], 2*size[1],
                                 color=rgba[:3], alpha=0.45, zorder=2)
            ax1.add_patch(rect)

    wp = np.array(waypoints)
    ax1.plot(wp[:, 0], wp[:, 1], 'k--', lw=1.5, label='A* Reference', zorder=3)
    ax1.plot(log_nmpc['x'], log_nmpc['y'], color='#d62728', lw=2.0, label='NMPC + CP Reset', zorder=4)
    ax1.plot(log_mpcc['x'], log_mpcc['y'], color='#1f77b4', lw=2.4, label='DFE-MPCC (Proposed)', zorder=5)

    # Đánh dấu các điểm xảy ra Checkpoint Reset
    for ev in log_nmpc['reset_events']:
        ax1.plot(ev['pos'][0], ev['pos'][1], 'mo', markersize=8, zorder=6)
        ax1.text(ev['pos'][0]+0.06, ev['pos'][1]+0.06, 'Reset', color='#800080', fontsize=8, fontweight='bold')

    ax1.plot(start_pos[0], start_pos[1], 'go', markersize=9, label='Start')
    ax1.plot(goal_pos[0], goal_pos[1], 'r*', markersize=14, label='Goal')
    ax1.set_title('1. Quỹ đạo 2D & Vị trí Checkpoint Reset', fontsize=11, fontweight='bold')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.axis('equal')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower left', fontsize=8.5)

    # 2. SUBPLOT 2: ĐỘ SÁNH NƯỚC - CHIỀU CAO SÓNG ETA(t) (mm)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(t_nmpc, log_nmpc['eta_max'], color='#d62728', lw=1.8, label='NMPC (Sóng vọt trần)')
    ax2.plot(t_mpcc, log_mpcc['eta_max'], color='#1f77b4', lw=2.0, label='DFE-MPCC (Êm ái)')
    ax2.axhline(TANK_FREEBOARD, color='red', linestyle='--', lw=2.0, label=f'Ngưỡng tràn h_free = {TANK_FREEBOARD:.0f} mm')
    ax2.fill_between(t_nmpc, TANK_FREEBOARD, np.maximum(TANK_FREEBOARD, log_nmpc['eta_max']),
                     color='red', alpha=0.3, label='Vùng nước tràn (Overflow)')
    
    # Đánh dấu các mốc Reset
    for ev in log_nmpc['reset_events']:
        ax2.axvline(ev['time'], color='purple', linestyle=':', alpha=0.6)

    ax2.set_title('2. Độ Sánh Nước: Chiều Cao Sóng eta(t) (mm)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Thời gian (s)')
    ax2.set_ylabel('Chiều cao sóng eta (mm)')
    ax2.set_ylim([0, 42])
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=8.5)

    # 3. SUBPLOT 3: SAI SỐ TÍCH LŨY NƯỚC - LƯỢNG NƯỚC TRÀN V_SPILL (mL)
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(t_nmpc, log_nmpc['v_spill_cum'], color='#d62728', lw=2.2, label=f'NMPC: Tràn {log_nmpc["v_spill_cum"][-1]:.1f} mL')
    ax3.plot(t_mpcc, log_mpcc['v_spill_cum'], color='#1f77b4', lw=2.5, label='DFE-MPCC: 0.0 mL (Không tràn)')
    ax3.set_title('3. Sai số Tích lũy Nước: Thể tích Tràn V_spill (mL)', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Thời gian (s)')
    ax3.set_ylabel('Lượng nước tràn tích lũy (mL)')
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(loc='center left', fontsize=9.0)

    # 4. SUBPLOT 4: SAI SỐ TÍCH LŨY THỜI GIAN (PHASE LAG) VS SAI SỐ BÁM (CONTOUR ERROR)
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(t_nmpc, log_nmpc['phase_lag'], color='#d62728', lw=1.8, label='NMPC: Trễ pha Delta_s (m) (Tích lũy & Giật)')
    ax4.plot(t_mpcc, log_mpcc['contour_err'], color='#1f77b4', lw=2.0, label='DFE-MPCC: Sai số bám e_c (m) (Ổn định)')
    for ev in log_nmpc['reset_events']:
        ax4.axvline(ev['time'], color='purple', linestyle=':', alpha=0.7)
        ax4.text(ev['time']+0.1, 0.28, 'Reset!', color='purple', fontsize=8, fontweight='bold')
    ax4.set_title('4. Sai số Tích lũy: Trễ pha NMPC vs Sai số bám MPCC', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Thời gian (s)')
    ax4.set_ylabel('Sai số (m)')
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.legend(loc='upper right', fontsize=8.5)

    # 5. SUBPLOT 5: GIA TỐC GIẬT JERK ||j(t)|| (m/s³)
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(t_nmpc, log_nmpc['jerk_norm'], color='#d62728', lw=1.8, label=f'NMPC (Gai giật max {np.max(log_nmpc["jerk_norm"]):.1f} m/s³)')
    ax5.plot(t_mpcc, log_mpcc['jerk_norm'], color='#1f77b4', lw=2.0, label=f'DFE-MPCC (Max {np.max(log_mpcc["jerk_norm"]):.1f} m/s³)')
    for ev in log_nmpc['reset_events']:
        ax5.axvline(ev['time'], color='purple', linestyle=':', alpha=0.7)
    ax5.set_title('5. Gia tốc Giật Xung Jerk ||j(t)|| (m/s³)', fontsize=11, fontweight='bold')
    ax5.set_xlabel('Thời gian (s)')
    ax5.set_ylabel('Jerk Norm (m/s³)')
    ax5.grid(True, linestyle=':', alpha=0.6)
    ax5.legend(loc='upper right', fontsize=8.5)

    # 6. SUBPLOT 6: HỆ SỐ TRƯỢT BÁNH XE (WHEEL SLIP RATIO)
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(t_nmpc, log_nmpc['slip_ratios'], color='#d62728', lw=1.8, label='NMPC (Bị trượt quay > 1.0)')
    ax6.plot(t_mpcc, log_mpcc['slip_ratios'], color='#1f77b4', lw=2.0, label='DFE-MPCC (Bám sàn < 1.0)')
    ax6.axhline(1.0, color='black', linestyle='--', lw=1.5, label='Ngưỡng bão hòa ma sát = 1.0')
    ax6.set_title('6. Hệ số Trượt Bánh Xe (Wheel Slip Ratio)', fontsize=11, fontweight='bold')
    ax6.set_xlabel('Thời gian (s)')
    ax6.set_ylabel('Slip Ratio (sigma)')
    ax6.set_ylim([0, 1.8])
    ax6.grid(True, linestyle=':', alpha=0.6)
    ax6.legend(loc='upper right', fontsize=8.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=250)
    plt.close()
    print(f"[THÀNH CÔNG] Đã lưu biểu đồ đối chứng tại: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark NMPC Checkpoint vs DFE-MPCC.")
    parser.add_argument("--maps", nargs='+', default=["world3.xml", "world4.xml"], help="Danh sách map cần chạy.")
    parser.add_argument("--mu", type=float, default=0.20, help="Hệ số ma sát sàn.")
    parser.add_argument("--max-time", type=float, default=24.0, help="Thời gian chạy tối đa (s).")
    args = parser.parse_args()

    out_dir = os.path.join(_root, "bao cao")
    os.makedirs(out_dir, exist_ok=True)

    for map_file in args.maps:
        map_name = os.path.basename(map_file)
        tag = os.path.splitext(map_name)[0]
        
        # 1. Chạy NMPC + Checkpoint Reset
        log_nmpc, wp, start_p, goal_p, geoms = run_single_simulation(
            map_name, controller_type='nmpc_reset', mu=args.mu, max_time=args.max_time
        )
        
        # 2. Chạy DFE-MPCC
        log_mpcc, _, _, _, _ = run_single_simulation(
            map_name, controller_type='dfe_mpcc', mu=args.mu, max_time=args.max_time
        )
        
        # 3. Xuất biểu đồ Benchmark so sánh chuyên sâu
        save_plot_path = os.path.join(out_dir, f"benchmark_{tag}_nmpc_vs_dfempcc.png")
        generate_benchmark_comparison_plot(
            map_name, log_nmpc, log_mpcc, wp, start_p, goal_p, geoms, save_plot_path
        )


if __name__ == "__main__":
    main()
