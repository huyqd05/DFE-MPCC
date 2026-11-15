"""
test_sim_amr_test3_traction.py
=============================================================================
KỊCH BẢN KIỂM CHỨNG TEST 3 (STRESS-TEST CHO BÀI BÁO Q1/Q2):
ĐÁNH GIÁ ẢNH HƯỞNG CỦA CHUYỂN DỊCH TẢI PHÁP TUYẾN (LOAD TRANSFER)
VÀ GIỚI HẠN BÁM MA SÁT BÁNH XE MECANUM (TRACTION LIMIT & WHEEL SLIP)

Mục đích khoa học:
1. "Bóc trần" điểm gãy của Baseline MPCC: Khi xe tăng tốc hoặc ôm cua gắt,
   mô-men lật do quán tính và sóng sánh FSI làm các bánh xe bị giảm tải (Fz giảm).
   Baseline MPCC không biết điều này, đòi hỏi lực kéo vượt quá elip ma sát (mu * Fz),
   gây hiện tượng trượt vi mô (Wheel slip), lệch quỹ đạo và rung lắc kích động sóng sánh.
2. Cung cấp bộ công cụ định lượng (Slip Ratio, Normal Load, Trajectory Drift)
   để làm minh chứng đối chứng (Benchmark Baseline) cho phương pháp Traction-Aware MPCC.
=============================================================================
"""

import os
import sys
import time
import argparse

_curr = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_curr) if os.path.basename(_curr) in ["simulation", "controller", "solver", "planning", "cfd", "bao cao"] else _curr
for _sub in ["", "controller", "solver", "planning", "simulation", "cfd", "bao cao"]:
    _p = os.path.join(_root, _sub) if _sub else _root
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mujoco
try:
    import mujoco.viewer
except ImportError:
    pass
import numpy as np
import matplotlib.pyplot as plt

from astar import AStarPlanner
from mpcc_nmpc_amr_controller import MPCCNMPCAMRController
from dfe_mpcc_amr_controller import DFEMPCCAMRController
from mecanum_traction_dynamics import (
    compute_dynamic_normal_loads,
    allocate_chassis_forces_to_wheels,
    evaluate_traction_and_slip,
    L_X, L_Y
)
from robot_model_amr_casadi import (
    PHI_MAX_X, PHI_MAX_Y,
    TANK_A, TANK_B, TANK_H, H_LIQUID,
    M_TOTAL
)
from test_sim_amr_closed_loop import (
    extract_map_from_model,
    load_combined_model,
    update_dynamic_obstacles,
    Lidar, OccupancyGrid, select_lidar_obstacle,
    draw_sphere, LIDAR_RANGE, L_THRESH, GRID_RES
)

# Đảm bảo in tiếng Việt chuẩn trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def plot_test3_traction_results(log, waypoints, start_pos, goal_pos, map_geoms, dyn_geoms=None, save_path=None, mu=0.35, mode='baseline'):
    """
    Vẽ biểu đồ phân tích 6 đồ thị chuyên sâu cho Kịch bản Test 3:
    1. Quỹ đạo X-Y và các vùng trượt bánh (Slip zones).
    2. Tải trọng pháp tuyến 4 bánh xe Fz_i(t) (Load Transfer).
    3. Lực yêu cầu vs Giới hạn bám ma sát mu*Fz (Traction Saturation).
    4. Hệ số trượt bánh xe (Wheel Slip Ratios).
    5. Góc sóng sánh phi_x, phi_y so với giới hạn an toàn.
    6. Sai số bám quỹ đạo (Contouring Error & Lag Error).
    """
    t = np.array(log['t'])
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('#f8f9fa')

    mode_title = "BASELINE MPCC (Traction-Unaware)" if mode == 'baseline' else "TRACTION-AWARE MPCC"
    fig.suptitle(f"KỊCH BẢN TEST 3: ĐỘNG LỰC HỌC TRƯỢT BÁNH MECANUM & GIỚI HẠN BÁM MA SÁT (slosh3.xml)\n"
                 f"Chế độ: {mode_title}   |   Hệ số ma sát sàn: mu = {mu:.2f}   |   Tổng thời gian: {t[-1]:.2f} s",
                 fontsize=14, fontweight='bold', y=0.985)

    # -------------------------------------------------------------
    # SUBPLOT 1: QUỸ ĐẠO TOÀN CỤC & VÙNG TRƯỢT BÁNH
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_facecolor('#ffffff')

    # Vẽ chướng ngại vật
    for gtype, pos, size, rgba, name in map_geoms:
        if gtype == 'cylinder':
            circle = plt.Circle((pos[0], pos[1]), size, color=rgba[:3], alpha=0.5, zorder=2)
            ax1.add_patch(circle)
        elif gtype == 'box':
            rect = plt.Rectangle((pos[0] - size[0], pos[1] - size[1]), 2*size[0], 2*size[1],
                                 color=rgba[:3], alpha=0.5, zorder=2)
            ax1.add_patch(rect)

    # Quỹ đạo tham chiếu A*
    if len(waypoints) > 0:
        wp = np.array(waypoints)
        ax1.plot(wp[:, 0], wp[:, 1], 'y--', lw=1.8, label='A* Global Path', zorder=3)

    # Quỹ đạo AMR thực tế
    ax1.plot(log['x'], log['y'], 'b-', lw=2.2, label='AMR Path', zorder=4)

    # Đánh dấu các vị trí có hiện tượng trượt bánh (màu đỏ)
    slip_mask = np.array(log['any_slip'])
    if np.any(slip_mask):
        ax1.scatter(np.array(log['x'])[slip_mask], np.array(log['y'])[slip_mask],
                    c='red', s=18, alpha=0.7, label='Wheel Slip Occurred', zorder=5)

    ax1.plot(start_pos[0], start_pos[1], 'go', markersize=9, label='Start')
    ax1.plot(goal_pos[0], goal_pos[1], 'r*', markersize=14, label='Goal')
    ax1.set_title('1. 2D Path & Slip Regions', fontsize=11, fontweight='bold')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.axis('equal')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower left', fontsize=8)

    # -------------------------------------------------------------
    # SUBPLOT 2: TẢI TRỌNG PHÁP TUYẾN 4 BÁNH XE Fz_i(t)
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set_facecolor('#ffffff')
    fz = np.array(log['fz_wheels']) # (N_steps, 4)
    ax2.plot(t, fz[:, 0], 'r-', lw=1.5, label='Fz_1 (Front-Left)')
    ax2.plot(t, fz[:, 1], 'g-', lw=1.5, label='Fz_2 (Front-Right)')
    ax2.plot(t, fz[:, 2], 'b-', lw=1.5, label='Fz_3 (Rear-Right)')
    ax2.plot(t, fz[:, 3], 'm-', lw=1.5, label='Fz_4 (Rear-Left)')
    ax2.axhline((M_TOTAL * 9.81)/4.0, color='gray', linestyle='--', lw=1.2, label='Static Fz (17.8N)')
    ax2.set_title('2. Dynamic Normal Load Transfer Fz(t)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Normal Force Fz (N)')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=8)

    # -------------------------------------------------------------
    # SUBPLOT 3: LỰC KÉO YÊU CẦU VS GIỚI HẠN BÁM mu*Fz
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_facecolor('#ffffff')
    f_demand = np.array(log['f_wheels_cmd'])
    f_limit = np.array(log['f_max_traction'])
    ax3.plot(t, np.max(np.abs(f_demand), axis=1), 'r-', lw=1.8, label='Max Demanded Wheel Force |f_cmd|')
    ax3.plot(t, np.min(f_limit, axis=1), 'k--', lw=1.8, label='Min Traction Limit min(mu*Fz)')
    ax3.fill_between(t, np.min(f_limit, axis=1), np.max(np.abs(f_demand), axis=1),
                     where=(np.max(np.abs(f_demand), axis=1) > np.min(f_limit, axis=1)),
                     color='red', alpha=0.25, label='Traction Saturation Zone')
    ax3.set_title('3. Force Demand vs. Traction Boundary', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Force (N)')
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(loc='upper right', fontsize=8)

    # -------------------------------------------------------------
    # SUBPLOT 4: HỆ SỐ TRƯỢT BÁNH XE (SLIP RATIOS)
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_facecolor('#ffffff')
    slips = np.array(log['slip_ratios']) # (N_steps, 4)
    ax4.plot(t, slips[:, 0], 'r-', lw=1.4, label='Slip Ratio Wheel 1 (FL)')
    ax4.plot(t, slips[:, 1], 'g-', lw=1.4, label='Slip Ratio Wheel 2 (FR)')
    ax4.plot(t, slips[:, 2], 'b-', lw=1.4, label='Slip Ratio Wheel 3 (RR)')
    ax4.plot(t, slips[:, 3], 'm-', lw=1.4, label='Slip Ratio Wheel 4 (RL)')
    ax4.set_title('4. Wheel Slip Ratio sigma_i(t) [0=No Slip, >0=Slipping]', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Slip Ratio')
    ax4.set_ylim(-0.05, 1.05)
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.legend(loc='upper right', fontsize=8)

    # -------------------------------------------------------------
    # SUBPLOT 5: GÓC SÓNG SÁNH SLOSHING PHI_X & PHI_Y
    # -------------------------------------------------------------
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_facecolor('#ffffff')
    phi_x_deg = np.degrees(log['phi_x'])
    phi_y_deg = np.degrees(log['phi_y'])
    ax5.plot(t, phi_x_deg, 'b-', lw=1.8, label='Phi_X (Longitudinal)')
    ax5.plot(t, phi_y_deg, 'g-', lw=1.8, label='Phi_Y (Transverse)')
    ax5.axhline(np.degrees(PHI_MAX_X), color='r', linestyle='--', lw=1.4, label=f'Limit X (+/-{np.degrees(PHI_MAX_X):.1f} deg)')
    ax5.axhline(-np.degrees(PHI_MAX_X), color='r', linestyle='--', lw=1.4)
    ax5.axhline(np.degrees(PHI_MAX_Y), color='darkorange', linestyle=':', lw=1.4, label=f'Limit Y (+/-{np.degrees(PHI_MAX_Y):.1f} deg)')
    ax5.axhline(-np.degrees(PHI_MAX_Y), color='darkorange', linestyle=':', lw=1.4)
    ax5.set_title('5. Sloshing Angles Under Wheel Traction', fontsize=11, fontweight='bold')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Angle (deg)')
    ax5.grid(True, linestyle=':', alpha=0.6)
    ax5.legend(loc='upper right', fontsize=8)

    # -------------------------------------------------------------
    # SUBPLOT 6: VẬN TỐC & SAI SỐ BÁM QUỸ ĐẠO
    # -------------------------------------------------------------
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_facecolor('#ffffff')
    ax6.plot(t, log['v_lin'], 'b-', lw=1.8, label='Linear Velocity (m/s)')
    ax6.plot(t, log['vs'], 'g--', lw=1.5, label='MPCC Progress vs (m/s)')
    drift_err = np.array(log['drift_err']) * 100.0 # cm
    ax6.plot(t, drift_err, 'r:', lw=1.6, label='Tracking Error (cm)')
    ax6.set_title('6. Robot Speed & Trajectory Tracking Drift', fontsize=11, fontweight='bold')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Speed (m/s) / Error (cm)')
    ax6.grid(True, linestyle=':', alpha=0.6)
    ax6.legend(loc='upper right', fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=250, bbox_inches='tight')
        print(f">> [TEST 3] Đã lưu thành công biểu đồ đồ thị kỹ thuật tại: {save_path}")

    return fig


def run_test3_traction_simulation(map_name=None, mu=0.35, mode='baseline', max_time=25.0, gui=False, save_path=None, no_plot=False):
    """
    Hàm thực thi Kịch bản Kiểm Chứng Test 3: Traction Limits & Wheel Slip.
    """
    print("=================================================================")
    print("KỊCH BẢN KIỂM CHỨNG TEST 3: TRACTION LIMITS & WHEEL SLIP")
    print(f"Chế độ: {mode.upper()}  |  Hệ số ma sát sàn: mu = {mu:.2f}")
    print("=================================================================")

    # 1. Nạp mô hình kết hợp (slosh3.xml làm chuẩn)
    model, active_map_name = load_combined_model(map_name)
    data = mujoco.MjData(model)

    dt_physics = model.opt.timestep # 0.005 s (200 Hz)
    nmpc_dt = 0.05                  # 0.05 s (20 Hz)
    substeps = int(round(nmpc_dt / dt_physics)) # 10 substeps

    print(f">> Bản đồ: {active_map_name} (Mô hình AMR từ slosh3.xml)")
    print(f">> Timestep vật lý: {dt_physics}s | Chu kỳ NMPC: {nmpc_dt}s ({substeps} substeps)")

    # 2. Quy hoạch đường đi toàn cục A*
    start_pos, goal_pos, obs_circles, obs_boxes, map_geoms, dyn_geoms = extract_map_from_model(model)
    print(f"Start: {start_pos} | Goal: {goal_pos}")

    waypoints = []
    for r_test in [0.26, 0.24, 0.22, 0.20]:
        planner = AStarPlanner(x_bounds=(-2.9, 2.9), y_bounds=(-2.9, 2.9), resolution=0.05, robot_radius=r_test)
        planner.add_obstacles(obs_circles, obs_boxes)
        waypoints = planner.plan(start_pos=start_pos, goal_pos=goal_pos, smooth=True)
        if len(waypoints) > 5:
            print(f"A* đã tìm thấy quỹ đạo gồm {len(waypoints)} điểm (Bán kính robot: {r_test}m).")
            break

    if len(waypoints) <= 5:
        raise RuntimeError(f"Không thể tìm thấy đường đi từ Start đến Goal trên bản đồ {active_map_name}!")

    # Khởi tạo bộ điều khiển theo chế độ
    if mode in ['traction-aware', 'dfe', 'dfempcc']:
        print(f">> [BỘ ĐIỀU KHIỂN] Khởi tạo PROPOSED DFE-MPCC (Traction-Aware, mu={mu:.2f})")
        controller = DFEMPCCAMRController(N=15, dt=nmpc_dt, mu=mu)
    else:
        print(">> [BỘ ĐIỀU KHIỂN] Khởi tạo BASELINE MPCC (Traction-Unaware)")
        controller = MPCCNMPCAMRController(N=15, dt=nmpc_dt)
    controller.set_path(waypoints)

    # 3. Khởi tạo vị trí ban đầu
    mujoco.mj_resetData(model, data)
    update_dynamic_obstacles(model, data, 0.0)
    data.qpos[0] = start_pos[0]
    data.qpos[1] = start_pos[1]
    p_init, t_init, heading_init = controller.get_path_reference(0.0)
    data.qpos[2] = heading_init
    mujoco.mj_forward(model, data)

    # Hệ số PD hồi tiếp cấp thấp
    kp_diag = np.array([120.0, 120.0, 15.0])
    kd_diag = np.array([45.0,  45.0,   2.0])
    M_full = np.zeros((model.nv, model.nv))

    # Cảm biến LiDAR
    lidar = Lidar(model)
    grid = OccupancyGrid()

    # Khởi tạo buffer logging Test 3
    log = {
        't': [], 'x': [], 'y': [], 'yaw': [], 'yaw_ref': [],
        'vx': [], 'vy': [], 'v_lin': [], 'vs': [],
        'ax': [], 'ay': [],
        'phi_x': [], 'phi_y': [],
        'eta_x_mm': [], 'eta_y_mm': [],
        'f_wheels_cmd': [], 'f_wheels_act': [],
        'f_max_traction': [], 'fz_wheels': [],
        'slip_ratios': [], 'any_slip': [],
        'drift_err': [], 'solve_ms': [], 'min_range': []
    }

    a_act = np.zeros(3)
    s_prog = 0.0
    vs_prog = 0.2
    as_prog = 0.0
    sim_time = 0.0
    max_steps = int(max_time / nmpc_dt)
    goal_reached = False

    viewer = None
    if gui:
        try:
            viewer = mujoco.viewer.launch_passive(model, data)
            print(">> Đã khởi động giao diện đồ họa 3D (MuJoCo Viewer) <<")
        except Exception as e:
            print(f"Cảnh báo: Không thể khởi chạy 3D Viewer ({e}). Chuyển sang console mode.")
            viewer = None

    print(f"\nBắt đầu mô phỏng Test 3: Traction & Slip ({max_steps} chu kỳ, T_max = {max_time} s)...")
    t_start_sim = time.time()

    for cycle in range(max_steps):
        if viewer is not None and not viewer.is_running():
            break

        # 1. Cập nhật vị trí vật thể động & Đọc trạng thái đo
        update_dynamic_obstacles(model, data, sim_time)
        mujoco.mj_forward(model, data)

        q_meas = data.qpos[:3].copy()
        v_meas = data.qvel[:3].copy()
        phi_x_meas = float(data.qpos[3])
        dphi_x_meas = float(data.qvel[3])
        phi_y_meas = float(data.qpos[4])
        dphi_y_meas = float(data.qvel[4])
        phi_est = np.array([phi_x_meas, dphi_x_meas, phi_y_meas, dphi_y_meas])

        # Kiểm tra điều kiện đích
        dist_to_goal = np.hypot(q_meas[0] - goal_pos[0], q_meas[1] - goal_pos[1])
        dist_to_path_end = np.hypot(q_meas[0] - waypoints[-1][0], q_meas[1] - waypoints[-1][1]) if len(waypoints) > 0 else dist_to_goal
        if (dist_to_goal < 0.18 or dist_to_path_end < 0.15) and s_prog > controller.path_length - 0.35:
            eff_dist = min(dist_to_goal, dist_to_path_end)
            print(f"\n>>> ĐÃ ĐẾN ĐÍCH THÀNH CÔNG tại chu kỳ {cycle+1} (t = {sim_time:.2f} s)! Sai số vị trí = {eff_dist*100:.1f} cm <<<")
            goal_reached = True
            break

        # 2. Xây dựng vector trạng thái NMPC
        x_nmpc = np.zeros(12)
        x_nmpc[0:3] = q_meas
        x_nmpc[3:6] = v_meas
        x_nmpc[6:9] = a_act
        x_nmpc[9] = s_prog
        x_nmpc[10] = vs_prog
        x_nmpc[11] = as_prog

        # 3. Quét LiDAR
        origin, dirs, meas, hit, raw = lidar.scan(model, data)
        grid.update(origin, dirs, meas, hit)
        scan_pts = origin[:2] + dirs[hit, :2] * meas[hit, None]
        min_range = float(raw.min())
        occ_pts = grid.occupied_points()
        closest_obs, min_d = select_lidar_obstacle(scan_pts, occ_pts, q_meas, v_meas, s_prog, controller)

        # 4. Giải tối ưu NMPC
        if mode in ['traction-aware', 'dfe', 'dfempcc']:
            u_opt, js_opt, x_pred, u_pred, info = controller.solve_step(x_nmpc, phi_est, closest_obs, mu_floor=mu)
        else:
            u_opt, js_opt, x_pred, u_pred, info = controller.solve_step(x_nmpc, phi_est, closest_obs)

        # Cập nhật tiến độ MPCC
        s_prog = float(x_pred[9, 1])
        vs_prog = float(x_pred[10, 1])
        as_prog = float(x_pred[11, 1])
        a_act = np.array(x_pred[6:9, 1])

        # 5. Mô phỏng vật lý đa bước tích phân với mô hình bão hòa lực bám Traction
        q0_cyc = q_meas.copy()
        v0_cyc = v_meas.copy()
        a0_cyc = a_act.copy()
        u0_cyc = u_opt[0:3].copy()

        # Biến theo dõi lực và trượt bánh trong chu kỳ con
        last_fz = np.zeros(4)
        last_f_cmd_w = np.zeros(4)
        last_f_act_w = np.zeros(4)
        last_f_max_tr = np.zeros(4)
        last_slip_ratios = np.zeros(4)
        last_any_slip = False

        for sub in range(substeps):
            tau_sub = (sub + 1) * dt_physics
            t_curr = sim_time + dt_physics
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

            # Lực yêu cầu trên thân xe từ vòng điều khiển phản hồi PD
            F_cmd_world = F_ff + fric_ff + kp_diag * err_q + kd_diag * err_v + F_bias

            # Chuyển lực yêu cầu sang hệ thân xe (Body Frame)
            th_now = q_now[2]
            c_th, s_th = np.cos(th_now), np.sin(th_now)
            F_cmd_bx = c_th * F_cmd_world[0] + s_th * F_cmd_world[1]
            F_cmd_by = -s_th * F_cmd_world[0] + c_th * F_cmd_world[1]
            Tau_cmd_bz = F_cmd_world[2]
            F_cmd_body = np.array([F_cmd_bx, F_cmd_by, Tau_cmd_bz])

            # Gia tốc thân xe và lực phản kháng FSI
            a_bx = c_th * a_act[0] + s_th * a_act[1]
            a_by = -s_th * a_act[0] + c_th * a_act[1]
            a_body_now = np.array([a_bx, a_by])

            # Phản lực FSI từ con lắc sóng sánh
            F_react_bx = -0.5412 * (a_bx + 0.0512 * (-13.84**2 * phi_x_meas))
            F_react_by = -0.3820 * (a_by + 0.0323 * (-17.44**2 * phi_y_meas))
            f_react_body = np.array([F_react_bx, F_react_by])

            # Tính toán phân bố tải trọng pháp tuyến tức thời 4 bánh xe Fz_i
            Fz_wheels, _ = compute_dynamic_normal_loads(M_TOTAL, a_body_now, f_react_body)

            # Phân bổ lực thân xe yêu cầu sang 4 bánh xe Mecanum
            f_wheels_cmd = allocate_chassis_forces_to_wheels(F_cmd_body)

            # Đánh giá giới hạn bám ma sát và trượt bánh
            f_wheels_act, f_chassis_act_body, slip_metrics = evaluate_traction_and_slip(
                f_wheels_cmd, Fz_wheels, mu=mu
            )

            last_fz = Fz_wheels
            last_f_cmd_w = f_wheels_cmd
            last_f_act_w = f_wheels_act
            last_f_max_tr = slip_metrics['f_max_traction']
            last_slip_ratios = slip_metrics['slip_ratios']
            last_any_slip = slip_metrics['any_slip']

            # Chuyển lực thực tế từ tiếp xúc sàn về World Frame để nạp vào MuJoCo actuator:
            # - Với Baseline: lực yêu cầu vượt quá mu*Fz nên bị bão hòa cắt lực (trượt bánh)
            # - Với DFE-MPCC: bộ điều khiển chủ động khống chế f_wheels <= mu*Fz nên truyền lực trọn vẹn (không trượt)
            F_act_world_x = c_th * f_chassis_act_body[0] - s_th * f_chassis_act_body[1]
            F_act_world_y = s_th * f_chassis_act_body[0] + c_th * f_chassis_act_body[1]
            Tau_act_world_z = f_chassis_act_body[2]
            F_cmd_applied = np.array([F_act_world_x, F_act_world_y, Tau_act_world_z])

            # Giới hạn an toàn động cơ
            F_cmd_applied[0:2] = np.clip(F_cmd_applied[0:2], -30.0, 30.0)
            F_cmd_applied[2] = np.clip(F_cmd_applied[2], -10.0, 10.0)

            data.ctrl[:3] = F_cmd_applied
            mujoco.mj_step(model, data)
            sim_time += dt_physics

        # Trực quan hóa 3D
        if viewer is not None:
            viewer.user_scn.ngeom = 0
            viewer.cam.lookat[:] = [q_meas[0] * 0.5, q_meas[1] * 0.5, 0.1]
            if len(waypoints) > 0:
                for pt in waypoints[::3]:
                    draw_sphere(viewer, np.array([pt[0], pt[1], 0.02]), [1.0, 1.0, 0.0, 0.6], size=0.025)
            draw_sphere(viewer, np.array([goal_pos[0], goal_pos[1], 0.05]), [0.0, 0.9, 0.2, 0.8], size=0.07)

            # Quỹ đạo dự đoán MPCC (màu xanh lục sáng)
            for i in range(controller.N):
                draw_sphere(viewer, np.array([x_pred[0, i], x_pred[1, i], 0.05]), [0.0, 1.0, 0.0, 0.8], size=0.02)

            # Tia LiDAR quét phản xạ (màu cyan)
            for p in scan_pts:
                draw_sphere(viewer, np.array([p[0], p[1], 0.35]), [0.0, 1.0, 1.0, 0.8], size=0.015)

            # Nếu đang bị trượt bánh: vẽ điểm cảnh báo màu đỏ rực ngay trên xe
            if last_any_slip:
                draw_sphere(viewer, np.array([q_meas[0], q_meas[1], 0.35]), [1.0, 0.0, 0.0, 0.9], size=0.05)
            viewer.sync()

        # Logging dữ liệu
        p_ref_now, _, heading_ref = controller.get_path_reference(s_prog)
        drift_distance = np.hypot(q_meas[0] - p_ref_now[0], q_meas[1] - p_ref_now[1])
        eta_x = 0.5 * TANK_A * np.tan(phi_x_meas) * 1000.0
        eta_y = 0.5 * TANK_B * np.tan(phi_y_meas) * 1000.0
        v_lin = float(np.hypot(v_meas[0], v_meas[1]))

        log['t'].append(sim_time)
        log['x'].append(q_meas[0])
        log['y'].append(q_meas[1])
        log['yaw'].append(q_meas[2])
        log['yaw_ref'].append(heading_ref)
        log['vx'].append(v_meas[0])
        log['vy'].append(v_meas[1])
        log['v_lin'].append(v_lin)
        log['vs'].append(vs_prog)
        log['ax'].append(a_act[0])
        log['ay'].append(a_act[1])
        log['phi_x'].append(phi_x_meas)
        log['phi_y'].append(phi_y_meas)
        log['eta_x_mm'].append(eta_x)
        log['eta_y_mm'].append(eta_y)
        log['f_wheels_cmd'].append(last_f_cmd_w)
        log['f_wheels_act'].append(last_f_act_w)
        log['f_max_traction'].append(last_f_max_tr)
        log['fz_wheels'].append(last_fz)
        log['slip_ratios'].append(last_slip_ratios)
        log['any_slip'].append(last_any_slip)
        log['drift_err'].append(drift_distance)
        log['solve_ms'].append(info['solve_time_ms'])
        log['min_range'].append(min_range)

        if (cycle + 1) % 10 == 0 or cycle == 0:
            slip_str = "SLIP!" if last_any_slip else "OK"
            print(f"Cycle {cycle+1:3d} (t={sim_time:.2f}s): Pos=[{q_meas[0]:.2f}, {q_meas[1]:.2f}] | "
                  f"v={v_lin:.2f}m/s | Slosh=[{np.degrees(phi_x_meas):.1f}°, {np.degrees(phi_y_meas):.1f}°] | "
                  f"Min_Fz={np.min(last_fz):.1f}N | Traction=[{slip_str}] | Drift={drift_distance*100:.1f}cm")

    # 4. Tổng kết số liệu Benchmark Test 3
    total_time = sim_time
    slip_array = np.array(log['any_slip'])
    pct_slipping = (np.sum(slip_array) / len(slip_array)) * 100.0 if len(slip_array) > 0 else 0.0
    all_slips = np.array(log['slip_ratios'])
    peak_slip = float(np.max(all_slips)) if len(all_slips) > 0 else 0.0
    all_fz = np.array(log['fz_wheels'])
    min_fz = float(np.min(all_fz)) if len(all_fz) > 0 else 0.0
    max_drift = float(np.max(log['drift_err'])) * 100.0
    avg_drift = float(np.mean(log['drift_err'])) * 100.0
    max_phi_x = np.degrees(np.max(np.abs(log['phi_x'])))
    max_phi_y = np.degrees(np.max(np.abs(log['phi_y'])))
    max_eta_x = np.max(np.abs(log['eta_x_mm']))
    max_eta_y = np.max(np.abs(log['eta_y_mm']))
    freeboard_mm = (TANK_H - H_LIQUID) * 1000.0

    print("\n=================================================================")
    print(f"KẾT QUẢ BENCHMARK TEST 3: TRACTION LIMITS & WHEEL SLIP (mu={mu:.2f})")
    print("=================================================================")
    print(f"• Trạng thái hoàn thành: {'VỀ ĐÍCH THÀNH CÔNG' if goal_reached else 'CHƯA CHẠM ĐÍCH'}")
    print(f"• Tổng thời gian di chuyển: {total_time:.2f} s")
    print(f"• Tỷ lệ thời gian xảy ra trượt bánh: {pct_slipping:.1f}% tổng hành trình")
    print(f"• Hệ số trượt bánh xe cực đại: sigma_max = {peak_slip:.3f} (1.0 = trượt hoàn toàn)")
    print(f"• Tải trọng pháp tuyến bánh xe nhỏ nhất: Fz_min = {min_fz:.2f} N (Danh định tĩnh: 17.78 N)")
    print(f"• Mức độ giảm tải trọng do mô-men lật: {(1.0 - min_fz/17.78)*100.0:.1f}%")
    print(f"• Sai số lệch quỹ đạo lớn nhất do trượt (Drift): {max_drift:.1f} cm (Trung bình: {avg_drift:.1f} cm)")
    print(f"• Góc sóng sánh cực đại: Phi_X = {max_phi_x:.2f}° (Limit ±{np.degrees(PHI_MAX_X):.1f}°), Phi_Y = {max_phi_y:.2f}° (Limit ±{np.degrees(PHI_MAX_Y):.1f}°)")
    print(f"• Sóng dâng cực đại tại vách: Eta_X = {max_eta_x:.1f} mm, Eta_Y = {max_eta_y:.1f} mm (Mép an toàn: {freeboard_mm:.0f} mm)")
    print(f"• Tình trạng tràn chất lỏng: {'AN TOÀN TUYỆT ĐỐI' if (max_eta_x < freeboard_mm and max_eta_y < freeboard_mm) else 'CẢNH BÁO TRÀN NƯỚC'}")
    print("=================================================================")

    # 5. Lưu biểu đồ
    workspace_dir = _root
    bm_folder = os.path.join(workspace_dir, "testbm")
    os.makedirs(bm_folder, exist_ok=True)

    if not save_path:
        map_tag = os.path.splitext(os.path.basename(active_map_name))[0]
        save_path = os.path.join(bm_folder, f"benchmark_test3_{map_tag}_mu{int(mu*100)}_{mode}.png")

    if not no_plot:
        plot_test3_traction_results(log, waypoints, start_pos, goal_pos, map_geoms, dyn_geoms, save_path=save_path, mu=mu, mode=mode)
        plt.show()
    elif save_path:
        plot_test3_traction_results(log, waypoints, start_pos, goal_pos, map_geoms, dyn_geoms, save_path=save_path, mu=mu, mode=mode)

    return log, goal_reached


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kịch bản Test 3: Động lực học trượt bánh Mecanum & Giới hạn bám.")
    parser.add_argument("--map", type=str, default="slosh3.xml", help="Tên file bản đồ XML.")
    parser.add_argument("--mu", type=float, default=0.35, help="Hệ số ma sát sàn (mặc định 0.35).")
    parser.add_argument("--mode", type=str, choices=['baseline', 'traction-aware'], default='baseline', help="Chế độ chạy.")
    parser.add_argument("--max-time", type=float, default=20.0, help="Thời gian mô phỏng tối đa (s).")
    parser.add_argument("--gui", action="store_true", help="Hiển thị giao diện đồ họa 3D MuJoCo.")
    parser.add_argument("--no-plot", action="store_true", help="Không hiển thị cửa sổ đồ thị Matplotlib.")
    parser.add_argument("--save-path", type=str, default=None, help="Đường dẫn lưu file ảnh kết quả.")

    args = parser.parse_args()
    run_test3_traction_simulation(
        map_name=args.map,
        mu=args.mu,
        mode=args.mode,
        max_time=args.max_time,
        gui=args.gui,
        save_path=args.save_path,
        no_plot=args.no_plot
    )
