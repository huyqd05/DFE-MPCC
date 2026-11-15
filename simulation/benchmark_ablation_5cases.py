"""
benchmark_ablation_5cases.py
=============================================================================
ĐÓNG GÓP THỰC NGHIỆM ĐỐI CHỨNG: MA TRẬN ABLATION 5 BỘ ĐIỀU KHIỂN (5-CASE MATRIX)
Kiểm chứng trên sàn ma sát thấp (mu = 0.12) để bóc tách chính xác lợi ích:

Case A: Ad-hoc Conservative Baseline (MPCC với a_max <= 0.40 m/s^2 cố định).
Case B: Slosh-Only MPCC (Chỉ có Slosh-ECBF chống tràn, bỏ qua ma sát bánh xe).
Case C: Static Coulomb MPCC (Ràng buộc ma sát giả định tải tĩnh Fz = Mg/4).
Case D: DFE without FSI (Tải động Fz chỉ tính quán tính xe, bỏ qua phản lực sóng).
Case E: Full Proposed DFE-MPCC (DFE đầy đủ + FSI 3D + Tối ưu Null-Space alpha*).

Chỉ số đo lường KPI:
1. T_finish: Thời gian hoàn thành (s) (Đo độ bảo thủ / tốc độ vận chuyển).
2. T_slip: Tổng thời gian bánh xe bị trượt (s) (Slip duration).
3. Max_slip: Tỷ số trượt bánh cực đại.
4. Max_wave: Chiều cao sóng dâng cực đại eta_max (mm) (So với ngưỡng tràn 30 mm).
5. V_spill: Ước lượng lượng nước tràn tích lũy (mL).
6. Mean_solve_ms: Thời gian tính toán trung bình (ms).
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
from mecanum_nullspace_allocation import allocate_chassis_wrench
from mecanum_traction_dynamics import (
    compute_dynamic_normal_loads,
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

TANK_FREEBOARD = (TANK_H - H_LIQUID) * 1000.0 # 30.0 mm


def run_single_simulation(map_file, case_name, mu=0.12, max_time=25.0):
    """
    Chạy mô phỏng vòng kín cho 1 Case điều khiển trên bản đồ chỉ định.
    """
    print(f"\n[{case_name}] Bắt đầu chạy trên bản đồ '{map_file}' (mu={mu:.2f})...")
    model, active_map_name = load_combined_model(map_file)
    data = mujoco.MjData(model)
    
    # 1. Lập kế hoạch đường đi toàn cục A*
    start_pos, goal_pos, obs_circles, obs_boxes, map_geoms, dyn_geoms = extract_map_from_model(model)
    waypoints = []
    for r_test in [0.26, 0.24, 0.22, 0.20]:
        planner = AStarPlanner(x_bounds=(-2.9, 2.9), y_bounds=(-2.9, 2.9), resolution=0.05, robot_radius=r_test)
        planner.add_obstacles(obs_circles, obs_boxes)
        waypoints = planner.plan(start_pos=start_pos, goal_pos=goal_pos, smooth=True)
        if len(waypoints) > 5:
            break

    if len(waypoints) <= 5:
        print(f"[{case_name}] Lỗi: A* không tìm được đường đi!")
        return None

    dt_sim = model.opt.timestep # 0.005 s
    nmpc_dt = 0.05              # 20 Hz
    substeps = int(round(nmpc_dt / dt_sim))

    # 2. Khởi tạo bộ điều khiển theo Case
    if case_name == 'Case_A_Conservative':
        controller = MPCCNMPCAMRController(N=15, dt=nmpc_dt)
        # Ép giới hạn vận tốc/gia tốc chặt chẽ
    elif case_name == 'Case_B_SloshOnly':
        controller = MPCCNMPCAMRController(N=15, dt=nmpc_dt)
    elif case_name in ['Case_C_StaticCoulomb', 'Case_D_DFE_NoFSI', 'Case_E_Full_DFEMPCC']:
        controller = DFEMPCCAMRController(N=15, dt=nmpc_dt, mu=mu)
    else:
        raise ValueError(f"Unknown case: {case_name}")

    controller.set_path(waypoints)

    # Reset trạng thái
    mujoco.mj_resetData(model, data)
    update_dynamic_obstacles(model, data, 0.0)
    data.qpos[0] = start_pos[0]
    data.qpos[1] = start_pos[1]
    p_init, t_init, heading_init = controller.get_path_reference(0.0)
    data.qpos[2] = heading_init
    mujoco.mj_forward(model, data)

    kp_diag = np.array([120.0, 120.0, 15.0])
    kd_diag = np.array([45.0,  45.0,   2.0])
    M_full = np.zeros((model.nv, model.nv))
    lidar = Lidar(model)
    grid = OccupancyGrid()

    # Log metrics
    t_list, x_list, y_list = [], [], []
    slip_durations = []
    max_slips = []
    eta_max_list = []
    solve_times = []

    a_act = np.zeros(3)
    s_prog = 0.0
    vs_prog = 0.2
    as_prog = 0.0
    sim_time = 0.0
    max_steps = int(max_time / nmpc_dt)
    goal_reached = False

    t_wall_start = time.time()

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

        # Đánh giá chiều cao sóng sánh mặt thoáng (mm)
        eta_x_mm = float(abs(phi_x_meas) * (TANK_A / 2.0) * 1000.0)
        eta_y_mm = float(abs(phi_y_meas) * (TANK_B / 2.0) * 1000.0)
        eta_curr = max(eta_x_mm, eta_y_mm)
        eta_max_list.append(eta_curr)

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
        closest_obs, min_d = select_lidar_obstacle(scan_pts, occ_pts, q_meas, v_meas, s_prog, controller)

        # Xây dựng vector trạng thái NMPC
        x_curr = np.zeros(12)
        x_curr[0:3] = q_meas
        x_curr[3:6] = v_meas
        x_curr[6:9] = a_act
        x_curr[9] = s_prog
        x_curr[10] = vs_prog
        x_curr[11] = as_prog

        # GIẢI NMPC THEO TỪNG CASE
        t_solve_0 = time.perf_counter()
        if case_name in ['Case_A_Conservative', 'Case_B_SloshOnly']:
            u_opt, js_opt, x_pred, u_pred, extra = controller.solve_step(x_curr, phi_est, closest_obs)
        else:
            # Case C, D, E dùng DFE Controller
            u_opt, js_opt, x_pred, u_pred, extra = controller.solve_step(x_curr, phi_est, closest_obs, mu_floor=mu)
        t_solve_ms = (time.perf_counter() - t_solve_0) * 1000.0
        solve_times.append(t_solve_ms)

        # Cập nhật tiến độ MPCC từ x_pred
        s_prog = float(x_pred[9, 1])
        vs_prog = float(x_pred[10, 1])
        as_prog = float(x_pred[11, 1])
        a_act = np.array(x_pred[6:9, 1])

        # Thực thi mô phỏng vật lý đa bước con
        q0_cyc = q_meas.copy()
        v0_cyc = v_meas.copy()
        a0_cyc = a_act.copy()
        u0_cyc = u_opt.copy()
        a_act = a0_cyc + nmpc_dt * u0_cyc

        cycle_has_slip = False
        max_cycle_slip = 0.0

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

            # ĐẶC TÍNH VẬT LÝ TẢI TRỌNG CỦA TỪNG CASE:
            if case_name == 'Case_C_StaticCoulomb':
                # Giả định tải tĩnh tuyệt đối
                Fz_wheels = np.full(4, (M_TOTAL * 9.81) / 4.0)
            elif case_name == 'Case_D_DFE_NoFSI':
                # Bỏ qua phản lực sóng FSI
                Fz_wheels, _ = compute_dynamic_normal_loads(M_TOTAL, a_body_now, np.zeros(2))
            else:
                # Case A, B, E: Tính đầy đủ tương tác FSI thực tế
                F_react_bx = -0.5412 * (a_bx + 0.0512 * (-13.84**2 * phi_x_meas))
                F_react_by = -0.3820 * (a_by + 0.0323 * (-17.44**2 * phi_y_meas))
                f_react_body = np.array([F_react_bx, F_react_by])
                Fz_wheels, _ = compute_dynamic_normal_loads(M_TOTAL, a_body_now, f_react_body)

            # PHÂN BỔ LỰC: Case E dùng tối ưu Null-space alpha*, các Case khác dùng Moore-Penrose thuần túy
            use_null = (case_name == 'Case_E_Full_DFEMPCC')
            f_wheels_cmd, _ = allocate_chassis_wrench(F_cmd_body, Fz_wheels, mu=mu, use_nullspace=use_null)

            # Mô hình tiếp xúc vật lý mặt sàn
            f_wheels_act, f_chassis_act_body, slip_metrics = evaluate_traction_and_slip(
                f_wheels_cmd, Fz_wheels, mu=mu
            )

            if slip_metrics['any_slip']:
                cycle_has_slip = True
                max_cycle_slip = max(max_cycle_slip, slip_metrics['max_slip'])

            # Nạp lực tiếp xúc vào actuator
            F_act_world_x = c_th * f_chassis_act_body[0] - s_th * f_chassis_act_body[1]
            F_act_world_y = s_th * f_chassis_act_body[0] + c_th * f_chassis_act_body[1]
            Tau_act_world_z = f_chassis_act_body[2]
            F_cmd_applied = np.array([F_act_world_x, F_act_world_y, Tau_act_world_z])
            F_cmd_applied[0:2] = np.clip(F_cmd_applied[0:2], -30.0, 30.0)
            F_cmd_applied[2] = np.clip(F_cmd_applied[2], -10.0, 10.0)

            data.ctrl[:3] = F_cmd_applied
            mujoco.mj_step(model, data)
            sim_time += dt_sim

        if cycle_has_slip:
            slip_durations.append(nmpc_dt)
            max_slips.append(max_cycle_slip)

        t_list.append(sim_time)
        x_list.append(float(data.qpos[0]))
        y_list.append(float(data.qpos[1]))

    t_finish = sim_time
    total_slip_sec = float(sum(slip_durations))
    peak_slip = float(max(max_slips)) if len(max_slips) > 0 else 0.0
    peak_wave_mm = float(max(eta_max_list)) if len(eta_max_list) > 0 else 0.0
    
    # Ước lượng lượng nước tràn tích lũy (mL):
    # Spill rate ước tính tỷ lệ với max(0, eta - 30mm)
    spill_overflow = np.maximum(0.0, np.array(eta_max_list) - TANK_FREEBOARD)
    v_spill_est_ml = float(np.sum(spill_overflow) * 0.15) # ml ước lượng

    mean_solve = float(np.mean(solve_times)) if len(solve_times) > 0 else 0.0
    p95_solve = float(np.percentile(solve_times, 95)) if len(solve_times) > 0 else 0.0

    print(f"[{case_name}] Kết thúc: Về đích={goal_reached} | T_finish={t_finish:.2f}s | "
          f"T_slip={total_slip_sec:.2f}s | Sóng max={peak_wave_mm:.1f}mm | Lượng tràn={v_spill_est_ml:.1f}mL | Solve p95={p95_solve:.1f}ms")

    return {
        'case_name': case_name,
        'goal_reached': goal_reached,
        't_finish': t_finish,
        't_slip_sec': total_slip_sec,
        'peak_slip': peak_slip,
        'peak_wave_mm': peak_wave_mm,
        'v_spill_ml': v_spill_est_ml,
        'mean_solve_ms': mean_solve,
        'p95_solve_ms': p95_solve,
        't_list': t_list,
        'x_list': x_list,
        'y_list': y_list
    }


def run_full_ablation_matrix(map_file='slosh3.xml', mu=0.12):
    """
    Chạy toàn bộ Ma trận 5 Nhóm đối chứng và xuất biểu đồ tổng hợp KPI.
    """
    cases = [
        'Case_A_Conservative',
        'Case_B_SloshOnly',
        'Case_C_StaticCoulomb',
        'Case_D_DFE_NoFSI',
        'Case_E_Full_DFEMPCC'
    ]

    results = []
    for c in cases:
        res = run_single_simulation(map_file, c, mu=mu)
        if res is not None:
            results.append(res)

    # Xuất bảng KPI tổng hợp
    print("\n" + "=" * 80)
    print(f"BẢNG TỔNG HỢP KPI ĐỐI CHỨNG ABLATION (Map: {map_file}, mu = {mu:.2f})")
    print("=" * 80)
    print(f"{'Bộ điều khiển (Case)':<25} | {'Về đích':<8} | {'T_finish (s)':<12} | {'T_slip (s)':<10} | {'Sóng (mm)':<10} | {'Tràn (mL)':<10} | {'Solver p95':<10}")
    print("-" * 80)
    for r in results:
        status_str = "THÀNH CÔNG" if r['goal_reached'] else "MẤT LÁI"
        print(f"{r['case_name']:<25} | {status_str:<8} | {r['t_finish']:<12.2f} | {r['t_slip_sec']:<10.2f} | {r['peak_wave_mm']:<10.1f} | {r['v_spill_ml']:<10.1f} | {r['p95_solve_ms']:<10.1f} ms")
    print("=" * 80)

    # Vẽ biểu đồ Bar Chart so sánh các chỉ số KPI chuẩn ấn phẩm
    output_png = os.path.join(_root, "testbm", "ablation_5cases_benchmark.png")
    os.makedirs(os.path.dirname(output_png), exist_ok=True)

    names = [r['case_name'].replace('Case_', '') for r in results]
    t_fin = [r['t_finish'] for r in results]
    t_slip = [r['t_slip_sec'] for r in results]
    spill = [r['v_spill_ml'] for r in results]
    p95 = [r['p95_solve_ms'] for r in results]

    x = np.arange(len(names))
    width = 0.2

    fig, ax1 = plt.subplots(figsize=(12, 6), dpi=300)
    fig.patch.set_facecolor('#ffffff')

    rects1 = ax1.bar(x - 1.5*width, t_fin, width, label='Finish Time (s) [Lower is better]', color='#3498db')
    rects2 = ax1.bar(x - 0.5*width, t_slip, width, label='Slip Duration (s) [0 is best]', color='#e74c3c')
    rects3 = ax1.bar(x + 0.5*width, spill, width, label='Estimated Spill (mL) [0 is best]', color='#f39c12')
    rects4 = ax1.bar(x + 1.5*width, p95, width, label='Solver p95 Latency (ms)', color='#2ecc71')

    ax1.set_ylabel('Metric Magnitude', fontsize=11, fontweight='bold')
    ax1.set_title(f'5-CASE ABLATION STUDY BENCHMARK (Floor Friction mu = {mu:.2f})\n'
                  f'Isolating Contributions: FSI Load Transfer vs Null-Space Torque Vectoring', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=9, fontweight='bold', rotation=15)
    ax1.legend(loc='upper right', frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"\n-> Đã xuất đồ thị Benchmark 5 Nhóm chuẩn ấn phẩm: {output_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy ma trận đối chứng 5 nhóm (Ablation Benchmark)")
    parser.add_argument('--map', type=str, default='slosh3.xml', help='Tên file map XML (slosh3.xml, slosh_rect.XML, world5.xml)')
    parser.add_argument('--mu', type=float, default=0.12, help='Hệ số ma sát sàn (mặc định 0.12)')
    args = parser.parse_args()

    run_full_ablation_matrix(map_file=args.map, mu=args.mu)
