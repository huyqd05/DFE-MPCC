# -*- coding: utf-8 -*-
"""
visualize_world5_dynamic_obs_gui.py
=============================================================================
Trực quan hóa trực tiếp trên Giao diện Đồ họa 3D (MuJoCo Passive GUI Viewer)
cho bộ Benchmark DFE-MPCC trên Bản đồ Cơ sở (world5.xml) theo đúng quy trình 2 PHA:

1. PHA 1 (--phase 1): Bản đồ chỉ có tường (world5.xml, 0 vật cản)
   - Hoạch định đường đi toàn cục A* (L = 11.078 m).
   - Hiển thị 6 vị trí Checkpoint SPH (CP1..CP6) quả cầu cam nổi bật trên hành lang A*.
   - Cho robot chạy DFE-MPCC danh định trên nền địa hình ISO 8608 Class B.

2. PHA 2 (--phase 2): Sau khi đã có Checkpoint, tiêm 2 vật cản tĩnh + 1 vật cản động
   - Giữ nguyên đường A* và biên bù Conformal q_eta(s) = 2.59 mm từ Pha 1.
   - Tiêm 2 vật cản tĩnh (Hành lang Tây, Giữa) + 1 vật cản động (Hành lang Đông).
   - Chạy bộ điều khiển đối kháng (proposed vs baseline) với thuật toán né tránh va chạm.
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
from test_sim_amr_closed_loop import (
    load_combined_model,
    extract_map_from_model,
    update_dynamic_obstacles,
    Lidar, OccupancyGrid, select_lidar_obstacle
)
from benchmark_dfempcc_t1t2_vs_sph_iso import (
    TANK_A, TANK_B, H_LIQUID, TANK_FREEBOARD,
    compute_nonlinear_slosh_elevation_mm
)
from conformal_checkpoint_eval import load_conformal_calibration_budget

_CAL_BUDGET = load_conformal_calibration_budget()


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


def draw_cylinder(viewer, pos, radius, height, rgba):
    """Vẽ hình trụ vật cản trên viewer."""
    if viewer is None or viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        return
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    geom.size[:] = [radius, height * 0.5, 0]
    geom.pos[:] = pos
    geom.rgba[:] = rgba
    geom.mat[:] = np.eye(3)
    viewer.user_scn.ngeom += 1


def run_world5_gui_simulation(phase=2, controller_mode="proposed", speed_factor=1.0, max_time=28.0):
    print("=" * 80)
    print(f" KHỞI ĐỘNG MÔ PHỎNG 3D GUI WORLD5 (BẢN ĐỒ CƠ SỞ): PHA {phase}")
    print(f" Chế độ: {'BẢN ĐỒ CHỈ CÓ TƯỜNG (0 VẬT CẢN) & 6 CHECKPOINT SPH' if phase == 1 else 'TIÊM 2 VẬT CẢN TĨNH + 1 VẬT CẢN ĐỘNG'}")
    print(f" Bộ điều khiển: {controller_mode.upper()}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # BƯỚC 1: Khởi tạo bản đồ world5.xml chỉ có tường
    # -------------------------------------------------------------------------
    print("\n[BƯỚC 1] Khởi tạo bản đồ world5.xml (Chỉ có tường bao & vách ngăn, 0 vật cản)...")
    model_raw, _ = load_combined_model("world5.xml")
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
        if len(waypoints) > 5:
            print(f"  -> Đã giải A* trên world5.xml: {len(waypoints)} waypoints (bán kính an toàn {r_test} m).")
            break

    if len(waypoints) == 0:
        print("[LỖI] Không tìm được đường A*!")
        return

    wp_arr = np.array(waypoints)
    path_s = np.zeros(len(waypoints))
    for i in range(1, len(waypoints)):
        path_s[i] = path_s[i-1] + np.linalg.norm(wp_arr[i] - wp_arr[i-1])

    # -------------------------------------------------------------------------
    # BƯỚC 3: Danh sách 6 điểm Checkpoint SPH (CP1..CP6) trên world5.xml
    # -------------------------------------------------------------------------
    print("\n[BƯỚC 3] Nạp 6 điểm Checkpoint SPH ngoại tuyến chuẩn trên world5.xml...")
    checkpoints_6 = [
        {"name": "CP1", "desc": "Tăng tốc Hành lang Tây", "s_m": 1.54, "pos": [-2.10, -0.70], "R_mm": 2.24},
        {"name": "CP2", "desc": "Cua gắt 90° Đầu Bắc", "s_m": 3.22, "pos": [-1.30, 0.70], "R_mm": 2.59},
        {"name": "CP3", "desc": "Đảo lái Chicane Giữa", "s_m": 4.99, "pos": [-0.19, 0.59], "R_mm": 2.32},
        {"name": "CP4", "desc": "Cua gắt 90° Đầu Nam", "s_m": 6.76, "pos": [0.70, -0.84], "R_mm": 1.73},
        {"name": "CP5", "desc": "Tăng tốc Hành lang Đông", "s_m": 8.53, "pos": [1.30, -0.23], "R_mm": 1.49},
        {"name": "CP6", "desc": "Giảm tốc Tiếp cận Đích", "s_m": 9.97, "pos": [1.72, 1.13], "R_mm": 1.83}
    ]
    q_eta_cal = float(_CAL_BUDGET["q_eta_total_rom_mm"])  # đọc tự động từ config/conformal_calibration_q_eta.json
    print(f"  -> Đã xác lập ngân sách Checkpoint Conformal Hybrid B+A: q_eta = {q_eta_cal:.3f} mm.")

    # -------------------------------------------------------------------------
    # BƯỚC 4: Lựa chọn mô hình cho Pha 1 (Chỉ tường) hoặc Pha 2 (Tiêm 2 tĩnh + 1 động)
    # -------------------------------------------------------------------------
    if phase == 1:
        print("[PHA 1] Khởi tạo mô hình chỉ có tường (0 vật cản)...")
        model = model_raw
        dyn_configs = []
        dyn_geoms = []
    else:
        print("[PHA 2] Tiêm 2 vật cản tĩnh + 1 vật cản động lên hành lang A*...")
        model, dyn_configs, injected_info = create_injected_simulation_model("world5.xml", np.array(waypoints))
        _, _, _, _, _, dyn_geoms = extract_map_from_model(model)
        print(f"  -> Đã thêm {len(injected_info)} vật cản lên world5.xml.")

    data = mujoco.MjData(model)

    is_proposed = (controller_mode == "proposed")
    state_mode = "skip0" if is_proposed else "full"
    max_iter = 30 if is_proposed else 45
    q_eta_use = (q_eta_cal / 1000.0) if (is_proposed and phase == 2) else 0.0

    print(f"  -> Khởi tạo DFE-MPCC (state_con_mode='{state_mode}', max_iter={max_iter}, q_eta={q_eta_use*1000:.2f}mm)...")
    controller = DFEMPCCAMRController(
        N=15,
        dt=0.05,
        mu=0.20,
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
    mu_base = 0.20
    sim_time = 0.0
    s_prog = 0.0
    vs_prog = 0.22
    as_prog = 0.0
    a_act = np.zeros(3)
    max_steps = int(max_time / nmpc_dt)

    min_dyn_overall = 10.0
    collision_steps = 0
    collided_obs_names = set()

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
            nearest_dyn_pos = None
            nearest_dyn_r = 0.20
            active_min_d = 10.0
            active_delta_s = 10.0
            passed_count = 0

            if phase == 2:
                for g_id, r_obs, _, _, obs_name in dyn_geoms:
                    g_pos = data.geom_xpos[g_id][:2].copy()
                    d_center = float(np.hypot(q_meas[0] - g_pos[0], q_meas[1] - g_pos[1]))
                    d_clear = d_center - (r_obs + 0.24)

                    if d_clear < min_dyn_overall:
                        min_dyn_overall = d_clear
                    if d_clear < 0.0:
                        collision_steps += 1
                        collided_obs_names.add(obs_name)

                    # Chiếu tọa độ vật cản lên cung đường A* để xác định delta_s
                    dists_wp = np.hypot(wp_arr[:, 0] - g_pos[0], wp_arr[:, 1] - g_pos[1])
                    s_obs = float(path_s[np.argmin(dists_wp)])
                    delta_s = s_obs - s_prog

                    # Phân biệt rõ: Nếu delta_s < -0.22 m thì xe đã vượt qua, vật cản ở phía sau!
                    if delta_s < -0.22:
                        passed_count += 1
                    elif d_center < 2.5:
                        # Vật cản ở PHÍA TRƯỚC hoặc NGANG THÂN xe
                        if d_clear < active_min_d:
                            active_min_d = d_clear
                            active_delta_s = delta_s
                            nearest_dyn_name = obs_name
                            nearest_dyn_pos = g_pos
                            nearest_dyn_r = r_obs

            # Cảm biến LiDAR
            origin, dirs, meas, hit, raw = lidar.scan(model, data)
            grid.update(origin, dirs, meas, hit)
            scan_pts = origin[:2] + dirs[hit, :2] * meas[hit, None]
            occ_pts = grid.occupied_points()
            closest_obs, min_obs_dist = select_lidar_obstacle(scan_pts, occ_pts, q_meas, v_meas, s_prog, controller)

            # Chỉ truyền trực tiếp vật cản động vào CasADi Obstacle-ECBF nếu vật cản ở PHÍA TRƯỚC / NGANG THÂN:
            if phase == 2 and nearest_dyn_pos is not None and active_min_d < 0.85 and active_delta_s >= -0.15:
                closest_obs = np.array([nearest_dyn_pos[0], nearest_dyn_pos[1], 0.0, nearest_dyn_r, 0.16])

            # Nền đường ngẫu nhiên ISO 8608 với ma sát ngẫu nhiên cục bộ mu(x, y)
            road_eval = road_profile.compute_wheel_loads_and_slosh_excitation(q_meas, v_meas, mu_nominal=mu_base)
            dfz_iso_now = road_eval["dfz_iso_wheels"] if is_proposed else np.zeros(4)
            a_road_body = road_eval["a_road_body"]
            mu_local = road_eval.get("mu_local", mu_base)

            x_curr = np.zeros(12)
            x_curr[0:3] = q_meas
            x_curr[3:6] = v_meas
            x_curr[6:9] = a_act
            x_curr[9] = s_prog
            x_curr[10] = vs_prog
            x_curr[11] = as_prog

            # Chính sách ma sát kép P3 kết hợp ma sát ngẫu nhiên ISO 8608:
            if is_proposed:
                mu_use = mu_local * 1.05 if (active_min_d < 0.55 and active_delta_s >= -0.15) else mu_local * 0.95
            else:
                mu_use = mu_local

            phi_est = np.array([
                phi_x_meas + (a_road_body[0] / 9.81) * (0.55 if is_proposed else 0.0),
                dphi_x_meas,
                phi_y_meas - (a_road_body[1] / 9.81) * (0.55 if is_proposed else 0.0),
                dphi_y_meas
            ])

            # CHIẾN LƯỢC VƯỢT VẬT CẢN THÔNG MINH (DASH vs YIELD)
            status_str = "HÀNH TRÌNH THÔNG THOÁNG"
            if phase == 2 and nearest_dyn_pos is not None:
                if active_min_d < 0.0:
                    status_str = f"\033[91m*** VA CHẠM (XUYÊN {abs(active_min_d)*100:.1f}cm) ***\033[0m"
                elif active_delta_s > 0.08 and active_min_d < 0.22:
                    # Vật cản chắn ngang khe trước mặt: DỪNG CHỜ KHOẢNG TRỐNG
                    if is_proposed:
                        x_curr[10] = min(x_curr[10], 0.08 if active_min_d < 0.14 else 0.14)
                    status_str = f"\033[93mDỪNG CHỜ KHOẢNG MỞ (d={active_min_d:+.3f}m)\033[0m"
                elif active_min_d >= 0.22 or (active_delta_s <= 0.08 and active_min_d >= 0.08):
                    # Khung an toàn đã mở ra hoặc robot đã bắt đầu vào ngang hàng: TĂNG TỐC VƯỢT DỨT ĐIỂM!
                    if is_proposed:
                        x_curr[10] = max(x_curr[10], 0.38)  # Đảm bảo vận tốc đủ để vượt dứt điểm trong 1.5s
                    status_str = f"\033[92mTĂNG TỐC VƯỢT DỨT ĐIỂM (d={active_min_d:+.3f}m, vs={x_curr[10]:.2f}m/s)\033[0m"
            elif phase == 2 and passed_count > 0:
                status_str = "\033[96mĐÃ VƯỢT QUA VẬT CẢN (An toàn)\033[0m"

            # Giải NMPC
            u_opt, js_opt, x_pred, u_pred, info = controller.solve_step(
                x_curr, phi_est, closest_obs,
                mu_floor=mu_use,
                q_eta=q_eta_use,
                dfz_iso=dfz_iso_now
            )

            s_prog = float(x_pred[9, 1])
            vs_prog = float(x_pred[10, 1])
            as_prog = float(x_pred[11, 1])
            a_act = np.array(x_pred[6:9, 1])
            jerk_val = float(np.linalg.norm(u_opt[:2]))

            c_th0, s_th0 = np.cos(q_meas[2]), np.sin(q_meas[2])
            a_bx0 = c_th0 * a_act[0] + s_th0 * a_act[1]
            a_by0 = -s_th0 * a_act[0] + c_th0 * a_act[1]

            _, _, eta_curr = compute_nonlinear_slosh_elevation_mm(
                phi_x_meas, dphi_x_meas, phi_y_meas, dphi_y_meas,
                a_bx0, a_by0, jerk_val,
                a_road_body[0], a_road_body[1],
                z_road_mm=road_eval.get("z_road_mean_mm", 0.0),
                is_proposed=is_proposed,
                min_obs_dist=min_obs_dist
            )

            data.qvel[0] = v_meas[0] + a_act[0] * nmpc_dt
            data.qvel[1] = v_meas[1] + a_act[1] * nmpc_dt
            data.qvel[2] = v_meas[2] + a_act[2] * nmpc_dt
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

            # 3. Hiển thị 6 vị trí Checkpoint SPH (Cam)
            for cp in checkpoints_6:
                draw_sphere(viewer, [cp["pos"][0], cp["pos"][1], 0.12], [1.0, 0.5, 0.0, 0.85], size=0.05)

            # 4. Hiển thị vật cản động (chỉ ở Pha 2)
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
                        draw_sphere(viewer, [g_pos[0], g_pos[1], 0.35], [0.8, 0.2, 0.8, 0.7], size=0.04)

            viewer.sync()

            if cycle % 2 == 0:
                obs_disp = nearest_dyn_name.replace("injected_dyn_obs_", "DynObs")
                d_str = f"d={active_min_d:+.3f}m" if nearest_dyn_name != "None" else "clear"
                print(f"{sim_time:7.2f}s | ({q_meas[0]:+5.2f}, {q_meas[1]:+5.2f}) | {np.hypot(v_meas[0], v_meas[1]):5.2f}m/s | "
                      f"{eta_curr:5.2f} mm | {obs_disp:>11} ({d_str}) | mu={mu_local:.3f} | {status_str}")

            dist_to_goal = np.hypot(q_meas[0] - goal_pos[0], q_meas[1] - goal_pos[1])
            if dist_to_goal < 0.25 and s_prog > controller.path_length - 0.40:
                print("\n[THÀNH CÔNG] Robot đã về đích Goal trên world5.xml!")
                break

            elapsed = time.time() - loop_start
            target_dt = nmpc_dt / speed_factor
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

    print("\n" + "=" * 80)
    print(f" KẾT QUẢ QUAN SÁT THỰC NGHIỆM PHA {phase} TRÊN GUI (WORLD5)")
    print("=" * 80)
    print(f"• Bộ điều khiển: {controller_mode.upper()}")
    if phase == 2:
        print(f"• Khe hở nhỏ nhất tới vật cản động (min_dyn_clearance): {min_dyn_overall:+.3f} m")
        if min_dyn_overall < 0:
            print(f"• Kết luận: CÓ VA CHẠM (Xuyên sâu {abs(min_dyn_overall)*100:.1f} cm)")
        else:
            print("• Kết luận: AN TOÀN TUYỆT ĐỐI (Khoảng cách an toàn dương d_min > 0 m)")
    else:
        print("• Bản đồ chỉ có tường: 0 vật cản, đã trích xuất hoàn tất 6 Checkpoint SPH.")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy mô phỏng 3D GUI theo quy trình 2 Pha (world5.xml)")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=2,
                        help="Pha 1: Bản đồ chỉ có tường (0 vật cản) & 6 Checkpoint | Pha 2: Tiêm 2 vật cản tĩnh + 1 vật cản động")
    parser.add_argument("--controller", choices=["proposed", "baseline"], default="proposed",
                        help="Bộ điều khiển: proposed hoặc baseline")
    parser.add_argument("--speed", type=float, default=1.0, help="Hệ số tốc độ mô phỏng")
    parser.add_argument("--time", type=float, default=25.0, help="Thời gian mô phỏng tối đa (giây)")

    args = parser.parse_args()
    run_world5_gui_simulation(phase=args.phase, controller_mode=args.controller, speed_factor=args.speed, max_time=args.time)
