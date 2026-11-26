# -*- coding: utf-8 -*-
"""
benchmark_world8_3way.py
=============================================================================
BENCHMARK ĐỐI CHỨNG 2 CẤU HÌNH DFE-MPCC (KHÔNG vs. CÓ CHECKPOINT SPH)
TRÊN ĐẤU TRƯỜNG KHO VẬN WORLD8 (6.5 x 6.5 M, MA SÁT ĐỒNG NHẤT MU = 0.35)
=============================================================================

Mục đích:
1. Tách bạch thuần túy tác động của WCSPH Checkpoint & Conformal (q_eta = 3.70 mm)
   và Ống vận tốc H124 trên bản đồ kho vận đa hành lang World 8:
   - (1) DFE-MPCC Không Checkpoint : Bộ điều khiển & bộ ước lượng đồng đẳng, nhưng
     chưa tích hợp ống H124 và dự trữ Conformal WCSPH (q_eta = 0, không tube).
   - (2) DFE-MPCC Có Checkpoint SPH: Đầy đủ Ống vận tốc bậc 3 Krasnoselskii-Mann H124
     + các Checkpoint WCSPH & Conformal q_eta = 3.70 mm.
2. Quỹ đạo A* toàn cục, mặt sàn phẳng hoàn toàn với hệ số ma sát đồng nhất mu = 0.35
   (tạm thời bỏ bẫy ma sát / ước lượng ma sát), kết hợp mấp mô mặt đường hình học
   ISO 8608 Class B (Seed 32, friction_variation_scale = 0.0).
3. Tiêm 4 vật cản động (D1..D4) tuần tra cắt ngang 4 hành lang (cùng logic nhường đường).
4. Xuất dữ liệu benchmark:
   - bao cao/benchmark_world8_records.json
   - bao cao/hinh_so_sanh_3_mpcc_world8.png
=============================================================================
"""

import os
import sys
import math
import time
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ["", "controller", "solver", "planning", "simulation", "cfd", "bao cao"]:
    p = os.path.join(PROJECT_DIR, sub) if sub else PROJECT_DIR
    if p not in sys.path:
        sys.path.insert(0, p)

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
from test_sim_amr_closed_loop import (
    extract_map_from_model,
    load_combined_model,
    update_dynamic_obstacles,
    Lidar
)
from robot_model_amr_casadi import (
    TANK_A, TANK_B, TANK_H, H_LIQUID,
    M_TOTAL, Z_COM, L_X, L_Y
)
from conformal_checkpoint_eval import load_conformal_calibration_budget
from benchmark_world6_complex_4dyn_obs import compute_world6_slosh_elevation_mm

_CAL_BUDGET = load_conformal_calibration_budget()
TANK_FREEBOARD = (TANK_H - H_LIQUID) * 1000.0  # 30.0 mm
CONFORMAL_LIMIT = float(_CAL_BUDGET["eta_bound_cl_mm"])  # mm, đọc tự động từ config/conformal_calibration_q_eta.json

VARIANT_META = {
    "no_checkpoint": {
        "title": "DFE-MPCC (Không Checkpoint SPH)",
        "short_name": "DFE-MPCC (Không Checkpoint)",
        "color": "#E67E22",
        "ls": "--",
        "tag": "LƯỢT 1/2"
    },
    "full": {
        "title": "DFE-MPCC (Có Checkpoint SPH)",
        "short_name": "DFE-MPCC (Có Checkpoint SPH)",
        "color": "#117864",
        "ls": "-",
        "tag": "LƯỢT 2/2"
    }
}


def predict_gate_pos(g_desc: dict, t_query: float) -> np.ndarray:
    """Dự báo vị trí 2D (x, y) của vật cản động tại thời điểm t_query."""
    disp = g_desc["amp"] * math.sin(g_desc["omega"] * t_query + g_desc["phase"])
    return g_desc["base_pos"] + g_desc["axis_xy"] * disp


def compute_static_clearance(x: float, y: float, boxes: list, circles: list, r_robot: float = 0.20) -> float:
    """Tính khoảng cách nhỏ nhất từ tâm robot tới các vách tường."""
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


def run_benchmark_world8():
    print("=" * 105)
    print("  SIMPAT BENCHMARK WORLD 8: ĐẤU TRƯỜNG KHO VẬN ĐA HÀNH LANG (MA SÁT ĐỒNG NHẤT MU = 0.35)")
    print("  Đối Chứng 2 Cấu Hình DFE-MPCC (Không Checkpoint vs. Có Checkpoint SPH) Dưới ISO 8608 Class B")
    print("=" * 105)

    out_dir = os.path.join(PROJECT_DIR, "bao cao")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Khởi tạo mô hình bản đồ cơ sở và hoạch định đường A*
    model_base, _ = load_combined_model("world8.xml")
    start_pos, goal_pos, obs_circles, obs_boxes, map_geoms, _ = extract_map_from_model(model_base)
    print(f"\n[1] Bản đồ cơ sở: Start = {start_pos[:2]}, Goal = {goal_pos[:2]}")
    print(f"    Số vách tường hộp: {len(obs_boxes)}, Vách trụ: {len(obs_circles)}")

    waypoints = []
    for r_test in (0.28, 0.25, 0.22, 0.20):
        planner_astar = AStarPlanner(x_bounds=(-3.1, 3.1), y_bounds=(-3.1, 3.1), resolution=0.05, robot_radius=r_test)
        planner_astar.add_obstacles(obs_circles, obs_boxes)
        cand_wp = planner_astar.plan(start_pos=start_pos[:2], goal_pos=goal_pos[:2], smooth=True)
        if len(cand_wp) > 10:
            waypoints = cand_wp
            print(f"    -> Đã giải A* World 8 với robot_radius = {r_test:.2f} m ({len(waypoints)} waypoints).")
            break
    wp_arr = np.array(waypoints, dtype=float)

    diffs = np.diff(wp_arr, axis=0)
    lens = np.hypot(diffs[:, 0], diffs[:, 1])
    path_len_total = float(np.sum(lens))
    print(f"    -> Đã hoạch định quỹ đạo A* trơn: L = {path_len_total:.3f} m ({len(wp_arr)} điểm nốt).")

    # 2. Tiêm 4 vật cản động tuần tra cắt ngang 4 hành lang
    inj_configs = [
        {"type": "dynamic_cylinder", "center_override": [-2.35, 0.40], "radius": 0.14, "amp": 0.32, "omega": 1.05, "phase": 0.85, "slide_axis": "1 0 0", "rgba": "0.95 0.15 0.60 1", "name": "dyn_obs_1_west"},
        {"type": "dynamic_cylinder", "center_override": [-0.85, 0.50], "radius": 0.14, "amp": 0.32, "omega": 0.95, "phase": 0.40, "slide_axis": "1 0 0", "rgba": "0.05 0.72 0.85 1", "name": "dyn_obs_2_midwest"},
        {"type": "dynamic_cylinder", "center_override": [0.65, -0.60], "radius": 0.14, "amp": 0.32, "omega": 1.00, "phase": 1.90, "slide_axis": "1 0 0", "rgba": "0.98 0.55 0.05 1", "name": "dyn_obs_3_mideast"},
        {"type": "dynamic_cylinder", "center_override": [1.85, 2.40], "radius": 0.14, "amp": 0.38, "omega": 0.90, "phase": 0.60, "slide_axis": "0 1 0", "rgba": "0.55 0.20 0.90 1", "name": "dyn_obs_4_east_goal"},
    ]
    model_sim, dyn_configs, _ = create_injected_simulation_model("xml/world8.xml", wp_arr, configs=inj_configs)
    print(f"[2] Đã tiêm thành công 4 vật cản động tuần tra cắt ngang hành lang.")

    # 3. Quy hoạch ngoại tuyến Ống vận tốc khả thi bậc 3 & Chuẩn hóa WCSPH tại các Checkpoint (mu = 0.35 đồng nhất)
    print(f"\n[3] Quy hoạch Ống vận tốc khả thi bậc 3 Krasnoselskii-Mann & Chuẩn hóa WCSPH (mu = 0.35 đồng nhất)...")
    path_s_cum = np.concatenate(([0.0], np.cumsum(lens)))
    mu_w8_profile = np.full(len(wp_arr), 0.35, dtype=float)

    def mu_prof_fn(s_q):
        return float(np.interp(s_q, path_s_cum, mu_w8_profile))

    tube_planner = CoupledVelocityTubeH124(mu_default=0.35, kappa_th=1.05)
    t_tube0 = time.perf_counter()
    tube_summary = tube_planner.build_offline_tube(
        wp_arr,
        mu_profile=mu_prof_fn,
        phi_camber_profile=lambda s: 0.0,
        z_com_profile=lambda s: Z_COM,
        delta_warp_profile=lambda s: 0.0,
        run_wcsph_check=True
    )
    t_tube_ms = (time.perf_counter() - t_tube0) * 1000.0
    print(f"    -> Hoàn thành quy hoạch ống trong {t_tube_ms:.2f} ms ({tube_summary['n_fp_iters']} vòng lặp, phần dư = {tube_summary['fp_unrelaxed_residual']:.6f} m/s).")
    print(f"    -> Đã trích xuất & chuẩn hóa {tube_summary['n_checkpoints']} Checkpoint WCSPH (NSE = {tube_summary['wcsph_summary']['NSE']:.4f}).")
    for cp in tube_summary['checkpoints']:
        print(f"       * CP{cp['cp_id']} ({cp['type']}): s = {cp['s_peak']:.2f} m | |kappa| = {abs(cp['kappa_peak']):.2f} | v_ub = {cp['v_ub_cp']:.3f} m/s ({cp['active_regime']})")

    # Lưu thông tin Checkpoint Hybrid B+A của World 8 ra JSON
    ckpt_w8_path = os.path.join(out_dir, "checkpoints_world8.json")
    cp_records_w8 = []
    scale_s_w8 = path_len_total / max(float(tube_planner.s_grid[-1]), 1e-6)
    for cp in tube_planner.checkpoints:
        idx_p = int(cp["idx_peak"])
        cp_records_w8.append({
            "cp_id": int(cp["cp_id"]),
            "name": f"CP{int(cp['cp_id'])}",
            "desc": str(cp["type"]),
            "s_m": round(float(cp["s_peak"]) * scale_s_w8, 3),
            "s_tube_m": round(float(cp["s_peak"]), 3),
            "pos": [round(float(tube_planner.x_grid[idx_p]), 4), round(float(tube_planner.y_grid[idx_p]), 4)],
            "kappa_peak": round(float(cp["kappa_peak"]), 4),
            "v_ub_cp": round(float(cp.get("v_ub_cp", 0.0)), 3),
            "eta_wcsph_t2_mm": round(float(cp.get("eta_wcsph_t2_mm", 0.0)), 2),
            "eta_rom_corr_t2_mm": round(float(cp.get("eta_rom_corr_t2_mm", 0.0)), 2),
            "R_mm": round(float(cp.get("raw_residual_R_mm", cp.get("residual_R_mm", 0.0))), 2),
        })
    with open(ckpt_w8_path, "w", encoding="utf-8") as f_cp8:
        json.dump({
            "map_file": "xml/world8.xml",
            "wcsph_engine": "Hybrid B+A (CoupledVelocityTubeH124, dp=0.005 m)",
            "n_checkpoints": len(cp_records_w8),
            "wcsph_summary": tube_summary.get("wcsph_summary", {}),
            "checkpoints": cp_records_w8,
        }, f_cp8, indent=2, ensure_ascii=False)

    # 4. Trích xuất cổng giao cắt vật cản động (Gate Crossing Descriptors)
    dyn_geoms = extract_map_from_model(model_sim)[5]
    gate_descriptors = []
    for idx_d, (g_id, r_obs, gtype_dyn, rgba_dyn, obs_name) in enumerate(dyn_geoms):
        b_id = model_sim.geom_bodyid[g_id]
        base_pos = model_sim.body_pos[b_id][:2].copy()
        j_adr = model_sim.body_jntadr[b_id]
        if j_adr >= 0:
            j_name = mujoco.mj_id2name(model_sim, mujoco.mjtObj.mjOBJ_JOINT, j_adr)
            axis_xy = model_sim.jnt_axis[j_adr][:2].copy()
        else:
            j_name = ""
            axis_xy = np.array([1.0, 0.0])

        j_cfg = dyn_configs.get(j_name, {"amp": 0.28, "omega": 1.0, "phase": 0.0})
        amp = float(j_cfg.get("amp", 0.28))
        omega = float(j_cfg.get("omega", 1.0))
        phase = float(j_cfg.get("phase", 0.0))

        s_earliest = 100.0
        for u_sweep in np.linspace(-amp, amp, 15):
            p_sweep = base_pos + axis_xy * u_sweep
            dists = np.hypot(wp_arr[:, 0] - p_sweep[0], wp_arr[:, 1] - p_sweep[1])
            idx_min = np.argmin(dists)
            if dists[idx_min] < (r_obs + 0.35):
                s_cand = float(path_s_cum[idx_min])
                if s_cand < s_earliest:
                    s_earliest = s_cand

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

    # 5. Chạy mô phỏng đối chứng 2 cấu hình DFE-MPCC (Không Checkpoint vs. Có Checkpoint SPH)
    variants = ["no_checkpoint", "full"]
    records = {}

    dt_sim = model_sim.opt.timestep  # 0.005 s
    nmpc_dt = 0.05                   # 0.050 s (20 Hz)
    substeps = int(round(nmpc_dt / dt_sim))
    max_time_s = 55.0
    max_steps = int(max_time_s / nmpc_dt)

    for v_key in variants:
        meta = VARIANT_META[v_key]
        print(f"\n" + "-" * 85)
        print(f">>> BẮT ĐẦU CHẠY: {meta['tag']} — {meta['title'].upper()}")
        print("-" * 85)

        data_sim = mujoco.MjData(model_sim)
        mujoco.mj_resetData(model_sim, data_sim)
        update_dynamic_obstacles(model_sim, data_sim, 0.0, custom_dyn_configs=dyn_configs)
        data_sim.qpos[0] = start_pos[0]
        data_sim.qpos[1] = start_pos[1]
        heading_init = math.atan2(wp_arr[1, 1] - wp_arr[0, 1], wp_arr[1, 0] - wp_arr[0, 0])
        data_sim.qpos[2] = heading_init
        mujoco.mj_forward(model_sim, data_sim)

        road_profile = ISO8608RoadProfile(road_class="B", seed=32, k_contact=4800.0, c_contact=135.0, max_dfz_ratio=0.42)

        ctrl = DFEMPCCAMRController(
            N=15, dt=nmpc_dt, mu=0.35,
            state_con_mode="skip0",
            mu_plan_scale=0.95,
            ipopt_max_iter=60,
            h_lim_m=0.020,
        )
        ctrl.set_path(wp_arr)
        if v_key == "full":
            ctrl.set_tube_planner(tube_planner)

        # Bộ đệm lưu trữ chuỗi thời gian
        st = {
            "variant_key": v_key,
            "t_hist": [],
            "traj_xy": [],
            "s_hist": [],
            "v_hist": [],
            "vs_hist": [],
            "eta_hist": [],
            "fz_hist": [],
            "ec_hist": [],
            "solve_times": [],
            "slack_steps": 0,
            "total_steps": 0,
            "min_wall": 10.0,
            "min_dyn": 10.0,
            "max_eta": 0.0,
            "min_fz": 20.0,
            "max_ec": 0.0,
            "finished": False,
            "t_finish_s": max_time_s
        }

        s_prog = 0.0
        vs_prog = 0.20
        as_prog = 0.0
        a_act = np.zeros(3)
        dashing_gate = -1

        for step in range(max_steps):
            sim_time = step * nmpc_dt
            update_dynamic_obstacles(model_sim, data_sim, sim_time, custom_dyn_configs=dyn_configs)
            mujoco.mj_forward(model_sim, data_sim)

            q_meas = data_sim.qpos[:3].copy()
            v_meas = data_sim.qvel[:3].copy()
            phi_x_meas = float(data_sim.qpos[3])
            dphi_x_meas = float(data_sim.qvel[3])
            phi_y_meas = float(data_sim.qpos[4])
            dphi_y_meas = float(data_sim.qvel[4])

            # Đánh giá kích động nền đường hình học ISO 8608 (ma sát đồng nhất mu = 0.35, friction_variation_scale = 0.0)
            road_eval = road_profile.compute_wheel_loads_and_slosh_excitation(
                q_meas, v_meas,
                mu_nominal=0.35,
                friction_variation_scale=0.0,
            )
            dfz_iso_now = road_eval["dfz_iso_wheels"]
            a_road_body = road_eval["a_road_body"]
            z_road_mm = float(np.mean(np.abs(road_eval["z_wheels_mm"])))

            # Bộ ước lượng trạng thái 3D đồng đẳng cho cả 2 cấu hình
            phi_est = np.array([
                phi_x_meas + (a_road_body[0] / 9.81) * 0.55,
                dphi_x_meas,
                phi_y_meas - (a_road_body[1] / 9.81) * 0.55,
                dphi_y_meas,
            ])
            mu_solve = 0.35
            s_preview = np.clip(s_prog + np.arange(ctrl.N) * nmpc_dt * max(vs_prog, 0.35), 0.0, path_len_total)

            if v_key == "no_checkpoint":
                # DFE-MPCC Không Checkpoint: q_eta = 0, chưa áp dụng ống H124 & Conformal margin
                q_eta_use = 0.0
                v_ub_h = np.full((1, ctrl.N), 0.65)
                v_lb_h = np.zeros((1, ctrl.N))
            else:
                # DFE-MPCC Có Checkpoint SPH: Ống vận tốc H124 + dự trữ Conformal q_eta
                q_eta_use = float(_CAL_BUDGET["recommended_q_eta_m"])
                v_ub_nom_q = np.interp(s_preview, tube_planner.s_grid, tube_planner.v_ub_nom)
                v_ub_online, _ = tube_planner.query_online_horizon_tube(
                    s_horizon=s_preview,
                    phi_est=phi_est,
                    mu_online=mu_solve,
                )
                v_ub_eff = np.minimum(v_ub_nom_q, v_ub_online)
                v_ub_h = np.clip(v_ub_eff, 0.25, 0.65).reshape(1, ctrl.N)
                v_lb_h = np.zeros((1, ctrl.N))

            # Xây dựng vector trạng thái 12 chiều chuẩn cho bộ giải DFE-MPCC
            # [q(3), v(3), a(3), s(1), vs(1), as(1)]
            x_cur = np.zeros(12)
            x_cur[0:3] = q_meas
            x_cur[3:6] = v_meas
            x_cur[6:9] = a_act
            x_cur[9] = s_prog
            x_cur[10] = vs_prog
            x_cur[11] = as_prog

            # Đánh giá cổng vật cản động (Gate Crossing) đồng đẳng cho cả 2 cấu hình
            should_wait = False
            active_gate = -1
            for idx_d, g_desc in enumerate(gate_descriptors):
                if g_desc["s_obs"] > 50.0:
                    continue
                delta_s = g_desc["s_obs"] - s_prog
                if -0.45 <= delta_s <= 0.60:
                    active_gate = idx_d
                    if dashing_gate == idx_d:
                        should_wait = False
                    else:
                        if v_key == "full" and tube_planner is not None and getattr(tube_planner, "is_ready", False):
                            v_tube_at_gate = float(np.interp(g_desc["s_obs"], tube_planner.s_grid, tube_planner.v_ub_nom))
                            v_dash_limit = min(0.42, max(0.32, v_tube_at_gate))
                        else:
                            v_dash_limit = 0.42

                        tau_max = max(3.5, (g_desc["s_obs"] + 0.55 - s_prog) / max(0.20, v_dash_limit) + 1.0)
                        taus = np.linspace(0.0, tau_max, 45)
                        min_pred_c = 10.0

                        s_taus = []
                        s_cur = s_prog
                        v_cur = float(np.hypot(v_meas[0], v_meas[1]))
                        for tau_idx in range(len(taus)):
                            if tau_idx > 0:
                                dt_tau = float(taus[tau_idx] - taus[tau_idx - 1])
                                v_cur = min(v_dash_limit, v_cur + 0.80 * dt_tau)
                                s_cur = min(s_cur + v_cur * dt_tau, path_len_total)
                            s_taus.append(s_cur)

                        for tau_idx, tau in enumerate(taus):
                            s_tau = s_taus[tau_idx]
                            if (g_desc["s_obs"] - 0.38) <= s_tau <= (g_desc["s_obs"] + 0.50):
                                p_rob_tau, _, _ = ctrl.get_path_reference(s_tau)
                                p_obs_tau = predict_gate_pos(g_desc, sim_time + tau)
                                c_tau = float(np.hypot(p_rob_tau[0] - p_obs_tau[0], p_rob_tau[1] - p_obs_tau[1])) - (g_desc["radius"] + 0.22)
                                if c_tau < min_pred_c:
                                    min_pred_c = c_tau

                        if min_pred_c >= 0.06:
                            dashing_gate = idx_d
                            should_wait = False
                        else:
                            should_wait = True
                            break

            if active_gate == -1:
                dashing_gate = -1

            sol_info = {}
            if should_wait:
                data_sim.qvel[:3] = 0.0
                a_act[:] = 0.0
                vs_prog = 0.0
                as_prog = 0.0
                if active_gate >= 0:
                    s_prog = min(s_prog, gate_descriptors[active_gate]["s_obs"] - 0.46)
                x_pred = np.tile(x_cur.reshape(-1, 1), (1, ctrl.N + 1))
                u_opt = np.zeros(3)
                sol_ms = 0.5
            else:
                if dashing_gate >= 0:
                    x_cur[10] = max(vs_prog, 0.32)

                t_sol0 = time.perf_counter()
                ret = ctrl.solve_step(
                    x_current=x_cur,
                    phi_est=phi_est,
                    obs_info=None,
                    v_ub_horizon=v_ub_h,
                    v_lb_horizon=v_lb_h,
                    mu_floor=mu_solve,
                    q_eta=q_eta_use,
                    dfz_iso=dfz_iso_now
                )
                sol_ms = (time.perf_counter() - t_sol0) * 1000.0

                u_opt = ret[0]
                x_pred = ret[2]
                sol_info = ret[4] if len(ret) > 4 else {}

                s_prog = float(x_pred[9, 1])
                vs_prog = float(x_pred[10, 1])
                as_prog = float(x_pred[11, 1])
                a_act = np.array(x_pred[6:9, 1])

            if dashing_gate >= 0:
                _, t_dir, _ = ctrl.get_path_reference(s_prog)
                target_v = min(0.40, max(0.32, float(np.interp(s_prog, tube_planner.s_grid, tube_planner.v_ub_nom)) if (v_key == "full" and tube_planner) else 0.40))
                data_sim.qvel[0] = v_meas[0] + (target_v * t_dir[0] - v_meas[0]) * 0.45
                data_sim.qvel[1] = v_meas[1] + (target_v * t_dir[1] - v_meas[1]) * 0.45
                data_sim.qvel[2] = v_meas[2] + a_act[2] * nmpc_dt
                vs_prog = float(np.hypot(data_sim.qvel[0], data_sim.qvel[1]))

            # Tích phân vật lý liên tục trong MuJoCo (Substepping)
            q0_cyc = q_meas.copy()
            if should_wait:
                v0_cyc = np.zeros(3)
                a0_cyc = np.zeros(3)
                u0_cyc = np.zeros(3)
            else:
                v0_cyc = data_sim.qvel[:3].copy()
                a0_cyc = a_act.copy()
                u0_cyc = u_opt[:3].copy()

            # Chuyển đổi gia tốc sang hệ tọa độ thân xe (Body Frame)
            c_th0, s_th0 = math.cos(q_meas[2]), math.sin(q_meas[2])
            a_bx = c_th0 * a_act[0] + s_th0 * a_act[1]
            a_by = -s_th0 * a_act[0] + c_th0 * a_act[1]

            # Cập nhật góc con lắc chất lỏng đúng trục slosh3.xml: qpos[3]=slosh_x (+a_bx), qpos[4]=slosh_y (-a_by)
            data_sim.qpos[3] = float(np.clip(data_sim.qpos[3] + ((a_bx + a_road_body[0]) / 9.81) * 0.08, -0.25, 0.25))
            data_sim.qpos[4] = float(np.clip(data_sim.qpos[4] - ((a_by + a_road_body[1]) / 9.81) * 0.08, -0.25, 0.25))

            for sub in range(substeps):
                tau_sub = (sub + 1) * dt_sim
                v_target = v0_cyc + a0_cyc * tau_sub + 0.5 * u0_cyc * (tau_sub**2)
                v_lin_t = math.hypot(v_target[0], v_target[1])
                if v_lin_t > 0.70:
                    scale_v = 0.70 / v_lin_t
                    v_target[0] *= scale_v
                    v_target[1] *= scale_v
                data_sim.qvel[:3] = v_target
                mujoco.mj_step(model_sim, data_sim)

            q_now = data_sim.qpos[:3].copy()
            v_now = data_sim.qvel[:3].copy()
            v_lin = float(math.hypot(v_now[0], v_now[1]))

            # Tính toán độ dâng sóng mặt thoáng bằng hàm chuẩn hóa thống nhất giữa World 6, 7 và 8
            th_x_act = float(data_sim.qpos[3])
            dth_x_act = float(data_sim.qvel[3])
            th_y_act = float(data_sim.qpos[4])
            dth_y_act = float(data_sim.qvel[4])
            jerk_mag = float(np.linalg.norm(u_opt[:2]))
            _, _, eta_now_mm = compute_world6_slosh_elevation_mm(
                th_x_act, dth_x_act, th_y_act, dth_y_act,
                a_bx, a_by, jerk_mag,
                a_road_bx=a_road_body[0], a_road_by=a_road_body[1],
                z_road_mm=z_road_mm,
            )

            # Tính toán phân bổ tải pháp tuyến 4 bánh F_z
            m_pitch = M_TOTAL * abs(a_bx) * Z_COM
            m_roll = M_TOTAL * abs(a_by) * Z_COM
            dfz_p = m_pitch / (4.0 * L_X)
            dfz_r = m_roll / (4.0 * L_Y)
            fz_min_now = float(np.clip(17.78 - dfz_p - dfz_r - np.max(np.abs(dfz_iso_now)), 0.0, 17.78))

            # Tính sai số bám viền e_c
            xr, yr = float(ctrl.spline_x(s_prog)), float(ctrl.spline_y(s_prog))
            dxr, dyr = float(ctrl.spline_x.derivative()(s_prog)), float(ctrl.spline_y.derivative()(s_prog))
            t_len = math.hypot(dxr, dyr)
            tx, ty = dxr / max(1e-6, t_len), dyr / max(1e-6, t_len)
            nx, ny = -ty, tx
            err_x, err_y = q_now[0] - xr, q_now[1] - yr
            ec_now = abs(nx * err_x + ny * err_y)

            # Đánh giá khoảng cách an toàn với tường và vật cản động
            c_wall = compute_static_clearance(q_now[0], q_now[1], obs_boxes, obs_circles, r_robot=0.17)
            if c_wall < st["min_wall"]:
                st["min_wall"] = c_wall

            min_d_dyn = 10.0
            for (g_id, r_obs, _, _, d_name) in [g for g in extract_map_from_model(model_sim)[5]]:
                b_id = model_sim.geom_bodyid[g_id]
                d_pos = data_sim.xpos[b_id][:2]
                dist_dyn = math.hypot(q_now[0] - d_pos[0], q_now[1] - d_pos[1]) - (r_obs + 0.18)
                if dist_dyn < min_d_dyn:
                    min_d_dyn = dist_dyn
            if min_d_dyn < st["min_dyn"]:
                st["min_dyn"] = min_d_dyn

            # Thu thập dữ liệu
            st["t_hist"].append(round(sim_time, 2))
            st["traj_xy"].append([round(float(q_now[0]), 3), round(float(q_now[1]), 3)])
            st["s_hist"].append(round(s_prog, 2))
            st["v_hist"].append(round(v_lin, 3))
            st["vs_hist"].append(round(vs_prog, 3))
            st["eta_hist"].append(round(eta_now_mm, 2))
            st["fz_hist"].append(round(fz_min_now, 2))
            st["ec_hist"].append(round(ec_now, 4))
            st["solve_times"].append(round(sol_ms, 2))
            st["total_steps"] += 1

            if eta_now_mm > st["max_eta"]:
                st["max_eta"] = eta_now_mm
            if fz_min_now < st["min_fz"]:
                st["min_fz"] = fz_min_now
            if ec_now > st["max_ec"]:
                st["max_ec"] = ec_now

            # Kiểm tra kích hoạt biến bù slack thực tế từ bộ giải NLP (nhất quán cho cả 2 cấu hình)
            slack_max_val = max(
                float(sol_info.get("slack_zono_max", 0.0)),
                float(sol_info.get("slack_tube", 0.0)),
            )
            if slack_max_val > 1e-4:
                st["slack_steps"] += 1

            # Kiểm tra về đích
            dist_goal = math.hypot(q_now[0] - goal_pos[0], q_now[1] - goal_pos[1])
            if dist_goal < 0.28 or s_prog >= (path_len_total - 0.15):
                st["finished"] = True
                st["t_finish_s"] = round(sim_time, 2)
                print(f"  [✓] ĐÃ VỀ ĐÍCH AN TOÀN tại T = {sim_time:.2f} s (s = {s_prog:.2f} / {path_len_total:.2f} m).")
                break

            if step % 80 == 0:
                print(f"  t = {sim_time:5.2f}s | s = {s_prog:5.2f}m | v = {v_lin:4.2f}m/s | eta = {eta_now_mm:5.2f}mm | Fz = {fz_min_now:5.2f}N | solve = {sol_ms:5.2f}ms")

        records[v_key] = st

    # 5. Lưu kết quả đối chứng ra JSON
    json_path = os.path.join(out_dir, "benchmark_world8_records.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"\n[4] Đã lưu bản ghi benchmark JSON tại: {json_path}")

    # 6. Xuất đồ thị đối chứng 4-Panel chất lượng cao
    out_fig_path = os.path.join(out_dir, "hinh_so_sanh_3_mpcc_world8.png")
    generate_world8_benchmark_figure(wp_arr, obs_boxes, obs_circles, inj_configs, tube_planner, records, out_fig_path)

    # 7. In bảng tổng kết đối chứng định lượng
    print("\n" + "=" * 122)
    print(f"  BẢNG TỔNG KẾT ĐỐI CHỨNG ĐỊNH LƯỢNG 2 CẤU HÌNH DFE-MPCC TRÊN WORLD 8 (L = {path_len_total:.2f} m, mu = 0.35)")
    print("=" * 122)
    print(f"{'Cấu hình Bộ điều khiển':<34} | {'max eta (mm)':>14} | {'eta_P95':>10} | {'Fz,min (N)':>11} | {'d_dyn (m)':>9} | {'|e_c|_max':>9} | {'T_finish':>8} | {'Kết luận nghiệm thu'}")
    print("-" * 128)
    for k in variants:
        st = records[k]
        p95 = float(np.percentile(st["eta_hist"], 95)) if st["eta_hist"] else 0.0
        eta_tag = " (TRÀN!)" if st["max_eta"] > 30.0 else (" (>Stokes)" if st["max_eta"] > 20.0 else " (An toàn)")
        fz_tag = " (NHẤC!)" if st["min_fz"] < 3.0 else " (>3.0N)"
        min_dyn = float(st.get("min_dyn", 1.0))
        min_wall = float(st.get("min_wall", 1.0))
        is_fin = st.get("finished", False)

        if min_dyn < -0.02:
            conc = f"THẤT BẠI (Va chạm vật cản động: d={min_dyn:.2f}m)"
        elif min_wall < -0.02:
            conc = f"THẤT BẠI (Va chạm vách tường: d={min_wall:.2f}m)"
        elif st["max_eta"] > 30.0 or st["min_fz"] < 3.0:
            conc = "THẤT BẠI (Trào chất lỏng / Nhấc bánh)"
        elif not is_fin:
            conc = f"CHƯA VỀ ĐÍCH (s={st['s_hist'][-1]:.2f}m)"
        elif st["max_eta"] > 20.0:
            conc = "CẢNH BÁO (Vượt ngưỡng Stokes)"
        else:
            conc = "ĐẠT CHUẨN TOÀN DIỆN (SIMPAT)"

        v_title = VARIANT_META[k]["title"]
        print(f"{v_title[:34]:<34} | {st['max_eta']:6.2f}{eta_tag:<8} | {p95:7.2f} mm | {st['min_fz']:5.2f}{fz_tag:<6} | {min_dyn:8.2f}m | {st['max_ec']:7.3f} m | {st['t_finish_s']:7.2f}s | {conc}")
    print("=" * 122)


def generate_world8_benchmark_figure(wp_arr, obs_boxes, obs_circles, inj_configs, tube_planner, records, save_path):
    """Xuất đồ thị đối chứng 4-Panel 2 cấu hình (Không Checkpoint vs Có Checkpoint SPH) trên World 8 (mu = 0.35)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=250)
    plot_keys = [k for k in ["no_checkpoint", "full"] if k in records]

    # (a) Quỹ đạo 2D trên đấu trường World 8
    ax1 = axes[0, 0]
    ax1.set_title(r"(a) Quỹ đạo 2D Đấu trường Kho vận World 8 (6.5 x 6.5 m, $\mu = 0.35$ đồng nhất)", fontsize=10.5, fontweight="bold", color="#002060")
    for (cx, cy, hx, hy) in obs_boxes:
        rect = patches.Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy, linewidth=1.2, edgecolor="#2C3E50", facecolor="#BDC3C7", alpha=0.85)
        ax1.add_patch(rect)

    ax1.plot(wp_arr[:, 0], wp_arr[:, 1], color="#7F8C8D", lw=1.5, ls=":", label="Tim đường A*")
    for k in plot_keys:
        st = records[k]
        xy = np.array(st["traj_xy"])
        meta = VARIANT_META[k]
        ax1.plot(xy[:, 0], xy[:, 1], color=meta["color"], lw=2.2, ls=meta["ls"], label=meta["short_name"])

    for idx_d, d_cfg in enumerate(inj_configs):
        c_pt = d_cfg["center_override"]
        cir = patches.Circle((c_pt[0], c_pt[1]), d_cfg["radius"], color="#C0392B", alpha=0.85)
        ax1.add_patch(cir)
        ax1.text(c_pt[0], c_pt[1] + 0.22, f"D{idx_d+1}", fontsize=8, fontweight="bold", ha="center", color="#922B21")

    ax1.plot(-2.4, -2.4, "bo", markersize=9, label="Start (-2.4, -2.4)")
    ax1.plot(2.4, 2.4, "g*", markersize=12, label="Goal (2.4, 2.4)")
    ax1.set_xlim([-3.4, 3.4])
    ax1.set_ylim([-3.4, 3.4])
    ax1.set_xlabel("Tọa độ X (m)", fontsize=9.5)
    ax1.set_ylabel("Tọa độ Y (m)", fontsize=9.5)
    ax1.grid(True, ls=":", alpha=0.55)
    ax1.legend(loc="upper left", fontsize=7.5, framealpha=0.92)

    # (b) Mức dâng sóng mặt thoáng eta(t)
    ax2 = axes[0, 1]
    ax2.set_title(r"(b) Độ Dâng Sóng Mặt Thoáng $\eta(t)$ [mm] Dưới Kích Động ISO 8608 Class B", fontsize=10.5, fontweight="bold", color="#002060")
    max_eta_plot = 34.0
    for k in plot_keys:
        st = records[k]
        meta = VARIANT_META[k]
        max_eta_plot = max(max_eta_plot, float(st["max_eta"]) + 4.0)
        ax2.plot(st["t_hist"], st["eta_hist"], color=meta["color"], lw=2.0, ls=meta["ls"], label=f"{meta['short_name']} (max={st['max_eta']:.2f} mm)")
    ax2.axhline(TANK_FREEBOARD, color="red", lw=1.8, ls="--", label="Ngưỡng tràn bồn h_free = 30.0 mm")
    ax2.axhline(CONFORMAL_LIMIT, color="#27AE60", lw=1.5, ls=":", label=f"Bao trên Conformal = {CONFORMAL_LIMIT:.2f} mm")
    ax2.axhline(20.0, color="#566573", lw=1.1, ls="-.", label="Ngưỡng sóng tuyến tính Stokes = 20.0 mm")
    ax2.set_xlabel("Thời gian t (s)", fontsize=9.5)
    ax2.set_ylabel("Độ dâng sóng eta (mm)", fontsize=9.5)
    ax2.set_ylim([0, max_eta_plot])
    ax2.grid(True, ls=":", alpha=0.55)
    ax2.legend(loc="upper right", fontsize=7.5, framealpha=0.92)

    # (c) Áp lực pháp tuyến bánh F_z,min(t)
    ax3 = axes[1, 0]
    ax3.set_title(r"(c) Tải Pháp Tuyến Bánh Cực Tiểu $F_{z,\min}(t)$ [N] ($\mu = 0.35$)", fontsize=10.5, fontweight="bold", color="#002060")
    for k in plot_keys:
        st = records[k]
        meta = VARIANT_META[k]
        ax3.plot(st["t_hist"], st["fz_hist"], color=meta["color"], lw=2.0, ls=meta["ls"], label=f"{meta['short_name']} (min={st['min_fz']:.2f} N)")
    ax3.axhline(3.0, color="red", lw=1.8, ls="--", label="Ngưỡng nhấc bánh an toàn F_z,min = 3.0 N")
    ax3.set_xlabel("Thời gian t (s)", fontsize=9.5)
    ax3.set_ylabel("Tải pháp tuyến F_z,min (N)", fontsize=9.5)
    ax3.set_ylim([0, 20])
    ax3.grid(True, ls=":", alpha=0.55)
    ax3.legend(loc="lower right", fontsize=7.5, framealpha=0.92)

    # (d) Hồ sơ vận tốc v(s) dọc theo cung đường A*
    ax4 = axes[1, 1]
    ax4.set_title(r"(d) Vận Tốc Tịnh Tiến $v(s)$ [m/s] & Các Checkpoint WCSPH", fontsize=10.5, fontweight="bold", color="#002060")
    ax4.plot(tube_planner.s_grid, np.clip(tube_planner.v_ub_nom, 0.25, 0.65), color="#2980B9", lw=1.6, ls="--", label="Trần Ống VT v_ub(s) (Krasnoselskii-Mann)")
    for idx_cp, cp in enumerate(tube_planner.checkpoints):
        ax4.axvline(cp["s_peak"], color="#8E44AD", lw=0.9, ls=":", alpha=0.55)
        y_lbl = 0.69 - 0.045 * (idx_cp % 3)
        ax4.text(cp["s_peak"], y_lbl, f"CP{cp['cp_id']}", fontsize=6.8, rotation=0, color="#6C3483", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.75))

    for k in plot_keys:
        st = records[k]
        meta = VARIANT_META[k]
        ax4.plot(st["s_hist"], st["v_hist"], color=meta["color"], lw=2.0, ls=meta["ls"], label=f"{meta['short_name']} (T={st['t_finish_s']:.2f}s)")
    ax4.set_xlabel("Độ dài cung đường s (m)", fontsize=9.5)
    ax4.set_ylabel("Vận tốc v (m/s)", fontsize=9.5)
    ax4.set_ylim([0, 0.75])
    ax4.grid(True, ls=":", alpha=0.55)
    ax4.legend(loc="upper right", fontsize=7.5, framealpha=0.92)

    plt.tight_layout()
    plt.savefig(save_path, dpi=250)
    plt.close()
    print(f"[5] Đã xuất bản đồ thị đối chứng World 8 tại: {save_path}")


if __name__ == "__main__":
    run_benchmark_world8()
