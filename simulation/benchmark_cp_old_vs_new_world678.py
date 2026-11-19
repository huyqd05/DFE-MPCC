# -*- coding: utf-8 -*-
"""
benchmark_cp_old_vs_new_world678.py
=============================================================================
BENCHMARK GHÉP CẶP ĐỐI CHỨNG QUY TẮC CHECKPOINT CŨ VS. MỚI (TÁCH DẤU)
TRÊN 3 BẢN ĐỒ KHO VẬN WORLD6, WORLD7, WORLD8 (MA SÁT ĐỒNG NHẤT MU = 0.35)

So sánh 1-đối-1 hai cấu hình đồng nhất bộ giải, Conformal d_p = 5 mm và
mặt đường ISO 8608 Class B (chỉ thay đổi bộ trích xuất Checkpoint):
  - OLD: CoupledVelocityTubeH124(use_physics_cp_constraints=False)
  - NEW: CoupledVelocityTubeH124(use_physics_cp_constraints=True)

Đầu ra:
  - bao cao/checkpoints_world6.json, checkpoints_world7.json, checkpoints_world8.json
  - bao cao/benchmark_cp_old_vs_new_world678.json
  - bao cao/cpnew_fig3_closed_loop_world678.png
=============================================================================
"""

import os
import sys

for _env_k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_k, "1")

import math
import time
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ["", "controller", "solver", "planning", "simulation", "cfd", "bao cao"]:
    p = os.path.join(PROJECT_DIR, sub) if sub else PROJECT_DIR
    if p not in sys.path:
        sys.path.insert(0, p)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import mujoco
from astar import AStarPlanner
from planning.velocity_tube_h124 import CoupledVelocityTubeH124
from controller.dfe_mpcc_amr_controller import DFEMPCCAMRController
from iso8608_road_profile import ISO8608RoadProfile
from test_sim_amr_closed_loop import extract_map_from_model, load_combined_model
from benchmark_world6_complex_4dyn_obs import compute_world6_slosh_elevation_mm
from benchmark_world8_3way import compute_static_clearance
from solver.robot_model_amr_casadi import TANK_A, TANK_B, TANK_H, H_LIQUID, M_TOTAL, Z_COM, L_X, L_Y
from cfd.conformal_checkpoint_eval import load_conformal_calibration_budget


MAP_CONFIGS = [
    {"name": "world6.xml", "key": "world6", "bounds": (-2.9, 2.9), "radius": 0.24, "ckpt_json": "checkpoints_world6.json"},
    {"name": "world7.xml", "key": "world7", "bounds": (-3.1, 3.1), "radius": 0.28, "ckpt_json": "checkpoints_world7.json"},
    {"name": "world8.xml", "key": "world8", "bounds": (-3.1, 3.1), "radius": 0.28, "ckpt_json": "checkpoints_world8.json"},
]


def _run_offline_stage_ab_worker(args: tuple) -> dict:
    """Chạy Stage A (KM) + Stage B (WCSPH Hybrid B+A dp=5mm) cho một cặp (bản đồ, old/new)."""
    wname, map_key, mode_key, wp_list = args
    use_new = (mode_key == "new")
    wp_arr = np.asarray(wp_list, dtype=float)
    diffs = np.diff(wp_arr, axis=0)
    lens = np.hypot(diffs[:, 0], diffs[:, 1])
    path_len_total = float(np.sum(lens))

    planner = CoupledVelocityTubeH124(
        mu_default=0.35,
        kappa_th=1.05,
        use_physics_cp_constraints=use_new,
    )

    t0 = time.perf_counter()
    summary = planner.build_offline_tube(
        wp_arr,
        mu_profile=lambda s: 0.35,
        phi_camber_profile=lambda s: 0.0,
        z_com_profile=lambda s: Z_COM,
        delta_warp_profile=lambda s: 0.0,
        run_wcsph_check=False,
    )
    t_tube_ms = (time.perf_counter() - t0) * 1000.0

    t_sph0 = time.perf_counter()
    sph_summary = planner.run_wcsph_checkpoint_standardization(dp_sph=0.005)
    t_wcsph_s = time.perf_counter() - t_sph0

    scale_s = path_len_total / max(float(planner.s_grid[-1]), 1e-6)
    cp_recs = []
    for cp in planner.checkpoints:
        idx_p = int(cp["idx_peak"])
        cp_recs.append({
            "cp_id": int(cp["cp_id"]),
            "name": f"CP{int(cp['cp_id'])}",
            "desc": str(cp["type"]),
            "s_m": round(float(cp["s_peak"]) * scale_s, 3),
            "s_tube_m": round(float(cp["s_peak"]), 3),
            "pos": [round(float(planner.x_grid[idx_p]), 4), round(float(planner.y_grid[idx_p]), 4)],
            "kappa_peak": round(float(cp["kappa_peak"]), 4),
            "v_ub_cp": round(float(cp.get("v_ub_cp", 0.0)), 3),
            "eta_dbep_t2_mm": round(float(cp.get("eta_dbep_t2_mm", 0.0)), 2),
            "eta_rom_corr_t2_mm": round(float(cp.get("eta_rom_corr_t2_mm", 0.0)), 2),
            "eta_wcsph_t2_mm": round(float(cp.get("eta_wcsph_t2_mm", 0.0)), 2),
            "eta_wcsph_cp_mm": round(float(cp.get("eta_wcsph_cp_mm", 0.0)), 2),
            "R_mm": round(float(cp.get("raw_residual_R_mm", cp.get("residual_R_mm", 0.0))), 2),
            "residual_R_mm": round(float(cp.get("raw_residual_R_mm", cp.get("residual_R_mm", 0.0))), 2),
            "debiased_residual_r_mm": round(float(cp.get("debiased_residual_r_mm", 0.0)), 3),
            "save_ram_ms": round(float(cp.get("save_ram_ms", 0.0)), 3),
        })

    return {
        "wname": wname,
        "map_key": map_key,
        "mode": mode_key,
        "use_new": use_new,
        "n_cp": len(planner.checkpoints),
        "t_tube_ms": round(float(t_tube_ms), 2),
        "t_wcsph_s": round(float(t_wcsph_s), 2),
        "n_fp_iters": int(summary["n_fp_iters"]),
        "wcsph_summary": sph_summary,
        "checkpoints_records": cp_recs,
    }


def _run_closed_loop_paired(wname: str, wp_arr: np.ndarray, tube_planner: CoupledVelocityTubeH124, q_eta_use: float) -> dict:
    """Chạy vòng kín MuJoCo + DFE-MPCC 20 Hz cho 1 cấu hình trên bản đồ tĩnh wname."""
    model_sim, _ = load_combined_model(wname)
    start_pos, goal_pos, obs_circles, obs_boxes, _, _ = extract_map_from_model(model_sim)

    diffs = np.diff(wp_arr, axis=0)
    path_len_total = float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))

    data_sim = mujoco.MjData(model_sim)
    mujoco.mj_resetData(model_sim, data_sim)
    data_sim.qpos[0] = start_pos[0]
    data_sim.qpos[1] = start_pos[1]
    heading_init = math.atan2(wp_arr[1, 1] - wp_arr[0, 1], wp_arr[1, 0] - wp_arr[0, 0])
    data_sim.qpos[2] = heading_init
    mujoco.mj_forward(model_sim, data_sim)

    road_profile = ISO8608RoadProfile(
        road_class="B", seed=32, k_contact=4800.0, c_contact=135.0, max_dfz_ratio=0.42
    )

    dt_sim = model_sim.opt.timestep
    nmpc_dt = 0.05
    substeps = int(round(nmpc_dt / dt_sim))
    max_time_s = 60.0
    max_steps = int(max_time_s / nmpc_dt)

    ctrl = DFEMPCCAMRController(
        N=15, dt=nmpc_dt, mu=0.35,
        state_con_mode="skip0",
        mu_plan_scale=0.95,
        ipopt_max_iter=60,
        h_lim_m=0.020,
    )
    ctrl.set_path(wp_arr)
    ctrl.set_tube_planner(tube_planner)

    s_prog = 0.0
    vs_prog = 0.20
    as_prog = 0.0
    a_act = np.zeros(3)

    eta_hist, fz_hist, ec_hist, solve_ms_list = [], [], [], []
    min_wall = 10.0
    slack_steps = 0
    total_steps = 0
    finished = False
    t_finish_s = max_time_s

    for step in range(max_steps):
        sim_time = step * nmpc_dt
        mujoco.mj_forward(model_sim, data_sim)

        q_meas = data_sim.qpos[:3].copy()
        v_meas = data_sim.qvel[:3].copy()
        phi_x_meas = float(data_sim.qpos[3])
        dphi_x_meas = float(data_sim.qvel[3])
        phi_y_meas = float(data_sim.qpos[4])
        dphi_y_meas = float(data_sim.qvel[4])

        road_eval = road_profile.compute_wheel_loads_and_slosh_excitation(
            q_meas, v_meas, mu_nominal=0.35, friction_variation_scale=0.0
        )
        dfz_iso_now = road_eval["dfz_iso_wheels"]
        a_road_body = road_eval["a_road_body"]
        z_road_mm = float(np.mean(np.abs(road_eval["z_wheels_mm"])))

        phi_est = np.array([
            phi_x_meas + (a_road_body[0] / 9.81) * 0.55,
            dphi_x_meas,
            phi_y_meas - (a_road_body[1] / 9.81) * 0.55,
            dphi_y_meas,
        ])

        s_preview = np.clip(s_prog + np.arange(ctrl.N) * nmpc_dt * max(vs_prog, 0.35), 0.0, path_len_total)
        v_ub_nom_q = np.interp(s_preview, tube_planner.s_grid, tube_planner.v_ub_nom)
        v_ub_online, _ = tube_planner.query_online_horizon_tube(
            s_horizon=s_preview, phi_est=phi_est, mu_online=0.35
        )
        v_ub_eff = np.minimum(v_ub_nom_q, v_ub_online)
        v_ub_h = np.clip(v_ub_eff, 0.25, 0.65).reshape(1, ctrl.N)
        v_lb_h = np.zeros((1, ctrl.N))

        x_cur = np.zeros(12)
        x_cur[0:3] = q_meas
        x_cur[3:6] = v_meas
        x_cur[6:9] = a_act
        x_cur[9] = s_prog
        x_cur[10] = vs_prog
        x_cur[11] = as_prog

        t_sol0 = time.perf_counter()
        ret = ctrl.solve_step(
            x_current=x_cur,
            phi_est=phi_est,
            obs_info=None,
            v_ub_horizon=v_ub_h,
            v_lb_horizon=v_lb_h,
            mu_floor=0.35,
            q_eta=q_eta_use,
            dfz_iso=dfz_iso_now,
        )
        sol_ms = (time.perf_counter() - t_sol0) * 1000.0
        u_opt = ret[0]
        x_pred = ret[2]
        sol_info = ret[4] if len(ret) > 4 else {}

        s_prog = float(x_pred[9, 1])
        vs_prog = float(x_pred[10, 1])
        as_prog = float(x_pred[11, 1])
        a_act = np.array(x_pred[6:9, 1])

        v0_cyc = data_sim.qvel[:3].copy()
        a0_cyc = a_act.copy()
        u0_cyc = u_opt[:3].copy()

        c_th0, s_th0 = math.cos(q_meas[2]), math.sin(q_meas[2])
        a_bx = c_th0 * a_act[0] + s_th0 * a_act[1]
        a_by = -s_th0 * a_act[0] + c_th0 * a_act[1]

        data_sim.qpos[3] = float(np.clip(data_sim.qpos[3] + ((a_by + a_road_body[1]) / 9.81) * 0.08, -0.25, 0.25))
        data_sim.qpos[4] = float(np.clip(data_sim.qpos[4] - ((a_bx + a_road_body[0]) / 9.81) * 0.08, -0.25, 0.25))

        for sub in range(substeps):
            tau_sub = (sub + 1) * dt_sim
            v_target = v0_cyc + a0_cyc * tau_sub + 0.5 * u0_cyc * (tau_sub ** 2)
            v_lin_t = math.hypot(v_target[0], v_target[1])
            if v_lin_t > 0.70:
                scale_v = 0.70 / v_lin_t
                v_target[0] *= scale_v
                v_target[1] *= scale_v
            data_sim.qvel[:3] = v_target
            mujoco.mj_step(model_sim, data_sim)

        q_now = data_sim.qpos[:3].copy()
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

        m_pitch = M_TOTAL * abs(a_bx) * Z_COM
        m_roll = M_TOTAL * abs(a_by) * Z_COM
        dfz_p = m_pitch / (4.0 * L_X)
        dfz_r = m_roll / (4.0 * L_Y)
        fz_min_now = float(np.clip(17.78 - dfz_p - dfz_r - np.max(np.abs(dfz_iso_now)), 0.0, 17.78))

        xr, yr = float(ctrl.spline_x(s_prog)), float(ctrl.spline_y(s_prog))
        dxr, dyr = float(ctrl.spline_x.derivative()(s_prog)), float(ctrl.spline_y.derivative()(s_prog))
        t_len = math.hypot(dxr, dyr)
        tx, ty = dxr / max(1e-6, t_len), dyr / max(1e-6, t_len)
        nx, ny = -ty, tx
        err_x, err_y = q_now[0] - xr, q_now[1] - yr
        ec_now = abs(nx * err_x + ny * err_y)

        c_wall = compute_static_clearance(q_now[0], q_now[1], obs_boxes, obs_circles, r_robot=0.17)
        if c_wall < min_wall:
            min_wall = c_wall

        eta_hist.append(float(eta_now_mm))
        fz_hist.append(float(fz_min_now))
        ec_hist.append(float(ec_now))
        solve_ms_list.append(float(sol_ms))
        total_steps += 1

        slack_max_val = max(
            float(sol_info.get("slack_zono_max", 0.0)),
            float(sol_info.get("slack_tube", 0.0)),
        )
        if slack_max_val > 1e-4:
            slack_steps += 1

        dist_goal = math.hypot(q_now[0] - goal_pos[0], q_now[1] - goal_pos[1])
        if dist_goal < 0.28 or s_prog >= (path_len_total - 0.15):
            finished = True
            t_finish_s = round(sim_time, 2)
            break

    return {
        "finished": bool(finished),
        "t_finish_s": float(t_finish_s),
        "max_eta_mm": round(float(max(eta_hist)), 3),
        "eta_p95_mm": round(float(np.percentile(eta_hist, 95)), 3),
        "mean_eta_mm": round(float(np.mean(eta_hist)), 3),
        "min_fz_N": round(float(min(fz_hist)), 3),
        "max_ec_m": round(float(max(ec_hist)), 4),
        "min_wall_m": round(float(min_wall), 4),
        "slack_steps": int(slack_steps),
        "total_steps": int(total_steps),
        "mean_solve_ms": round(float(np.mean(solve_ms_list)), 2),
        "p95_solve_ms": round(float(np.percentile(solve_ms_list, 95)), 2),
    }


def _vn_fig(val: float, dec: int = 2) -> str:
    return f"{float(val):.{dec}f}".replace(".", ",")


def _draw_comparison_figure(results_list: list, out_png: str, eta_bound_cl_mm: float = None):
    """Vẽ Hình 3: Đối chứng WCSPH Stage B (t_wcsph, n_cp) và Vòng kín MPCC (max_eta, T_finish) Cũ vs. Mới."""
    maps = [r["name"] for r in results_list]
    x = np.arange(len(maps))
    w = 0.36

    t_old = [r["old"]["t_wcsph_s"] for r in results_list]
    t_new = [r["new"]["t_wcsph_s"] for r in results_list]
    ncp_old = [r["old"]["n_cp"] for r in results_list]
    ncp_new = [r["new"]["n_cp"] for r in results_list]

    eta_old = [r["old"]["closed_loop"]["max_eta_mm"] for r in results_list]
    eta_new = [r["new"]["closed_loop"]["max_eta_mm"] for r in results_list]
    fin_old = [r["old"]["closed_loop"]["t_finish_s"] for r in results_list]
    fin_new = [r["new"]["closed_loop"]["t_finish_s"] for r in results_list]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.25), dpi=220)

    b1 = ax1.bar(x - w / 2, t_old, w, color="#95A5A6", edgecolor="#2C3E50", label="Quy tắc cũ (không ràng buộc)")
    b2 = ax1.bar(x + w / 2, t_new, w, color="#1F4E79", edgecolor="#002060", label="Quy tắc mới (NMS tách dấu)")
    for i, (bo, bn) in enumerate(zip(b1, b2)):
        ax1.text(x[i] - 0.03, bo.get_height() + 12, f"{_vn_fig(t_old[i], 1)} s\n({ncp_old[i]} CP)",
                 ha="right", va="bottom", fontsize=7.5, color="#2C3E50", fontweight="bold")
        ax1.text(x[i] + 0.03, bn.get_height() + 12, f"{_vn_fig(t_new[i], 1)} s\n({ncp_new[i]} CP)",
                 ha="left", va="bottom", fontsize=7.5, color="#002060", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(maps, fontsize=8.8, fontweight="bold")
    ax1.set_ylabel("Thời gian kiểm toán WCSPH (s)", fontsize=9.0)
    ax1.set_xlim(-0.58, len(maps) - 0.42)
    ax1.set_ylim(0, max(t_old) * 1.26)
    ax1.set_title("(a) Thời gian kiểm toán WCSPH Stage B (dp = 5 mm)", fontsize=9.5, fontweight="bold", color="#002060")
    ax1.grid(True, axis="y", alpha=0.28)
    ax1.legend(loc="upper left", fontsize=7.6, frameon=True)
    for spn in ("top", "right"):
        ax1.spines[spn].set_visible(False)

    b3 = ax2.bar(x - w / 2, eta_old, w, color="#E67E22", edgecolor="#935116", label="Vòng kín quy tắc cũ (max η)")
    b4 = ax2.bar(x + w / 2, eta_new, w, color="#117864", edgecolor="#0B5345", label="Vòng kín quy tắc mới (max η)")
    ax2.axhline(30.0, color="#C0392B", ls="--", lw=1.2, label="Ngưỡng tràn thành bình (30,0 mm)")
    if eta_bound_cl_mm is not None:
        ax2.axhline(eta_bound_cl_mm, color="#5B2C6F", ls=":", lw=1.3, label=f"Biên dự báo vòng kín ({_vn_fig(eta_bound_cl_mm, 2)} mm)")
    for i, (bo, bn) in enumerate(zip(b3, b4)):
        ax2.text(x[i] - 0.03, bo.get_height() + 0.55, f"{_vn_fig(eta_old[i], 2)} mm\n({_vn_fig(fin_old[i], 2)} s)",
                 ha="right", va="bottom", fontsize=6.9, color="#7E5109", fontweight="bold")
        ax2.text(x[i] + 0.03, bn.get_height() + 0.55, f"{_vn_fig(eta_new[i], 2)} mm\n({_vn_fig(fin_new[i], 2)} s)",
                 ha="left", va="bottom", fontsize=6.9, color="#0B5345", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(maps, fontsize=8.8, fontweight="bold")
    ax2.set_ylabel("Đỉnh sóng vòng kín max η (mm)", fontsize=9.0)
    ax2.set_xlim(-0.58, len(maps) - 0.42)
    ax2.set_ylim(0, 34.5)
    ax2.set_title("(b) Đỉnh sóng vòng kín MuJoCo & Thời gian về đích", fontsize=9.5, fontweight="bold", color="#002060")
    ax2.grid(True, axis="y", alpha=0.28)
    ax2.legend(loc="upper right", fontsize=7.2, frameon=True)
    for spn in ("top", "right"):
        ax2.spines[spn].set_visible(False)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.89, bottom=0.12, wspace=0.22)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main():
    t_all0 = time.perf_counter()
    out_dir = os.path.join(PROJECT_DIR, "bao cao")
    os.makedirs(out_dir, exist_ok=True)
    cal_budget = load_conformal_calibration_budget()
    q_eta_use = float(cal_budget["recommended_q_eta_m"])
    print("=" * 92)
    print(f"BENCHMARK GHÉP CẶP CHECKPOINT CŨ VS. MỚI (WORLD6/7/8) | P_q_eta = {q_eta_use*1000:.3f} mm")
    print("=" * 92)

    # 1. Sinh đường A* cố định một lần cho từng bản đồ
    map_wps = {}
    map_lens = {}
    for cfg in MAP_CONFIGS:
        m, _ = load_combined_model(cfg["name"])
        sp, gp, oc, ob, _, _ = extract_map_from_model(m)
        bnd = cfg["bounds"]
        pa = AStarPlanner(x_bounds=bnd, y_bounds=bnd, resolution=0.05, robot_radius=cfg["radius"])
        pa.add_obstacles(oc, ob)
        wp = np.asarray(pa.plan(start_pos=sp[:2], goal_pos=gp[:2], smooth=True), dtype=float)
        diffs = np.diff(wp, axis=0)
        L = float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))
        map_wps[cfg["name"]] = wp
        map_lens[cfg["name"]] = L
        print(f"  [A*] {cfg['name']}: r={cfg['radius']:.2f} m, L={L:.3f} m ({len(wp)} waypoints)", flush=True)

    # 2. Chạy song song Stage A + Stage B WCSPH (dp=5 mm) cho 6 tổ hợp (3 bản đồ x 2 quy tắc)
    tasks = []
    for cfg in MAP_CONFIGS:
        wp_list = map_wps[cfg["name"]].tolist()
        tasks.append((cfg["name"], cfg["key"], "old", wp_list))
        tasks.append((cfg["name"], cfg["key"], "new", wp_list))

    print("\n[Stage A + B] Đang chạy song song 6 tác vụ kiểm toán WCSPH dp = 0.005 m...", flush=True)
    stage_results = {}
    with ProcessPoolExecutor(max_workers=min(6, os.cpu_count() or 4)) as ex:
        fut_map = {ex.submit(_run_offline_stage_ab_worker, t): (t[0], t[2]) for t in tasks}
        for fut in as_completed(fut_map):
            res = fut.result()
            stage_results[(res["wname"], res["mode"])] = res
            print(
                f"  -> {res['wname']} [{res['mode'].upper()}]: n_cp={res['n_cp']:2d} | "
                f"t_tube={res['t_tube_ms']:.1f} ms | t_wcsph={res['t_wcsph_s']:.2f} s | "
                f"NSE={res['wcsph_summary']['NSE']:.4f}",
                flush=True,
            )

    # 3. Dựng lại planner Stage A gọn nhẹ (~15 ms) trong tiến trình chính & chạy vòng kín MuJoCo + DFE-MPCC
    combined_rows = []
    for cfg in MAP_CONFIGS:
        wname = cfg["name"]
        wp_arr = map_wps[wname]
        r_old = stage_results[(wname, "old")]
        r_new = stage_results[(wname, "new")]

        pl_old = CoupledVelocityTubeH124(mu_default=0.35, kappa_th=1.05, use_physics_cp_constraints=False)
        pl_old.build_offline_tube(
            wp_arr,
            mu_profile=lambda s: 0.35,
            phi_camber_profile=lambda s: 0.0,
            z_com_profile=lambda s: Z_COM,
            delta_warp_profile=lambda s: 0.0,
            run_wcsph_check=False,
        )
        pl_new = CoupledVelocityTubeH124(mu_default=0.35, kappa_th=1.05, use_physics_cp_constraints=True)
        pl_new.build_offline_tube(
            wp_arr,
            mu_profile=lambda s: 0.35,
            phi_camber_profile=lambda s: 0.0,
            z_com_profile=lambda s: Z_COM,
            delta_warp_profile=lambda s: 0.0,
            run_wcsph_check=False,
        )
        vub_rmse_mm_s = float(np.sqrt(np.mean(((pl_old.v_ub_nom - pl_new.v_ub_nom) * 1000.0) ** 2)))
        vub_max_diff_mm_s = float(np.max(np.abs((pl_old.v_ub_nom - pl_new.v_ub_nom) * 1000.0)))

        # Ghi cập nhật checkpoints_world6/7/8.json theo quy tắc mới & Conformal dp=5mm
        ckpt_path = os.path.join(out_dir, cfg["ckpt_json"])
        with open(ckpt_path, "w", encoding="utf-8") as f_cp:
            json.dump({
                "map": wname,
                "map_file": f"xml/{wname}",
                "wcsph_engine": "Hybrid B+A (CoupledVelocityTubeH124 SplitNMS, dp=0.005 m)",
                "n_checkpoints": r_new["n_cp"],
                "mu_floor": 0.35,
                "q_eta_calibrated_mm": round(q_eta_use * 1000.0, 3),
                "sph_offline_duration_s": r_new["t_wcsph_s"],
                "wcsph_summary": r_new["wcsph_summary"],
                "checkpoints": r_new["checkpoints_records"],
            }, f_cp, indent=2, ensure_ascii=False)

        print(f"\n[Closed-Loop] Đang chạy MuJoCo + DFE-MPCC cho {wname}...")
        cl_old = _run_closed_loop_paired(wname, wp_arr, pl_old, q_eta_use)
        print(f"  -> {wname} [OLD]: finished={cl_old['finished']} T={cl_old['t_finish_s']:.2f}s max_eta={cl_old['max_eta_mm']:.2f}mm Fz_min={cl_old['min_fz_N']:.2f}N slack={cl_old['slack_steps']}/{cl_old['total_steps']}")
        cl_new = _run_closed_loop_paired(wname, wp_arr, pl_new, q_eta_use)
        print(f"  -> {wname} [NEW]: finished={cl_new['finished']} T={cl_new['t_finish_s']:.2f}s max_eta={cl_new['max_eta_mm']:.2f}mm Fz_min={cl_new['min_fz_N']:.2f}N slack={cl_new['slack_steps']}/{cl_new['total_steps']}")

        combined_rows.append({
            "name": wname,
            "key": cfg["key"],
            "L_m": round(map_lens[wname], 3),
            "robot_radius_m": cfg["radius"],
            "v_ub_rmse_diff_mm_s": round(vub_rmse_mm_s, 3),
            "v_ub_max_diff_mm_s": round(vub_max_diff_mm_s, 3),
            "wcsph_speedup_pct": round(100.0 * (r_old["t_wcsph_s"] - r_new["t_wcsph_s"]) / max(r_old["t_wcsph_s"], 1e-6), 2),
            "old": {
                "n_cp": r_old["n_cp"],
                "t_tube_ms": r_old["t_tube_ms"],
                "t_wcsph_s": r_old["t_wcsph_s"],
                "nse": r_old["wcsph_summary"]["NSE"],
                "rmse_mm": r_old["wcsph_summary"]["RMSE_mm"],
                "max_abs_r_mm": r_old["wcsph_summary"]["max_abs_debiased_residual_eta_mm"],
                "closed_loop": cl_old,
            },
            "new": {
                "n_cp": r_new["n_cp"],
                "t_tube_ms": r_new["t_tube_ms"],
                "t_wcsph_s": r_new["t_wcsph_s"],
                "nse": r_new["wcsph_summary"]["NSE"],
                "rmse_mm": r_new["wcsph_summary"]["RMSE_mm"],
                "max_abs_r_mm": r_new["wcsph_summary"]["max_abs_debiased_residual_eta_mm"],
                "closed_loop": cl_new,
            },
        })

    out_fig = os.path.join(out_dir, "cpnew_fig3_closed_loop_world678.png")
    _draw_comparison_figure(combined_rows, out_fig, eta_bound_cl_mm=float(cal_budget["eta_bound_cl_mm"]))

    out_json = os.path.join(out_dir, "benchmark_cp_old_vs_new_world678.json")
    with open(out_json, "w", encoding="utf-8") as f_out:
        json.dump({
            "q_eta_calibrated_mm": round(q_eta_use * 1000.0, 3),
            "eta_bound_cl_mm": float(cal_budget["eta_bound_cl_mm"]),
            "dp_sph_m": 0.005,
            "road_profile": "ISO 8608 Class B (seed=32, mu=0.35)",
            "total_runtime_s": round(time.perf_counter() - t_all0, 2),
            "maps": combined_rows,
        }, f_out, indent=2, ensure_ascii=False)

    print(f"\n[HOÀN TẤT] Đã xuất {out_json} và {out_fig} trong {time.perf_counter() - t_all0:.2f} s.")


def redraw_from_saved_json():
    out_dir = os.path.join(PROJECT_DIR, "bao cao")
    in_json = os.path.join(out_dir, "benchmark_cp_old_vs_new_world678.json")
    out_fig = os.path.join(out_dir, "cpnew_fig3_closed_loop_world678.png")
    with open(in_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    _draw_comparison_figure(data["maps"], out_fig, eta_bound_cl_mm=data.get("eta_bound_cl_mm"))
    print(f"Đã vẽ lại {out_fig} từ {in_json}")


if __name__ == "__main__":
    if "--redraw" in sys.argv:
        redraw_from_saved_json()
    else:
        main()
