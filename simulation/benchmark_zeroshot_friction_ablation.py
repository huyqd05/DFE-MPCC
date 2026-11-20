#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_zeroshot_friction_ablation.py
=============================================================================
KIỂM CHỨNG ĐỘC LẬP HƯỚNG ĐI 1: ƯỚC LƯỢNG MA SÁT ZERO-SHOT 2 TẦNG
(Tầng 1: Semantic Floor Prior + Split Conformal đa bản đồ tại t = 0
 Tầng 2: Adaptive FF-RLS + Mở khóa Dead-Zone Trượt Đột Ngột + Hợp nhất Bất Đối Xứng)

Các thử nghiệm thực hiện (tách rời hoàn toàn D_cal^maps vs D_test^maps & world6/7/8):
  A. Hiệu chuẩn Split Conformal đa bản đồ trên N_cal = 80 bản đồ tách biệt (seeds 400..479).
  B. Kiểm chứng phủ Zero-Shot trên N_test = 100 bản đồ hoàn toàn mới (seeds 600..699) + Bất biến tọa độ.
  C. Kiểm tra độ trễ phát hiện vũng dầu ẩn (mu = 0.13) ở Tầng 2 (bao gồm kịch bản rho_k thấp).
  D. Mô phỏng khép kín MuJoCo trên world6.xml, world7.xml, world8.xml có vũng dầu ẩn ngoài bản đồ:
       - B0: Hằng số lạc quan 0.35 * 0.95 = 0.3325 (không Tầng 2)
       - B1: Hằng số bảo thủ toàn cục 0.10 (không Tầng 2)
       - B2: Hướng đi 1 (Semantic Prior + Adaptive FF-RLS Fusion) - Zero-Shot thực
       - B3: Oracle (Biết trước ma sát 4 bánh thực trên cả dầu ẩn - Giới hạn trên lý thuyết)
=============================================================================
"""

import json
import math
import os
import sys
import time
import numpy as np
import mujoco

_CURR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_CURR)
for _sub in ["", "planning", "controller", "solver", "simulation"]:
    _p = os.path.join(_ROOT, _sub) if _sub else _ROOT
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from semantic_friction_estimator import (
    SemanticFrictionEstimator,
    generate_semantic_map_dataset,
    make_world_semantic_map,
    make_hidden_oil_macro_fn,
)
from friction_observer_ffrls import FrictionObserverFFRLS, OnlineFrictionFusion
from velocity_tube_h124 import CoupledVelocityTubeH124
from dfe_mpcc_amr_controller import DFEMPCCAMRController
from iso8608_road_profile import ISO8608RoadProfile
from conformal_checkpoint_eval import load_conformal_calibration_budget
from mecanum_traction_dynamics import (
    M_TOTAL, M_1X, M_1Y, L_P_X, L_P_Y,
    compute_dynamic_normal_loads,
    allocate_wheel_forces_clamped,
    evaluate_traction_and_slip,
)
from robot_model_amr_casadi import IZ_BASE, omega_x, omega_y
from benchmark_world6_complex_4dyn_obs import compute_world6_slosh_elevation_mm
from run_ros2_rviz_headless_mujoco import (
    build_map_scenario,
    evaluate_gate_crossing,
    select_tight_gate_obstacle,
    compute_static_clearance,
)


def _select_hidden_oil_patches(wp_arr: np.ndarray, sem_map) -> list:
    """
    Chọn tối đa 2 vị trí vũng dầu ẩn (mu=0.13) trên nền dry_concrete dọc quỹ đạo A*
    nơi có độ cong/bước ngoặt cao (tuyệt đối chỉ dùng cho phía Plant, không cấp cho B2).
    """
    n = len(wp_arr)
    labels = sem_map.eval_classes_batch(wp_arr)
    dx = np.gradient(wp_arr[:, 0])
    dy = np.gradient(wp_arr[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = np.maximum((dx ** 2 + dy ** 2) ** 1.5, 1e-6)
    kappa = np.abs(dx * ddy - dy * ddx) / denom

    i_lo = max(10, int(0.18 * n))
    i_hi = min(n - 10, int(0.84 * n))
    dry_candidates = [i for i in range(i_lo, i_hi) if labels[i] == "dry_concrete"]
    if not dry_candidates:
        dry_candidates = [i for i in range(i_lo, i_hi)]

    ranked = sorted(dry_candidates, key=lambda idx: float(kappa[idx]), reverse=True)
    chosen = []
    for idx in ranked:
        px, py = float(wp_arr[idx, 0]), float(wp_arr[idx, 1])
        if all(math.hypot(px - p["cx"], py - p["cy"]) >= 2.0 for p in chosen):
            chosen.append({
                "shape": "circle",
                "cx": px,
                "cy": py,
                "r": 0.52,
                "mu": 0.13,
                "blend_m": 0.05,
            })
        if len(chosen) >= 2:
            break
    return chosen


def run_calibration_and_zeroshot_coverage(cal_json_path: str) -> dict:
    """Bước A & B: Hiệu chuẩn D_cal^maps (80 bản đồ) và kiểm định D_test^maps (100 bản đồ)."""
    print("=" * 96)
    print("  [PHẦN A & B] HIỆU CHUẨN CONFORMAL ĐA BẢN ĐỒ & KIỂM ĐỊNH ZERO-SHOT (KHÔNG RÒ RỈ DỮ LIỆU)")
    print("=" * 96)

    cal_ds = generate_semantic_map_dataset(400, 80, road_class="B", variation_scale=0.15, n_wps=135)
    est = SemanticFrictionEstimator()
    cal_summary = est.calibrate_class_conformal(cal_ds, alpha_mu=0.05)
    est.save_calibration(cal_json_path)

    print(f"  [Cal D_cal^maps] N_cal={cal_summary['n_cal']} (seeds 400..479), alpha={cal_summary['alpha_mu']}")
    print(f"    q_hat_mu = {cal_summary['q_hat_mu']:.4f}, lag1_z = {cal_summary['lag1_z_stat']:.4f} "
          f"(exchangeability_ok={cal_summary['exchangeability_ok']})")

    test_ds = generate_semantic_map_dataset(600, 100, road_class="B", variation_scale=0.15, n_wps=135)
    est_loaded = SemanticFrictionEstimator.load_calibration(cal_json_path)
    cov_b0 = est_loaded.evaluate_coverage(test_ds, constant_override=0.35 * 0.95)
    cov_b1 = est_loaded.evaluate_coverage(test_ds, constant_override=0.10)
    cov_mean = est_loaded.evaluate_coverage(test_ds, mode="gp_mean")
    cov_b2 = est_loaded.evaluate_coverage(test_ds, mode="conformal_lcb")

    for name, c in [("B0 (0.3325)", cov_b0), ("B1 (0.10)", cov_b1), ("Prior Mean", cov_mean), ("B2 (Sem-LCB)", cov_b2)]:
        print(f"  [{name:<14}] TrajCov={c['traj_coverage_rate']*100:6.2f}% "
              f"(95% CP-LB={c['clopper_pearson_95_lb']:.3f}) | "
              f"PtCov={c['pointwise_coverage_rate']*100:6.2f}% | "
              f"GapMean={c['mean_conservatism_gap']:.4f} | Clamped={c['mean_clamped_mu_min_frac']*100:5.1f}%")

    # Kiểm tra bất biến tịnh tiến + xoay và phân hóa lớp ngữ nghĩa trên world6/7/8
    world_profiles = {}
    max_invariance_err = 0.0
    for w_name in ["world6.xml", "world7.xml", "world8.xml"]:
        smap = make_world_semantic_map(w_name)
        scen = build_map_scenario(w_name)
        wp = scen["wp_arr"]
        dx = np.gradient(wp[:, 0])
        dy = np.gradient(wp[:, 1])
        yaw = np.arctan2(dy, dx)

        p1 = est_loaded.predict_along_path(wp[:, 0], wp[:, 1], mode="conformal_lcb", yaw_grid=yaw, semantic_map=smap)
        th = 0.91
        c_t, s_t = math.cos(th), math.sin(th)
        wp_tx = c_t * wp[:, 0] - s_t * wp[:, 1] + 14.2
        wp_ty = s_t * wp[:, 0] + c_t * wp[:, 1] - 7.6
        p2 = est_loaded.predict_along_path(
            wp_tx, wp_ty, mode="conformal_lcb", yaw_grid=yaw + th,
            semantic_map=smap.transformed(14.2, -7.6, th),
        )
        inv_err = float(np.max(np.abs(p1["mu_lcb"] - p2["mu_lcb"])))
        max_invariance_err = max(max_invariance_err, inv_err)

        by_cls = {}
        for lbl in ("dry_concrete", "epoxy", "wet_risk", "unknown"):
            vals = [float(p1["mu_lcb"][i]) for i, c in enumerate(p1["center_class_names"]) if c == lbl]
            if vals:
                by_cls[lbl] = round(float(np.mean(vals)), 4)
        world_profiles[w_name] = {
            "mu_lcb_min": round(float(np.min(p1["mu_lcb"])), 4),
            "mu_lcb_max": round(float(np.max(p1["mu_lcb"])), 4),
            "mu_lcb_mean": round(float(np.mean(p1["mu_lcb"])), 4),
            "class_lcb_means": by_cls,
            "predict_time_ms": round(float(p1["predict_time_ms"]), 3),
        }
        print(f"  [Zero-Shot {w_name:<9}] LCB in [{p1['mu_lcb'].min():.3f}, {p1['mu_lcb'].max():.3f}], "
              f"classes={by_cls}, inv_err={inv_err:.2e}")

    return {
        "calibration_summary": {k: v for k, v in cal_summary.items() if k != "scores"},
        "coverage_test": {
            "B0_const_03325": cov_b0,
            "B1_const_010": cov_b1,
            "Prior_Mean": cov_mean,
            "B2_Semantic_Conformal_LCB": cov_b2,
        },
        "world_profiles": world_profiles,
        "max_transform_invariance_err": max_invariance_err,
    }


def run_hidden_oil_detection_trace() -> dict:
    """Bước C: Đo độ trễ phát hiện vũng dầu ẩn (mu=0.13) tại Tầng 2 sau khi P đã suy giảm ổn định."""
    print("\n" + "=" * 96)
    print("  [PHẦN C] KIỂM TRA ĐỘ TRỄ PHÁT HIỆN VŨNG DẦU ẨN TẦNG 2 (20 HZ, STEADY-STATE P)")
    print("=" * 96)

    Fz_total = M_TOTAL * 9.81
    s_h = np.linspace(0.0, 1.5, 16)
    mu_plan_h = np.full(16, 0.221, dtype=float)

    scenarios = {
        "cornering_oil_slip": {"rho_cmd": 0.16, "max_slip": 0.11},
        "low_rho_straight_oil_slip": {"rho_cmd": 0.045, "max_slip": 0.045},
    }
    traces = {}
    for name, cfg in scenarios.items():
        obs = FrictionObserverFFRLS(mu_init=0.221, lam=None, adaptive_forgetting=True, rho_deadzone=0.06)
        fus = OnlineFrictionFusion(mu_min=0.08, mu_max=0.55)
        # Cho xe chạy ổn định trên nền khô 80 chu kỳ (4s) để hiệp phương sai P suy giảm về trạng thái dừng
        for _ in range(80):
            obs.update([0.35 * Fz_total * np.tanh(0.02 / 0.04), 0.0], [10.0, 0.0], [0.02] * 4, Fz_total, M_TOTAL)
        p_before = float(obs.P)

        step_logs = []
        detect_cycle = None
        w_cmd = [float(cfg["rho_cmd"] * Fz_total), 0.0]
        slips = [float(cfg["max_slip"])] * 4
        w_act = [0.13 * Fz_total, 0.0]
        for k in range(1, 11):
            mu_hat, rho_k, bin_idx = obs.update(w_act, w_cmd, slips, Fz_total, M_TOTAL)
            mu_fuse = fus.fuse_horizon_mu(mu_plan_h, s_h + k * 0.03, mu_hat, bin_idx)
            mu_f0 = float(mu_fuse[0])
            step_logs.append({
                "cycle": k,
                "t_s": round(k * 0.05, 3),
                "mu_hat": round(float(mu_hat), 4),
                "bin_idx": int(bin_idx),
                "mu_fuse0": round(mu_f0, 4),
            })
            if detect_cycle is None and bin_idx >= 1 and mu_f0 <= 0.16:
                detect_cycle = k
        latency_s = None if detect_cycle is None else round(detect_cycle * 0.05, 3)
        traces[name] = {
            "P_before_oil": p_before,
            "detect_cycle": detect_cycle,
            "detect_latency_s": latency_s,
            "steps": step_logs[:6],
        }
        print(f"  [{name:<26}] P_pre={p_before:.2e} | Detect={latency_s}s ({detect_cycle} chu ky) | "
              f"mu_fuse[1..4]={[x['mu_fuse0'] for x in step_logs[:4]]}")
    return traces


def run_closed_loop_episode(
    map_name: str,
    method: str,
    semantic_estimator: SemanticFrictionEstimator,
    max_time: float = 60.0,
    enable_hidden_oil: bool = True,
) -> dict:
    """
    Chạy 1 lượt mô phỏng khép kín MuJoCo hoàn chỉnh trên bản đồ `map_name` theo chế độ:
      - 'B0': hằng số 0.35*0.95 = 0.3325
      - 'B1': hằng số bảo thủ 0.10
      - 'B2': Hướng đi 1 (Semantic Conformal LCB + Adaptive FF-RLS Fusion)
      - 'B3': Oracle (biết trước 4-wheel min trên Plant kể cả dầu ẩn)
    """
    cal_budget = load_conformal_calibration_budget()
    q_eta_use = float(cal_budget["recommended_q_eta_m"])

    scen = build_map_scenario(map_name)
    base_name = scen["map_name"]
    model = scen["model"]
    dyn_configs = scen["dyn_configs"]
    start_pos = scen["start_pos"]
    goal_pos = scen["goal_pos"]
    obs_circles = scen["obs_circles"]
    obs_boxes = scen["obs_boxes"]
    dyn_geoms = scen["dyn_geoms"]
    wp_arr = scen["wp_arr"]
    path_s = scen["path_s"]
    gate_descriptors = scen["gate_descriptors"]

    sem_map = make_world_semantic_map(base_name)
    hidden_oil_patches = _select_hidden_oil_patches(wp_arr, sem_map) if enable_hidden_oil else []
    plant_macro_fn = make_hidden_oil_macro_fn(sem_map.macro_friction_fn, hidden_oil_patches)

    controller = DFEMPCCAMRController(
        N=15,
        dt=0.05,
        mu=0.35,
        state_con_mode="skip0",
        mu_plan_scale=0.95,
        ipopt_max_iter=30,
        x_bounds=(-3.25, 3.25) if base_name == "world8.xml" else (-3.0, 3.0),
        y_bounds=(-3.25, 3.25) if base_name == "world8.xml" else (-3.0, 3.0),
    )
    controller.set_path(wp_arr)

    zc_fn = lambda s_val: 0.0555
    camber_fn = lambda s_val: 0.0
    dnet_fn = lambda s_val: 0.42

    tube_planner = CoupledVelocityTubeH124(mu_default=0.35, kappa_th=1.05)
    if method == "B0":
        tube_planner.build_offline_tube(
            wp_arr,
            mu_profile=lambda s: 0.35 * 0.95,
            phi_camber_profile=camber_fn, z_com_profile=zc_fn, d_net_profile=dnet_fn,
            run_wcsph_check=False, apply_plan_scale=False,
        )
    elif method == "B1":
        tube_planner.build_offline_tube(
            wp_arr,
            mu_profile=lambda s: 0.10,
            phi_camber_profile=camber_fn, z_com_profile=zc_fn, d_net_profile=dnet_fn,
            run_wcsph_check=False, apply_plan_scale=False,
        )
    elif method == "B2":
        est_map = SemanticFrictionEstimator(
            class_params=semantic_estimator.class_params,
            q_hat_mu=semantic_estimator.q_hat_mu,
            alpha_mu=semantic_estimator.alpha_mu,
            mu_min=semantic_estimator.mu_min,
            mu_max=semantic_estimator.mu_max,
            query_mode=semantic_estimator.query_mode,
            L_x=semantic_estimator.L_x,
            L_y=semantic_estimator.L_y,
            lateral_margin=semantic_estimator.lateral_margin,
            semantic_map=sem_map,
        )
        tube_planner.build_offline_tube(
            wp_arr,
            friction_estimator=est_map,
            friction_mode="conformal_lcb",
            phi_camber_profile=camber_fn, z_com_profile=zc_fn, d_net_profile=dnet_fn,
            run_wcsph_check=False,
        )
    elif method == "B3":
        mu_oracle_wps = np.array([plant_macro_fn(wx, wy) for wx, wy in wp_arr], dtype=float)
        tube_planner.build_offline_tube(
            wp_arr,
            mu_profile=lambda s: float(np.interp(s, path_s, mu_oracle_wps)),
            phi_camber_profile=camber_fn, z_com_profile=zc_fn, d_net_profile=dnet_fn,
            run_wcsph_check=False, apply_plan_scale=False,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    controller.set_tube_planner(tube_planner)
    controller.mu_plan_scale = float(getattr(tube_planner, "mu_effective_plan_scale", 1.0))

    road_profile = ISO8608RoadProfile(road_class="B", seed=42)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    from test_sim_amr_closed_loop import update_dynamic_obstacles
    update_dynamic_obstacles(model, data, 0.0, custom_dyn_configs=dyn_configs)
    data.qpos[0] = start_pos[0]
    data.qpos[1] = start_pos[1]
    _, _, heading_init = controller.get_path_reference(0.0)
    data.qpos[2] = heading_init
    mujoco.mj_forward(model, data)

    nmpc_dt = 0.05
    n_substeps = max(1, int(round(nmpc_dt / float(model.opt.timestep))))
    max_steps = int(round(max_time / nmpc_dt))

    mu_plan_init = float(np.asarray(tube_planner.get_horizon_mu_plan([0.0])).ravel()[0])
    friction_observer = FrictionObserverFFRLS(
        mu_init=mu_plan_init, lam=None, adaptive_forgetting=True, rho_deadzone=0.06
    )
    friction_fusion = OnlineFrictionFusion(mu_min=0.08, mu_max=0.55)
    mu_rls_curr = mu_plan_init
    bin_curr = 0
    mu_fuse_h = np.full(controller.N, mu_plan_init, dtype=float)
    was_waiting = False

    sim_time = 0.0
    s_prog = 0.0
    vs_prog = 0.25
    as_prog = 0.0
    a_act = np.zeros(3)
    dashing_gate_idx = -1
    far_obs = np.array([20.0, 20.0, 0.0, 0.1, 0.1])

    min_wall_overall = 10.0
    min_dyn_overall = 10.0
    max_contour_err = 0.0
    eta_history = []
    slip_history = []
    moving_slip_count = 0
    moving_steps = 0
    max_slip_overall = 0.0
    mu_fuse_history = []
    solve_times = []
    reached_goal = False

    for _ in range(max_steps):
        update_dynamic_obstacles(model, data, sim_time, custom_dyn_configs=dyn_configs)
        mujoco.mj_forward(model, data)

        q_meas = data.qpos[:3].copy()
        v_meas = data.qvel[:3].copy()
        phi_x_meas = float(data.qpos[3])
        dphi_x_meas = float(data.qvel[3])
        phi_y_meas = float(data.qpos[4])
        dphi_y_meas = float(data.qvel[4])

        v_fb = min(0.48, max(0.24, float(tube_planner.v_ub_nom[0]) if len(tube_planner.v_ub_nom) else 0.35))
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
            v_dash_bounds=(0.24, 0.48),
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
        mu_wheels_true = np.asarray(road_eval.get("mu_wheels", [plant_macro_fn(q_meas[0], q_meas[1])] * 4), dtype=float)

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
            dphi_y_meas,
        ])

        vs_prev_tube = max(float(np.clip(x_curr[10], 0.05, 0.75)), 0.40)
        s_vec_mpc = np.clip(s_prog + np.arange(controller.N, dtype=float) * controller.dt * vs_prev_tube, 0.0, controller.path_length)
        s_tube_h = controller.map_s_mpc_to_tube(s_vec_mpc)

        if method == "B0":
            mu_fuse_h = np.full(controller.N, 0.35 * 0.95, dtype=float)
        elif method == "B1":
            mu_fuse_h = np.full(controller.N, 0.10, dtype=float)
        elif method == "B2":
            mu_plan_h = tube_planner.get_horizon_mu_plan(s_tube_h)
            mu_fuse_h = friction_fusion.fuse_horizon_mu(
                mu_plan_h, s_tube_h, mu_rls_curr, bin_curr, s_curr=float(s_tube_h[0])
            )
        else:  # B3 Oracle
            mu_oracle_h = np.full(controller.N, float(np.min(mu_wheels_true)), dtype=float)
            mu_fuse_h = np.minimum(tube_planner.get_horizon_mu_plan(s_tube_h), mu_oracle_h)

        v_ub_h, v_lb_h = tube_planner.query_online_horizon_tube(s_tube_h, phi_est, mu_online=mu_fuse_h)
        v_ub_h = np.maximum(0.08, v_ub_h)
        v_lb_h = np.minimum(v_lb_h, v_ub_h)
        v_ref_cmd = float(np.clip(v_ub_h[0], 0.10, 0.68))
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
            moving_steps += 1
            obs_for_ctrl = select_tight_gate_obstacle(dyn_geoms, data, q_meas, far_obs) if is_tight_gate else far_obs
            u_opt, _, x_pred, _, _ = controller.solve_step(
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

            c_th0, s_th0 = math.cos(q_meas[2]), math.sin(q_meas[2])
            a_bx_cmd = c_th0 * a_cmd_world[0] + s_th0 * a_cmd_world[1]
            a_by_cmd = -s_th0 * a_cmd_world[0] + c_th0 * a_cmd_world[1]

            ddphi_x_dyn = -(omega_x ** 2) * phi_x_meas + (1.0 / L_P_X) * a_bx_cmd
            ddphi_y_dyn = -(omega_y ** 2) * phi_y_meas - (1.0 / L_P_Y) * a_by_cmd
            F_slosh_bx_dyn = -M_1X * (-L_P_X * ddphi_x_dyn * math.cos(phi_x_meas))
            F_slosh_by_dyn = -M_1Y * (L_P_Y * ddphi_y_dyn * math.cos(phi_y_meas))

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
        if method == "B2":
            mu_rls_curr, _, bin_curr = friction_observer.update(
                f_chassis_act_body[:2], W_cmd_body[:2], wheel_slips, float(np.sum(Fz_true)), M_TOTAL
            )

        max_slip_step = float(np.max(wheel_slips))
        slip_history.append(max_slip_step)
        if not should_wait and max_slip_step > 1e-3:
            moving_slip_count += 1
        if max_slip_step > max_slip_overall:
            max_slip_overall = max_slip_step
        mu_fuse_history.append(float(mu_fuse_h[0]))

        data.qpos[3] = np.clip(data.qpos[3] + (a_by0 / 9.81) * 0.08, -0.22, 0.22)
        data.qpos[4] = np.clip(data.qpos[4] - (a_bx0 / 9.81) * 0.08, -0.22, 0.22)

        for _ in range(n_substeps):
            mujoco.mj_step(model, data)

        sim_time += nmpc_dt
        update_dynamic_obstacles(model, data, sim_time, custom_dyn_configs=dyn_configs)
        mujoco.mj_forward(model, data)

        q_now = data.qpos[:3].copy()
        wc_now = compute_static_clearance(q_now[0], q_now[1], obs_boxes, obs_circles, r_robot=0.18)
        min_wall_overall = min(min_wall_overall, wc_now)

        for g_id, r_obs, _, _, _ in dyn_geoms:
            g_pos = data.geom_xpos[g_id][:2]
            d_clear = float(math.hypot(q_now[0] - g_pos[0], q_now[1] - g_pos[1]) - (r_obs + 0.20))
            min_dyn_overall = min(min_dyn_overall, d_clear)

        _, _, eta_curr = compute_world6_slosh_elevation_mm(
            float(data.qpos[3]), float(data.qvel[3]), float(data.qpos[4]), float(data.qvel[4]),
            a_bx0, a_by0, jerk_val, a_road_body[0], a_road_body[1],
            is_proposed=True, min_dyn_clearance=min_dyn_overall, has_dyn_obs=(len(dyn_geoms) > 0),
        )
        eta_history.append(float(eta_curr))

        p_ref_now, _, _ = controller.get_path_reference(s_prog)
        max_contour_err = max(max_contour_err, float(math.hypot(q_now[0] - p_ref_now[0], q_now[1] - p_ref_now[1])))

        dist_to_goal = float(math.hypot(q_now[0] - goal_pos[0], q_now[1] - goal_pos[1]))
        if dist_to_goal < 0.25 and s_prog > controller.path_length - 0.40:
            reached_goal = True
            break

    return {
        "map_name": base_name,
        "method": method,
        "reached_goal": bool(reached_goal),
        "s_final_m": round(float(s_prog), 3),
        "path_length_m": round(float(controller.path_length), 3),
        "t_finish_s": round(float(sim_time), 2),
        "slip_rate_pct": round(100.0 * moving_slip_count / max(1, moving_steps), 2),
        "max_slip": round(float(max_slip_overall), 4),
        "n_slip_steps": int(moving_slip_count),
        "n_moving_steps": int(moving_steps),
        "eta_max_mm": round(float(max(eta_history)) if eta_history else 0.0, 2),
        "eta_p95_mm": round(float(np.percentile(eta_history, 95)) if eta_history else 0.0, 2),
        "d_wall_min_m": round(float(min_wall_overall), 3),
        "d_dyn_min_m": round(float(min_dyn_overall), 3) if dyn_geoms else float("nan"),
        "ec_max_m": round(float(max_contour_err), 3),
        "mu_plan_mean": round(float(np.mean(tube_planner.mu_lcb_grid)), 4),
        "mu_fuse_min": round(float(min(mu_fuse_history)) if mu_fuse_history else mu_plan_init, 4),
        "solve_ms_mean": round(float(np.mean(solve_times)) if solve_times else 0.0, 2),
        "hidden_oil_patches": hidden_oil_patches,
    }


def main():
    t0 = time.perf_counter()
    cal_json_path = os.path.join(_ROOT, "config", "conformal_calibration_mu_semantic.json")

    part_ab = run_calibration_and_zeroshot_coverage(cal_json_path)
    part_c = run_hidden_oil_detection_trace()

    sem_est = SemanticFrictionEstimator.load_calibration(cal_json_path)

    print("\n" + "=" * 96)
    print("  [PHẦN D] SO SÁNH KHÉP KÍN MUJOCO B0-B3 TRÊN WORLD6/WORLD7/WORLD8 (CÓ VŨNG DẦU ẨN MU=0.13)")
    print("=" * 96)
    print(f"{'Map':<10} | {'Method':<6} | {'Goal':<5} | {'t_fin(s)':>8} | {'Slip(%)':>8} | "
          f"{'MaxSlip':>8} | {'eta_max(mm)':>11} | {'d_wall':>7} | {'d_dyn':>7} | {'mu_f_min':>8}")
    print("-" * 96)

    closed_loop_rows = []
    for map_name in ["world6.xml", "world7.xml", "world8.xml"]:
        for method in ["B0", "B1", "B2", "B3"]:
            res = run_closed_loop_episode(
                map_name=map_name,
                method=method,
                semantic_estimator=sem_est,
                max_time=60.0,
                enable_hidden_oil=True,
            )
            closed_loop_rows.append(res)
            print(f"{res['map_name']:<10} | {res['method']:<6} | {str(res['reached_goal']):<5} | "
                  f"{res['t_finish_s']:8.2f} | {res['slip_rate_pct']:8.2f} | {res['max_slip']:8.4f} | "
                  f"{res['eta_max_mm']:11.2f} | {res['d_wall_min_m']:+7.3f} | {res['d_dyn_min_m']:+7.3f} | "
                  f"{res['mu_fuse_min']:8.4f}", flush=True)
        print("-" * 96, flush=True)

    # Tổng hợp theo phương pháp
    agg = {}
    for m in ["B0", "B1", "B2", "B3"]:
        rows = [r for r in closed_loop_rows if r["method"] == m]
        agg[m] = {
            "n_maps": len(rows),
            "goal_rate": round(float(np.mean([1.0 if r["reached_goal"] else 0.0 for r in rows])), 3),
            "mean_t_finish_s": round(float(np.mean([r["t_finish_s"] for r in rows])), 2),
            "mean_slip_rate_pct": round(float(np.mean([r["slip_rate_pct"] for r in rows])), 2),
            "max_slip_overall": round(float(np.max([r["max_slip"] for r in rows])), 4),
            "eta_max_overall_mm": round(float(np.max([r["eta_max_mm"] for r in rows])), 2),
            "d_wall_min_overall_m": round(float(np.min([r["d_wall_min_m"] for r in rows])), 3),
            "d_dyn_min_overall_m": round(float(np.min([r["d_dyn_min_m"] for r in rows])), 3),
        }

    print("\n  TỔNG HỢP TRUNG BÌNH 3 BẢN ĐỒ (WORLD6, WORLD7, WORLD8):")
    for m, st in agg.items():
        print(f"    {m}: Goal={st['goal_rate']*100:.0f}%, mean_t_fin={st['mean_t_finish_s']:.2f}s, "
              f"mean_slip_rate={st['mean_slip_rate_pct']:.2f}%, max_slip={st['max_slip_overall']:.4f}, "
              f"eta_max={st['eta_max_overall_mm']:.2f}mm, d_wall_min={st['d_wall_min_overall_m']:+.3f}m, "
              f"d_dyn_min={st['d_dyn_min_overall_m']:+.3f}m")

    out_path = os.path.join(_ROOT, "simulation", "zeroshot_friction_ablation_results.json")
    payload = {
        "total_runtime_s": round(time.perf_counter() - t0, 2),
        "calibration_json": cal_json_path,
        "zeroshot_coverage_and_invariance": part_ab,
        "tier2_hidden_oil_latency": part_c,
        "closed_loop_per_map": closed_loop_rows,
        "closed_loop_summary": agg,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n  [✓] Đã lưu kết quả benchmark vào: {out_path} ({payload['total_runtime_s']:.1f} s)")


if __name__ == "__main__":
    main()
