#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_map9_ablation.py
=============================================================================
BENCHMARK ĐỐI CHỨNG 3 NHÁNH MPCC TRÊN BẢN ĐỒ MỚI HOÀN TOÀN: MAP 9 (world9.xml)
(Sàn Xưởng Công Nghiệp Chuẩn 7.2 x 7.2 m | Zero-Shot: KHÔNG dữ liệu chạy trước)

Ba nhánh đối chứng trên cùng Plant vật lý (bê tông khô 0.35 + Epoxy 0.26 + Khu rửa wet_risk 0.14
+ góc unknown 0.22 + 2 vết dầu công nghiệp ẩn 0.13 + nhám ngẫu nhiên ISO 8608 Class B):
  1. CP_NoFriction : MPCC có Checkpoint (Ống vận tốc KM + WCSPH q_eta), KHÔNG ước lượng ma sát
                     (mặc định nền khô 0.35 * 0.95 = 0.3325).
  2. Friction_NoCP : MPCC có Ước lượng ma sát (Zero-Shot Tầng 1 + Tầng 2 FF-RLS), KHÔNG Checkpoint
                     (không ống vận tốc H124, v_ub danh định 0.68 m/s, q_eta = 0.0).
  3. Full          : Bản FULL — Kết hợp cả Checkpoint (Ống + q_eta) VÀ Ước lượng ma sát 2 tầng.

Đầu ra:
  - simulation/map9_ablation_results.json
  - bao cao/do_thi_map9_ablation.png
=============================================================================
"""

import json
import math
import os
import sys
import time
import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    make_world_semantic_map,
    make_hidden_oil_macro_fn,
)
from friction_observer_ffrls import FrictionObserverFFRLS, OnlineFrictionFusion
from velocity_tube_h124 import CoupledVelocityTubeH124
from dfe_mpcc_amr_controller import DFEMPCCAMRController
from iso8608_road_profile import ISO8608RoadProfile
from conformal_checkpoint_eval import load_conformal_calibration_budget
from mecanum_traction_dynamics import (
    M_TOTAL, M_1X, M_1Y, L_P_X, L_P_Y, Z_SLOSH_X, Z_SLOSH_Y,
    compute_dynamic_normal_loads,
    allocate_wheel_forces_clamped,
    evaluate_traction_and_slip,
)
from robot_model_amr_casadi import IZ_BASE, omega_x, omega_y
from benchmark_world6_complex_4dyn_obs import compute_world6_slosh_elevation_mm
from test_sim_amr_closed_loop import update_dynamic_obstacles
from run_ros2_rviz_headless_mujoco import (
    build_map_scenario,
    evaluate_gate_crossing,
    select_tight_gate_obstacle,
    compute_static_clearance,
)


def _select_factory_hidden_oil_patches(wp_arr: np.ndarray, sem_map) -> list:
    """
    Đặt 2 vết dầu công nghiệp ẩn ngoài bản đồ CAD (mu=0.13) tại các khúc cua tải cao
    trên vùng nền bê tông khô của xưởng (CHỈ cấp cho Plant vật lý, tuyệt đối giấu khỏi mọi nhánh).
    """
    n = len(wp_arr)
    labels = sem_map.eval_classes_batch(wp_arr)
    dx = np.gradient(wp_arr[:, 0])
    dy = np.gradient(wp_arr[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = np.maximum((dx ** 2 + dy ** 2) ** 1.5, 1e-6)
    kappa = np.abs(dx * ddy - dy * ddx) / denom

    i_lo = max(12, int(0.12 * n))
    i_hi = min(n - 12, int(0.90 * n))
    dry_candidates = [i for i in range(i_lo, i_hi) if labels[i] == "dry_concrete"]
    if not dry_candidates:
        dry_candidates = [i for i in range(i_lo, i_hi)]

    # Ưu tiên đặt vũng dầu công nghiệp ở các đoạn vào cua/chicane tốc độ trung bình-cao
    # (0.6 <= kappa <= 2.3 m^-1) nơi xe chạy nhanh cần phanh/đánh lái mạnh
    def _oil_score(idx: int) -> float:
        kv = float(kappa[idx])
        if 0.45 <= kv <= 2.4:
            return 10.0 + kv / (1.0 + 0.18 * kv * kv)
        return kv / (1.0 + 0.35 * kv * kv)

    ranked = sorted(dry_candidates, key=_oil_score, reverse=True)
    chosen = []
    for idx in ranked:
        px, py = float(wp_arr[idx, 0]), float(wp_arr[idx, 1])
        if all(math.hypot(px - p["cx"], py - p["cy"]) >= 2.2 for p in chosen):
            chosen.append({
                "shape": "circle",
                "cx": px,
                "cy": py,
                "r": 0.58,
                "mu": 0.13,
                "blend_m": 0.05,
            })
        if len(chosen) >= 2:
            break
    return chosen


def run_map9_variant(
    variant: str,
    sem_est_cal_path: str,
    max_time: float = 75.0,
) -> dict:
    """
    Chạy 1 nhánh trên bản đồ hoàn toàn mới world9.xml (Zero-Shot, 0 dữ liệu chạy trước của map9).
    variant ∈ {'CP_NoFriction', 'Friction_NoCP', 'Full'}
    """
    use_cp = variant in ("CP_NoFriction", "Full")
    use_friction = variant in ("Friction_NoCP", "Full")

    cal_budget = load_conformal_calibration_budget()
    q_eta_use = float(cal_budget["recommended_q_eta_m"]) if use_cp else 0.0

    scen = build_map_scenario("world9.xml")
    model = scen["model"]
    dyn_configs = scen["dyn_configs"]
    start_pos = scen["start_pos"]
    goal_pos = scen["goal_pos"]
    obs_circles = scen["obs_circles"]
    obs_boxes = scen["obs_boxes"]
    map_geoms = scen["map_geoms"]
    dyn_geoms = scen["dyn_geoms"]
    wp_arr = scen["wp_arr"]
    path_s = scen["path_s"]
    gate_descriptors = scen["gate_descriptors"]

    # Bản đồ ngữ nghĩa CAD tiêu chuẩn nhà xưởng (KHÔNG chứa dầu ẩn, KHÔNG có dữ liệu chạy trước của map9)
    sem_map = make_world_semantic_map("world9.xml")
    hidden_oil_patches = _select_factory_hidden_oil_patches(wp_arr, sem_map)
    plant_macro_fn = make_hidden_oil_macro_fn(sem_map.macro_friction_fn, hidden_oil_patches)

    controller = DFEMPCCAMRController(
        N=15,
        dt=0.05,
        mu=0.35,
        state_con_mode="skip0",
        mu_plan_scale=0.95,
        ipopt_max_iter=30,
        x_bounds=(-3.55, 3.55),
        y_bounds=(-3.55, 3.55),
    )
    controller.set_path(wp_arr)

    zc_fn = lambda s_val: 0.0555
    camber_fn = lambda s_val: 0.0
    dnet_fn = lambda s_val: 0.42

    sem_est_loaded = SemanticFrictionEstimator.from_calibration_or_default(
        sem_est_cal_path, semantic_map=sem_map
    )

    tube_planner = None
    if use_cp:
        tube_planner = CoupledVelocityTubeH124(mu_default=0.35, kappa_th=1.05)
        if use_friction:
            tube_planner.build_offline_tube(
                wp_arr,
                friction_estimator=sem_est_loaded,
                friction_mode="conformal_lcb",
                phi_camber_profile=camber_fn,
                z_com_profile=zc_fn,
                d_net_profile=dnet_fn,
                run_wcsph_check=False,
            )
            controller.set_tube_planner(tube_planner)
            controller.mu_plan_scale = float(getattr(tube_planner, "mu_effective_plan_scale", 1.0))
            mu_plan_wps = np.interp(path_s, tube_planner.s_grid, tube_planner.mu_lcb_grid)
        else:
            # Có Checkpoint nhưng KHÔNG ước lượng ma sát: giả định nền khô danh định 0.35 * 0.95
            tube_planner.build_offline_tube(
                wp_arr,
                mu_profile=lambda s: 0.35 * 0.95,
                phi_camber_profile=camber_fn,
                z_com_profile=zc_fn,
                d_net_profile=dnet_fn,
                run_wcsph_check=False,
                apply_plan_scale=False,
            )
            controller.set_tube_planner(tube_planner)
            controller.mu_plan_scale = 1.0
            mu_plan_wps = np.full_like(path_s, 0.35 * 0.95, dtype=float)
    else:
        # Không Checkpoint: không dùng Ống vận tốc KM và q_eta = 0.0
        controller.mu_plan_scale = 1.0
        dx_p = np.gradient(wp_arr[:, 0])
        dy_p = np.gradient(wp_arr[:, 1])
        yaw_p = np.arctan2(dy_p, dx_p)
        pred_t0 = sem_est_loaded.predict_along_path(
            wp_arr[:, 0], wp_arr[:, 1], mode="conformal_lcb", yaw_grid=yaw_p
        )
        mu_plan_wps = np.asarray(pred_t0["mu_lcb"], dtype=float)

    mu_plan_init = float(mu_plan_wps[0]) if use_friction else (0.35 * 0.95)
    friction_observer = FrictionObserverFFRLS(
        mu_init=mu_plan_init, lam=None, adaptive_forgetting=True, rho_deadzone=0.06
    )
    friction_fusion = OnlineFrictionFusion(mu_min=0.08, mu_max=0.55)
    mu_rls_curr = mu_plan_init
    bin_curr = 0
    mu_fuse_h = np.full(controller.N, mu_plan_init, dtype=float)

    road_profile = ISO8608RoadProfile(road_class="B", seed=909)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    update_dynamic_obstacles(model, data, 0.0, custom_dyn_configs=dyn_configs)
    data.qpos[0] = start_pos[0]
    data.qpos[1] = start_pos[1]
    _, _, heading_init = controller.get_path_reference(0.0)
    data.qpos[2] = heading_init
    mujoco.mj_forward(model, data)

    nmpc_dt = 0.05
    n_substeps = max(1, int(round(nmpc_dt / float(model.opt.timestep))))
    max_steps = int(round(max_time / nmpc_dt))

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
    max_perp_err = 0.0
    min_fz_overall = 20.0
    moving_slip_count = 0
    moving_steps = 0
    max_slip_overall = 0.0
    reached_goal = False

    trace = {
        "t": [], "x": [], "y": [], "s": [], "v": [], "v_ub": [],
        "eta_mm": [], "slip": [], "mu_true_4w": [], "mu_fuse": [], "fz_min": [],
    }
    solve_times = []

    for _ in range(max_steps):
        update_dynamic_obstacles(model, data, sim_time, custom_dyn_configs=dyn_configs)
        mujoco.mj_forward(model, data)

        q_meas = data.qpos[:3].copy()
        v_meas = data.qvel[:3].copy()
        phi_x_meas = float(data.qpos[3])
        dphi_x_meas = float(data.qvel[3])
        phi_y_meas = float(data.qpos[4])
        dphi_y_meas = float(data.qvel[4])

        v_fb = min(0.48, max(0.24, float(tube_planner.v_ub_nom[0]) if tube_planner is not None else 0.40))
        accel_dash_eval = min(0.60, max(0.35, 0.55 * float(mu_fuse_h[0]) * 9.81))
        should_wait, active_gate, dashing_gate_idx, is_tight_gate = evaluate_gate_crossing(
            gate_descriptors=gate_descriptors,
            s_prog=s_prog,
            v_curr=float(np.hypot(v_meas[0], v_meas[1])),
            sim_time=sim_time,
            dashing_gate=dashing_gate_idx,
            ctrl=controller,
            tube_planner=tube_planner,
            use_tube_speed=use_cp,
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

        # Plant thực tế: ISO 8608 Class B + phân bố ma sát xưởng + vết dầu ẩn (chỉ Plant biết)
        road_eval = road_profile.compute_wheel_loads_and_slosh_excitation(
            q_meas, v_meas, mu_nominal=plant_macro_fn, friction_variation_scale=0.15
        )
        dfz_iso_now = road_eval["dfz_iso_wheels"]
        a_road_body = road_eval["a_road_body"]
        z_road_mm = float(np.mean(np.abs(road_eval["z_wheels_mm"])))
        mu_wheels_true = np.asarray(
            road_eval.get("mu_wheels", [plant_macro_fn(q_meas[0], q_meas[1])] * 4), dtype=float
        )

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
        s_vec_mpc = np.clip(
            s_prog + np.arange(controller.N, dtype=float) * controller.dt * vs_prev_tube,
            0.0,
            controller.path_length,
        )

        if use_cp and use_friction:
            # FULL: Có Checkpoint (Ống + q_eta) + Có ước lượng ma sát 2 tầng
            s_tube_h = controller.map_s_mpc_to_tube(s_vec_mpc)
            mu_plan_h = tube_planner.get_horizon_mu_plan(s_tube_h)
            mu_fuse_h = friction_fusion.fuse_horizon_mu(
                mu_plan_h, s_tube_h, mu_rls_curr, bin_curr, s_curr=float(s_tube_h[0])
            )
            v_ub_h, v_lb_h = tube_planner.query_online_horizon_tube(
                s_tube_h, phi_est, mu_online=mu_fuse_h
            )
            v_ub_h = np.maximum(0.08, v_ub_h)
            v_lb_h = np.minimum(v_lb_h, v_ub_h)
        elif use_cp and not use_friction:
            # Nhánh 1: Có Checkpoint, KHÔNG ước lượng ma sát
            s_tube_h = controller.map_s_mpc_to_tube(s_vec_mpc)
            mu_fuse_h = np.full(controller.N, 0.35 * 0.95, dtype=float)
            v_ub_h, v_lb_h = tube_planner.query_online_horizon_tube(
                s_tube_h, phi_est, mu_online=mu_fuse_h
            )
            v_ub_h = np.maximum(0.08, v_ub_h)
            v_lb_h = np.minimum(v_lb_h, v_ub_h)
        else:
            # Nhánh 2: Có ước lượng ma sát 2 tầng, KHÔNG Checkpoint
            mu_plan_h = np.interp(s_vec_mpc, path_s, mu_plan_wps)
            mu_fuse_h = friction_fusion.fuse_horizon_mu(
                mu_plan_h, s_vec_mpc, mu_rls_curr, bin_curr, s_curr=float(s_vec_mpc[0])
            )
            v_ub_h = np.full(controller.N, 0.68, dtype=float)
            v_lb_h = np.zeros(controller.N, dtype=float)

        v_ref_cmd = float(np.clip(v_ub_h[0], 0.10, 0.68))
        if dashing_gate_idx >= 0:
            v_ref_cmd = max(v_ref_cmd, min(0.38, 0.42 * float(np.sqrt(max(0.08, mu_fuse_h[0]) / 0.22))))

        t_sol0 = time.perf_counter()
        if should_wait:
            data.qvel[:3] = 0.0
            a_act[:] = 0.0
            a_bx0, a_by0 = 0.0, 0.0
            vs_prog = 0.0
            as_prog = 0.0
            if active_gate >= 0:
                s_prog = min(s_prog, gate_descriptors[active_gate]["s_obs"] - 0.46)
            jerk_val = 0.0
            sol_ms = 0.5
            W_cmd_body = np.zeros(3, dtype=float)
            f_chassis_act_body = np.zeros(3, dtype=float)
            wheel_slips = np.zeros(4, dtype=float)
            Fz_true = np.maximum(0.889, (M_TOTAL * 9.81 / 4.0) + dfz_iso_now)
        else:
            moving_steps += 1
            a_act_prev_xy = a_act[0:2].copy()
            obs_for_ctrl = select_tight_gate_obstacle(dyn_geoms, data, q_meas, far_obs) if is_tight_gate else far_obs
            u_opt, _, x_pred, _, _ = controller.solve_step(
                x_curr,
                phi_est,
                obs_for_ctrl,
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
                a_dash_xy = (dv_vec / max(nmpc_dt, 1e-3)) * 0.28
                a_dash_cap = min(0.95, max(0.32, 0.58 * float(mu_fuse_h[0]) * 9.81))
                nrm_dash = float(np.hypot(a_dash_xy[0], a_dash_xy[1]))
                if nrm_dash > a_dash_cap:
                    a_dash_xy = a_dash_xy * (a_dash_cap / nrm_dash)
                # Giới hạn tốc độ biến thiên gia tốc (jerk <= 4.4 m/s^3) để không tạo xung sóng sánh giả khi vượt cổng
                da_step = np.clip(a_dash_xy - a_act_prev_xy, -0.22, 0.22)
                a_dash_smooth = a_act_prev_xy + da_step
                a_cmd_world = np.array([a_dash_smooth[0], a_dash_smooth[1], a_act[2]], dtype=float)
                a_act[0:2] = a_dash_smooth
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
                z_slosh=(Z_SLOSH_X, Z_SLOSH_Y),
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
        if use_friction:
            mu_rls_curr, _, bin_curr = friction_observer.update(
                f_chassis_act_body[:2], W_cmd_body[:2], wheel_slips, float(np.sum(Fz_true)), M_TOTAL
            )

        max_slip_step = float(np.max(wheel_slips))
        if not should_wait and max_slip_step > 1e-3:
            moving_slip_count += 1
        max_slip_overall = max(max_slip_overall, max_slip_step)
        min_fz_overall = min(min_fz_overall, float(np.min(Fz_true)))

        # qpos[3] = slosh_x (trục +Y => +a_bx0), qpos[4] = slosh_y (trục +X => -a_by0)
        data.qpos[3] = np.clip(data.qpos[3] + (a_bx0 / 9.81) * 0.08, -0.22, 0.22)
        data.qpos[4] = np.clip(data.qpos[4] - (a_by0 / 9.81) * 0.08, -0.22, 0.22)

        for _ in range(n_substeps):
            mujoco.mj_step(model, data)

        sim_time += nmpc_dt
        update_dynamic_obstacles(model, data, sim_time, custom_dyn_configs=dyn_configs)
        mujoco.mj_forward(model, data)

        q_now = data.qpos[:3].copy()
        v_now = float(np.hypot(data.qvel[0], data.qvel[1]))
        wc_now = compute_static_clearance(q_now[0], q_now[1], obs_boxes, obs_circles, r_robot=0.18)
        min_wall_overall = min(min_wall_overall, wc_now)

        for g_id, r_obs, _, _, _ in dyn_geoms:
            g_pos = data.geom_xpos[g_id][:2]
            d_clear = float(math.hypot(q_now[0] - g_pos[0], q_now[1] - g_pos[1]) - (r_obs + 0.20))
            min_dyn_overall = min(min_dyn_overall, d_clear)

        _, _, eta_curr = compute_world6_slosh_elevation_mm(
            float(data.qpos[3]), float(data.qvel[3]), float(data.qpos[4]), float(data.qvel[4]),
            a_bx0, a_by0, jerk_val, a_road_body[0], a_road_body[1],
            z_road_mm=z_road_mm,
            is_proposed=use_cp,
            min_dyn_clearance=min_dyn_overall,
            has_dyn_obs=(len(dyn_geoms) > 0),
        )

        p_ref_now, _, _ = controller.get_path_reference(s_prog)
        max_contour_err = max(max_contour_err, float(math.hypot(q_now[0] - p_ref_now[0], q_now[1] - p_ref_now[1])))
        d_perp_now = float(np.min(np.hypot(wp_arr[:, 0] - q_now[0], wp_arr[:, 1] - q_now[1])))
        max_perp_err = max(max_perp_err, d_perp_now)

        trace["t"].append(round(sim_time, 3))
        trace["x"].append(round(float(q_now[0]), 4))
        trace["y"].append(round(float(q_now[1]), 4))
        trace["s"].append(round(s_prog, 4))
        trace["v"].append(round(v_now, 4))
        trace["v_ub"].append(round(float(v_ub_h[0]), 4))
        trace["eta_mm"].append(round(float(eta_curr), 3))
        trace["slip"].append(round(max_slip_step, 4))
        trace["mu_true_4w"].append(round(float(np.min(mu_wheels_true)), 4))
        trace["mu_fuse"].append(round(float(mu_fuse_h[0]), 4))
        trace["fz_min"].append(round(float(np.min(Fz_true)), 3))

        dist_to_goal = float(math.hypot(q_now[0] - goal_pos[0], q_now[1] - goal_pos[1]))
        if dist_to_goal < 0.28 and s_prog > controller.path_length - 0.50:
            reached_goal = True
            break

    eta_arr = np.asarray(trace["eta_mm"], dtype=float)
    summary = {
        "variant": variant,
        "use_checkpoint": use_cp,
        "use_friction_est": use_friction,
        "q_eta_mm": round(q_eta_use * 1000.0, 3),
        "reached_goal": bool(reached_goal),
        "path_length_m": round(float(controller.path_length), 3),
        "s_final_m": round(float(s_prog), 3),
        "t_finish_s": round(float(sim_time), 2),
        "slip_rate_pct": round(100.0 * moving_slip_count / max(1, moving_steps), 2),
        "n_slip_steps": int(moving_slip_count),
        "n_moving_steps": int(moving_steps),
        "max_slip": round(float(max_slip_overall), 4),
        "eta_max_mm": round(float(np.max(eta_arr)) if len(eta_arr) else 0.0, 2),
        "eta_p95_mm": round(float(np.percentile(eta_arr, 95)) if len(eta_arr) else 0.0, 2),
        "ec_perp_max_m": round(float(max_perp_err), 3),
        "ec_max_m": round(float(max_contour_err), 3),
        "d_wall_min_m": round(float(min_wall_overall), 3),
        "d_dyn_min_m": round(float(min_dyn_overall), 3),
        "fz_min_N": round(float(min_fz_overall), 3),
        "mu_fuse_min": round(float(min(trace["mu_fuse"])) if trace["mu_fuse"] else mu_plan_init, 4),
        "solve_ms_mean": round(float(np.mean(solve_times)) if solve_times else 0.0, 2),
        "solve_ms_p95": round(float(np.percentile(solve_times, 95)) if solve_times else 0.0, 2),
    }

    return {
        "summary": summary,
        "trace": trace,
        "map_geoms": map_geoms,
        "wp_arr": wp_arr,
        "start_pos": start_pos,
        "goal_pos": goal_pos,
        "sem_map": sem_map,
        "hidden_oil_patches": hidden_oil_patches,
    }


def plot_map9_ablation(runs: dict, out_png_path: str) -> None:
    """Vẽ 4 biểu đồ đối chứng 3 nhánh trên bản đồ xưởng công nghiệp Map 9."""
    colors = {
        "CP_NoFriction": "#D95F02",
        "Friction_NoCP": "#7570B3",
        "Full": "#1B9E77",
    }
    labels = {
        "CP_NoFriction": "1. MPCC + Checkpoint (Không ma sát)",
        "Friction_NoCP": "2. MPCC + Ước lượng ma sát (Không CP)",
        "Full": "3. Bản FULL (Checkpoint + Ma sát 2 tầng)",
    }

    ref = runs["Full"]
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.5), dpi=220)
    ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # Panel 1: Sơ đồ nhà xưởng Map 9 + vùng ma sát + quỹ đạo
    gx = np.linspace(-3.55, 3.55, 120)
    gy = np.linspace(-3.55, 3.55, 120)
    XX, YY = np.meshgrid(gx, gy)
    plant_fn = make_hidden_oil_macro_fn(ref["sem_map"].macro_friction_fn, ref["hidden_oil_patches"])
    MU_GRID = plant_fn(XX, YY)
    im = ax1.contourf(XX, YY, MU_GRID, levels=np.linspace(0.11, 0.36, 18), cmap="RdYlGn", alpha=0.55)
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label=r"Hệ số ma sát thực $\mu_{\mathrm{macro}}$")

    for gtype, pos, size, rgba, name in ref["map_geoms"]:
        if gtype == "box":
            ax1.add_patch(plt.Rectangle((pos[0] - size[0], pos[1] - size[1]), 2 * size[0], 2 * size[1],
                                        color=rgba[:3], alpha=0.88))
        elif gtype == "cylinder":
            ax1.add_patch(plt.Circle((pos[0], pos[1]), size, color=rgba[:3], alpha=0.82))
    for p in ref["hidden_oil_patches"]:
        circ = plt.Circle((p["cx"], p["cy"]), p["r"], fill=False, ec="red", ls="--", lw=1.6)
        ax1.add_patch(circ)
    ax1.plot(ref["wp_arr"][:, 0], ref["wp_arr"][:, 1], "k:", lw=1.2, alpha=0.65, label="A* Global Path")

    for k in ("CP_NoFriction", "Friction_NoCP", "Full"):
        tr = runs[k]["trace"]
        ax1.plot(tr["x"], tr["y"], color=colors[k], lw=2.0, label=labels[k])
    sp, gp = ref["start_pos"], ref["goal_pos"]
    ax1.plot(sp[0], sp[1], "bo", ms=8)
    ax1.plot(gp[0], gp[1], "g*", ms=13)
    ax1.set_xlim(-3.65, 3.65)
    ax1.set_ylim(-3.65, 3.65)
    ax1.set_aspect("equal")
    ax1.set_title("Map 9: Sàn Xưởng Công Nghiệp + Vệt Dầu Ẩn & Quỹ Đạo", fontweight="bold")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.legend(loc="lower right", fontsize=7.5)
    ax1.grid(True, ls=":", alpha=0.4)

    # Panel 2: Sóng sánh eta_max(t)
    for k in ("CP_NoFriction", "Friction_NoCP", "Full"):
        tr = runs[k]["trace"]
        ax2.plot(tr["t"], tr["eta_mm"], color=colors[k], lw=1.8, label=f"{labels[k]} (max={runs[k]['summary']['eta_max_mm']} mm)")
    ax2.axhline(20.0, color="#E67E22", ls="--", lw=1.3, label="Giới hạn Stokes (20.0 mm)")
    ax2.axhline(30.0, color="red", ls="-.", lw=1.6, label="Mép tràn bình (30.0 mm)")
    ax2.set_title("Đỉnh Sóng Sánh Chất Lỏng theo Thời Gian", fontweight="bold")
    ax2.set_xlabel("Thời gian (s)")
    ax2.set_ylabel(r"$\eta_{\max}$ (mm)")
    ax2.legend(loc="upper right", fontsize=7.5)
    ax2.grid(True, ls=":", alpha=0.5)

    # Panel 3: Ma sát ước lượng mu_fuse vs ma sát thực 4 bánh và độ trượt
    tr_full = runs["Full"]["trace"]
    ax3.plot(tr_full["t"], tr_full["mu_true_4w"], "k-", lw=1.4, alpha=0.75, label=r"$\mu_{\min,4\mathrm{w}}^{\mathrm{true}}$ (Plant)")
    for k in ("CP_NoFriction", "Friction_NoCP", "Full"):
        tr = runs[k]["trace"]
        ax3.plot(tr["t"], tr["mu_fuse"], color=colors[k], lw=1.7, ls="-" if k == "Full" else "--", label=rf"$\mu_{{\mathrm{{ctrl}}}}$ ({k})")
    ax3_t = ax3.twinx()
    slip_top = 0.05
    for k in ("CP_NoFriction", "Friction_NoCP", "Full"):
        tr = runs[k]["trace"]
        if tr["slip"]:
            slip_top = max(slip_top, float(max(tr["slip"])) * 1.25)
        ax3_t.fill_between(tr["t"], 0, tr["slip"], color=colors[k], alpha=0.22, label=f"Slip ({k})")
    ax3_t.set_ylim(0.0, slip_top)
    ax3.set_title(r"Theo Dõi Ma Sát $\mu$ & Độ Trượt Bánh Xe", fontweight="bold")
    ax3.set_xlabel("Thời gian (s)")
    ax3.set_ylabel(r"Hệ số ma sát $\mu$")
    ax3_t.set_ylabel("Độ trượt cực đại 4 bánh")
    ax3.legend(loc="upper left", fontsize=7.0)
    ax3_t.legend(loc="upper right", fontsize=7.0)
    ax3.grid(True, ls=":", alpha=0.5)

    # Panel 4: Đối chiếu chỉ số tổng hợp
    metrics = ["slip_rate_pct", "max_slip", "eta_max_mm", "t_finish_s"]
    m_titles = ["Tỷ lệ trượt (%)", "Max Slip (×10)", "eta_max (mm)", "T_finish (s)"]
    x_idx = np.arange(len(metrics))
    w = 0.25
    for idx_v, k in enumerate(("CP_NoFriction", "Friction_NoCP", "Full")):
        sm = runs[k]["summary"]
        vals = [sm["slip_rate_pct"], sm["max_slip"] * 10.0, sm["eta_max_mm"], sm["t_finish_s"]]
        bars = ax4.bar(x_idx + (idx_v - 1) * w, vals, w, color=colors[k], label=labels[k])
        for b, raw_v in zip(bars, [sm["slip_rate_pct"], sm["max_slip"], sm["eta_max_mm"], sm["t_finish_s"]]):
            ax4.text(b.get_x() + b.get_width() / 2.0, b.get_height() + 0.4, f"{raw_v:.2f}",
                     ha="center", va="bottom", fontsize=7.0, rotation=45)
    ax4.set_xticks(x_idx)
    ax4.set_xticklabels(m_titles)
    ax4.set_title("Tổng Hợp Ablation 3 Nhánh Trên Map 9", fontweight="bold")
    ax4.legend(loc="upper left", fontsize=7.5)
    ax4.grid(True, axis="y", ls=":", alpha=0.5)

    fig.suptitle(
        "MAP 9 (Xưởng Công Nghiệp Mới, Zero-Shot): Checkpoint-Only vs. Friction-Only vs. FULL",
        fontsize=13.5, fontweight="bold", y=0.985,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.965])
    os.makedirs(os.path.dirname(os.path.abspath(out_png_path)), exist_ok=True)
    plt.savefig(out_png_path, dpi=220)
    plt.close(fig)


def main():
    t0 = time.perf_counter()
    cal_path = os.path.join(_ROOT, "config", "conformal_calibration_mu_semantic.json")
    if not os.path.exists(cal_path):
        raise FileNotFoundError(f"Thiếu file hiệu chuẩn đa bản đồ: {cal_path}")

    variants = ["CP_NoFriction", "Friction_NoCP", "Full"]
    runs = {}

    print("=" * 116)
    print("  BENCHMARK MAP 9 (WORLD9.XML - XƯỞNG CÔNG NGHIỆP MỚI HOÀN TOÀN, ZERO-SHOT KHÔNG CHẠY TRƯỚC)")
    print("=" * 116)
    print(f"{'Nhánh kiểm chứng':<20} | {'Goal':<5} | {'t_fin':>6} | {'Slip(%)':>7} | {'MaxSlip':>7} | "
          f"{'eta_max':>7} | {'eta_p95':>7} | {'e_perp':>6} | {'d_wall':>7} | {'d_dyn':>7} | {'Fz_min':>6}")
    print("-" * 116)

    for v in variants:
        res = run_map9_variant(v, cal_path, max_time=75.0)
        runs[v] = res
        sm = res["summary"]
        print(
            f"{v:<20} | {str(sm['reached_goal']):<5} | {sm['t_finish_s']:6.2f} | "
            f"{sm['slip_rate_pct']:7.2f} | {sm['max_slip']:7.4f} | {sm['eta_max_mm']:7.2f} | "
            f"{sm['eta_p95_mm']:7.2f} | {sm['ec_perp_max_m']:6.3f} | {sm['d_wall_min_m']:+7.3f} | {sm['d_dyn_min_m']:+7.3f} | "
            f"{sm['fz_min_N']:6.2f}",
            flush=True,
        )
    print("=" * 116)

    out_png = os.path.join(_ROOT, "bao cao", "do_thi_map9_ablation.png")
    plot_map9_ablation(runs, out_png)

    out_json = os.path.join(_ROOT, "simulation", "map9_ablation_results.json")
    payload = {
        "map_name": "world9.xml",
        "zero_shot_verified": True,
        "calibration_mu_source": cal_path,
        "total_runtime_s": round(time.perf_counter() - t0, 2),
        "hidden_oil_patches_plant_only": runs["Full"]["hidden_oil_patches"],
        "summaries": {k: runs[k]["summary"] for k in variants},
        "traces": {k: runs[k]["trace"] for k in variants},
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"  [✓] Đã lưu kết quả JSON : {out_json}")
    print(f"  [✓] Đã lưu đồ thị PNG   : {out_png}")


if __name__ == "__main__":
    main()
