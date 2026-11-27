#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_dual_layer_ablation.py
=============================================================================
BENCHMARK ĐỐI CHỨNG ABLATION TRÊN ĐA BẢN ĐỒ (world6.xml & world9.xml):
So sánh 4 biến thể trên cùng Plant vật lý (sàn đa vùng + 2 vũng dầu ẩn mu = 0.09
+ nhám ngẫu nhiên ISO 8608 Class B). Mọi nhánh KHÔNG được biết trước bản đồ vùng
hay vũng dầu; các nhánh có ước lượng ma sát CHỈ được cấp khoảng nền [0.28, 0.42]:

  1. CP_NoFriction : Có Checkpoint (Ống vận tốc H124 + q_eta), KHÔNG ước lượng ma sát
                     (giả định cố định 0.35 * 0.95 = 0.3325).
  2. Friction_NoCP : Có Ước lượng ma sát Hướng 2 (Dual-Layer), KHÔNG Checkpoint
                     (không ống vận tốc H124, v_ub = 0.68 m/s, q_eta = 0.0).
  3. Full_Lap1     : Bản FULL (Checkpoint H124 + q_eta + Dual-Layer) — Lượt đầu tiên (Zero-Shot).
  4. Full_Lap2     : Bản FULL — Lượt thứ hai dùng lại trí nhớ dị thường 2D đã học ở Lượt 1.

Đầu ra:
  - simulation/dual_layer_ablation_results.json
  - bao cao/do_thi_dual_layer_ablation.png
=============================================================================
"""

import json
import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

_CURR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_CURR)
for _sub in ["", "planning", "controller", "solver", "simulation", "cfd"]:
    _p = os.path.join(_ROOT, _sub) if _sub else _ROOT
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from benchmark_map9_ablation import _select_factory_hidden_oil_patches
from benchmark_world6_complex_4dyn_obs import compute_world6_slosh_elevation_mm
from conformal_checkpoint_eval import load_conformal_calibration_budget
from controller.dfe_mpcc_amr_controller import DFEMPCCAMRController
from controller.iso8608_road_profile import ISO8608RoadProfile
from controller.mecanum_traction_dynamics import (
    L_P_X,
    L_P_Y,
    M_1X,
    M_1Y,
    M_TOTAL,
    Z_SLOSH_X,
    Z_SLOSH_Y,
    allocate_wheel_forces_clamped,
    compute_dynamic_normal_loads,
    evaluate_traction_and_slip,
)
from planning.dual_layer_friction_estimator import DualLayerFrictionEstimator
from planning.semantic_friction_estimator import (
    make_hidden_oil_macro_fn,
    make_world_semantic_map,
)
from planning.velocity_tube_h124 import CoupledVelocityTubeH124
from run_ros2_rviz_headless_mujoco import (
    build_map_scenario,
    compute_static_clearance,
    evaluate_gate_crossing,
    select_tight_gate_obstacle,
)
from solver.robot_model_amr_casadi import IZ_BASE, omega_x, omega_y
from test_sim_amr_closed_loop import update_dynamic_obstacles


def run_variant_episode(
    map_name: str,
    variant: str,
    dual_estimator: DualLayerFrictionEstimator = None,
    max_time: float = 75.0,
) -> dict:
    """
    Chạy 1 lượt mô phỏng khép kín MuJoCo cho biến thể `variant` trên `map_name`.
    variant ∈ {'CP_NoFriction', 'Friction_NoCP', 'Full_Lap1', 'Full_Lap2'}
    """
    use_cp = variant in ("CP_NoFriction", "Full_Lap1", "Full_Lap2")
    use_friction = variant in ("Friction_NoCP", "Full_Lap1", "Full_Lap2")

    cal_budget = load_conformal_calibration_budget()
    q_eta_use = float(cal_budget["recommended_q_eta_m"]) if use_cp else 0.0

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

    # Plant vật lý: nền đa vùng + 2 vũng dầu ẩn mu=0.09 (giấu hoàn toàn khỏi mọi nhánh)
    sem_map_plant = make_world_semantic_map(base_name)
    hidden_oil_patches = _select_factory_hidden_oil_patches(wp_arr, sem_map_plant)
    for p in hidden_oil_patches:
        p["mu"] = 0.09
        p["r"] = 0.58
        p["blend_m"] = 0.05
    plant_macro_fn = make_hidden_oil_macro_fn(
        sem_map_plant.macro_friction_fn, hidden_oil_patches, mu_clip=(0.06, 0.60)
    )

    bounds = (-3.55, 3.55) if "world9" in base_name else (-3.0, 3.0)
    controller = DFEMPCCAMRController(
        N=12,
        dt=0.05,
        mu=0.35,
        state_con_mode="skip0",
        mu_plan_scale=0.95,
        ipopt_max_iter=25,
        x_bounds=bounds,
        y_bounds=bounds,
    )
    controller.set_path(wp_arr)

    zc_fn = lambda s_val: 0.0555
    camber_fn = lambda s_val: 0.0
    dnet_fn = lambda s_val: 0.42

    if use_friction:
        if dual_estimator is None:
            est = DualLayerFrictionEstimator(
                mu_env_low=0.28,
                mu_env_high=0.42,
                mu_hard_min=0.05,
            )
        else:
            est = dual_estimator
            est.reset(keep_anomaly_map=(variant == "Full_Lap2"))
    else:
        est = None

    tube_planner = None
    if use_cp:
        tube_planner = CoupledVelocityTubeH124(mu_default=0.35, kappa_th=1.05)
        if use_friction:
            tube_planner.build_offline_tube(
                wp_arr,
                friction_estimator=est,
                friction_mode="conformal_lcb",
                phi_camber_profile=camber_fn,
                z_com_profile=zc_fn,
                d_net_profile=dnet_fn,
                run_wcsph_check=False,
            )
            controller.set_tube_planner(tube_planner)
            controller.mu_plan_scale = float(getattr(tube_planner, "mu_effective_plan_scale", 1.0))
        else:
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
    else:
        controller.mu_plan_scale = 1.0

    yaw_wps = np.arctan2(np.gradient(wp_arr[:, 1]), np.gradient(wp_arr[:, 0]))
    sin_yaw_wps = np.sin(yaw_wps)
    cos_yaw_wps = np.cos(yaw_wps)
    if tube_planner is not None:
        sin_yaw_tube = np.sin(tube_planner.yaw_grid)
        cos_yaw_tube = np.cos(tube_planner.yaw_grid)

    mu_init = float(est.bg_model.get_safe_bg()) if use_friction else (0.35 * 0.95)
    mu_fuse_h = np.full(controller.N, mu_init, dtype=float)

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
    min_fz_overall = 20.0
    moving_slip_count = 0
    moving_steps = 0
    max_slip_overall = 0.0
    reached_goal = False

    trace = {
        "t": [], "s": [], "v": [], "eta_mm": [],
        "slip": [], "mu_true_4w": [], "mu_fuse": [],
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
            s_tube_h = controller.map_s_mpc_to_tube(s_vec_mpc)
            x_h = np.interp(s_tube_h, tube_planner.s_grid, tube_planner.x_grid)
            y_h = np.interp(s_tube_h, tube_planner.s_grid, tube_planner.y_grid)
            yaw_h = np.arctan2(
                np.interp(s_tube_h, tube_planner.s_grid, sin_yaw_tube),
                np.interp(s_tube_h, tube_planner.s_grid, cos_yaw_tube),
            )
            mu_fuse_h = est.fuse_horizon_xy(
                x_h, y_h, yaw_h, s_tube_h, t_now=sim_time, s_curr=float(s_tube_h[0])
            )
            v_ub_h, v_lb_h = tube_planner.query_online_horizon_tube(
                s_tube_h, phi_est, mu_online=mu_fuse_h
            )
            v_ub_h = np.maximum(0.08, v_ub_h)
            v_lb_h = np.minimum(v_lb_h, v_ub_h)
        elif use_cp and not use_friction:
            s_tube_h = controller.map_s_mpc_to_tube(s_vec_mpc)
            mu_fuse_h = np.full(controller.N, 0.35 * 0.95, dtype=float)
            v_ub_h, v_lb_h = tube_planner.query_online_horizon_tube(
                s_tube_h, phi_est, mu_online=mu_fuse_h
            )
            v_ub_h = np.maximum(0.08, v_ub_h)
            v_lb_h = np.minimum(v_lb_h, v_ub_h)
        else:
            x_h = np.interp(s_vec_mpc, path_s, wp_arr[:, 0])
            y_h = np.interp(s_vec_mpc, path_s, wp_arr[:, 1])
            yaw_h = np.arctan2(
                np.interp(s_vec_mpc, path_s, sin_yaw_wps),
                np.interp(s_vec_mpc, path_s, cos_yaw_wps),
            )
            mu_fuse_h = est.fuse_horizon_xy(
                x_h, y_h, yaw_h, s_vec_mpc, t_now=sim_time, s_curr=float(s_vec_mpc[0])
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
            est.update_online(
                q_meas[0], q_meas[1], q_meas[2],
                f_chassis_act_body[:2], W_cmd_body[:2], wheel_slips,
                float(np.sum(Fz_true)), M_TOTAL, t_now=sim_time,
            )

        max_slip_step = float(np.max(wheel_slips))
        if not should_wait and max_slip_step > 1e-3:
            moving_slip_count += 1
        max_slip_overall = max(max_slip_overall, max_slip_step)
        min_fz_overall = min(min_fz_overall, float(np.min(Fz_true)))

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

        trace["t"].append(round(sim_time, 3))
        trace["s"].append(round(s_prog, 4))
        trace["v"].append(round(v_now, 4))
        trace["eta_mm"].append(round(float(eta_curr), 3))
        trace["slip"].append(round(max_slip_step, 4))
        trace["mu_true_4w"].append(round(float(np.min(mu_wheels_true)), 4))
        trace["mu_fuse"].append(round(float(mu_fuse_h[0]), 4))

        dist_to_goal = float(math.hypot(q_now[0] - goal_pos[0], q_now[1] - goal_pos[1]))
        if dist_to_goal < 0.28 and s_prog > controller.path_length - 0.50:
            reached_goal = True
            break

    eta_arr = np.asarray(trace["eta_mm"], dtype=float)
    summary = {
        "map_name": base_name,
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
        "fz_min_N": round(float(min_fz_overall), 3),
        "d_wall_min_m": round(float(min_wall_overall), 3),
        "d_dyn_min_m": round(float(min_dyn_overall), 3),
        "mu_fuse_min": round(float(min(trace["mu_fuse"])) if trace["mu_fuse"] else mu_init, 4),
        "mu_bg_final": round(float(est.bg_model.mu_bg), 4) if est is not None else None,
        "n_anomaly_cells": int(len(est.anomaly_map.cells)) if est is not None else 0,
        "solve_ms_mean": round(float(np.mean(solve_times)) if solve_times else 0.0, 2),
        "solve_ms_p95": round(float(np.percentile(solve_times, 95)) if solve_times else 0.0, 2),
    }

    return {
        "summary": summary,
        "trace": trace,
        "estimator": est,
        "hidden_oil_patches": hidden_oil_patches,
    }


def plot_dual_layer_ablation(all_results: dict, out_png_path: str) -> None:
    """Vẽ biểu đồ 2x2 đối chứng 4 biến thể trên world6.xml và world9.xml."""
    variants = ["CP_NoFriction", "Friction_NoCP", "Full_Lap1", "Full_Lap2"]
    v_labels = ["CP_NoFric", "Fric_NoCP", "Full_L1", "Full_L2"]
    colors = ["#D95F02", "#7570B3", "#1B9E77", "#0F766E"]
    maps = ["world6.xml", "world9.xml"]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), dpi=160)
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]

    x_base = np.arange(len(maps))
    w = 0.18

    # (0, 0): Slip Rate (%)
    ax = axes[0, 0]
    for i, (var, lbl, col) in enumerate(zip(variants, v_labels, colors)):
        vals = [all_results[m][var]["summary"]["slip_rate_pct"] for m in maps]
        bars = ax.bar(x_base + (i - 1.5) * w, vals, width=w, label=lbl, color=col, alpha=0.92)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.2f}%", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x_base)
    ax.set_xticklabels(["world6.xml (13.1 m)", "world9.xml (20.0 m)"])
    ax.set_ylabel("Tỷ lệ bước trượt (%)")
    ax.set_title("(a) So sánh Tỷ lệ bước trượt trên 2 bản đồ có dầu ẩn (mu=0.09)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    # (0, 1): Peak & p95 Slosh (mm)
    ax = axes[0, 1]
    for i, (var, lbl, col) in enumerate(zip(variants, v_labels, colors)):
        vals_max = [all_results[m][var]["summary"]["eta_max_mm"] for m in maps]
        vals_p95 = [all_results[m][var]["summary"]["eta_p95_mm"] for m in maps]
        bars = ax.bar(x_base + (i - 1.5) * w, vals_max, width=w, label=f"{lbl} max", color=col, alpha=0.85)
        ax.bar(x_base + (i - 1.5) * w, vals_p95, width=w * 0.55, color="#0F172A", alpha=0.35)
        for b, v in zip(bars, vals_max):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.35, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)
    ax.axhline(30.0, color="#DC2626", ls="--", lw=1.2, label="Mép tràn 30 mm")
    ax.axhline(15.0, color="#F59E0B", ls=":", lw=1.2, label="Thiết kế 15 mm")
    ax.set_xticks(x_base)
    ax.set_xticklabels(["world6.xml", "world9.xml"])
    ax.set_ylabel("Độ dâng sóng eta (mm)")
    ax.set_title("(b) Đỉnh sóng max (màu) và p95 (lõi đậm)")
    ax.set_ylim(0, max(24.0, max(all_results[m][v]["summary"]["eta_max_mm"] for m in maps for v in variants) + 5.0))
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7.5, ncol=2, loc="upper left")

    # (1, 0): world9.xml mu_fuse vs mu_true_4w
    ax = axes[1, 0]
    tr_full1 = all_results["world9.xml"]["Full_Lap1"]["trace"]
    tr_full2 = all_results["world9.xml"]["Full_Lap2"]["trace"]
    tr_cp_nf = all_results["world9.xml"]["CP_NoFriction"]["trace"]
    ax.plot(tr_full1["s"], tr_full1["mu_true_4w"], "k--", lw=1.3, label="mu_true_4w (Plant)", alpha=0.8)
    ax.plot(tr_cp_nf["s"], tr_cp_nf["mu_fuse"], color=colors[0], lw=1.3, ls="-.", label="CP_NoFriction (0.3325)")
    ax.plot(tr_full1["s"], tr_full1["mu_fuse"], color=colors[2], lw=1.8, label="Full_Lap1 (Zero-Shot)")
    ax.plot(tr_full2["s"], tr_full2["mu_fuse"], color=colors[3], lw=1.8, ls=":", label="Full_Lap2 (Nhớ 2D)")
    ax.set_xlabel("Quãng đường cung s (m)")
    ax.set_ylabel("Hệ số ma sát mu")
    ax.set_title("(c) world9.xml: Dự báo mu_fuse dọc quỹ đạo (Lượt 1 vs Lượt 2)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")

    # (1, 1): world9.xml eta_mm
    ax = axes[1, 1]
    tr_fr_nc = all_results["world9.xml"]["Friction_NoCP"]["trace"]
    ax.plot(tr_fr_nc["t"], tr_fr_nc["eta_mm"], color=colors[1], lw=1.5, label="Friction_NoCP", alpha=0.9)
    ax.plot(tr_cp_nf["t"], tr_cp_nf["eta_mm"], color=colors[0], lw=1.3, label="CP_NoFriction", alpha=0.8)
    ax.plot(tr_full1["t"], tr_full1["eta_mm"], color=colors[2], lw=1.7, label="Full_Lap1")
    ax.plot(tr_full2["t"], tr_full2["eta_mm"], color=colors[3], lw=1.6, ls="--", label="Full_Lap2")
    ax.axhline(15.0, color="#F59E0B", ls=":", lw=1.1, label="Ngưỡng thiết kế 15 mm")
    ax.set_xlabel("Thời gian (s)")
    ax.set_ylabel("Độ dâng sóng eta (mm)")
    ax.set_title("(d) world9.xml: Diễn biến độ dâng sóng theo thời gian")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png_path)), exist_ok=True)
    fig.savefig(out_png_path, bbox_inches="tight")
    plt.close(fig)


def main():
    t0 = time.perf_counter()
    print("=" * 102)
    print("BENCHMARK ABLATION: CP_NoFriction vs Friction_NoCP vs Full_Lap1 vs Full_Lap2 (Dual-Layer)")
    print("=" * 102)

    maps = ["world6.xml", "world9.xml"]
    all_results = {}
    json_payload = {"maps": {}, "mu_env_range": [0.28, 0.42], "hidden_oil_mu": 0.09}

    for m_name in maps:
        print(f"\n>>> Đang chạy bản đồ {m_name} ...")
        all_results[m_name] = {}
        json_payload["maps"][m_name] = {"summaries": {}, "traces": {}, "hidden_oil_patches": []}

        r_cp = run_variant_episode(m_name, "CP_NoFriction")
        r_fr = run_variant_episode(m_name, "Friction_NoCP")
        r_f1 = run_variant_episode(m_name, "Full_Lap1")
        r_f2 = run_variant_episode(m_name, "Full_Lap2", dual_estimator=r_f1["estimator"])

        for var, res in [
            ("CP_NoFriction", r_cp),
            ("Friction_NoCP", r_fr),
            ("Full_Lap1", r_f1),
            ("Full_Lap2", r_f2),
        ]:
            all_results[m_name][var] = res
            json_payload["maps"][m_name]["summaries"][var] = res["summary"]
            json_payload["maps"][m_name]["traces"][var] = res["trace"]
        json_payload["maps"][m_name]["hidden_oil_patches"] = r_f1["hidden_oil_patches"]

    total_s = time.perf_counter() - t0
    json_payload["total_runtime_s"] = round(total_s, 2)

    out_json = os.path.join(_ROOT, "simulation", "dual_layer_ablation_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, ensure_ascii=False)

    out_png = os.path.join(_ROOT, "bao cao", "do_thi_dual_layer_ablation.png")
    plot_dual_layer_ablation(all_results, out_png)

    print("\n" + "=" * 102)
    print(f"{'Map':<11} | {'Variant':<14} | {'Goal':<4} | {'t(s)':>5} | {'Slip(%)':>7} | {'MaxSlip':>7} | "
          f"{'eta_max':>7} | {'eta_p95':>7} | {'Cells':>5} | {'SolveTB/p95':>12}")
    print("-" * 102)
    for m_name in maps:
        for var in ["CP_NoFriction", "Friction_NoCP", "Full_Lap1", "Full_Lap2"]:
            s = all_results[m_name][var]["summary"]
            print(
                f"{m_name:<11} | {var:<14} | {str(s['reached_goal']):<4} | {s['t_finish_s']:5.2f} | "
                f"{s['slip_rate_pct']:6.2f}% | {s['max_slip']:7.4f} | {s['eta_max_mm']:7.2f} | "
                f"{s['eta_p95_mm']:7.2f} | {s['n_anomaly_cells']:5d} | "
                f"{s['solve_ms_mean']:5.1f}/{s['solve_ms_p95']:5.1f}"
            )
        print("-" * 102)
    print(f"Hoàn tất trong {total_s:.2f} s. JSON: {out_json} | PNG: {out_png}")


if __name__ == "__main__":
    main()
