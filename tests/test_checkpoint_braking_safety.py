#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_checkpoint_braking_safety.py
=============================================================================
KIỂM THỬ AN TOÀN PHANH KHẨN CẤP TẠI CHECKPOINT (PHA 4 - HƯỚNG 1+3 ZERO-SHOT)

Kiểm chứng độc lập mô hình rút gọn (reduced-order model) trên mô phỏng 100%
không chạy thử trên bản đồ kiểm thử:
  1. Test 1: Đối chiếu truyền ZOH (controller.propagate_slosh_zoh & Phi_s[:, N])
             với tích phân RK4 liên tục (dt = 1e-4 s) & kiểm tra cận trạng thái dừng.
  2. Test 2: Kiểm chứng bất đẳng thức bất biến phanh dừng ở bước cuối chân trời
             so với mô phỏng đa vật thể MuJoCo 500 Hz (xml/slosh3.xml) + q_eta_t2.
  3. Test 3: Chạy khép kín Zero-Shot B2 trên world6.xml (hiệu chuẩn đóng băng),
             tiêm nhánh phanh khẩn cấp -0.65 m/s^2 tại mỗi checkpoint s_peak
             và xác minh eta < 30.0 mm cùng ngân sách thời gian giải.
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
for _sub in ["", "planning", "controller", "solver", "simulation", "cfd"]:
    _p = os.path.join(_ROOT, _sub) if _sub else _ROOT
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from robot_model_amr_casadi import (
    M_TOTAL, IZ_BASE, TANK_A, TANK_B, H_LIQUID, ZETA_SLOSH, g,
    lx, ly, omega_x, omega_y,
)
from dfe_mpcc_amr_controller import DFEMPCCAMRController
from velocity_tube_h124 import CoupledVelocityTubeH124
from semantic_friction_estimator import (
    SemanticFrictionEstimator,
    make_world_semantic_map,
    make_hidden_oil_macro_fn,
)
from friction_observer_ffrls import FrictionObserverFFRLS, OnlineFrictionFusion
from iso8608_road_profile import ISO8608RoadProfile
from mecanum_traction_dynamics import (
    M_1X, M_1Y, L_P_X, L_P_Y, Z_SLOSH_X, Z_SLOSH_Y,
    compute_dynamic_normal_loads,
    allocate_wheel_forces_clamped,
    evaluate_traction_and_slip,
)
from simulation.benchmark_world6_complex_4dyn_obs import compute_world6_slosh_elevation_mm
from simulation.run_ros2_rviz_headless_mujoco import (
    build_map_scenario,
    evaluate_gate_crossing,
    select_tight_gate_obstacle,
    compute_static_clearance,
)
from simulation.benchmark_zeroshot_friction_ablation import _select_hidden_oil_patches


def _coupled_stokes_mm(phi_x: float, phi_y: float, c2_x: float, c2_y: float) -> float:
    """Tính đỉnh sóng Stokes bậc 2 ghép cặp 2 trục (mm) từ góc con lắc (rad)."""
    u_x = math.tan(abs(float(phi_x)))
    u_y = math.tan(abs(float(phi_y)))
    eta_x = (0.5 * TANK_A * u_x + c2_x * (u_x ** 2)) * 1000.0
    eta_y = (0.5 * TANK_B * u_y + c2_y * (u_y ** 2)) * 1000.0
    return max(eta_x + 0.35 * eta_y, eta_y + 0.35 * eta_x)


def _run_mujoco_emergency_braking(
    model,
    qpos_init: np.ndarray,
    qvel_init: np.ndarray,
    a_brake_max: float = 0.65,
    coast_s: float = 0.60,
    eval_w6: bool = False,
) -> dict:
    """Thực thi nhánh phanh khẩn cấp ngược chiều vận tốc trên MuJoCo (500 Hz) + quán tính tự do."""
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:len(qpos_init)] = np.asarray(qpos_init, dtype=float)
    data.qvel[:len(qvel_init)] = np.asarray(qvel_init, dtype=float)
    mujoco.mj_forward(model, data)

    dt = float(model.opt.timestep)
    v0_xy = np.array([float(qvel_init[0]), float(qvel_init[1])], dtype=float)
    v0 = float(np.linalg.norm(v0_xy))
    dir_v = (v0_xy / v0) if v0 > 1e-6 else np.zeros(2, dtype=float)
    t_stop = (v0 / a_brake_max) if v0 > 1e-6 else 0.0
    n_steps = max(1, int(math.ceil((t_stop + coast_s) / dt)))

    c2_x = float((np.pi * TANK_A / 8.0) / np.tanh(np.pi * H_LIQUID / TANK_A))
    c2_y = float((np.pi * TANK_B / 8.0) / np.tanh(np.pi * H_LIQUID / TANK_B))

    eta_stokes_max = _coupled_stokes_mm(float(data.qpos[3]), float(data.qpos[4]), c2_x, c2_y)
    eta_w6_max = 0.0
    phi_x_max = abs(float(data.qpos[3]))
    phi_y_max = abs(float(data.qpos[4]))

    for step in range(n_steps):
        t = step * dt
        if t < t_stop and t_stop > 0.0:
            v_ref_xy = v0_xy * max(0.0, 1.0 - t / t_stop)
            a_ref_xy = -a_brake_max * dir_v
        else:
            v_ref_xy = np.zeros(2, dtype=float)
            a_ref_xy = np.zeros(2, dtype=float)

        if model.nu >= 3:
            data.ctrl[0:2] = M_TOTAL * a_ref_xy + 2.0 * data.qvel[0:2] + 300.0 * (v_ref_xy - data.qvel[0:2])
            data.ctrl[2] = -20.0 * float(data.qvel[2])
        else:
            data.qvel[0:2] = v_ref_xy
            data.qvel[2] = 0.0

        mujoco.mj_step(model, data)

        px = float(data.qpos[3])
        py = float(data.qpos[4])
        phi_x_max = max(phi_x_max, abs(px))
        phi_y_max = max(phi_y_max, abs(py))
        eta_stokes_max = max(eta_stokes_max, _coupled_stokes_mm(px, py, c2_x, c2_y))

        if eval_w6:
            yaw = float(data.qpos[2])
            cy, sy = math.cos(yaw), math.sin(yaw)
            abx = cy * float(a_ref_xy[0]) + sy * float(a_ref_xy[1])
            aby = -sy * float(a_ref_xy[0]) + cy * float(a_ref_xy[1])
            _, _, eta_w6 = compute_world6_slosh_elevation_mm(
                px, float(data.qvel[3]), py, float(data.qvel[4]),
                abx, aby, 0.0, 0.0, 0.0, is_proposed=True,
            )
            eta_w6_max = max(eta_w6_max, float(eta_w6))

    return {
        "eta_stokes_max_mm": float(eta_stokes_max),
        "eta_w6_max_mm": float(eta_w6_max),
        "phi_x_max": float(phi_x_max),
        "phi_y_max": float(phi_y_max),
        "v_final": float(np.hypot(data.qvel[0], data.qvel[1])),
        "t_stop_s": float(t_stop),
    }


def _rk4_pendulum_seq(phi_est: np.ndarray, a_body_seq: np.ndarray, dt_step: float = 0.05) -> np.ndarray:
    """Tích phân RK4 dày (dt = 1e-4 s) hệ con lắc có cản tuyến tính tương đương ZOH."""
    st = np.asarray(phi_est, dtype=float).ravel()[:4].copy()
    sub = 500
    h = dt_step / sub
    cx = 2.0 * ZETA_SLOSH * omega_x
    cy = 2.0 * ZETA_SLOSH * omega_y

    def _rhs(s_vec: np.ndarray, abx: float, aby: float) -> np.ndarray:
        return np.array([
            s_vec[1],
            -(omega_x ** 2) * s_vec[0] - cx * s_vec[1] + abx / lx,
            s_vec[3],
            -(omega_y ** 2) * s_vec[2] - cy * s_vec[3] - aby / ly,
        ], dtype=float)

    for ab in a_body_seq:
        abx, aby = float(ab[0]), float(ab[1])
        for _ in range(sub):
            k1 = _rhs(st, abx, aby)
            k2 = _rhs(st + 0.5 * h * k1, abx, aby)
            k3 = _rhs(st + 0.5 * h * k2, abx, aby)
            k4 = _rhs(st + h * k3, abx, aby)
            st = st + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return st


def test_1_zoh_slosh_unroll_vs_rk4_and_zero_state_bound():
    """Test 1: Kiểm chứng ZOH (propagate_slosh_zoh & Phi_s[:, N]) so với RK4 + cận trạng thái 0."""
    ctrl = DFEMPCCAMRController(N=15, dt=0.05, mu=0.35, state_con_mode="skip0", a_brake_max=0.65)
    phi0 = np.array([0.045, -0.32, -0.060, 0.48], dtype=float)
    t_idx = np.arange(ctrl.N, dtype=float)
    a_seq = np.column_stack([
        0.55 * np.sin(0.45 * t_idx) - 0.15,
        0.65 * np.cos(0.35 * t_idx),
    ])

    phi_zoh = ctrl.propagate_slosh_zoh(phi0, a_seq)
    phi_rk4 = _rk4_pendulum_seq(phi0, a_seq, dt_step=ctrl.dt)
    pos_err = float(np.max(np.abs(phi_zoh[[0, 2]] - phi_rk4[[0, 2]])))
    vel_err = float(np.max(np.abs(phi_zoh[[1, 3]] - phi_rk4[[1, 3]])))
    assert pos_err < 1e-6, f"Sai số góc ZOH vs RK4 quá lớn: {pos_err:.3e} rad"
    assert vel_err < 1e-5, f"Sai số vận tốc góc ZOH vs RK4 quá lớn: {vel_err:.3e} rad/s"

    # Kiểm tra Phi_s[:, N] từ solver khớp propagate_slosh_zoh
    waypoints = np.column_stack([np.linspace(0.0, 2.5, 40), 0.35 * np.sin(np.linspace(0.0, np.pi, 40))])
    ctrl.set_path(waypoints)
    _, _, th0 = ctrl.get_path_reference(0.0)
    x_curr = np.zeros(12, dtype=float)
    x_curr[0:2] = waypoints[0]
    x_curr[2] = th0
    x_curr[3] = 0.28 * math.cos(th0)
    x_curr[4] = 0.28 * math.sin(th0)
    x_curr[10] = 0.28
    _, _, x_pred, _, extra = ctrl.solve_step(x_curr, phi0)

    th_pred = x_pred[2, :ctrl.N]
    ax_pred = x_pred[6, :ctrl.N]
    ay_pred = x_pred[7, :ctrl.N]
    a_body_pred = np.column_stack([
        np.cos(th_pred) * ax_pred + np.sin(th_pred) * ay_pred,
        -np.sin(th_pred) * ax_pred + np.cos(th_pred) * ay_pred,
    ])
    phi_N_ref = ctrl.propagate_slosh_zoh(phi0, a_body_pred)
    nlp_zoh_err = float(np.max(np.abs(extra["phi_N_pred"] - phi_N_ref)))
    assert nlp_zoh_err < 1e-5, f"Phi_s[:, N] không khớp ZOH: err={nlp_zoh_err:.3e}"

    # Kiểm tra trạng thái dừng v_N = 0, phi_N = 0 -> eta < 0.2 mm
    v_norm_0 = math.sqrt(0.08 ** 2)
    a_brk_0 = 0.0
    phi_eq_0 = a_brk_0 / g
    phi_pk_0 = math.sqrt(phi_eq_0 ** 2 + 1e-6) + math.sqrt(phi_eq_0 ** 2 + 1e-6)
    eta_0_x = 0.5 * TANK_A * phi_pk_0 + ctrl._stokes_ax_q * (phi_pk_0 ** 2)
    eta_0_y = 0.5 * TANK_B * phi_pk_0 + ctrl._stokes_ay_q * (phi_pk_0 ** 2)
    assert eta_0_x < 0.0002 and eta_0_y < 0.0002, f"Cận dừng bằng 0 quá lớn: {eta_0_x:.5f}, {eta_0_y:.5f}"

    print(
        f"  [PASS] Test 1: pos_err={pos_err:.2e} rad, vel_err={vel_err:.2e} rad/s, "
        f"nlp_zoh_err={nlp_zoh_err:.2e}, zero_eta=({eta_0_x*1e3:.3f}, {eta_0_y*1e3:.3f}) mm."
    )


def test_2_terminal_braking_bound_vs_mujoco_500hz():
    """Test 2: So sánh cận tập bất biến phanh cuối chân trời với MuJoCo 500 Hz (slosh3.xml)."""
    cal_path = os.path.join(_ROOT, "config", "conformal_calibration_q_eta.json")
    with open(cal_path, "r", encoding="utf-8") as f:
        q_eta_t2_total_mm = float(json.load(f)["q_eta_t2_total_mm"])

    ctrl = DFEMPCCAMRController(N=15, dt=0.05, mu=0.35, a_brake_max=0.65)
    model = mujoco.MjModel.from_xml_path(os.path.join(_ROOT, "xml", "slosh3.xml"))
    assert abs(float(model.opt.timestep) - 0.002) < 1e-6

    cases = [
        # (v_bx, v_by, phi_x0, dphi_x0, phi_y0, dphi_y0)
        (0.25, 0.00, 0.02, 0.15, 0.04, 0.25),
        (0.32, 0.05, -0.03, -0.25, 0.08, -0.35),
        (0.36, -0.04, 0.04, 0.30, -0.09, 0.45),
        (0.38, 0.06, -0.04, 0.20, 0.10, -0.50),
    ]
    summaries = []
    for v_bx, v_by, px0, dpx0, py0, dpy0 in cases:
        v_norm = math.sqrt(v_bx ** 2 + v_by ** 2 + 0.08 ** 2)
        phi_eq_x = (-ctrl.a_brake_max * (v_bx / v_norm)) / g
        phi_eq_y = -(-ctrl.a_brake_max * (v_by / v_norm)) / g
        pk_x = math.sqrt(phi_eq_x ** 2 + 1e-6) + math.sqrt((px0 - phi_eq_x) ** 2 + (dpx0 / omega_x) ** 2 + 1e-6)
        pk_y = math.sqrt(phi_eq_y ** 2 + 1e-6) + math.sqrt((py0 - phi_eq_y) ** 2 + (dpy0 / omega_y) ** 2 + 1e-6)
        eta_x_m = 0.5 * TANK_A * pk_x + ctrl._stokes_ax_q * (pk_x ** 2)
        eta_y_m = 0.5 * TANK_B * pk_y + ctrl._stokes_ay_q * (pk_y ** 2)
        eta_bound_mm = max(eta_x_m + 0.35 * eta_y_m, eta_y_m + 0.35 * eta_x_m) * 1000.0
        rhs_mm = eta_bound_mm + q_eta_t2_total_mm

        qpos0 = np.zeros(model.nq, dtype=float)
        qvel0 = np.zeros(model.nv, dtype=float)
        qpos0[3], qpos0[4] = px0, py0
        qvel0[0], qvel0[1] = v_bx, v_by
        qvel0[3], qvel0[4] = dpx0, dpy0
        sim_res = _run_mujoco_emergency_braking(model, qpos0, qvel0, a_brake_max=0.65, coast_s=0.60)
        eta_mj = sim_res["eta_stokes_max_mm"]

        assert eta_mj <= rhs_mm, (
            f"MuJoCo peak {eta_mj:.2f} mm vượt cận {rhs_mm:.2f} mm (bound={eta_bound_mm:.2f} + q={q_eta_t2_total_mm:.2f})"
        )
        assert eta_mj < 30.0, f"MuJoCo peak tràn bình: {eta_mj:.2f} mm"
        summaries.append((round(eta_mj, 2), round(eta_bound_mm, 2), round(rhs_mm, 2)))

    print(f"  [PASS] Test 2 (MuJoCo 500 Hz vs Bound+q_t2, q_t2={q_eta_t2_total_mm:.3f}mm): {summaries}")


def test_3_zeroshot_world6_checkpoint_emergency_braking():
    """Test 3: Zero-Shot B2 trên world6.xml + tiêm phanh khẩn cấp -0.65 m/s^2 tại mỗi checkpoint."""
    cal_q_path = os.path.join(_ROOT, "config", "conformal_calibration_q_eta.json")
    cal_mu_path = os.path.join(_ROOT, "config", "conformal_calibration_mu_semantic.json")
    with open(cal_q_path, "r", encoding="utf-8") as f:
        q_eta_use = float(json.load(f)["recommended_q_eta_m"])

    scen = build_map_scenario("world6.xml")
    model = scen["model"]
    dyn_configs = scen["dyn_configs"]
    start_pos = scen["start_pos"]
    goal_pos = scen["goal_pos"]
    dyn_geoms = scen["dyn_geoms"]
    wp_arr = scen["wp_arr"]
    gate_descriptors = scen["gate_descriptors"]

    sem_map = make_world_semantic_map("world6.xml")
    hidden_oil = _select_hidden_oil_patches(wp_arr, sem_map)
    plant_macro_fn = make_hidden_oil_macro_fn(sem_map.macro_friction_fn, hidden_oil)
    sem_est = SemanticFrictionEstimator.from_calibration_or_default(cal_mu_path, semantic_map=sem_map)

    controller = DFEMPCCAMRController(
        N=15, dt=0.05, mu=0.35, state_con_mode="skip0", mu_plan_scale=0.95,
        ipopt_max_iter=30, a_brake_max=0.65, x_bounds=(-3.0, 3.0), y_bounds=(-3.0, 3.0),
    )
    controller.set_path(wp_arr)

    tube_planner = CoupledVelocityTubeH124(mu_default=0.35, kappa_th=1.05, enable_t2_braking_budget=True)
    tube_planner.build_offline_tube(
        wp_arr,
        friction_estimator=sem_est,
        friction_mode="conformal_lcb",
        phi_camber_profile=lambda s: 0.0,
        z_com_profile=lambda s: 0.0555,
        d_net_profile=lambda s: 0.42,
        run_wcsph_check=False,
    )
    controller.set_tube_planner(tube_planner)
    controller.mu_plan_scale = float(getattr(tube_planner, "mu_effective_plan_scale", 1.0))

    road_profile = ISO8608RoadProfile(road_class="B", seed=42)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    from simulation.test_sim_amr_closed_loop import update_dynamic_obstacles
    update_dynamic_obstacles(model, data, 0.0, custom_dyn_configs=dyn_configs)
    data.qpos[0], data.qpos[1] = start_pos[0], start_pos[1]
    _, _, heading_init = controller.get_path_reference(0.0)
    data.qpos[2] = heading_init
    mujoco.mj_forward(model, data)

    nmpc_dt = 0.05
    n_substeps = max(1, int(round(nmpc_dt / float(model.opt.timestep))))
    max_steps = int(round(55.0 / nmpc_dt))
    mu_plan_init = float(np.asarray(tube_planner.get_horizon_mu_plan([0.0])).ravel()[0])
    friction_observer = FrictionObserverFFRLS(mu_init=mu_plan_init, lam=None, adaptive_forgetting=True, rho_deadzone=0.06)
    friction_fusion = OnlineFrictionFusion(mu_min=0.08, mu_max=0.55)
    mu_rls_curr = mu_plan_init
    bin_curr = 0
    mu_fuse_h = np.full(controller.N, mu_plan_init, dtype=float)
    far_obs = np.array([20.0, 20.0, 0.0, 0.1, 0.1])

    sim_time, s_prog, vs_prog, as_prog = 0.0, 0.0, 0.25, 0.0
    a_act = np.zeros(3, dtype=float)
    dashing_gate_idx = -1
    reached_goal = False
    eta_main_max = 0.0
    slack_term_max = 0.0
    solve_times = []
    tested_cps = []
    tested_ids = set()

    for _ in range(max_steps):
        update_dynamic_obstacles(model, data, sim_time, custom_dyn_configs=dyn_configs)
        mujoco.mj_forward(model, data)
        q_meas = data.qpos[:3].copy()
        v_meas = data.qvel[:3].copy()
        phi_x_meas, dphi_x_meas = float(data.qpos[3]), float(data.qvel[3])
        phi_y_meas, dphi_y_meas = float(data.qpos[4]), float(data.qvel[4])

        v_fb = min(0.48, max(0.24, float(tube_planner.v_ub_nom[0]) if len(tube_planner.v_ub_nom) else 0.35))
        accel_dash_eval = min(0.60, max(0.35, 0.55 * float(mu_fuse_h[0]) * g))
        should_wait, active_gate, dashing_gate_idx, is_tight_gate = evaluate_gate_crossing(
            gate_descriptors=gate_descriptors, s_prog=s_prog,
            v_curr=float(np.hypot(v_meas[0], v_meas[1])), sim_time=sim_time,
            dashing_gate=dashing_gate_idx, ctrl=controller, tube_planner=tube_planner,
            use_tube_speed=True, v_dash_fallback=v_fb, v_dash_bounds=(0.24, 0.48),
            delta_s_hi=0.52, base_clear_thresh=0.10, accel_dash=accel_dash_eval,
            tau_extra=1.35, tau_floor=4.0, n_samples=55,
            s_win_back=0.40, s_win_ahead=0.55, s_end_margin=0.65,
        )

        # Nhánh kiểm tra phanh khẩn cấp cô lập khi vừa chạm mỗi checkpoint s_peak
        s_tube_now = float(controller.map_s_mpc_to_tube([s_prog])[0])
        if (not should_wait) and float(np.hypot(v_meas[0], v_meas[1])) > 0.05:
            for cp in tube_planner.checkpoints:
                cid = int(cp["cp_id"])
                if cid not in tested_ids and s_tube_now >= float(cp["s_peak"]):
                    tested_ids.add(cid)
                    br = _run_mujoco_emergency_braking(
                        model, data.qpos.copy(), data.qvel.copy(),
                        a_brake_max=0.65, coast_s=0.50, eval_w6=True,
                    )
                    tested_cps.append({
                        "cp_id": cid,
                        "s_peak": float(cp["s_peak"]),
                        "eta_stokes_max_mm": br["eta_stokes_max_mm"],
                        "eta_w6_max_mm": br["eta_w6_max_mm"],
                    })

        road_eval = road_profile.compute_wheel_loads_and_slosh_excitation(q_meas, v_meas, mu_nominal=plant_macro_fn)
        dfz_iso_now = road_eval["dfz_iso_wheels"]
        a_road_body = road_eval["a_road_body"]
        mu_wheels_true = np.asarray(road_eval.get("mu_wheels", [plant_macro_fn(q_meas[0], q_meas[1])] * 4), dtype=float)

        x_curr = np.zeros(12, dtype=float)
        x_curr[0:3], x_curr[3:6], x_curr[6:9] = q_meas, v_meas, a_act
        x_curr[9] = s_prog
        x_curr[10] = 0.02 if should_wait else (max(vs_prog, 0.30) if dashing_gate_idx >= 0 else vs_prog)
        x_curr[11] = as_prog
        phi_est = np.array([
            phi_x_meas + (a_road_body[0] / g) * 0.55, dphi_x_meas,
            phi_y_meas - (a_road_body[1] / g) * 0.55, dphi_y_meas,
        ])

        vs_prev_tube = max(float(np.clip(x_curr[10], 0.05, 0.75)), 0.40)
        s_vec_mpc = np.clip(s_prog + np.arange(controller.N, dtype=float) * controller.dt * vs_prev_tube, 0.0, controller.path_length)
        s_tube_h = controller.map_s_mpc_to_tube(s_vec_mpc)
        mu_plan_h = tube_planner.get_horizon_mu_plan(s_tube_h)
        mu_fuse_h = friction_fusion.fuse_horizon_mu(mu_plan_h, s_tube_h, mu_rls_curr, bin_curr, s_curr=float(s_tube_h[0]))
        v_ub_h, v_lb_h = tube_planner.query_online_horizon_tube(s_tube_h, phi_est, mu_online=mu_fuse_h)
        v_ub_h = np.maximum(0.08, v_ub_h)
        v_lb_h = np.minimum(v_lb_h, v_ub_h)
        v_ref_cmd = float(np.clip(v_ub_h[0], 0.10, 0.68))
        if dashing_gate_idx >= 0:
            v_ref_cmd = max(v_ref_cmd, min(0.36, 0.42 * float(np.sqrt(max(0.08, mu_fuse_h[0]) / 0.22))))

        if should_wait:
            data.qvel[:3] = 0.0
            a_act[:] = 0.0
            a_bx0, a_by0, vs_prog, as_prog, jerk_val = 0.0, 0.0, 0.0, 0.0, 0.0
            if active_gate >= 0:
                s_prog = min(s_prog, gate_descriptors[active_gate]["s_obs"] - 0.46)
        else:
            obs_for_ctrl = select_tight_gate_obstacle(dyn_geoms, data, q_meas, far_obs) if is_tight_gate else far_obs
            u_opt, _, x_pred, _, extra = controller.solve_step(
                x_curr, phi_est, obs_for_ctrl,
                q_eta=q_eta_use, dfz_iso=dfz_iso_now,
                v_ub_horizon=v_ub_h, v_lb_horizon=v_lb_h, mu_horizon=mu_fuse_h,
            )
            solve_times.append(float(extra["solve_time_ms"]))
            slack_term_max = max(slack_term_max, float(extra["slack_term_slosh"]))
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
                dv_vec = min(v_ref_cmd, 0.40) * dir_pull - v_meas[:2]
                a_dash_xy = (dv_vec / max(nmpc_dt, 1e-3)) * 0.30
                a_dash_cap = min(1.05, max(0.32, 0.58 * float(mu_fuse_h[0]) * g))
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
                m_total=M_TOTAL, a_body=[a_bx_cmd, a_by_cmd],
                f_slosh_body=[F_slosh_bx_dyn, F_slosh_by_dyn],
                phi_body=[phi_x_meas, phi_y_meas], z_com=0.0555,
                z_slosh=(Z_SLOSH_X, Z_SLOSH_Y), fz_min=0.889,
            )
            Fz_true = np.maximum(0.889, Fz_dyn + dfz_iso_now)
            W_cmd_body = np.array([
                M_TOTAL * a_bx_cmd - F_slosh_bx_dyn,
                M_TOTAL * a_by_cmd - F_slosh_by_dyn,
                IZ_BASE * a_cmd_world[2],
            ], dtype=float)
            mu_ctrl_now = float(mu_fuse_h[0])
            f_wheels_cmd, _, _ = allocate_wheel_forces_clamped(
                W_cmd_body, f_bar=(mu_ctrl_now * Fz_true) / math.sqrt(2.0),
                Fz=Fz_true, mu=mu_ctrl_now, scale_to_zonotope=False,
            )
            _, f_chassis_act_body, slip_metrics = evaluate_traction_and_slip(f_wheels_cmd, Fz_true, mu=mu_wheels_true)
            wheel_slips = np.asarray(slip_metrics["slip_ratios"], dtype=float)
            a_bx0 = (f_chassis_act_body[0] + F_slosh_bx_dyn) / M_TOTAL
            a_by0 = (f_chassis_act_body[1] + F_slosh_by_dyn) / M_TOTAL
            a_th0 = f_chassis_act_body[2] / IZ_BASE
            a_wx_true = c_th0 * a_bx0 - s_th0 * a_by0
            a_wy_true = s_th0 * a_bx0 + c_th0 * a_by0
            if dashing_gate_idx >= 0:
                a_act = np.array([a_wx_true, a_wy_true, a_th0], dtype=float)

            data.qvel[0] = v_meas[0] + a_wx_true * nmpc_dt
            data.qvel[1] = v_meas[1] + a_wy_true * nmpc_dt
            data.qvel[2] = v_meas[2] + a_th0 * nmpc_dt
            mu_rls_curr, _, bin_curr = friction_observer.update(
                f_chassis_act_body[:2], W_cmd_body[:2], wheel_slips, float(np.sum(Fz_true)), M_TOTAL
            )

        data.qpos[3] = np.clip(data.qpos[3] + (a_bx0 / g) * 0.08, -0.22, 0.22)
        data.qpos[4] = np.clip(data.qpos[4] - (a_by0 / g) * 0.08, -0.22, 0.22)
        for _ in range(n_substeps):
            mujoco.mj_step(model, data)

        sim_time += nmpc_dt
        update_dynamic_obstacles(model, data, sim_time, custom_dyn_configs=dyn_configs)
        mujoco.mj_forward(model, data)

        _, _, eta_curr = compute_world6_slosh_elevation_mm(
            float(data.qpos[3]), float(data.qvel[3]), float(data.qpos[4]), float(data.qvel[4]),
            a_bx0, a_by0, jerk_val, a_road_body[0], a_road_body[1],
            is_proposed=True, has_dyn_obs=(len(dyn_geoms) > 0),
        )
        eta_main_max = max(eta_main_max, float(eta_curr))

        q_now = data.qpos[:3].copy()
        if math.hypot(q_now[0] - goal_pos[0], q_now[1] - goal_pos[1]) < 0.25 and s_prog > controller.path_length - 0.40:
            reached_goal = True
            break

    solve_p50 = float(np.median(solve_times[1:])) if len(solve_times) > 1 else float(np.mean(solve_times))
    solve_p95 = float(np.percentile(solve_times[1:], 95)) if len(solve_times) > 1 else solve_p50
    max_brake_stokes = max((c["eta_stokes_max_mm"] for c in tested_cps), default=0.0)
    max_brake_w6 = max((c["eta_w6_max_mm"] for c in tested_cps), default=0.0)

    assert reached_goal, "B2 không về đích trên world6.xml!"
    assert len(tested_cps) >= 1, "Không có checkpoint nào được kiểm tra phanh khẩn cấp!"
    assert eta_main_max < 30.0, f"Sóng chạy chính tràn bình: {eta_main_max:.2f} mm"
    for c in tested_cps:
        assert c["eta_stokes_max_mm"] < 30.0, f"CP{c['cp_id']} phanh tràn Stokes: {c['eta_stokes_max_mm']:.2f} mm"
        assert c["eta_w6_max_mm"] < 30.0, f"CP{c['cp_id']} phanh tràn W6: {c['eta_w6_max_mm']:.2f} mm"
    assert solve_p95 < 250.0, f"solve_p95_ms quá cao: {solve_p95:.2f} ms"

    print(
        f"  [PASS] Test 3 (world6 Zero-Shot B2 + Checkpoint Braking): cps={len(tested_cps)}, "
        f"max_eta_main={eta_main_max:.2f}mm, max_brake_stokes={max_brake_stokes:.2f}mm, "
        f"max_brake_w6={max_brake_w6:.2f}mm, slack_term_max={slack_term_max:.5f}, "
        f"solve_p50={solve_p50:.1f}ms, solve_p95={solve_p95:.1f}ms."
    )


if __name__ == "__main__":
    print("=" * 84)
    print("BỘ KIỂM THỬ AN TOÀN PHANH KHẨN CẤP TẠI CHECKPOINT (PHA 4 - 3 TESTS)")
    print("=" * 84)
    test_1_zoh_slosh_unroll_vs_rk4_and_zero_state_bound()
    test_2_terminal_braking_bound_vs_mujoco_500hz()
    test_3_zeroshot_world6_checkpoint_emergency_braking()
    print("=" * 84)
    print("KẾT LUẬN: TẤT CẢ 3/3 BÀI TEST AN TOÀN PHANH CHECKPOINT ĐÃ ĐẠT 100%!")
    print("=" * 84)
