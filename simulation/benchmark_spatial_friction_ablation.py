# -*- coding: utf-8 -*-
"""
benchmark_spatial_friction_ablation.py
=============================================================================
KIỂM CHỨNG SIM-ONLY MỞ RỘNG (ĐỢT 1 + ĐỢT 2) CHO MÔ HÌNH ƯỚC LƯỢNG MA SÁT
KHÔNG GIAN GP + HIỆU CHUẨN CONFORMAL CẤP QUỸ ĐẠO DẤU CHÂN 4 BÁNH

Bao gồm 7 nghiên cứu định lượng giải quyết toàn diện các nhận xét kỹ thuật:
1. Study 1 (Protocol A - Cùng bố trí sàn mở rộng):
   D_hist = 15 lượt hai chiều qua FF-RLS từng bánh bù trễ nhân quả;
   D_cal = 80 quỹ đạo (seeds 200..279 -> k*=77 tại alpha=0.05, k*=79 < 80 tại 0.025);
   D_test = 100 quỹ đạo chưa từng thấy (seeds 300..399) đánh giá M0..M3.
2. Study 2 (M3 + M5 Component Ablation):
   Tách từng thành phần (Center vs Footprint GP; Unidir Chassis vs Bidir Per-Wheel LagComp).
3. Study 3 (M4 & Prior Attribution):
   Khảo sát mật độ phủ lịch sử (Dense 15 / Sparse 5 / Lệch tuyến +0.65 m) và đóng góp của mu_prior.
4. Study 4 (Mục 6 Kernel Hyperparameter Sensitivity & LML):
   Quét (ell_macro, sigma_f) và tính Log Marginal Likelihood (LML).
5. Study 5 (Protocol B - Đa bố trí sàn ngẫu nhiên, M7 & M8):
   80 site hiệu chuẩn + 100 site kiểm thử hoàn toàn mới (make_random_site_ground_truth).
6. Study 6 (M1 + M2 + M8 Vòng kín MuJoCo 10 seed liên tiếp 300..309 & Ablation 2x2):
   Bộ giải thời gian thực N=10, ipopt_max_iter=15, warm-start primal-dual, ánh xạ s_mpc->s_tube;
   Báo cáo mean/p50/p95/max solve time, % > 50 ms, cận trên Clopper-Pearson tỷ lệ trượt.
7. Study 7 (M1 Kiểm tra trễ chấp hành 1 chu kỳ = 50 ms):
   Chạy lại trên 3 seed có min(mu_4w) thấp nhất với trễ 1 chu kỳ điều khiển.
=============================================================================
"""

import os
import sys
import json
import time
import numpy as np
import mujoco
from scipy.stats import beta

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

_curr = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_curr) if os.path.basename(_curr) in ["simulation", "controller", "solver", "planning", "tests"] else _curr
for _p in [
    _root,
    os.path.join(_root, "planning"),
    os.path.join(_root, "controller"),
    os.path.join(_root, "solver"),
    os.path.join(_root, "simulation"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from planning.spatial_friction_gp import (
    SpatialFrictionGP,
    make_benchmark_curve_ground_truth,
    make_random_site_ground_truth,
    collect_ffrls_history_from_rollouts,
)
from planning.velocity_tube_h124 import (
    CoupledVelocityTubeH124,
    _analytical_benchmark_curve,
    generate_multi_regime_benchmark_scenario,
)
from controller.dfe_mpcc_amr_controller import DFEMPCCAMRController
from controller.mecanum_traction_dynamics import (
    M_TOTAL,
    G,
    compute_dynamic_normal_loads,
    allocate_wheel_forces_clamped,
    evaluate_traction_and_slip,
)
from solver.robot_model_amr_casadi import (
    TANK_A,
    TANK_B,
    H_LIQUID,
    m_1x,
    lx,
    omega_x,
    m_1y,
    ly,
    omega_y,
)


def _perturbed_curve_waypoints(
    y_offset: float = 0.0,
    y_amp_scale: float = 1.0,
    n_pts: int = 200,
) -> np.ndarray:
    x_arr = np.linspace(0.0, 6.75, n_pts)
    y_base, _, _, _ = _analytical_benchmark_curve(x_arr)
    envelope = np.sin(np.pi * (x_arr / 6.75))
    y_arr = y_amp_scale * y_base + y_offset * envelope
    return np.column_stack([x_arr, y_arr])


def _collect_hist_data(
    n_runs: int = 15,
    observation_mode: str = "per_wheel_rls_lagcomp",
    bidirectional: bool = True,
    lam_rls: float = 0.80,
    seed_start: int = 100,
) -> dict:
    """Thu thập dữ liệu lịch sử D_hist từ n_runs lượt chạy (kiểm soát được cùng seed gốc 100)."""
    rng_hist = np.random.default_rng(100)
    pts_all, mu_all, var_all = [], [], []
    n_total_steps, n_accepted, n_skipped = 0, 0, 0

    for idx in range(int(n_runs)):
        s_id = int(seed_start) + idx
        gt_h = make_benchmark_curve_ground_truth(seed=s_id, road_class="B", variation_scale=0.15)
        wp_h = _perturbed_curve_waypoints(
            y_offset=float(rng_hist.uniform(-0.32, 0.32)),
            y_amp_scale=float(rng_hist.uniform(0.88, 1.12)),
            n_pts=185,
        )
        wp_pass = wp_h[::-1].copy() if (bidirectional and (idx % 2 == 1)) else wp_h
        res_h = collect_ffrls_history_from_rollouts(
            gt_h,
            [wp_pass],
            lam_rls=lam_rls,
            seed=s_id + 1000,
            observation_mode=observation_mode,
            bidirectional=False,  # Đã đảo chiều xen kẽ ngay tại từng lượt
        )
        n_total_steps += res_h["n_total_steps"]
        n_accepted += res_h["n_accepted"]
        n_skipped += res_h["n_deadzone_skipped"]
        if res_h["n_accepted"] > 0:
            pts_all.append(res_h["points_xy"])
            mu_all.append(res_h["mu_obs"])
            var_all.append(res_h["obs_var"])

    return {
        "n_runs": int(n_runs),
        "n_total_steps": int(n_total_steps),
        "n_accepted": int(n_accepted),
        "n_deadzone_skipped": int(n_skipped),
        "observation_mode": observation_mode,
        "bidirectional": bool(bidirectional),
        "lam_rls": float(lam_rls),
        "points_xy": np.vstack(pts_all) if pts_all else np.zeros((0, 2)),
        "mu_obs": np.concatenate(mu_all) if mu_all else np.zeros(0),
        "obs_var": np.concatenate(var_all) if var_all else np.zeros(0),
    }


def _build_traj_split(seed_start: int, count: int, y_shift: float = 0.0, is_test: bool = False) -> list:
    rng = np.random.default_rng(int(seed_start))
    trajs = []
    for idx in range(int(count)):
        s_id = int(seed_start) + idx
        gt = make_benchmark_curve_ground_truth(seed=s_id, road_class="B", variation_scale=0.15)
        if is_test and s_id == 300 and abs(y_shift) < 1e-9:
            wp = _perturbed_curve_waypoints(y_offset=0.0, y_amp_scale=1.0, n_pts=250)
        else:
            y_off = float(rng.uniform(-0.22, 0.22)) + float(y_shift)
            y_amp = float(rng.uniform(0.92, 1.08))
            wp = _perturbed_curve_waypoints(y_offset=y_off, y_amp_scale=y_amp, n_pts=200 if is_test else 185)
        mu_c, mu_4w, split_del = gt.eval_path_4w_min(wp[:, 0], wp[:, 1])
        trajs.append({
            "seed": s_id,
            "waypoints": wp,
            "x_grid": wp[:, 0],
            "y_grid": wp[:, 1],
            "mu_center_true": mu_c,
            "mu_min_4w_true": mu_4w,
            "split_delta": split_del,
            "ground_truth": gt,
        })
    return trajs


def _build_disjoint_datasets(
    n_hist: int = 15,
    n_cal: int = 80,
    n_test: int = 100,
    observation_mode: str = "per_wheel_rls_lagcomp",
    bidirectional: bool = True,
    y_test_shift: float = 0.0,
):
    """Tạo bộ 3 tập dữ liệu tách rời mặc định cho Protocol A."""
    hist_summary = _collect_hist_data(
        n_runs=n_runs if (n_runs := n_hist) else 15,
        observation_mode=observation_mode,
        bidirectional=bidirectional,
        seed_start=100,
    )
    cal_trajs = _build_traj_split(seed_start=200, count=n_cal, y_shift=0.0, is_test=False)
    test_trajs = _build_traj_split(seed_start=300, count=n_test, y_shift=y_test_shift, is_test=True)
    return hist_summary, cal_trajs, test_trajs


def run_closed_loop_mujoco_trial(
    mode_key: str,
    waypoints: np.ndarray,
    tube_planner: CoupledVelocityTubeH124,
    ground_truth,
    mpc_mu_planner: CoupledVelocityTubeH124 = None,
    d_net_fn=None,
    max_time_s: float = 28.0,
    controller_N: int = 8,
    ipopt_max_iter: int = 8,
    actuation_delay_cycles: int = 0,
    enable_deadline_miss: bool = False,
    shared_ctrl: DFEMPCCAMRController = None,
    shared_mj_model=None,
) -> dict:
    """
    Chạy mô phỏng đóng vòng thực tế trong MuJoCo (xml/slosh3.xml) với DFEMPCCAMRController
    cấu hình thời gian thực (N=8, ipopt_max_iter=8, warm-start primal-dual, ánh xạ s_mpc->s_tube),
    tách bạch độc lập nguồn mu của Ống vận tốc (tube_planner) và nguồn mu của MPC (mpc_mu_planner),
    hỗ trợ bù dự đoán cho trễ chấp hành 1 chu kỳ (50 ms) và mô phỏng lỡ hạn giải (> 50 ms -> 100 ms).
    """
    mu_src = tube_planner if mpc_mu_planner is None else mpc_mu_planner

    if shared_mj_model is None:
        xml_path = os.path.join(_root, "xml", "slosh3.xml")
        model = mujoco.MjModel.from_xml_path(xml_path)
    else:
        model = shared_mj_model
    data = mujoco.MjData(model)

    dt_sim = float(model.opt.timestep)  # 0.002 s trên slosh3.xml
    nmpc_dt = 0.05
    substeps = int(round(nmpc_dt / dt_sim))

    if shared_ctrl is None:
        ctrl = DFEMPCCAMRController(
            N=int(controller_N),
            dt=nmpc_dt,
            mu=0.35,
            state_con_mode="skip0",
            mu_plan_scale=mu_src.mu_effective_plan_scale,
            ipopt_max_iter=int(ipopt_max_iter),
            h_lim_m=0.030,
        )
    else:
        ctrl = shared_ctrl
        ctrl.mu_plan_scale = float(mu_src.mu_effective_plan_scale)
    ctrl.set_path(waypoints)
    ctrl.set_tube_planner(tube_planner)
    ctrl.reset_warm_start()

    mujoco.mj_resetData(model, data)
    start_pos = waypoints[0]
    goal_pos = waypoints[-1]
    _, _, heading_init = ctrl.get_path_reference(0.0)
    data.qpos[0] = start_pos[0]
    data.qpos[1] = start_pos[1]
    data.qpos[2] = heading_init
    mujoco.mj_forward(model, data)

    # Warm-up 1 lần tại t=0 trước khi khởi hành để kích hoạt bộ nhớ đệm CasADi/IPOPT (không tính vào thống kê 20 Hz)
    x_warm = np.zeros(12)
    x_warm[0:3] = [start_pos[0], start_pos[1], heading_init]
    x_warm[10] = 0.05
    s_w_tube = ctrl.map_s_mpc_to_tube(np.clip(np.arange(ctrl.N) * nmpc_dt * 0.40, 0.0, ctrl.path_length))
    v_w_h = np.clip(np.interp(s_w_tube, tube_planner.s_grid, tube_planner.v_ub_nom), 0.15, 0.70).reshape(1, ctrl.N)
    mu_w_h = mu_src.get_horizon_mu_plan(np.clip(s_w_tube, 0.0, mu_src.s_grid[-1])).reshape(1, ctrl.N)
    ctrl.solve_step(
        x_current=x_warm,
        phi_est=np.zeros(4),
        mu_floor=None,
        q_eta=ctrl.default_q_eta_m,
        dfz_iso=np.zeros(4),
        v_ub_horizon=v_w_h,
        v_lb_horizon=np.zeros((1, ctrl.N)),
        mu_horizon=mu_w_h,
    )

    M_full = np.zeros((model.nv, model.nv))
    kp_diag = np.array([95.0, 95.0, 14.0])
    kd_diag = np.array([32.0, 32.0, 3.0])

    s_prog = 0.0
    vs_prog = 0.05
    as_prog = 0.0
    a_act = np.zeros(3)
    sim_time = 0.0
    max_steps = int(round(max_time_s / nmpc_dt))
    goal_reached = False
    u_committed = np.zeros(4, dtype=float)
    u_fallback_next = np.zeros(4, dtype=float)

    coth_x = 1.0 / float(np.tanh(np.pi * H_LIQUID / TANK_A))
    coth_y = 1.0 / float(np.tanh(np.pi * H_LIQUID / TANK_B))

    slip_steps = 0
    total_cycles = 0
    deadline_miss_steps = 0
    max_slip_val = 0.0
    max_eta_mm = 0.0
    min_fz_N = 100.0
    max_contour_err_m = 0.0
    solve_times_ms = []
    iter_counts = []
    suboptimal_steps = 0

    def _execute_mujoco_cycle(q0_c, v0_c, a0_c, u0_c, s_ref_c):
        nonlocal sim_time, total_cycles, slip_steps, max_slip_val, max_eta_mm, min_fz_N
        cyc_slip = False
        s_prog_tube = float(np.clip(ctrl.map_s_mpc_to_tube(s_ref_c), 0.0, mu_src.s_grid[-1]))
        mu_belief = float(np.interp(s_prog_tube, mu_src.s_grid, mu_src.mu_grid)) * mu_src.mu_effective_plan_scale

        dfz_iso_cached = None
        mu_wheels_true_cached = None
        for sub in range(substeps):
            tau_sub = (sub + 1) * dt_sim
            q_target = q0_c + tau_sub * v0_c + 0.5 * (tau_sub ** 2) * a0_c + (tau_sub ** 3 / 6.0) * u0_c
            v_target = v0_c + tau_sub * a0_c + 0.5 * (tau_sub ** 2) * u0_c
            a_target = a0_c + tau_sub * u0_c

            mujoco.mj_fullM(model, data, M_full)
            F_ff = M_full[:3, :3] @ a_target
            q_now = data.qpos[:3].copy()
            v_now = data.qvel[:3].copy()
            F_bias = data.qfrc_bias[:3].copy()

            err_q = q_target - q_now
            err_q[2] = np.arctan2(np.sin(err_q[2]), np.cos(err_q[2]))
            err_v = v_target - v_now

            F_cmd_world = F_ff + kp_diag * err_q + kd_diag * err_v + F_bias
            th_now = float(q_now[2])
            c_th, s_th = np.cos(th_now), np.sin(th_now)
            F_cmd_body = np.array([
                c_th * F_cmd_world[0] + s_th * F_cmd_world[1],
                -s_th * F_cmd_world[0] + c_th * F_cmd_world[1],
                F_cmd_world[2],
            ])

            a_bx = c_th * a_target[0] + s_th * a_target[1]
            a_by = -s_th * a_target[0] + c_th * a_target[1]
            ph_x_now = float(data.qpos[3])
            ph_y_now = float(data.qpos[4])
            F_react_bx = -m_1x * (a_bx + lx * (-(omega_x ** 2) * ph_x_now))
            F_react_by = -m_1y * (a_by + ly * (-(omega_y ** 2) * ph_y_now))

            Fz_nom, _ = compute_dynamic_normal_loads(
                M_TOTAL,
                np.array([a_bx, a_by]),
                np.array([F_react_bx, F_react_by]),
                phi_body=np.array([ph_x_now, ph_y_now]),
            )
            # Lấy mẫu biên dạng nhám không gian ISO 8608 & 4 bánh ở 100 Hz (< 6 mm dịch chuyển/mẫu)
            if (sub % 5) == 0 or dfz_iso_cached is None:
                road_sub = ground_truth.road_profile.compute_wheel_loads_and_slosh_excitation(
                    q_now, v_now, mu_nominal=ground_truth.macro_fn, friction_variation_scale=ground_truth.variation_scale
                )
                dfz_iso_cached = road_sub["dfz_iso_wheels"]
                _, mu_wheels_true_cached, _ = ground_truth.eval_four_wheels(q_now[0], q_now[1], th_now)

            Fz_true = np.maximum(1.5, Fz_nom + dfz_iso_cached)
            min_fz_N = min(min_fz_N, float(np.min(Fz_true)))

            f_bar_belief = (mu_belief / np.sqrt(2.0)) * Fz_true
            f_wheels_cmd, _, _ = allocate_wheel_forces_clamped(F_cmd_body, f_bar_belief, Fz=Fz_true, mu=mu_belief)

            _, f_chassis_act_body, slip_metrics = evaluate_traction_and_slip(
                f_wheels_cmd, Fz_true, mu=mu_wheels_true_cached
            )

            if slip_metrics["any_slip"]:
                cyc_slip = True
            max_slip_val = max(max_slip_val, float(slip_metrics["max_slip"]))

            data.qfrc_applied[:3] = -data.qfrc_passive[:3]
            F_act_world = np.array([
                c_th * f_chassis_act_body[0] - s_th * f_chassis_act_body[1],
                s_th * f_chassis_act_body[0] + c_th * f_chassis_act_body[1],
                f_chassis_act_body[2],
            ])
            data.ctrl[:3] = np.clip(F_act_world, [-40.0, -40.0, -15.0], [40.0, 40.0, 15.0])
            mujoco.mj_step(model, data)
            sim_time += dt_sim

            ux = abs(np.tan(float(data.qpos[3])))
            uy = abs(np.tan(float(data.qpos[4])))
            ex_mm = (0.5 * TANK_A * ux + (np.pi * TANK_A / 8.0) * coth_x * (ux ** 2)) * 1000.0
            ey_mm = (0.5 * TANK_B * uy + (np.pi * TANK_B / 8.0) * coth_y * (uy ** 2)) * 1000.0
            eta_now = max(ey_mm + 0.35 * ex_mm, ex_mm + 0.35 * ey_mm)
            max_eta_mm = max(max_eta_mm, eta_now)

        total_cycles += 1
        if cyc_slip:
            slip_steps += 1

    while total_cycles < max_steps:
        q_meas = data.qpos[:3].copy()
        v_meas = data.qvel[:3].copy()
        phi_x = float(data.qpos[3])
        dphi_x = float(data.qvel[3])
        phi_y = float(data.qpos[4])
        dphi_y = float(data.qvel[4])

        dist_goal = float(np.hypot(q_meas[0] - goal_pos[0], q_meas[1] - goal_pos[1]))
        if dist_goal < 0.14 and s_prog > ctrl.path_length - 0.25:
            goal_reached = True
            break

        road_eval = ground_truth.road_profile.compute_wheel_loads_and_slosh_excitation(
            q_meas, v_meas, mu_nominal=ground_truth.macro_fn, friction_variation_scale=ground_truth.variation_scale
        )
        dfz_iso_now = road_eval["dfz_iso_wheels"]

        phi_est = np.array([phi_x, dphi_x, phi_y, dphi_y], dtype=float)
        x_cur = np.zeros(12)
        x_cur[0:3] = q_meas
        x_cur[3:6] = v_meas
        x_cur[6:9] = a_act
        x_cur[9] = s_prog
        x_cur[10] = vs_prog
        x_cur[11] = as_prog

        if int(actuation_delay_cycles) <= 0:
            vs_preview = max(vs_prog, 0.35)
            s_preview_mpc = np.clip(s_prog + np.arange(ctrl.N) * nmpc_dt * vs_preview, 0.0, ctrl.path_length)
            s_preview_tube = ctrl.map_s_mpc_to_tube(s_preview_mpc)
            v_ub_h = np.clip(np.interp(s_preview_tube, tube_planner.s_grid, tube_planner.v_ub_nom), 0.15, 0.70).reshape(1, ctrl.N)
            v_lb_h = np.zeros((1, ctrl.N))
            mu_h = mu_src.get_horizon_mu_plan(np.clip(s_preview_tube, 0.0, mu_src.s_grid[-1])).reshape(1, ctrl.N)

            u_opt, _, x_pred, _, info = ctrl.solve_step(
                x_current=x_cur,
                phi_est=phi_est,
                obs_info=None,
                mu_floor=None,
                q_eta=ctrl.default_q_eta_m,
                dfz_iso=dfz_iso_now,
                v_ub_horizon=v_ub_h,
                v_lb_horizon=v_lb_h,
                mu_horizon=mu_h,
            )
            solve_times_ms.append(float(info["solve_time_ms"]))
            if info.get("iter_count", -1) >= 0:
                iter_counts.append(int(info["iter_count"]))
            if info.get("status") != "optimal":
                suboptimal_steps += 1

            a0_cyc = a_act.copy()
            s_prog = float(x_pred[9, 1])
            vs_prog = float(x_pred[10, 1])
            as_prog = float(x_pred[11, 1])
            a_act = np.array(x_pred[6:9, 1], dtype=float)
            p_ref_now, _, _ = ctrl.get_path_reference(s_prog)
            max_contour_err_m = max(max_contour_err_m, float(np.hypot(q_meas[0] - p_ref_now[0], q_meas[1] - p_ref_now[1])))
            _execute_mujoco_cycle(q_meas, v_meas, a0_cyc, u_opt[:3], s_prog)
        else:
            # Bù dự đoán cho trễ 1 chu kỳ (50 ms): tối ưu xuất phát từ trạng thái sau khi lệnh u_committed tác động
            x_solve_start = ctrl.Ad @ x_cur + ctrl.Bd @ u_committed
            x_solve_start[9] = float(np.clip(x_solve_start[9], 0.0, ctrl.path_length))
            x_solve_start[10] = float(np.clip(x_solve_start[10], 0.05, 0.75))

            vs_preview = max(float(x_solve_start[10]), 0.35)
            s_preview_mpc = np.clip(float(x_solve_start[9]) + np.arange(ctrl.N) * nmpc_dt * vs_preview, 0.0, ctrl.path_length)
            s_preview_tube = ctrl.map_s_mpc_to_tube(s_preview_mpc)
            v_ub_h = np.clip(np.interp(s_preview_tube, tube_planner.s_grid, tube_planner.v_ub_nom), 0.15, 0.70).reshape(1, ctrl.N)
            v_lb_h = np.zeros((1, ctrl.N))
            mu_h = mu_src.get_horizon_mu_plan(np.clip(s_preview_tube, 0.0, mu_src.s_grid[-1])).reshape(1, ctrl.N)

            u_opt, js_opt, _, u_pred, info = ctrl.solve_step(
                x_current=x_solve_start,
                phi_est=phi_est,
                obs_info=None,
                mu_floor=None,
                q_eta=ctrl.default_q_eta_m,
                dfz_iso=dfz_iso_now,
                v_ub_horizon=v_ub_h,
                v_lb_horizon=v_lb_h,
                mu_horizon=mu_h,
            )
            solve_times_ms.append(float(info["solve_time_ms"]))
            if info.get("iter_count", -1) >= 0:
                iter_counts.append(int(info["iter_count"]))
            if info.get("status") != "optimal":
                suboptimal_steps += 1

            u_next = np.array([u_opt[0], u_opt[1], u_opt[2], js_opt], dtype=float)
            u_next_step1 = np.array(u_pred[:, 1], dtype=float) if u_pred.shape[1] > 1 else u_next.copy()

            # Thực thi lệnh u_committed đang giữ hàng trong [t_k, t_k + 50 ms]
            a0_cyc = a_act.copy()
            s_prog = float(x_solve_start[9])
            vs_prog = float(x_solve_start[10])
            as_prog = float(x_solve_start[11])
            a_act = np.array(x_solve_start[6:9], dtype=float)
            p_ref_now, _, _ = ctrl.get_path_reference(s_prog)
            max_contour_err_m = max(max_contour_err_m, float(np.hypot(q_meas[0] - p_ref_now[0], q_meas[1] - p_ref_now[1])))
            _execute_mujoco_cycle(q_meas, v_meas, a0_cyc, u_committed[:3], s_prog)

            if enable_deadline_miss and float(info["solve_time_ms"]) > 50.0 and total_cycles < max_steps:
                q_mid = data.qpos[:3].copy()
                v_mid = data.qvel[:3].copy()
                dist_mid = float(np.hypot(q_mid[0] - goal_pos[0], q_mid[1] - goal_pos[1]))
                if dist_mid < 0.14 and s_prog > ctrl.path_length - 0.25:
                    goal_reached = True
                    break
                deadline_miss_steps += 1
                x_mid = np.zeros(12)
                x_mid[0:3] = q_mid
                x_mid[3:6] = v_mid
                x_mid[6:9] = a_act
                x_mid[9] = s_prog
                x_mid[10] = vs_prog
                x_mid[11] = as_prog
                x_post_miss = ctrl.Ad @ x_mid + ctrl.Bd @ u_fallback_next
                x_post_miss[9] = float(np.clip(x_post_miss[9], 0.0, ctrl.path_length))
                x_post_miss[10] = float(np.clip(x_post_miss[10], 0.05, 0.75))
                a0_mid = a_act.copy()
                s_prog = float(x_post_miss[9])
                vs_prog = float(x_post_miss[10])
                as_prog = float(x_post_miss[11])
                a_act = np.array(x_post_miss[6:9], dtype=float)
                _execute_mujoco_cycle(q_mid, v_mid, a0_mid, u_fallback_next[:3], s_prog)

            u_committed = u_next
            u_fallback_next = u_next_step1

    st_arr = np.asarray(solve_times_ms, dtype=float) if solve_times_ms else np.zeros(1)
    slip_rate_pct = 100.0 * slip_steps / max(1, total_cycles)
    return {
        "mode": mode_key,
        "goal_reached": bool(goal_reached),
        "t_finish_s": round(float(sim_time), 2),
        "final_dist_to_goal_m": round(float(np.hypot(data.qpos[0] - goal_pos[0], data.qpos[1] - goal_pos[1])), 3),
        "slip_steps": int(slip_steps),
        "total_cycles": int(total_cycles),
        "slip_rate_pct": round(float(slip_rate_pct), 2),
        "max_slip": round(float(max_slip_val), 4),
        "max_eta_mm": round(float(max_eta_mm), 2),
        "min_fz_N": round(float(min_fz_N), 2),
        "max_contour_err_m": round(float(max_contour_err_m), 4),
        "mean_solve_ms": round(float(np.mean(st_arr)), 2),
        "p50_solve_ms": round(float(np.median(st_arr)), 2),
        "p95_solve_ms": round(float(np.percentile(st_arr, 95)), 2),
        "max_solve_ms": round(float(np.max(st_arr)), 2),
        "pct_over_50ms": round(float(100.0 * np.mean(st_arr > 50.0)), 2),
        "mean_iters": round(float(np.mean(iter_counts)) if iter_counts else 0.0, 2),
        "suboptimal_steps": int(suboptimal_steps),
        "deadline_miss_steps": int(deadline_miss_steps),
        "s_mapping_max_drift_m": round(float(ctrl.s_mapping_max_drift_m), 4),
        "actuation_delay_cycles": int(actuation_delay_cycles),
        "enable_deadline_miss": bool(enable_deadline_miss),
        "_raw_solve_times_ms": [round(float(v), 2) for v in solve_times_ms],
    }


def _aggregate_cl_metrics(runs_list: list) -> dict:
    """Tổng hợp số liệu nhiều seed vòng kín kèm cận trên Clopper-Pearson tỷ lệ lượt bị trượt."""
    n_runs = len(runs_list)
    n_slip_runs = int(sum(1 for r in runs_list if r["slip_steps"] > 0))
    # Cận trên một phía 95% Clopper-Pearson cho P(lượt bị trượt >= 1 chu kỳ)
    ub95 = 1.0 - (0.05 ** (1.0 / n_runs)) if n_slip_runs == 0 else (
        1.0 if n_slip_runs >= n_runs else float(beta.ppf(0.95, n_slip_runs + 1, n_runs - n_slip_runs))
    )
    tot_slip = int(sum(r["slip_steps"] for r in runs_list))
    tot_cyc = int(sum(r["total_cycles"] for r in runs_list))
    all_solves = np.concatenate([np.asarray(r.get("_raw_solve_times_ms", [r["mean_solve_ms"]]), dtype=float) for r in runs_list])
    return {
        "n_runs": n_runs,
        "n_runs_with_slip": n_slip_runs,
        "run_slip_prob_95_ub": round(float(ub95), 4),
        "total_slip_cycles": tot_slip,
        "total_cycles": tot_cyc,
        "aggregate_slip_rate_pct": round(100.0 * tot_slip / max(1, tot_cyc), 3),
        "worst_max_slip": round(float(max(r["max_slip"] for r in runs_list)), 4),
        "mean_t_finish_s": round(float(np.mean([r["t_finish_s"] for r in runs_list])), 3),
        "mean_eta_max_mm": round(float(np.mean([r["max_eta_mm"] for r in runs_list])), 2),
        "worst_eta_max_mm": round(float(np.max([r["max_eta_mm"] for r in runs_list])), 2),
        "min_fz_N": round(float(np.min([r["min_fz_N"] for r in runs_list])), 2),
        "all_goals_reached": bool(all(r["goal_reached"] for r in runs_list)),
        "mean_solve_ms": round(float(np.mean(all_solves)), 2),
        "p50_solve_ms": round(float(np.median(all_solves)), 2),
        "p95_solve_ms": round(float(np.percentile(all_solves, 95)), 2),
        "max_solve_ms": round(float(np.max(all_solves)), 2),
        "pct_over_50ms": round(float(100.0 * np.mean(all_solves > 50.0)), 2),
        "suboptimal_rate_pct": round(100.0 * sum(r["suboptimal_steps"] for r in runs_list) / max(1, tot_cyc), 3),
        "total_deadline_miss_steps": int(sum(r.get("deadline_miss_steps", 0) for r in runs_list)),
        "mean_iters": round(float(np.mean([r["mean_iters"] for r in runs_list])), 2),
        "max_s_mapping_drift_m": round(float(max(r["s_mapping_max_drift_m"] for r in runs_list)), 4),
    }


def _strip_raw(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "_raw_solve_times_ms"}


def _run_protocol_b_multisite(n_cal_sites: int = 80, n_test_sites: int = 100) -> dict:
    """Study 5 (Protocol B - M7/M8): Đánh giá xuyên bố trí sàn ngẫu nhiên độc lập."""
    rng_m = np.random.default_rng(777)

    def _eval_site_group(seed_base: int, count: int):
        site_recs = []
        for idx in range(count):
            s_id = int(seed_base) + idx
            gt = make_random_site_ground_truth(s_id, road_class="B", variation_scale=0.15)
            rollouts = []
            for r_i in range(8):
                wp_r = _perturbed_curve_waypoints(
                    y_offset=float(rng_m.uniform(-0.30, 0.30)),
                    y_amp_scale=float(rng_m.uniform(0.90, 1.10)),
                    n_pts=110,
                )
                if r_i % 2 == 1:
                    wp_r = wp_r[::-1].copy()
                rollouts.append(wp_r)
            hist_s = collect_ffrls_history_from_rollouts(
                gt, rollouts, seed=s_id + 10000, observation_mode="per_wheel_rls_lagcomp", bidirectional=False
            )
            gp_s = SpatialFrictionGP(query_mode="footprint")
            gp_s.fit(hist_s["points_xy"], hist_s["mu_obs"], hist_s["obs_var"], voxel_size=0.15)

            wp_t = _perturbed_curve_waypoints(
                y_offset=float(rng_m.uniform(-0.22, 0.22)),
                y_amp_scale=float(rng_m.uniform(0.92, 1.08)),
                n_pts=140,
            )
            _, mu_4w_t, _ = gt.eval_path_4w_min(wp_t[:, 0], wp_t[:, 1])
            site_recs.append({"gp": gp_s, "x_grid": wp_t[:, 0], "y_grid": wp_t[:, 1], "mu_min_4w_true": mu_4w_t})
        return site_recs

    cal_sites = _eval_site_group(1000, n_cal_sites)
    cal_scores = np.empty(n_cal_sites, dtype=float)
    for j, rec in enumerate(cal_sites):
        cal_res = rec["gp"].calibrate_trajectory_conformal(
            [{"x_grid": rec["x_grid"], "y_grid": rec["y_grid"], "mu_min_4w_true": rec["mu_min_4w_true"]}],
            alpha_mu=0.05,
        )
        cal_scores[j] = float(cal_res["score_max"])

    scores_sorted = np.sort(cal_scores)
    k_050 = int(np.clip(np.ceil((n_cal_sites + 1) * 0.95), 1, n_cal_sites))
    k_025 = int(np.clip(np.ceil((n_cal_sites + 1) * 0.975), 1, n_cal_sites))
    q_050 = float(scores_sorted[k_050 - 1])
    q_025 = float(scores_sorted[k_025 - 1])
    sc_c = cal_scores - np.mean(cal_scores)
    r1 = float(np.sum(sc_c[:-1] * sc_c[1:]) / max(np.sum(sc_c ** 2), 1e-12))
    z_r1 = abs(r1) * np.sqrt(n_cal_sites)

    test_sites = _eval_site_group(2000, n_test_sites)

    def _agg_test(mode_name: str):
        cov_flags, pt_fracs, overests, gaps, priors, clamps, pred_ms = [], [], [], [], [], [], []
        for rec in test_sites:
            gp_s = rec["gp"]
            gp_s.q_hat_mu = q_050
            mu_true = rec["mu_min_4w_true"]
            if mode_name == "M0":
                mu_tube = np.full_like(mu_true, 0.35 * 0.95)
                p_frac, c_frac, ms_v = 0.0, 0.0, 0.0
            elif mode_name == "M1":
                r = gp_s.predict_along_path(rec["x_grid"], rec["y_grid"], mode="gp_mean")
                mu_tube = r["mu_tube"]
                p_frac, c_frac, ms_v = r["prior_region_frac"], float(np.mean(mu_tube <= gp_s.mu_min + 1e-6)), r["predict_time_ms"]
            else:
                r = gp_s.predict_along_path(rec["x_grid"], rec["y_grid"], mode="conformal_lcb")
                mu_tube = r["mu_tube"]
                p_frac, c_frac, ms_v = r["prior_region_frac"], r["clamped_mu_min_frac"], r["predict_time_ms"]

            diff = mu_true - mu_tube
            is_c = diff >= -1e-9
            cov_flags.append(bool(np.all(is_c)))
            pt_fracs.append(float(np.mean(is_c)))
            overests.append(float(np.max(np.maximum(0.0, mu_tube - mu_true))))
            gaps.append(float(np.mean(diff)))
            priors.append(p_frac)
            clamps.append(c_frac)
            pred_ms.append(ms_v)

        n_cov = int(sum(cov_flags))
        lb_1s = float(beta.ppf(0.05, n_cov, n_test_sites - n_cov + 1)) if n_cov > 0 else 0.0
        lo_2s = float(beta.ppf(0.025, n_cov, n_test_sites - n_cov + 1)) if n_cov > 0 else 0.0
        hi_2s = float(beta.ppf(0.975, n_cov + 1, n_test_sites - n_cov)) if n_cov < n_test_sites else 1.0
        return {
            "n_test_sites": n_test_sites,
            "n_traj_covered": n_cov,
            "traj_coverage_rate": round(n_cov / float(n_test_sites), 4),
            "pointwise_coverage_rate": round(float(np.mean(pt_fracs)), 4),
            "clopper_pearson_95_lb": round(lb_1s, 4),
            "clopper_pearson_95_ci_2sided": [round(lo_2s, 4), round(hi_2s, 4)],
            "max_worst_overestimation": round(float(np.max(overests)), 5),
            "mean_conservatism_gap": round(float(np.mean(gaps)), 5),
            "mean_prior_region_frac": round(float(np.mean(priors)), 4),
            "mean_clamped_mu_min_frac": round(float(np.mean(clamps)), 4),
            "mean_predict_time_ms": round(float(np.mean(pred_ms)), 3),
        }

    return {
        "n_cal_sites": n_cal_sites,
        "n_test_sites": n_test_sites,
        "calibration_alpha_005": {
            "k_star": k_050,
            "q_hat_mu": round(q_050, 6),
            "score_mean": round(float(np.mean(cal_scores)), 6),
            "score_max": round(float(np.max(cal_scores)), 6),
            "lag1_z_stat": round(float(z_r1), 4),
        },
        "calibration_alpha_0025": {
            "k_star": k_025,
            "q_hat_mu": round(q_025, 6),
        },
        "test_coverage": {
            "M0": _agg_test("M0"),
            "M1": _agg_test("M1"),
            "M2_multisite": _agg_test("M2"),
        },
    }


def main():
    t_all0 = time.perf_counter()
    print("=" * 96)
    print("CHUẨN BỊ & CHẠY KIỂM CHỨNG SIM-ONLY ĐỢT 1 + ĐỢT 2 (STUDIES 1 - 7)")
    print("=" * 96)

    # -------------------------------------------------------------------------
    # STUDY 1: Protocol A (Same-Site, D_hist=15 Bidir Per-Wheel, D_cal=80, D_test=100)
    # -------------------------------------------------------------------------
    hist_data, cal_trajs, test_trajs = _build_disjoint_datasets(
        n_hist=15, n_cal=80, n_test=100,
        observation_mode="per_wheel_rls_lagcomp", bidirectional=True,
    )
    gp = SpatialFrictionGP(
        mu_prior=0.25, sigma_f=0.12, ell_macro=0.85,
        sigma_micro=0.045, ell_micro=0.22, sigma_obs_floor=0.020,
        mu_min=0.08, mu_max=0.55, sigma_0=0.015, query_mode="footprint",
    )
    gp.fit(hist_data["points_xy"], hist_data["mu_obs"], hist_data["obs_var"], voxel_size=0.15)
    gp_lml = gp.log_marginal_likelihood()
    print(f"[Study 1] D_hist (15 runs bidir per-wheel): {hist_data['n_accepted']}/{hist_data['n_total_steps']} -> "
          f"{len(gp.X_train)} voxels (fit={gp.last_fit_time_ms:.2f}ms, LML={gp_lml:.2f})")

    cal_025 = gp.calibrate_trajectory_conformal(cal_trajs, alpha_mu=0.025)
    cal_050 = gp.calibrate_trajectory_conformal(cal_trajs, alpha_mu=0.05)
    gp.cal_meta["bonferroni_joint_budget"] = {
        "alpha_eta": 0.05,
        "alpha_fz": 0.025,
        "alpha_mu_025": 0.025,
        "alpha_total_with_025": 0.10,
        "q_hat_mu_at_alpha_025": cal_025["q_hat_mu"],
        "k_star_at_alpha_025": cal_025["k_star"],
    }
    cal_out_path = os.path.join(_root, "config", "conformal_calibration_mu.json")
    gp.save_calibration(cal_out_path, include_support=True)
    print(f"          D_cal (N=80): alpha=0.05 -> k*={cal_050['k_star']}, q_hat={cal_050['q_hat_mu']:.4f}, lag1_z={cal_050['lag1_z_stat']:.3f} | "
          f"alpha=0.025 -> k*={cal_025['k_star']}<80, q_hat={cal_025['q_hat_mu']:.4f}")

    cov_m0 = gp.evaluate_coverage(test_trajs, mu_override_fn=lambda x, y: np.full_like(x, 0.35 * 0.95))
    cov_m1 = gp.evaluate_coverage(test_trajs, mode="gp_mean")
    cov_m2 = gp.evaluate_coverage(test_trajs, mode="conformal_lcb")

    _, _, d_net_fn, zc_fn, camber_fn = generate_multi_regime_benchmark_scenario()
    tube_stats = {k: {"v_cp3": [], "v_mean": [], "t_plan_ms": [], "t_gp_ms": [], "prior_f": [], "clamp_f": []}
                  for k in ("M0", "M1", "M2", "M3")}
    for tr in test_trajs:
        wp = tr["waypoints"]
        p0 = CoupledVelocityTubeH124(mu_default=0.35, kappa_th=1.05)
        s0 = p0.build_offline_tube(wp, mu_profile=lambda s: 0.35, d_net_profile=d_net_fn, z_com_profile=zc_fn, phi_camber_profile=camber_fn, run_wcsph_check=False, apply_plan_scale=True)
        p1 = CoupledVelocityTubeH124(mu_default=0.35, kappa_th=1.05)
        s1 = p1.build_offline_tube(wp, d_net_profile=d_net_fn, z_com_profile=zc_fn, phi_camber_profile=camber_fn, run_wcsph_check=False, friction_estimator=gp, friction_mode="gp_mean", apply_plan_scale=False)
        p2 = CoupledVelocityTubeH124(mu_default=0.35, kappa_th=1.05)
        s2 = p2.build_offline_tube(wp, d_net_profile=d_net_fn, z_com_profile=zc_fn, phi_camber_profile=camber_fn, run_wcsph_check=False, friction_estimator=gp, friction_mode="conformal_lcb", apply_plan_scale=False)
        p3 = CoupledVelocityTubeH124(mu_default=0.35, kappa_th=1.05)
        p3._extract_geometry_and_checkpoints(wp)
        _, mu_4w_ex, _ = tr["ground_truth"].eval_path_4w_min(p3.x_grid, p3.y_grid)
        s3 = p3.build_offline_tube(wp, mu_profile=mu_4w_ex, d_net_profile=d_net_fn, z_com_profile=zc_fn, phi_camber_profile=camber_fn, run_wcsph_check=False, apply_plan_scale=False)

        for key, pl, sm in [("M0", p0, s0), ("M1", p1, s1), ("M2", p2, s2), ("M3", p3, s3)]:
            mask_oil = (pl.s_grid >= 2.9) & (pl.s_grid <= 4.1)
            tube_stats[key]["v_cp3"].append(float(np.min(pl.v_ub_nom[mask_oil])))
            tube_stats[key]["v_mean"].append(float(np.mean(pl.v_ub_nom)))
            tube_stats[key]["t_plan_ms"].append(float(sm["plan_time_total_ms"]))
            tube_stats[key]["t_gp_ms"].append(float(sm["friction_predict_time_ms"]))
            tube_stats[key]["prior_f"].append(float(sm["friction_prior_region_frac"]))
            tube_stats[key]["clamp_f"].append(float(sm["friction_clamped_mu_min_frac"]))

    print(f"          D_test (N=100): M0 cov={cov_m0['traj_coverage_rate']*100:.1f}%, "
          f"M1 cov={cov_m1['traj_coverage_rate']*100:.1f}%, "
          f"M2 cov={cov_m2['traj_coverage_rate']*100:.1f}% ({cov_m2['n_traj_covered']}/100, "
          f"gap={cov_m2['mean_conservatism_gap']:+.4f}, clamp={cov_m2['mean_clamped_mu_min_frac']*100:.1f}%, "
          f"prior={cov_m2['mean_prior_region_frac']*100:.1f}%)")
    sys.stdout.flush()

    # -------------------------------------------------------------------------
    # STUDY 2: M3 + M5 Component Ablation (4 Combinations on D_cal=80 / D_test=100)
    # -------------------------------------------------------------------------
    print("[Study 2] M3 + M5 Component Ablation (Center vs Footprint; Chassis vs Per-Wheel LagComp)...")
    hist_legacy = _collect_hist_data(15, observation_mode="chassis_ffrls_legacy", bidirectional=False, seed_start=100)
    comp_ablation = []
    for label, h_src, q_m in [
        ("Legacy (Center + Unidir Chassis)", hist_legacy, "center"),
        ("M5 Only (Center + Bidir Per-Wheel LagComp)", hist_data, "center"),
        ("M3 Only (Footprint + Unidir Chassis)", hist_legacy, "footprint"),
        ("Upgraded M3+M5 (Footprint + Bidir Per-Wheel)", hist_data, "footprint"),
    ]:
        gp_ab = SpatialFrictionGP(query_mode=q_m)
        gp_ab.fit(h_src["points_xy"], h_src["mu_obs"], h_src["obs_var"], voxel_size=0.15)
        c025 = gp_ab.calibrate_trajectory_conformal(cal_trajs, alpha_mu=0.025)
        c050 = gp_ab.calibrate_trajectory_conformal(cal_trajs, alpha_mu=0.05)
        ev = gp_ab.evaluate_coverage(test_trajs, mode="conformal_lcb")
        comp_ablation.append({
            "label": label,
            "query_mode": q_m,
            "observation_mode": h_src["observation_mode"],
            "bidirectional": h_src["bidirectional"],
            "n_voxels": int(len(gp_ab.X_train)),
            "q_hat_mu_005": c050["q_hat_mu"],
            "q_hat_mu_0025": c025["q_hat_mu"],
            "traj_coverage_rate": ev["traj_coverage_rate"],
            "max_worst_overestimation": ev["max_worst_overestimation"],
            "mean_conservatism_gap": ev["mean_conservatism_gap"],
            "mean_prior_region_frac": ev["mean_prior_region_frac"],
            "mean_clamped_mu_min_frac": ev["mean_clamped_mu_min_frac"],
            "mean_predict_time_ms": ev["mean_predict_time_ms"],
        })
        print(f"          {label:<45s} | M={len(gp_ab.X_train):3d} | q={c050['q_hat_mu']:.3f} | "
              f"cov={ev['traj_coverage_rate']*100:5.1f}% | gap={ev['mean_conservatism_gap']:+.4f} | "
              f"clamp={ev['mean_clamped_mu_min_frac']*100:4.1f}% | prior={ev['mean_prior_region_frac']*100:4.1f}%")
    sys.stdout.flush()

    # -------------------------------------------------------------------------
    # STUDY 3: M4 Coverage Density & Prior (mu_prior) Attribution
    # -------------------------------------------------------------------------
    print("[Study 3] M4 Coverage Density & Prior (mu_prior) Attribution...")
    hist_sparse = _collect_hist_data(5, observation_mode="per_wheel_rls_lagcomp", bidirectional=True, seed_start=100)
    test_shifted = _build_traj_split(seed_start=300, count=100, y_shift=0.65, is_test=True)

    density_sweep = []
    for d_lbl, h_src, ts_set in [
        ("Dense (15 runs, nominal corridor)", hist_data, test_trajs),
        ("Sparse (5 runs, nominal corridor)", hist_sparse, test_trajs),
        ("Shifted (+0.65 m unexplored corridor)", hist_data, test_shifted),
    ]:
        gp_d = SpatialFrictionGP(query_mode="footprint")
        gp_d.fit(h_src["points_xy"], h_src["mu_obs"], h_src["obs_var"], voxel_size=0.15)
        c_d = gp_d.calibrate_trajectory_conformal(cal_trajs, alpha_mu=0.05)
        ev_d = gp_d.evaluate_coverage(ts_set, mode="conformal_lcb")
        density_sweep.append({
            "label": d_lbl,
            "n_voxels": int(len(gp_d.X_train)),
            "q_hat_mu": c_d["q_hat_mu"],
            "traj_coverage_rate": ev_d["traj_coverage_rate"],
            "mean_conservatism_gap": ev_d["mean_conservatism_gap"],
            "mean_prior_region_frac": ev_d["mean_prior_region_frac"],
            "mean_clamped_mu_min_frac": ev_d["mean_clamped_mu_min_frac"],
        })
        print(f"          {d_lbl:<40s} | q={c_d['q_hat_mu']:.3f} | cov={ev_d['traj_coverage_rate']*100:5.1f}% | "
              f"prior={ev_d['mean_prior_region_frac']*100:5.1f}% | clamp={ev_d['mean_clamped_mu_min_frac']*100:5.1f}%")

    prior_attribution = []
    for mp in [0.25, 0.30, 0.35]:
        gp_p = SpatialFrictionGP(mu_prior=mp, query_mode="footprint")
        gp_p.fit(hist_data["points_xy"], hist_data["mu_obs"], hist_data["obs_var"], voxel_size=0.15)
        cp = gp_p.calibrate_trajectory_conformal(cal_trajs, alpha_mu=0.05)
        evp = gp_p.evaluate_coverage(test_trajs, mode="conformal_lcb")
        prior_attribution.append({
            "mu_prior": mp,
            "q_hat_mu": cp["q_hat_mu"],
            "traj_coverage_rate": evp["traj_coverage_rate"],
            "max_worst_overestimation": evp["max_worst_overestimation"],
            "mean_conservatism_gap": evp["mean_conservatism_gap"],
            "mean_clamped_mu_min_frac": evp["mean_clamped_mu_min_frac"],
            "lml": round(gp_p.log_marginal_likelihood(), 2),
        })
    sys.stdout.flush()

    # -------------------------------------------------------------------------
    # STUDY 4: Section 6 Kernel Hyperparameter Sensitivity & LML Sweep
    # -------------------------------------------------------------------------
    print("[Study 4] Kernel Hyperparameter Sensitivity & Log Marginal Likelihood (LML)...")
    kernel_sweep = []
    for ell_m, sig_f in [
        (0.25, 0.12), (0.35, 0.12), (0.45, 0.12), (0.65, 0.12),
        (0.85, 0.12), (1.15, 0.12), (0.85, 0.09), (0.85, 0.16),
    ]:
        gp_k = SpatialFrictionGP(sigma_f=sig_f, ell_macro=ell_m, query_mode="footprint")
        gp_k.fit(hist_data["points_xy"], hist_data["mu_obs"], hist_data["obs_var"], voxel_size=0.15)
        ck = gp_k.calibrate_trajectory_conformal(cal_trajs, alpha_mu=0.05)
        evk = gp_k.evaluate_coverage(test_trajs, mode="conformal_lcb")
        kernel_sweep.append({
            "ell_macro_m": ell_m,
            "sigma_f": sig_f,
            "lml": round(gp_k.log_marginal_likelihood(), 2),
            "q_hat_mu": ck["q_hat_mu"],
            "traj_coverage_rate": evk["traj_coverage_rate"],
            "max_worst_overestimation": evk["max_worst_overestimation"],
            "mean_conservatism_gap": evk["mean_conservatism_gap"],
            "mean_clamped_mu_min_frac": evk["mean_clamped_mu_min_frac"],
        })
    sys.stdout.flush()

    # -------------------------------------------------------------------------
    # STUDY 5: Protocol B (Multi-Site Random Floor Layouts: 80 cal sites + 100 unseen test sites)
    # -------------------------------------------------------------------------
    print("[Study 5] Protocol B Multi-Site Random Layouts (80 cal sites + 100 unseen test sites)...")
    protocol_b = _run_protocol_b_multisite(n_cal_sites=80, n_test_sites=100)
    print(f"          Multi-site cal: q_005={protocol_b['calibration_alpha_005']['q_hat_mu']:.4f}, "
          f"q_0025={protocol_b['calibration_alpha_0025']['q_hat_mu']:.4f} | "
          f"100 unseen test sites M2 cov={protocol_b['test_coverage']['M2_multisite']['traj_coverage_rate']*100:.1f}% "
          f"(CI95={protocol_b['test_coverage']['M2_multisite']['clopper_pearson_95_ci_2sided']})")
    sys.stdout.flush()

    # -------------------------------------------------------------------------
    # STUDY 6: Multi-Seed Real-Time Closed-Loop MuJoCo (Seeds 300..309) + 2x2 Tube x MPC
    # -------------------------------------------------------------------------
    print("[Study 6] Real-Time Closed-Loop MuJoCo (10 consecutive seeds 300..309, N=8, it=8, 6 configs incl. 2x2)...", flush=True)
    shared_mj_model = mujoco.MjModel.from_xml_path(os.path.join(_root, "xml", "slosh3.xml"))
    shared_ctrl = DFEMPCCAMRController(
        N=8,
        dt=0.05,
        mu=0.35,
        state_con_mode="skip0",
        mu_plan_scale=1.0,
        ipopt_max_iter=12,
        h_lim_m=0.030,
    )
    cl_seeds = list(range(10))  # Chỉ số trong test_trajs -> seeds 300..309
    config_names = ("M0", "M1", "M2_tube_only", "M2_mpc_only", "M2", "M3")
    per_seed_details = {}
    grouped_runs = {cfg: [] for cfg in config_names}

    ckpt_path = os.path.join(_root, "simulation", "_ablation_study6_ckpt.json")
    ckpt_data = {"study6": {}, "study7": {}}
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path, "r", encoding="utf-8") as f_ck:
                loaded_ck = json.load(f_ck)
            if isinstance(loaded_ck, dict) and "study6" in loaded_ck:
                ckpt_data = loaded_ck
                print(f"          [Resume] Tìm thấy checkpoint {ckpt_path}: {len(ckpt_data.get('study6', {}))} seeds Study 6, {len(ckpt_data.get('study7', {}))} seeds Study 7.", flush=True)
        except Exception:
            ckpt_data = {"study6": {}, "study7": {}}

    def _save_ckpt():
        tmp_p = ckpt_path + ".tmp"
        with open(tmp_p, "w", encoding="utf-8") as f_ck:
            json.dump(ckpt_data, f_ck, ensure_ascii=False)
        os.replace(tmp_p, ckpt_path)

    for i_tr in cl_seeds:
        tr = test_trajs[i_tr]
        s_id = tr["seed"]
        seed_key = f"seed_{s_id}"
        wp = tr["waypoints"]
        gt = tr["ground_truth"]
        min_mu_tr = float(np.min(tr["mu_min_4w_true"]))

        cached_s6 = ckpt_data.get("study6", {}).get(seed_key)
        if isinstance(cached_s6, dict) and all(cfg in cached_s6.get("runs", {}) for cfg in config_names):
            seed_res_raw = cached_s6
            seed_res = {"seed": s_id, "min_mu_4w_true": seed_res_raw["min_mu_4w_true"], "runs": {}}
            for cfg in config_names:
                r_raw = seed_res_raw["runs"][cfg]
                grouped_runs[cfg].append(r_raw)
                seed_res["runs"][cfg] = _strip_raw(r_raw)
            per_seed_details[seed_key] = seed_res
            print(f"          [Resume] seed={s_id} (min_mu={min_mu_tr:.4f}): M0={seed_res['runs']['M0']['slip_rate_pct']:.2f}% | "
                  f"M2_tube={seed_res['runs']['M2_tube_only']['slip_rate_pct']:.2f}% | "
                  f"M2_mpc={seed_res['runs']['M2_mpc_only']['slip_rate_pct']:.2f}% | "
                  f"M2={seed_res['runs']['M2']['slip_rate_pct']:.2f}% (p95={seed_res['runs']['M2']['p95_solve_ms']:.1f}ms)", flush=True)
            continue

        pl_m0 = CoupledVelocityTubeH124(mu_default=0.35, kappa_th=1.05)
        pl_m0.build_offline_tube(wp, mu_profile=lambda s: 0.35, d_net_profile=d_net_fn, z_com_profile=zc_fn, phi_camber_profile=camber_fn, run_wcsph_check=False, apply_plan_scale=True)

        pl_m1 = CoupledVelocityTubeH124(mu_default=0.35, kappa_th=1.05)
        pl_m1.build_offline_tube(wp, d_net_profile=d_net_fn, z_com_profile=zc_fn, phi_camber_profile=camber_fn, run_wcsph_check=False, friction_estimator=gp, friction_mode="gp_mean", apply_plan_scale=False)

        pl_m2 = CoupledVelocityTubeH124(mu_default=0.35, kappa_th=1.05)
        pl_m2.build_offline_tube(wp, d_net_profile=d_net_fn, z_com_profile=zc_fn, phi_camber_profile=camber_fn, run_wcsph_check=False, friction_estimator=gp, friction_mode="conformal_lcb", apply_plan_scale=False)

        pl_m3 = CoupledVelocityTubeH124(mu_default=0.35, kappa_th=1.05)
        pl_m3._extract_geometry_and_checkpoints(wp)
        _, mu_4w_exact, _ = gt.eval_path_4w_min(pl_m3.x_grid, pl_m3.y_grid)
        pl_m3.build_offline_tube(wp, mu_profile=mu_4w_exact, d_net_profile=d_net_fn, z_com_profile=zc_fn, phi_camber_profile=camber_fn, run_wcsph_check=False, apply_plan_scale=False)

        pair_map = {
            "M0": (pl_m0, pl_m0),
            "M1": (pl_m1, pl_m1),
            "M2_tube_only": (pl_m2, pl_m0),
            "M2_mpc_only": (pl_m0, pl_m2),
            "M2": (pl_m2, pl_m2),
            "M3": (pl_m3, pl_m3),
        }
        seed_res = {"seed": s_id, "min_mu_4w_true": round(min_mu_tr, 4), "runs": {}}
        seed_res_raw = {"seed": s_id, "min_mu_4w_true": round(min_mu_tr, 4), "runs": {}}
        for cfg in config_names:
            tp, mp_ctrl = pair_map[cfg]
            r = run_closed_loop_mujoco_trial(
                cfg, wp, tp, gt, mpc_mu_planner=mp_ctrl, d_net_fn=d_net_fn,
                controller_N=8, ipopt_max_iter=12, actuation_delay_cycles=0, enable_deadline_miss=False,
                shared_ctrl=shared_ctrl, shared_mj_model=shared_mj_model,
            )
            grouped_runs[cfg].append(r)
            seed_res_raw["runs"][cfg] = r
            seed_res["runs"][cfg] = _strip_raw(r)
        per_seed_details[seed_key] = seed_res
        ckpt_data["study6"][seed_key] = seed_res_raw
        _save_ckpt()
        print(f"          seed={s_id} (min_mu={min_mu_tr:.4f}): M0={seed_res['runs']['M0']['slip_rate_pct']:.2f}% | "
              f"M2_tube={seed_res['runs']['M2_tube_only']['slip_rate_pct']:.2f}% | "
              f"M2_mpc={seed_res['runs']['M2_mpc_only']['slip_rate_pct']:.2f}% | "
              f"M2={seed_res['runs']['M2']['slip_rate_pct']:.2f}% (p95={seed_res['runs']['M2']['p95_solve_ms']:.1f}ms)", flush=True)

    agg_study6 = {cfg: _aggregate_cl_metrics(grouped_runs[cfg]) for cfg in config_names}
    print("-" * 96)
    print(f"{'Cấu hình (10 seeds)':<16} | {'Lượt trượt':<10} | {'UB95 P(trượt)':<13} | {'Chu kỳ trượt':<12} | {'T_fin TB':<9} | {'Solve TB/p95':<14} | {'>50ms':<7}")
    print("-" * 96)
    for cfg in config_names:
        a = agg_study6[cfg]
        print(f"{cfg:<16} | {a['n_runs_with_slip']:2d}/10      | {a['run_slip_prob_95_ub']*100:8.1f}%     | "
              f"{a['aggregate_slip_rate_pct']:8.3f}%    | {a['mean_t_finish_s']:6.2f} s  | "
              f"{a['mean_solve_ms']:5.1f} / {a['p95_solve_ms']:5.1f}  | {a['pct_over_50ms']:5.2f}%")
    print("-" * 96, flush=True)

    # -------------------------------------------------------------------------
    # STUDY 7: 1-Cycle Predictive Delay (50 ms) + Deadline-Miss (100 ms) Stress Check on 3 Hardest Seeds of 300..309
    # -------------------------------------------------------------------------
    sorted_10 = sorted(test_trajs[:10], key=lambda t: float(np.min(t["mu_min_4w_true"])))
    stress_trajs = sorted_10[:3]
    stress_seeds = [t["seed"] for t in stress_trajs]
    print(f"[Study 7] Stress Check trễ 1 chu kỳ (50 ms) + lỡ hạn (>50ms->100ms) trên 3 seed thấp nhất của 300..309: {stress_seeds}", flush=True)

    delay_grouped = {cfg: [] for cfg in ("M0", "M1", "M2", "M3")}
    nodelay_subset = {cfg: [] for cfg in ("M0", "M1", "M2", "M3")}
    delay_per_seed = {}

    for tr in stress_trajs:
        s_id = tr["seed"]
        seed_key = f"seed_{s_id}"
        wp = tr["waypoints"]
        gt = tr["ground_truth"]
        cached_s7 = ckpt_data.get("study7", {}).get(seed_key)
        if isinstance(cached_s7, dict) and all(cfg in cached_s7.get("runs", {}) for cfg in ("M0", "M1", "M2", "M3")):
            seed_d = {"seed": s_id, "min_mu_4w_true": cached_s7["min_mu_4w_true"], "runs": {}}
            for cfg in ("M0", "M1", "M2", "M3"):
                r_del = cached_s7["runs"][cfg]
                delay_grouped[cfg].append(r_del)
                seed_d["runs"][cfg] = _strip_raw(r_del)
                idx_seed = s_id - 300
                nodelay_subset[cfg].append(grouped_runs[cfg][idx_seed])
            delay_per_seed[seed_key] = seed_d
            continue

        pl_m0 = CoupledVelocityTubeH124(mu_default=0.35, kappa_th=1.05)
        pl_m0.build_offline_tube(wp, mu_profile=lambda s: 0.35, d_net_profile=d_net_fn, z_com_profile=zc_fn, phi_camber_profile=camber_fn, run_wcsph_check=False, apply_plan_scale=True)
        pl_m1 = CoupledVelocityTubeH124(mu_default=0.35, kappa_th=1.05)
        pl_m1.build_offline_tube(wp, d_net_profile=d_net_fn, z_com_profile=zc_fn, phi_camber_profile=camber_fn, run_wcsph_check=False, friction_estimator=gp, friction_mode="gp_mean", apply_plan_scale=False)
        pl_m2 = CoupledVelocityTubeH124(mu_default=0.35, kappa_th=1.05)
        pl_m2.build_offline_tube(wp, d_net_profile=d_net_fn, z_com_profile=zc_fn, phi_camber_profile=camber_fn, run_wcsph_check=False, friction_estimator=gp, friction_mode="conformal_lcb", apply_plan_scale=False)
        pl_m3 = CoupledVelocityTubeH124(mu_default=0.35, kappa_th=1.05)
        pl_m3._extract_geometry_and_checkpoints(wp)
        _, mu_4w_exact, _ = gt.eval_path_4w_min(pl_m3.x_grid, pl_m3.y_grid)
        pl_m3.build_offline_tube(wp, mu_profile=mu_4w_exact, d_net_profile=d_net_fn, z_com_profile=zc_fn, phi_camber_profile=camber_fn, run_wcsph_check=False, apply_plan_scale=False)

        seed_d = {"seed": s_id, "min_mu_4w_true": round(float(np.min(tr["mu_min_4w_true"])), 4), "runs": {}}
        seed_d_raw = {"seed": s_id, "min_mu_4w_true": seed_d["min_mu_4w_true"], "runs": {}}
        for cfg, pl_obj in [("M0", pl_m0), ("M1", pl_m1), ("M2", pl_m2), ("M3", pl_m3)]:
            r_del = run_closed_loop_mujoco_trial(
                cfg, wp, pl_obj, gt, mpc_mu_planner=pl_obj, d_net_fn=d_net_fn,
                controller_N=8, ipopt_max_iter=12, actuation_delay_cycles=1, enable_deadline_miss=True,
                shared_ctrl=shared_ctrl, shared_mj_model=shared_mj_model,
            )
            delay_grouped[cfg].append(r_del)
            seed_d_raw["runs"][cfg] = r_del
            seed_d["runs"][cfg] = _strip_raw(r_del)
            # Match no-delay run on exact same seed
            idx_seed = s_id - 300
            nodelay_subset[cfg].append(grouped_runs[cfg][idx_seed])
        delay_per_seed[seed_key] = seed_d
        ckpt_data["study7"][seed_key] = seed_d_raw
        _save_ckpt()

    agg_delay = {cfg: _aggregate_cl_metrics(delay_grouped[cfg]) for cfg in ("M0", "M1", "M2", "M3")}
    agg_nodelay_sub = {cfg: _aggregate_cl_metrics(nodelay_subset[cfg]) for cfg in ("M0", "M1", "M2", "M3")}
    for cfg in ("M0", "M1", "M2", "M3"):
        print(f"          {cfg} (delay=1 + deadline_miss): slip={agg_delay[cfg]['aggregate_slip_rate_pct']:.2f}%, "
              f"max_slip={agg_delay[cfg]['worst_max_slip']:.4f}, miss={agg_delay[cfg]['total_deadline_miss_steps']}, "
              f"T_fin={agg_delay[cfg]['mean_t_finish_s']:.2f}s, p95={agg_delay[cfg]['p95_solve_ms']:.1f}ms")

    # -------------------------------------------------------------------------
    # XUẤT JSON TỔNG HỢP (BAO GỒM CẢ ALIAS CHO TƯƠNG THÍCH NGƯỢC)
    # -------------------------------------------------------------------------
    out_json = os.path.join(_root, "simulation", "spatial_friction_ablation_results.json")
    payload_out = {
        "version": "2.0",
        "total_runtime_s": round(float(time.perf_counter() - t_all0), 2),
        "history_collection": {k: v for k, v in hist_data.items() if k not in ("points_xy", "mu_obs", "obs_var")},
        "n_gp_voxel_support": int(len(gp.X_train)),
        "gp_fit_time_ms": round(float(gp.last_fit_time_ms), 3),
        "gp_log_marginal_likelihood": round(float(gp_lml), 3),
        "calibration_alpha_005": {k: v for k, v in cal_050.items() if k != "scores"},
        "calibration_alpha_0025": {k: v for k, v in cal_025.items() if k != "scores"},
        "test_coverage": {
            "M0": cov_m0,
            "M1": cov_m1,
            "M2": cov_m2,
        },
        "tube_summary_100_seeds": {
            k: {
                "mean_v_min_cp3_m_s": round(float(np.mean(v["v_cp3"])), 4),
                "mean_v_tube_m_s": round(float(np.mean(v["v_mean"])), 4),
                "mean_plan_time_ms": round(float(np.mean(v["t_plan_ms"])), 2),
                "mean_gp_predict_ms": round(float(np.mean(v["t_gp_ms"])), 4),
                "mean_prior_region_frac": round(float(np.mean(v["prior_f"])), 4),
                "mean_clamped_mu_min_frac": round(float(np.mean(v["clamp_f"])), 4),
            }
            for k, v in tube_stats.items()
        },
        "study2_component_ablation": comp_ablation,
        "study3_m4_and_prior": {
            "coverage_density": density_sweep,
            "prior_attribution": prior_attribution,
        },
        "study4_kernel_sensitivity": kernel_sweep,
        "study5_protocol_b_multisite": protocol_b,
        "study6_closed_loop_10seeds_2x2": {
            "seeds": [int(300 + i) for i in cl_seeds],
            "controller_config": {
                "N": 8,
                "dt_s": 0.05,
                "ipopt_max_iter": 12,
                "warm_start": "to_function+adaptive+dual_push5e-6",
                "predeparture_warmup": True,
            },
            "aggregate_summary": agg_study6,
            "per_seed_details": per_seed_details,
        },
        "study7_actuation_delay_50ms": {
            "stress_seeds": stress_seeds,
            "actuation_delay_cycles": 1,
            "enable_deadline_miss": True,
            "actuation_delay_mode": "1_cycle_predictive + fallback_deadline_miss_50ms",
            "aggregate_summary_delayed": agg_delay,
            "aggregate_summary_nodelay_same_3_seeds": agg_nodelay_sub,
            "per_seed_details": delay_per_seed,
        },
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload_out, f, indent=2, ensure_ascii=False)
    if os.path.exists(ckpt_path):
        try:
            os.remove(ckpt_path)
        except OSError:
            pass
    print(f"\n[HOÀN TẤT] Đã lưu toàn bộ kết quả 7 nghiên cứu ra: {out_json} (trong {payload_out['total_runtime_s']:.1f} s)")


if __name__ == "__main__":
    main()
