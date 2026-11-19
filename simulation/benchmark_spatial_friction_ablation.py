# -*- coding: utf-8 -*-
"""
benchmark_spatial_friction_ablation.py
=============================================================================
KIỂM CHỨNG SIM-ONLY ĐỊNH LƯỢNG 4 CẤU HÌNH (M0 - M3) CHO MÔ HÌNH ƯỚC LƯỢNG
MA SÁT TIỀN KHỞI HÀNH (SPATIAL GP + TRAJECTORY-LEVEL CONFORMAL LCB)

Quy trình tách tập nghiêm ngặt (không Data Leakage):
1. D_hist (15 lượt chạy quá khứ, seeds 100..114):
   Thu thập mẫu nhân quả qua FrictionObserverFFRLS -> gom cụm voxel -> fit SpatialFrictionGP.
2. D_cal (60 quỹ đạo độc lập, seeds 200..259):
   Hiệu chuẩn Split Conformal cấp quỹ đạo (bao trọn 4-wheel split-mu & ISO 8608 Class B)
   tại alpha_mu = 0.05 (và ghi nhận ngưỡng Bonferroni alpha_mu = 0.025).
   Xuất ra config/conformal_calibration_mu.json.
3. D_test (30 quỹ đạo chưa từng thấy, seeds 300..329):
   Đánh giá 4 cấu hình trên bài toán hoạch định Ống vận tốc tiền khởi hành:
     - M0 (Constant Naive)   : mu = 0.35 * 0.95 (chưa biết vùng trơn / vi mô)
     - M1 (GP Mean Only)     : mu = mu_hat_GP (chưa hiệu chuẩn Conformal)
     - M2 (GP+Conformal LCB) : ĐỀ XUẤT mu = mu_hat - q_hat*(sigma + sigma_0), scale=1.0
     - M3 (Oracle 4-Wheel)   : Biết trước min_w mu_true(p_w(s)), scale=1.0
4. Đóng vòng MuJoCo (xml/slosh3.xml) + DFEMPCCAMRController (P_mu horizon) trên seed=300:
   Mô phỏng bão hòa bám 4 bánh thực tế với mu_wheels_true(x, y, yaw) & ISO 8608 Class B.
=============================================================================
"""

import os
import sys
import json
import time
import numpy as np
import mujoco

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

_curr = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_curr) if os.path.basename(_curr) in ["simulation", "controller", "solver", "planning", "tests"] else _curr
for _p in [_root, os.path.join(_root, "planning"), os.path.join(_root, "controller"),
           os.path.join(_root, "solver"), os.path.join(_root, "simulation")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from planning.spatial_friction_gp import (
    SpatialFrictionGP,
    make_benchmark_curve_ground_truth,
    collect_ffrls_history_from_rollouts,
)
from planning.velocity_tube_h124 import (
    CoupledVelocityTubeH124,
    _analytical_benchmark_curve,
    generate_multi_regime_benchmark_scenario,
)
from controller.dfe_mpcc_amr_controller import DFEMPCCAMRController
from controller.mecanum_traction_dynamics import (
    M_TOTAL, G,
    compute_dynamic_normal_loads,
    allocate_wheel_forces_clamped,
    evaluate_traction_and_slip,
)
from solver.robot_model_amr_casadi import (
    TANK_A, TANK_B, H_LIQUID, m_1x, lx, omega_x, m_1y, ly, omega_y
)


def _perturbed_curve_waypoints(y_offset: float = 0.0, y_amp_scale: float = 1.0, n_pts: int = 200) -> np.ndarray:
    x_arr = np.linspace(0.0, 6.75, n_pts)
    y_base, _, _, _ = _analytical_benchmark_curve(x_arr)
    envelope = np.sin(np.pi * (x_arr / 6.75))
    y_arr = y_amp_scale * y_base + y_offset * envelope
    return np.column_stack([x_arr, y_arr])


def _build_disjoint_datasets():
    # 1. D_hist: 15 lượt chạy quá khứ qua FF-RLS (seeds 100..114)
    rng_hist = np.random.default_rng(100)
    pts_all, mu_all, var_all = [], [], []
    n_total_steps, n_accepted, n_skipped = 0, 0, 0

    for s_id in range(100, 115):
        gt_h = make_benchmark_curve_ground_truth(seed=s_id, road_class="B", variation_scale=0.15)
        wp_h = _perturbed_curve_waypoints(
            y_offset=float(rng_hist.uniform(-0.32, 0.32)),
            y_amp_scale=float(rng_hist.uniform(0.88, 1.12)),
            n_pts=185,
        )
        res_h = collect_ffrls_history_from_rollouts(gt_h, [wp_h], seed=s_id + 1000)
        n_total_steps += res_h["n_total_steps"]
        n_accepted += res_h["n_accepted"]
        n_skipped += res_h["n_deadzone_skipped"]
        if res_h["n_accepted"] > 0:
            pts_all.append(res_h["points_xy"])
            mu_all.append(res_h["mu_obs"])
            var_all.append(res_h["obs_var"])

    hist_summary = {
        "n_runs": 15,
        "n_total_steps": int(n_total_steps),
        "n_accepted": int(n_accepted),
        "n_deadzone_skipped": int(n_skipped),
        "points_xy": np.vstack(pts_all),
        "mu_obs": np.concatenate(mu_all),
        "obs_var": np.concatenate(var_all),
    }

    # 2. D_cal: 60 quỹ đạo hiệu chuẩn độc lập (seeds 200..259)
    rng_cal = np.random.default_rng(200)
    cal_trajs = []
    for s_id in range(200, 260):
        gt_c = make_benchmark_curve_ground_truth(seed=s_id, road_class="B", variation_scale=0.15)
        wp_c = _perturbed_curve_waypoints(
            y_offset=float(rng_cal.uniform(-0.22, 0.22)),
            y_amp_scale=float(rng_cal.uniform(0.92, 1.08)),
            n_pts=185,
        )
        mu_c, mu_4w, split_del = gt_c.eval_path_4w_min(wp_c[:, 0], wp_c[:, 1])
        cal_trajs.append({
            "seed": s_id,
            "waypoints": wp_c,
            "x_grid": wp_c[:, 0],
            "y_grid": wp_c[:, 1],
            "mu_center_true": mu_c,
            "mu_min_4w_true": mu_4w,
            "split_delta": split_del,
            "ground_truth": gt_c,
        })

    # 3. D_test: 30 quỹ đạo kiểm thử chưa từng thấy (seeds 300..329)
    rng_test = np.random.default_rng(300)
    test_trajs = []
    for s_id in range(300, 330):
        gt_t = make_benchmark_curve_ground_truth(seed=s_id, road_class="B", variation_scale=0.15)
        # Đặt seed=300 trùng quỹ đạo benchmark chuẩn để đối chiếu trực tiếp với kịch bản 4 Checkpoint
        if s_id == 300:
            wp_t = _perturbed_curve_waypoints(y_offset=0.0, y_amp_scale=1.0, n_pts=250)
        else:
            wp_t = _perturbed_curve_waypoints(
                y_offset=float(rng_test.uniform(-0.22, 0.22)),
                y_amp_scale=float(rng_test.uniform(0.92, 1.08)),
                n_pts=200,
            )
        mu_c, mu_4w, split_del = gt_t.eval_path_4w_min(wp_t[:, 0], wp_t[:, 1])
        test_trajs.append({
            "seed": s_id,
            "waypoints": wp_t,
            "x_grid": wp_t[:, 0],
            "y_grid": wp_t[:, 1],
            "mu_center_true": mu_c,
            "mu_min_4w_true": mu_4w,
            "split_delta": split_del,
            "ground_truth": gt_t,
        })

    return hist_summary, cal_trajs, test_trajs


def run_closed_loop_mujoco_trial(
    mode_key: str,
    waypoints: np.ndarray,
    tube_planner: CoupledVelocityTubeH124,
    ground_truth,
    d_net_fn=None,
    max_time_s: float = 28.0,
) -> dict:
    """
    Chạy mô phỏng đóng vòng thực tế trong MuJoCo (xml/slosh3.xml) với DFEMPCCAMRController
    và bão hòa lực bám 4 bánh Coulomb dưới ma sát thực mu_wheels_true(x, y, yaw) + ISO 8608.
    """
    xml_path = os.path.join(_root, "xml", "slosh3.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    dt_sim = float(model.opt.timestep)  # 0.002 s trên slosh3.xml
    nmpc_dt = 0.05
    substeps = int(round(nmpc_dt / dt_sim))

    # Trong chế độ M0/M1/M2/M3, bộ điều khiển dùng đúng ma sát quy hoạch của mô hình đó
    ctrl = DFEMPCCAMRController(
        N=15,
        dt=nmpc_dt,
        mu=0.35,
        state_con_mode="skip0",
        mu_plan_scale=tube_planner.mu_effective_plan_scale,
        ipopt_max_iter=50,
        h_lim_m=0.030,
    )
    ctrl.set_path(waypoints)
    ctrl.set_tube_planner(tube_planner)

    mujoco.mj_resetData(model, data)
    start_pos = waypoints[0]
    goal_pos = waypoints[-1]
    _, _, heading_init = ctrl.get_path_reference(0.0)
    data.qpos[0] = start_pos[0]
    data.qpos[1] = start_pos[1]
    data.qpos[2] = heading_init
    mujoco.mj_forward(model, data)

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

    coth_x = 1.0 / float(np.tanh(np.pi * H_LIQUID / TANK_A))
    coth_y = 1.0 / float(np.tanh(np.pi * H_LIQUID / TANK_B))

    slip_steps = 0
    total_cycles = 0
    max_slip_val = 0.0
    max_eta_mm = 0.0
    min_fz_N = 100.0
    max_contour_err_m = 0.0
    solve_times_ms = []
    suboptimal_steps = 0

    for _ in range(max_steps):
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

        # Nhiễu tải pháp tuyến & độ nghiêng ISO 8608 Class B tại vị trí hiện tại
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

        # Truy vấn chân trời ống vận tốc v_ub(s_k) và ma sát hoạch định mu_plan(s_k) tại t = 0..N-1
        vs_preview = max(vs_prog, 0.35)
        s_preview = np.clip(s_prog + np.arange(ctrl.N) * nmpc_dt * vs_preview, 0.0, ctrl.path_length)
        v_ub_h = np.clip(np.interp(s_preview, tube_planner.s_grid, tube_planner.v_ub_nom), 0.15, 0.70).reshape(1, ctrl.N)
        v_lb_h = np.zeros((1, ctrl.N))
        mu_h = tube_planner.get_horizon_mu_plan(s_preview).reshape(1, ctrl.N)

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
        if info.get("status") != "optimal":
            suboptimal_steps += 1

        s_prog = float(x_pred[9, 1])
        vs_prog = float(x_pred[10, 1])
        as_prog = float(x_pred[11, 1])
        a_act_next = np.array(x_pred[6:9, 1], dtype=float)

        p_ref_now, _, _ = ctrl.get_path_reference(s_prog)
        max_contour_err_m = max(max_contour_err_m, float(np.hypot(q_meas[0] - p_ref_now[0], q_meas[1] - p_ref_now[1])))

        q0_cyc = q_meas.copy()
        v0_cyc = v_meas.copy()
        a0_cyc = a_act.copy()
        u0_cyc = u_opt[:3].copy()
        a_act = a_act_next

        cyc_slip = False
        for sub in range(substeps):
            tau_sub = (sub + 1) * dt_sim
            q_target = q0_cyc + tau_sub * v0_cyc + 0.5 * (tau_sub ** 2) * a0_cyc + (tau_sub ** 3 / 6.0) * u0_cyc
            v_target = v0_cyc + tau_sub * a0_cyc + 0.5 * (tau_sub ** 2) * u0_cyc
            a_target = a0_cyc + tau_sub * u0_cyc

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
            road_sub = ground_truth.road_profile.compute_wheel_loads_and_slosh_excitation(
                q_now, v_now, mu_nominal=ground_truth.macro_fn, friction_variation_scale=ground_truth.variation_scale
            )
            Fz_true = np.maximum(1.5, Fz_nom + road_sub["dfz_iso_wheels"])
            min_fz_N = min(min_fz_N, float(np.min(Fz_true)))

            # Bộ điều khiển phân bổ lực theo ma sát hoạch định hiện thời (của M0/M1/M2/M3)
            mu_belief = float(np.interp(s_prog, tube_planner.s_grid, tube_planner.mu_grid)) * tube_planner.mu_effective_plan_scale
            f_bar_belief = (mu_belief / np.sqrt(2.0)) * Fz_true
            f_wheels_cmd, _, _ = allocate_wheel_forces_clamped(F_cmd_body, f_bar_belief, Fz=Fz_true, mu=mu_belief)

            # Đánh giá bão hòa bám thực tế dưới ma sát 4 bánh THẬT (không lộ cho bộ điều khiển!)
            _, mu_wheels_true, _ = ground_truth.eval_four_wheels(q_now[0], q_now[1], th_now)
            _, f_chassis_act_body, slip_metrics = evaluate_traction_and_slip(
                f_wheels_cmd, Fz_true, mu=mu_wheels_true
            )

            if slip_metrics["any_slip"]:
                cyc_slip = True
            max_slip_val = max(max_slip_val, float(slip_metrics["max_slip"]))

            # Bù trừ suy hao khớp trượt giả tạo trong XML để động lực học khớp chặt cân bằng Newton-Euler
            data.qfrc_applied[:3] = -data.qfrc_passive[:3]
            F_act_world = np.array([
                c_th * f_chassis_act_body[0] - s_th * f_chassis_act_body[1],
                s_th * f_chassis_act_body[0] + c_th * f_chassis_act_body[1],
                f_chassis_act_body[2],
            ])
            data.ctrl[:3] = np.clip(F_act_world, [-40.0, -40.0, -15.0], [40.0, 40.0, 15.0])
            mujoco.mj_step(model, data)
            sim_time += dt_sim

            # Tính độ cao sóng mặt thoáng (Stokes bậc 2)
            ux = abs(np.tan(float(data.qpos[3])))
            uy = abs(np.tan(float(data.qpos[4])))
            ex_mm = (0.5 * TANK_A * ux + (np.pi * TANK_A / 8.0) * coth_x * (ux ** 2)) * 1000.0
            ey_mm = (0.5 * TANK_B * uy + (np.pi * TANK_B / 8.0) * coth_y * (uy ** 2)) * 1000.0
            eta_now = max(ey_mm + 0.35 * ex_mm, ex_mm + 0.35 * ey_mm)
            max_eta_mm = max(max_eta_mm, eta_now)

        total_cycles += 1
        if cyc_slip:
            slip_steps += 1

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
        "mean_solve_ms": round(float(np.mean(solve_times_ms)) if solve_times_ms else 0.0, 2),
        "suboptimal_steps": int(suboptimal_steps),
    }


def _run_4way_closed_loop(tr: dict, gp: SpatialFrictionGP, d_net_fn, zc_fn, camber_fn) -> dict:
    """Chạy đóng vòng MuJoCo (xml/slosh3.xml) cho cả 4 cấu hình M0..M3 trên cùng một quỹ đạo test."""
    wp = tr["waypoints"]
    gt = tr["ground_truth"]

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

    cl_results = {}
    for m_key, pl_obj in [("M0", pl_m0), ("M1", pl_m1), ("M2", pl_m2), ("M3", pl_m3)]:
        cl_results[m_key] = run_closed_loop_mujoco_trial(m_key, wp, pl_obj, gt, d_net_fn=d_net_fn)

    print("-" * 94)
    print(f"{'Cấu hình':<8} | {'Đích':<6} | {'T_hoàn thành':<12} | {'Tỷ lệ trượt':<11} | {'Max Slip':<9} | {'Eta_max':<9} | {'Fz_min':<8} | {'Solve TB':<8}")
    print("-" * 94)
    for m_key in ("M0", "M1", "M2", "M3"):
        r = cl_results[m_key]
        print(f"{m_key:<8} | {str(r['goal_reached']):<6} | {r['t_finish_s']:8.2f} s   | {r['slip_rate_pct']:8.2f} %  | "
              f"{r['max_slip']:9.4f} | {r['max_eta_mm']:6.2f} mm | {r['min_fz_N']:5.2f} N | {r['mean_solve_ms']:5.2f} ms")
    print("-" * 94)
    return cl_results


def main():
    print("=" * 94)
    print("KIỂM CHỨNG SIM-ONLY: ƯỚC LƯỢNG MA SÁT KHÔNG GIAN GP + CONFORMAL LCB (M0 - M3)")
    print("=" * 94)

    # 1. Tạo 3 tập dữ liệu tách rời nghiêm ngặt
    hist_data, cal_trajs, test_trajs = _build_disjoint_datasets()
    print(f"[1/4] D_hist (FF-RLS nhân quả, 15 runs): {hist_data['n_accepted']}/{hist_data['n_total_steps']} mẫu kích thích "
          f"(bỏ qua {hist_data['n_deadzone_skipped']} bước dead-zone)")

    gp = SpatialFrictionGP(
        mu_prior=0.25,
        sigma_f=0.12,
        ell_macro=0.85,
        sigma_micro=0.045,
        ell_micro=0.22,
        sigma_obs_floor=0.020,
        mu_min=0.08,
        mu_max=0.55,
        sigma_0=0.015,
    )
    gp.fit(hist_data["points_xy"], hist_data["mu_obs"], hist_data["obs_var"], voxel_size=0.15)
    print(f"      -> Gom cụm còn {len(gp.X_train)} điểm hỗ trợ voxel (fit trong {gp.last_fit_time_ms:.2f} ms)")

    # 2. Hiệu chuẩn Conformal cấp quỹ đạo trên D_cal (60 quỹ đạo độc lập)
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
    print(f"[2/4] D_cal (N_cal={cal_050['n_cal']} quỹ đạo, bao trọn 4-wheel split-mu):")
    print(f"      -> alpha_mu=0.050: k*={cal_050['k_star']}, q_hat_mu={cal_050['q_hat_mu']:.4f}, "
          f"lag1_z={cal_050['lag1_z_stat']:.3f} (exchangeability_ok={cal_050['exchangeability_ok']})")
    print(f"      -> alpha_mu=0.025 (Bonferroni alpha_tot=0.10): k*={cal_025['k_star']}, q_hat_mu={cal_025['q_hat_mu']:.4f}")
    print(f"      -> Đã lưu vào: {cal_out_path}")

    # 3. Đánh giá bao phủ & độ bảo thủ trên D_test (30 quỹ đạo chưa từng thấy)
    print(f"\n[3/4] ĐÁNH GIÁ TRÊN TẬP KIỂM THỬ ĐỘC LẬP D_test (N_test = {len(test_trajs)} quỹ đạo):")
    cov_m0 = gp.evaluate_coverage(test_trajs, mu_override_fn=lambda x, y: np.full_like(x, 0.35 * 0.95))
    cov_m1 = gp.evaluate_coverage(test_trajs, mode="gp_mean")
    cov_m2 = gp.evaluate_coverage(test_trajs, mode="conformal_lcb")

    # Đánh giá tốc độ & nút thắt ống vận tốc trên 30 quỹ đạo cho M0..M3
    _, _, d_net_fn, zc_fn, camber_fn = generate_multi_regime_benchmark_scenario()
    tube_stats = {k: {"v_cp3": [], "v_mean": [], "t_plan_ms": [], "t_gp_ms": []} for k in ("M0", "M1", "M2", "M3")}
    for tr in test_trajs:
        wp = tr["waypoints"]
        # M0
        p0 = CoupledVelocityTubeH124(mu_default=0.35, kappa_th=1.05)
        s0 = p0.build_offline_tube(wp, mu_profile=lambda s: 0.35, d_net_profile=d_net_fn, z_com_profile=zc_fn, phi_camber_profile=camber_fn, run_wcsph_check=False, apply_plan_scale=True)
        # M1
        p1 = CoupledVelocityTubeH124(mu_default=0.35, kappa_th=1.05)
        s1 = p1.build_offline_tube(wp, d_net_profile=d_net_fn, z_com_profile=zc_fn, phi_camber_profile=camber_fn, run_wcsph_check=False, friction_estimator=gp, friction_mode="gp_mean", apply_plan_scale=False)
        # M2
        p2 = CoupledVelocityTubeH124(mu_default=0.35, kappa_th=1.05)
        s2 = p2.build_offline_tube(wp, d_net_profile=d_net_fn, z_com_profile=zc_fn, phi_camber_profile=camber_fn, run_wcsph_check=False, friction_estimator=gp, friction_mode="conformal_lcb", apply_plan_scale=False)
        # M3 (Oracle 4-wheel min)
        p3 = CoupledVelocityTubeH124(mu_default=0.35, kappa_th=1.05)
        # Đánh giá đúng trên s_grid của ống
        s3 = p3.build_offline_tube(
            wp,
            mu_profile=lambda s_q, _wp=wp, _tr=tr: float(np.interp(s_q, np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(_wp[:,0]), np.diff(_wp[:,1]))))), _tr["mu_min_4w_true"])),
            d_net_profile=d_net_fn, z_com_profile=zc_fn, phi_camber_profile=camber_fn, run_wcsph_check=False, apply_plan_scale=False
        )

        for key, pl, sm in [("M0", p0, s0), ("M1", p1, s1), ("M2", p2, s2), ("M3", p3, s3)]:
            mask_oil = (pl.s_grid >= 2.9) & (pl.s_grid <= 4.1)
            tube_stats[key]["v_cp3"].append(float(np.min(pl.v_ub_nom[mask_oil])))
            tube_stats[key]["v_mean"].append(float(np.mean(pl.v_ub_nom)))
            tube_stats[key]["t_plan_ms"].append(float(sm["plan_time_total_ms"]))
            tube_stats[key]["t_gp_ms"].append(float(sm["friction_predict_time_ms"]))

    print("-" * 94)
    print(f"{'Cấu hình':<26} | {'Phủ quỹ đạo':<12} | {'Phủ điểm':<10} | {'Overest max':<11} | {'Gap TB':<8} | {'v_min(CP3)':<10} | {'t_GP (ms)':<9}")
    print("-" * 94)
    for name, cov, tk in [
        ("M0: Constant (0.35*0.95)", cov_m0, "M0"),
        ("M1: GP Mean Only", cov_m1, "M1"),
        ("M2: GP + Conformal LCB", cov_m2, "M2"),
    ]:
        print(f"{name:<26} | {cov['traj_coverage_rate']*100:6.1f}% ({cov['n_traj_covered']:2d}/{cov['n_test']}) | "
              f"{cov['pointwise_coverage_rate']*100:7.2f}%  | {cov['max_worst_overestimation']:11.4f} | "
              f"{cov['mean_conservatism_gap']:+8.4f} | {np.mean(tube_stats[tk]['v_cp3']):7.3f} m/s | "
              f"{cov['mean_predict_time_ms']:7.3f}")
    print(f"{'M3: Oracle 4-Wheel Min':<26} | {'100.0% (30/30)':<12} | {'100.00%':<10} | {'0.0000':<11} | {'0.0000':<8} | "
          f"{np.mean(tube_stats['M3']['v_cp3']):7.3f} m/s | {'0.000':<9}")
    print("-" * 94)

    # 4. Đóng vòng MuJoCo + DFEMPCCAMRController trên seed danh định (seed=300) và seed bất lợi nhất trong D_test
    tr_nom = test_trajs[0]
    worst_tr = min(test_trajs, key=lambda t: float(np.min(t["mu_min_4w_true"])))

    print(f"\n[4/4] CHẠY MÔ PHỎNG ĐÓNG VÒNG MUJOCO (xml/slosh3.xml):")
    print(f"  (A) Quỹ đạo danh định seed={tr_nom['seed']} (min mu_4w={float(np.min(tr_nom['mu_min_4w_true'])):.4f}):")
    cl_nom = _run_4way_closed_loop(tr_nom, gp, d_net_fn, zc_fn, camber_fn)

    print(f"  (B) Quỹ đạo ISO Class B bất lợi seed={worst_tr['seed']} (min mu_4w={float(np.min(worst_tr['mu_min_4w_true'])):.4f}):")
    cl_adv = _run_4way_closed_loop(worst_tr, gp, d_net_fn, zc_fn, camber_fn)
    print("=" * 94)

    out_json = os.path.join(_root, "simulation", "spatial_friction_ablation_results.json")
    payload_out = {
        "history_collection": {k: v for k, v in hist_data.items() if k not in ("points_xy", "mu_obs", "obs_var")},
        "n_gp_voxel_support": int(len(gp.X_train)),
        "gp_fit_time_ms": round(float(gp.last_fit_time_ms), 3),
        "calibration_alpha_005": {k: v for k, v in cal_050.items() if k != "scores"},
        "calibration_alpha_0025": {k: v for k, v in cal_025.items() if k != "scores"},
        "test_coverage": {
            "M0": cov_m0,
            "M1": cov_m1,
            "M2": cov_m2,
        },
        "tube_summary_30_seeds": {
            k: {
                "mean_v_min_cp3_m_s": round(float(np.mean(v["v_cp3"])), 4),
                "mean_v_tube_m_s": round(float(np.mean(v["v_mean"])), 4),
                "mean_plan_time_ms": round(float(np.mean(v["t_plan_ms"])), 2),
                "mean_gp_predict_ms": round(float(np.mean(v["t_gp_ms"])), 4),
            }
            for k, v in tube_stats.items()
        },
        "closed_loop_mujoco": {
            f"nominal_seed_{tr_nom['seed']}": cl_nom,
            f"adverse_iso_seed_{worst_tr['seed']}": cl_adv,
        },
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload_out, f, indent=2, ensure_ascii=False)
    print(f"Đã xuất toàn bộ kết quả định lượng ra: {out_json}")


if __name__ == "__main__":
    main()
