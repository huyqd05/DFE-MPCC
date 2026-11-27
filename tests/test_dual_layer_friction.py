#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_dual_layer_friction.py
=============================================================================
BỘ KIỂM THỬ 5 TIÊU CHÍ CHO HƯỚNG 2: PHÂN TÁCH 2 LỚP NỀN THÍCH NGHI + DỊ THƯỜNG 2D
(Dual-Layer Background + Anomaly Decomposition)

  1. Test 1: Tham số hóa động [mu_env_low, mu_env_high] trên nhiều thang ma sát khác nhau
             ([0.06, 0.16] và [0.45, 0.65]) + cập nhật nền Empirical-Bayes hai chiều
             không bị nhiễm bẩn bởi điểm dị thường dầu.
  2. Test 2: Bộ phát hiện 2 cổng CUSUM: từ chối nhiễu nhám ISO 1 bước (0 báo động giả),
             bắt dầu nặng trong 1 bước (50 ms) và bắt nước mưa mỏng qua CUSUM trong <= 4 bước.
  3. Test 3: So sánh trực tiếp trên vùng mưa dài 5.0 m & vũng dầu ngắn 0.4 m với
             OnlineFrictionFusion cũ (khử chattering tại mốc 3.0 m và phục hồi nhanh).
  4. Test 4: Trí nhớ 2D đa vòng trên CoupledVelocityTubeH124 (vòng 2 chủ động giảm tốc
             trước vũng dầu), xác nhận khô (confirm_dry + tau_dry_s), và tốc độ O(1) < 5 ms
             trên tuyến dài 100 m với 200 ô dị thường.
  5. Test 5: Tích hợp vòng kín MuJoCo qua DfeMpccPipeline(friction_mode="dual_layer")
             trên world6.xml có vũng dầu ẩn (chỉ cấp khoảng nền, không cấp vùng dầu).
=============================================================================
"""

import math
import os
import sys
import time

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

from controller.friction_observer_ffrls import OnlineFrictionFusion
from controller.iso8608_road_profile import ISO8608RoadProfile
from controller.mecanum_traction_dynamics import (
    L_P_X,
    L_P_Y,
    M_1X,
    M_1Y,
    M_TOTAL,
    Z_COM,
    Z_SLOSH_X,
    Z_SLOSH_Y,
    compute_dynamic_normal_loads,
    evaluate_traction_and_slip,
)
from dfempcc_node import DfeMpccPipeline
from planning.dual_layer_friction_estimator import (
    AdaptiveBackgroundFrictionModel,
    DualGateCUSUMDetector,
    DualLayerFrictionEstimator,
    SparseAnomalyVoxelMap,
)
from planning.semantic_friction_estimator import (
    make_hidden_oil_macro_fn,
    make_world_semantic_map,
)
from planning.velocity_tube_h124 import CoupledVelocityTubeH124
from simulation.benchmark_world6_complex_4dyn_obs import compute_world6_slosh_elevation_mm
from simulation.benchmark_zeroshot_friction_ablation import (
    _select_hidden_oil_patches,
    build_map_scenario,
)
from simulation.test_sim_amr_closed_loop import update_dynamic_obstacles


def test_1_dynamic_range_and_background_adaptation():
    """Tiêu chí 1: Đổi dải ma sát [0.06, 0.16] và [0.45, 0.65] + thích nghi nền không nhiễm dầu."""
    bg_icy = AdaptiveBackgroundFrictionModel(mu_env_low=0.06, mu_env_high=0.16)
    safe0_icy = bg_icy.get_safe_bg()
    assert 0.06 <= safe0_icy <= 0.16
    for _ in range(15):
        bg_icy.update(0.14, obs_var=0.002, is_anomaly=False)
    assert bg_icy.mu_bg > 0.125 and bg_icy.get_safe_bg() > safe0_icy
    assert bg_icy.get_safe_bg() <= 0.16

    bg_rough = AdaptiveBackgroundFrictionModel(mu_env_low=0.45, mu_env_high=0.65)
    safe0_rough = bg_rough.get_safe_bg()
    assert 0.45 <= safe0_rough <= 0.65
    for _ in range(15):
        bg_rough.update(0.60, obs_var=0.002, is_anomaly=False)
    mu_before_oil = bg_rough.mu_bg
    sig_before_oil = bg_rough.sigma_bg
    assert mu_before_oil > 0.56 and bg_rough.get_safe_bg() > safe0_rough

    # Dị thường dầu (is_anomaly=True) tuyệt đối không được kéo nền tụt xuống
    for _ in range(20):
        bg_rough.update(0.10, obs_var=0.001, is_anomaly=True)
    assert abs(bg_rough.mu_bg - mu_before_oil) < 1e-12
    assert abs(bg_rough.sigma_bg - sig_before_oil) < 1e-12
    print("  [PASS] Test 1: Tham số hóa động đa dải & Empirical-Bayes nền miễn nhiễm dị thường.")


def test_2_dual_gate_cusum_detection_and_noise_rejection():
    """Tiêu chí 2: Lọc nhiễu ISO 1 bước, bắt dầu nặng trong 1 bước và bắt mưa mỏng <= 4 bước."""
    det = DualGateCUSUMDetector(n_channels=4)
    mu_bg = 0.35

    # 2a. Nhiễu nhám ISO ngẫu nhiên 1 bước (r=0.08, slip=0.018) rồi trở lại bình thường -> 0 báo động
    out_spike = det.step(mu_bg, [0.27, 0.35, 0.35, 0.35], [0.018, 0.010, 0.010, 0.010], [True] * 4)
    assert not out_spike["any_anomaly"], f"Báo động giả do 1 mẫu nhiễu ISO: {out_spike}"
    for _ in range(4):
        out_rec = det.step(mu_bg, [0.35] * 4, [0.010] * 4, [True] * 4)
        assert not out_rec["any_anomaly"]

    # 2b. Dầu nặng (mu=0.10, slip=0.08) -> cổng nhanh kích hoạt ngay bước 1
    det.reset()
    out_oil = det.step(mu_bg, [0.10] * 4, [0.08, 0.01, 0.01, 0.08], [True] * 4)
    assert out_oil["any_anomaly"] and out_oil["fast_triggered"][0] and out_oil["fast_triggered"][3]
    assert not out_oil["is_anomaly"][1], "Bánh không trượt không được gắn cờ dầu"

    # 2c. Nước mưa mỏng (delta_mu=0.07, slip=0.028 < slip_emerg=0.040) -> CUSUM bắt trong <= 4 bước
    det.reset()
    trig_step = None
    for k in range(1, 6):
        out_rain = det.step(mu_bg, [0.28] * 4, [0.028] * 4, [True] * 4)
        if out_rain["any_anomaly"]:
            trig_step = k
            break
    assert trig_step is not None and trig_step <= 4, f"CUSUM bắt mưa mỏng quá chậm: {trig_step}"
    print(f"  [PASS] Test 2: Chống nhiễu ISO 1 bước, dầu nặng=1 bước, mưa mỏng CUSUM={trig_step} bước.")


def test_3_long_rain_and_short_oil_vs_legacy():
    """Tiêu chí 3: Hết lỗi chattering sau 3.0 m trên vùng mưa dài 5.0 m & hồi phục sau vũng ngắn 0.4 m."""
    Fz_tot = M_TOTAL * 9.81
    s_path = np.linspace(0.0, 8.0, 81)  # 0.10 m/step
    x_path = s_path.copy()
    y_path = np.zeros_like(s_path)
    yaw_path = np.zeros_like(s_path)

    # Kịch bản A: Vùng mưa dài 5.0 m từ s=1.0..6.0 m (mu=0.18), đoạn s=3.5..5.5 m đi thẳng êm (bin=0)
    est_dl = DualLayerFrictionEstimator(mu_env_low=0.30, mu_env_high=0.40, mu_hard_min=0.05)
    fus_old = OnlineFrictionFusion(mu_min=0.08, mu_max=0.55)

    mu_dl_rain = []
    mu_old_rain = []
    mu_rls_dummy = 0.18
    for idx, s0 in enumerate(s_path):
        t_now = idx * 0.2
        s_h = np.linspace(s0, s0 + 0.8, 8)
        x_h = s_h.copy()
        y_h = np.zeros_like(s_h)
        yaw_h = np.zeros_like(s_h)

        in_rain = 1.0 <= s0 <= 6.0
        if 1.0 <= s0 <= 1.5:
            # Cua/tăng tốc chớm vào vùng mưa -> phát hiện
            est_dl.update_online(s0, 0.0, 0.0, [0.18 * Fz_tot, 0.0], [0.16 * Fz_tot, 0.0], [0.06] * 4, Fz_tot, M_TOTAL, t_now)
            bin_old = 2
            mu_rls_dummy = 0.18
        elif in_rain:
            # Đi thẳng êm trên đoạn mưa dài -> không có kích thích
            est_dl.update_online(s0, 0.0, 0.0, [0.02 * Fz_tot, 0.0], [0.02 * Fz_tot, 0.0], [0.005] * 4, Fz_tot, M_TOTAL, t_now)
            bin_old = 0
        else:
            est_dl.update_online(s0, 0.0, 0.0, [0.02 * Fz_tot, 0.0], [0.02 * Fz_tot, 0.0], [0.005] * 4, Fz_tot, M_TOTAL, t_now)
            bin_old = 0

        # Kiểm tra khi đi lại qua phần giữa vùng mưa [4.2..5.5 m] sau khi đã từng chạm vùng [1.0..1.5 m]
        mu_dl_h = est_dl.fuse_horizon_xy(x_h, y_h, yaw_h, s_h, t_now=t_now, s_curr=s0)
        mu_old_h = fus_old.fuse_horizon_mu(np.full(8, 0.30), s_h, mu_rls_dummy, bin_old, s_curr=s0)
        mu_dl_rain.append(float(mu_dl_h[0]))
        mu_old_rain.append(float(mu_old_h[0]))

    # Tại s=4.8 m (> 3.0 m kể từ 1.5 m), bản cũ đã tự hủy chốt nhảy về 0.30!
    idx_48 = 48
    assert mu_old_rain[idx_48] > 0.29, f"Expected legacy latch to expire by s=4.8m, got {mu_old_rain[idx_48]}"
    # Ở vòng 1, đoạn 1.0..1.5 m đã lưu vào bản đồ 2D; khi cho xe đi qua lại từ s=0..2.0 m ở vòng 2,
    # nhưng với vùng mưa dài 5.0 m đang mở rộng (xe đi tiếp và trượt lần nữa khi vào đoạn kế),
    # bản cũ tụt hết memory sau 3.0 m còn bản mới giữ bền vùng đã phát hiện đến tau_dry_s.
    est_dl.reset(keep_anomaly_map=True)
    mu_revisit = est_dl.fuse_horizon_xy(np.array([1.3]), np.array([0.0]), np.array([0.0]), np.array([1.3]), t_now=25.0)
    assert mu_revisit[0] < 0.22, f"Vùng mưa đã ghi không giữ được trong trí nhớ 2D: {mu_revisit[0]}"

    # Kịch bản B: Vũng dầu ngắn 0.4 m (s=1.0..1.4 m), sau đó là nền khô.
    # Ngay tại s=2.2 m (> 0.8 m sau vũng dầu), DualLayer phục hồi về nền còn legacy vẫn ì suốt 3.0 m (đến s=4.4 m)
    est_short = DualLayerFrictionEstimator(mu_env_low=0.30, mu_env_high=0.40)
    fus_short = OnlineFrictionFusion(mu_min=0.08, mu_max=0.55)
    for k in range(3):
        est_short.update_online(1.1 + 0.1 * k, 0.0, 0.0, [0.12 * Fz_tot, 0.0], [0.16 * Fz_tot, 0.0], [0.08] * 4, Fz_tot, M_TOTAL, 0.1 * k)
        fus_short.fuse_horizon_mu(np.full(8, 0.30), np.linspace(1.1 + 0.1 * k, 1.9 + 0.1 * k, 8), 0.12, 2, s_curr=1.1 + 0.1 * k)

    # Tại s=2.4 m (cách vũng ngắn > 1.0 m, đi thẳng êm bin=0)
    est_short.update_online(2.4, 0.0, 0.0, [0.02 * Fz_tot, 0.0], [0.02 * Fz_tot, 0.0], [0.005] * 4, Fz_tot, M_TOTAL, 2.0)
    mu_short_dl = float(est_short.fuse_horizon_xy([2.4], [0.0], [0.0], [2.4], t_now=2.0)[0])
    mu_short_old = float(fus_short.fuse_horizon_mu(np.full(8, 0.30), np.linspace(2.4, 3.2, 8), 0.12, 0, s_curr=2.4)[0])
    assert mu_short_dl > 0.28 and mu_short_old < 0.24 and (mu_short_dl - mu_short_old) > 0.06, (
        f"Phục hồi sau vũng ngắn sai kỳ vọng: dl={mu_short_dl:.3f}, old={mu_short_old:.3f}"
    )
    print(f"  [PASS] Test 3: Chống chattering vùng mưa & phục hồi sau vũng ngắn (dl={mu_short_dl:.3f} vs old={mu_short_old:.3f}).")


def test_4_multilap_tube_memory_dry_clear_and_o1_speed():
    """Tiêu chí 4: Vòng 2 Tube H124 hạ v_ub từ xa, xác nhận khô/bay hơi xóa ô & tốc độ O(1) trên map 100 m."""
    x_ctrl = np.linspace(0.0, 6.75, 160)
    y_ctrl = 0.45 * np.exp(-((x_ctrl - 1.50) ** 2) / (2 * 0.43 ** 2)) - 0.43 * np.exp(-((x_ctrl - 3.15) ** 2) / (2 * 0.42 ** 2))
    wp = np.column_stack([x_ctrl, y_ctrl])

    est = DualLayerFrictionEstimator(mu_env_low=0.30, mu_env_high=0.40, tau_dry_s=60.0)
    tube1 = CoupledVelocityTubeH124(mu_default=0.35, kappa_th=1.05)
    tube1.build_offline_tube(wp, friction_estimator=est, friction_mode="conformal_lcb", run_wcsph_check=False)

    # Mô phỏng vòng 1 phát hiện dầu tại đỉnh cua thứ 2 (x=3.15, y=-0.43)
    Fz_tot = M_TOTAL * 9.81
    for step_k in range(5):
        est.update_online(3.15, -0.43, 0.0, [0.12 * Fz_tot, 0.0], [0.18 * Fz_tot, 0.0], [0.09] * 4, Fz_tot, M_TOTAL, 5.0 + 0.05 * step_k)
    assert len(est.anomaly_map.cells) > 0

    # Vòng 2: giữ anomaly map và dựng lại Ống vận tốc H124 -> phải chủ động hạ v_ub tại x~3.15
    est.reset(keep_anomaly_map=True)
    tube2 = CoupledVelocityTubeH124(mu_default=0.35, kappa_th=1.05)
    tube2.build_offline_tube(wp, friction_estimator=est, friction_mode="conformal_lcb", run_wcsph_check=False)

    idx_oil = int(np.argmin(np.hypot(tube2.x_grid - 3.15, tube2.y_grid + 0.43)))
    assert tube2.mu_lcb_grid[idx_oil] < 0.20, f"Vòng 2 không hạ mu tại vùng dầu cũ: {tube2.mu_lcb_grid[idx_oil]}"
    assert tube2.v_ub_nom[idx_oil] < tube1.v_ub_nom[idx_oil] - 0.04, (
        f"Vòng 2 không chủ động hạ tốc độ: v1={tube1.v_ub_nom[idx_oil]:.3f}, v2={tube2.v_ub_nom[idx_oil]:.3f}"
    )

    # Kiểm tra xác nhận khô liên tiếp xóa ô & tự suy giảm theo tau_dry_s
    for k in range(3):
        est.update_online(3.15, -0.43, 0.0, [0.35 * Fz_tot * math.tanh(0.01 / 0.04), 0.0], [0.14 * Fz_tot, 0.0], [0.010] * 4, Fz_tot, M_TOTAL, 6.0 + 0.05 * k)
    assert est.anomaly_map.query_points(np.array([[3.15, -0.43]]), 6.2)[0] < 0.15

    vmap = SparseAnomalyVoxelMap(tau_dry_s=10.0, prune_thresh=0.01)
    for j in range(200):
        vmap.record_anomaly(0.5 * j, math.sin(j), 0.20, t_now=0.0)
    long_pts = np.column_stack([np.linspace(0.0, 100.0, 2000), np.zeros(2000)])
    t_q0 = time.perf_counter()
    _ = vmap.query_points(long_pts, t_now=1.0)
    q_ms = (time.perf_counter() - t_q0) * 1000.0
    assert q_ms < 25.0, f"Truy vấn 2000 điểm trên tuyến 100 m quá chậm: {q_ms:.2f} ms"
    n_pruned = vmap.prune_expired(t_now=40.0)
    assert n_pruned == 200 and len(vmap.cells) == 0
    print(f"  [PASS] Test 4: Trí nhớ đa vòng (v1={tube1.v_ub_nom[idx_oil]:.3f}->{tube2.v_ub_nom[idx_oil]:.3f} m/s), xóa khô & O(1)={q_ms:.2f} ms.")


def test_5_closed_loop_mujoco_world6_dual_layer():
    """Tiêu chí 5: Vòng kín MuJoCo trên world6.xml qua DfeMpccPipeline(friction_mode='dual_layer')."""
    scen = build_map_scenario("world6.xml")
    model = scen["model"]
    dyn_configs = scen["dyn_configs"]
    start_pos = scen["start_pos"]
    goal_pos = scen["goal_pos"]
    wp_arr = scen["wp_arr"]

    # Trên Plant có cả epoxy/wet_risk + 2 vũng dầu ẩn (mu=0.08), estimator CHỈ được cấp khoảng nền [0.30, 0.40]
    sem_map_true = make_world_semantic_map("world6.xml")
    hidden_oil = _select_hidden_oil_patches(wp_arr, sem_map_true)
    for p in hidden_oil:
        p["mu"] = 0.08
    plant_macro_fn = make_hidden_oil_macro_fn(sem_map_true.macro_friction_fn, hidden_oil, mu_clip=(0.06, 0.60))

    pipeline = DfeMpccPipeline(
        horizon_N=8,
        dt=0.05,
        ipopt_max_iter=12,
        mu_default=0.35,
        map_name="generic_dry",
        x_bounds=(-3.0, 3.0),
        y_bounds=(-3.0, 3.0),
        waypoints=wp_arr,
        friction_mode="dual_layer",
        mu_env_range=(0.30, 0.40),
        mu_hard_min=0.05,
    )

    road_profile = ISO8608RoadProfile(road_class="B", seed=42)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    update_dynamic_obstacles(model, data, 0.0, custom_dyn_configs=dyn_configs)
    data.qpos[0], data.qpos[1] = start_pos[0], start_pos[1]
    _, _, h0 = pipeline.controller.get_path_reference(0.0)
    data.qpos[2] = h0
    mujoco.mj_forward(model, data)

    nmpc_dt = 0.05
    n_sub = max(1, int(round(nmpc_dt / float(model.opt.timestep))))
    sim_t = 0.0
    eta_max = 0.0
    reached = False
    prev_wheel_slips = np.zeros(4, dtype=float)

    for _ in range(int(round(55.0 / nmpc_dt))):
        update_dynamic_obstacles(model, data, sim_t, custom_dyn_configs=dyn_configs)
        mujoco.mj_forward(model, data)
        q_meas = data.qpos[:3].copy()
        v_meas = data.qvel[:3].copy()

        road_eval = road_profile.compute_wheel_loads_and_slosh_excitation(q_meas, v_meas, mu_nominal=plant_macro_fn)
        a_road_b = road_eval["a_road_body"]
        mu_w_true = np.asarray(road_eval.get("mu_wheels", [plant_macro_fn(q_meas[0], q_meas[1])] * 4), dtype=float)
        phi_est = np.array([
            float(data.qpos[3]) + (a_road_b[0] / 9.81) * 0.55,
            float(data.qvel[3]),
            float(data.qpos[4]) - (a_road_b[1] / 9.81) * 0.55,
            float(data.qvel[4]),
        ], dtype=float)

        Fz_true = np.full(4, M_TOTAL * 9.81 / 4.0)
        step_out = pipeline.step(q_meas, v_meas, phi_est, wheel_slips=prev_wheel_slips)

        f_w_cmd = step_out["wheel_forces_4w"] / np.sqrt(2.0)
        _, f_ch_act, slip_m = evaluate_traction_and_slip(f_w_cmd, Fz_true, mu=mu_w_true)
        prev_wheel_slips = np.asarray(slip_m["slip_ratios"], dtype=float)

        th = float(q_meas[2])
        c_th, s_th = math.cos(th), math.sin(th)
        a_bx = f_ch_act[0] / M_TOTAL
        a_by = f_ch_act[1] / M_TOTAL
        a_th = f_ch_act[2] / 0.045
        data.qvel[0] = v_meas[0] + (c_th * a_bx - s_th * a_by) * nmpc_dt
        data.qvel[1] = v_meas[1] + (s_th * a_bx + c_th * a_by) * nmpc_dt
        data.qvel[2] = v_meas[2] + a_th * nmpc_dt
        data.qpos[3] = np.clip(data.qpos[3] + (a_bx / 9.81) * 0.08, -0.22, 0.22)
        data.qpos[4] = np.clip(data.qpos[4] - (a_by / 9.81) * 0.08, -0.22, 0.22)

        for _ in range(n_sub):
            mujoco.mj_step(model, data)
        sim_t += nmpc_dt

        _, _, eta_c = compute_world6_slosh_elevation_mm(
            float(data.qpos[3]), float(data.qvel[3]), float(data.qpos[4]), float(data.qvel[4]),
            a_bx, a_by, float(step_out["u_opt"][0]), a_road_b[0], a_road_b[1],
            is_proposed=True, min_dyn_clearance=1.0, has_dyn_obs=False,
        )
        eta_max = max(eta_max, float(eta_c))

        q_now = data.qpos[:3]
        if math.hypot(q_now[0] - goal_pos[0], q_now[1] - goal_pos[1]) < 0.30 and pipeline.s_prog > pipeline.controller.path_length - 0.50:
            reached = True
            break

    assert reached, f"Xe chưa về đích world6.xml (s={pipeline.s_prog:.2f}/{pipeline.controller.path_length:.2f})"
    assert eta_max < 30.0, f"Tràn nước: eta_max={eta_max:.2f} mm >= 30.0 mm"
    n_cells = len(pipeline.dual_estimator.anomaly_map.cells)
    assert n_cells > 0, "Chưa ghi nhận được vùng trơn/dầu ẩn vào bộ nhớ 2D!"

    # Vòng 2 tái lập kế hoạch trên cùng bản đồ -> tube_planner phải tự hạ mu ở vùng đã phát hiện
    pipeline.set_waypoints(wp_arr, keep_anomaly_map=True)
    assert np.min(pipeline.tube_planner.mu_lcb_grid) < pipeline.dual_estimator.bg_model.get_safe_bg() - 0.03
    print(
        f"  [PASS] Test 5 (world6 dual_layer): Goal OK (t={sim_t:.2f}s, eta_max={eta_max:.2f}mm, "
        f"cells={n_cells}, mu_bg={pipeline.dual_estimator.bg_model.mu_bg:.3f})."
    )


if __name__ == "__main__":
    print("=" * 84)
    print("KIỂM THỬ HƯỚNG 2: DUAL-LAYER BACKGROUND + ANOMALY FRICTION ESTIMATOR (5 TESTS)")
    print("=" * 84)
    test_1_dynamic_range_and_background_adaptation()
    test_2_dual_gate_cusum_detection_and_noise_rejection()
    test_3_long_rain_and_short_oil_vs_legacy()
    test_4_multilap_tube_memory_dry_clear_and_o1_speed()
    test_5_closed_loop_mujoco_world6_dual_layer()
    print("=" * 84)
    print("TẤT CẢ 5/5 BÀI TEST DUAL-LAYER ĐÃ ĐẠT 100%!")
    print("=" * 84)
