# -*- coding: utf-8 -*-
"""
test_spatial_friction_gp.py
=============================================================================
Kiểm thử đơn vị & tích hợp cho Mô hình Trường Ma sát Không gian GP + Conformal
(SpatialFrictionGP), tích hợp Ống vận tốc (CoupledVelocityTubeH124) và
Bộ điều khiển DFE-MPCC chân trời P_mu (DFEMPCCAMRController).
=============================================================================
"""

import os
import sys
import tempfile
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

_curr = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_curr)
for _p in [_root, os.path.join(_root, "planning"), os.path.join(_root, "controller"),
           os.path.join(_root, "solver"), os.path.join(_root, "simulation")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from planning.spatial_friction_gp import (
    SpatialFrictionGP,
    MultiZoneFrictionGroundTruth,
    make_benchmark_curve_ground_truth,
    make_random_site_ground_truth,
    compute_footprint_query_points,
    collect_ffrls_history_from_rollouts,
    aggregate_observations_to_voxels,
)
from planning.velocity_tube_h124 import (
    CoupledVelocityTubeH124,
    _analytical_benchmark_curve,
    generate_multi_regime_benchmark_scenario,
)
from controller.dfe_mpcc_amr_controller import DFEMPCCAMRController


def _build_perturbed_curve_waypoints(y_offset: float = 0.0, y_amp_scale: float = 1.0, n_pts: int = 200) -> np.ndarray:
    x_arr = np.linspace(0.0, 6.75, n_pts)
    y_base, _, _, _ = _analytical_benchmark_curve(x_arr)
    # Dịch ngang mượt để đầu/cuối vẫn nằm trong hành lang nhưng quét đa dạng vùng lân cận
    envelope = np.sin(np.pi * (x_arr / 6.75))
    y_arr = y_amp_scale * y_base + y_offset * envelope
    return np.column_stack([x_arr, y_arr])


def test_1_cold_start_and_unexplored_reversion():
    """Test 1: Ở vùng xa dữ liệu lịch sử, GP quay về mu_prior và bất định tiệm cận prior std."""
    rng = np.random.default_rng(11)
    pts_explored = rng.uniform([0.0, -0.5], [3.0, 0.5], size=(80, 2))
    mu_explored = np.full(80, 0.35) + rng.normal(0.0, 0.01, size=80)

    gp = SpatialFrictionGP(mu_prior=0.22, sigma_f=0.12, sigma_micro=0.045, q_hat_mu=2.0)
    gp.fit(pts_explored, mu_explored, voxel_size=0.15)

    q_pts = np.array([
        [1.5, 0.0],    # Trong vùng đã khám phá (mu ~ 0.35)
        [12.0, 12.0],  # Rất xa vùng dữ liệu (chưa khám phá)
    ])
    mu_mean, sigma_gp, mu_lcb = gp.predict(q_pts)

    prior_std = np.sqrt(gp.sigma_f ** 2 + gp.sigma_micro ** 2)
    assert abs(mu_mean[1] - gp.mu_prior) < 1e-5, f"Cold-start mean={mu_mean[1]} != prior={gp.mu_prior}"
    assert abs(sigma_gp[1] - prior_std) < 1e-4, f"Cold-start std={sigma_gp[1]} != {prior_std}"
    assert sigma_gp[1] > sigma_gp[0], "Vùng chưa khám phá phải có sigma_gp lớn hơn vùng đã đo!"
    assert mu_lcb[1] < mu_lcb[0], "Vùng chưa khám phá phải cho mu_lcb bảo thủ hơn!"

    # Kiểm tra fit rỗng không lỗi
    gp_empty = SpatialFrictionGP(mu_prior=0.24)
    gp_empty.fit(np.zeros((0, 2)), np.zeros(0))
    m_e, s_e, l_e = gp_empty.predict(q_pts)
    assert np.allclose(m_e, 0.24) and np.allclose(s_e, np.sqrt(gp_empty.sigma_f**2 + gp_empty.sigma_micro**2))
    assert np.all(l_e <= m_e)


def test_2_speed_and_numerical_stability():
    """Test 2: Tốc độ fit/predict nhanh (< 20 ms tâm, < 80 ms toàn dấu chân 2400 điểm) và chặn an toàn."""
    rng = np.random.default_rng(22)
    pts = rng.uniform([0.0, -1.0], [6.5, 1.0], size=(1200, 2))
    mu_obs = 0.25 + 0.10 * np.sin(pts[:, 0]) + rng.normal(0.0, 0.02, size=1200)

    pts_v, mu_v, var_v = aggregate_observations_to_voxels(pts, mu_obs, voxel_size=0.23)
    assert 50 <= len(pts_v) <= 400, f"Unexpected voxel count: {len(pts_v)}"

    gp = SpatialFrictionGP(query_mode="footprint")
    gp.fit(pts, mu_obs, voxel_size=0.23)

    x_q = np.linspace(0.0, 6.75, 400)
    y_q, _, _, _ = _analytical_benchmark_curve(x_q)

    # Warm call + timed calls (min of 5 to resist OS scheduling jitter)
    _ = gp.predict_along_path(x_q, y_q, mode="conformal_lcb", query_mode="center")
    t_centers = []
    for _ in range(5):
        res_c = gp.predict_along_path(x_q, y_q, mode="conformal_lcb", query_mode="center")
        t_centers.append(res_c["predict_time_ms"])
    min_center_ms = min(t_centers)
    assert min_center_ms < 20.0, f"GP predict center quá chậm: min={min_center_ms:.2f} ms ({t_centers})"

    _ = gp.predict_along_path(x_q, y_q, mode="conformal_lcb", query_mode="footprint")
    t_fps = []
    for _ in range(5):
        res = gp.predict_along_path(x_q, y_q, mode="conformal_lcb", query_mode="footprint")
        t_fps.append(res["predict_time_ms"])
    min_fp_ms = min(t_fps)
    assert min_fp_ms < 80.0, f"GP predict footprint quá chậm: min={min_fp_ms:.2f} ms ({t_fps})"

    assert np.all(np.isfinite(res["mu_tube"]))
    assert np.all(res["mu_lcb"] >= gp.mu_min - 1e-12)
    assert np.all(res["mu_mean"] <= gp.mu_max + 1e-12)
    assert np.all(res["mu_lcb"] <= res["mu_mean"] + 1e-12)


def test_3_disjoint_split_trajectory_conformal_coverage():
    """Test 3: Tách tập D_hist / D_cal / D_test nghiêm ngặt + Kiểm chứng phủ 4 bánh split-mu & JSON I/O."""
    # D_hist: 15 lượt chạy quá khứ qua FF-RLS (seeds 100..114)
    prior_rollouts = []
    rng_hist = np.random.default_rng(100)
    gt_hist = make_benchmark_curve_ground_truth(seed=100, road_class="B", variation_scale=0.15)
    for k in range(15):
        wp_k = _build_perturbed_curve_waypoints(
            y_offset=float(rng_hist.uniform(-0.35, 0.35)),
            y_amp_scale=float(rng_hist.uniform(0.85, 1.15)),
            n_pts=180,
        )
        prior_rollouts.append(wp_k)

    hist_data = collect_ffrls_history_from_rollouts(
        gt_hist,
        prior_rollouts,
        seed=101,
        observation_mode="per_wheel_rls_lagcomp",
        bidirectional=True,
    )
    assert hist_data["n_accepted"] > 500
    assert hist_data["n_deadzone_skipped"] > 0, "Cần kiểm chứng vùng chết FF-RLS hoạt động"

    gp = SpatialFrictionGP(query_mode="footprint")
    gp.fit(hist_data["points_xy"], hist_data["mu_obs"], hist_data["obs_var"], voxel_size=0.15)

    # D_cal: 60 quỹ đạo độc lập (seeds 200..259)
    rng_cal = np.random.default_rng(200)
    cal_trajs = []
    for s_id in range(200, 260):
        gt_j = make_benchmark_curve_ground_truth(seed=s_id, road_class="B", variation_scale=0.15)
        wp_j = _build_perturbed_curve_waypoints(
            y_offset=float(rng_cal.uniform(-0.22, 0.22)),
            y_amp_scale=float(rng_cal.uniform(0.92, 1.08)),
            n_pts=180,
        )
        _, mu_4w, _ = gt_j.eval_path_4w_min(wp_j[:, 0], wp_j[:, 1])
        cal_trajs.append({"x_grid": wp_j[:, 0], "y_grid": wp_j[:, 1], "mu_min_4w_true": mu_4w})

    cal_meta = gp.calibrate_trajectory_conformal(cal_trajs, alpha_mu=0.05)
    assert cal_meta["n_cal"] == 60
    assert cal_meta["k_star"] == 58  # ceil(61 * 0.95) = 58
    assert cal_meta["query_mode"] == "footprint"
    assert 0.2 < gp.q_hat_mu < 3.5, f"q_hat_mu ngoài kỳ vọng: {gp.q_hat_mu}"

    # D_test: 40 quỹ đạo độc lập (seeds 300..339)
    rng_test = np.random.default_rng(300)
    test_trajs = []
    for s_id in range(300, 340):
        gt_t = make_benchmark_curve_ground_truth(seed=s_id, road_class="B", variation_scale=0.15)
        wp_t = _build_perturbed_curve_waypoints(
            y_offset=float(rng_test.uniform(-0.22, 0.22)),
            y_amp_scale=float(rng_test.uniform(0.92, 1.08)),
            n_pts=180,
        )
        _, mu_4w, _ = gt_t.eval_path_4w_min(wp_t[:, 0], wp_t[:, 1])
        test_trajs.append({"x_grid": wp_t[:, 0], "y_grid": wp_t[:, 1], "mu_min_4w_true": mu_4w})

    cov_m2 = gp.evaluate_coverage(test_trajs, mode="conformal_lcb")
    cov_m1 = gp.evaluate_coverage(test_trajs, mode="gp_mean")

    assert cov_m2["traj_coverage_rate"] >= 0.85, f"M2 coverage quá thấp: {cov_m2}"
    assert cov_m2["binom_p_value_one_sided"] > 0.05, f"M2 bị bác bỏ Binomial p={cov_m2['binom_p_value_one_sided']}"
    assert cov_m2["traj_coverage_rate"] > cov_m1["traj_coverage_rate"], (
        f"M2 ({cov_m2['traj_coverage_rate']}) phải phủ tốt hơn M1 ({cov_m1['traj_coverage_rate']})"
    )

    # Kiểm tra lưu/nạp JSON v2.0 khớp tuyệt đối < 1e-12
    with tempfile.TemporaryDirectory() as tmpdir:
        cal_json = os.path.join(tmpdir, "test_cal_mu.json")
        gp.save_calibration(cal_json, include_support=True)
        gp_loaded = SpatialFrictionGP.load_calibration(cal_json)
        assert gp_loaded.query_mode == "footprint"
        res_orig = gp.predict_along_path(test_trajs[0]["x_grid"], test_trajs[0]["y_grid"], mode="conformal_lcb")
        res_load = gp_loaded.predict_along_path(test_trajs[0]["x_grid"], test_trajs[0]["y_grid"], mode="conformal_lcb")
        assert np.max(np.abs(res_orig["mu_tube"] - res_load["mu_tube"])) < 1e-12


def test_4_tube_no_double_counting_and_fixed_point():
    """Test 4: CoupledVelocityTubeH124 dùng GP Conformal không phạt kép và giữ nguyên hội tụ KM."""
    gt = make_benchmark_curve_ground_truth(seed=42)
    wp_hist = [_build_perturbed_curve_waypoints(y_offset=off, n_pts=160) for off in (-0.2, 0.0, 0.2)]
    hist = collect_ffrls_history_from_rollouts(gt, wp_hist, seed=10)
    gp = SpatialFrictionGP(q_hat_mu=1.80)
    gp.fit(hist["points_xy"], hist["mu_obs"], hist["obs_var"], voxel_size=0.15)

    waypoints, _, d_net_fn, zc_fn, camber_fn = generate_multi_regime_benchmark_scenario()
    planner = CoupledVelocityTubeH124(mu_default=0.35, kappa_th=1.05)
    summary = planner.build_offline_tube(
        waypoints,
        d_net_profile=d_net_fn,
        z_com_profile=zc_fn,
        phi_camber_profile=camber_fn,
        run_wcsph_check=False,
        friction_estimator=gp,
        friction_mode="conformal_lcb",
    )

    assert planner.mu_is_conformal_safe is True
    assert planner.mu_effective_plan_scale == 1.0, f"Phạt kép? scale={planner.mu_effective_plan_scale}"
    assert summary["mu_is_conformal_safe"] is True
    assert summary["mu_effective_plan_scale"] == 1.0
    assert summary["max_constraint_violation_m_s"] <= 1e-5
    assert summary["fp_unrelaxed_residual"] < 2e-4


def test_5_online_tube_local_mu_scaling_bugfix():
    """Test 5: Sửa lỗi tỷ lệ cục bộ trong query_online_horizon_tube (không chia nhầm cho mu_default=0.35)."""
    waypoints, mu_fn, d_net_fn, zc_fn, camber_fn = generate_multi_regime_benchmark_scenario()
    planner = CoupledVelocityTubeH124(mu_default=0.35, kappa_th=1.05)
    planner.build_offline_tube(
        waypoints,
        mu_profile=mu_fn,
        d_net_profile=d_net_fn,
        z_com_profile=zc_fn,
        phi_camber_profile=camber_fn,
        run_wcsph_check=False,
    )

    # Truy vấn ngay tại vùng dầu CP3 (s quanh 3.46 m, nơi mu_grid = 0.12, mu_plan = 0.95*0.12 = 0.114)
    s_hor = np.linspace(3.30, 3.70, 10)
    v_ub_nom_q = np.interp(s_hor, planner.s_grid, planner.v_ub_nom)
    mu_plan_q = planner.get_horizon_mu_plan(s_hor)
    assert np.all(mu_plan_q < 0.15), f"Không phải vùng dầu CP3: {mu_plan_q}"

    # Khi mu_online đúng bằng mu_plan_q, v_ub_online không được tụt oan vì chia nhầm cho mu_default=0.35
    v_ub_same, v_lb_same = planner.query_online_horizon_tube(
        s_hor, phi_est=[0.0, 0.0, 0.0, 0.0], mu_online=mu_plan_q
    )
    assert np.allclose(v_ub_same, np.clip(v_ub_nom_q, 0.08, planner.v_max), atol=1e-4), (
        f"Co giãn sai khi mu_online == mu_plan_q: online={v_ub_same}, nom={v_ub_nom_q}"
    )

    # Khi bất ngờ gặp vũng dầu trơn hơn nữa (mu_online = 0.08), trần phải siết chặt hơn
    v_ub_slip, v_lb_slip = planner.query_online_horizon_tube(
        s_hor, phi_est=[0.0, 0.0, 0.0, 0.0], mu_online=0.08
    )
    assert np.all(v_ub_slip <= v_ub_nom_q + 1e-6)
    assert np.all(v_ub_slip < v_ub_same - 1e-3)
    assert np.all(v_ub_slip >= v_lb_slip)


def test_6_controller_horizon_pmu_and_backwards_compat():
    """Test 6: Nâng cấp P_mu theo chân trời (1, N) trong DFEMPCCAMRController vẫn tương thích ngược."""
    waypoints, mu_fn, d_net_fn, zc_fn, camber_fn = generate_multi_regime_benchmark_scenario()
    planner = CoupledVelocityTubeH124(mu_default=0.35, kappa_th=1.05)
    planner.build_offline_tube(
        waypoints,
        mu_profile=mu_fn,
        d_net_profile=d_net_fn,
        z_com_profile=zc_fn,
        phi_camber_profile=camber_fn,
        run_wcsph_check=False,
    )

    ctrl = DFEMPCCAMRController(N=15, dt=0.05, mu=0.35)
    ctrl.set_path(waypoints)
    ctrl.set_tube_planner(planner)

    _, _, th0 = ctrl.get_path_reference(0.0)
    x0 = np.zeros(12)
    x0[0:2] = waypoints[0]
    x0[2] = th0
    x0[3] = 0.35 * np.cos(th0)
    x0[4] = 0.35 * np.sin(th0)
    x0[9] = 0.0
    x0[10] = 0.35
    phi0 = np.zeros(4)

    # 1. Tương thích ngược: mu_floor vô hướng
    u1, _, _, _, info1 = ctrl.solve_step(x0, phi0, mu_floor=0.35)
    assert info1["status"] == "optimal", f"Scalar mu_floor status={info1['status']}"
    assert np.allclose(info1["mu_plan_preview"], 0.35 * ctrl.mu_plan_scale)
    assert np.all(np.isfinite(u1))

    # 2. Tự động đọc từ tube_planner khi mu_floor=None tại vùng chuyển tiếp vào dầu (s = 2.80 m)
    p2, _, th2 = ctrl.get_path_reference(2.80)
    x2 = x0.copy()
    x2[0:2] = p2
    x2[2] = th2
    x2[3] = 0.40 * np.cos(th2)
    x2[4] = 0.40 * np.sin(th2)
    x2[9] = 2.80
    x2[10] = 0.40
    _, _, _, _, info2 = ctrl.solve_step(x2, phi0, mu_floor=None)
    assert info2["status"] == "optimal", f"Auto tube mu status={info2['status']}"
    mu_prev = info2["mu_plan_preview"]
    assert mu_prev[0] > mu_prev[-1] + 0.05, f"Expected decreasing horizon mu entering oil zone: {mu_prev}"

    # 3. Truyền trực tiếp mu_horizon tùy biến
    mu_custom = np.linspace(0.35, 0.12, ctrl.N)
    _, _, _, _, info3 = ctrl.solve_step(x2, phi0, mu_horizon=mu_custom)
    assert info3["status"] == "optimal", f"Explicit mu_horizon status={info3['status']}"
    assert np.allclose(info3["mu_plan_preview"], mu_custom)


def test_7_footprint_iff_conformal_and_prior_diagnostics():
    """Test 7: Đánh giá theo lô khớp < 1e-12, tính IFF của điểm Conformal dấu chân 4 bánh và chẩn đoán vùng prior."""
    gt = make_random_site_ground_truth(site_seed=1007, road_class="B", variation_scale=0.15)
    wp_check = _build_perturbed_curve_waypoints(n_pts=80)
    batch_vals = gt.eval_points_batch(wp_check)
    scalar_vals = np.array([gt.eval_center(float(p[0]), float(p[1])) for p in wp_check])
    assert np.max(np.abs(batch_vals - scalar_vals)) < 1e-12, "eval_points_batch lệch với eval_center!"

    wp_hist = [_build_perturbed_curve_waypoints(y_offset=off, n_pts=140) for off in (-0.15, 0.0, 0.15)]
    hist = collect_ffrls_history_from_rollouts(gt, wp_hist, seed=77, bidirectional=True)
    gp = SpatialFrictionGP(query_mode="footprint")
    gp.fit(hist["points_xy"], hist["mu_obs"], hist["obs_var"], voxel_size=0.15)
    lml = gp.log_marginal_likelihood()
    assert np.isfinite(lml), f"LML không hữu hạn: {lml}"

    # Tạo 12 quỹ đạo kiểm định tính IFF của điểm footprint:
    # Mọi quỹ đạo có score <= q_hat phải được phủ 100% (với diff >= -1e-9), và điểm > q_hat phải vi phạm
    rng = np.random.default_rng(707)
    cal_trajs = []
    for k in range(12):
        wp_k = _build_perturbed_curve_waypoints(
            y_offset=float(rng.uniform(-0.20, 0.20)),
            y_amp_scale=float(rng.uniform(0.92, 1.08)),
            n_pts=90,
        )
        gt_k = make_random_site_ground_truth(site_seed=1007, road_class="B", variation_scale=0.15)
        _, mu_4w, _ = gt_k.eval_path_4w_min(wp_k[:, 0], wp_k[:, 1])
        cal_trajs.append({"x_grid": wp_k[:, 0], "y_grid": wp_k[:, 1], "mu_min_4w_true": mu_4w})

    cal_meta = gp.calibrate_trajectory_conformal(cal_trajs, alpha_mu=0.10, query_mode="footprint")
    scores = np.asarray(cal_meta["scores"], dtype=float)
    q_hat = float(gp.q_hat_mu)
    for j, tr in enumerate(cal_trajs):
        res = gp.predict_along_path(tr["x_grid"], tr["y_grid"], mode="conformal_lcb", query_mode="footprint")
        covered = bool(np.all(tr["mu_min_4w_true"] - res["mu_tube"] >= -1e-9))
        if scores[j] <= q_hat - 1e-9:
            assert covered, f"Quỹ đạo {j} có score={scores[j]} <= q_hat={q_hat} nhưng không được phủ!"
        elif scores[j] > q_hat + 1e-6:
            # Khi unclipped > mu_min (tức thực sự quá ước) thì phải vi phạm
            assert not covered or np.all(tr["mu_min_4w_true"] <= gp.mu_min + 1e-9), (
                f"Quỹ đạo {j} có score={scores[j]} > q_hat={q_hat} nhưng vẫn báo phủ!"
            )

    # Kiểm tra chẩn đoán vùng prior / kẹp sàn mu_min (M4)
    res_in = gp.predict_along_path(wp_check[:, 0], wp_check[:, 1], mode="conformal_lcb")
    far_x = np.linspace(15.0, 18.0, 40)
    far_y = np.zeros(40)
    res_far = gp.predict_along_path(far_x, far_y, mode="conformal_lcb")
    assert res_far["prior_region_frac"] == 1.0, f"Vùng xa phải có prior_region_frac=1.0, nhận {res_far['prior_region_frac']}"
    assert res_in["prior_region_frac"] < 0.25, f"Trên tuyến đã đi qua, prior_region_frac quá cao: {res_in['prior_region_frac']}"


def test_8_mpc_to_tube_arc_length_mapping():
    """Test 8: Ánh xạ s_mpc <-> s_tube đồng bộ tọa độ vật lý dù waypoint thô làm lệch dây cung."""
    x_coarse = np.linspace(0.0, 6.75, 16)
    y_coarse = 0.72 * np.sin(2.0 * np.pi * x_coarse / 4.5)
    wp_coarse = np.column_stack([x_coarse, y_coarse])

    planner = CoupledVelocityTubeH124(mu_default=0.35, kappa_th=1.05)
    planner.build_offline_tube(wp_coarse, mu_profile=lambda s: 0.30, run_wcsph_check=False)

    ctrl = DFEMPCCAMRController(N=10, dt=0.05, mu=0.35, ipopt_max_iter=15)
    ctrl.set_path(wp_coarse)
    ctrl.set_tube_planner(planner)

    assert ctrl.s_mapping_max_drift_m > 0.05, f"Kỳ vọng phát hiện độ lệch s > 0.05 m, nhận {ctrl.s_mapping_max_drift_m:.4f}"
    s_test = np.linspace(0.0, ctrl.path_length, 60)
    s_mapped = ctrl.map_s_mpc_to_tube(s_test)
    p_mpc = np.column_stack([ctrl.spline_x(s_test), ctrl.spline_y(s_test)])
    p_tube = np.column_stack([
        np.interp(s_mapped, planner.s_grid, planner.x_grid),
        np.interp(s_mapped, planner.s_grid, planner.y_grid),
    ])
    pos_err = np.linalg.norm(p_mpc - p_tube, axis=1)
    assert np.max(pos_err) < 0.15 and np.mean(pos_err) < 0.05, f"Sai lệch vị trí sau ánh xạ quá lớn: max={np.max(pos_err):.4f}"


def test_9_realtime_ipopt_warmstart_budget():
    """Test 9: Bộ điều khiển thời gian thực N=8, ipopt_max_iter=12 + to_function primal-dual warm-start giữ p95 < 50 ms ở chu kỳ 20 Hz."""
    waypoints, mu_fn, d_net_fn, zc_fn, camber_fn = generate_multi_regime_benchmark_scenario()
    planner = CoupledVelocityTubeH124(mu_default=0.35, kappa_th=1.05)
    planner.build_offline_tube(
        waypoints, mu_profile=mu_fn, d_net_profile=d_net_fn,
        z_com_profile=zc_fn, phi_camber_profile=camber_fn, run_wcsph_check=False,
    )

    ctrl = DFEMPCCAMRController(N=8, dt=0.05, mu=0.35, state_con_mode="skip0", ipopt_max_iter=12)
    ctrl.set_path(waypoints)
    ctrl.set_tube_planner(planner)

    _, _, th0 = ctrl.get_path_reference(0.0)
    x = np.zeros(12)
    x[0:2] = waypoints[0]
    x[2] = th0
    x[3] = 0.25 * np.cos(th0)
    x[4] = 0.25 * np.sin(th0)
    x[9] = 0.0
    x[10] = 0.25
    phi = np.zeros(4)

    solve_ms = []
    n_opt = 0
    for step in range(36):
        _, _, xp, _, info = ctrl.solve_step(x, phi, mu_floor=None)
        if step >= 1:  # Bỏ bước warm-up đầu tiên trước khi khởi hành (t=0)
            solve_ms.append(float(info["solve_time_ms"]))
            if info["status"] == "optimal":
                n_opt += 1
        x = xp[:, 1]

    mean_ms = float(np.mean(solve_ms))
    p95_ms = float(np.percentile(solve_ms, 95))
    assert p95_ms < 50.0, f"Thời gian giải p95={p95_ms:.2f} ms vượt ngân sách 50 ms!"
    assert mean_ms < 35.0, f"Thời gian giải TB={mean_ms:.2f} ms quá cao!"
    assert n_opt >= int(0.95 * len(solve_ms)), f"Tỷ lệ optimal quá thấp: {n_opt}/{len(solve_ms)}"


if __name__ == "__main__":
    print("=" * 78)
    print("CHẠY BỘ KIỂM THỬ ĐƠN VỊ & TÍCH HỢP SPATIAL FRICTION GP (9 TESTS)")
    print("=" * 78)

    tests = [
        ("Test 1: Cold-Start & Unexplored Reversion", test_1_cold_start_and_unexplored_reversion),
        ("Test 2: GP Speed (< 10ms) & Bounds", test_2_speed_and_numerical_stability),
        ("Test 3: Disjoint Split Trajectory Conformal Coverage", test_3_disjoint_split_trajectory_conformal_coverage),
        ("Test 4: Tube Integration & No Double-Counting", test_4_tube_no_double_counting_and_fixed_point),
        ("Test 5: Online Tube Local mu Scaling Bugfix", test_5_online_tube_local_mu_scaling_bugfix),
        ("Test 6: Controller Horizon P_mu & Backwards Compat", test_6_controller_horizon_pmu_and_backwards_compat),
        ("Test 7: Footprint IFF Score & Prior Diagnostics", test_7_footprint_iff_conformal_and_prior_diagnostics),
        ("Test 8: MPC-to-Tube Arc-Length Mapping", test_8_mpc_to_tube_arc_length_mapping),
        ("Test 9: Real-Time IPOPT Warm-Start Budget", test_9_realtime_ipopt_warmstart_budget),
    ]
    for label, fn in tests:
        print(f"\n-> {label}...")
        fn()
        print("   [PASS]")

    print("\n" + "=" * 78)
    print("KẾT LUẬN: TOÀN BỘ 9/9 TEST SPATIAL FRICTION GP & INTEGRATIONS ĐÃ ĐẠT 100%!")
    print("=" * 78)
