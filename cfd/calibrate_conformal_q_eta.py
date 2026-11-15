# -*- coding: utf-8 -*-
"""
calibrate_conformal_q_eta.py
=============================================================================
Hiệu chuẩn Ngoại tuyến (Offline 1-Time) Từ điển Nguyên hàm Hình học A* & Split Conformal
=============================================================================
Quy trình thực thi độc lập, hoàn toàn phi tuần hoàn (không dùng rng.gumbel tổng hợp):
1. Khởi tạo bộ dữ liệu quỹ đạo A* chuẩn hóa hình học gồm:
   - Tập huấn luyện độ chệch tất định D_train: 20 quỹ đạo x 4 Checkpoint (N_train = 80).
   - Tập hiệu chuẩn Conformal D_cal (độc lập): 50 quỹ đạo x 4 Checkpoint (N_cal = 200).
   Bao phủ đủ 4 loại nguyên hàm công nghiệp: góc rẽ 45°, góc rẽ 90°, đảo chiều chữ S
   (S-chicane biến thiên khoảng cách liên kết ds_link) và vòng tránh vật cản (obstacle detour),
   cùng dải ma sát mu in [0.12, 0.35], bệ cao Z_COM in [0.0555, 0.22] m, camber, warp.
2. Tại mọi quỹ đạo, trích xuất hồ sơ ngoại sinh thuần hình học (v_ref^geom, a_x^geom, a_y^geom)
   và đối chứng trực tiếp 2-DOF DBEP + Stokes bậc 2 với trình giải hạt DualSliceWCSPH
   (kiểm chứng neo thêm bằng lưới mịn 1286 hạt dp = 0.005 m và RobustWCSPH2D toàn miền).
3. Khớp hồi quy tuyến tính OLS (chỉ trên D_train) cho độ chệch tất định b_bar_eta(a_tot/g)
   và độ chệch tải pháp tuyến, sau đó đánh giá phần dư ngoài mẫu trên D_cal (N_cal = 200).
4. Kiểm định tự tương quan lag-1 toàn cục & nội quỹ đạo (n = 150 cặp chuyển tiếp) và
   tính lượng tử Split Conformal 2 cấp (cấp điểm k*=191/201 và cấp quỹ đạo k_traj*=49/51)
   theo Mệnh đề 1, xuất cấu hình chuẩn ra config/conformal_calibration_q_eta.json.
"""

import os
import sys

# Ngăn NumPy/OpenBLAS tranh chấp luồng khi chạy ProcessPoolExecutor song song 10 worker
for _env_k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_k, "1")

import json
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import numpy as np

_curr = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_curr) if os.path.basename(_curr) in ["cfd", "controller", "solver", "planning", "simulation"] else _curr
for _p in [_root, os.path.join(_root, "cfd"), os.path.join(_root, "controller"), os.path.join(_root, "planning"), os.path.join(_root, "solver")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from robot_model_amr_casadi import TANK_A, TANK_B, TANK_H, H_LIQUID, Z_COM, ZETA_SLOSH, g, M_TOTAL, m_1x, m_1y, lx, ly
from mecanum_traction_dynamics import L_X, L_Y, Z_SLOSH_X, Z_SLOSH_Y
from coupled_wcsph_2d_dual import DualSliceWCSPH
from robust_wcsph import RobustWCSPH2D
from planning.velocity_tube_h124 import CoupledVelocityTubeH124, FZ_NOM
from conformal_checkpoint_eval import (
    compute_conformal_quantile_q_eta,
    verify_coverage_one_sided_betabinomial,
    compute_cold_start_analytical_slosh_bound,
    test_residual_autocorrelation_and_exchangeability,
    compute_joint_and_trajectory_conformal_bounds,
)


def _generate_astar_primitive_trajectory(traj_idx: int, rng: np.random.Generator):
    """
    Sinh một quỹ đạo A* làm trơn C^inf có đúng 4 nguyên hàm hình học đặc trưng
    trải rộng dải gia tốc công nghiệp (45° dịu, vòng tránh trung bình, 90° gắt, S-chicane)
    cùng mặt sàn không đồng nhất.
    """
    c1 = float(rng.uniform(1.15, 1.55))
    ds12 = float(rng.uniform(0.75, 1.50))
    ds23 = float(rng.uniform(1.20, 2.00))
    ds34 = float(rng.uniform(1.25, 2.15))
    c2 = c1 + ds12
    c3 = c2 + ds23
    c4 = c3 + ds34
    x_end = c4 + float(rng.uniform(1.05, 1.45))

    # Phân tầng đa dải độ cong: dịu (A=0.14-0.24, sig=0.52-0.68 -> a/g~0.025-0.05), trung bình (a/g~0.05-0.08), gắt (a/g~0.08-0.12)
    tier_amps_sigs = [
        (0.14, 0.24, 0.52, 0.68),  # Góc rẽ 45° thoáng
        (0.24, 0.34, 0.45, 0.56),  # Vòng tránh / cong trung bình
        (0.36, 0.48, 0.39, 0.46),  # Góc rẽ 90° / chicane tải cao
    ]
    amps, sigs = [], []
    s_cur = 1.0 if (traj_idx % 2 == 0) else -1.0
    for k_pos in range(4):
        tier = (traj_idx + k_pos) % 3
        if traj_idx % 5 == 0:
            tier = 0 if k_pos in (0, 2) else 1
        elif traj_idx % 5 == 1:
            tier = 2
        lo_a, hi_a, lo_s, hi_s = tier_amps_sigs[tier]
        amps.append(s_cur * float(rng.uniform(lo_a, hi_a)))
        sigs.append(float(rng.uniform(lo_s, hi_s)))
        s_cur *= -1.0 if (k_pos != 1 or traj_idx % 3 != 0) else 1.0

    centers = [c1, c2, c3, c4]
    x_ctrl = np.linspace(0.0, x_end, 260)
    y_ctrl = np.zeros_like(x_ctrl)
    for A_k, c_k, sig_k in zip(amps, centers, sigs):
        y_ctrl += A_k * np.exp(-((x_ctrl - c_k) ** 2) / (2.0 * (sig_k ** 2)))
    waypoints = np.column_stack([x_ctrl, y_ctrl])

    # Biến thiên điều kiện mặt sàn theo từng quỹ đạo để đại diện phân phối P_Z
    mu_wet = float(rng.uniform(0.12, 0.24))
    z_ramp = float(rng.uniform(0.14, 0.22))
    camber_deg = float(rng.uniform(2.5, 5.5))

    def _win(x: float, x0: float, x1w: float) -> float:
        return float(0.5 * (np.tanh((x - x0) / 0.08) - np.tanh((x - x1w) / 0.08)))

    # Ánh xạ x theo chiều dài cung s thực tế để vùng ướt (CP3) và cầu dẫn (CP4) khớp tâm khúc cua
    diffs_wp = np.diff(waypoints, axis=0)
    s_wp = np.concatenate(([0.0], np.cumsum(np.hypot(diffs_wp[:, 0], diffs_wp[:, 1]))))
    x_wp = waypoints[:, 0]

    def _x_of_s(s: float) -> float:
        return float(np.interp(s, s_wp, x_wp))

    def mu_fn(s: float) -> float:
        return 0.35 - (0.35 - mu_wet) * _win(_x_of_s(s), c3 - 0.55, c3 + 0.55)

    def zc_fn(s: float) -> float:
        return Z_COM + (z_ramp - Z_COM) * _win(_x_of_s(s), c4 - 0.65, c4 + 0.70)

    def camber_fn(s: float) -> float:
        return float(np.radians(camber_deg)) * _win(_x_of_s(s), c4 - 0.65, c4 + 0.70)

    return waypoints, mu_fn, zc_fn, camber_fn


def _select_four_primitive_windows(planner: CoupledVelocityTubeH124, waypoints: np.ndarray):
    """
    Trích xuất đường cong theo chiều dài cung s và chọn đúng 4 cửa sổ nguyên hàm Checkpoint chính.
    """
    s_grid, kappa, dkappa, cps_raw = planner._extract_geometry_and_checkpoints(waypoints)
    n = len(s_grid)
    # Tách theo 4 phần tư dọc tuyến dựa trên cực trị |kappa| nếu bộ lọc hợp nhất chỉ trả < 4 CP
    corners = [cp for cp in cps_raw if cp.get("type") == "corner_slosh"]
    if len(corners) >= 4:
        corners_sorted = sorted(corners, key=lambda c: c["s_peak"])[:4]
        selected = corners_sorted
    else:
        selected = []
        bounds = np.linspace(0.60, s_grid[-1] - 0.60, 5)
        for k_seg in range(4):
            mask = (s_grid >= bounds[k_seg]) & (s_grid <= bounds[k_seg + 1])
            idxs = np.where(mask)[0]
            if len(idxs) == 0:
                idx_p = int(np.clip((k_seg + 1) * n // 5, 10, n - 11))
            else:
                idx_p = int(idxs[np.argmax(np.abs(kappa[idxs]))])
            idx_in = max(0, idx_p - int(round(0.24 / planner.ds)))
            idx_out = min(n - 1, idx_p + int(round(0.24 / planner.ds)))
            selected.append({
                "cp_id": k_seg + 1,
                "idx_in": idx_in,
                "idx_peak": idx_p,
                "idx_out": idx_out,
                "s_in": float(s_grid[idx_in]),
                "s_peak": float(s_grid[idx_p]),
                "s_out": float(s_grid[idx_out]),
                "kappa_peak": float(kappa[idx_p]),
                "dkappa_peak": float(dkappa[idx_p]),
            })
    selected = sorted(selected, key=lambda c: c["s_peak"])
    for idx, cp in enumerate(selected):
        cp["cp_id"] = idx + 1
    return s_grid, kappa, dkappa, selected


def _evaluate_trajectory_exogenous_vs_wcsph(
    planner: CoupledVelocityTubeH124,
    dual_sph: DualSliceWCSPH,
    base_ckpt: dict,
    dt_cfl: float,
    cw_x: float,
    cw_y: float,
    dp_sph: float,
    traj_idx: int,
    rng: np.random.Generator,
):
    """
    Chạy đối chứng độc lập DBEP ngoại sinh vs DualSliceWCSPH trên 1 quỹ đạo (4 Checkpoint).
    Trả về danh sách 4 dict kết quả từng Checkpoint.
    """
    waypoints, mu_fn, zc_fn, camber_fn = _generate_astar_primitive_trajectory(traj_idx, rng)
    s_grid, kappa, dkappa, cps = _select_four_primitive_windows(planner, waypoints)
    planner.s_grid = s_grid
    planner.kappa_grid = kappa
    planner.dkappa_grid = dkappa
    mu_grid = np.array([float(mu_fn(s)) for s in s_grid], dtype=float)
    zc_grid = np.array([float(zc_fn(s)) for s in s_grid], dtype=float)
    camber_grid = np.array([float(camber_fn(s)) for s in s_grid], dtype=float)
    w_ramp = np.clip((zc_grid - Z_COM) / max(0.22 - Z_COM, 1e-6), 0.0, 1.0)
    warp_grid = 0.0018 * w_ramp

    geom_prof = planner._compute_exogenous_geom_profile(
        s_grid, kappa, dkappa, mu_grid,
        z_com_eff_grid=zc_grid, phi_camber_grid=camber_grid, delta_warp_grid=warp_grid,
    )
    eta_dbep_grid, fz_dbep_grid = planner._evaluate_exogenous_dbep_profile(geom_prof)

    omega_1y = float(np.sqrt((np.pi * g / TANK_B) * np.tanh(np.pi * H_LIQUID / TANK_B)))
    coth_y = float(1.0 / np.tanh(np.pi * H_LIQUID / TANK_B))
    coth_x = float(1.0 / np.tanh(np.pi * H_LIQUID / TANK_A))
    f0_wheels = np.full(4, (M_TOTAL * g) / 4.0)
    s_pitch = np.array([-1.0, -1.0, +1.0, +1.0])
    s_roll = np.array([-1.0, +1.0, -1.0, +1.0])
    m_fl_g = max(1e-3, dual_sph.bridge.m_fl_total * g)

    records = []
    for idx_cp, cp in enumerate(cps):
        idx_in = cp["idx_in"]
        idx_p = cp["idx_peak"]
        idx_out = cp["idx_out"]
        sl = slice(idx_in, idx_out + 1)
        idx_eta_max = idx_in + int(np.argmax(eta_dbep_grid[sl]))
        idx_ay_max = idx_in + int(np.argmax(np.abs(geom_prof["ay_ref_geom"][sl])))

        if idx_cp == 0:
            ds_link = float(cp["s_peak"] - s_grid[0])
            gamma_link = 0.0
            phi_link = 0.0
            ay_prev = 0.0
        else:
            s_prev = float(cps[idx_cp - 1]["s_peak"])
            ds_link = float(cp["s_peak"] - s_prev)
            mask_link = (s_grid >= s_prev) & (s_grid <= cp["s_peak"])
            if np.sum(mask_link) >= 2:
                t_link = float(np.trapezoid(1.0 / np.maximum(geom_prof["v_ref_geom"][mask_link], 0.08), s_grid[mask_link]))
            else:
                t_link = ds_link / max(float(geom_prof["v_ref_geom"][idx_p]), 0.08)
            gamma_link = float(np.exp(-ZETA_SLOSH * omega_1y * t_link))
            phi_link = float((omega_1y * t_link) % (2.0 * np.pi))
            ay_prev = float(records[-1]["ay_geom"])

        ax_exc = float(geom_prof["ax_ref_geom"][idx_eta_max])
        ay_exc = float(geom_prof["ay_ref_geom"][idx_ay_max])
        a_tot = float(np.hypot(ax_exc, ay_exc))
        delta_psi = float(np.trapezoid(kappa[sl], s_grid[sl])) if idx_out > idx_in else 0.0

        eta_dbep_cmp = float(eta_dbep_grid[idx_eta_max])
        fz_dbep_cmp = float(np.min(fz_dbep_grid[sl]))

        v_exc = float(cp.get("v_ub_cp", geom_prof["v_ref_geom"][idx_p]))
        a_brk_x = -min(planner.a_brake_max, 0.45) * (v_exc / max(planner.v_max, 1e-6))
        a_brk_y = -0.35 * ay_exc

        hyb = dual_sph.evaluate_hybrid_ba_checkpoint(
            base_ckpt=base_ckpt,
            dt=dt_cfl,
            ax_exc=ax_exc,
            ay_exc=ay_exc,
            a_brk_x=a_brk_x,
            a_brk_y=a_brk_y,
            ay_prev_geom=ay_prev,
            gamma_link=gamma_link,
            phi_link_rad=phi_link,
            delta_psi_rad=delta_psi,
            t1_window_s=0.150,
            t2_window_s=0.090,
        )
        eta_wcsph = hyb["eta_wcsph_t1_mm"]
        eta_wcsph_t2 = hyb["eta_wcsph_t2_mm"]
        res_sph = hyb["res_t1"]

        # Phản thực tế T2 thuần ngoại sinh đồng bộ ghép cặp 2D (giảm 15% sóng ngang Y khi đảo chiều + 0.70 sóng dọc X)
        eta_x_brk_mm = (0.5 * TANK_A * (abs(a_brk_x) / g)) * 1000.0 * 0.65
        eta_y_brk_mm = (0.5 * TANK_B * (abs(a_brk_y) / g)) * 1000.0 * 0.35
        eta_dbep_t2 = max(eta_dbep_cmp, 0.85 * eta_dbep_cmp + 0.70 * eta_x_brk_mm + eta_y_brk_mm)
        a_tot_t2_eff = max(a_tot, float(np.hypot(a_brk_x, a_brk_y)))

        zc_cp = float(zc_grid[idx_p])
        phi_cp = float(camber_grid[idx_p])
        warp_cp = float(warp_grid[idx_p])
        gamma_roll_cp = 1.0 / max(0.25, 1.0 - (M_TOTAL * g * zc_cp) / max(planner.k_roll, 1e-6))
        dFz_slosh = res_sph["dFz_slosh_wheels"]
        dFz_pitch = (ax_exc * zc_cp * M_TOTAL) / (4.0 * L_X)
        dFz_roll = gamma_roll_cp * (ay_exc * zc_cp * M_TOTAL * np.cos(phi_cp)) / (4.0 * L_Y)
        dFz_camber = gamma_roll_cp * (M_TOTAL * g * zc_cp * abs(np.sin(phi_cp))) / (4.0 * L_Y)
        dFz_warp = planner.k_z_corner * abs(warp_cp)
        fz_wcsph = max(0.0, float(np.min(
            f0_wheels + s_pitch * dFz_pitch + s_roll * dFz_roll + dFz_slosh
            - dFz_camber - dFz_warp - planner.dfz_iso_traj_max
        )))

        records.append({
            "traj_idx": traj_idx,
            "cp_id": idx_cp + 1,
            "ax_geom": ax_exc,
            "ay_geom": ay_exc,
            "a_tot_geom": a_tot,
            "a_tot_t2_eff_geom": a_tot_t2_eff,
            "ds_link_m": ds_link,
            "gamma_link": gamma_link,
            "delta_psi_rad": delta_psi,
            "eta_dbep_mm": eta_dbep_cmp,
            "eta_wcsph_mm": float(eta_wcsph),
            "e_raw_eta_mm": float(eta_wcsph - eta_dbep_cmp),
            "eta_dbep_t2_mm": float(eta_dbep_t2),
            "eta_wcsph_t2_mm": float(eta_wcsph_t2),
            "e_raw_eta_t2_mm": float(eta_wcsph_t2 - eta_dbep_t2),
            "t2_surge_mm": float(max(0.0, eta_wcsph_t2 - eta_wcsph)),
            "worst_phase_gain": float(hyb["worst_phase_gain"]),
            "fz_dbep_N": fz_dbep_cmp,
            "fz_wcsph_N": float(fz_wcsph),
            "e_raw_fz_N": float(fz_dbep_cmp - fz_wcsph),
        })
    return records


_WORKER_CTX = {}


def _eval_cal_traj_worker(task_args: tuple) -> list:
    """Worker song song cấp tiến trình cho hiệu chuẩn Offline Conformal Hybrid B+A."""
    dp_cal, dt_cal, cw_x, cw_y, traj_idx, seed_val = task_args
    ctx = _WORKER_CTX
    if ctx.get("dp_cal") != dp_cal or ctx.get("dt_cal") != dt_cal:
        planner = CoupledVelocityTubeH124(mu_default=0.35, kappa_th=1.05)
        dual_sph = DualSliceWCSPH(Lx=TANK_A, Ly=TANK_B, tank_height=TANK_H, h=H_LIQUID, dp=dp_cal)
        for _ in range(10):
            dual_sph.step(dt_cal, ax=0.0, ay=0.0)
        base_ckpt = dual_sph.save_checkpoint()
        ctx["dp_cal"] = dp_cal
        ctx["dt_cal"] = dt_cal
        ctx["planner"] = planner
        ctx["dual_sph"] = dual_sph
        ctx["base_ckpt"] = base_ckpt
    rng = np.random.default_rng(int(seed_val))
    return _evaluate_trajectory_exogenous_vs_wcsph(
        ctx["planner"], ctx["dual_sph"], ctx["base_ckpt"],
        dt_cal, cw_x, cw_y, dp_cal, traj_idx, rng
    )


def run_conformal_calibration_pipeline():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    t0 = time.perf_counter()
    print("=" * 80)
    print("CALIBRATE SPLIT CONFORMAL & TỪ ĐIỂN NGUYÊN HÀM HÌNH HỌC A* (OFFLINE 1-TIME)")
    print("=" * 80)

    planner = CoupledVelocityTubeH124(mu_default=0.35, kappa_th=1.05)

    # Khởi tạo trình giải WCSPH lưới tương xứng dp = 0.010 m (80+120=200 hạt lỏng, dt = 100 us <= 0.50*CFL)
    dp_cal = 0.010
    dt_cal = 0.000100
    dual_sph = DualSliceWCSPH(Lx=TANK_A, Ly=TANK_B, tank_height=TANK_H, h=H_LIQUID, dp=dp_cal)
    for _ in range(10):
        dual_sph.step(dt_cal, ax=0.0, ay=0.0)
    base_ckpt = dual_sph.save_checkpoint()
    for s_cal in range(36):
        al = min(1.0, (s_cal + 1) / 14.0)
        dual_sph.step(dt_cal, ax=1.0 * al, ay=1.0 * al)
    cw_y = max(0.25, abs(dual_sph.solver_yz.f_wall_2d[0] * dual_sph.Lx) / max(1e-3, dual_sph.bridge.m_fl_total))
    cw_x = max(0.20, abs(dual_sph.solver_xz.f_wall_2d[0] * dual_sph.Ly) / max(1e-3, dual_sph.bridge.m_fl_total))

    max_workers = min(10, os.cpu_count() or 4)
    print(f"\n[Chuẩn bị] Hybrid B+A (T1=150 ms, T2=90 ms) | dp={dp_cal*1000:.1f} mm, dt={dt_cal*1e6:.0f} us | {max_workers} tiến trình song song")

    # Bước 1: Chạy 20 quỹ đạo huấn luyện D_train (80 CP -> 160 điểm T1+T2) để khớp OLS độ chệch tất định b_bar_eta(a/g) & b_bar_Fz(a/g)
    n_train_traj = 20
    rng_train = np.random.default_rng(101)
    train_seeds = rng_train.integers(0, 2**31 - 1, size=n_train_traj)
    train_tasks = [
        (dp_cal, dt_cal, cw_x, cw_y, t_i, int(train_seeds[t_i]))
        for t_i in range(n_train_traj)
    ]
    train_recs = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for recs_i in executor.map(_eval_cal_traj_worker, train_tasks):
            train_recs.extend(recs_i)

    a_g_train_t1 = np.array([r["a_tot_geom"] / g for r in train_recs], dtype=float)
    e_eta_train_t1 = np.array([r["e_raw_eta_mm"] for r in train_recs], dtype=float)
    a_g_train_t2 = np.array([r["a_tot_t2_eff_geom"] / g for r in train_recs], dtype=float)
    e_eta_train_t2 = np.array([r["e_raw_eta_t2_mm"] for r in train_recs], dtype=float)
    a_g_train_all = np.concatenate([a_g_train_t1, a_g_train_t2])
    e_eta_train_all = np.concatenate([e_eta_train_t1, e_eta_train_t2])
    e_fz_train = np.array([r["e_raw_fz_N"] for r in train_recs], dtype=float)

    # Hồi quy tuyến tính OLS tự nhiên trên tập Huấn luyện (gộp T1+T2, không ép kẹp cứng b0 = 0.15 mm)
    slope_eta, intercept_eta = np.polyfit(a_g_train_all, e_eta_train_all, 1)
    if intercept_eta < 0.0:
        b0_mm = 0.0
        b1_mm = float(max(0.0, np.dot(a_g_train_all, e_eta_train_all) / max(np.dot(a_g_train_all, a_g_train_all), 1e-12)))
    else:
        b0_mm = float(intercept_eta)
        b1_mm = float(max(0.0, slope_eta))
    bias_pred_train = np.maximum(0.0, b0_mm + b1_mm * a_g_train_all)
    bias_eta_max_mm = float(round(float(np.percentile(bias_pred_train, 95)), 3))

    slope_fz, intercept_fz = np.polyfit(a_g_train_t1, e_fz_train, 1)
    if intercept_fz < 0.0:
        bias_fz_b0_N = 0.0
        bias_fz_b1_N = float(max(0.0, np.dot(a_g_train_t1, e_fz_train) / max(np.dot(a_g_train_t1, a_g_train_t1), 1e-12)))
    else:
        bias_fz_b0_N = float(intercept_fz)
        bias_fz_b1_N = float(max(0.0, slope_fz))
    bias_fz_pred_train = np.maximum(0.0, bias_fz_b0_N + bias_fz_b1_N * a_g_train_t1)
    bias_fz_max_N = float(round(float(np.percentile(bias_fz_pred_train, 95)), 4))

    print(f"\n[Bước 1] Hồi quy OLS tự nhiên trên D_train ({len(train_recs)} CP x 2 nhánh = {len(a_g_train_all)} điểm T1+T2):")
    print(f"         b0 = {b0_mm:.4f} mm, b1 = {b1_mm:.4f} mm/(a/g), b_eta,max = {bias_eta_max_mm:.3f} mm")
    print(f"         b_Fz0 = {bias_fz_b0_N:.4f} N, b_Fz1 = {bias_fz_b1_N:.4f} N/(a/g), b_Fz,max = {bias_fz_max_N:.4f} N")

    # Bước 2: Chạy tập hiệu chuẩn độc lập D_cal (50 quỹ đạo x 4 CP = 200 CP -> 400 điểm T1+T2) KHÔNG rò rỉ dữ liệu train
    n_cal_traj = 50
    cps_per_traj = 4
    n_cal_samples = n_cal_traj * cps_per_traj
    alpha = 0.05
    rng_cal = np.random.default_rng(2026)
    cal_seeds = rng_cal.integers(0, 2**31 - 1, size=n_cal_traj)
    cal_tasks = [
        (dp_cal, dt_cal, cw_x, cw_y, n_train_traj + t_i, int(cal_seeds[t_i]))
        for t_i in range(n_cal_traj)
    ]
    cal_recs = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for recs_i in executor.map(_eval_cal_traj_worker, cal_tasks):
            cal_recs.extend(recs_i)

    a_g_t1_cal = np.array([r["a_tot_geom"] / g for r in cal_recs], dtype=float)
    a_g_t2_cal = np.array([r["a_tot_t2_eff_geom"] / g for r in cal_recs], dtype=float)
    eta_dbep_t1_cal = np.array([r["eta_dbep_mm"] for r in cal_recs], dtype=float)
    eta_wcsph_t1_cal = np.array([r["eta_wcsph_mm"] for r in cal_recs], dtype=float)
    eta_dbep_t2_cal = np.array([r["eta_dbep_t2_mm"] for r in cal_recs], dtype=float)
    eta_wcsph_t2_cal = np.array([r["eta_wcsph_t2_mm"] for r in cal_recs], dtype=float)
    t2_surge_cal = np.array([r["t2_surge_mm"] for r in cal_recs], dtype=float)
    fz_dbep_cal = np.array([r["fz_dbep_N"] for r in cal_recs], dtype=float)
    fz_wcsph_cal = np.array([r["fz_wcsph_N"] for r in cal_recs], dtype=float)

    bias_eta_t1_cal = np.minimum(bias_eta_max_mm, b0_mm + b1_mm * a_g_t1_cal)
    eta_rom_corr_t1_cal = eta_dbep_t1_cal + bias_eta_t1_cal
    r_eta_t1_cal = eta_wcsph_t1_cal - eta_rom_corr_t1_cal

    bias_eta_t2_cal = np.minimum(bias_eta_max_mm, b0_mm + b1_mm * a_g_t2_cal)
    eta_rom_corr_t2_cal = eta_dbep_t2_cal + bias_eta_t2_cal
    r_eta_t2_cal = eta_wcsph_t2_cal - eta_rom_corr_t2_cal

    # Chọn phần dư khắt khe nhất giữa T1 và T2 tại mỗi Checkpoint để kiểm định conformal N_cal = 200 (k*=191, k_traj*=49)
    r_eta_cal = np.where(np.abs(r_eta_t2_cal) >= np.abs(r_eta_t1_cal), r_eta_t2_cal, r_eta_t1_cal)

    # Thống kê thuần túy trên 400 điểm vật lý (T1 + T2), KHÔNG chèn điểm 0.0
    eta_wcsph_all_cal = np.concatenate([eta_wcsph_t1_cal, eta_wcsph_t2_cal])
    r_eta_all_cal = np.concatenate([r_eta_t1_cal, r_eta_t2_cal])

    bias_fz_cal = np.minimum(bias_fz_max_N, bias_fz_b0_N + bias_fz_b1_N * a_g_t1_cal)
    fz_rom_corr_cal = np.maximum(0.0, fz_dbep_cal - bias_fz_cal)
    r_fz_cal = fz_rom_corr_cal - fz_wcsph_cal

    rmse_mm = float(np.sqrt(np.mean(r_eta_all_cal ** 2)))
    nrmse = float(rmse_mm / max(float(np.max(eta_wcsph_all_cal) - np.min(eta_wcsph_all_cal)), 1e-8))
    nse = 1.0 - float(np.sum(r_eta_all_cal ** 2) / max(np.sum((eta_wcsph_all_cal - np.mean(eta_wcsph_all_cal)) ** 2), 1e-8))
    rmse_t1_mm = float(np.sqrt(np.mean(r_eta_t1_cal ** 2)))
    nse_t1 = 1.0 - float(np.sum(r_eta_t1_cal ** 2) / max(np.sum((eta_wcsph_t1_cal - np.mean(eta_wcsph_t1_cal)) ** 2), 1e-8))
    mean_res_mm = float(np.mean(r_eta_all_cal))
    sigma_eta = float(max(0.25, np.std(r_eta_cal, ddof=1)))
    sigma_fz = float(max(0.20, np.std(r_fz_cal, ddof=1)))

    # Bước 3: Kiểm định tự tương quan lag-1 toàn cục & nội quỹ đạo (n = 150 cặp chuyển tiếp)
    ac_res = test_residual_autocorrelation_and_exchangeability(
        r_eta_cal, n_trajectories=n_cal_traj, cps_per_traj=cps_per_traj, alpha=alpha
    )
    print(f"\n[Bước 2] Đánh giá ngoài mẫu trên D_cal (N_cal = {n_cal_samples} CP | {len(eta_wcsph_all_cal)} điểm T1+T2):")
    print(f"         mean(r_eta) = {mean_res_mm:+.4f} mm | RMSE = {rmse_mm:.4f} mm (NRMSE={nrmse:.4f}) | NSE = {nse:.4f} (T1 NSE={nse_t1:.4f})")
    print(f"         Lag-1 Autocorrelation: r1_global = {ac_res['r1_global']:.4f}, r1_intra = {ac_res['r1_intra']:.4f} "
          f"(SE={ac_res['se']:.4f}, z={ac_res['z_stat']:.3f}, p={ac_res['p_value_two_sided']:.4f})")

    # Bước 4: Lượng tử Split Conformal 2 cấp theo Mệnh đề 1 (cấp điểm k*=191/201 & cấp quỹ đạo k_traj*=49/51)
    joint_bounds = compute_joint_and_trajectory_conformal_bounds(
        np.abs(r_eta_cal), np.abs(r_fz_cal),
        sigma_eta=sigma_eta, sigma_fz=sigma_fz,
        cps_per_traj=cps_per_traj, alpha=alpha,
    )
    # Bổ sung phân vị Split Conformal 1 chiều cho biên sóng sánh |r_eta| và độ dâng phanh T2
    uni_eta = compute_conformal_quantile_q_eta(np.abs(r_eta_cal), alpha=alpha)
    uni_t2 = compute_conformal_quantile_q_eta(t2_surge_cal, alpha=alpha)
    beta_cov = verify_coverage_one_sided_betabinomial(np.abs(r_eta_cal), float(uni_eta["q_hat"]), n_cal_samples, alpha)

    q_eta_rom_mm = float(round(max(joint_bounds["q_eta_rom_mm"], uni_eta["q_hat"]), 3))
    q_fz_rom_N = float(round(joint_bounds["q_fz_rom_N"], 3))
    q_eta_t2_surge_mm = float(round(float(uni_t2["q_hat"]), 3))
    q_eta_total_rom_mm = float(round(bias_eta_max_mm + q_eta_rom_mm, 3))
    q_eta_cl_mm = float(round(q_eta_total_rom_mm + q_eta_t2_surge_mm, 3))

    # Bước 5: Neo kiểm chứng bổ sung bằng DualSliceWCSPH mịn 1286 hạt (dp=0.005 m, 180 ms) & RobustWCSPH2D (dp=0.005 m, 160 ms)
    sph_fine = DualSliceWCSPH(Lx=TANK_A, Ly=TANK_B, tank_height=TANK_H, h=H_LIQUID, dp=0.005)
    for _ in range(10):
        sph_fine.step(6e-5, ax=0.0, ay=0.0)
    fine_res = sph_fine.integrate_window(n_steps=3000, dt=6e-5, ax=0.12, ay=0.91, ramp_steps=250)
    fy_fine = float(fine_res["peak_fy_wall_N"])
    fine_peak_eta_y_mm = float(fine_res["peak_eta_surf_y_mm"])

    robust_sph = RobustWCSPH2D(tank_width=TANK_B, liquid_height=H_LIQUID, dp=0.005)
    max_eta_robust_mm = 0.0
    for k_rb in range(2000):
        robust_sph.step(8e-5, 0.91 * np.sin(np.pi * min(1.0, k_rb / 1200.0)))
        xf = robust_sph.pos[:robust_sph.n_fluid, 0]
        eta_cur_mm = 6.0 * H_LIQUID * abs(float(np.mean(xf))) / TANK_B * 1000.0
        if eta_cur_mm > max_eta_robust_mm:
            max_eta_robust_mm = eta_cur_mm

    print(f"\n[Bước 3] Phân vị Split Conformal Hợp nhất & Toàn quỹ đạo (Mệnh đề 1):")
    print(f"         k_star = {joint_bounds['k_star']}/{n_cal_samples + 1} | q_eta_ROM = {q_eta_rom_mm:.3f} mm | q_Fz_ROM = {q_fz_rom_N:.3f} N")
    print(f"         q_eta_total_ROM = {q_eta_total_rom_mm:.3f} mm | q_eta_T2_surge = {q_eta_t2_surge_mm:.3f} mm | q_eta_CL = {q_eta_cl_mm:.3f} mm")
    print(f"         k_traj = {joint_bounds['k_traj']}/{n_cal_traj + 1} | q_eta_traj = {joint_bounds['q_eta_traj_mm']:.3f} mm | q_Fz_traj = {joint_bounds['q_fz_traj_N']:.3f} N")
    print(f"         Neo kiểm chứng 1286 hạt (dp=5mm, 180 ms): Fy_wall={fy_fine:.4f} N, peak_eta_y={fine_peak_eta_y_mm:.2f} mm | RobustWCSPH2D max_eta={max_eta_robust_mm:.2f} mm")

    # Bước 6: Kiểm tra chặn giải tích khởi động nguội (Cold-Start Bound)
    cold_start_res = compute_cold_start_analytical_slosh_bound(TANK_B, H_LIQUID, 0.91)
    max_delta_eta_mm = float(cold_start_res["q_eta_analytical_mm"])

    output_path = os.path.join(_root, "config", "conformal_calibration_q_eta.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    elapsed_s = time.perf_counter() - t0

    cal_payload = {
        "calibrated_at": datetime.now().isoformat(),
        "pipeline_runtime_s": round(elapsed_s, 3),
        "method": "Split Conformal over Offline A* Geometric Primitives Dictionary (Hybrid B+A: Long-Window T1=150ms & Braking T2=90ms DualSliceWCSPH vs Exogenous DBEP)",
        "hybrid_ba_enabled": True,
        "d_cal_includes_t2": True,
        "dp_cal_m": dp_cal,
        "dt_cal_s": dt_cal,
        "t1_window_ms": 150.0,
        "t2_window_ms": 90.0,
        "n_train": len(train_recs),
        "n_cal": n_cal_samples,
        "n_cal_points_total": int(len(eta_wcsph_all_cal)),
        "n_trajectories": n_cal_traj,
        "cps_per_traj": cps_per_traj,
        "alpha": alpha,
        "k_star": int(joint_bounds["k_star"]),
        "k_traj": int(joint_bounds["k_traj"]),
        "cp_finite_lower_bound": round(float(joint_bounds["cp_finite"]), 6),
        "traj_finite_lower_bound": round(float(joint_bounds["traj_finite"]), 6),
        "bias_b0_mm": round(b0_mm, 4),
        "bias_b1_mm": round(b1_mm, 4),
        "bias_eta_max_mm": round(bias_eta_max_mm, 3),
        "bias_fz_b0_N": round(bias_fz_b0_N, 4),
        "bias_fz_b1_N": round(bias_fz_b1_N, 4),
        "bias_fz_max_N": round(bias_fz_max_N, 4),
        "q_eta_rom_mm": q_eta_rom_mm,
        "q_fz_rom_N": q_fz_rom_N,
        "q_eta_t2_surge_mm": q_eta_t2_surge_mm,
        "q_eta_total_rom_mm": q_eta_total_rom_mm,
        "q_eta_traj_mm": round(float(joint_bounds["q_eta_traj_mm"]), 3),
        "q_fz_traj_N": round(float(joint_bounds["q_fz_traj_N"]), 3),
        "q_eta_cl_mm": q_eta_cl_mm,
        "q_hat_mm": q_eta_rom_mm,
        "sigma_eta_mm": round(sigma_eta, 4),
        "sigma_fz_N": round(sigma_fz, 4),
        "rmse_mm": round(rmse_mm, 4),
        "nrmse": round(nrmse, 4),
        "nse": round(nse, 4),
        "rmse_t1_mm": round(rmse_t1_mm, 4),
        "nse_t1": round(nse_t1, 4),
        "mean_residual_mm": round(mean_res_mm, 4),
        "r1_global": round(float(ac_res["r1_global"]), 4),
        "r1_intra": round(float(ac_res["r1_intra"]), 4),
        "se_intra": round(float(ac_res["se"]), 4),
        "z_stat_intra": round(float(ac_res["z_stat"]), 4),
        "p_value_intra": round(float(ac_res["p_value_two_sided"]), 4),
        "no_significant_autocorr": bool(ac_res["no_significant_autocorr"]),
        "recommended_q_eta_m": round(q_eta_total_rom_mm / 1000.0, 6),
        "cold_start_margin_mm": round(max_delta_eta_mm, 3),
        "coverage_beta_binom": {
            "empirical_coverage": round(float(beta_cov["empirical_coverage"]), 4),
            "one_sided_5pct_lower_bound": round(float(beta_cov["one_sided_5pct_lower_bound"]), 4),
            "p_value_one_sided": round(float(beta_cov["p_value_one_sided"]), 4),
        },
        "fine_anchor_check": {
            "dp_m": 0.005,
            "fine_window_ms": 180.0,
            "fine_peak_eta_surf_y_mm": round(fine_peak_eta_y_mm, 3),
            "fy_wall_1286_N": round(fy_fine, 4),
            "robust_wcsph2d_max_eta_mm": round(max_eta_robust_mm, 2),
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cal_payload, f, indent=2, ensure_ascii=False)

    print(f"\n[Bước 4] Xuất thành công cấu hình hiệu chuẩn thực vào: {output_path} ({elapsed_s:.2f} s)")
    print("=" * 80)
    return cal_payload


if __name__ == "__main__":
    run_conformal_calibration_pipeline()
