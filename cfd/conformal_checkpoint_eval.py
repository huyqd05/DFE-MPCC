"""
conformal_checkpoint_eval.py
=============================================================================
ĐỊNH LƯỢNG BẤT ĐỊNH CONFORMAL CÓ RẼ NHÁNH CHECKPOINT & KIỂM ĐỊNH THỐNG KÊ
(Dự án: DFE-MPCC AMR Chở Chất Lỏng - Định hướng SIMPAT v2.0)

Thực thi các thuật toán từ Huong_Dan_Sua_Code_Chi_Tiet_SIMPAT.md (Module 2 & Module 4):
1. compute_run_nonconformity_score_eta_F: Tính điểm không tương thích cấp run s_i^(η,F)
   từ K nhánh phanh T2 và quỹ đạo chính (Lỗi A.1, A.4).
2. verify_coverage_one_sided_betabinomial: Kiểm định tỷ lệ phủ mẫu hữu hạn một phía dưới
   bằng phân phối Beta-Binomial chính xác (Lỗi A.3).
3. check_iterative_calibration_convergence: Kiểm tra hội tụ giữa 2 vòng lặp R1 và R2
   bằng trung bình cắt cụt phần đuôi và khoảng cách Wasserstein-1 (Lỗi A.4).
=============================================================================
"""

import numpy as np
from scipy.stats import betabinom, wasserstein_distance


def compute_run_nonconformity_score_eta_F(
    branch_results: list,
    main_run_data: dict,
    sigma_eta_h: float,
    sigma_F_h: float
) -> dict:
    """
    Tính điểm không tương thích cấp độ lượt chạy s_i^(η,F) từ K nhánh phanh T2 và quỹ đạo chính.

    Parameters:
    -----------
    branch_results : list of K dicts
        Mỗi phần tử m ứng với nhánh phanh T2 xuất phát từ checkpoint k_m, chứa:
        - 'eta_sph_max_x': max_{t >= t_{k_m}} max(eta_SPH_xL(t), eta_SPH_xR(t)) trên nhánh m
        - 'eta_sph_max_y': max_{t >= t_{k_m}} max(eta_SPH_yL(t), eta_SPH_yR(t)) trên nhánh m
        - 'eta_hat_peak_x': Cận đỉnh dự báo trục x tính tại đúng bước quyết định k_m
        - 'eta_hat_peak_y': Cận đỉnh dự báo trục y tính tại đúng bước quyết định k_m
        - 'Fz_model_branch': mảng (T_b, 4) tải pháp tuyến dự báo trên nhánh m
        - 'Fz_sph_branch': mảng (T_b, 4) tải pháp tuyến thực tế từ SPH trên nhánh m
    main_run_data : dict
        Chứa 'Fz_model_main' (T_main, 4) và 'Fz_sph_main' (T_main, 4) trên quỹ đạo chính.
    sigma_eta_h : float
        Thang sai số đặc trưng mức dâng tại mực nước h [m].
    sigma_F_h : float
        Thang sai số đặc trưng tải pháp tuyến tại mực nước h [N].
    """
    # 1. Điểm thiếu hụt cận đỉnh mức dâng qua K nhánh phanh T2 (đúng lịch gia tốc backup!)
    e_eta_branches = []
    for br in branch_results:
        err_x = max(0.0, float(br['eta_sph_max_x']) - float(br['eta_hat_peak_x']))
        err_y = max(0.0, float(br['eta_sph_max_y']) - float(br['eta_hat_peak_y']))
        e_eta_branches.append(max(err_x, err_y))
    e_eta_i = float(np.max(e_eta_branches)) if e_eta_branches else 0.0

    # 2. Điểm sụt giảm tải pháp tuyến qua cả 4 bánh trên quỹ đạo chính và các nhánh phanh
    diff_F_main = np.maximum(0.0, np.asarray(main_run_data['Fz_model_main']) - np.asarray(main_run_data['Fz_sph_main']))
    e_F_list = [float(np.max(diff_F_main))] if diff_F_main.size > 0 else [0.0]
    for br in branch_results:
        diff_F_br = np.maximum(0.0, np.asarray(br['Fz_model_branch']) - np.asarray(br['Fz_sph_branch']))
        if diff_F_br.size > 0:
            e_F_list.append(float(np.max(diff_F_br)))
    e_F_i = float(np.max(e_F_list))

    # 3. Điểm gộp chuẩn hóa cấp độ lượt chạy
    s_i = max(e_eta_i / max(sigma_eta_h, 1e-6), e_F_i / max(sigma_F_h, 1e-6))
    return {
        "e_eta_i": e_eta_i,
        "e_F_i": e_F_i,
        "s_i_eta_F": float(s_i)
    }


def verify_coverage_one_sided_betabinomial(
    test_scores: np.ndarray,
    q_hat: float,
    n_cal: int,
    eps: float,
    alpha_sig: float = 0.05
) -> dict:
    """
    Kiểm định một phía dưới bằng phân phối Beta-Binomial chính xác (Sửa lỗi A.3).
    Coverage cao hơn kỳ vọng (ví dụ 30/30) cho p-value = 1.0 (ĐẠT), không bị bác bỏ sai!
    """
    test_scores = np.asarray(test_scores)
    n_test = len(test_scores)
    k_cov = int(np.sum(test_scores <= q_hat))
    empirical_cov = k_cov / float(n_test)

    l_param = int(np.floor((n_cal + 1) * eps))
    l_param = max(1, min(l_param, n_cal))
    a_beta = n_cal + 1 - l_param
    b_beta = l_param

    # Xác suất quan sát được số lần phủ <= k_cov dưới giả thiết H0 (exchangeability đúng)
    p_value_lower = float(betabinom.cdf(k_cov, n_test, a_beta, b_beta))
    # Phân vị 5% một phía dưới của tỷ lệ phủ hữu hạn mẫu
    lower_bound_95 = float(betabinom.ppf(alpha_sig, n_test, a_beta, b_beta)) / float(n_test)

    return {
        "n_cal": n_cal,
        "n_test": n_test,
        "k_cov": k_cov,
        "empirical_coverage": empirical_cov,
        "target_coverage": 1.0 - eps,
        "one_sided_5pct_lower_bound": lower_bound_95,
        "p_value_one_sided": p_value_lower,
        "passed": bool(p_value_lower >= alpha_sig)
    }


def check_iterative_calibration_convergence(
    scores_prev: np.ndarray,
    scores_curr: np.ndarray,
    tol_rel: float = 0.10
) -> dict:
    """
    Kiểm tra hội tụ giữa 2 vòng lặp R_(r-1) và R_r (n = 60 runs gộp qua 3 mực nước)
    bằng trung bình cắt cụt phần đuôi (70% - 95% quantile mean) và khoảng cách Wasserstein-1 (Sửa lỗi A.4).
    """
    scores_prev = np.asarray(scores_prev)
    scores_curr = np.asarray(scores_curr)

    q_low_prev, q_high_prev = np.quantile(scores_prev, [0.70, 0.95])
    q_low_curr, q_high_curr = np.quantile(scores_curr, [0.70, 0.95])

    tail_prev = scores_prev[(scores_prev >= q_low_prev) & (scores_prev <= q_high_prev)]
    tail_curr = scores_curr[(scores_curr >= q_low_curr) & (scores_curr <= q_high_curr)]

    mean_tail_prev = float(np.mean(tail_prev)) if len(tail_prev) > 0 else float(np.mean(scores_prev))
    mean_tail_curr = float(np.mean(tail_curr)) if len(tail_curr) > 0 else float(np.mean(scores_curr))
    rel_change_tail = abs(mean_tail_curr - mean_tail_prev) / max(mean_tail_prev, 1e-6)

    w1_dist = float(wasserstein_distance(scores_prev, scores_curr))
    w1_rel = w1_dist / max(float(np.std(scores_prev)), 1e-6)

    return {
        "rel_change_tail_mean": rel_change_tail,
        "wasserstein_rel": w1_rel,
        "converged": bool(rel_change_tail < tol_rel and w1_rel < 0.25)
    }


def compute_conformal_quantile_q_eta(
    nonconformity_scores: np.ndarray,
    alpha: float = 0.05
) -> dict:
    """
    Tính lượng tử Conformal mẫu hữu hạn q_eta ở độ tin cậy 1 - alpha (mặc định 95%).
    Theo chuẩn Split-Conformal Prediction (Vovk et al. 2005; Tibshirani et al. 2019):
        k_idx = ceil((n_cal + 1) * (1 - alpha)) - 1
        q_hat = sorted_scores[k_idx]
    """
    scores = np.sort(np.asarray(nonconformity_scores, dtype=float))
    n = len(scores)
    if n == 0:
        raise ValueError("Mảng nonconformity_scores không được rỗng!")

    k_idx = int(np.ceil((n + 1) * (1.0 - alpha))) - 1
    k_idx = max(0, min(k_idx, n - 1))
    q_hat = float(scores[k_idx])

    return {
        "n_samples": n,
        "alpha": alpha,
        "confidence_level": 1.0 - alpha,
        "k_idx": k_idx,
        "q_hat": q_hat,
        "median": float(np.median(scores)),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores))
    }


def compute_cold_start_analytical_slosh_bound(
    tank_length: float = 0.150,
    liquid_height: float = 0.080,
    a_max: float = 1.20,
    g_acc: float = 9.81
) -> dict:
    """
    Nguồn Giải Tích Cận Trên Khởi Động Lạnh (First-Principles Analytical Bound - Zero Prior).
    Tính trực tiếp biên sai số cực đại q_eta^(analytical) giữa mô hình con lắc tuyến tính (mode 1)
    và động lực học chất lưu phi tuyến thực tế từ phương trình Navier-Stokes:

    1. Phần dư bỏ qua tổng vô hạn các mode sóng bậc cao (n = 2, 3, ...):
       Delta_eta_modes = (L / (2*g)) * [1 - (8/pi^2) * tanh(pi*h/L) / (pi*h/L)] * a_max
    2. Bất đối xứng đỉnh-đáy phi tuyến theo khai triển Stokes bậc 2:
       Delta_eta_Stokes = (pi * eta_linear^2) / (2 * L) * coth(pi * h / L)

    Tổng hai thành phần cho ra biên an toàn giải tích thuần túy (mm) tại t = 0 mà không cần chạy mô phỏng.
    """
    L = float(tank_length)
    h = float(liquid_height)
    a = float(a_max)

    # Mode 1 wavenumber
    k1 = np.pi / L
    k1_h = k1 * h

    # 1. Phần dư các mode bậc cao n >= 2
    # Khối lượng hữu hiệu mode 1 tỷ lệ với (8/pi^2) * tanh(k1*h) / (k1*h)
    ratio_mode1 = (8.0 / (np.pi ** 2)) * (np.tanh(k1_h) / max(k1_h, 1e-6))
    residual_modes_ratio = max(0.0, 1.0 - ratio_mode1)
    delta_eta_modes_m = (L / (2.0 * g_acc)) * residual_modes_ratio * a

    # 2. Sai số phi tuyến mặt thoáng Stokes bậc 2
    # Biên độ sóng tuyến tính danh định dưới gia tốc a_max
    omega_1 = np.sqrt(g_acc * k1 * np.tanh(k1_h))
    # Độ nghiêng mặt nước tĩnh theta_eq ~ a / g
    eta_linear_m = 0.5 * L * (a / g_acc)
    # Hiệu chỉnh Stokes bậc 2: chênh lệch đỉnh sóng so với đáy sóng
    coth_k1_h = 1.0 / max(np.tanh(k1_h), 1e-6)
    delta_eta_stokes_m = (np.pi * (eta_linear_m ** 2) / (2.0 * L)) * coth_k1_h

    # Tổng biên an toàn giải tích (chuyển sang mm)
    delta_modes_mm = delta_eta_modes_m * 1000.0
    delta_stokes_mm = delta_eta_stokes_m * 1000.0
    q_eta_analytical_mm = delta_modes_mm + delta_stokes_mm

    return {
        "tank_length_m": L,
        "liquid_height_m": h,
        "a_max_m_s2": a,
        "ratio_mode1_mass": float(ratio_mode1),
        "residual_higher_modes_mm": float(delta_modes_mm),
        "stokes_2nd_order_mm": float(delta_stokes_mm),
        "q_eta_analytical_mm": float(q_eta_analytical_mm),
        "q_eta_analytical_m": float(q_eta_analytical_mm / 1000.0)
    }


def test_residual_autocorrelation_and_exchangeability(
    residuals,
    n_trajectories: int = 50,
    cps_per_traj: int = 4,
    alpha: float = 0.05,
) -> dict:
    """
    Kiểm định tự tương quan bậc một (Lag-1) trên chuỗi phần dư toàn cục và nội quỹ đạo.
    Lưu ý thống kê: p > 0.05 cho thấy chưa phát hiện tự tương quan tuyến tính bậc một
    có ý nghĩa thống kê, củng cố (chứ không chứng minh tuyệt đối) giả thiết khả hoán
    có điều kiện theo vector đặc trưng hình học ngoại sinh z_m^{A*}.
    """
    import math
    import numpy as np

    res = np.asarray(residuals, dtype=float).ravel()
    if len(res) < 3:
        raise ValueError("Cần ít nhất 3 phần dư để kiểm định tự tương quan.")

    res_c = res - np.mean(res)
    denom_g = float(np.dot(res_c, res_c))
    r1_global = float(np.dot(res_c[:-1], res_c[1:]) / denom_g) if denom_g > 1e-12 else 0.0

    used = min(len(res), int(n_trajectories) * int(cps_per_traj))
    n_traj_eff = max(1, used // max(1, int(cps_per_traj)))
    if cps_per_traj >= 2 and n_traj_eff * cps_per_traj >= 2:
        mat = res[: n_traj_eff * cps_per_traj].reshape(n_traj_eff, cps_per_traj)
        x = mat[:, :-1].ravel()
        y = mat[:, 1:].ravel()
        n_pairs = len(x)
        xc = x - np.mean(x)
        yc = y - np.mean(y)
        denom_i = math.sqrt(float(np.dot(xc, xc)) * float(np.dot(yc, yc)))
        r1_intra = float(np.dot(xc, yc) / denom_i) if denom_i > 1e-12 else 0.0
    else:
        n_pairs = max(1, len(res) - 1)
        r1_intra = r1_global

    se = 1.0 / math.sqrt(max(1, n_pairs))
    z_stat = r1_intra / se
    p_value_two_sided = math.erfc(abs(z_stat) / math.sqrt(2.0))

    return {
        "n_total": int(len(res)),
        "n_trajectories": int(n_traj_eff),
        "cps_per_traj": int(cps_per_traj),
        "n_intra_pairs": int(n_pairs),
        "r1_global": float(r1_global),
        "r1_intra": float(r1_intra),
        "se": float(se),
        "z_stat": float(z_stat),
        "p_value_two_sided": float(p_value_two_sided),
        "no_significant_autocorr": bool(p_value_two_sided > alpha),
    }


def compute_joint_and_trajectory_conformal_bounds(
    scores_eta,
    scores_fz,
    sigma_eta: float,
    sigma_fz: float,
    cps_per_traj: int = 4,
    alpha: float = 0.05,
) -> dict:
    """
    Tính ngưỡng Split Conformal hợp nhất (Joint) ở cả hai cấp độ:
      1) Cấp độ từng Checkpoint đơn lẻ: k* = ceil((N_cal + 1)(1 - alpha))
         với phân phối bao phủ biên Beta(k*, N_cal + 1 - k*).
      2) Cấp độ toàn quỹ đạo (Trajectory-level over M = cps_per_traj checkpoints):
         lấy R_traj,i = max_{1<=m<=M} R_joint,i,m trên N_traj quỹ đạo độc lập,
         k_traj* = ceil((N_traj + 1)(1 - alpha)) để khử mất mát hợp nhất (union bound).
    """
    import math
    import numpy as np

    s_eta = np.abs(np.asarray(scores_eta, dtype=float).ravel())
    s_fz = np.abs(np.asarray(scores_fz, dtype=float).ravel())
    n_cal = len(s_eta)
    if n_cal == 0 or len(s_fz) != n_cal:
        raise ValueError("scores_eta và scores_fz phải cùng kích thước và khác rỗng.")

    sig_eta = max(1e-9, float(sigma_eta))
    sig_fz = max(1e-9, float(sigma_fz))
    r_joint = np.maximum(s_eta / sig_eta, s_fz / sig_fz)

    # 1. Cấp độ Checkpoint (single-checkpoint coverage)
    k_star = int(math.ceil((n_cal + 1) * (1.0 - alpha)))
    r_sorted = np.sort(r_joint)
    if k_star <= n_cal:
        q_joint_cp = float(r_sorted[max(0, k_star - 1)])
        cp_finite = True
    else:
        q_joint_cp = float("inf")
        cp_finite = False

    a_cp = min(k_star, n_cal + 1)
    b_cp = max(1, n_cal + 1 - a_cp)
    beta_cp_mean = a_cp / (a_cp + b_cp)
    beta_cp_std = math.sqrt((a_cp * b_cp) / (((a_cp + b_cp) ** 2) * (a_cp + b_cp + 1.0)))

    q_eta_rom = float(q_joint_cp * sig_eta)
    q_fz_rom = float(q_joint_cp * sig_fz)
    empirical_cp_cov = float(np.mean(r_joint <= q_joint_cp)) if cp_finite else 1.0

    # 2. Cấp độ toàn quỹ đạo (Trajectory-level coverage trên M Checkpoint)
    m_cps = max(1, int(cps_per_traj))
    n_traj = max(1, n_cal // m_cps)
    r_traj = np.max(r_joint[: n_traj * m_cps].reshape(n_traj, m_cps), axis=1)
    k_traj = int(math.ceil((n_traj + 1) * (1.0 - alpha)))
    r_traj_sorted = np.sort(r_traj)
    if k_traj <= n_traj:
        q_joint_traj = float(r_traj_sorted[max(0, k_traj - 1)])
        traj_finite = True
    else:
        q_joint_traj = float("inf")
        traj_finite = False

    a_tr = min(k_traj, n_traj + 1)
    b_tr = max(1, n_traj + 1 - a_tr)
    beta_traj_mean = a_tr / (a_tr + b_tr)
    beta_traj_std = math.sqrt((a_tr * b_tr) / (((a_tr + b_tr) ** 2) * (a_tr + b_tr + 1.0)))

    q_eta_traj = float(q_joint_traj * sig_eta)
    q_fz_traj = float(q_joint_traj * sig_fz)
    empirical_traj_cov = float(np.mean(r_traj <= q_joint_traj)) if traj_finite else 1.0

    return {
        "n_cal": int(n_cal),
        "alpha": float(alpha),
        "k_star": int(k_star),
        "cp_finite": cp_finite,
        "beta_cp_params": [int(a_cp), int(b_cp)],
        "beta_cp_mean": float(beta_cp_mean),
        "beta_cp_std": float(beta_cp_std),
        "q_joint_cp": float(q_joint_cp),
        "q_eta_rom_mm": float(q_eta_rom),
        "q_fz_rom_N": float(q_fz_rom),
        "empirical_cp_coverage": float(empirical_cp_cov),
        "bonferroni_traj_lower_bound": float(max(0.0, 1.0 - m_cps * alpha)),
        "cps_per_traj": int(m_cps),
        "n_traj": int(n_traj),
        "k_traj": int(k_traj),
        "traj_finite": traj_finite,
        "beta_traj_params": [int(a_tr), int(b_tr)],
        "beta_traj_mean": float(beta_traj_mean),
        "beta_traj_std": float(beta_traj_std),
        "q_joint_traj": float(q_joint_traj),
        "q_eta_traj_mm": float(q_eta_traj),
        "q_fz_traj_N": float(q_fz_traj),
        "empirical_traj_coverage": float(empirical_traj_cov),
    }


def load_conformal_calibration_budget(config_path: str | None = None) -> dict:
    """
    Nạp tập tham số hiệu chuẩn Split Conformal Hybrid B+A chính thức từ
    config/conformal_calibration_q_eta.json để đồng bộ hóa cho Bộ điều khiển,
    Bộ lập ống vận tốc Stage B và toàn bộ kịch bản mô phỏng/đối chứng.
    """
    import os
    import json

    if config_path is None:
        _cfd_dir = os.path.dirname(os.path.abspath(__file__))
        _root_dir = os.path.dirname(_cfd_dir)
        config_path = os.path.join(_root_dir, "config", "conformal_calibration_q_eta.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Không tìm thấy tệp hiệu chuẩn Conformal tại '{config_path}'. "
            "Hãy chạy 'python cfd/calibrate_conformal_q_eta.py' trước."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    bias_b0_mm = float(cfg["bias_b0_mm"])
    bias_b1_mm = float(cfg["bias_b1_mm"])
    bias_eta_max_mm = float(cfg["bias_eta_max_mm"])
    q_eta_rom_mm = float(cfg["q_eta_rom_mm"])
    q_eta_total_rom_mm = float(cfg.get("q_eta_total_rom_mm", round(bias_eta_max_mm + q_eta_rom_mm, 3)))
    q_eta_t2_surge_mm = float(cfg.get("q_eta_t2_surge_mm", 0.0))
    q_eta_cl_mm = float(cfg.get("q_eta_cl_mm", round(q_eta_total_rom_mm + q_eta_t2_surge_mm, 3)))
    recommended_q_eta_m = float(cfg.get("recommended_q_eta_m", round(q_eta_total_rom_mm / 1000.0, 6)))
    eta_rom_target_mm = round(max(0.0, 15.0 - q_eta_total_rom_mm), 3)
    eta_bound_cl_mm = round(eta_rom_target_mm + q_eta_cl_mm, 3)

    out = dict(cfg)
    out.update({
        "config_path": os.path.abspath(config_path),
        "bias_b0_mm": bias_b0_mm,
        "bias_b1_mm": bias_b1_mm,
        "bias_eta_max_mm": bias_eta_max_mm,
        "q_eta_rom_mm": q_eta_rom_mm,
        "q_eta_total_rom_mm": q_eta_total_rom_mm,
        "q_eta_t2_surge_mm": q_eta_t2_surge_mm,
        "q_eta_cl_mm": q_eta_cl_mm,
        "recommended_q_eta_m": recommended_q_eta_m,
        "q_eta_cl_m": round(q_eta_cl_mm / 1000.0, 6),
        "eta_rom_target_mm": eta_rom_target_mm,
        "eta_bound_cl_mm": eta_bound_cl_mm,
    })
    return out


