"""
friction_observer_ffrls.py
=============================================================================
BỘ QUAN SÁT MA SÁT FF-RLS & HIỆU CHUẨN CONFORMAL CHUẨN HÓA CẤP LƯỢT CHẠY
(Dự án: DFE-MPCC AMR Chở Chất Lỏng - Định hướng SIMPAT v2.0)

Thực thi chuẩn mực từ Huong_Dan_Sua_Code_Chi_Tiet_SIMPAT.md (Module 3):
1. FrictionObserverFFRLS: Bộ ước lượng ma sát trực tuyến bằng bình phương tối thiểu
   đệ quy có hệ số quên (FF-RLS), kèm vùng chết kích thích (dead-zone) khi xe đi đều (Lỗi A.11).
2. calibrate_mu_run_level_mondrian: Hiệu chuẩn Conformal cấp lượt chạy (1 điểm / run)
   chuẩn hóa theo bin kích thích, bảo đảm xác suất phủ đồng thời >= 1 - eps_mu mà không
   bị nhân B lần sai số (Lỗi A.2).
=============================================================================
"""

import numpy as np


class FrictionObserverFFRLS:
    def __init__(
        self,
        mu_init: float = 0.25,
        mu_min: float = 0.12,
        mu_max: float = 0.50,
        lam: float = 0.985,
        P_init: float = 10.0,
        rho_deadzone: float = 0.06
    ):
        self.mu_hat = mu_init
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.lam = lam                  # Hệ số quên (Forgetting factor)
        self.P = P_init                 # Hiệp phương sai ước lượng vô hướng
        self.rho_deadzone = rho_deadzone

    def update(
        self,
        W_actual_xy: np.ndarray,
        W_cmd_xy: np.ndarray,
        wheel_slip_ratios: np.ndarray,
        Fz_total: float,
        m_total: float,
        g: float = 9.81
    ) -> tuple:
        """
        Cập nhật ước lượng ma sát μ̂ trực tuyến ở mỗi bước thời gian.

        Parameters:
        -----------
        W_actual_xy : np.ndarray shape (2,) lực mặt sàn đo được [Fx, Fy]
        W_cmd_xy : np.ndarray shape (2,) lực lệnh [Fx_cmd, Fy_cmd]
        wheel_slip_ratios : np.ndarray shape (4,) độ trượt 4 bánh xe s_w
        Fz_total : Tổng tải pháp tuyến 4 bánh (N)
        m_total : Tổng khối lượng xe + chất lỏng (kg)

        Returns:
        --------
        (mu_hat, rho_k, bin_index)
        """
        norm_cmd = float(np.linalg.norm(W_cmd_xy))
        denom = max(m_total * g, 1e-3)
        rho_k = norm_cmd / denom
        max_slip = float(np.max(np.abs(wheel_slip_ratios))) if len(wheel_slip_ratios) > 0 else 0.0

        # Phân nhóm kích thích (Bin b ∈ {0: Thấp, 1: Trung bình, 2: Cao})
        if rho_k < 0.08 or max_slip < 0.02:
            bin_idx = 0
        elif rho_k < 0.16 or max_slip < 0.06:
            bin_idx = 1
        else:
            bin_idx = 2

        # Chỉ cập nhật RLS khi có kích thích trượt đủ quan sát được (tránh trôi tham số khi đi đều)
        if rho_k >= self.rho_deadzone and max_slip >= 0.015:
            # Hàm bão hòa trượt phi tuyến ψ(s) = tanh(s / s_char)
            s_char = 0.04
            phi = Fz_total * np.tanh(max_slip / s_char)
            y_meas = float(np.linalg.norm(W_actual_xy))

            # Bước cập nhật FF-RLS
            K_gain = (self.P * phi) / (self.lam + phi * self.P * phi)
            innovation = y_meas - phi * self.mu_hat
            self.mu_hat = float(np.clip(self.mu_hat + K_gain * innovation, self.mu_min, self.mu_max))
            self.P = (1.0 / self.lam) * (1.0 - K_gain * phi) * self.P
            self.P = min(self.P, 20.0)  # Chặn trên chống covariance windup

        return self.mu_hat, rho_k, bin_idx


def calibrate_mu_run_level_mondrian(
    cal_runs_mu: list,
    sigma_mu_bins: np.ndarray,
    eps_mu: float = 0.04
) -> dict:
    """
    Hiệu chuẩn biên sai số ước lượng ma sát ở cấp độ lượt chạy (1 điểm chuẩn hóa / run) (Sửa lỗi A.2).

    Parameters:
    -----------
    cal_runs_mu : list of dicts (mỗi dict là 1 run j ∈ D_cal)
        Chứa mảng thời gian 'mu_hat' (T,), 'mu_true' (T,), và 'bin_idx' (T,) ∈ {0, 1, 2}
    sigma_mu_bins : np.ndarray shape (B,)
        Thang sai số đặc trưng σ_(μ,b) của từng bin b tính từ tập nhận dạng trước đó.
    eps_mu : float
        Ngân sách rủi ro cho ma sát (mặc định 0.04).
    """
    n_runs = len(cal_runs_mu)
    sigma_mu_bins = np.asarray(sigma_mu_bins, dtype=np.float64)
    B = len(sigma_mu_bins)
    run_scores = []

    for run in cal_runs_mu:
        mu_hat = np.asarray(run['mu_hat'], dtype=np.float64)
        mu_true = np.asarray(run['mu_true'], dtype=np.float64)
        bins = np.asarray(run['bin_idx'], dtype=int)
        one_sided_err = np.maximum(0.0, mu_hat - mu_true)

        bin_normalized_max = []
        for b in range(B):
            mask_b = (bins == b)
            if np.any(mask_b):
                max_err_in_bin_b = float(np.max(one_sided_err[mask_b]))
                bin_normalized_max.append(max_err_in_bin_b / max(sigma_mu_bins[b], 1e-6))

        # Điểm số duy nhất của lượt chạy j là cực đại qua tất cả các bin mà lượt chạy đi qua
        s_j_mu = float(np.max(bin_normalized_max)) if bin_normalized_max else 0.0
        run_scores.append(s_j_mu)

    run_scores_sorted = np.sort(run_scores)
    k_idx = int(np.ceil((n_runs + 1) * (1.0 - eps_mu))) - 1
    k_idx = int(np.clip(k_idx, 0, n_runs - 1))
    q_hat_mu_norm = float(run_scores_sorted[k_idx]) if n_runs > 0 else 0.0

    # Biên vật lý áp dụng trực tuyến cho từng bin b: Δμ_safe(b) = q̂_μ * σ_(μ,b)
    delta_mu_bins = q_hat_mu_norm * sigma_mu_bins
    return {
        "q_hat_mu_norm": q_hat_mu_norm,
        "delta_mu_bins": delta_mu_bins,
        "run_scores": run_scores
    }
