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

from typing import Dict, List, Optional, Sequence, Union
import numpy as np


class FrictionObserverFFRLS:
    def __init__(
        self,
        mu_init: float = 0.25,
        mu_min: Optional[float] = None,
        mu_max: Optional[float] = None,
        lam: Optional[float] = None,
        P_init: float = 10.0,
        rho_deadzone: float = 0.06,
        adaptive_forgetting: Optional[bool] = None,
        lam_fast: float = 0.75,
        lam_slow: float = 0.95,
        slip_trigger: float = 0.025,
        innovation_neg_trigger: float = -0.055,
        P_shock_floor: float = 0.0025,
    ):
        self.mu_init = float(mu_init)
        self.mu_hat = float(mu_init)
        # Preserve legacy fixed-lam bounds (0.12..0.50) when caller passes explicit `lam`
        # without opting into adaptive_forgetting; use wider bounds (0.08..0.55) for Tier 2.
        use_adaptive = (lam is None) if adaptive_forgetting is None else bool(adaptive_forgetting)
        self.adaptive_forgetting = use_adaptive
        if mu_min is None:
            self.mu_min = 0.08 if use_adaptive else 0.12
        else:
            self.mu_min = float(mu_min)
        if mu_max is None:
            self.mu_max = 0.55 if use_adaptive else 0.50
        else:
            self.mu_max = float(mu_max)

        self.lam_fast = float(lam_fast)
        self.lam_slow = float(lam_slow)
        self.lam = float(self.lam_slow if lam is None else lam)
        self.lam_eff_last = float(self.lam)
        self.P_init = float(P_init)
        self.P = float(P_init)          # Hiệp phương sai ước lượng vô hướng
        self.rho_deadzone = float(rho_deadzone)
        self.slip_trigger = float(slip_trigger)
        self.innovation_neg_trigger = float(innovation_neg_trigger)
        self.P_shock_floor = float(P_shock_floor)
        self.last_fault_triggered = False
        self.last_rel_innovation = 0.0

    def reset(self, mu_init: Optional[float] = None) -> None:
        if mu_init is not None:
            self.mu_init = float(mu_init)
        self.mu_hat = float(np.clip(self.mu_init, self.mu_min, self.mu_max))
        self.P = float(self.P_init)
        self.lam_eff_last = float(self.lam)
        self.last_fault_triggered = False
        self.last_rel_innovation = 0.0

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
        if self.adaptive_forgetting and max_slip >= self.slip_trigger:
            bin_idx = 2 if (max_slip >= 0.06 or rho_k >= 0.16) else 1
        elif rho_k < 0.08 or max_slip < 0.02:
            bin_idx = 0
        elif rho_k < 0.16 or max_slip < 0.06:
            bin_idx = 1
        else:
            bin_idx = 2

        self.last_fault_triggered = False
        self.last_rel_innovation = 0.0

        # Chỉ cập nhật RLS khi có kích thích trượt đủ quan sát được (tránh trôi tham số khi đi đều);
        # ở chế độ adaptive, trượt đột ngột (max_slip >= slip_trigger) mở khóa dead-zone ngay cả khi rho_k thấp.
        update_active = (rho_k >= self.rho_deadzone and max_slip >= 0.015) or (
            self.adaptive_forgetting and max_slip >= self.slip_trigger
        )
        if update_active:
            # Hàm bão hòa trượt phi tuyến ψ(s) = tanh(s / s_char)
            s_char = 0.04
            phi = Fz_total * np.tanh(max_slip / s_char)
            y_meas = float(np.linalg.norm(W_actual_xy))

            # Kiểm tra đổi mới âm hoặc trượt lớn để kích hoạt hệ số quên nhanh + sàn hiệp phương sai
            innovation = y_meas - phi * self.mu_hat
            rel_innovation = innovation / max(phi, 1.0)
            self.last_rel_innovation = float(rel_innovation)

            if self.adaptive_forgetting:
                fault_trigger = (
                    (max_slip > 0.03)
                    or (max_slip >= self.slip_trigger and rel_innovation < self.innovation_neg_trigger)
                )
                self.last_fault_triggered = bool(fault_trigger)
                lam_eff = self.lam_fast if fault_trigger else self.lam_slow
                if (max_slip > 0.06) or (
                    max_slip >= self.slip_trigger and rel_innovation < self.innovation_neg_trigger
                ):
                    self.P = max(self.P, self.P_shock_floor)
            else:
                lam_eff = self.lam

            self.lam_eff_last = float(lam_eff)

            # Bước cập nhật FF-RLS
            K_gain = (self.P * phi) / (lam_eff + phi * self.P * phi)
            self.mu_hat = float(np.clip(self.mu_hat + K_gain * innovation, self.mu_min, self.mu_max))
            self.P = (1.0 / lam_eff) * (1.0 - K_gain * phi) * self.P
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


class OnlineFrictionFusion:
    """
    Tầng 2 (Online 20 Hz): Hợp nhất cận dưới Conformal từ bản đồ ngữ nghĩa (`mu_plan_h`)
    với ước lượng ma sát tức thời từ `FrictionObserverFFRLS` (`mu_rls`, `bin_idx`)
    bằng phân rã không gian mũ và cơ chế bám bất đối xứng ưu tiên an toàn.
    """

    def __init__(
        self,
        delta_mu_bins: Union[Sequence[float], Dict[int, float]] = (0.04, 0.025, 0.012),
        w_exc_bins: Union[Sequence[float], Dict[int, float]] = (0.0, 0.6, 1.0),
        L_corr: float = 1.2,
        L_anomaly_hold: float = 1.2,
        mu_min: float = 0.08,
        mu_max: float = 0.55,
        asymmetric_safety: bool = True,
    ):
        self.delta_mu_bins = delta_mu_bins
        self.w_exc_bins = w_exc_bins
        self.L_corr = float(L_corr)
        self.L_anomaly_hold = float(L_anomaly_hold)
        self.mu_min = float(mu_min)
        self.mu_max = float(mu_max)
        self.asymmetric_safety = bool(asymmetric_safety)

        self.history: List[Dict[str, float]] = []
        self._latched_s: Optional[float] = None
        self._latched_mu_safe: Optional[float] = None
        self._latched_w: float = 0.0

    @staticmethod
    def _get_bin_val(mapping_or_seq: Union[Sequence[float], Dict[int, float]], bin_idx: int) -> float:
        if isinstance(mapping_or_seq, dict):
            if bin_idx in mapping_or_seq:
                return float(mapping_or_seq[bin_idx])
            keys = sorted(mapping_or_seq.keys())
            return float(mapping_or_seq[keys[min(max(bin_idx, 0), len(keys) - 1)]])
        idx = min(max(int(bin_idx), 0), len(mapping_or_seq) - 1)
        return float(mapping_or_seq[idx])

    def reset(self) -> None:
        self.history.clear()
        self._latched_s = None
        self._latched_mu_safe = None
        self._latched_w = 0.0

    def fuse_horizon_mu(
        self,
        mu_plan_h: Union[Sequence[float], np.ndarray, float],
        s_h: Union[Sequence[float], np.ndarray, float],
        mu_rls: float,
        bin_idx: int,
        s_curr: Optional[float] = None,
        record_history: bool = True,
    ) -> np.ndarray:
        mu_plan_arr = np.asarray(mu_plan_h, dtype=float)
        scalar_mode = mu_plan_arr.ndim == 0
        mu_plan_1d = mu_plan_arr.reshape(-1)
        s_arr = np.asarray(s_h, dtype=float).reshape(-1)
        if len(s_arr) == 1 and len(mu_plan_1d) > 1:
            s_arr = np.full(len(mu_plan_1d), float(s_arr[0]), dtype=float)

        s_0 = float(s_arr[0]) if s_curr is None else float(s_curr)
        delta_s = np.maximum(0.0, s_arr - s_0)
        rho_s = np.exp(-delta_s / max(self.L_corr, 1e-6))

        b = int(np.clip(int(bin_idx), 0, 2))
        w_exc = self._get_bin_val(self.w_exc_bins, b)
        delta_mu = self._get_bin_val(self.delta_mu_bins, b)
        mu_rls_safe = float(np.clip(float(mu_rls) - delta_mu, self.mu_min, self.mu_max))
        # Khi thực nghiệm trượt chứng minh ma sát thực thấp hơn nhiều so với tiên nghiệm (> 0.045),
        # tăng trọng số bám theo quan sát tức thời để không giữ niềm tin lạc quan nguy hiểm.
        if b >= 1 and mu_rls_safe < float(mu_plan_1d[0]) - 0.045:
            w_exc = max(w_exc, 0.85)

        # Quản lý bộ nhớ bất thường ma sát theo quãng đường để tránh dao động bật/tắt trên vệt dầu
        if b >= 1 and mu_rls_safe < float(mu_plan_1d[0]):
            if (
                self._latched_s is None
                or mu_rls_safe <= float(self._latched_mu_safe or mu_rls_safe) + 0.015
                or w_exc >= self._latched_w
            ):
                self._latched_s = s_0
                self._latched_mu_safe = mu_rls_safe
                self._latched_w = w_exc
        elif b >= 1 and float(mu_rls) >= float(mu_plan_1d[0]) - 0.015:
            # Kích thích trên nền khô xác nhận ma sát đã phục hồi -> giải phóng chốt
            self._latched_s = None
            self._latched_mu_safe = None
            self._latched_w = 0.0

        gamma_now = w_exc * rho_s
        mu_blend_now = (1.0 - gamma_now) * mu_plan_1d + gamma_now * mu_rls_safe

        if self._latched_s is not None and self._latched_mu_safe is not None:
            ds_since = max(0.0, s_0 - self._latched_s)
            if ds_since <= 2.5 * self.L_anomaly_hold:
                hold_decay = float(np.exp(-ds_since / max(self.L_anomaly_hold, 1e-6)))
                gamma_hold = (self._latched_w * hold_decay) * rho_s
                mu_blend_hold = (1.0 - gamma_hold) * mu_plan_1d + gamma_hold * float(self._latched_mu_safe)
                mu_blend = np.minimum(mu_blend_now, mu_blend_hold)
                gamma_vec = np.maximum(gamma_now, gamma_hold) if np.mean(mu_blend_hold) < np.mean(mu_blend_now) else gamma_now
            else:
                self._latched_s = None
                self._latched_mu_safe = None
                self._latched_w = 0.0
                mu_blend = mu_blend_now
                gamma_vec = gamma_now
        else:
            mu_blend = mu_blend_now
            gamma_vec = gamma_now

        if self.asymmetric_safety:
            mu_fuse = np.minimum(mu_plan_1d, mu_blend)
        else:
            mu_fuse = mu_blend
        mu_fuse = np.clip(mu_fuse, self.mu_min, self.mu_max)

        if record_history:
            self.history.append(
                {
                    "s": s_0,
                    "mu_plan0": float(mu_plan_1d[0]),
                    "mu_rls": float(mu_rls),
                    "bin_idx": b,
                    "gamma_k": float(gamma_vec[0]),
                    "mu_fuse0": float(mu_fuse[0]),
                }
            )

        return mu_fuse[0] if scalar_mode else mu_fuse

    fuse = fuse_horizon_mu


def fuse_horizon_mu(
    mu_plan_h: Union[Sequence[float], np.ndarray, float],
    s_h: Union[Sequence[float], np.ndarray, float],
    mu_rls: float,
    bin_idx: int,
    s_curr: Optional[float] = None,
    delta_mu_bins: Union[Sequence[float], Dict[int, float]] = (0.04, 0.025, 0.012),
    w_exc_bins: Union[Sequence[float], Dict[int, float]] = (0.0, 0.6, 1.0),
    L_corr: float = 1.2,
    mu_min: float = 0.08,
    mu_max: float = 0.55,
    asymmetric_safety: bool = True,
) -> np.ndarray:
    """Hàm hợp nhất không trạng thái (stateless) cho horizon ma sát Tầng 1 + Tầng 2."""
    fusion = OnlineFrictionFusion(
        delta_mu_bins=delta_mu_bins,
        w_exc_bins=w_exc_bins,
        L_corr=L_corr,
        mu_min=mu_min,
        mu_max=mu_max,
        asymmetric_safety=asymmetric_safety,
    )
    return fusion.fuse_horizon_mu(
        mu_plan_h=mu_plan_h,
        s_h=s_h,
        mu_rls=mu_rls,
        bin_idx=bin_idx,
        s_curr=s_curr,
        record_history=False,
    )
