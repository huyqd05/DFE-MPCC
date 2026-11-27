# -*- coding: utf-8 -*-
"""
dual_layer_friction_estimator.py
=============================================================================
HƯỚNG 2: PHÂN TÁCH 2 LỚP NỀN THÍCH NGHI + DỊ THƯỜNG CỤC BỘ CHO ĐA BẢN ĐỒ TOÀN CỤC
(Dual-Layer Background + Anomaly Decomposition: mu(x, t) = mu_bg - delta_mu_anom(x, t))

Kiến trúc tách bạch hai hiện tượng có bản chất thống kê đối lập:
1. AdaptiveBackgroundFrictionModel (Lớp A - Nền toàn cục):
   - Khởi tạo từ khoảng ma sát môi trường [mu_env_low, mu_env_high] bất kỳ.
   - Tinh chỉnh Bayesian hai chiều trên các đoạn bình thường (không dị thường) để gỡ khóa
     bảo thủ quá mức khi khoảng nền đầu vào rộng.
2. DualGateCUSUMDetector (Bộ phát hiện đổi điểm 2 cổng):
   - Cổng nhanh (s_w >= slip_emerg) bắt sự cố nặng (vết dầu) trong 1 chu kỳ (50 ms).
   - Cổng CUSUM một phía lọc nhiễu 1 bước (nhám ISO 8608) và bắt vùng mưa mỏng.
   - Xác nhận khô liên tiếp (dry_streak) để xóa ô dị thường khi mặt sàn đã khô.
3. SparseAnomalyVoxelMap (Lớp B - Bộ nhớ dị thường thưa 2D O(1)):
   - Lưu chỉ các ô lưới 2D có dị thường (bảng băm dict[(ix, iy)]), suy giảm theo thời gian
     exp(-(t - t_last) / tau_dry_s), truy vấn O(1) không phụ thuộc diện tích bản đồ.
4. DualLayerFrictionEstimator:
   - Giữ nguyên interface predict_along_path(...) cho CoupledVelocityTubeH124.
   - Hỗ trợ update_online(...) và fuse_horizon_xy(...) 20 Hz cho DFE-MPCC.
=============================================================================
"""

import math
import os
import sys
import time
from typing import Optional

import numpy as np

_curr = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_curr) if os.path.basename(_curr) in ["planning", "controller", "solver", "simulation", "tests"] else _curr
for _p in [_root, os.path.join(_root, "planning"), os.path.join(_root, "controller"), os.path.join(_root, "solver")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from controller.friction_observer_ffrls import FrictionObserverFFRLS
from controller.mecanum_traction_dynamics import L_X, L_Y, WHEEL_POS
from planning.spatial_friction_gp import compute_footprint_query_points


class AdaptiveBackgroundFrictionModel:
    """
    Lớp A: Mô hình ma sát nền thích nghi Empirical-Bayes (Normal-Normal conjugate).
    Khởi tạo từ khoảng [mu_env_low, mu_env_high] và chỉ cập nhật trên vùng không dị thường.
    """

    def __init__(
        self,
        mu_env_low: float = 0.28,
        mu_env_high: float = 0.42,
        q_bg: float = 1.96,
        sigma_bg_floor: float = 0.015,
    ):
        low = float(mu_env_low)
        high = float(mu_env_high)
        if not (low > 0.0 and high > low):
            raise ValueError(f"Khoảng ma sát nền không hợp lệ: [{low}, {high}] (yêu cầu 0 < low < high)")

        self.mu_env_low = low
        self.mu_env_high = high
        self.q_bg = float(q_bg)
        self.sigma_bg_floor = max(1e-4, float(sigma_bg_floor))

        self.mu_bg_0 = 0.5 * (low + high)
        self.sigma_bg_0 = max(self.sigma_bg_floor, (high - low) / (2.0 * 1.96))
        self.mu_bg = float(self.mu_bg_0)
        self.sigma_bg = float(self.sigma_bg_0)
        self.n_updates = 0

    def get_safe_bg(self) -> float:
        """Trả về cận dưới tin cậy của nền mu_bg_safe bị chặn trong [mu_env_low, mu_env_high]."""
        return float(np.clip(self.mu_bg - self.q_bg * self.sigma_bg, self.mu_env_low, self.mu_env_high))

    def update(self, mu_obs: float, obs_var: float, is_anomaly: bool = False) -> float:
        """Cập nhật Bayesian hai chiều cho nền; bỏ qua hoàn toàn khi đang có dị thường."""
        if is_anomaly:
            return self.get_safe_bg()

        prec_prev = 1.0 / (self.sigma_bg ** 2)
        prec_obs = 1.0 / max(float(obs_var), self.sigma_bg_floor ** 2)
        prec_new = prec_prev + prec_obs

        mu_new = (prec_prev * self.mu_bg + prec_obs * float(mu_obs)) / prec_new
        self.mu_bg = float(np.clip(mu_new, self.mu_env_low, self.mu_env_high))
        self.sigma_bg = float(max(self.sigma_bg_floor, math.sqrt(1.0 / prec_new)))
        self.n_updates += 1
        return self.get_safe_bg()

    def reset(self) -> None:
        self.mu_bg = float(self.mu_bg_0)
        self.sigma_bg = float(self.sigma_bg_0)
        self.n_updates = 0


class DualGateCUSUMDetector:
    """
    Bộ phát hiện đổi điểm 2 cổng theo từng kênh bánh xe:
      - Cổng nhanh (Emergency): trượt lớn & thiếu hụt ma sát -> kích hoạt tức thời (1 bước).
      - Cổng CUSUM một phía: tích lũy thiếu hụt vượt ngưỡng trôi -> lọc nhiễu 1 bước, bắt mưa mỏng.
    """

    def __init__(
        self,
        n_channels: int = 4,
        delta_drift: float = 0.035,
        h_cusum: float = 0.075,
        slip_emerg: float = 0.040,
        slip_confirm: float = 0.020,
        slip_clear: float = 0.015,
        dry_streak_req: int = 2,
    ):
        self.n_channels = int(n_channels)
        self.delta_drift = float(delta_drift)
        self.h_cusum = float(h_cusum)
        self.slip_emerg = float(slip_emerg)
        self.slip_confirm = float(slip_confirm)
        self.slip_clear = float(slip_clear)
        self.dry_streak_req = int(dry_streak_req)

        self.g_stat = np.zeros(self.n_channels, dtype=float)
        self.dry_streak = np.zeros(self.n_channels, dtype=int)

    def _to_channels(self, val, dtype=float) -> np.ndarray:
        arr = np.asarray(val, dtype=dtype).ravel()
        if arr.size == 1:
            return np.full(self.n_channels, arr[0], dtype=dtype)
        if arr.size != self.n_channels:
            raise ValueError(f"Kích thước kênh không khớp: cần {self.n_channels}, nhận {arr.size}")
        return arr

    def step(self, mu_bg: float, mu_obs_vec, slip_vec, active_mask) -> dict:
        mu_obs_arr = self._to_channels(mu_obs_vec, dtype=float)
        slip_arr = np.abs(self._to_channels(slip_vec, dtype=float))
        act_arr = self._to_channels(active_mask, dtype=bool)

        r = np.where(act_arr, float(mu_bg) - mu_obs_arr, 0.0)
        self.g_stat = np.where(
            act_arr,
            np.maximum(0.0, self.g_stat + (r - self.delta_drift)),
            self.g_stat * 0.85,
        )

        fast_trig = (slip_arr >= self.slip_emerg) & (r > 0.025)
        cusum_trig = (self.g_stat > self.h_cusum) & (slip_arr >= self.slip_confirm)
        is_anomaly = fast_trig | cusum_trig

        dry_step = act_arr & (r <= 0.015) & (slip_arr <= self.slip_clear)
        self.dry_streak = np.where(dry_step, self.dry_streak + 1, 0)
        is_dry_confirmed = self.dry_streak >= self.dry_streak_req

        return {
            "is_anomaly": is_anomaly,
            "any_anomaly": bool(np.any(is_anomaly)),
            "fast_triggered": fast_trig,
            "cusum_triggered": cusum_trig,
            "g_stat": self.g_stat.copy(),
            "is_dry_confirmed": is_dry_confirmed,
        }

    def reset(self) -> None:
        self.g_stat.fill(0.0)
        self.dry_streak.fill(0)


class SparseAnomalyVoxelMap:
    """
    Lớp B: Bản đồ dị thường thưa 2D lưu theo bảng băm voxel (O(1) cập nhật & truy vấn).
    Chỉ lưu những ô thực sự có tụt ma sát (vết dầu, vũng nước mưa), kèm suy giảm theo thời gian.
    """

    def __init__(
        self,
        voxel_size: float = 0.18,
        ell_anom: float = 0.22,
        query_radius_cells: int = 2,
        tau_dry_s: float = 300.0,
        prune_thresh: float = 0.005,
    ):
        self.voxel_size = max(0.02, float(voxel_size))
        self.ell_anom = max(0.02, float(ell_anom))
        self.query_radius_cells = max(1, int(query_radius_cells))
        self.tau_dry_s = max(1e-3, float(tau_dry_s))
        self.prune_thresh = max(1e-5, float(prune_thresh))
        self.cells: dict = {}

    def _key(self, x: float, y: float) -> tuple:
        return (
            int(math.floor(float(x) / self.voxel_size)),
            int(math.floor(float(y) / self.voxel_size)),
        )

    def _decayed_val(self, cell: dict, t_now: float) -> float:
        dt = max(0.0, float(t_now) - float(cell["t_last"]))
        return float(cell["delta_mu"]) * math.exp(-dt / self.tau_dry_s)

    def record_anomaly(self, x: float, y: float, delta_mu: float, t_now: float) -> None:
        dmu = float(delta_mu)
        if dmu <= self.prune_thresh:
            return
        key = self._key(x, y)
        prev = self.cells.get(key)
        if prev is not None:
            prev_eff = self._decayed_val(prev, t_now)
            prev["delta_mu"] = max(prev_eff, dmu)
            prev["t_last"] = float(t_now)
            prev["hits"] += 1
        else:
            ix, iy = key
            self.cells[key] = {
                "cx": (ix + 0.5) * self.voxel_size,
                "cy": (iy + 0.5) * self.voxel_size,
                "delta_mu": dmu,
                "t_last": float(t_now),
                "hits": 1,
            }

    def confirm_dry(self, x: float, y: float) -> bool:
        return self.cells.pop(self._key(x, y), None) is not None

    def query_points(self, pts_xy: np.ndarray, t_now: float) -> np.ndarray:
        arr = np.asarray(pts_xy, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, 2)
        n = arr.shape[0]
        if n == 0 or not self.cells:
            return np.zeros(n, dtype=float)

        out = np.zeros(n, dtype=float)
        inv_vs = 1.0 / self.voxel_size
        inv_2ell2 = 1.0 / (2.0 * (self.ell_anom ** 2))
        inv_tau = 1.0 / self.tau_dry_s
        r_c = self.query_radius_cells
        t_cur = float(t_now)
        cells = self.cells

        for i in range(n):
            px = float(arr[i, 0])
            py = float(arr[i, 1])
            ix = int(math.floor(px * inv_vs))
            iy = int(math.floor(py * inv_vs))
            best = 0.0
            for dx in range(-r_c, r_c + 1):
                for dy in range(-r_c, r_c + 1):
                    c = cells.get((ix + dx, iy + dy))
                    if c is None:
                        continue
                    dt = t_cur - c["t_last"]
                    eff = c["delta_mu"] * math.exp(-dt * inv_tau) if dt > 0.0 else c["delta_mu"]
                    if eff <= best:
                        continue
                    d2 = (px - c["cx"]) ** 2 + (py - c["cy"]) ** 2
                    val = eff * math.exp(-d2 * inv_2ell2)
                    if val > best:
                        best = val
            out[i] = best
        return out

    def prune_expired(self, t_now: float) -> int:
        to_del = [k for k, c in self.cells.items() if self._decayed_val(c, t_now) < self.prune_thresh]
        for k in to_del:
            del self.cells[k]
        return len(to_del)

    def reset(self) -> None:
        self.cells.clear()


class DualLayerFrictionEstimator:
    """
    Bộ ước lượng ma sát toàn cục 2 lớp:
      mu(x, t) = mu_bg - delta_mu_anom(x, t)
    Tương thích trực tiếp với CoupledVelocityTubeH124.build_offline_tube(...) và DFE-MPCC 20 Hz.
    """

    def __init__(
        self,
        mu_env_low: float = 0.28,
        mu_env_high: float = 0.42,
        mu_hard_min: float = 0.05,
        q_bg: float = 1.96,
        sigma_bg_floor: float = 0.015,
        voxel_size: float = 0.18,
        ell_anom: float = 0.22,
        tau_dry_s: float = 300.0,
        L_corr_now: float = 0.80,
        delta_mu_rls_margin: float = 0.015,
        L_x: float = L_X,
        L_y: float = L_Y,
        lateral_margin: float = 0.05,
        query_mode: str = "footprint",
    ):
        if query_mode not in ("footprint", "center"):
            raise ValueError(f"query_mode không hợp lệ: '{query_mode}'")

        self.mu_env_low = float(mu_env_low)
        self.mu_env_high = float(mu_env_high)
        self.mu_hard_min = float(mu_hard_min)
        self.L_corr_now = max(0.05, float(L_corr_now))
        self.delta_mu_rls_margin = float(delta_mu_rls_margin)
        self.L_x = float(L_x)
        self.L_y = float(L_y)
        self.lateral_margin = float(lateral_margin)
        self.query_mode = str(query_mode)

        self.bg_model = AdaptiveBackgroundFrictionModel(
            mu_env_low=self.mu_env_low,
            mu_env_high=self.mu_env_high,
            q_bg=q_bg,
            sigma_bg_floor=sigma_bg_floor,
        )
        self.detector = DualGateCUSUMDetector(n_channels=4)
        self.anomaly_map = SparseAnomalyVoxelMap(
            voxel_size=voxel_size,
            ell_anom=ell_anom,
            tau_dry_s=tau_dry_s,
        )
        self.rls_observer = FrictionObserverFFRLS(
            mu_init=self.bg_model.get_safe_bg(),
            mu_min=self.mu_hard_min,
            mu_max=self.mu_env_high,
            lam=None,
            adaptive_forgetting=True,
            rho_deadzone=0.06,
        )

        self._active_anom_delta = 0.0
        self._active_anom_t = -1e9
        self._active_anom_hold_s = 0.25
        self.last_t_now = 0.0
        self.last_predict_time_ms = 0.0
        self.last_det_status = {"any_anomaly": False}

    def predict_along_path(
        self,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        mode: str = "conformal_lcb",
        yaw_grid: Optional[np.ndarray] = None,
        query_mode: Optional[str] = None,
        t_now: Optional[float] = None,
    ) -> dict:
        """
        Dự báo trường ma sát dọc quỹ đạo A* cho CoupledVelocityTubeH124.build_offline_tube().
        Kết hợp nền mu_bg_safe hiện hành trừ bản đồ dị thường 2D đã học trên dấu chân K=6.
        """
        if mode not in ("conformal_lcb", "gp_mean"):
            raise ValueError(f"mode không hợp lệ: '{mode}'. Chọn 'conformal_lcb' hoặc 'gp_mean'.")
        qm = self.query_mode if query_mode is None else str(query_mode)
        if qm not in ("footprint", "center"):
            raise ValueError(f"query_mode không hợp lệ: '{qm}'")

        t0 = time.perf_counter()
        xg = np.asarray(x_grid, dtype=float).ravel()
        yg = np.asarray(y_grid, dtype=float).ravel()
        n = len(xg)
        if n == 0:
            raise ValueError("x_grid và y_grid không được rỗng.")

        t_use = self.last_t_now if t_now is None else float(t_now)

        if yaw_grid is None:
            if n >= 2:
                yg_yaw = np.arctan2(np.gradient(yg), np.gradient(xg))
            else:
                yg_yaw = np.zeros(n, dtype=float)
        else:
            yg_yaw = np.asarray(yaw_grid, dtype=float).ravel()

        if qm == "footprint":
            fp_pts = np.asarray(
                compute_footprint_query_points(
                    xg, yg, yaw_grid=yg_yaw,
                    L_x=self.L_x, L_y=self.L_y, lateral_margin=self.lateral_margin,
                ),
                dtype=float,
            )  # (n, k_fp, 2)
            k_fp = fp_pts.shape[1]
            delta_fp = self.anomaly_map.query_points(fp_pts.reshape(-1, 2), t_use)
            delta_max = np.max(delta_fp.reshape(n, k_fp), axis=1)
        else:
            pts = np.column_stack([xg, yg])
            delta_max = self.anomaly_map.query_points(pts, t_use)

        mu_bg_safe = self.bg_model.get_safe_bg()
        mu_lcb = np.clip(mu_bg_safe - delta_max, self.mu_hard_min, self.mu_env_high)
        mu_mean = np.clip(self.bg_model.mu_bg - delta_max, self.mu_hard_min, self.mu_env_high)
        sigma_arr = np.full(n, self.bg_model.sigma_bg, dtype=float)

        if mode == "conformal_lcb":
            mu_tube = mu_lcb.copy()
            is_conf = True
        else:
            mu_tube = mu_mean.copy()
            is_conf = False

        clamped_frac = float(np.mean(mu_lcb <= self.mu_hard_min + 1e-9))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.last_predict_time_ms = elapsed_ms

        return {
            "mode": mode,
            "query_mode": qm,
            "is_conformal_safe": is_conf,
            "mu_mean": mu_mean,
            "sigma_gp": sigma_arr,
            "mu_lcb": mu_lcb,
            "mu_tube": mu_tube,
            "q_hat_mu": float(self.bg_model.q_bg),
            "alpha_mu": 0.05,
            "n_support": int(len(self.anomaly_map.cells)),
            "erosion_radius_m": 0.0,
            "use_raster_cache": False,
            "grid_res_m": float(self.anomaly_map.voxel_size),
            "prior_region_frac": 0.0,
            "clamped_mu_min_frac": round(clamped_frac, 4),
            "predict_time_ms": float(elapsed_ms),
            "mu_bg": float(self.bg_model.mu_bg),
            "mu_bg_safe": float(mu_bg_safe),
            "n_anomaly_cells": int(len(self.anomaly_map.cells)),
        }

    def update_online(
        self,
        x: float,
        y: float,
        yaw: float,
        W_actual_xy: np.ndarray,
        W_cmd_xy: np.ndarray,
        wheel_slips: np.ndarray,
        Fz_total: float,
        m_total: float,
        t_now: float,
        g: float = 9.81,
    ) -> dict:
        """
        Cập nhật trực tuyến 20 Hz: chạy FF-RLS thích nghi -> 2-Gate CUSUM -> tách biệt cập nhật
        Lớp A (nền Bayesian) và Lớp B (bảng băm dị thường 2D tại tọa độ từng bánh).
        """
        slips_arr = np.asarray(wheel_slips, dtype=float).ravel()
        if slips_arr.size == 1:
            slips_4 = np.full(4, float(slips_arr[0]), dtype=float)
        else:
            slips_4 = slips_arr[:4].astype(float)

        mu_rls, rho_k, bin_idx = self.rls_observer.update(
            W_actual_xy, W_cmd_xy, slips_4, float(Fz_total), float(m_total), g=float(g)
        )
        active_mask = (bin_idx >= 1) | (np.abs(slips_4) >= self.detector.slip_confirm)
        det = self.detector.step(self.bg_model.mu_bg, mu_rls, slips_4, active_mask)
        self.last_det_status = det

        c_y = math.cos(float(yaw))
        s_y = math.sin(float(yaw))
        t_cur = float(t_now)

        if det["any_anomaly"]:
            mu_rls_safe = float(np.clip(mu_rls - self.delta_mu_rls_margin, self.mu_hard_min, self.mu_env_high))
            delta_mu = max(0.0, self.bg_model.mu_bg - mu_rls_safe)
            for w in range(4):
                if det["is_anomaly"][w]:
                    bx, by = float(WHEEL_POS[w, 0]), float(WHEEL_POS[w, 1])
                    wx = float(x) + c_y * bx - s_y * by
                    wy = float(y) + s_y * bx + c_y * by
                    self.anomaly_map.record_anomaly(wx, wy, delta_mu, t_cur)
            self._active_anom_delta = max(self._active_anom_delta, delta_mu)
            self._active_anom_t = t_cur
        else:
            if (t_cur - self._active_anom_t) > self._active_anom_hold_s:
                self._active_anom_delta = 0.0
            if bin_idx >= 1:
                obs_var = float(np.clip(2.0 * self.rls_observer.P + 0.02 ** 2, 0.02 ** 2, 0.04))
                self.bg_model.update(mu_rls, obs_var, is_anomaly=False)
                for w in range(4):
                    if det["is_dry_confirmed"][w]:
                        bx, by = float(WHEEL_POS[w, 0]), float(WHEEL_POS[w, 1])
                        wx = float(x) + c_y * bx - s_y * by
                        wy = float(y) + s_y * bx + c_y * by
                        self.anomaly_map.confirm_dry(wx, wy)

        self.last_t_now = t_cur
        return {
            "mu_rls": float(mu_rls),
            "rho_k": float(rho_k),
            "bin_idx": int(bin_idx),
            "any_anomaly": bool(det["any_anomaly"]),
            "mu_bg": float(self.bg_model.mu_bg),
            "mu_bg_safe": float(self.bg_model.get_safe_bg()),
            "n_anomaly_cells": int(len(self.anomaly_map.cells)),
        }

    def fuse_horizon_xy(
        self,
        x_h: np.ndarray,
        y_h: np.ndarray,
        yaw_h: np.ndarray,
        s_h: np.ndarray,
        t_now: Optional[float] = None,
        s_curr: Optional[float] = None,
    ) -> np.ndarray:
        """
        Truy vấn ma sát hợp nhất dọc chân trời dự báo trên dấu chân K=6:
        mu_fuse(p_k, t) = clip(mu_bg_safe - max(delta_map(p_k, t), delta_now(s_q)), mu_hard_min, mu_env_high)
        """
        xh = np.asarray(x_h, dtype=float).ravel()
        yh = np.asarray(y_h, dtype=float).ravel()
        yawh = np.asarray(yaw_h, dtype=float).ravel()
        sh = np.asarray(s_h, dtype=float).ravel()
        n_h = len(sh)
        t_use = self.last_t_now if t_now is None else float(t_now)

        if self.query_mode == "footprint":
            fp_pts = np.asarray(
                compute_footprint_query_points(
                    xh, yh, yaw_grid=yawh,
                    L_x=self.L_x, L_y=self.L_y, lateral_margin=self.lateral_margin,
                ),
                dtype=float,
            )
            k_fp = fp_pts.shape[1]
            delta_fp = self.anomaly_map.query_points(fp_pts.reshape(-1, 2), t_use)
            delta_map_h = np.max(delta_fp.reshape(n_h, k_fp), axis=1)
        else:
            delta_map_h = self.anomaly_map.query_points(np.column_stack([xh, yh]), t_use)

        if self._active_anom_delta > 0.0 and (t_use - self._active_anom_t) <= self._active_anom_hold_s:
            s_0 = float(sh[0]) if s_curr is None else float(s_curr)
            ds = np.maximum(0.0, sh - s_0)
            delta_now_h = self._active_anom_delta * np.exp(-ds / self.L_corr_now)
        else:
            delta_now_h = np.zeros(n_h, dtype=float)

        mu_bg_safe = self.bg_model.get_safe_bg()
        mu_fuse = mu_bg_safe - np.maximum(delta_map_h, delta_now_h)
        return np.clip(mu_fuse, self.mu_hard_min, self.mu_env_high)

    def reset(self, keep_anomaly_map: bool = False) -> None:
        self.detector.reset()
        self.rls_observer.reset(mu_init=self.bg_model.get_safe_bg())
        self._active_anom_delta = 0.0
        self._active_anom_t = -1e9
        if not keep_anomaly_map:
            self.bg_model.reset()
            self.anomaly_map.reset()
            self.rls_observer.reset(mu_init=self.bg_model.get_safe_bg())
            self.last_t_now = 0.0
