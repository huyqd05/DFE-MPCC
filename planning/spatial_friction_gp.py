# -*- coding: utf-8 -*-
"""
spatial_friction_gp.py
=============================================================================
MÔ HÌNH TRƯỜNG MA SÁT KHÔNG GIAN GAUSSIAN PROCESS + HIỆU CHUẨN CONFORMAL
CẤP QUỸ ĐẠO (SIM-ONLY PRE-DEPARTURE FRICTION FIELD ESTIMATOR)

Định hướng 1 & 4 trong kiến trúc Sim-Only:
1. MultiZoneFrictionGroundTruth:
   Môi trường mô phỏng đa vùng ma sát (bê tông khô, hành lang epoxy, vũng dầu)
   kết hợp nhiễu nhám vi mô 2D chuẩn ISO 8608:2016 Class B (tách biệt hoàn toàn
   với bộ ước lượng). Hỗ trợ tính ma sát 4 bánh độc lập & split-mu.
2. collect_ffrls_history_from_rollouts & aggregate_observations_to_voxels:
   Thu thập dữ liệu nhân quả từ các lượt chạy quá khứ qua bộ quan sát
   FrictionObserverFFRLS (chỉ nhận mẫu kích thích bin_idx >= 1, có trễ và
   nhiễu đo dị phương sai gắn với hiệp phương sai P của RLS - không lộ ground truth).
3. SpatialFrictionGP:
   - Kernel tổng hợp: Matérn-3/2 (thang vĩ mô C^1, hỗ trợ làm trơn ống vận tốc)
     + RBF (thang vi mô ISO 8608).
   - Truy vấn tại t = 0 dọc toàn bộ quỹ đạo A* trong vài mili-giây:
     trả về (mu_mean, sigma_gp, mu_lcb).
   - Hiệu chuẩn Conformal cấp quỹ đạo (Split Conformal Supremum Score) bao trọn
     cả hiệu ứng split-mu 4 bánh:
       R_j = max_s max(0, mu_mean(s) - min_{w=1..4} mu_true(p_w(s))) / (sigma_gp(s) + sigma_0)
   - Xuất/nạp nhanh cấu hình + điểm hỗ trợ JSON (config/conformal_calibration_mu.json).
=============================================================================
"""

import os
import sys
import json
import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from scipy.stats import binom, beta

_curr = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_curr) if os.path.basename(_curr) in ["planning", "controller", "solver", "simulation", "tests"] else _curr
for _p in [_root, os.path.join(_root, "controller"), os.path.join(_root, "planning"), os.path.join(_root, "solver")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from controller.iso8608_road_profile import ISO8608RoadProfile
from controller.friction_observer_ffrls import FrictionObserverFFRLS
from controller.mecanum_traction_dynamics import L_X, L_Y, M_TOTAL, G


def default_warehouse_macro_friction(x: float, y: float) -> float:
    """
    Hàm phân vùng ma sát vĩ mô mặc định cho sàn kho công nghiệp (trơn cục bộ mượt):
    - Nền bê tông khô: mu = 0.35
    - Hành lang epoxy bóng (khu trung tâm): mu = 0.25
    - Vũng dầu/nước tràn: mu = 0.13
    """
    # Vùng epoxy quanh dải x in [-1.2, 0.8], y in [-0.8, 1.5]
    wx_ep = 0.5 * (np.tanh((x + 1.2) / 0.12) - np.tanh((x - 0.8) / 0.12))
    wy_ep = 0.5 * (np.tanh((y + 0.8) / 0.12) - np.tanh((y - 1.5) / 0.12))
    w_epoxy = float(np.clip(wx_ep * wy_ep, 0.0, 1.0))

    # Vũng dầu tại tâm (x0=0.45, y0=-0.55) bán kính hiệu dụng ~0.70 m
    r2_oil = ((x - 0.45) / 0.70) ** 2 + ((y + 0.55) / 0.55) ** 2
    w_oil = float(np.exp(-1.5 * r2_oil))

    mu_base = 0.35 - (0.35 - 0.25) * w_epoxy
    mu_val = mu_base * (1.0 - w_oil) + 0.13 * w_oil
    return float(np.clip(mu_val, 0.08, 0.60))


def benchmark_curve_macro_friction(x: float, y: float) -> float:
    """
    Hàm phân vùng ma sát vĩ mô 2D tương thích với đường cong chuẩn 7.42 m (x in [0, 6.75]):
    - Sàn bê tông khô danh định: mu = 0.35
    - Đoạn hành lang epoxy chuyển tiếp quanh CP1/CP2 (x ~ 1.4..2.3): mu = 0.26
    - Vùng sàn ướt/dầu quanh CP3 (tâm x=3.15, y=-0.43): mu = 0.13
    """
    # Vùng epoxy nhẹ quanh khúc cua 1 -> sườn S
    wx_ep = 0.5 * (np.tanh((x - 1.15) / 0.10) - np.tanh((x - 2.30) / 0.10))
    wy_ep = 0.5 * (np.tanh((y + 0.35) / 0.12) - np.tanh((y - 0.90) / 0.12))
    w_epoxy = float(np.clip(wx_ep * wy_ep, 0.0, 1.0))

    # Vùng dầu/ướt quanh CP3 (bao phủ toàn bộ dải ngang khúc cua 2 quanh x in [2.6, 3.9])
    wx_oil = 0.5 * (np.tanh((x - 2.65) / 0.09) - np.tanh((x - 3.95) / 0.09))
    wy_oil = 0.5 * (np.tanh((y + 0.95) / 0.12) - np.tanh((y - 0.35) / 0.12))
    w_oil = float(np.clip(wx_oil * wy_oil, 0.0, 1.0))

    mu_base = 0.35 - (0.35 - 0.26) * w_epoxy
    mu_val = mu_base * (1.0 - w_oil) + 0.13 * w_oil
    return float(np.clip(mu_val, 0.08, 0.60))


class MultiZoneFrictionGroundTruth:
    """
    Trường ma sát thực tế (Ground Truth) cho mô phỏng Sim-Only:
    Kết hợp phân vùng ma sát vĩ mô 2D và nhiễu độ nhám không gian vi mô ISO 8608:2016.
    """

    def __init__(
        self,
        macro_fn=None,
        road_class: str = "B",
        seed: int = 42,
        variation_scale: float = 0.15,
        mu_clip: tuple = (0.08, 0.60),
    ):
        self.macro_fn = default_warehouse_macro_friction if macro_fn is None else macro_fn
        self.road_class = str(road_class)
        self.seed = int(seed)
        self.variation_scale = float(variation_scale)
        self.mu_clip = (float(mu_clip[0]), float(mu_clip[1]))
        self.road_profile = ISO8608RoadProfile(road_class=self.road_class, seed=self.seed)

    def eval_center(self, x: float, y: float) -> float:
        """Trả về hệ số ma sát cục bộ tại điểm tiếp xúc (x, y)."""
        mu_loc = self.road_profile.compute_local_friction(
            float(x),
            float(y),
            mu_nominal=self.macro_fn,
            variation_scale=self.variation_scale,
        )
        return float(np.clip(mu_loc, self.mu_clip[0], self.mu_clip[1]))

    def eval_four_wheels(
        self,
        x: float,
        y: float,
        yaw: float,
        L_x: float = L_X,
        L_y: float = L_Y,
    ) -> tuple:
        """
        Tính hệ số ma sát thực tế dưới 4 bánh xe Mecanum [FL, FR, RR, RL].
        Returns: (mu_min_4w, mu_wheels, split_mu_delta)
        """
        c_y = float(np.cos(yaw))
        s_y = float(np.sin(yaw))
        offsets = (
            (+L_x, +L_y),  # FL
            (+L_x, -L_y),  # FR
            (-L_x, -L_y),  # RR
            (-L_x, +L_y),  # RL
        )
        mu_wheels = np.empty(4, dtype=float)
        for idx, (bx, by) in enumerate(offsets):
            wx = float(x) + c_y * bx - s_y * by
            wy = float(y) + s_y * bx + c_y * by
            mu_wheels[idx] = self.eval_center(wx, wy)

        mu_min_4w = float(np.min(mu_wheels))
        split_delta = float(np.max(mu_wheels) - mu_min_4w)
        return mu_min_4w, mu_wheels, split_delta

    def eval_path_4w_min(
        self,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        L_x: float = L_X,
        L_y: float = L_Y,
    ) -> tuple:
        """
        Đánh giá dọc một quỹ đạo rời rạc (x_grid, y_grid) với hướng tiếp tuyến yaw(s).
        Returns: (mu_center_arr, mu_min_4w_arr, split_delta_arr)
        """
        x_arr = np.asarray(x_grid, dtype=float)
        y_arr = np.asarray(y_grid, dtype=float)
        n = len(x_arr)
        if n == 0:
            return np.zeros(0), np.zeros(0), np.zeros(0)

        if n >= 2:
            dx = np.gradient(x_arr)
            dy = np.gradient(y_arr)
            yaw_arr = np.arctan2(dy, dx)
        else:
            yaw_arr = np.zeros(1, dtype=float)

        mu_center = np.empty(n, dtype=float)
        mu_min_4w = np.empty(n, dtype=float)
        split_delta = np.empty(n, dtype=float)
        for i in range(n):
            mu_center[i] = self.eval_center(x_arr[i], y_arr[i])
            m_min, _, s_del = self.eval_four_wheels(x_arr[i], y_arr[i], yaw_arr[i], L_x=L_x, L_y=L_y)
            mu_min_4w[i] = m_min
            split_delta[i] = s_del
        return mu_center, mu_min_4w, split_delta


def make_benchmark_curve_ground_truth(
    seed: int = 42,
    road_class: str = "B",
    variation_scale: float = 0.15,
    mu_clip: tuple = (0.08, 0.60),
) -> MultiZoneFrictionGroundTruth:
    """Tạo trường ma sát ground truth chuẩn khớp kịch bản chữ S 7.42 m."""
    return MultiZoneFrictionGroundTruth(
        macro_fn=benchmark_curve_macro_friction,
        road_class=road_class,
        seed=seed,
        variation_scale=variation_scale,
        mu_clip=mu_clip,
    )


def aggregate_observations_to_voxels(
    points_xy: np.ndarray,
    mu_obs: np.ndarray,
    obs_var: np.ndarray = None,
    voxel_size: float = 0.15,
    min_var_floor: float = 4.0e-4,
) -> tuple:
    """
    Gom cụm mẫu quan sát vào lưới voxel không gian theo trung bình trọng số nghịch đảo phương sai.
    Giữ kích thước điểm hỗ trợ GP gọn (M <= ~400) và chặn dưới phương sai để tránh tự tin thái quá
    do các mẫu gần nhau cùng chịu một độ nhám ISO 8608.
    """
    pts = np.asarray(points_xy, dtype=float)
    y = np.asarray(mu_obs, dtype=float).ravel()
    if pts.ndim != 2 or pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=float), np.zeros(0, dtype=float), np.zeros(0, dtype=float)

    if obs_var is None:
        var = np.full(len(y), max(min_var_floor, 1e-3), dtype=float)
    else:
        var = np.maximum(np.asarray(obs_var, dtype=float).ravel(), min_var_floor)

    if voxel_size is None or voxel_size <= 0.0:
        return pts.copy(), y.copy(), var.copy()

    cells = np.floor(pts / float(voxel_size)).astype(int)
    buckets = {}
    for idx in range(len(y)):
        key = (int(cells[idx, 0]), int(cells[idx, 1]))
        if key not in buckets:
            buckets[key] = []
        buckets[key].append(idx)

    agg_pts = []
    agg_mu = []
    agg_var = []
    for idxs in buckets.values():
        idx_arr = np.asarray(idxs, dtype=int)
        v_sub = var[idx_arr]
        w = 1.0 / np.maximum(v_sub, 1e-8)
        w_sum = float(np.sum(w))
        p_cell = np.sum(pts[idx_arr] * w[:, None], axis=0) / w_sum
        mu_cell = float(np.sum(y[idx_arr] * w) / w_sum)
        # Không cho phương sai sau gom cụm sụt giảm quá 0.5*min(v_sub) nhằm bảo toàn bất định vi mô nội ô
        var_cell = max(1.0 / w_sum, 0.5 * float(np.min(v_sub)), float(min_var_floor))
        agg_pts.append(p_cell)
        agg_mu.append(mu_cell)
        agg_var.append(var_cell)

    return np.asarray(agg_pts, dtype=float), np.asarray(agg_mu, dtype=float), np.asarray(agg_var, dtype=float)


def collect_ffrls_history_from_rollouts(
    ground_truth: MultiZoneFrictionGroundTruth,
    rollout_trajectories: list,
    m_total: float = M_TOTAL,
    g: float = G,
    s_char: float = 0.04,
    noise_std_rel: float = 0.025,
    c_P_var: float = 2.0,
    sigma_obs_floor: float = 0.020,
    lam_rls: float = 0.80,
    seed: int = 0,
) -> dict:
    """
    Thu thập tập lịch sử nhân quả D_hist thông qua bộ quan sát FrictionObserverFFRLS
    trên các lượt chạy quá khứ (KHÔNG lộ ground truth cho bộ ước lượng).

    Mỗi mẫu chỉ được ghi nhận khi kích thích vượt ngưỡng quan sát (bin_idx >= 1,
    tức rho_k >= 0.08 và max_slip >= 0.02), với phương sai dị phương sai gắn với P của RLS.
    """
    rng = np.random.default_rng(seed)
    pts_list = []
    mu_hat_list = []
    var_list = []
    n_total_steps = 0
    n_deadzone_skipped = 0

    for traj in rollout_trajectories:
        if isinstance(traj, dict):
            x_arr = np.asarray(traj["x_grid"], dtype=float)
            y_arr = np.asarray(traj["y_grid"], dtype=float)
            yaw_arr = (
                np.asarray(traj["yaw_grid"], dtype=float)
                if "yaw_grid" in traj
                else np.arctan2(np.gradient(y_arr), np.gradient(x_arr))
            )
            rho_arr = np.asarray(traj["rho_grid"], dtype=float) if "rho_grid" in traj else None
            slip_arr = np.asarray(traj["slip_grid"], dtype=float) if "slip_grid" in traj else None
        else:
            wp = np.asarray(traj, dtype=float)
            x_arr = wp[:, 0]
            y_arr = wp[:, 1]
            yaw_arr = np.arctan2(np.gradient(y_arr), np.gradient(x_arr))
            rho_arr = None
            slip_arr = None

        n_pts = len(x_arr)
        if n_pts < 3:
            continue

        # Tính độ cong cục bộ để mô phỏng kích thích tự nhiên ở đoạn cua & đoạn tăng/giảm tốc
        dx = np.gradient(x_arr)
        dy = np.gradient(y_arr)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        sp = np.maximum(np.hypot(dx, dy), 1e-6)
        kappa = np.abs(dx * ddy - dy * ddx) / (sp ** 3)

        observer = FrictionObserverFFRLS(
            mu_init=0.25,
            mu_min=0.06,
            mu_max=0.60,
            lam=lam_rls,
            P_init=10.0,
            rho_deadzone=0.06,
        )

        fz_nom_total = m_total * g
        for i in range(n_pts):
            n_total_steps += 1
            if rho_arr is not None:
                rho_cmd = float(rho_arr[i])
            else:
                # Gia tốc ở đầu/cuối tuyến hoặc vùng cua; đoạn thẳng giữa có thể rơi vào vùng chết
                frac = i / max(n_pts - 1, 1)
                if frac < 0.14 or frac > 0.86:
                    a_tot = 0.85 + 0.25 * rng.uniform()
                else:
                    a_lat = (0.58 ** 2) * float(kappa[i])
                    a_probe = 0.72 if (i % 7 < 4) else 0.18
                    a_tot = float(np.hypot(a_probe, a_lat))
                rho_cmd = a_tot / g

            if slip_arr is not None:
                max_slip = float(slip_arr[i])
            else:
                if rho_cmd >= 0.08:
                    max_slip = float(np.clip(0.022 + 0.18 * (rho_cmd - 0.06) + rng.normal(0.0, 0.004), 0.018, 0.09))
                else:
                    max_slip = float(np.clip(0.006 + 0.05 * rho_cmd, 0.002, 0.014))

            # Tải pháp tuyến thực bị dao động nhẹ bởi gập ghềnh ISO 8608
            fz_total_meas = float(fz_nom_total + rng.normal(0.0, 1.8))
            fz_total_meas = max(0.5 * fz_nom_total, fz_total_meas)

            # Lực bám thực tế sinh ra tại tiếp xúc 4 bánh dưới ma sát cục bộ + nhiễu cảm biến lực
            _, mu_wheels_true, _ = ground_truth.eval_four_wheels(x_arr[i], y_arr[i], yaw_arr[i])
            mu_eff_true = float(np.mean(mu_wheels_true))
            phi_true = fz_total_meas * np.tanh(max_slip / s_char)
            y_meas = max(0.0, mu_eff_true * phi_true * (1.0 + rng.normal(0.0, noise_std_rel)))

            w_cmd_xy = np.array([rho_cmd * fz_nom_total, 0.0])
            w_act_xy = np.array([y_meas, 0.0])
            wheel_slips = np.full(4, max_slip, dtype=float)

            mu_hat, _, bin_idx = observer.update(
                W_actual_xy=w_act_xy,
                W_cmd_xy=w_cmd_xy,
                wheel_slip_ratios=wheel_slips,
                Fz_total=fz_total_meas,
                m_total=m_total,
                g=g,
            )

            if bin_idx >= 1:
                obs_var_i = float(np.clip(c_P_var * observer.P + sigma_obs_floor ** 2, sigma_obs_floor ** 2, 0.04))
                pts_list.append([float(x_arr[i]), float(y_arr[i])])
                mu_hat_list.append(float(mu_hat))
                var_list.append(obs_var_i)
            else:
                n_deadzone_skipped += 1

    if len(pts_list) == 0:
        return {
            "points_xy": np.zeros((0, 2), dtype=float),
            "mu_obs": np.zeros(0, dtype=float),
            "obs_var": np.zeros(0, dtype=float),
            "n_total_steps": n_total_steps,
            "n_accepted": 0,
            "n_deadzone_skipped": n_deadzone_skipped,
        }

    return {
        "points_xy": np.asarray(pts_list, dtype=float),
        "mu_obs": np.asarray(mu_hat_list, dtype=float),
        "obs_var": np.asarray(var_list, dtype=float),
        "n_total_steps": n_total_steps,
        "n_accepted": len(pts_list),
        "n_deadzone_skipped": n_deadzone_skipped,
    }


class SpatialFrictionGP:
    """
    Bộ ước lượng trường ma sát không gian 2D bằng Heteroscedastic Gaussian Process
    (Matérn-3/2 vĩ mô + RBF vi mô) kèm Hiệu chuẩn Conformal cấp quỹ đạo (LCB).
    """

    def __init__(
        self,
        mu_prior: float = 0.25,
        sigma_f: float = 0.12,
        ell_macro: float = 0.85,
        sigma_micro: float = 0.045,
        ell_micro: float = 0.22,
        sigma_obs_floor: float = 0.020,
        mu_min: float = 0.08,
        mu_max: float = 0.55,
        sigma_0: float = 0.015,
        q_hat_mu: float = 1.96,
        alpha_mu: float = 0.05,
    ):
        self.mu_prior = float(mu_prior)
        self.sigma_f = float(sigma_f)
        self.ell_macro = float(ell_macro)
        self.sigma_micro = float(sigma_micro)
        self.ell_micro = float(ell_micro)
        self.sigma_obs_floor = float(sigma_obs_floor)
        self.mu_min = float(mu_min)
        self.mu_max = float(mu_max)
        self.sigma_0 = float(sigma_0)
        self.q_hat_mu = float(q_hat_mu)
        self.alpha_mu = float(alpha_mu)

        self.X_train = np.zeros((0, 2), dtype=float)
        self.y_train = np.zeros(0, dtype=float)
        self.obs_var_train = np.zeros(0, dtype=float)
        self._cho_factor = None
        self._alpha_vec = None
        self._K_inv = None
        self.last_fit_time_ms = 0.0
        self.last_predict_time_ms = 0.0
        self.cal_meta = {}

    def _compute_kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Tính ma trận hiệp phương sai K(X1, X2) = K_matern32_macro + K_rbf_micro bằng C-cdist."""
        dist = cdist(X1, X2, metric="euclidean")

        # Macro Matérn-3/2 (đảm bảo liên tục khả vi C^1 cho đường ống vận tốc)
        arg_m = (np.sqrt(3.0) / max(self.ell_macro, 1e-6)) * dist
        k_macro = (self.sigma_f ** 2) * (1.0 + arg_m) * np.exp(-arg_m)

        # Micro RBF (nắm bắt tương quan độ nhám không gian vi mô ISO 8608)
        inv_2ell2 = -0.5 / (max(self.ell_micro, 1e-6) ** 2)
        k_micro = (self.sigma_micro ** 2) * np.exp(inv_2ell2 * (dist * dist))
        return k_macro + k_micro

    def _kernel_diag(self, n: int) -> np.ndarray:
        return np.full(n, (self.sigma_f ** 2) + (self.sigma_micro ** 2), dtype=float)

    def fit(
        self,
        points_xy: np.ndarray,
        mu_obs: np.ndarray,
        obs_var: np.ndarray = None,
        voxel_size: float = 0.15,
    ) -> "SpatialFrictionGP":
        """Huấn luyện posterior GP bằng phân rã Cholesky trên tập dữ liệu (có gom cụm voxel)."""
        t0 = time.perf_counter()
        pts = np.asarray(points_xy, dtype=float)
        y = np.asarray(mu_obs, dtype=float).ravel()

        if pts.ndim != 2 or pts.shape[0] == 0 or len(y) == 0:
            self.X_train = np.zeros((0, 2), dtype=float)
            self.y_train = np.zeros(0, dtype=float)
            self.obs_var_train = np.zeros(0, dtype=float)
            self._cho_factor = None
            self._alpha_vec = None
            self._K_inv = None
            self.last_fit_time_ms = (time.perf_counter() - t0) * 1000.0
            return self

        if pts.shape[1] != 2:
            raise ValueError(f"points_xy phải có shape (M, 2), nhận được {pts.shape}")

        var_floor = self.sigma_obs_floor ** 2
        if obs_var is None:
            var = np.full(len(y), var_floor, dtype=float)
        else:
            var = np.maximum(np.asarray(obs_var, dtype=float).ravel(), var_floor)

        if voxel_size is not None and voxel_size > 0.0:
            pts, y, var = aggregate_observations_to_voxels(
                pts, y, var, voxel_size=voxel_size, min_var_floor=var_floor
            )

        self.X_train = np.asarray(pts, dtype=float)
        self.y_train = np.asarray(y, dtype=float)
        self.obs_var_train = np.maximum(np.asarray(var, dtype=float), var_floor)

        K_mm = self._compute_kernel_matrix(self.X_train, self.X_train)
        diag_idx = np.diag_indices_from(K_mm)
        K_mm[diag_idx] += self.obs_var_train + 1e-6

        self._cho_factor = cho_factor(K_mm, lower=True, check_finite=False)
        self._alpha_vec = cho_solve(self._cho_factor, self.y_train - self.mu_prior, check_finite=False)
        self._K_inv = cho_solve(self._cho_factor, np.eye(len(self.X_train)), check_finite=False)
        self.last_fit_time_ms = (time.perf_counter() - t0) * 1000.0
        return self

    def predict(self, points_xy: np.ndarray) -> tuple:
        """
        Dự báo vector hóa tại tập tọa độ truy vấn (N, 2).
        Returns: (mu_mean, sigma_gp, mu_lcb)
        """
        t0 = time.perf_counter()
        X_q = np.asarray(points_xy, dtype=float)
        if X_q.ndim == 1:
            X_q = X_q.reshape(1, 2)
        n_q = X_q.shape[0]
        if n_q == 0:
            self.last_predict_time_ms = (time.perf_counter() - t0) * 1000.0
            return np.zeros(0), np.zeros(0), np.zeros(0)

        prior_std = float(np.sqrt((self.sigma_f ** 2) + (self.sigma_micro ** 2)))
        if self._cho_factor is None or len(self.X_train) == 0:
            mu_mean = np.full(n_q, self.mu_prior, dtype=float)
            sigma_gp = np.full(n_q, prior_std, dtype=float)
        else:
            K_qs = self._compute_kernel_matrix(X_q, self.X_train)
            mu_mean = self.mu_prior + K_qs @ self._alpha_vec
            var_red = np.sum((K_qs @ self._K_inv) * K_qs, axis=1)
            var_post = np.maximum(1e-10, self._kernel_diag(n_q) - var_red)
            sigma_gp = np.sqrt(var_post)

        mu_lcb = np.clip(
            mu_mean - self.q_hat_mu * (sigma_gp + self.sigma_0),
            self.mu_min,
            self.mu_max,
        )
        self.last_predict_time_ms = (time.perf_counter() - t0) * 1000.0
        return mu_mean, sigma_gp, mu_lcb

    def predict_along_path(
        self,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        mode: str = "conformal_lcb",
    ) -> dict:
        """
        Truy vấn trường ma sát dọc lưới quỹ đạo A* (x_grid, y_grid) tại t = 0.
        mode:
          - 'conformal_lcb': trả về cận dưới Conformal mu_lcb (an toàn mặc định)
          - 'gp_mean': chỉ dùng trung bình hậu nghiệm GP cắt ngưỡng [mu_min, mu_max] (đối chứng M1)
        """
        if mode not in ("conformal_lcb", "gp_mean"):
            raise ValueError(f"mode không hợp lệ: '{mode}'. Chọn 'conformal_lcb' hoặc 'gp_mean'.")

        pts = np.column_stack([np.asarray(x_grid, dtype=float), np.asarray(y_grid, dtype=float)])
        mu_mean, sigma_gp, mu_lcb = self.predict(pts)
        mu_mean_clipped = np.clip(mu_mean, self.mu_min, self.mu_max)

        is_safe = (mode == "conformal_lcb")
        mu_tube = mu_lcb if is_safe else mu_mean_clipped

        return {
            "mu_tube": mu_tube,
            "mu_mean": mu_mean_clipped,
            "mu_mean_raw": mu_mean,
            "sigma_gp": sigma_gp,
            "mu_lcb": mu_lcb,
            "is_conformal_safe": is_safe,
            "mode": mode,
            "q_hat_mu": float(self.q_hat_mu),
            "alpha_mu": float(self.alpha_mu),
            "sigma_0": float(self.sigma_0),
            "n_support": int(len(self.X_train)),
            "predict_time_ms": float(self.last_predict_time_ms),
        }

    def calibrate_trajectory_conformal(
        self,
        cal_trajectories: list,
        alpha_mu: float = None,
    ) -> dict:
        """
        Hiệu chuẩn Split Conformal cấp quỹ đạo (1 điểm supremum chuẩn hóa / quỹ đạo)
        bao trọn ma sát cực tiểu 4 bánh xe:
            R_j = max_i max(0, mu_mean[i] - mu_min_4w_true[i]) / (sigma_gp[i] + sigma_0)
        """
        alpha_use = self.alpha_mu if alpha_mu is None else float(alpha_mu)
        n_cal = len(cal_trajectories)
        if n_cal == 0:
            raise ValueError("cal_trajectories rỗng, không thể hiệu chuẩn Conformal!")

        scores = np.empty(n_cal, dtype=float)
        for j, traj in enumerate(cal_trajectories):
            pts = np.column_stack([
                np.asarray(traj["x_grid"], dtype=float),
                np.asarray(traj["y_grid"], dtype=float),
            ])
            mu_min_4w_true = np.asarray(traj["mu_min_4w_true"], dtype=float).ravel()
            mu_mean, sigma_gp, _ = self.predict(pts)
            norm_err = np.maximum(0.0, mu_mean - mu_min_4w_true) / (sigma_gp + self.sigma_0)
            scores[j] = float(np.max(norm_err)) if norm_err.size > 0 else 0.0

        scores_sorted = np.sort(scores)
        k_star = int(np.clip(np.ceil((n_cal + 1) * (1.0 - alpha_use)), 1, n_cal))
        q_hat = float(scores_sorted[k_star - 1])

        # Kiểm tra tự tương quan bậc 1 (Lag-1 autocorrelation) giữa các quỹ đạo hiệu chuẩn
        if n_cal >= 4 and np.std(scores) > 1e-12:
            s_c = scores - np.mean(scores)
            r1 = float(np.sum(s_c[:-1] * s_c[1:]) / np.sum(s_c ** 2))
            se_r1 = float(1.0 / np.sqrt(n_cal))
            z_r1 = abs(r1) / se_r1
        else:
            r1 = 0.0
            se_r1 = 1.0
            z_r1 = 0.0

        self.q_hat_mu = q_hat
        self.alpha_mu = alpha_use
        self.cal_meta = {
            "method": "Trajectory-Level Split Conformal (4-Wheel Min Supremum)",
            "n_cal": int(n_cal),
            "alpha_mu": float(alpha_use),
            "target_coverage": float(1.0 - alpha_use),
            "k_star": int(k_star),
            "q_hat_mu": round(q_hat, 6),
            "score_mean": round(float(np.mean(scores)), 6),
            "score_median": round(float(np.median(scores)), 6),
            "score_max": round(float(np.max(scores)), 6),
            "lag1_autocorr": round(r1, 5),
            "lag1_z_stat": round(z_r1, 4),
            "exchangeability_ok": bool(z_r1 < 1.96),
            "scores": [round(float(s), 6) for s in scores.tolist()],
        }
        return self.cal_meta

    def evaluate_coverage(
        self,
        test_trajectories: list,
        mode: str = "conformal_lcb",
        mu_override_fn=None,
    ) -> dict:
        """
        Đánh giá tỷ lệ phủ cấp quỹ đạo & cấp điểm trên tập test độc lập D_test.
        Nếu mu_override_fn được truyền vào (ví dụ cho baseline hằng M0), dùng giá trị đó làm mu_tube.
        """
        n_test = len(test_trajectories)
        if n_test == 0:
            raise ValueError("test_trajectories rỗng!")

        traj_covered = []
        point_cov_fracs = []
        worst_overest_list = []
        mean_gap_list = []
        max_gap_list = []
        pred_times = []

        for traj in test_trajectories:
            x_g = np.asarray(traj["x_grid"], dtype=float)
            y_g = np.asarray(traj["y_grid"], dtype=float)
            mu_true_4w = np.asarray(traj["mu_min_4w_true"], dtype=float).ravel()

            if mu_override_fn is not None:
                t0 = time.perf_counter()
                mu_tube = np.asarray(mu_override_fn(x_g, y_g), dtype=float).ravel()
                pred_times.append((time.perf_counter() - t0) * 1000.0)
            else:
                res = self.predict_along_path(x_g, y_g, mode=mode)
                mu_tube = res["mu_tube"]
                pred_times.append(res["predict_time_ms"])

            diff = mu_true_4w - mu_tube  # >= 0 nghĩa là an toàn (mu_tube <= mu_true_4w)
            is_cov_pts = (diff >= -1e-9)
            traj_covered.append(bool(np.all(is_cov_pts)))
            point_cov_fracs.append(float(np.mean(is_cov_pts)))
            worst_overest_list.append(float(np.max(np.maximum(0.0, mu_tube - mu_true_4w))))
            mean_gap_list.append(float(np.mean(diff)))
            max_gap_list.append(float(np.max(diff)))

        n_cov = int(np.sum(traj_covered))
        traj_cov_rate = n_cov / float(n_test)
        pt_cov_rate = float(np.mean(point_cov_fracs))

        # Khoảng tin cậy một phía dưới Clopper-Pearson (95%) & p-value Binomial
        clopper_95_lb = float(beta.ppf(0.05, n_cov, n_test - n_cov + 1)) if n_cov > 0 else 0.0
        binom_p_one_sided = float(binom.cdf(n_cov, n_test, 1.0 - self.alpha_mu))

        return {
            "mode": mode if mu_override_fn is None else "override",
            "n_test": int(n_test),
            "n_traj_covered": n_cov,
            "traj_coverage_rate": round(traj_cov_rate, 4),
            "pointwise_coverage_rate": round(pt_cov_rate, 4),
            "clopper_pearson_95_lb": round(clopper_95_lb, 4),
            "binom_p_value_one_sided": round(binom_p_one_sided, 4),
            "mean_worst_overestimation": round(float(np.mean(worst_overest_list)), 5),
            "max_worst_overestimation": round(float(np.max(worst_overest_list)), 5),
            "mean_conservatism_gap": round(float(np.mean(mean_gap_list)), 5),
            "max_conservatism_gap": round(float(np.max(max_gap_list)), 5),
            "mean_predict_time_ms": round(float(np.mean(pred_times)), 4),
        }

    def save_calibration(self, path: str, include_support: bool = True) -> None:
        """Lưu tham số kernel, lượng tử Conformal và tùy chọn tập hỗ trợ GP ra JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "version": "1.0",
            "hyperparams": {
                "mu_prior": self.mu_prior,
                "sigma_f": self.sigma_f,
                "ell_macro": self.ell_macro,
                "sigma_micro": self.sigma_micro,
                "ell_micro": self.ell_micro,
                "sigma_obs_floor": self.sigma_obs_floor,
                "mu_min": self.mu_min,
                "mu_max": self.mu_max,
                "sigma_0": self.sigma_0,
            },
            "q_hat_mu": float(self.q_hat_mu),
            "alpha_mu": float(self.alpha_mu),
            "n_support": int(len(self.X_train)),
            "calibration_summary": {k: v for k, v in self.cal_meta.items() if k != "scores"},
        }
        if include_support and len(self.X_train) > 0:
            payload["support"] = {
                "X_train": self.X_train.tolist(),
                "y_train": self.y_train.tolist(),
                "obs_var_train": self.obs_var_train.tolist(),
            }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_calibration(cls, path: str) -> "SpatialFrictionGP":
        """Nạp lại mô hình GP + lượng tử Conformal từ JSON trong < 2 ms."""
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        hp = payload.get("hyperparams", {})
        gp = cls(
            mu_prior=hp.get("mu_prior", 0.25),
            sigma_f=hp.get("sigma_f", 0.12),
            ell_macro=hp.get("ell_macro", 0.85),
            sigma_micro=hp.get("sigma_micro", 0.045),
            ell_micro=hp.get("ell_micro", 0.22),
            sigma_obs_floor=hp.get("sigma_obs_floor", 0.020),
            mu_min=hp.get("mu_min", 0.08),
            mu_max=hp.get("mu_max", 0.55),
            sigma_0=hp.get("sigma_0", 0.015),
            q_hat_mu=payload.get("q_hat_mu", 1.96),
            alpha_mu=payload.get("alpha_mu", 0.05),
        )
        gp.cal_meta = payload.get("calibration_summary", {})
        sup = payload.get("support")
        if sup is not None and len(sup.get("X_train", [])) > 0:
            gp.fit(
                points_xy=np.asarray(sup["X_train"], dtype=float),
                mu_obs=np.asarray(sup["y_train"], dtype=float),
                obs_var=np.asarray(sup["obs_var_train"], dtype=float),
                voxel_size=0.0,  # Đã gom cụm voxel sẵn trước khi lưu
            )
        return gp
