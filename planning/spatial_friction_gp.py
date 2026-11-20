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
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.ndimage import minimum_filter, maximum_filter
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


def default_warehouse_macro_friction(x, y):
    """
    Hàm phân vùng ma sát vĩ mô mặc định cho sàn kho công nghiệp (trơn cục bộ mượt):
    Hỗ trợ cả đầu vào vô hướng lẫn mảng NumPy để tính nhanh theo lô.
    - Nền bê tông khô: mu = 0.35
    - Hành lang epoxy bóng (khu trung tâm): mu = 0.25
    - Vũng dầu/nước tràn: mu = 0.13
    """
    is_scalar = np.isscalar(x) and np.isscalar(y)
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)

    wx_ep = 0.5 * (np.tanh((xa + 1.2) / 0.12) - np.tanh((xa - 0.8) / 0.12))
    wy_ep = 0.5 * (np.tanh((ya + 0.8) / 0.12) - np.tanh((ya - 1.5) / 0.12))
    w_epoxy = np.clip(wx_ep * wy_ep, 0.0, 1.0)

    r2_oil = ((xa - 0.45) / 0.70) ** 2 + ((ya + 0.55) / 0.55) ** 2
    w_oil = np.exp(-1.5 * r2_oil)

    mu_base = 0.35 - (0.35 - 0.25) * w_epoxy
    mu_val = np.clip(mu_base * (1.0 - w_oil) + 0.13 * w_oil, 0.08, 0.60)
    return float(mu_val) if is_scalar else mu_val


def benchmark_curve_macro_friction(x, y):
    """
    Hàm phân vùng ma sát vĩ mô 2D tương thích với đường cong chuẩn 7.42 m (x in [0, 6.75]):
    Hỗ trợ cả đầu vào vô hướng lẫn mảng NumPy.
    - Sàn bê tông khô danh định: mu = 0.35
    - Đoạn hành lang epoxy chuyển tiếp quanh CP1/CP2 (x ~ 1.4..2.3): mu = 0.26
    - Vùng sàn ướt/dầu quanh CP3 (tâm x=3.15, y=-0.43): mu = 0.13
    """
    is_scalar = np.isscalar(x) and np.isscalar(y)
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)

    wx_ep = 0.5 * (np.tanh((xa - 1.15) / 0.10) - np.tanh((xa - 2.30) / 0.10))
    wy_ep = 0.5 * (np.tanh((ya + 0.35) / 0.12) - np.tanh((ya - 0.90) / 0.12))
    w_epoxy = np.clip(wx_ep * wy_ep, 0.0, 1.0)

    wx_oil = 0.5 * (np.tanh((xa - 2.65) / 0.09) - np.tanh((xa - 3.95) / 0.09))
    wy_oil = 0.5 * (np.tanh((ya + 0.95) / 0.12) - np.tanh((ya - 0.35) / 0.12))
    w_oil = np.clip(wx_oil * wy_oil, 0.0, 1.0)

    mu_base = 0.35 - (0.35 - 0.26) * w_epoxy
    mu_val = np.clip(mu_base * (1.0 - w_oil) + 0.13 * w_oil, 0.08, 0.60)
    return float(mu_val) if is_scalar else mu_val


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

    def eval_points_batch(self, pts_xy: np.ndarray) -> np.ndarray:
        """
        Đánh giá vector hóa trường ma sát thực tại (M, 2) tọa độ.
        Khớp với eval_center từng điểm trong sai số số học máy (< 1e-12).
        """
        pts = np.asarray(pts_xy, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(1, 2)
        m = pts.shape[0]
        if m == 0:
            return np.zeros(0, dtype=float)

        xs = pts[:, 0]
        ys = pts[:, 1]
        try:
            base_mu = np.asarray(self.macro_fn(xs, ys), dtype=float)
            if base_mu.shape != (m,):
                base_mu = np.array([float(self.macro_fn(float(xs[i]), float(ys[i]))) for i in range(m)], dtype=float)
        except Exception:
            base_mu = np.array([float(self.macro_fn(float(xs[i]), float(ys[i]))) for i in range(m)], dtype=float)

        rp = self.road_profile
        arg = xs[:, None] * rp.kx[None, :] + ys[:, None] * rp.ky[None, :] + rp.phases[None, :]
        norm_factor = float(np.sqrt(np.sum(rp.amplitudes ** 2)))
        spatial_noise = (np.sin(arg) @ rp.amplitudes) / (norm_factor * np.sqrt(2.0) + 1e-6)
        spatial_noise = np.clip(spatial_noise, -1.0, 1.0)
        mu_local = np.clip(base_mu * (1.0 + self.variation_scale * spatial_noise), 0.05, 0.80)
        return np.clip(mu_local, self.mu_clip[0], self.mu_clip[1])

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
        wheel_pts = np.empty((4, 2), dtype=float)
        for idx, (bx, by) in enumerate(offsets):
            wheel_pts[idx, 0] = float(x) + c_y * bx - s_y * by
            wheel_pts[idx, 1] = float(y) + s_y * bx + c_y * by

        mu_wheels = self.eval_points_batch(wheel_pts)
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

        center_pts = np.column_stack([x_arr, y_arr])
        mu_center = self.eval_points_batch(center_pts)

        fp_4w = compute_footprint_query_points(
            x_arr, y_arr, yaw_grid=yaw_arr, L_x=L_x, L_y=L_y, lateral_margin=0.0
        )  # (N, 4, 2)
        mu_4w_mat = self.eval_points_batch(fp_4w.reshape(-1, 2)).reshape(n, 4)
        mu_min_4w = np.min(mu_4w_mat, axis=1)
        split_delta = np.max(mu_4w_mat, axis=1) - mu_min_4w
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


def make_random_site_ground_truth(
    site_seed: int,
    road_class: str = "B",
    variation_scale: float = 0.15,
    mu_clip: tuple = (0.08, 0.60),
) -> MultiZoneFrictionGroundTruth:
    """
    Tạo một bố trí sàn kho ngẫu nhiên độc lập (random floor site) để kiểm chứng
    khả năng tổng quát hóa xuyên bố trí sàn (M7):
      - Nền khô ngẫu nhiên mu_dry in [0.32, 0.38]
      - 1 vùng epoxy với vị trí, kích thước, ma sát ngẫu nhiên
      - 1 đến 2 vũng dầu/nước tràn với vị trí, kích thước, ma sát ngẫu nhiên
      - Độ nhám vi mô ISO 8608:2016 Class B dùng seed độc lập site_seed + 50000.
    """
    rng = np.random.default_rng(int(site_seed))
    mu_dry = float(rng.uniform(0.32, 0.38))

    ep_mu = float(rng.uniform(0.22, 0.28))
    ep_xc = float(rng.uniform(0.9, 5.6))
    ep_hw = float(rng.uniform(0.45, 0.85))
    ep_yc = float(rng.uniform(-0.50, 0.50))
    ep_hh = float(rng.uniform(0.45, 0.80))
    ep_edge = float(rng.uniform(0.09, 0.12))

    n_spills = int(rng.integers(1, 3))
    spills = []
    for _ in range(n_spills):
        spills.append({
            "mu": float(rng.uniform(0.11, 0.16)),
            "xc": float(rng.uniform(1.4, 5.5)),
            "hw": float(rng.uniform(0.45, 0.80)),
            "yc": float(rng.uniform(-0.55, 0.55)),
            "hh": float(rng.uniform(0.45, 0.70)),
            "edge": float(rng.uniform(0.09, 0.12)),
        })

    def _site_macro(x, y):
        is_scalar = np.isscalar(x) and np.isscalar(y)
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)

        wx_ep = 0.5 * (np.tanh((xa - (ep_xc - ep_hw)) / ep_edge) - np.tanh((xa - (ep_xc + ep_hw)) / ep_edge))
        wy_ep = 0.5 * (np.tanh((ya - (ep_yc - ep_hh)) / ep_edge) - np.tanh((ya - (ep_yc + ep_hh)) / ep_edge))
        w_ep = np.clip(wx_ep * wy_ep, 0.0, 1.0)
        mu_val = mu_dry - (mu_dry - ep_mu) * w_ep

        for sp in spills:
            e_sp = sp["edge"]
            wx_sp = 0.5 * (np.tanh((xa - (sp["xc"] - sp["hw"])) / e_sp) - np.tanh((xa - (sp["xc"] + sp["hw"])) / e_sp))
            wy_sp = 0.5 * (np.tanh((ya - (sp["yc"] - sp["hh"])) / e_sp) - np.tanh((ya - (sp["yc"] + sp["hh"])) / e_sp))
            w_sp = np.clip(wx_sp * wy_sp, 0.0, 1.0)
            mu_val = mu_val * (1.0 - w_sp) + sp["mu"] * w_sp

        mu_val = np.clip(mu_val, mu_clip[0], mu_clip[1])
        return float(mu_val) if is_scalar else mu_val

    return MultiZoneFrictionGroundTruth(
        macro_fn=_site_macro,
        road_class=road_class,
        seed=int(site_seed) + 50000,
        variation_scale=variation_scale,
        mu_clip=mu_clip,
    )


def compute_footprint_query_points(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    yaw_grid: np.ndarray = None,
    L_x: float = L_X,
    L_y: float = L_Y,
    lateral_margin: float = 0.05,
) -> np.ndarray:
    """
    Tính tọa độ 2D dưới dấu chân 4 bánh xe (FL, FR, RR, RL) và biên sai số bám ngang
    tại mỗi nút dọc quỹ đạo tham chiếu (M3).
    Returns: np.ndarray có shape (N, K, 2), với K = 6 nếu lateral_margin > 0 hoặc K = 4.
    """
    x_arr = np.asarray(x_grid, dtype=float).ravel()
    y_arr = np.asarray(y_grid, dtype=float).ravel()
    n = len(x_arr)

    offsets = [
        (+float(L_x), +float(L_y)),  # FL
        (+float(L_x), -float(L_y)),  # FR
        (-float(L_x), -float(L_y)),  # RR
        (-float(L_x), +float(L_y)),  # RL
    ]
    if float(lateral_margin) > 1e-9:
        offsets.extend([
            (0.0, +(float(L_y) + float(lateral_margin))),
            (0.0, -(float(L_y) + float(lateral_margin))),
        ])
    off_arr = np.asarray(offsets, dtype=float)  # (K, 2)
    k_pts = off_arr.shape[0]

    if n == 0:
        return np.zeros((0, k_pts, 2), dtype=float)

    if yaw_grid is None:
        if n >= 2:
            yaw_arr = np.arctan2(np.gradient(y_arr), np.gradient(x_arr))
        else:
            yaw_arr = np.zeros(1, dtype=float)
    else:
        yaw_arr = np.asarray(yaw_grid, dtype=float).ravel()

    cos_y = np.cos(yaw_arr)[:, None]  # (N, 1)
    sin_y = np.sin(yaw_arr)[:, None]  # (N, 1)
    bx = off_arr[None, :, 0]          # (1, K)
    by = off_arr[None, :, 1]          # (1, K)

    pts_x = x_arr[:, None] + cos_y * bx - sin_y * by
    pts_y = y_arr[:, None] + sin_y * bx + cos_y * by
    return np.stack([pts_x, pts_y], axis=-1)


def aggregate_observations_to_voxels(
    points_xy: np.ndarray,
    mu_obs: np.ndarray,
    obs_var: np.ndarray = None,
    voxel_size: float = 0.15,
    min_var_floor: float = 4.0e-4,
    obs_age_days: np.ndarray = None,
    aging_rate: float = 0.0,
) -> tuple:
    """
    Gom cụm mẫu quan sát vào lưới voxel không gian theo trung bình trọng số nghịch đảo phương sai.
    Hỗ trợ cộng thêm phương sai lão hóa theo thời gian obs_var += aging_rate * age_days (PT 2 PDF)
    và chặn dưới phương sai để tránh tự tin thái quá do độ nhám ISO 8608.
    """
    pts = np.asarray(points_xy, dtype=float)
    y = np.asarray(mu_obs, dtype=float).ravel()
    if pts.ndim != 2 or pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=float), np.zeros(0, dtype=float), np.zeros(0, dtype=float)

    if obs_var is None:
        var = np.full(len(y), max(min_var_floor, 1e-3), dtype=float)
    else:
        var = np.maximum(np.asarray(obs_var, dtype=float).ravel(), min_var_floor)
    if obs_age_days is not None and float(aging_rate) > 0.0:
        ages = np.maximum(0.0, np.asarray(obs_age_days, dtype=float).ravel())
        var = var + float(aging_rate) * ages

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
    observation_mode: str = "per_wheel_rls_lagcomp",
    bidirectional: bool = False,
    L_x: float = L_X,
    L_y: float = L_Y,
    collect_nonslip_lb: bool = False,
    nonslip_k_sigma: float = 2.5,
) -> dict:
    """
    Thu thập tập lịch sử nhân quả D_hist thông qua bộ quan sát FrictionObserverFFRLS
    trên các lượt chạy quá khứ (KHÔNG lộ ground truth cho bộ ước lượng).

    Hỗ trợ 2 chế độ:
      - 'per_wheel_rls_lagcomp' (mặc định M5): 4 bộ FF-RLS độc lập theo từng bánh
        (FL, FR, RR, RL) gán vào tâm không gian trễ nhân quả p_bar_{k,w} của cửa sổ RLS.
      - 'chassis_ffrls_legacy': 1 bộ FF-RLS toàn xe gán tại tọa độ tâm tức thời (đối chứng).
    Tùy chọn collect_nonslip_lb=True ghi nhận thêm chặn dưới ma sát phi-trượt
    ell_{k,w} = max(0, |F_meas,w|/Fz_w - nonslip_k_sigma * sigma_u) tại các bước không trượt.
    """
    if observation_mode not in ("per_wheel_rls_lagcomp", "chassis_ffrls_legacy"):
        raise ValueError(f"observation_mode không hợp lệ: '{observation_mode}'")

    rng = np.random.default_rng(seed)
    pts_list = []
    mu_hat_list = []
    var_list = []
    nonslip_pts_list = []
    nonslip_lb_list = []
    n_total_steps = 0
    n_deadzone_skipped = 0

    wheel_offsets = np.array([
        [+float(L_x), +float(L_y)],  # FL
        [+float(L_x), -float(L_y)],  # FR
        [-float(L_x), -float(L_y)],  # RR
        [-float(L_x), +float(L_y)],  # RL
    ], dtype=float)

    for traj_idx, traj in enumerate(rollout_trajectories):
        if isinstance(traj, dict):
            x_arr = np.asarray(traj["x_grid"], dtype=float).copy()
            y_arr = np.asarray(traj["y_grid"], dtype=float).copy()
            yaw_explicit = np.asarray(traj["yaw_grid"], dtype=float).copy() if "yaw_grid" in traj else None
            rho_arr = np.asarray(traj["rho_grid"], dtype=float).copy() if "rho_grid" in traj else None
            slip_arr = np.asarray(traj["slip_grid"], dtype=float).copy() if "slip_grid" in traj else None
        else:
            wp = np.asarray(traj, dtype=float)
            x_arr = wp[:, 0].copy()
            y_arr = wp[:, 1].copy()
            yaw_explicit = None
            rho_arr = None
            slip_arr = None

        n_pts = len(x_arr)
        if n_pts < 3:
            continue

        if bidirectional and (traj_idx % 2 == 1):
            x_arr = x_arr[::-1].copy()
            y_arr = y_arr[::-1].copy()
            yaw_explicit = yaw_explicit[::-1].copy() if yaw_explicit is not None else None
            rho_arr = rho_arr[::-1].copy() if rho_arr is not None else None
            slip_arr = slip_arr[::-1].copy() if slip_arr is not None else None

        yaw_arr = yaw_explicit if yaw_explicit is not None else np.arctan2(np.gradient(y_arr), np.gradient(x_arr))

        # Tính độ cong cục bộ để mô phỏng kích thích tự nhiên ở đoạn cua & đoạn tăng/giảm tốc
        dx = np.gradient(x_arr)
        dy = np.gradient(y_arr)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        sp = np.maximum(np.hypot(dx, dy), 1e-6)
        kappa = np.abs(dx * ddy - dy * ddx) / (sp ** 3)

        if observation_mode == "per_wheel_rls_lagcomp":
            observers = [
                FrictionObserverFFRLS(
                    mu_init=0.25,
                    mu_min=0.06,
                    mu_max=0.60,
                    lam=lam_rls,
                    P_init=10.0,
                    rho_deadzone=0.06,
                )
                for _ in range(4)
            ]
            S_w = np.zeros(4, dtype=float)
            p_bar_w = [None, None, None, None]
            fz_nom_wheel = (m_total * g) / 4.0
            m_wheel = m_total / 4.0
            c_P_eff = c_P_var / 16.0  # Quy đổi phi_w = phi_chassis / 4 -> P_w ~ 16*P_chassis

            for i in range(n_pts):
                if rho_arr is not None:
                    rho_cmd = float(rho_arr[i])
                else:
                    frac = i / max(n_pts - 1, 1)
                    if frac < 0.14 or frac > 0.86:
                        a_tot = 0.85 + 0.25 * rng.uniform()
                    else:
                        a_lat = (0.58 ** 2) * float(kappa[i])
                        a_probe = 0.72 if (i % 7 < 4) else 0.18
                        a_tot = float(np.hypot(a_probe, a_lat))
                    rho_cmd = a_tot / g

                if slip_arr is not None:
                    base_slip = float(slip_arr[i])
                else:
                    if rho_cmd >= 0.08:
                        base_slip = float(np.clip(0.022 + 0.18 * (rho_cmd - 0.06) + rng.normal(0.0, 0.004), 0.018, 0.09))
                    else:
                        base_slip = float(np.clip(0.006 + 0.05 * rho_cmd, 0.002, 0.014))

                th_i = float(yaw_arr[i])
                c_y, s_y = float(np.cos(th_i)), float(np.sin(th_i))
                _, mu_wheels_true, _ = ground_truth.eval_four_wheels(x_arr[i], y_arr[i], th_i, L_x=L_x, L_y=L_y)

                for w in range(4):
                    n_total_steps += 1
                    bx, by = wheel_offsets[w]
                    p_w = np.array([
                        float(x_arr[i]) + c_y * bx - s_y * by,
                        float(y_arr[i]) + s_y * bx + c_y * by,
                    ], dtype=float)
                    slip_w = float(np.clip(base_slip * (1.0 + rng.normal(0.0, 0.03)), 0.001, 0.12))
                    fz_w = float(max(0.5 * fz_nom_wheel, fz_nom_wheel + rng.normal(0.0, 0.6)))
                    phi_w = fz_w * np.tanh(slip_w / s_char)
                    y_w = max(0.0, float(mu_wheels_true[w]) * phi_w * (1.0 + rng.normal(0.0, noise_std_rel)))

                    w_cmd_xy = np.array([rho_cmd * fz_nom_wheel, 0.0])
                    w_act_xy = np.array([y_w, 0.0])
                    mu_hat_w, _, bin_idx = observers[w].update(
                        W_actual_xy=w_act_xy,
                        W_cmd_xy=w_cmd_xy,
                        wheel_slip_ratios=np.array([slip_w]),
                        Fz_total=fz_w,
                        m_total=m_wheel,
                        g=g,
                    )

                    if rho_cmd >= observers[w].rho_deadzone and slip_w >= 0.015:
                        w_energy = phi_w ** 2
                        s_new = lam_rls * S_w[w] + w_energy
                        if S_w[w] <= 0.0 or p_bar_w[w] is None:
                            p_bar_w[w] = p_w.copy()
                        else:
                            p_bar_w[w] = (lam_rls * S_w[w] * p_bar_w[w] + w_energy * p_w) / s_new
                        S_w[w] = s_new

                    if bin_idx >= 1 and p_bar_w[w] is not None:
                        obs_var_i = float(np.clip(
                            c_P_eff * observers[w].P + sigma_obs_floor ** 2,
                            sigma_obs_floor ** 2,
                            0.04,
                        ))
                        pts_list.append([float(p_bar_w[w][0]), float(p_bar_w[w][1])])
                        mu_hat_list.append(float(mu_hat_w))
                        var_list.append(obs_var_i)
                    else:
                        n_deadzone_skipped += 1
                        if collect_nonslip_lb and fz_w > 1e-3:
                            u_ns = y_w / fz_w
                            sig_u = max(0.005, u_ns * float(np.hypot(noise_std_rel, 0.6 / fz_w)))
                            ell_ns = float(np.clip(u_ns - float(nonslip_k_sigma) * sig_u, 0.05, 0.55))
                            nonslip_pts_list.append([float(p_w[0]), float(p_w[1])])
                            nonslip_lb_list.append(ell_ns)
        else:
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
                _, mu_wheels_true, _ = ground_truth.eval_four_wheels(x_arr[i], y_arr[i], yaw_arr[i], L_x=L_x, L_y=L_y)
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

    ns_pts = np.asarray(nonslip_pts_list, dtype=float) if nonslip_pts_list else np.zeros((0, 2), dtype=float)
    ns_lb = np.asarray(nonslip_lb_list, dtype=float) if nonslip_lb_list else np.zeros(0, dtype=float)

    if len(pts_list) == 0:
        return {
            "points_xy": np.zeros((0, 2), dtype=float),
            "mu_obs": np.zeros(0, dtype=float),
            "obs_var": np.zeros(0, dtype=float),
            "nonslip_points_xy": ns_pts,
            "nonslip_mu_lb": ns_lb,
            "n_total_steps": n_total_steps,
            "n_accepted": 0,
            "n_deadzone_skipped": n_deadzone_skipped,
            "observation_mode": observation_mode,
            "bidirectional": bool(bidirectional),
            "lam_rls": float(lam_rls),
        }

    return {
        "points_xy": np.asarray(pts_list, dtype=float),
        "mu_obs": np.asarray(mu_hat_list, dtype=float),
        "obs_var": np.asarray(var_list, dtype=float),
        "nonslip_points_xy": ns_pts,
        "nonslip_mu_lb": ns_lb,
        "n_total_steps": n_total_steps,
        "n_accepted": len(pts_list),
        "n_deadzone_skipped": n_deadzone_skipped,
        "observation_mode": observation_mode,
        "bidirectional": bool(bidirectional),
        "lam_rls": float(lam_rls),
    }


class SpatialFrictionGP:
    """
    Bộ ước lượng trường ma sát không gian 2D bằng Heteroscedastic Gaussian Process
    (Matérn-3/2 vĩ mô + RBF vi mô) kèm Hiệu chuẩn Conformal cấp quỹ đạo trên dấu chân 4 bánh.
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
        query_mode: str = "footprint",
        L_x: float = L_X,
        L_y: float = L_Y,
        lateral_margin: float = 0.05,
        erosion_radius: float = 0.0,
        grid_res: float = 0.05,
        use_raster_cache: bool = False,
        aging_rate: float = 0.0,
    ):
        if query_mode not in ("footprint", "center"):
            raise ValueError(f"query_mode không hợp lệ: '{query_mode}'")
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
        self.query_mode = str(query_mode)
        self.L_x = float(L_x)
        self.L_y = float(L_y)
        self.lateral_margin = float(lateral_margin)
        self.erosion_radius = max(0.0, float(erosion_radius))
        self.grid_res = max(0.01, float(grid_res))
        self.use_raster_cache = bool(use_raster_cache or (self.erosion_radius > 1e-9))
        self.aging_rate = max(0.0, float(aging_rate))

        self.X_train = np.zeros((0, 2), dtype=float)
        self.y_train = np.zeros(0, dtype=float)
        self.obs_var_train = np.zeros(0, dtype=float)
        self.nonslip_points_xy = np.zeros((0, 2), dtype=float)
        self.nonslip_mu_lb = np.zeros(0, dtype=float)
        self._cho_factor = None
        self._alpha_vec = None
        self._L_inv_T = None
        self._K_inv = None
        self._raster_cache = None
        self.last_fit_time_ms = 0.0
        self.last_raster_build_ms = 0.0
        self.last_predict_time_ms = 0.0
        self.cal_meta = {}

    def _compute_kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Tính ma trận hiệp phương sai K(X1, X2) = K_matern32_macro + K_rbf_micro bằng C-cdist tại chỗ."""
        d = cdist(X1, X2, metric="euclidean")
        d2 = d * d

        # Macro Matérn-3/2 (đảm bảo liên tục khả vi C^1 cho đường ống vận tốc)
        d *= np.sqrt(3.0) / max(self.ell_macro, 1e-6)
        k_mat = np.exp(-d)
        d += 1.0
        k_mat *= d
        k_mat *= (self.sigma_f ** 2)

        # Micro RBF (nắm bắt tương quan độ nhám không gian vi mô ISO 8608)
        if self.sigma_micro > 0.0:
            d2 *= -0.5 / (max(self.ell_micro, 1e-6) ** 2)
            np.exp(d2, out=d2)
            d2 *= (self.sigma_micro ** 2)
            k_mat += d2
        return k_mat

    def _kernel_diag(self, n: int) -> np.ndarray:
        return np.full(n, (self.sigma_f ** 2) + (self.sigma_micro ** 2), dtype=float)

    def fit(
        self,
        points_xy: np.ndarray,
        mu_obs: np.ndarray,
        obs_var: np.ndarray = None,
        voxel_size: float = 0.15,
        obs_age_days: np.ndarray = None,
        nonslip_points_xy: np.ndarray = None,
        nonslip_mu_lb: np.ndarray = None,
        raster_bounds: tuple = None,
    ) -> "SpatialFrictionGP":
        """Huấn luyện posterior GP bằng phân rã Cholesky trên tập dữ liệu (kèm voxel, lão hóa và cache lưới 2D/co biên)."""
        t0 = time.perf_counter()
        pts = np.asarray(points_xy, dtype=float)
        y = np.asarray(mu_obs, dtype=float).ravel()

        if nonslip_points_xy is not None and nonslip_mu_lb is not None:
            ns_p = np.asarray(nonslip_points_xy, dtype=float)
            ns_l = np.asarray(nonslip_mu_lb, dtype=float).ravel()
            if ns_p.ndim == 2 and len(ns_p) == len(ns_l) and len(ns_p) > 0:
                self.nonslip_points_xy = ns_p.copy()
                self.nonslip_mu_lb = ns_l.copy()
            else:
                self.nonslip_points_xy = np.zeros((0, 2), dtype=float)
                self.nonslip_mu_lb = np.zeros(0, dtype=float)
        else:
            self.nonslip_points_xy = np.zeros((0, 2), dtype=float)
            self.nonslip_mu_lb = np.zeros(0, dtype=float)

        if pts.ndim != 2 or pts.shape[0] == 0 or len(y) == 0:
            self.X_train = np.zeros((0, 2), dtype=float)
            self.y_train = np.zeros(0, dtype=float)
            self.obs_var_train = np.zeros(0, dtype=float)
            self._cho_factor = None
            self._alpha_vec = None
            self._L_inv_T = None
            self._K_inv = None
            self._raster_cache = None
            self.last_raster_build_ms = 0.0
            self.last_fit_time_ms = (time.perf_counter() - t0) * 1000.0
            return self

        if pts.shape[1] != 2:
            raise ValueError(f"points_xy phải có shape (M, 2), nhận được {pts.shape}")

        var_floor = self.sigma_obs_floor ** 2
        if obs_var is None:
            var = np.full(len(y), var_floor, dtype=float)
        else:
            var = np.maximum(np.asarray(obs_var, dtype=float).ravel(), var_floor)

        if obs_age_days is not None and self.aging_rate > 0.0:
            ages = np.maximum(0.0, np.asarray(obs_age_days, dtype=float).ravel())
            var = var + self.aging_rate * ages

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
        L_inv = solve_triangular(self._cho_factor[0], np.eye(len(self.X_train)), lower=True, check_finite=False)
        self._L_inv_T = L_inv.T
        self._K_inv = L_inv.T @ L_inv
        self._raster_cache = None
        self.last_raster_build_ms = 0.0
        if self.use_raster_cache or self.erosion_radius > 1e-9:
            self.build_raster_cache(
                bounds=raster_bounds,
                nonslip_points_xy=self.nonslip_points_xy,
                nonslip_mu_lb=self.nonslip_mu_lb,
            )
        self.last_fit_time_ms = (time.perf_counter() - t0) * 1000.0
        return self

    def log_marginal_likelihood(self) -> float:
        """Tính log-marginal likelihood (LML) trên tập dữ liệu hỗ trợ để đối chiếu độ nhạy siêu tham số."""
        if self._cho_factor is None or len(self.X_train) == 0:
            return 0.0
        y_c = self.y_train - self.mu_prior
        L_mat = self._cho_factor[0]
        log_det = 2.0 * float(np.sum(np.log(np.diag(L_mat))))
        quad = float(y_c @ self._alpha_vec)
        n = len(self.y_train)
        return float(-0.5 * quad - 0.5 * log_det - 0.5 * n * np.log(2.0 * np.pi))

    def _predict_exact(self, X_q: np.ndarray) -> tuple:
        """Đánh giá trung bình và độ lệch chuẩn hậu nghiệm GP chính xác tại tập điểm (N, 2)."""
        n_q = X_q.shape[0]
        if n_q == 0:
            return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
        prior_std = float(np.sqrt((self.sigma_f ** 2) + (self.sigma_micro ** 2)))
        if self._cho_factor is None or len(self.X_train) == 0:
            return np.full(n_q, self.mu_prior, dtype=float), np.full(n_q, prior_std, dtype=float)
        K_qs = self._compute_kernel_matrix(X_q, self.X_train)
        mu_mean = self.mu_prior + K_qs @ self._alpha_vec
        V = K_qs @ self._L_inv_T
        var_red = np.sum(V * V, axis=1)
        var_post = np.maximum(1e-10, self._kernel_diag(n_q) - var_red)
        return mu_mean, np.sqrt(var_post)

    def build_raster_cache(
        self,
        bounds: tuple = None,
        nonslip_points_xy: np.ndarray = None,
        nonslip_mu_lb: np.ndarray = None,
        chunk_size: int = 4096,
    ) -> dict:
        """
        Raster hóa hậu nghiệm GP trên lưới 2D đều (mặc định ô 5 cm), áp dụng phép co biên
        hình thái học đĩa bán kính erosion_radius cho mu_mean (và giãn biên cho sigma_gp),
        kèm kẹp chặn dưới phi-trượt l_c theo PT (7) của bản PDF để truy vấn O(1) theo song tuyến tính.
        """
        t0 = time.perf_counter()
        if self._cho_factor is None or len(self.X_train) == 0:
            self._raster_cache = None
            self.last_raster_build_ms = 0.0
            return {}

        res = max(0.01, float(self.grid_res))
        pad = max(1.60, 1.5 * self.ell_macro + self.erosion_radius)
        if bounds is None:
            x_min = np.floor((float(np.min(self.X_train[:, 0])) - pad) / res) * res
            x_max = np.ceil((float(np.max(self.X_train[:, 0])) + pad) / res) * res
            y_min = np.floor((float(np.min(self.X_train[:, 1])) - pad) / res) * res
            y_max = np.ceil((float(np.max(self.X_train[:, 1])) + pad) / res) * res
        else:
            x_min, x_max, y_min, y_max = [float(v) for v in bounds]

        xs = np.arange(x_min, x_max + 0.5 * res, res, dtype=float)
        ys = np.arange(y_min, y_max + 0.5 * res, res, dtype=float)
        nx, ny = len(xs), len(ys)
        XX, YY = np.meshgrid(xs, ys, indexing="ij")
        grid_pts = np.column_stack([XX.ravel(), YY.ravel()])

        n_tot = grid_pts.shape[0]
        mu_flat = np.empty(n_tot, dtype=float)
        sig_flat = np.empty(n_tot, dtype=float)
        for i0 in range(0, n_tot, int(chunk_size)):
            i1 = min(n_tot, i0 + int(chunk_size))
            m_c, s_c = self._predict_exact(grid_pts[i0:i1])
            mu_flat[i0:i1] = m_c
            sig_flat[i0:i1] = s_c

        mu_grid = mu_flat.reshape(nx, ny)
        sigma_grid = sig_flat.reshape(nx, ny)

        if self.erosion_radius > 1e-6 and nx >= 2 and ny >= 2:
            r_cells = int(np.ceil(self.erosion_radius / res))
            off = np.arange(-r_cells, r_cells + 1, dtype=float) * res
            dx_d, dy_d = np.meshgrid(off, off, indexing="ij")
            disk_fp = (dx_d ** 2 + dy_d ** 2) <= (self.erosion_radius + 1e-9) ** 2
            mu_grid = minimum_filter(mu_grid, footprint=disk_fp, mode="nearest")
            sigma_grid = maximum_filter(sigma_grid, footprint=disk_fp, mode="nearest")

        if nonslip_points_xy is not None and nonslip_mu_lb is not None and len(nonslip_points_xy) > 0:
            ns_p = np.asarray(nonslip_points_xy, dtype=float)
            ns_l = np.asarray(nonslip_mu_lb, dtype=float).ravel()
            in_g = (
                (ns_p[:, 0] >= xs[0] - 0.5 * res) & (ns_p[:, 0] <= xs[-1] + 0.5 * res)
                & (ns_p[:, 1] >= ys[0] - 0.5 * res) & (ns_p[:, 1] <= ys[-1] + 0.5 * res)
            )
            if np.any(in_g):
                ix = np.clip(np.rint((ns_p[in_g, 0] - xs[0]) / res).astype(int), 0, nx - 1)
                iy = np.clip(np.rint((ns_p[in_g, 1] - ys[0]) / res).astype(int), 0, ny - 1)
                ell_grid = np.full((nx, ny), -np.inf, dtype=float)
                np.maximum.at(ell_grid, (ix, iy), ns_l[in_g])
                has_lb = np.isfinite(ell_grid)
                mu_grid[has_lb] = np.maximum(mu_grid[has_lb], ell_grid[has_lb])

        self._raster_cache = {
            "xs": xs,
            "ys": ys,
            "res": res,
            "nx": nx,
            "ny": ny,
            "mu_grid": mu_grid,
            "sigma_grid": sigma_grid,
        }
        self.last_raster_build_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "grid_shape": (nx, ny),
            "raster_build_ms": round(self.last_raster_build_ms, 3),
            "erosion_radius_m": self.erosion_radius,
            "grid_res_m": res,
        }

    def _predict_from_raster(self, X_q: np.ndarray) -> tuple:
        """Nội suy song tuyến tính O(1) trên lưới 2D; rơi về GP chính xác cho điểm ngoài khung lưới."""
        rc = self._raster_cache
        xs, ys, res = rc["xs"], rc["ys"], rc["res"]
        nx, ny = rc["nx"], rc["ny"]
        mu_g, sig_g = rc["mu_grid"], rc["sigma_grid"]

        in_box = (
            (X_q[:, 0] >= xs[0]) & (X_q[:, 0] <= xs[-1])
            & (X_q[:, 1] >= ys[0]) & (X_q[:, 1] <= ys[-1])
        )
        n_q = X_q.shape[0]
        if nx < 2 or ny < 2 or not np.any(in_box):
            return self._predict_exact(X_q)

        mu_out = np.empty(n_q, dtype=float)
        sig_out = np.empty(n_q, dtype=float)

        ux = np.clip((X_q[in_box, 0] - xs[0]) / res, 0.0, nx - 1.0 - 1e-9)
        uy = np.clip((X_q[in_box, 1] - ys[0]) / res, 0.0, ny - 1.0 - 1e-9)
        ix = ux.astype(int)
        iy = uy.astype(int)
        wx = ux - ix
        wy = uy - iy

        w00 = (1.0 - wx) * (1.0 - wy)
        w10 = wx * (1.0 - wy)
        w01 = (1.0 - wx) * wy
        w11 = wx * wy

        mu_out[in_box] = (
            w00 * mu_g[ix, iy] + w10 * mu_g[ix + 1, iy]
            + w01 * mu_g[ix, iy + 1] + w11 * mu_g[ix + 1, iy + 1]
        )
        sig_out[in_box] = (
            w00 * sig_g[ix, iy] + w10 * sig_g[ix + 1, iy]
            + w01 * sig_g[ix, iy + 1] + w11 * sig_g[ix + 1, iy + 1]
        )

        if not np.all(in_box):
            m_oob, s_oob = self._predict_exact(X_q[~in_box])
            mu_out[~in_box] = m_oob
            sig_out[~in_box] = s_oob
        return mu_out, sig_out

    def predict(self, points_xy: np.ndarray) -> tuple:
        """
        Dự báo vector hóa tại tập tọa độ truy vấn (N, 2) bằng lưới O(1) (nếu bật) hoặc GP chính xác.
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

        if self._raster_cache is not None and self._cho_factor is not None and len(self.X_train) > 0:
            mu_mean, sigma_gp = self._predict_from_raster(X_q)
        else:
            mu_mean, sigma_gp = self._predict_exact(X_q)

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
        yaw_grid: np.ndarray = None,
        query_mode: str = None,
    ) -> dict:
        """
        Truy vấn trường ma sát dọc lưới quỹ đạo A* (x_grid, y_grid) tại t = 0.
        mode:
          - 'conformal_lcb': trả về cận dưới Conformal mu_lcb (an toàn mặc định)
          - 'gp_mean': chỉ dùng trung bình hậu nghiệm GP cắt ngưỡng [mu_min, mu_max] (đối chứng M1)
        query_mode:
          - 'footprint' (mặc định M3): dự báo tại chính 4 bánh + biên lệch ngang rồi lấy min
          - 'center': chỉ dự báo tại tâm quỹ đạo (đối chứng M3 cũ)
        """
        if mode not in ("conformal_lcb", "gp_mean"):
            raise ValueError(f"mode không hợp lệ: '{mode}'. Chọn 'conformal_lcb' hoặc 'gp_mean'.")
        qm = self.query_mode if query_mode is None else str(query_mode)
        if qm not in ("footprint", "center"):
            raise ValueError(f"query_mode không hợp lệ: '{qm}'.")

        x_arr = np.asarray(x_grid, dtype=float).ravel()
        y_arr = np.asarray(y_grid, dtype=float).ravel()
        n_q = len(x_arr)
        prior_std = float(np.sqrt((self.sigma_f ** 2) + (self.sigma_micro ** 2)))

        if n_q == 0:
            self.last_predict_time_ms = 0.0
            empty = np.zeros(0, dtype=float)
            return {
                "mu_tube": empty,
                "mu_mean": empty,
                "mu_mean_raw": empty,
                "sigma_gp": empty,
                "mu_lcb": empty,
                "is_conformal_safe": (mode == "conformal_lcb"),
                "mode": mode,
                "query_mode": qm,
                "q_hat_mu": float(self.q_hat_mu),
                "alpha_mu": float(self.alpha_mu),
                "sigma_0": float(self.sigma_0),
                "n_support": int(len(self.X_train)),
                "erosion_radius_m": round(self.erosion_radius, 4),
                "use_raster_cache": bool(self._raster_cache is not None),
                "grid_res_m": round(self.grid_res, 4),
                "predict_time_ms": 0.0,
                "prior_region_frac": 0.0,
                "clamped_mu_min_frac": 0.0,
            }

        if qm == "footprint":
            fp_pts = compute_footprint_query_points(
                x_arr, y_arr, yaw_grid=yaw_grid,
                L_x=self.L_x, L_y=self.L_y, lateral_margin=self.lateral_margin,
            )  # (N, K, 2)
            k_fp = fp_pts.shape[1]
            mu_m_flat, sig_flat, _ = self.predict(fp_pts.reshape(-1, 2))
            mu_m_all = mu_m_flat.reshape(n_q, k_fp)
            sig_all = sig_flat.reshape(n_q, k_fp)

            lcb_all = mu_m_all - self.q_hat_mu * (sig_all + self.sigma_0)
            idx_min = np.argmin(lcb_all, axis=1)
            rows = np.arange(n_q)
            mu_lcb = np.clip(lcb_all[rows, idx_min], self.mu_min, self.mu_max)
            mu_mean_raw = np.min(mu_m_all, axis=1)
            mu_mean_clipped = np.clip(mu_mean_raw, self.mu_min, self.mu_max)
            sigma_gp = sig_all[rows, idx_min]
            sigma_gp_max = np.max(sig_all, axis=1)
        else:
            pts = np.column_stack([x_arr, y_arr])
            mu_mean_raw, sigma_gp, mu_lcb = self.predict(pts)
            mu_mean_clipped = np.clip(mu_mean_raw, self.mu_min, self.mu_max)
            sigma_gp_max = sigma_gp

        is_safe = (mode == "conformal_lcb")
        mu_tube = mu_lcb if is_safe else mu_mean_clipped

        # Chẩn đoán M4: tỷ lệ chiều dài tuyến rơi vào vùng prior (ít/không dữ liệu) và vùng bị kẹp sàn mu_min
        prior_region_frac = float(np.mean(sigma_gp_max >= 0.85 * prior_std))
        clamped_mu_min_frac = float(np.mean(mu_lcb <= self.mu_min + 1e-6))

        return {
            "mu_tube": mu_tube,
            "mu_mean": mu_mean_clipped,
            "mu_mean_raw": mu_mean_raw,
            "sigma_gp": sigma_gp,
            "mu_lcb": mu_lcb,
            "is_conformal_safe": is_safe,
            "mode": mode,
            "query_mode": qm,
            "q_hat_mu": float(self.q_hat_mu),
            "alpha_mu": float(self.alpha_mu),
            "sigma_0": float(self.sigma_0),
            "n_support": int(len(self.X_train)),
            "erosion_radius_m": round(self.erosion_radius, 4),
            "use_raster_cache": bool(self._raster_cache is not None),
            "grid_res_m": round(self.grid_res, 4),
            "predict_time_ms": float(self.last_predict_time_ms),
            "prior_region_frac": round(prior_region_frac, 4),
            "clamped_mu_min_frac": round(clamped_mu_min_frac, 4),
        }

    def calibrate_trajectory_conformal(
        self,
        cal_trajectories: list,
        alpha_mu: float = None,
        query_mode: str = None,
    ) -> dict:
        """
        Hiệu chuẩn Split Conformal cấp quỹ đạo (1 điểm supremum chuẩn hóa / quỹ đạo)
        bao trọn ma sát cực tiểu 4 bánh xe:
          - Chế độ 'footprint' (mặc định IFF):
              R_j = max_i min_{k in footprint} max(0, mu_mean[i, k] - mu_min_4w_true[i]) / (sigma_gp[i, k] + sigma_0)
          - Chế độ 'center' (cũ để đối chứng):
              R_j = max_i max(0, mu_mean_center[i] - mu_min_4w_true[i]) / (sigma_gp_center[i] + sigma_0)
        """
        alpha_use = self.alpha_mu if alpha_mu is None else float(alpha_mu)
        qm = self.query_mode if query_mode is None else str(query_mode)
        if qm not in ("footprint", "center"):
            raise ValueError(f"query_mode không hợp lệ: '{qm}'.")

        n_cal = len(cal_trajectories)
        if n_cal == 0:
            raise ValueError("cal_trajectories rỗng, không thể hiệu chuẩn Conformal!")

        scores = np.empty(n_cal, dtype=float)
        for j, traj in enumerate(cal_trajectories):
            x_g = np.asarray(traj["x_grid"], dtype=float).ravel()
            y_g = np.asarray(traj["y_grid"], dtype=float).ravel()
            yaw_g = np.asarray(traj["yaw_grid"], dtype=float).ravel() if "yaw_grid" in traj else None
            mu_min_4w_true = np.asarray(traj["mu_min_4w_true"], dtype=float).ravel()

            if len(x_g) == 0:
                scores[j] = 0.0
                continue

            if qm == "footprint":
                fp_pts = compute_footprint_query_points(
                    x_g, y_g, yaw_grid=yaw_g,
                    L_x=self.L_x, L_y=self.L_y, lateral_margin=self.lateral_margin,
                )
                n_pts_j, k_fp, _ = fp_pts.shape
                mu_m_flat, sig_flat, _ = self.predict(fp_pts.reshape(-1, 2))
                mu_m_all = mu_m_flat.reshape(n_pts_j, k_fp)
                sig_all = sig_flat.reshape(n_pts_j, k_fp)
                norm_err_all = np.maximum(0.0, mu_m_all - mu_min_4w_true[:, None]) / (sig_all + self.sigma_0)
                node_scores = np.min(norm_err_all, axis=1)
                scores[j] = float(np.max(node_scores))
            else:
                pts = np.column_stack([x_g, y_g])
                mu_mean, sigma_gp, _ = self.predict(pts)
                norm_err = np.maximum(0.0, mu_mean - mu_min_4w_true) / (sigma_gp + self.sigma_0)
                scores[j] = float(np.max(norm_err))

        scores_sorted = np.sort(scores)
        k_star = int(np.clip(np.ceil((n_cal + 1) * (1.0 - alpha_use)), 1, n_cal))
        q_hat = float(scores_sorted[k_star - 1])

        # Kiểm tra tự tương quan bậc 1 (lưu ý: chỉ kiểm định phụ thuộc tuyến tính bậc 1, không chứng minh hoàn toàn khả hoán vị)
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
            "method": (
                "Trajectory-Level Split Conformal (IFF 4-Wheel + Lateral Footprint)"
                if qm == "footprint"
                else "Trajectory-Level Split Conformal (Center vs 4-Wheel Min Legacy)"
            ),
            "query_mode": qm,
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
            "lag1_no_serial_dep_detected": bool(z_r1 < 1.96),
            "exchangeability_ok": bool(z_r1 < 1.96),
            "scores": [round(float(s), 6) for s in scores.tolist()],
        }
        return self.cal_meta

    def evaluate_coverage(
        self,
        test_trajectories: list,
        mode: str = "conformal_lcb",
        mu_override_fn=None,
        query_mode: str = None,
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
        prior_fracs = []
        clamped_fracs = []

        for traj in test_trajectories:
            x_g = np.asarray(traj["x_grid"], dtype=float)
            y_g = np.asarray(traj["y_grid"], dtype=float)
            yaw_g = np.asarray(traj["yaw_grid"], dtype=float) if "yaw_grid" in traj else None
            mu_true_4w = np.asarray(traj["mu_min_4w_true"], dtype=float).ravel()

            if mu_override_fn is not None:
                t0 = time.perf_counter()
                mu_tube = np.asarray(mu_override_fn(x_g, y_g), dtype=float).ravel()
                pred_times.append((time.perf_counter() - t0) * 1000.0)
                prior_fracs.append(0.0)
                clamped_fracs.append(float(np.mean(mu_tube <= self.mu_min + 1e-6)))
            else:
                res = self.predict_along_path(x_g, y_g, mode=mode, yaw_grid=yaw_g, query_mode=query_mode)
                mu_tube = res["mu_tube"]
                pred_times.append(res["predict_time_ms"])
                prior_fracs.append(res["prior_region_frac"])
                clamped_fracs.append(res["clamped_mu_min_frac"])

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

        # Khoảng tin cậy Clopper-Pearson một phía dưới (95%), hai phía (95%) & p-value Binomial
        clopper_95_lb = float(beta.ppf(0.05, n_cov, n_test - n_cov + 1)) if n_cov > 0 else 0.0
        clopper_95_ci_lo = float(beta.ppf(0.025, n_cov, n_test - n_cov + 1)) if n_cov > 0 else 0.0
        clopper_95_ci_hi = float(beta.ppf(0.975, n_cov + 1, n_test - n_cov)) if n_cov < n_test else 1.0
        binom_p_one_sided = float(binom.cdf(n_cov, n_test, 1.0 - self.alpha_mu))

        return {
            "mode": mode if mu_override_fn is None else "override",
            "query_mode": (self.query_mode if query_mode is None else str(query_mode)) if mu_override_fn is None else "override",
            "n_test": int(n_test),
            "n_traj_covered": n_cov,
            "traj_coverage_rate": round(traj_cov_rate, 4),
            "pointwise_coverage_rate": round(pt_cov_rate, 4),
            "clopper_pearson_95_lb": round(clopper_95_lb, 4),
            "clopper_pearson_95_ci_2sided": [round(clopper_95_ci_lo, 4), round(clopper_95_ci_hi, 4)],
            "binom_p_value_one_sided": round(binom_p_one_sided, 4),
            "mean_worst_overestimation": round(float(np.mean(worst_overest_list)), 5),
            "max_worst_overestimation": round(float(np.max(worst_overest_list)), 5),
            "mean_conservatism_gap": round(float(np.mean(mean_gap_list)), 5),
            "max_conservatism_gap": round(float(np.max(max_gap_list)), 5),
            "mean_prior_region_frac": round(float(np.mean(prior_fracs)), 4),
            "max_prior_region_frac": round(float(np.max(prior_fracs)), 4),
            "mean_clamped_mu_min_frac": round(float(np.mean(clamped_fracs)), 4),
            "max_clamped_mu_min_frac": round(float(np.max(clamped_fracs)), 4),
            "mean_predict_time_ms": round(float(np.mean(pred_times)), 4),
        }

    def save_calibration(self, path: str, include_support: bool = True) -> None:
        """Lưu tham số kernel, lượng tử Conformal và tùy chọn tập hỗ trợ GP ra JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "version": "2.0",
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
                "query_mode": self.query_mode,
                "L_x": self.L_x,
                "L_y": self.L_y,
                "lateral_margin": self.lateral_margin,
                "erosion_radius": self.erosion_radius,
                "grid_res": self.grid_res,
                "use_raster_cache": self.use_raster_cache,
                "aging_rate": self.aging_rate,
            },
            "q_hat_mu": float(self.q_hat_mu),
            "alpha_mu": float(self.alpha_mu),
            "n_support": int(len(self.X_train)),
            "calibration_summary": {k: v for k, v in self.cal_meta.items() if k != "scores"},
        }
        if include_support and len(self.X_train) > 0:
            sup_dict = {
                "X_train": self.X_train.tolist(),
                "y_train": self.y_train.tolist(),
                "obs_var_train": self.obs_var_train.tolist(),
            }
            if len(self.nonslip_points_xy) > 0 and len(self.nonslip_mu_lb) > 0:
                sup_dict["nonslip_points_xy"] = self.nonslip_points_xy.tolist()
                sup_dict["nonslip_mu_lb"] = self.nonslip_mu_lb.tolist()
            payload["support"] = sup_dict
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_calibration(cls, path: str) -> "SpatialFrictionGP":
        """Nạp lại mô hình GP + lượng tử Conformal từ JSON trong < 2 ms."""
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        hp = payload.get("hyperparams", {})
        cal_sum = payload.get("calibration_summary", {})
        default_qm = "center" if payload.get("version", "1.0") == "1.0" else "footprint"
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
            query_mode=hp.get("query_mode", cal_sum.get("query_mode", default_qm)),
            L_x=hp.get("L_x", L_X),
            L_y=hp.get("L_y", L_Y),
            lateral_margin=hp.get("lateral_margin", 0.05),
            erosion_radius=hp.get("erosion_radius", 0.0),
            grid_res=hp.get("grid_res", 0.05),
            use_raster_cache=hp.get("use_raster_cache", False),
            aging_rate=hp.get("aging_rate", 0.0),
        )
        gp.cal_meta = cal_sum
        sup = payload.get("support")
        if sup is not None and len(sup.get("X_train", [])) > 0:
            ns_p = sup.get("nonslip_points_xy")
            ns_l = sup.get("nonslip_mu_lb")
            gp.fit(
                points_xy=np.asarray(sup["X_train"], dtype=float),
                mu_obs=np.asarray(sup["y_train"], dtype=float),
                obs_var=np.asarray(sup["obs_var_train"], dtype=float),
                voxel_size=0.0,  # Đã gom cụm voxel sẵn trước khi lưu
                nonslip_points_xy=np.asarray(ns_p, dtype=float) if ns_p else None,
                nonslip_mu_lb=np.asarray(ns_l, dtype=float) if ns_l else None,
            )
        return gp
