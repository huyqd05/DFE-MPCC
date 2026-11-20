# -*- coding: utf-8 -*-
"""
semantic_friction_estimator.py
=============================================================================
BỘ ƯỚC LƯỢNG MA SÁT TIỀN KHỞI HÀNH ZERO-SHOT CHO BẢN ĐỒ MỚI (TẦNG 1 - HƯỚNG ĐI 1)
(Semantic Floor Prior + Multi-Map Split Conformal Calibration at t = 0)

Thiết kế tuân thủ ràng buộc: KHÔNG cần chạy thử trước trên bản đồ mới (zero-shot).
1. SemanticFloorMap:
   Biểu diễn bản đồ ngữ nghĩa vật liệu sàn (từ CAD/BIM hoặc occupancy semantic map)
   gồm tập hợp vùng hình học: 'dry_concrete', 'epoxy', 'wet_risk', 'unknown'.
   Hỗ trợ phân lớp vectorized cực nhanh tại t = 0 và xuất hàm ma sát vĩ mô trơn
   cho bộ mô phỏng vật lý (có thể gắn thêm vũng dầu ẩn ngoài bản đồ để kiểm tra Tầng 2).
2. SemanticFrictionEstimator:
   - Giữ nguyên interface predict_along_path(x_grid, y_grid, mode, yaw_grid)
     để cắm trực tiếp vào CoupledVelocityTubeH124.build_offline_tube().
   - Hỗ trợ truy vấn dấu chân 4 bánh + biên lệch ngang +-0.05 m (K = 6 điểm)
     và cận dưới Split Conformal đa bản đồ:
       mu_plan^(0)(s) = clip(min_k [mu_0(c_k) - q_hat_mu * sigma_0(c_k)], mu_min, mu_max)
   - Bất biến theo phép tịnh tiến và xoay hệ tọa độ bản đồ mới.
=============================================================================
"""

import json
import os
import sys
import time
import numpy as np
from scipy.stats import beta, binom

_curr = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_curr) if os.path.basename(_curr) in ["planning", "controller", "solver", "simulation", "tests"] else _curr
for _p in [_root, os.path.join(_root, "planning"), os.path.join(_root, "controller"), os.path.join(_root, "solver")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from controller.mecanum_traction_dynamics import L_X, L_Y
from planning.spatial_friction_gp import (
    MultiZoneFrictionGroundTruth,
    compute_footprint_query_points,
)


DEFAULT_SEMANTIC_CLASSES = {
    "dry_concrete": {"mu_0": 0.35, "sigma_0": 0.035},
    "epoxy": {"mu_0": 0.26, "sigma_0": 0.030},
    "wet_risk": {"mu_0": 0.14, "sigma_0": 0.025},
    "unknown": {"mu_0": 0.22, "sigma_0": 0.045},
}


class SemanticFloorMap:
    """
    Bản đồ phân vùng vật liệu sàn 2D (tịnh tiến & xoay bất biến theo danh sách vùng).
    Mỗi zone có dạng:
      - circle: {"shape": "circle", "cx": float, "cy": float, "r": float, "label": str, "blend_m": float}
      - rect / rotated_rect: {"shape": "rect", "cx": float, "cy": float, "hx": float, "hy": float,
                              "angle": float, "label": str, "blend_m": float}
    Vùng sau ghi đè vùng trước (khi tạo lớp CAD hoặc lớp nền thực).
    """

    def __init__(
        self,
        zones: list = None,
        default_label: str = "dry_concrete",
        class_params: dict = None,
        name: str = "custom_semantic_map",
    ):
        self.zones = list(zones) if zones is not None else []
        self.default_label = str(default_label)
        self.class_params = {
            k: {"mu_0": float(v["mu_0"]), "sigma_0": float(v["sigma_0"])}
            for k, v in (DEFAULT_SEMANTIC_CLASSES if class_params is None else class_params).items()
        }
        self.name = str(name)

    def _zone_membership_weight(self, zone: dict, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Tính hàm thuộc mềm [0, 1] (dùng cho ground truth trơn) và khoảng cách dấu tới biên vùng."""
        shape = zone.get("shape", "rect")
        cx = float(zone["cx"])
        cy = float(zone["cy"])
        blend = max(float(zone.get("blend_m", 0.08)), 1e-3)

        if shape == "circle":
            r = float(zone["r"])
            dist = np.hypot(x - cx, y - cy)
            # d_signed < 0 bên trong, > 0 bên ngoài
            d_signed = dist - r
            weight = 0.5 * (1.0 - np.tanh(d_signed / blend))
            return np.clip(weight, 0.0, 1.0), d_signed <= 0.0

        # Hình chữ nhật tổng quát (cho phép xoay góc angle rad)
        hx = float(zone["hx"])
        hy = float(zone["hy"])
        ang = float(zone.get("angle", 0.0))
        ca, sa = np.cos(ang), np.sin(ang)
        dx = x - cx
        dy = y - cy
        u = ca * dx + sa * dy
        v = -sa * dx + ca * dy
        d_signed = np.maximum(np.abs(u) - hx, np.abs(v) - hy)
        weight = 0.5 * (1.0 - np.tanh(d_signed / blend))
        return np.clip(weight, 0.0, 1.0), d_signed <= 0.0

    def eval_classes_batch(self, pts_xy: np.ndarray) -> list:
        """Trả về danh sách nhãn ngữ nghĩa rời rạc c(x, y) cho mảng tọa độ (..., 2)."""
        arr = np.asarray(pts_xy, dtype=float)
        if arr.size == 0:
            return []
        pts = arr.reshape(-1, 2)
        m = pts.shape[0]
        xs = pts[:, 0]
        ys = pts[:, 1]
        labels = np.full(m, self.default_label, dtype=object)
        for z in self.zones:
            _, inside = self._zone_membership_weight(z, xs, ys)
            if np.any(inside):
                labels[inside] = str(z["label"])
        return labels.tolist()

    def eval_class(self, x: float, y: float) -> str:
        return self.eval_classes_batch(np.array([[float(x), float(y)]]))[0]

    def macro_friction_fn(self, x, y):
        """
        Hàm ma sát vĩ mô liên tục trơn tương ứng với bố trí vật liệu (chỉ biết lớp danh định).
        Hỗ trợ cả vô hướng lẫn mảng NumPy.
        """
        is_scalar = np.isscalar(x) and np.isscalar(y)
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        mu_base = float(self.class_params.get(self.default_label, self.class_params["unknown"])["mu_0"])
        mu_arr = np.full_like(xa, mu_base, dtype=float)

        for z in self.zones:
            lbl = str(z["label"])
            mu_z = float(self.class_params.get(lbl, self.class_params["unknown"])["mu_0"])
            w, _ = self._zone_membership_weight(z, xa, ya)
            mu_arr = (1.0 - w) * mu_arr + w * mu_z

        mu_arr = np.clip(mu_arr, 0.06, 0.65)
        return float(mu_arr) if is_scalar else mu_arr

    def transformed(self, dx: float = 0.0, dy: float = 0.0, dtheta: float = 0.0) -> "SemanticFloorMap":
        """Tạo bản sao bản đồ đã tịnh tiến (dx, dy) và xoay góc dtheta quanh gốc tọa độ."""
        c_t, s_t = float(np.cos(dtheta)), float(np.sin(dtheta))
        new_zones = []
        for z in self.zones:
            zn = dict(z)
            cx0, cy0 = float(z["cx"]), float(z["cy"])
            zn["cx"] = c_t * cx0 - s_t * cy0 + float(dx)
            zn["cy"] = s_t * cx0 + c_t * cy0 + float(dy)
            if zn.get("shape", "rect") != "circle":
                zn["angle"] = float(z.get("angle", 0.0)) + float(dtheta)
            new_zones.append(zn)
        return SemanticFloorMap(
            zones=new_zones,
            default_label=self.default_label,
            class_params=self.class_params,
            name=f"{self.name}_transformed",
        )


def make_hidden_oil_macro_fn(
    base_macro_fn,
    oil_patches: list,
    mu_clip: tuple = (0.08, 0.60),
):
    """
    Gắn thêm các vũng dầu/nước tràn bất ngờ NGOÀI bản đồ ngữ nghĩa để mô phỏng sự cố thực tế tại t > 0.
    oil_patches: list các dict {"cx": float, "cy": float, "r": float, "mu": float, "blend_m": float}
    """
    patches = list(oil_patches) if oil_patches is not None else []

    def _macro(x, y):
        is_scalar = np.isscalar(x) and np.isscalar(y)
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        base_val = np.asarray(base_macro_fn(xa, ya), dtype=float)
        out = base_val.copy()
        for p in patches:
            cx = float(p["cx"])
            cy = float(p["cy"])
            r = float(p.get("r", 0.55))
            mu_p = float(p.get("mu", 0.13))
            blend = max(float(p.get("blend_m", 0.08)), 1e-3)
            dist = np.hypot(xa - cx, ya - cy)
            w = np.clip(0.5 * (1.0 - np.tanh((dist - r) / blend)), 0.0, 1.0)
            out = (1.0 - w) * out + w * mu_p
        out = np.clip(out, float(mu_clip[0]), float(mu_clip[1]))
        return float(out) if is_scalar else out

    return _macro


def make_world_semantic_map(map_name: str) -> SemanticFloorMap:
    """
    Bản đồ ngữ nghĩa thiết kế sàn CAD (chỉ gồm bê tông khô, epoxy, wet_risk - không chứa vũng dầu ẩn)
    cho các đấu trường MuJoCo (world6.xml, world7.xml, world8.xml và world* mặc định).
    """
    b = os.path.basename(str(map_name)).lower()
    if b in ("world6.xml", "world6"):
        zones = [
            {"shape": "rect", "cx": -1.85, "cy": 0.45, "hx": 0.75, "hy": 1.15, "angle": 0.0, "label": "epoxy", "blend_m": 0.09},
            {"shape": "rect", "cx": -0.25, "cy": -1.30, "hx": 0.70, "hy": 0.65, "angle": 0.0, "label": "epoxy", "blend_m": 0.09},
            {"shape": "circle", "cx": 1.55, "cy": 0.85, "r": 0.65, "label": "wet_risk", "blend_m": 0.09},
        ]
    elif b in ("world7.xml", "world7"):
        zones = [
            {"shape": "rect", "cx": -1.95, "cy": 0.25, "hx": 0.70, "hy": 1.10, "angle": 0.0, "label": "epoxy", "blend_m": 0.09},
            {"shape": "rect", "cx": -0.15, "cy": 0.85, "hx": 0.85, "hy": 0.70, "angle": 0.0, "label": "epoxy", "blend_m": 0.09},
            {"shape": "circle", "cx": 1.25, "cy": -0.35, "r": 0.60, "label": "wet_risk", "blend_m": 0.09},
        ]
    elif b in ("world8.xml", "world8"):
        zones = [
            {"shape": "rect", "cx": -2.25, "cy": 1.15, "hx": 0.75, "hy": 1.10, "angle": 0.0, "label": "epoxy", "blend_m": 0.09},
            {"shape": "rect", "cx": -0.55, "cy": -0.55, "hx": 0.80, "hy": 0.90, "angle": 0.0, "label": "epoxy", "blend_m": 0.09},
            {"shape": "circle", "cx": 1.75, "cy": 1.45, "r": 0.68, "label": "wet_risk", "blend_m": 0.09},
        ]
    elif b in ("world9.xml", "world9", "map9.xml", "map9"):
        # Sàn xưởng công nghiệp chuẩn: nền bê tông khô (mặc định) + khu gia công Epoxy
        # + khu rửa chi tiết / dung dịch tưới nguội (wet_risk) + góc chuyển tiếp chưa khảo sát (unknown)
        zones = [
            {"shape": "rect", "cx": -0.85, "cy": 0.20, "hx": 0.85, "hy": 1.35, "angle": 0.0, "label": "epoxy", "blend_m": 0.09},
            {"shape": "rect", "cx": 0.85, "cy": 0.15, "hx": 0.78, "hy": 1.05, "angle": 0.0, "label": "wet_risk", "blend_m": 0.09},
            {"shape": "rect", "cx": 2.65, "cy": 0.55, "hx": 0.85, "hy": 1.05, "angle": 0.0, "label": "epoxy", "blend_m": 0.09},
            {"shape": "circle", "cx": -2.10, "cy": -0.35, "r": 0.45, "label": "unknown", "blend_m": 0.09},
        ]
    else:
        zones = [
            {"shape": "rect", "cx": 0.0, "cy": 0.0, "hx": 0.90, "hy": 0.90, "angle": 0.0, "label": "epoxy", "blend_m": 0.09},
        ]
    return SemanticFloorMap(zones=zones, default_label="dry_concrete", name=f"semantic_{b}")


def _make_random_path_waypoints(rng: np.random.Generator, n_pts: int = 135) -> np.ndarray:
    """Sinh quỹ đạo đa dạng (uốn S, dịch chuyển, xoay góc và đổi thang) để kiểm định bất biến tọa độ."""
    length = float(rng.uniform(6.0, 8.2))
    s_raw = np.linspace(0.0, length, n_pts)
    amp1 = float(rng.uniform(0.35, 0.95)) * (1.0 if rng.uniform() > 0.5 else -1.0)
    amp2 = float(rng.uniform(-0.30, 0.30))
    freq1 = float(rng.uniform(0.75, 1.25))
    y_local = amp1 * np.sin(freq1 * np.pi * s_raw / length) + amp2 * np.sin(2.1 * np.pi * s_raw / length)
    pts_local = np.column_stack([s_raw, y_local])

    rot = float(rng.uniform(-np.pi, np.pi))
    shift = rng.uniform(-3.0, 3.0, size=2)
    c_r, s_r = float(np.cos(rot)), float(np.sin(rot))
    R = np.array([[c_r, -s_r], [s_r, c_r]], dtype=float)
    return pts_local @ R.T + shift


def make_random_semantic_map_scenario(
    seed: int,
    road_class: str = "B",
    variation_scale: float = 0.15,
    n_wps: int = 135,
    allow_unknown_zone: bool = False,
) -> dict:
    """
    Sinh một kịch bản bản đồ ngẫu nhiên hoàn chỉnh (Zero-Shot, tách rời với mọi bản đồ khác):
    - Đường đi quay góc và tịnh tiến ngẫu nhiên trong mặt phẳng [-6, 6] m.
    - Ngẫu nhiên 2-3 vùng epoxy, 1-2 vùng wet_risk (và tùy chọn unknown) nằm rải rác trên tuyến.
    - Trả về đầy đủ semantic_map, ground_truth, waypoints, x_grid, y_grid, mu_min_4w_true.
    """
    rng = np.random.default_rng(int(seed))
    wp = _make_random_path_waypoints(rng, n_pts=n_wps)
    diffs = np.diff(wp, axis=0)
    yaw_grid = np.arctan2(np.concatenate([diffs[:, 1], [diffs[-1, 1]]]),
                          np.concatenate([diffs[:, 0], [diffs[-1, 0]]]))

    zones = []
    # 2 vùng epoxy ngẫu nhiên dọc theo tuyến
    for frac in (float(rng.uniform(0.20, 0.38)), float(rng.uniform(0.45, 0.65))):
        idx = int(np.clip(round(frac * (len(wp) - 1)), 5, len(wp) - 6))
        cx = float(wp[idx, 0] + rng.uniform(-0.25, 0.25))
        cy = float(wp[idx, 1] + rng.uniform(-0.25, 0.25))
        zones.append({
            "shape": "rect",
            "cx": cx,
            "cy": cy,
            "hx": float(rng.uniform(0.55, 0.95)),
            "hy": float(rng.uniform(0.50, 0.85)),
            "angle": float(rng.uniform(-np.pi, np.pi)),
            "label": "epoxy",
            "blend_m": float(rng.uniform(0.06, 0.11)),
        })

    # 1 vùng wet_risk ngẫu nhiên
    idx_w = int(np.clip(round(float(rng.uniform(0.68, 0.88)) * (len(wp) - 1)), 5, len(wp) - 6))
    zones.append({
        "shape": "circle",
        "cx": float(wp[idx_w, 0] + rng.uniform(-0.18, 0.18)),
        "cy": float(wp[idx_w, 1] + rng.uniform(-0.18, 0.18)),
        "r": float(rng.uniform(0.45, 0.72)),
        "label": "wet_risk",
        "blend_m": float(rng.uniform(0.06, 0.10)),
    })

    if allow_unknown_zone:
        idx_u = int(np.clip(round(0.12 * (len(wp) - 1)), 3, len(wp) - 4))
        zones.append({
            "shape": "circle",
            "cx": float(wp[idx_u, 0]),
            "cy": float(wp[idx_u, 1]),
            "r": 0.45,
            "label": "unknown",
            "blend_m": 0.08,
        })

    sem_map = SemanticFloorMap(zones=zones, default_label="dry_concrete", name=f"rand_sem_{seed}")
    gt = MultiZoneFrictionGroundTruth(
        macro_fn=sem_map.macro_friction_fn,
        road_class=road_class,
        seed=int(seed) + 5000,
        variation_scale=variation_scale,
    )
    mu_c, mu_4w, _ = gt.eval_path_4w_min(wp[:, 0], wp[:, 1])

    return {
        "seed": int(seed),
        "waypoints": wp,
        "x_grid": wp[:, 0].copy(),
        "y_grid": wp[:, 1].copy(),
        "yaw_grid": yaw_grid,
        "semantic_map": sem_map,
        "ground_truth": gt,
        "mu_center_true": mu_c,
        "mu_min_4w_true": mu_4w,
    }


def generate_semantic_map_dataset(
    seed_start: int,
    count: int,
    road_class: str = "B",
    variation_scale: float = 0.15,
    n_wps: int = 135,
) -> list:
    """Sinh danh sách nhiều bản đồ ngữ nghĩa hoàn toàn độc lập để hiệu chuẩn / kiểm thử Zero-Shot."""
    return [
        make_random_semantic_map_scenario(
            seed=int(seed_start) + i,
            road_class=road_class,
            variation_scale=variation_scale,
            n_wps=n_wps,
        )
        for i in range(int(count))
    ]


class SemanticFrictionEstimator:
    """
    Bộ ước lượng ma sát tiền khởi hành Zero-Shot dựa trên bản đồ ngữ nghĩa CAD/BIM
    và hiệu chuẩn Split Conformal đa bản đồ (Tầng 1 - Hướng đi 1).
    """

    def __init__(
        self,
        class_params: dict = None,
        q_hat_mu: float = 1.96,
        alpha_mu: float = 0.05,
        mu_min: float = 0.08,
        mu_max: float = 0.55,
        query_mode: str = "footprint",
        L_x: float = L_X,
        L_y: float = L_Y,
        lateral_margin: float = 0.05,
        semantic_map=None,
    ):
        if query_mode not in ("footprint", "center"):
            raise ValueError(f"query_mode không hợp lệ: '{query_mode}'")
        src_cls = DEFAULT_SEMANTIC_CLASSES if class_params is None else class_params
        self.class_params = {
            str(k): {"mu_0": float(v["mu_0"]), "sigma_0": float(v["sigma_0"])}
            for k, v in src_cls.items()
        }
        if "unknown" not in self.class_params:
            self.class_params["unknown"] = dict(DEFAULT_SEMANTIC_CLASSES["unknown"])

        self.q_hat_mu = float(q_hat_mu)
        self.alpha_mu = float(alpha_mu)
        self.mu_min = float(mu_min)
        self.mu_max = float(mu_max)
        self.query_mode = str(query_mode)
        self.L_x = float(L_x)
        self.L_y = float(L_y)
        self.lateral_margin = float(lateral_margin)
        self.semantic_map = semantic_map

        self.cal_meta = {}
        self.last_predict_time_ms = 0.0

    def set_semantic_map(self, semantic_map) -> "SemanticFrictionEstimator":
        """Gắn bản đồ ngữ nghĩa sàn của nhiệm vụ hiện tại (Zero-Shot tại t = 0)."""
        self.semantic_map = semantic_map
        return self

    def _lookup_classes(self, pts_xy: np.ndarray, semantic_map=None) -> tuple:
        """Tra cứu (mu_0, sigma_0, labels) tại danh sách điểm (..., 2)."""
        arr = np.asarray(pts_xy, dtype=float)
        pts = arr.reshape(-1, 2) if arr.size > 0 else np.zeros((0, 2), dtype=float)
        m = pts.shape[0]
        s_map = self.semantic_map if semantic_map is None else semantic_map

        if s_map is None:
            labels = ["unknown"] * m
        elif isinstance(s_map, SemanticFloorMap):
            labels = s_map.eval_classes_batch(pts)
        elif hasattr(s_map, "eval_classes_batch"):
            labels = list(s_map.eval_classes_batch(pts))
        elif callable(s_map):
            labels = [str(s_map(float(pts[i, 0]), float(pts[i, 1]))) for i in range(m)]
        else:
            raise TypeError("semantic_map phải là SemanticFloorMap hoặc callable(x, y) -> str")

        mu_0_arr = np.empty(m, dtype=float)
        sig_0_arr = np.empty(m, dtype=float)
        unk = self.class_params["unknown"]
        for i, lbl in enumerate(labels):
            p = self.class_params.get(str(lbl), unk)
            mu_0_arr[i] = float(p["mu_0"])
            sig_0_arr[i] = float(p["sigma_0"])
        return mu_0_arr, sig_0_arr, labels

    def predict_along_path(
        self,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        mode: str = "conformal_lcb",
        yaw_grid: np.ndarray = None,
        query_mode: str = None,
        semantic_map=None,
    ) -> dict:
        """
        Dự báo trường ma sát Zero-Shot dọc quỹ đạo A* tại t = 0.
        Tương thích 100% chữ ký và các khóa trả về của CoupledVelocityTubeH124.build_offline_tube().
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

        if yaw_grid is None:
            if n >= 2:
                dx = np.gradient(xg)
                dy = np.gradient(yg)
                yg_yaw = np.arctan2(dy, dx)
            else:
                yg_yaw = np.zeros(n, dtype=float)
        else:
            yg_yaw = np.asarray(yaw_grid, dtype=float).ravel()

        s_map = self.semantic_map if semantic_map is None else semantic_map

        if qm == "footprint":
            fp_pts = np.asarray(
                compute_footprint_query_points(
                    xg, yg, yg_yaw,
                    L_x=self.L_x, L_y=self.L_y, lateral_margin=self.lateral_margin,
                ),
                dtype=float,
            ).reshape(-1, 2)
            k_fp = len(fp_pts) // n
            mu0_all, sig0_all, lbl_all = self._lookup_classes(fp_pts, semantic_map=s_map)
            mu0_mat = mu0_all.reshape(n, k_fp)
            sig0_mat = sig0_all.reshape(n, k_fp)
            lcb_mat = mu0_mat - self.q_hat_mu * sig0_mat
            worst_k = np.argmin(lcb_mat, axis=1)
            row_idx = np.arange(n)

            mu_mean = mu0_mat[row_idx, worst_k]
            sigma_arr = sig0_mat[row_idx, worst_k]
            mu_lcb = np.clip(lcb_mat[row_idx, worst_k], self.mu_min, self.mu_max)

            lbl_mat = [lbl_all[i * k_fp + int(worst_k[i])] for i in range(n)]
            center_lbls = [lbl_all[i * k_fp] for i in range(n)]
        else:
            pts = np.column_stack([xg, yg])
            mu_mean, sigma_arr, lbl_mat = self._lookup_classes(pts, semantic_map=s_map)
            mu_lcb = np.clip(mu_mean - self.q_hat_mu * sigma_arr, self.mu_min, self.mu_max)
            center_lbls = list(lbl_mat)

        if mode == "conformal_lcb":
            mu_tube = mu_lcb.copy()
            is_conf = True
        else:
            mu_tube = np.clip(mu_mean, self.mu_min, self.mu_max)
            is_conf = False

        unknown_frac = float(np.mean([1.0 if lb == "unknown" else 0.0 for lb in lbl_mat]))
        clamped_frac = float(np.mean(mu_lcb <= self.mu_min + 1e-9))

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
            "q_hat_mu": float(self.q_hat_mu),
            "alpha_mu": float(self.alpha_mu),
            "n_support": 0,
            "erosion_radius_m": 0.0,
            "use_raster_cache": False,
            "grid_res_m": 0.0,
            "prior_region_frac": round(unknown_frac, 4),
            "clamped_mu_min_frac": round(clamped_frac, 4),
            "predict_time_ms": float(elapsed_ms),
            "class_names": lbl_mat,
            "center_class_names": center_lbls,
        }

    def calibrate_class_conformal(
        self,
        cal_maps_trajs: list,
        alpha_mu: float = None,
        query_mode: str = None,
    ) -> dict:
        """
        Hiệu chuẩn Split Conformal đa bản đồ trên tập D_cal^maps tách rời bản đồ kiểm thử:
            R_j = max_{s, k in footprint} max(0, mu_0(c(p_{j,k}(s))) - mu_{min,4w}^true(s)) / sigma_0(c(p_{j,k}(s)))
        """
        n_cal = len(cal_maps_trajs)
        if n_cal < 10:
            raise ValueError(f"Số quỹ đạo hiệu chuẩn quá nhỏ ({n_cal}); yêu cầu >= 10.")

        alpha_use = float(self.alpha_mu if alpha_mu is None else alpha_mu)
        qm = self.query_mode if query_mode is None else str(query_mode)
        scores = np.empty(n_cal, dtype=float)

        for j, tr in enumerate(cal_maps_trajs):
            xg = np.asarray(tr["x_grid"], dtype=float).ravel()
            yg = np.asarray(tr["y_grid"], dtype=float).ravel()
            mu_min_4w = np.asarray(tr["mu_min_4w_true"], dtype=float).ravel()
            s_map = tr.get("semantic_map", self.semantic_map)
            if "yaw_grid" in tr and tr["yaw_grid"] is not None:
                yw = np.asarray(tr["yaw_grid"], dtype=float).ravel()
            else:
                dx = np.gradient(xg)
                dy = np.gradient(yg)
                yw = np.arctan2(dy, dx)

            if qm == "footprint":
                fp_pts = np.asarray(
                    compute_footprint_query_points(
                        xg, yg, yw, L_x=self.L_x, L_y=self.L_y, lateral_margin=self.lateral_margin
                    ),
                    dtype=float,
                ).reshape(-1, 2)
                k_fp = len(fp_pts) // len(xg)
                mu0_all, sig0_all, _ = self._lookup_classes(fp_pts, semantic_map=s_map)
                mu0_mat = mu0_all.reshape(len(xg), k_fp)
                sig0_mat = sig0_all.reshape(len(xg), k_fp)
                over_mat = np.maximum(0.0, mu0_mat - mu_min_4w[:, None]) / np.maximum(sig0_mat, 1e-6)
                scores[j] = float(np.max(np.min(over_mat, axis=1)))
            else:
                pts = np.column_stack([xg, yg])
                mu0_all, sig0_all, _ = self._lookup_classes(pts, semantic_map=s_map)
                over = np.maximum(0.0, mu0_all - mu_min_4w) / np.maximum(sig0_all, 1e-6)
                scores[j] = float(np.max(over))

        scores_sorted = np.sort(scores)
        k_star = int(np.clip(np.ceil((n_cal + 1) * (1.0 - alpha_use)), 1, n_cal))
        q_val = float(scores_sorted[k_star - 1])

        self.alpha_mu = alpha_use
        self.q_hat_mu = q_val

        # Tự tương quan bậc 1 (Kiểm tra giả thiết khả hoán vị giữa các bản đồ D_cal)
        sc_c = scores - float(np.mean(scores))
        denom = float(np.sum(sc_c ** 2))
        r1 = float(np.sum(sc_c[:-1] * sc_c[1:]) / denom) if denom > 1e-12 else 0.0
        z_stat = abs(r1) * float(np.sqrt(n_cal))

        k_025 = int(np.clip(np.ceil((n_cal + 1) * 0.975), 1, n_cal))
        q_025 = float(scores_sorted[k_025 - 1])

        summary = {
            "method": "Zero-Shot Multi-Map Class Split Conformal",
            "query_mode": qm,
            "n_cal": n_cal,
            "alpha_mu": alpha_use,
            "target_coverage": round(1.0 - alpha_use, 4),
            "k_star": k_star,
            "q_hat_mu": round(q_val, 6),
            "score_mean": round(float(np.mean(scores)), 6),
            "score_median": round(float(np.median(scores)), 6),
            "score_max": round(float(np.max(scores)), 6),
            "lag1_autocorr": round(r1, 4),
            "lag1_z_stat": round(z_stat, 4),
            "lag1_no_serial_dep_detected": bool(z_stat < 1.96),
            "exchangeability_ok": bool(z_stat < 1.96),
            "bonferroni_joint_budget": {
                "alpha_eta": 0.05,
                "alpha_fz": 0.025,
                "alpha_mu_025": 0.025,
                "alpha_total_with_025": 0.10,
                "q_hat_mu_at_alpha_025": round(q_025, 6),
                "k_star_at_alpha_025": k_025,
            },
            "scores": [round(float(v), 6) for v in scores],
        }
        self.cal_meta = {k: v for k, v in summary.items() if k != "scores"}
        return summary

    def evaluate_coverage(
        self,
        test_maps_trajs: list,
        mode: str = "conformal_lcb",
        constant_override: float = None,
        query_mode: str = None,
    ) -> dict:
        """Chấm tỷ lệ phủ toàn quỹ đạo & điểm trên tập bản đồ mới tách biệt (không chạy trước)."""
        n_test = len(test_maps_trajs)
        if n_test == 0:
            raise ValueError("test_maps_trajs không được rỗng.")

        n_covered = 0
        total_pts = 0
        covered_pts = 0
        worst_over_list = []
        mean_gaps = []
        max_gaps = []
        unknown_fracs = []
        clamp_fracs = []
        pred_times = []

        for tr in test_maps_trajs:
            xg = np.asarray(tr["x_grid"], dtype=float).ravel()
            yg = np.asarray(tr["y_grid"], dtype=float).ravel()
            mu_min_4w = np.asarray(tr["mu_min_4w_true"], dtype=float).ravel()
            s_map = tr.get("semantic_map", self.semantic_map)
            yw = tr.get("yaw_grid", None)

            if constant_override is not None:
                t0 = time.perf_counter()
                mu_pred = np.full(len(xg), float(constant_override), dtype=float)
                pred_times.append((time.perf_counter() - t0) * 1000.0)
                unknown_fracs.append(0.0)
                clamp_fracs.append(1.0 if float(constant_override) <= self.mu_min + 1e-9 else 0.0)
            else:
                res = self.predict_along_path(
                    xg, yg, mode=mode, yaw_grid=yw, query_mode=query_mode, semantic_map=s_map
                )
                mu_pred = res["mu_tube"]
                pred_times.append(res["predict_time_ms"])
                unknown_fracs.append(float(res.get("prior_region_frac", 0.0)))
                clamp_fracs.append(float(res.get("clamped_mu_min_frac", 0.0)))

            diff_over = mu_pred - mu_min_4w
            pts_ok = diff_over <= 1e-9
            covered_pts += int(np.sum(pts_ok))
            total_pts += len(mu_pred)
            if bool(np.all(pts_ok)):
                n_covered += 1

            worst_over_list.append(float(np.max(np.maximum(0.0, diff_over))))
            gap_arr = mu_min_4w - mu_pred
            mean_gaps.append(float(np.mean(gap_arr)))
            max_gaps.append(float(np.max(gap_arr)))

        cov_rate = float(n_covered) / float(n_test)
        p_target = 1.0 - float(self.alpha_mu)
        cp_lb = 0.0 if n_covered == 0 else float(beta.ppf(0.05, n_covered, n_test - n_covered + 1))
        ci_low_2 = 0.0 if n_covered == 0 else float(beta.ppf(0.025, n_covered, n_test - n_covered + 1))
        ci_high_2 = 1.0 if n_covered >= n_test else float(beta.ppf(0.975, n_covered + 1, n_test - n_covered))
        p_val_binom = float(binom.cdf(n_covered, n_test, p_target))

        return {
            "mode": "constant_override" if constant_override is not None else mode,
            "query_mode": "override" if constant_override is not None else (self.query_mode if query_mode is None else query_mode),
            "n_test": n_test,
            "n_traj_covered": n_covered,
            "traj_coverage_rate": round(cov_rate, 4),
            "pointwise_coverage_rate": round(float(covered_pts) / max(1, total_pts), 4),
            "clopper_pearson_95_lb": round(cp_lb, 4),
            "clopper_pearson_95_ci_2sided": [round(ci_low_2, 4), round(ci_high_2, 4)],
            "binom_p_value_one_sided": round(p_val_binom, 4),
            "mean_worst_overestimation": round(float(np.mean(worst_over_list)), 5),
            "max_worst_overestimation": round(float(np.max(worst_over_list)), 5),
            "mean_conservatism_gap": round(float(np.mean(mean_gaps)), 5),
            "max_conservatism_gap": round(float(np.max(max_gaps)), 5),
            "mean_unknown_region_frac": round(float(np.mean(unknown_fracs)), 4),
            "mean_clamped_mu_min_frac": round(float(np.mean(clamp_fracs)), 4),
            "mean_predict_time_ms": round(float(np.mean(pred_times)), 4),
        }

    def save_calibration(self, json_path: str) -> str:
        """Lưu cấu hình lớp ngữ nghĩa & lượng tử Conformal ra file JSON."""
        abs_p = os.path.abspath(json_path)
        os.makedirs(os.path.dirname(abs_p), exist_ok=True)
        payload = {
            "version": "1.0_zeroshot_semantic",
            "class_params": self.class_params,
            "hyperparams": {
                "mu_min": self.mu_min,
                "mu_max": self.mu_max,
                "query_mode": self.query_mode,
                "L_x": self.L_x,
                "L_y": self.L_y,
                "lateral_margin": self.lateral_margin,
            },
            "q_hat_mu": float(self.q_hat_mu),
            "alpha_mu": float(self.alpha_mu),
            "calibration_summary": self.cal_meta,
        }
        tmp_p = abs_p + ".tmp"
        with open(tmp_p, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp_p, abs_p)
        return abs_p

    @classmethod
    def load_calibration(cls, json_path: str, semantic_map=None) -> "SemanticFrictionEstimator":
        """Nạp cấu hình Conformal ngữ nghĩa từ file JSON."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        hp = data.get("hyperparams", {})
        est = cls(
            class_params=data.get("class_params", DEFAULT_SEMANTIC_CLASSES),
            q_hat_mu=float(data.get("q_hat_mu", 1.96)),
            alpha_mu=float(data.get("alpha_mu", 0.05)),
            mu_min=float(hp.get("mu_min", 0.08)),
            mu_max=float(hp.get("mu_max", 0.55)),
            query_mode=str(hp.get("query_mode", "footprint")),
            L_x=float(hp.get("L_x", L_X)),
            L_y=float(hp.get("L_y", L_Y)),
            lateral_margin=float(hp.get("lateral_margin", 0.05)),
            semantic_map=semantic_map,
        )
        est.cal_meta = data.get("calibration_summary", {})
        return est

    @classmethod
    def from_calibration_or_default(cls, json_path: str = None, semantic_map=None) -> "SemanticFrictionEstimator":
        """Nạp file hiệu chuẩn mặc định nếu có, ngược lại dùng tham số Conformal danh định."""
        if json_path is None:
            json_path = os.path.join(_root, "config", "conformal_calibration_mu_semantic.json")
        if os.path.exists(json_path):
            return cls.load_calibration(json_path, semantic_map=semantic_map)
        return cls(semantic_map=semantic_map)
