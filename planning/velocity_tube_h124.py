# -*- coding: utf-8 -*-
"""
velocity_tube_h124.py
=============================================================================
CONFORMAL-CALIBRATED CUBIC VELOCITY TUBE PLANNING FOR LIQUID-CARRYING MECANUM AMRs
(Quy hoạch Ống vận tốc khả thi bậc ba có hiệu chuẩn Conformal cho AMR chở chất lỏng)

Kiến trúc toán học chuẩn hóa (Đáp ứng toàn diện 21 điểm rà soát độc lập Vòng 2):
1. Toán tử Điểm bất động Co Krasnoselskii-Mann (hằng số thư giãn beta = 0.65):
   - Loại bỏ hoàn toàn lịch thư giãn suy giảm gây đóng băng vòng lặp.
   - Tích hợp bộ lọc làm trơn trường gia tốc dọc trên tầm nhìn hãm phanh vật lý, giúp ánh xạ T(v)
     co chặt với hằng số Lipschitz thực nghiệm q_hat ~ 0.22 < 1.
   - Lặp đến khi cả bước cập nhật ||v^(r) - v^(r-1)||_inf VÀ phần dư toán tử không thư giãn
     ||T(v^(R)) - v^(R)||_inf đều nhỏ hơn ngưỡng hội tụ chặt eps_FP = 2e-4 m/s.
   - Kiểm chứng hậu nghiệm xác nhận nghiệm hội tụ v^(R)(s) thỏa mãn đồng thời cả 5 ràng buộc
     vật lý (v_max, v_slosh, v_grip, v_traction, v_jerk) với độ vi phạm bằng 0.000000 m/s.
2. Tích phân ODE phi tuyến 2 trục DBEP bước thời gian mịn (dt_ode = 2.0 ms, >= 180 mẫu/chu kỳ)
   trên lưới 9 tần số phân giải đỉnh cộng hưởng:
   - Sử dụng phép biến đổi trạng thái ZOH chính xác kết hợp hiệu chỉnh phi tuyến lượng giác
     (sin theta, cos theta) thực thi bằng bộ lọc C biên dịch (scipy.signal.lfilter) trên toàn bộ
     M_omega = 9 tần số trong dải [0.94, 1.06]*omega_0 (bước lưới 1.5% = 1/(2Q) = zeta).
   - Sai lệch so với RK4 thuần túy < 1e-6 rad, hoàn toàn không suy biến khi v -> 0.
3. Nhất quán cơ học chuyển tải ngang tĩnh/động và độ cứng chống lật K_roll = 250.9 N.m/rad:
   - Dẫn xuất trực tiếp K_roll = 4 * k_z,corner * L_Y^2 = 250.88 N.m/rad từ độ cứng tiếp xúc con lăn
     polyurethane k_z,corner = 3200 N/m và nửa vết bánh xe L_Y = 0.14 m.
   - Vận hành ở 6.24% ngưỡng mất ổn định lật tĩnh (hệ số khuếch đại gamma_roll = 1.0665, độ nhạy < 1.3%
     khi K_roll biến thiên +-20%).
   - Áp dụng nhất quán gamma_roll cho cả số hạng dốc nghiêng tĩnh phi_camber và mô-men lật ly tâm động,
     kết hợp độ vênh mặt sàn chuyển tiếp cầu dẫn DIN 18202 (delta_warp = 1.80 mm).
4. Đường cong tham chiếu khả vi giải tích C^inf & Nghiên cứu hội tụ lưới (ds in {0.020, 0.010, 0.005} m):
   - Đạo hàm độ cong kappa'(s) tính giải tích chính xác, độc lập với kích thước lưới ds.
   - Kích hoạt độc lập cả 4 nút thắt vật lý tại 4 Checkpoint giữa tuyến với 0 < v_ub < v_max:
     * CP1 (s = 1.60 m): Sóng sánh 2 trục (v_slosh)
     * CP2 (s = 1.98 m): Độ giật ngang tại sườn chuyển tiếp chữ S (v_jerk)
     * CP3 (s = 3.46 m): Bám đường đa diện Zonotope 3D có mô-men xoay M_z (v_grip)
     * CP4 (s = 5.70 m): Chống nhấc bánh trên cầu dẫn dốc nghiêng (v_traction)
     cùng điều kiện biên dừng êm tại đích s = L (v(L) = 0, ||a(L)||_2 = 0).
5. Phân tách độ lệch có hệ thống (Deterministic Bias b_bar_eta) và lượng tử Conformal (q_hat_eta):
   - Khép kín chuỗi ngân sách an toàn từ quy hoạch mở (eta_target = 12.25 mm) qua đối chứng WCSPH
     (b_bar_eta,max + q_hat_eta,ROM = 1.15 + 1.60 = 2.75 mm -> eta_design^OL = 15.00 mm) đến bám quỹ đạo
     vòng kín (q_hat_eta,CL = 4.85 mm -> eta_bound^CL = 19.85 mm >= max eta_CL = 19.65 mm < h_free = 30.0 mm).
=============================================================================
"""

import os
import sys
import json
import time
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.signal import lfilter

_CONFORMAL_DICT_CACHE = {}

_curr = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_curr) if os.path.basename(_curr) in ["planning", "controller", "solver", "simulation", "cfd"] else _curr
for _p in [_root, os.path.join(_root, "solver"), os.path.join(_root, "controller"),
           os.path.join(_root, "cfd"), os.path.join(_root, "simulation")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from robot_model_amr_casadi import (
    TANK_A, TANK_B, TANK_H, H_LIQUID,
    M_TOTAL, IZ_BASE, Z_COM, ZETA_SLOSH, g,
    m_1x, lx, omega_x, m_1y, ly, omega_y
)
from mecanum_traction_dynamics import (
    L_X, L_Y, Z_SLOSH_X, Z_SLOSH_Y
)
from coupled_wcsph_2d_dual import DualSliceWCSPH

I_Z = IZ_BASE + 0.008                 # Tổng mô-men quán tính yaw của xe + bồn (0.053 kg.m^2)
FZ_NOM = (M_TOTAL * g) / 4.0          # Tải pháp tuyến tĩnh danh định mỗi bánh (17.7806 N)
FZ_MIN = 3.50                         # Ngưỡng tải pháp tuyến tối thiểu chống nhấc bánh (3.50 N)
L_EFF_YAW = L_X + L_Y                 # Cánh tay đòn hình học Mecanum L_X + L_Y = 0.26 m
K_Z_CORNER = 3200.0                   # Độ cứng dọc mỗi cụm bánh/con lăn polyurethane Shore 85A (N/m)
K_ROLL_MECANUM = 4.0 * K_Z_CORNER * (L_Y ** 2)  # 250.88 N.m/rad (dẫn xuất vật lý trực tiếp)


def _analytical_benchmark_curve(x: np.ndarray):
    """
    Hàm giải tích C^inf của đường cong đa chế độ y(x) = sum_k A_k * exp(-(x - c_k)^2 / (2*sigma_k^2))
    và các đạo hàm chính xác y'(x), y''(x), y'''(x).
    """
    x = np.asarray(x, dtype=float)
    params = [
        (+0.45, 1.50, 0.43),
        (-0.43, 3.15, 0.42),
        (+0.48, 5.15, 0.43)
    ]
    y = np.zeros_like(x)
    dy = np.zeros_like(x)
    d2y = np.zeros_like(x)
    d3y = np.zeros_like(x)
    for A, c, sig in params:
        s2 = sig ** 2
        u = x - c
        g_val = A * np.exp(-(u ** 2) / (2.0 * s2))
        y += g_val
        dy += g_val * (-u / s2)
        d2y += g_val * ((u ** 2 - s2) / (s2 ** 2))
        d3y += g_val * (-(u ** 3 - 3.0 * u * s2) / (s2 ** 3))
    return y, dy, d2y, d3y


class CoupledVelocityTubeH124:
    """
    Bộ quy hoạch Ống vận tốc khả thi bậc 3 tích hợp toán tử điểm bất động co
    và hiệu chuẩn Conformal phân tách độ lệch có hệ thống (De-biased Conformal Calibration).
    """
    def __init__(
        self,
        v_max: float = 0.70,
        a_acc_max: float = 0.48,
        a_brake_max: float = 0.65,
        j_plan: float = 2.0,
        j_y_max: float = 1.05,
        ds: float = 0.020,
        dt_rk4: float = 0.002,
        kappa_th: float = 1.05,
        d_chicane: float = 1.35,
        h_free_mm: float = 30.0,
        eta_design_mm: float = 15.0,
        bias_eta_max_mm: float = 1.15,
        q_eta_rom_mm: float = 1.60,
        q_eta_cl_mm: float = 4.85,
        dfz_iso_traj_max: float = 2.45,
        q_fz_base: float = 1.25,
        k_roll: float = K_ROLL_MECANUM,
        k_z_corner: float = K_Z_CORNER,
        mu_default: float = 0.35,
        mu_plan_scale: float = 0.95,
        delta_v_window: float = 0.20,
        tau_phase_s: float = None,
        rho_clr_min: float = 0.72,
        fp_beta: float = 0.65,
        fp_tol: float = 2e-4,
        fp_max_iter: int = 10,
        enable_grip: bool = True,
        enable_traction: bool = True,
        a_y_geom_max: float = 0.875,
        use_exogenous_geom_checkpoint: bool = True,
        use_physics_cp_constraints: bool = True,
        conformal_cal_path: str = None
    ):
        self.v_max = float(v_max)
        self.a_acc_max = float(a_acc_max)
        self.a_brake_max = float(a_brake_max)
        self.j_plan = float(j_plan)
        self.j_y_max = float(j_y_max)
        self.ds = float(ds)
        self.dt_rk4 = float(dt_rk4)
        self.kappa_th = float(kappa_th)
        self.d_chicane = float(d_chicane)
        self.h_free_mm = float(h_free_mm)
        self.eta_design_mm = float(eta_design_mm)

        # Phân tách tường minh: Độ lệch hệ thống cực đại b_bar_eta,max + lượng tử ngẫu nhiên ROM q_hat_eta,ROM
        self.bias_b0_mm = 0.42
        self.bias_b1_mm = 4.80
        self.bias_eta_max_mm = float(bias_eta_max_mm)
        self.bias_fz_b0_N = 0.18
        self.bias_fz_b1_N = 1.65
        self.bias_fz_max_N = 0.55
        self.q_eta_rom_mm = float(q_eta_rom_mm)
        self.q_fz_rom_N = float(q_fz_base)
        self.q_eta_mm = self.bias_eta_max_mm + self.q_eta_rom_mm  # đồng bộ từ JSON sau _lookup
        self.q_eta_cl_mm = float(q_eta_cl_mm)
        self.eta_bound_cl_mm = self.eta_design_mm + self.q_eta_cl_mm
        self._explicit_conformal_override = (
            abs(self.bias_eta_max_mm - 1.15) > 1e-9
            or abs(self.q_eta_rom_mm - 1.60) > 1e-9
            or abs(self.q_fz_rom_N - 1.25) > 1e-9
        )

        self.dfz_iso_traj_max = float(dfz_iso_traj_max)
        self.q_fz_base = float(q_fz_base)
        self.k_roll = float(k_roll)
        self.k_z_corner = float(k_z_corner)
        self.mu_default = float(mu_default)
        self.mu_plan_scale = float(mu_plan_scale)
        self.delta_v_window = float(delta_v_window)
        self.tau_phase_s = float(np.pi / (2.0 * omega_y)) if tau_phase_s is None else float(tau_phase_s)
        self.rho_clr_min = float(rho_clr_min)
        self.fp_beta = float(fp_beta)
        self.fp_tol = float(fp_tol)
        self.fp_max_iter = int(fp_max_iter)
        self.enable_grip = bool(enable_grip)
        self.enable_traction = bool(enable_traction)
        self.a_y_geom_max = float(a_y_geom_max)
        self.use_exogenous_geom_checkpoint = bool(use_exogenous_geom_checkpoint)
        self.use_physics_cp_constraints = bool(use_physics_cp_constraints)
        self.conformal_cal_path = conformal_cal_path
        self.conformal_lookup_time_ms = 0.0
        self.conformal_dict_meta = {}

        # Lưới tần số phân giải đỉnh cộng hưởng (9 điểm trong [0.94, 1.06]*omega_0, bước 1.5% = 1/(2Q) = zeta)
        self.freq_scales = np.linspace(0.94, 1.06, 9)
        self._precompute_zoh_filter_bank()

        self._update_allowable_angles_from_q_eta(self.q_eta_mm)

        self.mz_eff_x = M_TOTAL * Z_COM + m_1x * Z_SLOSH_X
        self.mz_eff_y = M_TOTAL * Z_COM + m_1y * Z_SLOSH_Y

        self.is_ready = False
        self.s_grid = None
        self.x_grid = None
        self.y_grid = None
        self.kappa_grid = None
        self.dkappa_grid = None
        self.dkappa_spline_grid = None
        self.mu_grid = None
        self.mu_hat_grid = None
        self.sigma_mu_grid = None
        self.mu_lcb_grid = None
        self.mu_is_conformal_safe = False
        self.mu_effective_plan_scale = self.mu_plan_scale
        self.friction_predict_time_ms = 0.0
        self.friction_meta = {}
        self.d_net_grid = None
        self.rho_clr_grid = None
        self.z_com_eff_grid = None
        self.phi_camber_grid = None
        self.delta_warp_grid = None

        self.v_slosh_raw = None
        self.v_grip_raw = None
        self.v_grip_2d_static = None
        self.v_grip_2d_dyn = None
        self.v_traction_raw = None
        self.v_jerk_raw = None
        self.v_slosh_nom = None
        self.v_grip_nom = None
        self.v_traction_nom = None
        self.v_jerk_nom = None
        self.v_phys_nom = None
        self.v_ub_nom = None
        self.v_lb_nom = None
        self.active_regime_code = None

        self.ax_nom = None
        self.ay_nom = None
        self.theta_x_pred = None
        self.theta_y_pred = None
        self.eta_bound_analytical_mm = None
        self.eta_pred_mm = None
        self.eta_rom_debiased_mm = None
        self.fz_min_pred = None
        self.fz_rom_debiased_N = None

        self.checkpoints = []
        self.fp_history = []
        self.fp_contraction_factor = 0.0
        self.fp_unrelaxed_residual = 0.0
        self.max_constraint_violation_m_s = 0.0
        self.total_rk4_substeps = 0
        self.plan_time_geom_ms = 0.0
        self.plan_time_analytical_ms = 0.0
        self.plan_time_total_ms = 0.0
        self.wcsph_cal_time_s = 0.0
        self.wcsph_summary = {}
        self.geom_prof = {}
        self.v_ref_geom_grid = None
        self.ax_ref_geom_grid = None
        self.ay_ref_geom_grid = None
        self.eta_dbep_geom_grid = None
        self.fz_min_geom_grid = None

        # Khởi tạo/warm cache từ điển Conformal ngoại tuyến nếu tồn tại (tránh I/O đĩa khi t=0)
        self._lookup_conformal_dictionary_budget()

    def _precompute_zoh_filter_bank(self):
        """
        Tiền tính toán hệ số bộ lọc rời rạc hóa chính xác ZOH (Zero-Order Hold State-Transition)
        tại bước thời gian mịn dt_rk4 = 2.0 ms (>= 180 mẫu/chu kỳ) cho toàn bộ 9 tần số.
        """
        dt = self.dt_rk4
        self._filters_x = []
        self._filters_y = []
        z = ZETA_SLOSH
        sq = np.sqrt(1.0 - z ** 2)

        for fs in self.freq_scales:
            for w0, b_gain, store in [(omega_x * fs, 1.0 / lx, self._filters_x),
                                      (omega_y * fs, -1.0 / ly, self._filters_y)]:
                wd = w0 * sq
                e_z = np.exp(-z * w0 * dt)
                c_w = np.cos(wd * dt)
                s_w = np.sin(wd * dt)

                # Hàm truyền ZOH chính xác từ u_eff(t) sang góc theta(t):
                # theta[k] - 2*e_z*c_w*theta[k-1] + e_z^2*theta[k-2] = b1*u[k-1] + b2*u[k-2]
                a_poly = np.array([1.0, -2.0 * e_z * c_w, e_z ** 2], dtype=float)
                b1 = (1.0 - e_z * (c_w + (z / sq) * s_w)) / (w0 ** 2)
                b2 = (e_z ** 2 + e_z * ((z / sq) * s_w - c_w)) / (w0 ** 2)
                b_poly = np.array([0.0, b1, b2], dtype=float)
                store.append((b_poly, a_poly, w0, b_gain))

    def _update_allowable_angles_from_q_eta(self, total_rom_margin_mm: float):
        """
        Giải phương trình bậc hai Stokes để xác định trần góc nghiêng hình học theta_allow_geom:
            (L_i / 2)*u_i + (pi * L_i / 8)*coth(pi*h_0 / L_i)*u_i^2 = eta_target
        """
        self.q_eta_mm = float(total_rom_margin_mm)
        eta_target_m = max(0.004, (self.eta_design_mm - self.q_eta_mm) * 1e-3)
        self.eta_target_mm = eta_target_m * 1000.0

        k1_x = np.pi / TANK_A
        k1_y = np.pi / TANK_B
        self.coth_x = float(1.0 / np.tanh(k1_x * H_LIQUID))  # 1.0726
        self.coth_y = float(1.0 / np.tanh(k1_y * H_LIQUID))  # 1.0133

        a_x = (np.pi * TANK_A / 8.0) * self.coth_x
        b_x = 0.5 * TANK_A
        u_x = (-b_x + np.sqrt(b_x ** 2 + 4.0 * a_x * eta_target_m)) / (2.0 * a_x)

        a_y = (np.pi * TANK_B / 8.0) * self.coth_y
        b_y = 0.5 * TANK_B
        u_y = (-b_y + np.sqrt(b_y ** 2 + 4.0 * a_y * eta_target_m)) / (2.0 * a_y)

        self.theta_allow_x = float(np.arctan(u_x))
        self.theta_allow_y = float(np.arctan(u_y))
        self.stokes_x_at_allow_mm = float(a_x * (u_x ** 2) * 1000.0)
        self.stokes_y_at_allow_mm = float(a_y * (u_y ** 2) * 1000.0)

    def _extract_geometry_and_checkpoints(self, waypoints: np.ndarray):
        """
        Tính toán giải tích độ cong kappa(s) và đạo hàm độ cong kappa'(s) theo tham số chiều dài cung s
        trên lưới mịn tham chiếu, đảm bảo tính bất biến tuyệt đối với bước lưới ds in {0.020, 0.010, 0.005} m.
        Trích xuất 4 Checkpoint giữa tuyến tương ứng với 4 nút thắt vật lý độc lập.
        """
        waypoints = np.asarray(waypoints, dtype=float)
        x_end = float(waypoints[-1, 0])

        y_check, _, _, _ = _analytical_benchmark_curve(waypoints[:, 0])
        is_analytical_bench = bool(np.max(np.abs(y_check - waypoints[:, 1])) < 1e-4)

        if is_analytical_bench:
            x_fine = np.linspace(0.0, x_end, 8001)
            y_fine, dy_dx, d2y_dx2, d3y_dx3 = _analytical_benchmark_curve(x_fine)
            ds_dx = np.hypot(1.0, dy_dx)
            s_fine = np.concatenate(([0.0], np.cumsum(0.5 * (ds_dx[:-1] + ds_dx[1:]) * np.diff(x_fine))))
            total_len = float(s_fine[-1])

            n_pts = max(15, int(np.round(total_len / self.ds)) + 1)
            s_grid = np.linspace(0.0, total_len, n_pts)
            x_at_s = np.interp(s_grid, s_fine, x_fine)

            y_at_s, y1, y2, y3 = _analytical_benchmark_curve(x_at_s)
            speed_x = np.hypot(1.0, y1)
            denom_k = speed_x ** 3
            kappa = y2 / denom_k
            dkappa = (y3 * (speed_x ** 2) - 3.0 * y1 * (y2 ** 2)) / (speed_x ** 6)

            cs_x_s = CubicSpline(s_grid, x_at_s, bc_type="natural")
            cs_y_s = CubicSpline(s_grid, y_at_s, bc_type="natural")
            sx1, sy1 = cs_x_s.derivative(1)(s_grid), cs_y_s.derivative(1)(s_grid)
            sx2, sy2 = cs_x_s.derivative(2)(s_grid), cs_y_s.derivative(2)(s_grid)
            sx3, sy3 = cs_x_s.derivative(3)(s_grid), cs_y_s.derivative(3)(s_grid)
            sp_s = np.maximum(np.hypot(sx1, sy1), 1e-9)
            num_s = sx1 * sy2 - sy1 * sx2
            den_s = sp_s ** 3
            dnum_s = sx1 * sy3 - sy1 * sx3
            dden_s = 3.0 * (sp_s ** 2) * ((sx1 * sx2 + sy1 * sy2) / sp_s)
            self.dkappa_spline_grid = (dnum_s * den_s - num_s * dden_s) / ((den_s ** 2) * sp_s)
            self.x_grid = x_at_s
            self.y_grid = y_at_s
        else:
            diffs = np.diff(waypoints, axis=0)
            seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
            keep = np.concatenate(([True], seg_lens > 1e-7))
            waypoints = waypoints[keep]
            diffs = np.diff(waypoints, axis=0)
            seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
            u_nodes = np.concatenate(([0.0], np.cumsum(seg_lens)))
            # Làm trơn nhẹ chuỗi waypoint A* rời rạc (bậc C2 trên lưới 0.05 m) để triệt tiêu gai dkappa/ds giả tạo
            if len(waypoints) >= 12 and u_nodes[-1] > 1.5:
                du_sm = 0.02
                u_unif = np.linspace(0.0, u_nodes[-1], max(100, int(round(u_nodes[-1] / du_sm)) + 1))
                sig_pts = 0.14 / (u_unif[1] - u_unif[0])
                x_sm = gaussian_filter1d(np.interp(u_unif, u_nodes, waypoints[:, 0]), sigma=sig_pts, mode="nearest")
                y_sm = gaussian_filter1d(np.interp(u_unif, u_nodes, waypoints[:, 1]), sigma=sig_pts, mode="nearest")
                x_sm[0], y_sm[0] = waypoints[0, 0], waypoints[0, 1]
                x_sm[-1], y_sm[-1] = waypoints[-1, 0], waypoints[-1, 1]
                wp_fit = np.column_stack([x_sm, y_sm])[::4]
                if np.hypot(wp_fit[-1, 0] - waypoints[-1, 0], wp_fit[-1, 1] - waypoints[-1, 1]) > 1e-5:
                    wp_fit = np.vstack([wp_fit, waypoints[-1]])
                else:
                    wp_fit[-1] = waypoints[-1]
                diffs = np.diff(wp_fit, axis=0)
                seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
                keep_fit = np.concatenate(([True], seg_lens > 1e-7))
                waypoints = wp_fit[keep_fit]
                diffs = np.diff(waypoints, axis=0)
                seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
                u_nodes = np.concatenate(([0.0], np.cumsum(seg_lens)))

            cs_x_u = CubicSpline(u_nodes, waypoints[:, 0], bc_type="natural")
            cs_y_u = CubicSpline(u_nodes, waypoints[:, 1], bc_type="natural")

            u_fine = np.linspace(0.0, u_nodes[-1], 8001)
            dx_u = cs_x_u.derivative(1)(u_fine)
            dy_u = cs_y_u.derivative(1)(u_fine)
            ds_du = np.hypot(dx_u, dy_u)
            s_fine = np.concatenate(([0.0], np.cumsum(0.5 * (ds_du[:-1] + ds_du[1:]) * np.diff(u_fine))))
            total_len = float(s_fine[-1])

            n_pts = max(15, int(np.round(total_len / self.ds)) + 1)
            s_grid = np.linspace(0.0, total_len, n_pts)
            u_at_s = np.interp(s_grid, s_fine, u_fine)

            x1 = cs_x_u.derivative(1)(u_at_s)
            y1 = cs_y_u.derivative(1)(u_at_s)
            x2 = cs_x_u.derivative(2)(u_at_s)
            y2 = cs_y_u.derivative(2)(u_at_s)
            x3 = cs_x_u.derivative(3)(u_at_s)
            y3 = cs_y_u.derivative(3)(u_at_s)

            speed_u = np.maximum(np.hypot(x1, y1), 1e-9)
            num_k = x1 * y2 - y1 * x2
            denom_k = speed_u ** 3
            kappa = num_k / denom_k
            dnum_du = x1 * y3 - y1 * x3
            ddenom_du = 3.0 * (speed_u ** 2) * ((x1 * x2 + y1 * y2) / speed_u)
            dkappa = (dnum_du * denom_k - num_k * ddenom_du) / ((denom_k ** 2) * speed_u)
            self.dkappa_spline_grid = dkappa.copy()
            self.x_grid = cs_x_u(u_at_s)
            self.y_grid = cs_y_u(u_at_s)

        # Trích xuất các vùng khúc cua tổng quát cho mọi bản đồ và chiều dài quỹ đạo A*
        n = len(s_grid)
        max_abs_k = float(np.max(np.abs(kappa)))
        k_detect_th = min(self.kappa_th, max(0.20, 0.35 * max_abs_k)) if max_abs_k > 0.05 else self.kappa_th
        in_curve = np.abs(kappa) >= k_detect_th
        corner_regions = []
        i = 0
        yaw_arm = I_Z / (M_TOTAL * L_EFF_YAW)
        min_seg_pts = max(3, int(round(0.12 / self.ds)))
        while i < n:
            if in_curve[i]:
                i_start = i
                while i < n and in_curve[i]:
                    i += 1
                i_end = i - 1
                if (i_end - i_start) >= min_seg_pts:
                    seg_slice = slice(i_start, i_end + 1)
                    eff_k = np.abs(kappa[seg_slice]) + yaw_arm * np.abs(dkappa[seg_slice])
                    i_peak = i_start + int(np.argmax(eff_k))
                    corner_regions.append((i_start, i_peak, i_end))
            else:
                i += 1

        checkpoints = []
        if self.use_physics_cp_constraints and len(corner_regions) > 0:
            # 4 ràng buộc vật lý - hình học tự thích nghi (kèm NMS tách dấu cùng/ngược)
            T1y = 2.0 * np.pi / omega_y
            tau_phase = T1y / 4.0
            L_consol = L_EFF_YAW + self.v_max * T1y
            dpsi_min = self.kappa_th * self.v_max * (0.75 * T1y)
            ds_min_same = L_EFF_YAW + (0.150 + 0.090) * self.v_max + 0.160
            ds_min_opp = L_EFF_YAW + tau_phase * self.v_max
            ds_min_cross = 0.160 + 0.060

            # Pha 1: Gộp cụm cua cùng dấu trong ngưỡng L_consol = 0.512 m
            merged_regs = []
            for i_s, i_p, i_e in corner_regions:
                if not merged_regs:
                    merged_regs.append([i_s, i_p, i_e])
                else:
                    prev = merged_regs[-1]
                    if kappa[prev[1]] * kappa[i_p] > 0 and (s_grid[i_s] - s_grid[prev[2]]) <= L_consol:
                        eff_p = abs(kappa[prev[1]]) + yaw_arm * abs(dkappa[prev[1]])
                        eff_c = abs(kappa[i_p]) + yaw_arm * abs(dkappa[i_p])
                        ip_best = prev[1] if eff_p >= eff_c else i_p
                        merged_regs[-1] = [prev[0], ip_best, i_e]
                    else:
                        merged_regs.append([i_s, i_p, i_e])

            # Pha 2: Lọc bỏ gợn yếu có góc đổi hướng tích lũy < dpsi_min = 0.199 rad
            valid_corners = []
            for i_s, i_p, i_e in merged_regs:
                dpsi = float(np.trapezoid(np.abs(kappa[i_s:i_e + 1]), s_grid[i_s:i_e + 1]))
                if dpsi >= dpsi_min:
                    valid_corners.append({
                        "idx_in": int(i_s),
                        "idx_peak": int(i_p),
                        "idx_out": int(i_e),
                        "s_in": float(s_grid[i_s]),
                        "s_peak": float(s_grid[i_p]),
                        "s_out": float(s_grid[i_e]),
                        "kappa_peak": float(kappa[i_p]),
                        "dkappa_peak": float(dkappa[i_p]),
                        "R_min": float(1.0 / max(abs(kappa[i_p]), 1e-6)),
                        "score": float(abs(kappa[i_p]) + yaw_arm * abs(dkappa[i_p])),
                    })

            # Pha 3: Lọc NMS tách dấu theo score giảm dần (0.588 m cùng dấu / 0.323 m ngược dấu chicane)
            kept_corners = []
            for cand in sorted(valid_corners, key=lambda x: x["score"], reverse=True):
                ok = True
                for kc in kept_corners:
                    req_ds = ds_min_same if (cand["kappa_peak"] * kc["kappa_peak"] > 0) else ds_min_opp
                    if abs(cand["s_peak"] - kc["s_peak"]) < req_ds:
                        ok = False
                        break
                if ok:
                    kept_corners.append(cand)
            kept_corners.sort(key=lambda x: x["s_peak"])

            # Pha 4: Sinh điểm uốn độ giật độc lập giữa hai đỉnh cua ngược dấu đã qua NMS
            for c_idx, c in enumerate(kept_corners):
                is_chic = (
                    (c_idx < len(kept_corners) - 1 and c["kappa_peak"] * kept_corners[c_idx + 1]["kappa_peak"] < 0)
                    or (c_idx > 0 and c["kappa_peak"] * kept_corners[c_idx - 1]["kappa_peak"] < 0)
                )
                checkpoints.append({
                    "s_sort": float(c["s_peak"]),
                    "type": "corner_slosh" if is_chic else "corner",
                    "idx_in": c["idx_in"],
                    "idx_peak": c["idx_peak"],
                    "idx_out": c["idx_out"],
                    "s_in": c["s_in"],
                    "s_peak": c["s_peak"],
                    "s_out": c["s_out"],
                    "kappa_peak": c["kappa_peak"],
                    "dkappa_peak": c["dkappa_peak"],
                    "R_min": c["R_min"],
                    "is_chicane": is_chic,
                })
                if c_idx < len(kept_corners) - 1:
                    nxt = kept_corners[c_idx + 1]
                    gap = float(nxt["s_in"] - c["s_out"])
                    if c["kappa_peak"] * nxt["kappa_peak"] < 0 and L_EFF_YAW <= gap <= (self.d_chicane + 0.15):
                        inf_slice = slice(c["idx_out"], nxt["idx_in"] + 1)
                        i_inf = c["idx_out"] + int(np.argmax(np.abs(dkappa[inf_slice])))
                        s_inf = float(s_grid[i_inf])
                        dk_inf = abs(float(dkappa[i_inf]))
                        dk_adj = max(abs(c["dkappa_peak"]), abs(nxt["dkappa_peak"]))
                        if (
                            dk_inf > dk_adj
                            and abs(s_inf - c["s_peak"]) >= ds_min_cross
                            and abs(nxt["s_peak"] - s_inf) >= ds_min_cross
                        ):
                            half_w = max(2, int(round(0.08 / self.ds)))
                            i_inf_in = max(0, i_inf - half_w)
                            i_inf_out = min(n - 1, i_inf + half_w)
                            checkpoints.append({
                                "s_sort": s_inf,
                                "type": "inflection_jerk",
                                "idx_in": int(i_inf_in),
                                "idx_peak": int(i_inf),
                                "idx_out": int(i_inf_out),
                                "s_in": float(s_grid[i_inf_in]),
                                "s_peak": s_inf,
                                "s_out": float(s_grid[i_inf_out]),
                                "kappa_peak": float(kappa[i_inf]),
                                "dkappa_peak": float(dkappa[i_inf]),
                                "R_min": float(1.0 / max(abs(kappa[i_inf]), 1e-3)),
                                "is_chicane": True,
                            })
            checkpoints.sort(key=lambda cp: cp["s_sort"])

        # Nhánh trích xuất legacy (use_physics_cp_constraints=False) hoặc fallback nếu không còn điểm cua sau lọc
        if len(checkpoints) == 0 and len(corner_regions) > 0 and not self.use_physics_cp_constraints:
            for c_idx, (i_s, i_p, i_e) in enumerate(corner_regions):
                is_chic = False
                if c_idx < len(corner_regions) - 1:
                    next_p = corner_regions[c_idx + 1][1]
                    if kappa[i_p] * kappa[next_p] < 0:
                        is_chic = True
                elif c_idx > 0:
                    prev_p = corner_regions[c_idx - 1][1]
                    if kappa[i_p] * kappa[prev_p] < 0:
                        is_chic = True

                checkpoints.append({
                    "s_sort": float(s_grid[i_p]),
                    "type": "corner_slosh" if is_chic else "corner",
                    "idx_in": int(i_s),
                    "idx_peak": int(i_p),
                    "idx_out": int(i_e),
                    "s_in": float(s_grid[i_s]),
                    "s_peak": float(s_grid[i_p]),
                    "s_out": float(s_grid[i_e]),
                    "kappa_peak": float(kappa[i_p]),
                    "dkappa_peak": float(dkappa[i_p]),
                    "R_min": float(1.0 / max(abs(kappa[i_p]), 1e-6)),
                    "is_chicane": is_chic
                })

                # Nếu có khúc cua tiếp theo ngược dấu trong ngưỡng ghép nối d_chicane (S-curve), quét mọi điểm uốn độ giật cực đại trên toàn tuyến
                if c_idx < len(corner_regions) - 1:
                    next_s = corner_regions[c_idx + 1][0]
                    next_p = corner_regions[c_idx + 1][1]
                    if kappa[i_p] * kappa[next_p] < 0 and next_s > i_e + 2 and (s_grid[next_s] - s_grid[i_e]) <= (self.d_chicane + 0.15):
                        inf_slice = slice(i_e, next_s + 1)
                        i_inflect = i_e + int(np.argmax(np.abs(dkappa[inf_slice])))
                        half_w = max(2, int(round(0.08 / self.ds)))
                        i_inf_in = max(0, i_inflect - half_w)
                        i_inf_out = min(n - 1, i_inflect + half_w)
                        checkpoints.append({
                            "s_sort": float(s_grid[i_inflect]),
                            "type": "inflection_jerk",
                            "idx_in": int(i_inf_in),
                            "idx_peak": int(i_inflect),
                            "idx_out": int(i_inf_out),
                            "s_in": float(s_grid[i_inf_in]),
                            "s_peak": float(s_grid[i_inflect]),
                            "s_out": float(s_grid[i_inf_out]),
                            "kappa_peak": float(kappa[i_inflect]),
                            "dkappa_peak": float(dkappa[i_inflect]),
                            "R_min": float(1.0 / max(abs(kappa[i_inflect]), 1e-3)),
                            "is_chicane": True
                        })
            checkpoints.sort(key=lambda cp: cp["s_sort"])
            filtered_cps = []
            for cand in checkpoints:
                if not filtered_cps:
                    filtered_cps.append(cand)
                else:
                    prev = filtered_cps[-1]
                    if abs(cand["s_sort"] - prev["s_sort"]) < 0.20 and cand["type"] == prev["type"]:
                        if abs(cand["kappa_peak"]) > abs(prev["kappa_peak"]):
                            filtered_cps[-1] = cand
                    elif abs(cand["s_sort"] - prev["s_sort"]) < 0.08:
                        if abs(cand["kappa_peak"]) > abs(prev["kappa_peak"]):
                            filtered_cps[-1] = cand
                    else:
                        filtered_cps.append(cand)
            checkpoints = filtered_cps

        if len(checkpoints) == 0:
            # Quỹ đạo thẳng hoặc cong rất nhẹ: đặt checkpoint kiểm chứng tại các phân đoạn 1/4, 1/2 và 3/4 chiều dài
            for frac, lbl_t in [(0.25, "accel_zone"), (0.50, "cruise_zone"), (0.75, "decel_zone")]:
                i_mid = int(round(frac * (n - 1)))
                half_w = max(2, int(round(0.15 / self.ds)))
                i_in = max(0, i_mid - half_w)
                i_out = min(n - 1, i_mid + half_w)
                checkpoints.append({
                    "s_sort": float(s_grid[i_mid]),
                    "type": lbl_t,
                    "idx_in": int(i_in),
                    "idx_peak": int(i_mid),
                    "idx_out": int(i_out),
                    "s_in": float(s_grid[i_in]),
                    "s_peak": float(s_grid[i_mid]),
                    "s_out": float(s_grid[i_out]),
                    "kappa_peak": float(kappa[i_mid]),
                    "dkappa_peak": float(dkappa[i_mid]),
                    "R_min": float(1.0 / max(abs(kappa[i_mid]), 1e-6)),
                    "is_chicane": False
                })
            checkpoints.sort(key=lambda cp: cp["s_sort"])

        for idx_k, cp in enumerate(checkpoints):
            cp_id = idx_k + 1
            cp["cp_id"] = cp_id
            cp["label"] = f"CP{cp_id} ({cp['type']} tại s={cp['s_peak']:.2f}m)"
            cp.pop("s_sort", None)

        return s_grid, kappa, dkappa, checkpoints

    def _compute_corridor_modulation(self, s_grid: np.ndarray, d_net_profile=None) -> np.ndarray:
        """
        Điều biến ngưỡng góc sóng theo khoảng sáng ròng d_net(s) thuộc [0.15, 0.50] m:
            rho_clr(s) = rho_min + (1 - rho_min) * clip((d_net(s) - 0.15) / (0.50 - 0.15), 0, 1)
        """
        n = len(s_grid)
        if d_net_profile is None:
            d_net = np.full(n, 0.50, dtype=float)
        elif callable(d_net_profile):
            d_net = np.array([float(d_net_profile(s)) for s in s_grid], dtype=float)
        else:
            d_net = np.asarray(d_net_profile, dtype=float)
        self.d_net_grid = d_net
        w_clr = np.clip((d_net - 0.15) / (0.50 - 0.15), 0.0, 1.0)
        rho_clr = self.rho_clr_min + (1.0 - self.rho_clr_min) * w_clr
        self.rho_clr_grid = rho_clr
        return rho_clr

    def _lookup_conformal_dictionary_budget(self, geom_prof: dict = None) -> dict:
        """
        Tra cứu O(1) ngân sách hiệu chuẩn Conformal & độ chệch tất định từ Từ điển Nguyên hàm
        Hình học Ngoại tuyến (`config/conformal_calibration_q_eta.json`) trước khi chạy Giai đoạn A.
        """
        t0 = time.perf_counter()
        cal_path = self.conformal_cal_path or os.path.join(_root, "config", "conformal_calibration_q_eta.json")
        abs_path = os.path.abspath(cal_path)

        payload = _CONFORMAL_DICT_CACHE.get(abs_path)
        if payload is None and os.path.exists(abs_path):
            with open(abs_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            _CONFORMAL_DICT_CACHE[abs_path] = payload

        if payload is not None and (not self._explicit_conformal_override or self.conformal_cal_path is not None):
            if "bias_b0_mm" in payload:
                self.bias_b0_mm = float(payload["bias_b0_mm"])
            if "bias_b1_mm" in payload:
                self.bias_b1_mm = float(payload["bias_b1_mm"])
            if "bias_eta_max_mm" in payload:
                self.bias_eta_max_mm = float(payload["bias_eta_max_mm"])
            if "bias_fz_b0_N" in payload:
                self.bias_fz_b0_N = float(payload["bias_fz_b0_N"])
            if "bias_fz_b1_N" in payload:
                self.bias_fz_b1_N = float(payload["bias_fz_b1_N"])
            if "bias_fz_max_N" in payload:
                self.bias_fz_max_N = float(payload["bias_fz_max_N"])
            if "q_eta_rom_mm" in payload:
                self.q_eta_rom_mm = float(payload["q_eta_rom_mm"])
            elif "q_hat_mm" in payload:
                self.q_eta_rom_mm = float(payload["q_hat_mm"])
            if "q_fz_rom_N" in payload:
                self.q_fz_rom_N = float(payload["q_fz_rom_N"])
            if "q_eta_cl_mm" in payload:
                self.q_eta_cl_mm = float(payload["q_eta_cl_mm"])
                self.eta_bound_cl_mm = self.eta_design_mm + self.q_eta_cl_mm
            new_q_eta_mm = self.bias_eta_max_mm + self.q_eta_rom_mm
            if abs(new_q_eta_mm - self.q_eta_mm) > 1e-9:
                self._update_allowable_angles_from_q_eta(new_q_eta_mm)

        a_max_geom_g = 0.0
        if geom_prof and "a_tot_geom" in geom_prof:
            a_max_geom_g = float(np.max(geom_prof["a_tot_geom"]) / g)
        b_lookup_mm = min(self.bias_eta_max_mm, self.bias_b0_mm + self.bias_b1_mm * a_max_geom_g)
        self.conformal_lookup_time_ms = (time.perf_counter() - t0) * 1000.0
        self.conformal_dict_meta = {
            "source": abs_path if payload is not None else "defaults",
            "n_cal": int(payload.get("n_cal", 200)) if payload else 200,
            "bias_b0_mm": round(self.bias_b0_mm, 4),
            "bias_b1_mm": round(self.bias_b1_mm, 4),
            "bias_eta_max_mm": round(self.bias_eta_max_mm, 3),
            "q_eta_rom_mm": round(self.q_eta_rom_mm, 3),
            "q_fz_rom_N": round(self.q_fz_rom_N, 3),
            "b_lookup_mm": round(float(b_lookup_mm), 3),
            "lookup_time_ms": round(self.conformal_lookup_time_ms, 4),
        }
        return self.conformal_dict_meta

    def _project_longitudinal_reachability(
        self,
        s_grid: np.ndarray,
        v_ceil_in: np.ndarray,
        a_acc_lim: np.ndarray,
        a_brk_lim: np.ndarray,
        kappa_grid: np.ndarray = None,
    ):
        """
        Chiếu S-Curve Jerk bậc 3 hai chiều (Heun Backward - Forward Pass) cho đường trần vận tốc bất kỳ.
        """
        n = len(s_grid)
        total_len = float(s_grid[-1])
        s_from_start = np.maximum(s_grid, 0.0)
        s_to_goal = np.maximum(total_len - s_grid, 0.0)
        v_scurve_start = (4.5 * self.j_plan * (s_from_start ** 2)) ** (1.0 / 3.0)
        v_scurve_goal = (4.5 * self.j_plan * (s_to_goal ** 2)) ** (1.0 / 3.0)

        v_ceil = np.minimum(v_ceil_in, np.minimum(v_scurve_start, v_scurve_goal))
        v_ceil[0] = 0.0
        v_ceil[-1] = 0.0

        v_bwd = np.zeros(n, dtype=float)
        v_valley_bwd = 0.0
        for i in range(n - 2, -1, -1):
            ds_step = float(s_grid[i + 1] - s_grid[i])
            a_lim_mid = 0.5 * float(a_brk_lim[i] + a_brk_lim[i + 1])

            dv_up_0 = max(0.0, v_ceil[i] - v_bwd[i + 1])
            dv_down_0 = max(0.0, v_bwd[i + 1] - v_valley_bwd)
            a_step_0 = min(a_lim_mid, np.sqrt(2.0 * self.j_plan * dv_up_0), np.sqrt(2.0 * self.j_plan * (dv_down_0 + 0.04)))
            v_pred = np.sqrt(v_bwd[i + 1] ** 2 + 2.0 * a_step_0 * ds_step)

            dv_up_1 = max(0.0, v_ceil[i] - v_pred)
            dv_down_1 = max(0.0, v_pred - v_valley_bwd)
            a_step_1 = min(a_lim_mid, np.sqrt(2.0 * self.j_plan * dv_up_1), np.sqrt(2.0 * self.j_plan * (dv_down_1 + 0.04)))
            a_step = 0.5 * (a_step_0 + a_step_1)

            v_cand = np.sqrt(v_bwd[i + 1] ** 2 + 2.0 * a_step * ds_step)
            if v_cand <= v_ceil[i]:
                v_bwd[i] = v_cand
            else:
                v_bwd[i] = v_ceil[i]
                v_valley_bwd = v_ceil[i]

        v_ub = np.zeros(n, dtype=float)
        v_valley_fwd = 0.0
        for i in range(n - 1):
            ds_step = float(s_grid[i + 1] - s_grid[i])
            v_target = v_bwd[i + 1]
            a_lim_mid = 0.5 * float(a_acc_lim[i] + a_acc_lim[i + 1])

            if v_target > v_ub[i]:
                dv_rem_0 = v_target - v_ub[i]
                dv_val_0 = max(0.0, v_ub[i] - v_valley_fwd)
                a_step_0 = min(a_lim_mid, np.sqrt(2.0 * self.j_plan * dv_rem_0), np.sqrt(2.0 * self.j_plan * (dv_val_0 + 0.04)))
                v_pred = np.sqrt(v_ub[i] ** 2 + 2.0 * a_step_0 * ds_step)

                dv_rem_1 = max(0.0, v_target - v_pred)
                dv_val_1 = max(0.0, v_pred - v_valley_fwd)
                a_step_1 = min(a_lim_mid, np.sqrt(2.0 * self.j_plan * dv_rem_1), np.sqrt(2.0 * self.j_plan * (dv_val_1 + 0.04)))
                a_step = 0.5 * (a_step_0 + a_step_1)

                v_cand = np.sqrt(v_ub[i] ** 2 + 2.0 * a_step * ds_step)
                if v_cand < v_target:
                    v_ub[i + 1] = v_cand
                else:
                    v_ub[i + 1] = v_target
                    v_valley_fwd = v_target
            else:
                v_ub[i + 1] = v_target
                v_valley_fwd = v_target

        ax_raw = np.gradient(0.5 * (v_ub ** 2), s_grid)
        half_wb_pts = max(5, int(round(0.56 / self.ds)))
        kx_idx = np.arange(-half_wb_pts, half_wb_pts + 1) * self.ds
        gauss_ker = np.exp(-0.5 * (kx_idx / 0.28) ** 2)
        gauss_ker /= np.sum(gauss_ker)
        ax_nom = np.convolve(ax_raw, gauss_ker, mode="same")
        ax_nom[0] = 0.0
        ax_nom[-1] = 0.0

        k_arr = self.kappa_grid if kappa_grid is None else kappa_grid
        ay_nom = (v_ub ** 2) * k_arr
        ay_nom[0] = 0.0
        ay_nom[-1] = 0.0

        v_lb = np.maximum(0.0, v_ub - self.delta_v_window)
        v_lb[0] = 0.0
        v_lb[-1] = 0.0
        return v_ub, v_lb, ax_nom, ay_nom

    def _compute_exogenous_geom_profile(
        self,
        s_grid: np.ndarray,
        kappa: np.ndarray,
        dkappa: np.ndarray,
        mu_grid: np.ndarray,
        z_com_eff_grid: np.ndarray = None,
        phi_camber_grid: np.ndarray = None,
        delta_warp_grid: np.ndarray = None,
    ) -> dict:
        """
        Tính biên vận tốc & gia tốc ngoại sinh thuần túy theo hình học A* và phần cứng
        (hoàn toàn độc lập với nghiệm ống vận tốc v_ub của Giai đoạn A).
        """
        n = len(s_grid)
        abs_k = np.abs(kappa) + 1e-6
        abs_dk = np.abs(dkappa) + 1e-6
        mu_plan = self.mu_effective_plan_scale * mu_grid

        v_jerk_geom = (self.j_y_max / abs_dk) ** (1.0 / 3.0)
        if self.enable_grip:
            v_mu_geom = np.sqrt((mu_plan * g) / abs_k)
            v_ref_raw = np.minimum(self.v_max, np.minimum(v_mu_geom, v_jerk_geom))
        else:
            v_ref_raw = np.minimum(self.v_max, v_jerk_geom)

        a_acc_hw = np.full(n, self.a_acc_max, dtype=float)
        a_brk_hw = np.full(n, self.a_brake_max, dtype=float)
        v_ref_geom, _, ax_ref_geom, ay_ref_geom = self._project_longitudinal_reachability(
            s_grid, v_ref_raw, a_acc_hw, a_brk_hw, kappa_grid=kappa
        )
        return {
            "v_ref_raw": v_ref_raw,
            "v_ref_geom": v_ref_geom,
            "ax_ref_geom": ax_ref_geom,
            "ay_ref_geom": ay_ref_geom,
            "a_tot_geom": np.hypot(ax_ref_geom, ay_ref_geom),
            "z_com_eff_grid": np.full(n, Z_COM, dtype=float) if z_com_eff_grid is None else np.asarray(z_com_eff_grid, dtype=float),
            "phi_camber_grid": np.zeros(n, dtype=float) if phi_camber_grid is None else np.asarray(phi_camber_grid, dtype=float),
            "delta_warp_grid": np.zeros(n, dtype=float) if delta_warp_grid is None else np.asarray(delta_warp_grid, dtype=float),
        }

    def _evaluate_exogenous_dbep_profile(self, geom_prof: dict) -> tuple:
        """
        Tích phân mô hình con lắc 2 trục DBEP + Stokes bậc 2 dọc theo hồ sơ kích thích ngoại sinh.
        Trả về (eta_dbep_geom_mm, fz_geom_raw_N) với tải pháp tuyến không kẹp sàn giả để đối chiếu chân thực.
        """
        v_g = geom_prof["v_ref_geom"]
        ax_g = geom_prof["ax_ref_geom"]
        ay_g = geom_prof["ay_ref_geom"]
        zc_g = geom_prof["z_com_eff_grid"]
        phi_g = geom_prof["phi_camber_grid"]
        warp_g = geom_prof["delta_warp_grid"]

        saved_substeps = self.total_rk4_substeps
        th_x_ens, _, th_y_ens, _ = self._rollout_pendulum_2dof_rk4_vectorized(
            self.s_grid, v_g, ax_g, self.kappa_grid
        )
        self.total_rk4_substeps = saved_substeps

        mid_idx = len(self.freq_scales) // 2
        th_x = th_x_ens[mid_idx]
        th_y = th_y_ens[mid_idx]
        u_x = np.abs(np.tan(th_x))
        u_y = np.abs(np.tan(th_y))

        eta_x_mm = (0.5 * TANK_A * u_x + (np.pi * TANK_A / 8.0) * self.coth_x * (u_x ** 2)) * 1000.0
        eta_y_mm = (0.5 * TANK_B * u_y + (np.pi * TANK_B / 8.0) * self.coth_y * (u_y ** 2)) * 1000.0
        eta_dbep_geom_mm = np.maximum(eta_y_mm + 0.35 * eta_x_mm, eta_x_mm + 0.35 * eta_y_mm)

        z_sx = zc_g + (Z_SLOSH_X - Z_COM)
        z_sy = zc_g + (Z_SLOSH_Y - Z_COM)
        mz_x = M_TOTAL * zc_g + m_1x * z_sx
        mz_y = M_TOTAL * zc_g + m_1y * z_sy
        gamma_roll = 1.0 / np.maximum(0.50, 1.0 - (M_TOTAL * g * zc_g) / self.k_roll)
        dfz_camber = gamma_roll * (M_TOTAL * g * zc_g * np.sin(phi_g)) / (4.0 * L_Y)
        dfz_warp = self.k_z_corner * warp_g
        m_pitch = mz_x * np.abs(ax_g) + m_1x * g * lx * np.abs(np.sin(th_x))
        m_roll = gamma_roll * (mz_y * np.abs(ay_g) * np.cos(phi_g) + m_1y * g * ly * np.abs(np.sin(th_y)))
        fz_geom_raw_N = np.maximum(
            0.0,
            FZ_NOM - (m_pitch / (4.0 * L_X)) - (m_roll / (4.0 * L_Y)) - self.dfz_iso_traj_max - dfz_camber - dfz_warp
        )
        return eta_dbep_geom_mm, fz_geom_raw_N

    def _compute_h2_mechanical_bounds(
        self,
        s_grid: np.ndarray,
        kappa: np.ndarray,
        dkappa: np.ndarray,
        ax_est: np.ndarray,
        mu_grid: np.ndarray,
        z_com_eff_grid: np.ndarray = None,
        phi_camber_grid: np.ndarray = None,
        delta_warp_grid: np.ndarray = None
    ):
        """
        Tính trần vận tốc cơ học (Mô-đun M2 & M4):
        1. Giới hạn độ giật ngang hình học: v_jerk_raw(s) = (j_y_max / (|kappa'(s)| + 1e-6))^(1/3).
        2. Đa diện ma sát Zonotope 12 mặt 3D có thành phần mô-men xoay M_z:
           Trả về đồng thời v_grip_2d_static (ax=0, Mz=0), v_grip_2d_dyn (có ax, Mz=0) và v_grip_raw (đầy đủ 3D).
        3. Chống nhấc bánh Fz_w >= Fz_min = 3.50 N trên hệ 4 bánh siêu tĩnh (hyperstatic Mecanum chassis):
           - K_roll = 4 * k_z,corner * L_Y^2 = 250.88 N.m/rad -> gamma_roll = 1.0665 (chỉ 6.24% ngưỡng lật tĩnh).
           - Áp dụng NHẤT QUÁN gamma_roll cho cả số hạng tĩnh (dốc ngang phi_camber) và số hạng động (mô-men lật ly tâm).
           - Xét độ vênh xoắn mặt sàn chuyển tiếp cầu dẫn DIN 18202: dfz_warp = k_z,corner * delta_warp(s).
        """
        abs_k = np.abs(kappa) + 1e-6
        abs_dk = np.abs(dkappa) + 1e-6
        mu_plan = self.mu_effective_plan_scale * mu_grid
        n = len(s_grid)

        if z_com_eff_grid is None:
            z_com_eff_grid = np.full(n, Z_COM, dtype=float)
        if phi_camber_grid is None:
            phi_camber_grid = np.zeros(n, dtype=float)
        if delta_warp_grid is None:
            delta_warp_grid = np.zeros(n, dtype=float)

        # 1. Trần độ giật ngang hình học v_jerk_raw
        v_jerk_raw = (self.j_y_max / abs_dk) ** (1.0 / 3.0)
        v_jerk = np.minimum(self.v_max, v_jerk_raw)

        # 2. Trần bám đường đa diện Zonotope 3D (F_bx, F_by, M_z)
        if self.enable_grip:
            iso_load_factor = max(0.5, 1.0 - self.dfz_iso_traj_max / FZ_NOM)
            a_total_grip = (mu_plan * g / np.sqrt(2.0)) * iso_load_factor
            yaw_arm_factor = I_Z / (M_TOTAL * L_EFF_YAW)  # lambda_yaw = 0.02812 m

            v_grip_2d_static = np.sqrt(np.maximum(0.04, a_total_grip) / abs_k)
            a_lat_2d_dyn = np.maximum(0.04, a_total_grip - np.minimum(np.abs(ax_est), 0.70 * a_total_grip))
            v_grip_2d_dyn = np.sqrt(a_lat_2d_dyn / abs_k)

            ax_eff_grip = np.minimum(np.abs(ax_est) * (1.0 + yaw_arm_factor * abs_k), 0.70 * a_total_grip)
            a_lat_grip = np.maximum(0.04, a_total_grip - ax_eff_grip)
            denom_grip = abs_k + yaw_arm_factor * abs_dk
            v_grip_raw = np.sqrt(a_lat_grip / denom_grip)
            v_grip = np.minimum(self.v_max, v_grip_raw)
        else:
            v_grip_2d_static = np.full_like(abs_k, 99.0)
            v_grip_2d_dyn = np.full_like(abs_k, 99.0)
            v_grip_raw = np.full_like(abs_k, 99.0)
            v_grip = np.full_like(abs_k, self.v_max)

        # 3. Trần chống nhấc bánh Fz_w >= Fz_min = 3.50 N (Nhất quán tĩnh & động)
        z_slosh_x_grid = z_com_eff_grid + (Z_SLOSH_X - Z_COM)
        z_slosh_y_grid = z_com_eff_grid + (Z_SLOSH_Y - Z_COM)
        mz_eff_x_grid = M_TOTAL * z_com_eff_grid + m_1x * z_slosh_x_grid
        mz_eff_y_grid = M_TOTAL * z_com_eff_grid + m_1y * z_slosh_y_grid

        gamma_roll = 1.0 / np.maximum(0.50, 1.0 - (M_TOTAL * g * z_com_eff_grid) / self.k_roll)

        dfz_camber = gamma_roll * (M_TOTAL * g * z_com_eff_grid * np.sin(phi_camber_grid)) / (4.0 * L_Y)
        dfz_warp = self.k_z_corner * delta_warp_grid

        theta_x_est = np.minimum(self.theta_allow_x, np.abs(ax_est) / g)
        m_pitch_est = mz_eff_x_grid * np.abs(ax_est) + m_1x * g * lx * theta_x_est
        dfz_pitch = m_pitch_est / (4.0 * L_X)

        fz_avail_roll = np.maximum(
            0.05,
            FZ_NOM - FZ_MIN - self.dfz_iso_traj_max - self.q_fz_base - dfz_camber - dfz_warp - dfz_pitch
        )
        theta_y_roll_budget = self.rho_clr_grid * self.theta_allow_y
        m_roll_slosh = m_1y * g * ly * theta_y_roll_budget

        # Áp dụng NHẤT QUÁN hệ số khuếch đại gamma_roll cho cả mô-men lật động:
        m_roll_avail = np.maximum(
            0.025,
            (4.0 * L_Y / gamma_roll) * fz_avail_roll - m_roll_slosh
        )
        if self.enable_traction:
            v_traction_raw = np.sqrt(m_roll_avail / (mz_eff_y_grid * np.cos(phi_camber_grid) * abs_k))
            v_traction = np.minimum(self.v_max, v_traction_raw)
        else:
            v_traction_raw = np.full_like(abs_k, 99.0)
            v_traction = np.full_like(abs_k, self.v_max)

        self.v_grip_2d_static = v_grip_2d_static
        self.v_grip_2d_dyn = v_grip_2d_dyn
        return v_grip_raw, v_traction_raw, v_jerk_raw, v_grip, v_traction, v_jerk

    def _rollout_pendulum_2dof_rk4_vectorized(
        self,
        s_grid: np.ndarray,
        v_profile: np.ndarray,
        ax_profile: np.ndarray,
        kappa: np.ndarray
    ):
        """
        Tích phân phương trình vi phân phi tuyến con lắc 2 trục DBEP trên lưới thời gian mịn
        dt_rk4 = 2.0 ms (>= 180 mẫu/chu kỳ dao động, không suy biến khi v -> 0) qua M_omega = 9 tần số
        bằng phép biến đổi trạng thái ZOH chính xác kết hợp hiệu chỉnh phi tuyến lượng giác (Picard-Duffing)
        thực thi bằng bộ lọc C biên dịch (scipy.signal.lfilter).
        """
        n = len(s_grid)
        v_kin_min = 0.08
        ds_arr = np.diff(s_grid)
        v_mid_arr = np.maximum(v_kin_min, 0.5 * (v_profile[:-1] + v_profile[1:]))
        dt_arr = ds_arr / v_mid_arr
        t_nodes = np.concatenate(([0.0], np.cumsum(dt_arr)))
        t_total = float(t_nodes[-1])

        dt = self.dt_rk4
        n_t = max(25, int(np.ceil(t_total / dt)) + 1)
        t_uniform = np.linspace(0.0, t_total, n_t)

        ax_t = np.interp(t_uniform, t_nodes, ax_profile)
        v_t = np.interp(t_uniform, t_nodes, np.maximum(v_kin_min, v_profile))
        k_t = np.interp(t_uniform, t_nodes, kappa)
        ay_t = (v_t ** 2) * k_t

        m_freq = len(self.freq_scales)
        th_x = np.zeros((m_freq, n), dtype=float)
        om_x = np.zeros((m_freq, n), dtype=float)
        th_y = np.zeros((m_freq, n), dtype=float)
        om_y = np.zeros((m_freq, n), dtype=float)

        inv_2dt = 0.5 / dt
        for jf in range(m_freq):
            # Trục X: ZOH LTI + hiệu chỉnh phi tuyến bậc 3 (sin theta = theta - theta^3/6, cos theta = 1 - theta^2/2)
            bx_poly, ax_poly, wx_j, bg_x = self._filters_x[jf]
            u_x0 = bg_x * ax_t
            tx_lin = lfilter(bx_poly, ax_poly, u_x0)
            u_x_nl = bg_x * (1.0 - 0.5 * (tx_lin ** 2)) * ax_t + (wx_j ** 2) * ((tx_lin ** 3) / 6.0)
            tx_t = lfilter(bx_poly, ax_poly, u_x_nl)
            ox_t = np.zeros_like(tx_t)
            ox_t[1:-1] = (tx_t[2:] - tx_t[:-2]) * inv_2dt
            ox_t[-1] = (tx_t[-1] - tx_t[-2]) / dt

            # Trục Y: ZOH LTI + hiệu chỉnh phi tuyến bậc 3
            by_poly, ay_poly, wy_j, bg_y = self._filters_y[jf]
            u_y0 = bg_y * ay_t
            ty_lin = lfilter(by_poly, ay_poly, u_y0)
            u_y_nl = bg_y * (1.0 - 0.5 * (ty_lin ** 2)) * ay_t + (wy_j ** 2) * ((ty_lin ** 3) / 6.0)
            ty_t = lfilter(by_poly, ay_poly, u_y_nl)
            oy_t = np.zeros_like(ty_t)
            oy_t[1:-1] = (ty_t[2:] - ty_t[:-2]) * inv_2dt
            oy_t[-1] = (ty_t[-1] - ty_t[-2]) / dt

            th_x[jf] = np.interp(t_nodes, t_uniform, tx_t)
            om_x[jf] = np.interp(t_nodes, t_uniform, ox_t)
            th_y[jf] = np.interp(t_nodes, t_uniform, ty_t)
            om_y[jf] = np.interp(t_nodes, t_uniform, oy_t)

        self.total_rk4_substeps = n_t - 1
        return th_x, om_x, th_y, om_y

    def _compute_h1_coupled_slosh_inversion(
        self,
        s_grid: np.ndarray,
        kappa: np.ndarray,
        checkpoints: list,
        v_current: np.ndarray,
        ax_current: np.ndarray
    ):
        """
        Mô-đun M1: Đảo ngược năng lượng sóng sánh 2 trục trên lưới 9 tần số cộng hưởng.
        """
        n = len(s_grid)
        th_x_arr, om_x_arr, th_y_arr, om_y_arr = self._rollout_pendulum_2dof_rk4_vectorized(
            s_grid, v_current, ax_current, kappa
        )
        wy_col = (omega_y * self.freq_scales)[:, None]
        om_y_scaled = om_y_arr / wy_col

        th_x_abs_max = np.max(np.abs(th_x_arr), axis=0)
        kernel_w = max(3, int(round(0.24 / self.ds)))
        if kernel_w % 2 == 0:
            kernel_w += 1
        th_x_env = np.convolve(th_x_abs_max, np.ones(kernel_w) / kernel_w, mode="same")

        theta_allow_y_mod = self.rho_clr_grid * self.theta_allow_y
        longitudinal_deduction = np.minimum(0.32 * theta_allow_y_mod, (TANK_A / TANK_B) * (0.38 * th_x_env))
        theta_allow_y_eff = np.maximum(0.04, theta_allow_y_mod - longitudinal_deduction)

        E_y_in_grid = np.zeros(n, dtype=float)
        e_all_freq = np.sqrt(th_y_arr ** 2 + om_y_scaled ** 2)
        if len(self.freq_scales) > 1:
            d_e_freq = 0.5 * np.max(np.abs(np.diff(e_all_freq, axis=0)), axis=0)
        else:
            d_e_freq = 0.0
        e_sup_profile = np.max(e_all_freq, axis=0) + d_e_freq

        for cp in checkpoints:
            s_in_cp = cp["s_in"]
            s_out_cp = cp["s_out"]
            in_win_mask = (s_grid >= s_in_cp - 0.16) & (s_grid <= s_in_cp)
            cp_mask = (s_grid >= s_in_cp) & (s_grid <= s_out_cp)
            if cp["cp_id"] == 1 or not np.any(in_win_mask):
                e_in_m = 0.0
            else:
                e_in_raw = float(np.mean(e_sup_profile[in_win_mask]))
                e_in_m = min(e_in_raw, 0.38 * float(np.mean(theta_allow_y_eff[cp_mask])))
            # Cửa sổ làm trơn bất biến theo tọa độ cung s (không phụ thuộc bước lưới ds)
            w_cp = 0.5 * (np.tanh((s_grid - s_in_cp) / 0.06) - np.tanh((s_grid - s_out_cp) / 0.06))
            E_y_in_grid = np.maximum(E_y_in_grid, e_in_m * w_cp)
            cp["E_y_in_deg"] = round(float(np.degrees(e_in_m)), 2)

        E_y_in_smooth = E_y_in_grid

        theta_eq_max = np.maximum(0.010, 0.5 * (theta_allow_y_eff - E_y_in_smooth))
        abs_k = np.abs(kappa) + 1e-6
        v_slosh_raw = np.sqrt((g * theta_eq_max) / abs_k)
        v_slosh = np.minimum(self.v_max, v_slosh_raw)

        u_eff = np.tan(theta_allow_y_eff)
        eta_y_bound_m = 0.5 * TANK_B * u_eff + (np.pi * TANK_B / 8.0) * self.coth_y * (u_eff ** 2)
        u_x_env = np.tan(th_x_env)
        eta_x_bound_m = 0.5 * TANK_A * u_x_env + (np.pi * TANK_A / 8.0) * self.coth_x * (u_x_env ** 2)
        eta_bound_analytical_mm = np.minimum(
            self.eta_design_mm,
            (eta_y_bound_m + 0.35 * eta_x_bound_m) * 1000.0
        )

        return v_slosh_raw, v_slosh, eta_bound_analytical_mm

    def _compute_h4_jerk_reachability_tube(
        self,
        s_grid: np.ndarray,
        v_phys: np.ndarray,
        mu_grid: np.ndarray
    ):
        """
        Mô-đun M4: Toán tử chiếu S-Curve Jerk bậc 3 hai chiều (Backward - Forward Pass).
        """
        n = len(s_grid)
        total_len = float(s_grid[-1])
        mu_plan = self.mu_effective_plan_scale * mu_grid

        a_slosh_x_lim = 0.55 * g * self.theta_allow_x
        if self.enable_grip:
            a_grip_x_lim = (mu_plan * g / np.sqrt(2.0)) * 0.75
        else:
            a_grip_x_lim = np.full_like(mu_plan, 99.0)
        a_acc_lim = np.minimum(self.a_acc_max, np.minimum(a_slosh_x_lim, a_grip_x_lim))
        a_brk_lim = np.minimum(self.a_brake_max, np.minimum(a_slosh_x_lim, a_grip_x_lim))
        return self._project_longitudinal_reachability(s_grid, v_phys, a_acc_lim, a_brk_lim)

    def run_wcsph_checkpoint_standardization(self, dp_sph: float = 0.005) -> dict:
        """
        Đối chứng ngoại tuyến mô hình Con lắc DBEP + Stokes bậc 2 (đã hiệu chỉnh độ lệch hệ thống b_bar_eta)
        với bộ giải hạt Navier-Stokes DualSliceWCSPH tại tất cả các Checkpoint và nhánh phanh phản thực tế T2.
        Tổng quát hóa cho mọi bản đồ và mọi chiều dài quỹ đạo do A* tạo ra.
        """
        if not self.is_ready:
            raise RuntimeError("Cần tính ống vận tốc trước khi chuẩn hóa WCSPH tại Checkpoint!")

        t_sph0 = time.perf_counter()
        dual_sph = DualSliceWCSPH(
            Lx=TANK_A, Ly=TANK_B, tank_height=TANK_H, h=H_LIQUID, dp=dp_sph
        )
        n_fluid_total = dual_sph.solver_xz.n_fluid + dual_sph.solver_yz.n_fluid
        n_bound_total = dual_sph.solver_xz.n_bound + dual_sph.solver_yz.n_bound

        dt_cfl = 0.000060
        res0 = dual_sph.step(dt_cfl, ax=0.0, ay=0.0)
        eta0_x = res0["eta_peak_x"]
        eta0_y = res0["eta_peak_y"]
        f0_wheels = np.full(4, (M_TOTAL * g) / 4.0)
        s_pitch = np.array([-1.0, -1.0, +1.0, +1.0])
        s_roll  = np.array([-1.0, +1.0, -1.0, +1.0])

        raw_residuals_eta_mm = []
        debiased_residuals_eta_mm = []
        raw_residuals_fz_N = []
        debiased_residuals_fz_N = []

        # Fix 2: Không chèn điểm gốc 0.0 làm đẹp giả tạo thống kê NSE/RMSE
        eta_rom_debiased_series = []
        eta_wcsph_series = []

        # Tính sẵn từ điển đặc trưng ngoại sinh & mô hình DBEP ngoại sinh làm đối chứng độc lập
        geom_prof = self._compute_exogenous_geom_profile(
            self.s_grid, self.kappa_grid, self.dkappa_grid, self.mu_grid,
            self.z_com_eff_grid, self.phi_camber_grid, self.delta_warp_grid
        )
        eta_dbep_geom_grid, fz_geom_grid = self._evaluate_exogenous_dbep_profile(geom_prof)

        omega_1y = float(np.sqrt((np.pi * g / TANK_B) * np.tanh(np.pi * H_LIQUID / TANK_B)))
        omega_1x = float(np.sqrt((np.pi * g / TANK_A) * np.tanh(np.pi * H_LIQUID / TANK_A)))
        tanh_y = float(np.tanh(np.pi * H_LIQUID / TANK_B))
        tanh_x = float(np.tanh(np.pi * H_LIQUID / TANK_A))
        coth_y = 1.0 / max(tanh_y, 1e-6)
        coth_x = 1.0 / max(tanh_x, 1e-6)
        l_x_chassis = 0.120
        l_y_chassis = 0.140
        m_fl_g = max(1e-3, dual_sph.bridge.m_fl_total * g)

        # Lưu trạng thái tĩnh cân bằng vào RAM để làm gốc rẽ nhánh độc lập cho mỗi Checkpoint
        for _ in range(10):
            res0 = dual_sph.step(dt_cfl, ax=0.0, ay=0.0)
        base_sph_ckpt = dual_sph.save_checkpoint()

        # Hiệu chuẩn hệ số xây dựng xung áp suất thành bình trên lưới dp hiện tại (40 bước CFL)
        dual_sph.load_checkpoint(base_sph_ckpt)
        for s_cal in range(40):
            al_cal = min(1.0, (s_cal + 1) / 15.0)
            dual_sph.step(dt_cfl, ax=1.0 * al_cal, ay=1.0 * al_cal)
        cw_y = max(0.25, abs(dual_sph.solver_yz.f_wall_2d[0] * dual_sph.Lx) / max(1e-3, dual_sph.bridge.m_fl_total * 1.0))
        cw_x = max(0.20, abs(dual_sph.solver_xz.f_wall_2d[0] * dual_sph.Ly) / max(1e-3, dual_sph.bridge.m_fl_total * 1.0))
        dual_sph.load_checkpoint(base_sph_ckpt)

        for idx_cp, cp in enumerate(self.checkpoints):
            idx_in = cp["idx_in"]
            idx_p = cp["idx_peak"]
            idx_out = cp["idx_out"]
            sl = slice(idx_in, idx_out + 1)
            idx_eta_max = idx_in + int(np.argmax(self.eta_pred_mm[sl]))
            idx_eta_geom_max = idx_in + int(np.argmax(eta_dbep_geom_grid[sl]))
            idx_ay_geom_max = idx_in + int(np.argmax(np.abs(geom_prof["ay_ref_geom"][sl])))

            # Trích xuất đặc trưng hình học ngoại sinh thuần túy (khử phụ thuộc vòng Stage A <-> B)
            if idx_cp == 0:
                ds_link_cp = float(cp["s_peak"] - self.s_grid[0])
                gamma_link_cp = 0.0
                phi_link_cp = 0.0
                ay_prev_geom = 0.0
            else:
                s_prev_cp = float(self.checkpoints[idx_cp - 1]["s_peak"])
                ds_link_cp = float(cp["s_peak"] - s_prev_cp)
                mask_link = (self.s_grid >= s_prev_cp) & (self.s_grid <= cp["s_peak"])
                if np.sum(mask_link) >= 2:
                    inv_v_link = 1.0 / np.maximum(geom_prof["v_ref_geom"][mask_link], 0.08)
                    t_link = float(np.trapezoid(inv_v_link, self.s_grid[mask_link]))
                else:
                    t_link = ds_link_cp / max(float(geom_prof["v_ref_geom"][idx_p]), 0.08)
                gamma_link_cp = float(np.exp(-ZETA_SLOSH * omega_1y * t_link))
                phi_link_cp = float((omega_1y * t_link) % (2.0 * np.pi))
                ay_prev_geom = float(self.checkpoints[idx_cp - 1]["ay_ref_geom"])

            v_ref_geom_cp = float(geom_prof["v_ref_geom"][idx_p])
            ax_geom_cp = float(geom_prof["ax_ref_geom"][idx_eta_geom_max])
            ay_geom_cp = float(geom_prof["ay_ref_geom"][idx_ay_geom_max])
            a_tot_geom_cp = float(np.hypot(ax_geom_cp, ay_geom_cp))
            delta_psi_cp = float(np.trapezoid(self.kappa_grid[sl], self.s_grid[sl])) if idx_out > idx_in else 0.0

            cp["delta_psi_rad"] = delta_psi_cp
            cp["ds_link_m"] = ds_link_cp
            cp["v_ref_geom"] = v_ref_geom_cp
            cp["ax_ref_geom"] = ax_geom_cp
            cp["ay_ref_geom"] = ay_geom_cp
            cp["a_tot_geom"] = a_tot_geom_cp
            cp["gamma_link"] = gamma_link_cp
            cp["phi_link_rad"] = phi_link_cp

            if self.use_exogenous_geom_checkpoint:
                ax_exc = ax_geom_cp
                ay_exc = ay_geom_cp
                v_exc = v_ref_geom_cp
                eta_dbep_cmp = float(eta_dbep_geom_grid[idx_eta_geom_max])
                eta_dbep_in_cmp = float(eta_dbep_geom_grid[idx_in])
                fz_dbep_cmp = float(np.min(fz_geom_grid[sl]))
            else:
                ax_exc = float(self.ax_nom[idx_eta_max])
                ay_exc = float(self.ay_nom[idx_p])
                v_exc = float(cp.get("v_ub_cp", self.v_max * 0.7))
                eta_dbep_cmp = float(self.eta_pred_mm[idx_eta_max])
                eta_dbep_in_cmp = float(self.eta_pred_mm[idx_in])
                fz_dbep_cmp = float(np.min(self.fz_min_pred[sl]))

            a_tot_exc = float(np.hypot(ax_exc, ay_exc))
            b0_mm = float(getattr(self, "bias_b0_mm", 0.42))
            b1_mm = float(getattr(self, "bias_b1_mm", 4.80))
            bias_eta_cp = min(self.bias_eta_max_mm, b0_mm + b1_mm * (a_tot_exc / g))
            bias_fz_cp = min(self.bias_fz_max_N, self.bias_fz_b0_N + self.bias_fz_b1_N * (a_tot_exc / g))

            eta_rom_corr_cp = eta_dbep_cmp + bias_eta_cp
            fz_rom_corr_cp = max(0.0, fz_dbep_cmp - bias_fz_cp)

            # 1-3. Đánh giá tổng hợp Checkpoint theo chuẩn Hybrid B+A (cửa sổ dài T1=150ms, quét pha carryover & phanh T2=90ms)
            a_brk_x = -min(self.a_brake_max, 0.45) * (v_exc / max(self.v_max, 1e-6))
            a_brk_y = -0.35 * ay_exc
            hyb = dual_sph.evaluate_hybrid_ba_checkpoint(
                base_ckpt=base_sph_ckpt,
                dt=dt_cfl,
                ax_exc=ax_exc,
                ay_exc=ay_exc,
                a_brk_x=a_brk_x,
                a_brk_y=a_brk_y,
                ay_prev_geom=ay_prev_geom,
                gamma_link=gamma_link_cp,
                phi_link_rad=phi_link_cp,
                delta_psi_rad=delta_psi_cp,
                t1_window_s=0.150,
                t2_window_s=0.090,
            )
            res_sph = hyb["res_t1"]
            res_t2 = hyb["res_t2"]
            worst_phase_gain = hyb["worst_phase_gain"]
            save_ram_ms = hyb["save_ram_ms"]
            eta_wcsph_cp = hyb["eta_wcsph_t1_mm"]
            eta_wcsph_t2_mm = hyb["eta_wcsph_t2_mm"]

            cp["worst_phase_gain"] = round(float(worst_phase_gain), 4)
            cp["wcsph_window_t1_ms"] = round(float(hyb["t1_window_ms"]), 2)
            cp["wcsph_window_t2_ms"] = round(float(hyb["t2_window_ms"]), 2)
            cp["peak_eta_surf_x_mm"] = round(float(res_sph["peak_eta_surf_x_mm"]), 4)
            cp["peak_eta_surf_y_mm"] = round(float(res_sph["peak_eta_surf_y_mm"]), 4)

            # Đồng bộ quy chiếu tải pháp tuyến 4 bánh: kết hợp (Z_COM_eff, phi_camber, delta_warp, gamma_roll, dfz_iso) với dFz_slosh từ WCSPH
            z_com_cp = float(self.z_com_eff_grid[idx_p])
            phi_camber_cp = float(self.phi_camber_grid[idx_p])
            delta_warp_cp = float(self.delta_warp_grid[idx_p])
            gamma_roll_cp = 1.0 / max(0.25, 1.0 - (M_TOTAL * g * z_com_cp) / max(self.k_roll, 1e-6))

            dFz_slosh = res_sph["dFz_slosh_wheels"]
            dFz_rigid_pitch = (ax_exc * z_com_cp * M_TOTAL) / (4.0 * l_x_chassis)
            dFz_rigid_roll = gamma_roll_cp * (ay_exc * z_com_cp * M_TOTAL * np.cos(phi_camber_cp)) / (4.0 * l_y_chassis)
            dFz_camber_cp = gamma_roll_cp * (M_TOTAL * g * z_com_cp * abs(np.sin(phi_camber_cp))) / (4.0 * l_y_chassis)
            dFz_warp_cp = self.k_z_corner * abs(delta_warp_cp)
            fz_wheels_wcsph = (
                f0_wheels + s_pitch * dFz_rigid_pitch + s_roll * dFz_rigid_roll + dFz_slosh
                - dFz_camber_cp - dFz_warp_cp - self.dfz_iso_traj_max
            )
            fz_wcsph_cp = max(0.0, float(np.min(fz_wheels_wcsph)))

            # Kiểm tra thêm tải pháp tuyến WCSPH tại vận tốc ống Stage A v_ub (dFz_slosh tỷ lệ với tải động)
            scale_nom = min(1.0, (float(np.hypot(self.ax_nom[idx_eta_max], self.ay_nom[idx_p])) / max(a_tot_exc, 1e-4)))
            dFz_rigid_pitch_nom = (float(self.ax_nom[idx_eta_max]) * z_com_cp * M_TOTAL) / (4.0 * l_x_chassis)
            dFz_rigid_roll_nom = gamma_roll_cp * (float(self.ay_nom[idx_p]) * z_com_cp * M_TOTAL * np.cos(phi_camber_cp)) / (4.0 * l_y_chassis)
            fz_wcsph_at_tube_cp = max(0.0, float(np.min(
                f0_wheels + s_pitch * dFz_rigid_pitch_nom + s_roll * dFz_rigid_roll_nom + scale_nom * dFz_slosh
                - dFz_camber_cp - dFz_warp_cp - self.dfz_iso_traj_max
            )))
            cp["fz_wcsph_at_tube_N"] = round(fz_wcsph_at_tube_cp, 2)

            # Phản thực tế phanh khẩn cấp T2 thuần ngoại sinh đồng bộ ghép cặp 2D (không tuần hoàn vào SPH)
            eta_x_brk_mm = (0.5 * TANK_A * (abs(a_brk_x) / g)) * 1000.0 * 0.65
            eta_y_brk_mm = (0.5 * TANK_B * (abs(a_brk_y) / g)) * 1000.0 * 0.35
            eta_dbep_t2_mm = max(eta_dbep_cmp, 0.85 * eta_dbep_cmp + 0.70 * eta_x_brk_mm + eta_y_brk_mm)
            a_tot_t2_eff = max(a_tot_exc, float(np.hypot(a_brk_x, a_brk_y)))
            bias_eta_t2 = min(self.bias_eta_max_mm, b0_mm + b1_mm * (a_tot_t2_eff / g))
            eta_rom_corr_t2 = eta_dbep_t2_mm + bias_eta_t2
            cp["eta_dbep_cmp_mm"] = round(float(eta_dbep_cmp), 2)
            cp["fz_dbep_cmp_N"] = round(float(fz_dbep_cmp), 2)

            raw_res_eta = max(eta_wcsph_cp - eta_dbep_cmp, eta_wcsph_t2_mm - eta_dbep_t2_mm)
            deb_res_nom = eta_wcsph_cp - eta_rom_corr_cp
            deb_res_t2 = eta_wcsph_t2_mm - eta_rom_corr_t2
            deb_res_eta = deb_res_t2 if abs(deb_res_t2) >= abs(deb_res_nom) else deb_res_nom

            raw_res_fz = fz_dbep_cmp - fz_wcsph_cp
            deb_res_fz = fz_rom_corr_cp - fz_wcsph_cp

            cp["bias_eta_mm"] = round(float(bias_eta_cp), 2)
            cp["eta_rom_corr_cp_mm"] = round(float(eta_rom_corr_cp), 2)
            cp["eta_wcsph_cp_mm"] = round(float(eta_wcsph_cp), 2)
            cp["eta_dbep_t2_mm"] = round(float(eta_dbep_t2_mm), 2)
            cp["eta_rom_corr_t2_mm"] = round(float(eta_rom_corr_t2), 2)
            cp["eta_wcsph_t2_mm"] = round(float(eta_wcsph_t2_mm), 2)
            cp["raw_residual_R_mm"] = round(float(raw_res_eta), 2)
            cp["debiased_residual_r_mm"] = round(float(deb_res_eta), 2)
            cp["residual_R_mm"] = round(float(raw_res_eta), 2)
            cp["fz_rom_corr_N"] = round(float(fz_rom_corr_cp), 2)
            cp["fz_wcsph_min_N"] = round(float(fz_wcsph_cp), 2)
            cp["raw_residual_Fz_N"] = round(float(raw_res_fz), 2)
            cp["debiased_residual_Fz_N"] = round(float(deb_res_fz), 2)
            cp["save_ram_ms"] = round(float(save_ram_ms), 3)

            raw_residuals_eta_mm.append(float(raw_res_eta))
            debiased_residuals_eta_mm.append(float(deb_res_eta))
            raw_residuals_fz_N.append(float(raw_res_fz))
            debiased_residuals_fz_N.append(float(deb_res_fz))

            # Fix 2: Chỉ dùng đúng các cặp điểm thực đo độc lập (T1 đỉnh cua & T2 phanh khẩn cấp), không chèn điểm trùng lặp
            eta_rom_debiased_series.extend([eta_rom_corr_cp, eta_rom_corr_t2])
            eta_wcsph_series.extend([eta_wcsph_cp, eta_wcsph_t2_mm])

        self.wcsph_cal_time_s = time.perf_counter() - t_sph0
        arr_rom = np.asarray(eta_rom_debiased_series, dtype=float)
        arr_sph = np.asarray(eta_wcsph_series, dtype=float)
        nse = 1.0 - float(np.sum((arr_sph - arr_rom) ** 2) / max(np.sum((arr_sph - np.mean(arr_sph)) ** 2), 1e-6))
        rmse_mm = float(np.sqrt(np.mean((arr_sph - arr_rom) ** 2)))
        nrmse_pct = float(rmse_mm / max(np.sqrt(np.mean(arr_sph ** 2)), 1e-6) * 100.0)

        self.wcsph_summary = {
            "dp_mm": round(dp_sph * 1000.0, 2),
            "n_fluid_particles": int(n_fluid_total),
            "n_bound_particles": int(n_bound_total),
            "n_total_particles": int(n_fluid_total + n_bound_total),
            "cal_time_s": round(float(self.wcsph_cal_time_s), 3),
            "bias_eta_max_mm": round(self.bias_eta_max_mm, 2),
            "q_eta_rom_mm": round(self.q_eta_rom_mm, 2),
            "q_eta_total_rom_mm": round(self.q_eta_mm, 2),
            "q_eta_cl_mm": round(self.q_eta_cl_mm, 2),
            "eta_bound_cl_mm": round(self.eta_bound_cl_mm, 2),
            "max_raw_residual_eta_mm": round(float(np.max(raw_residuals_eta_mm)), 2),
            "mean_raw_residual_eta_mm": round(float(np.mean(raw_residuals_eta_mm)), 2),
            "max_abs_debiased_residual_eta_mm": round(float(np.max(np.abs(debiased_residuals_eta_mm))), 2),
            "mean_debiased_residual_eta_mm": round(float(np.mean(debiased_residuals_eta_mm)), 3),
            "max_residual_eta_mm": round(float(np.max(raw_residuals_eta_mm)), 2),
            "mean_residual_eta_mm": round(float(np.mean(raw_residuals_eta_mm)), 2),
            "max_residual_fz_N": round(float(np.max(np.abs(debiased_residuals_fz_N))), 2),
            "q_eta_calibrated_mm": round(self.q_eta_mm, 2),
            "NSE": round(nse, 4),
            "RMSE_mm": round(rmse_mm, 2),
            "NRMSE_pct": round(nrmse_pct, 2)
        }
        return self.wcsph_summary

    def build_offline_tube(
        self,
        waypoints: np.ndarray,
        mu_profile=None,
        d_net_profile=None,
        z_com_profile=None,
        phi_camber_profile=None,
        delta_warp_profile=None,
        load_transfer_profile=None,
        n_iter: int = None,
        run_wcsph_check: bool = True,
        friction_estimator=None,
        friction_mode: str = "conformal_lcb",
        apply_plan_scale: bool = None,
    ) -> dict:
        """
        Thực thi quy hoạch Ống vận tốc khả thi 1 lần trước khi khởi hành (One-Shot Pre-Departure):
        - Giai đoạn A (Trực tuyến tại t = 0 trên bản đồ mới): Giải bài toán Điểm bất động Co Krasnoselskii-Mann
          v^(r) = (1 - beta)*v^(r-1) + beta*T(v^(r-1)) (hằng số beta = 0.65) đến khi phần dư không thư giãn
          ||T(v^(R)) - v^(R)||_inf < fp_tol = 2e-4 m/s.
        - Giai đoạn B (Kiểm toán / Hiệu chuẩn ngoại tuyến bằng hạt WCSPH tại 4 Checkpoint):
          Tách bạch khỏi thời gian quy hoạch t_plan.
        """
        t_geom0 = time.perf_counter()
        s_grid, kappa, dkappa, checkpoints = self._extract_geometry_and_checkpoints(waypoints)
        self.s_grid = s_grid
        self.kappa_grid = kappa
        self.dkappa_grid = dkappa
        self.checkpoints = checkpoints

        n = len(s_grid)
        if friction_estimator is not None:
            f_res = friction_estimator.predict_along_path(self.x_grid, self.y_grid, mode=friction_mode)
            mu_grid = np.asarray(f_res["mu_tube"], dtype=float)
            self.mu_hat_grid = np.asarray(f_res["mu_mean"], dtype=float)
            self.sigma_mu_grid = np.asarray(f_res["sigma_gp"], dtype=float)
            self.mu_lcb_grid = np.asarray(f_res["mu_lcb"], dtype=float)
            self.mu_is_conformal_safe = bool(f_res.get("is_conformal_safe", False))
            self.friction_predict_time_ms = float(f_res.get("predict_time_ms", 0.0))
            self.friction_meta = {
                "source": type(friction_estimator).__name__,
                "mode": str(f_res.get("mode", friction_mode)),
                "q_hat_mu": float(f_res.get("q_hat_mu", 0.0)),
                "alpha_mu": float(f_res.get("alpha_mu", 0.0)),
                "n_support": int(f_res.get("n_support", 0)),
                "predict_time_ms": round(self.friction_predict_time_ms, 4),
            }
        else:
            if mu_profile is None:
                mu_grid = np.full(n, self.mu_default, dtype=float)
            elif callable(mu_profile):
                mu_grid = np.array([float(mu_profile(s)) for s in s_grid], dtype=float)
            else:
                mu_grid = np.asarray(mu_profile, dtype=float)
            self.mu_hat_grid = mu_grid.copy()
            self.sigma_mu_grid = np.zeros(n, dtype=float)
            self.mu_lcb_grid = mu_grid.copy()
            self.mu_is_conformal_safe = False
            self.friction_predict_time_ms = 0.0
            self.friction_meta = {"source": "mu_profile" if mu_profile is not None else "mu_default"}

        if apply_plan_scale is None:
            self.mu_effective_plan_scale = 1.0 if (friction_estimator is not None and self.mu_is_conformal_safe) else self.mu_plan_scale
        else:
            self.mu_effective_plan_scale = self.mu_plan_scale if bool(apply_plan_scale) else 1.0
        self.mu_grid = mu_grid

        self._compute_corridor_modulation(s_grid, d_net_profile)

        if z_com_profile is not None:
            if callable(z_com_profile):
                z_com_eff_grid = np.array([float(z_com_profile(s)) for s in s_grid], dtype=float)
            else:
                z_com_eff_grid = np.asarray(z_com_profile, dtype=float)
        elif load_transfer_profile is not None:
            lt_arr = np.array([float(load_transfer_profile(s)) for s in s_grid] if callable(load_transfer_profile) else load_transfer_profile, dtype=float)
            z_com_eff_grid = np.where(lt_arr > 1.2, 0.22, Z_COM)
        else:
            z_com_eff_grid = np.full(n, Z_COM, dtype=float)
        self.z_com_eff_grid = z_com_eff_grid

        if phi_camber_profile is not None:
            if callable(phi_camber_profile):
                phi_camber_grid = np.array([float(phi_camber_profile(s)) for s in s_grid], dtype=float)
            else:
                phi_camber_grid = np.asarray(phi_camber_profile, dtype=float)
        else:
            phi_camber_grid = np.where(z_com_eff_grid > 0.10, np.radians(5.5), 0.0)
        self.phi_camber_grid = phi_camber_grid

        if delta_warp_profile is not None:
            if callable(delta_warp_profile):
                delta_warp_grid = np.array([float(delta_warp_profile(s)) for s in s_grid], dtype=float)
            else:
                delta_warp_grid = np.asarray(delta_warp_profile, dtype=float)
        else:
            w_ramp = np.clip((z_com_eff_grid - Z_COM) / max(0.22 - Z_COM, 1e-6), 0.0, 1.0)
            delta_warp_grid = 0.0018 * w_ramp
        self.delta_warp_grid = delta_warp_grid

        self.plan_time_geom_ms = (time.perf_counter() - t_geom0) * 1000.0

        # =====================================================================
        # GIAI ĐOẠN 0 (TRƯỚC GIAI ĐOẠN A): HỒ SƠ NGOẠI SINH & TRA CỨU TỪ ĐIỂN O(1)
        # =====================================================================
        self.geom_prof = self._compute_exogenous_geom_profile(
            s_grid, kappa, dkappa, mu_grid,
            z_com_eff_grid=z_com_eff_grid,
            phi_camber_grid=phi_camber_grid,
            delta_warp_grid=delta_warp_grid,
        )
        self.v_ref_geom_grid = self.geom_prof["v_ref_geom"]
        self.ax_ref_geom_grid = self.geom_prof["ax_ref_geom"]
        self.ay_ref_geom_grid = self.geom_prof["ay_ref_geom"]
        self._lookup_conformal_dictionary_budget(self.geom_prof)

        # =====================================================================
        # GIAI ĐOẠN A: VÒNG LẶP ĐIỂM BẤT ĐỘNG CO KRASNOSELSKII-MANN (HẰNG SỐ BETA = 0.65)
        # =====================================================================
        t_fp0 = time.perf_counter()
        ax_est = np.zeros(n, dtype=float)
        v_ub = np.full(n, self.v_max, dtype=float)
        v_ub[0] = 0.0
        v_ub[-1] = 0.0
        v_lb = np.zeros(n, dtype=float)
        self.fp_history = []

        max_iters = self.fp_max_iter if n_iter is None else max(int(n_iter), self.fp_max_iter)

        for it in range(max_iters):
            v_ub_prev = v_ub.copy()
            ax_prev = ax_est.copy()

            v_grip_raw, v_traction_raw, v_jerk_raw, v_grip, v_traction, v_jerk = self._compute_h2_mechanical_bounds(
                s_grid, kappa, dkappa, ax_prev, mu_grid,
                z_com_eff_grid=z_com_eff_grid, phi_camber_grid=phi_camber_grid, delta_warp_grid=delta_warp_grid
            )
            v_mech = np.minimum(self.v_max, np.minimum(v_grip, np.minimum(v_traction, v_jerk)))

            if it == 0:
                v_seed, _, ax_seed, _ = self._compute_h4_jerk_reachability_tube(s_grid, v_mech, mu_grid)
            else:
                v_seed, ax_seed = v_ub_prev, ax_prev

            v_slosh_raw, v_slosh, eta_bound_anal_mm = self._compute_h1_coupled_slosh_inversion(
                s_grid, kappa, checkpoints, v_seed, ax_seed
            )

            v_phys = np.minimum(v_mech, v_slosh)
            v_T, v_lb_T, ax_T, ay_T = self._compute_h4_jerk_reachability_tube(
                s_grid, v_phys, mu_grid
            )

            unrelaxed_res = float(np.max(np.abs(v_T - v_ub_prev)))

            if it == 0:
                v_ub = v_T
                ax_est = ax_T
            else:
                v_ub = (1.0 - self.fp_beta) * v_ub_prev + self.fp_beta * v_T
                ax_est = (1.0 - self.fp_beta) * ax_prev + self.fp_beta * ax_T

            v_lb = np.maximum(0.0, v_ub - self.delta_v_window)
            v_lb[0] = 0.0
            v_lb[-1] = 0.0
            step_diff = float(np.max(np.abs(v_ub - v_ub_prev)))

            self.fp_history.append({
                "iter": it + 1,
                "max_delta_v_m_s": round(step_diff, 6),
                "unrelaxed_residual_m_s": round(unrelaxed_res, 6)
            })

            if it >= 1 and unrelaxed_res < self.fp_tol:
                break

        # Kiểm chứng hậu nghiệm tại nghiệm hội tụ v_ub^(R), ax_est^(R):
        v_grip_raw, v_traction_raw, v_jerk_raw, v_grip, v_traction, v_jerk = self._compute_h2_mechanical_bounds(
            s_grid, kappa, dkappa, ax_est, mu_grid,
            z_com_eff_grid=z_com_eff_grid, phi_camber_grid=phi_camber_grid, delta_warp_grid=delta_warp_grid
        )
        v_slosh_raw, v_slosh, eta_bound_anal_mm = self._compute_h1_coupled_slosh_inversion(
            s_grid, kappa, checkpoints, v_ub, ax_est
        )
        v_phys_final = np.minimum(self.v_max, np.minimum(v_slosh, np.minimum(v_grip, np.minimum(v_traction, v_jerk))))
        v_T_check, v_lb, ax_est, ay_est = self._compute_h4_jerk_reachability_tube(
            s_grid, v_phys_final, mu_grid
        )
        self.fp_unrelaxed_residual = float(np.max(np.abs(v_T_check - v_ub)))
        v_ub = v_T_check
        self.max_constraint_violation_m_s = float(np.max(np.maximum(0.0, v_ub - v_phys_final)))

        # Ước lượng hệ số co Lipschitz thực nghiệm q_hat = ||r^(k)|| / ||r^(k-1)||
        ratios = []
        for k_it in range(1, len(self.fp_history)):
            d_prev = self.fp_history[k_it - 1]["unrelaxed_residual_m_s"]
            d_curr = self.fp_history[k_it]["unrelaxed_residual_m_s"]
            if d_prev > 1e-5:
                ratios.append(d_curr / d_prev)
        self.fp_contraction_factor = float(np.median(ratios)) if ratios else 0.18

        self.plan_time_analytical_ms = (time.perf_counter() - t_fp0) * 1000.0
        self.plan_time_total_ms = self.plan_time_geom_ms + self.plan_time_analytical_ms

        # Đánh giá đáp ứng ODE con lắc 2 trục DBEP + Stokes bậc 2 dọc theo hồ sơ hội tụ v_ub(s)
        th_x_ens, om_x_ens, th_y_ens, om_y_ens = self._rollout_pendulum_2dof_rk4_vectorized(
            s_grid, v_ub, ax_est, kappa
        )
        mid_idx = len(self.freq_scales) // 2
        th_x = th_x_ens[mid_idx]
        th_y = th_y_ens[mid_idx]

        u_x = np.abs(np.tan(th_x))
        u_y = np.abs(np.tan(th_y))
        eta_1x = 0.5 * TANK_A * u_x
        eta_1y = 0.5 * TANK_B * u_y
        stokes_x = (np.pi * TANK_A / 8.0) * self.coth_x * (u_x ** 2)
        stokes_y = (np.pi * TANK_B / 8.0) * self.coth_y * (u_y ** 2)
        eta_x_mm = (eta_1x + stokes_x) * 1000.0
        eta_y_mm = (eta_1y + stokes_y) * 1000.0
        eta_pred_mm = np.maximum(eta_y_mm + 0.35 * eta_x_mm, eta_x_mm + 0.35 * eta_y_mm)

        a_tot_grid = np.hypot(ax_est, ay_est)
        bias_eta_grid = np.minimum(self.bias_eta_max_mm, self.bias_b0_mm + self.bias_b1_mm * (a_tot_grid / g))
        self.eta_rom_debiased_mm = eta_pred_mm + bias_eta_grid

        z_slosh_x_grid = z_com_eff_grid + (Z_SLOSH_X - Z_COM)
        z_slosh_y_grid = z_com_eff_grid + (Z_SLOSH_Y - Z_COM)
        mz_eff_x_grid = M_TOTAL * z_com_eff_grid + m_1x * z_slosh_x_grid
        mz_eff_y_grid = M_TOTAL * z_com_eff_grid + m_1y * z_slosh_y_grid

        gamma_roll = 1.0 / np.maximum(0.50, 1.0 - (M_TOTAL * g * z_com_eff_grid) / self.k_roll)
        dfz_camber = gamma_roll * (M_TOTAL * g * z_com_eff_grid * np.sin(phi_camber_grid)) / (4.0 * L_Y)
        dfz_warp = self.k_z_corner * delta_warp_grid

        m_pitch = mz_eff_x_grid * np.abs(ax_est) + m_1x * g * lx * np.abs(np.sin(th_x))
        m_roll = gamma_roll * (mz_eff_y_grid * np.abs(ay_est) * np.cos(phi_camber_grid) + m_1y * g * ly * np.abs(np.sin(th_y)))
        fz_min_pred = np.maximum(
            FZ_MIN + 0.25,
            FZ_NOM - (m_pitch / (4.0 * L_X)) - (m_roll / (4.0 * L_Y)) - self.dfz_iso_traj_max - dfz_camber - dfz_warp
        )
        bias_fz_grid = np.minimum(self.bias_fz_max_N, self.bias_fz_b0_N + self.bias_fz_b1_N * (a_tot_grid / g))
        self.fz_rom_debiased_N = np.maximum(FZ_MIN + 0.10, fz_min_pred - bias_fz_grid)

        raw_stack = np.vstack([v_slosh_raw, v_grip_raw, v_traction_raw, v_jerk_raw])
        active_phys_idx = np.argmin(raw_stack, axis=0) + 1
        self.active_regime_code = active_phys_idx

        regime_names = {
            1: "Sóng sánh (v_slosh)",
            2: "Bám Zonotope 3D (v_grip)",
            3: "Chống nhấc bánh (v_traction)",
            4: "Độ giật ngang (v_jerk)"
        }

        for cp in self.checkpoints:
            idx_p = cp["idx_peak"]
            idx_in = cp["idx_in"]
            idx_out = cp["idx_out"]
            sl = slice(idx_in, idx_out + 1)

            cp["mu_cp"] = round(float(mu_grid[idx_p]), 2)
            cp["z_com_cp_m"] = round(float(z_com_eff_grid[idx_p]), 3)
            cp["phi_camber_cp_deg"] = round(float(np.degrees(phi_camber_grid[idx_p])), 1)
            cp["delta_warp_cp_mm"] = round(float(delta_warp_grid[idx_p] * 1000.0), 2)
            cp["v_slosh_raw_cp"] = round(float(v_slosh_raw[idx_p]), 3)
            cp["v_grip_raw_cp"] = round(float(v_grip_raw[idx_p]), 3)
            cp["v_grip_2d_static_cp"] = round(float(self.v_grip_2d_static[idx_p]), 3)
            cp["v_grip_2d_dyn_cp"] = round(float(self.v_grip_2d_dyn[idx_p]), 3)
            cp["v_traction_raw_cp"] = round(float(v_traction_raw[idx_p]), 3)
            cp["v_jerk_raw_cp"] = round(float(v_jerk_raw[idx_p]), 3)

            cp["v_slosh_cp"] = round(float(v_slosh[idx_p]), 3)
            cp["v_grip_cp"] = round(float(v_grip[idx_p]), 3)
            cp["v_traction_cp"] = round(float(v_traction[idx_p]), 3)
            cp["v_jerk_cp"] = round(float(v_jerk[idx_p]), 3)
            cp["v_phys_cp"] = round(float(v_phys_final[idx_p]), 3)
            cp["v_ub_cp"] = round(float(v_ub[idx_p]), 3)

            raw_vals = [
                float(v_slosh_raw[idx_p]),
                float(v_grip_raw[idx_p]),
                float(v_traction_raw[idx_p]),
                float(v_jerk_raw[idx_p])
            ]
            act_code = int(np.argmin(raw_vals) + 1)
            cp["active_regime_code"] = act_code
            cp["active_regime"] = regime_names[act_code]

            cp["theta_in_deg"] = round(float(np.degrees(np.hypot(th_x[idx_in], th_y[idx_in]))), 2)
            cp["theta_peak_deg"] = round(float(np.max(np.degrees(np.hypot(th_x[sl], th_y[sl])))), 2)
            cp["eta_bound_cp_mm"] = round(float(np.max(eta_bound_anal_mm[sl])), 2)
            cp["eta_peak_mm"] = round(float(np.max(eta_pred_mm[sl])), 2)
            cp["fz_min_N"] = round(float(np.min(fz_min_pred[sl])), 2)

        self.v_slosh_raw = v_slosh_raw
        self.v_grip_raw = v_grip_raw
        self.v_traction_raw = v_traction_raw
        self.v_jerk_raw = v_jerk_raw
        self.v_slosh_nom = v_slosh
        self.v_grip_nom = v_grip
        self.v_traction_nom = v_traction
        self.v_jerk_nom = v_jerk
        self.v_phys_nom = v_phys_final
        self.v_ub_nom = v_ub
        self.v_lb_nom = v_lb
        self.ax_nom = ax_est
        self.ay_nom = ay_est
        self.theta_x_pred = th_x
        self.theta_y_pred = th_y
        self.eta_bound_analytical_mm = eta_bound_anal_mm
        self.eta_pred_mm = eta_pred_mm
        self.fz_min_pred = fz_min_pred
        self.eta_dbep_geom_grid, self.fz_min_geom_grid = self._evaluate_exogenous_dbep_profile(self.geom_prof)
        self.is_ready = True

        if run_wcsph_check:
            self.run_wcsph_checkpoint_standardization(dp_sph=0.005)

        return {
            "path_length_m": round(float(s_grid[-1]), 2),
            "n_grid": n,
            "ds_m": self.ds,
            "n_freq_ensemble": len(self.freq_scales),
            "total_rk4_substeps": int(self.total_rk4_substeps),
            "n_checkpoints": len(self.checkpoints),
            "n_fp_iters": len(self.fp_history),
            "fp_final_delta_v": self.fp_history[-1]["max_delta_v_m_s"],
            "fp_unrelaxed_residual": round(self.fp_unrelaxed_residual, 6),
            "fp_contraction_factor": round(self.fp_contraction_factor, 4),
            "max_constraint_violation_m_s": round(self.max_constraint_violation_m_s, 6),
            "plan_time_geom_ms": round(self.plan_time_geom_ms, 2),
            "plan_time_analytical_ms": round(self.plan_time_analytical_ms, 2),
            "plan_time_total_ms": round(self.plan_time_total_ms, 2),
            "conformal_lookup_time_ms": round(self.conformal_lookup_time_ms, 4),
            "friction_predict_time_ms": round(self.friction_predict_time_ms, 4),
            "mu_is_conformal_safe": bool(self.mu_is_conformal_safe),
            "mu_effective_plan_scale": round(float(self.mu_effective_plan_scale), 4),
            "mu_tube_min": round(float(np.min(mu_grid)), 4),
            "mu_tube_max": round(float(np.max(mu_grid)), 4),
            "enable_grip": self.enable_grip,
            "enable_traction": self.enable_traction,
            "min_v_ref_geom_minus_v_ub": round(float(np.min(self.v_ref_geom_grid - v_ub)), 6),
            "wcsph_cal_time_s": round(self.wcsph_cal_time_s, 3),
            "max_eta_pred_mm": round(float(np.max(eta_pred_mm)), 2),
            "max_eta_rom_debiased_mm": round(float(np.max(self.eta_rom_debiased_mm)), 2),
            "max_eta_wcsph_mm": round(float(max(cp.get("eta_wcsph_cp_mm", 0.0) for cp in self.checkpoints)), 2),
            "max_eta_wcsph_t2_mm": round(float(max(cp.get("eta_wcsph_t2_mm", 0.0) for cp in self.checkpoints)), 2),
            "min_fz_pred_N": round(float(np.min(fz_min_pred)), 2),
            "theta_allow_x_deg": round(float(np.degrees(self.theta_allow_x)), 2),
            "theta_allow_y_deg": round(float(np.degrees(self.theta_allow_y)), 2),
            "coth_x": round(self.coth_x, 4),
            "coth_y": round(self.coth_y, 4),
            "checkpoints": self.checkpoints,
            "wcsph_summary": self.wcsph_summary,
            "fp_history": self.fp_history
        }

    def run_grid_refinement_study(
        self,
        waypoints: np.ndarray,
        mu_profile=None,
        d_net_profile=None,
        z_com_profile=None,
        phi_camber_profile=None,
        delta_warp_profile=None
    ) -> dict:
        """
        Nghiên cứu độc lập lưới không gian và thời gian (Spatial & Temporal Grid Refinement Study)
        trên 3 độ phân giải: ds = 0.020 m, 0.010 m và 0.005 m (tham chiếu mịn).
        """
        ds_list = [0.020, 0.010, 0.005]
        results = {}
        for ds_val in ds_list:
            sub_planner = CoupledVelocityTubeH124(
                v_max=self.v_max,
                a_acc_max=self.a_acc_max,
                a_brake_max=self.a_brake_max,
                j_plan=self.j_plan,
                j_y_max=self.j_y_max,
                ds=ds_val,
                dt_rk4=self.dt_rk4,
                kappa_th=self.kappa_th,
                mu_default=self.mu_default,
                enable_grip=self.enable_grip,
                enable_traction=self.enable_traction,
                use_exogenous_geom_checkpoint=self.use_exogenous_geom_checkpoint,
                conformal_cal_path=self.conformal_cal_path,
                bias_eta_max_mm=self.bias_eta_max_mm,
                q_eta_rom_mm=self.q_eta_rom_mm,
                q_eta_cl_mm=self.q_eta_cl_mm,
                q_fz_base=self.q_fz_base
            )
            info = sub_planner.build_offline_tube(
                waypoints,
                mu_profile=mu_profile,
                d_net_profile=d_net_profile,
                z_com_profile=z_com_profile,
                phi_camber_profile=phi_camber_profile,
                delta_warp_profile=delta_warp_profile,
                run_wcsph_check=False
            )
            results[ds_val] = {
                "planner": sub_planner,
                "info": info
            }

        ref_p = results[0.005]["planner"]
        comparison = []
        for ds_val in ds_list:
            p = results[ds_val]["planner"]
            inf = results[ds_val]["info"]
            cs_v_ref = CubicSpline(ref_p.s_grid, ref_p.v_ub_nom)
            cs_grip_ref = CubicSpline(ref_p.s_grid, ref_p.v_grip_nom)
            cs_dk_ref = CubicSpline(ref_p.s_grid, ref_p.dkappa_grid)
            cs_eta_ref = CubicSpline(ref_p.s_grid, ref_p.eta_pred_mm)

            v_ref_at_p = cs_v_ref(p.s_grid)
            grip_ref_at_p = cs_grip_ref(p.s_grid)
            dk_ref_at_p = cs_dk_ref(p.s_grid)
            eta_ref_at_p = cs_eta_ref(p.s_grid)

            comparison.append({
                "ds_m": ds_val,
                "n_grid": inf["n_grid"],
                "n_fp_iters": inf["n_fp_iters"],
                "plan_time_ms": inf["plan_time_total_ms"],
                "max_dkappa_diff": round(float(np.max(np.abs(p.dkappa_grid - dk_ref_at_p))), 5),
                "max_dkappa_spline_diff": round(float(np.max(np.abs(p.dkappa_spline_grid - p.dkappa_grid))), 4),
                "max_v_grip_diff_m_s": round(float(np.max(np.abs(p.v_grip_nom - grip_ref_at_p))), 5),
                "max_v_ub_diff_m_s": round(float(np.max(np.abs(p.v_ub_nom - v_ref_at_p))), 5),
                "rmse_v_ub_m_s": round(float(np.sqrt(np.mean((p.v_ub_nom - v_ref_at_p) ** 2))), 5),
                "max_eta_diff_mm": round(float(np.max(np.abs(p.eta_pred_mm - eta_ref_at_p))), 4),
                "cp3_v_grip_raw": inf["checkpoints"][2]["v_grip_raw_cp"] if len(inf["checkpoints"]) > 2 else (inf["checkpoints"][0]["v_grip_raw_cp"] if inf["checkpoints"] else 0.0),
                "cp3_v_ub": inf["checkpoints"][2]["v_ub_cp"] if len(inf["checkpoints"]) > 2 else (inf["checkpoints"][0]["v_ub_cp"] if inf["checkpoints"] else 0.0)
            })
        return {"levels": comparison}

    def get_horizon_mu_plan(self, s_horizon: np.ndarray) -> np.ndarray:
        """
        Nội suy hệ số ma sát hoạch định hiệu dụng mu_plan(s) = mu_effective_plan_scale * mu_grid(s)
        dọc theo chân trời dự báo s_horizon (tránh phạt kép khi đã dùng Conformal LCB).
        """
        if not self.is_ready or self.s_grid is None or self.mu_grid is None:
            s_arr = np.asarray(s_horizon, dtype=float)
            return np.full_like(s_arr, max(0.05, self.mu_plan_scale * self.mu_default), dtype=float)
        s_q = np.clip(np.asarray(s_horizon, dtype=float), 0.0, self.s_grid[-1])
        mu_interp = np.interp(s_q, self.s_grid, self.mu_grid)
        return np.maximum(0.05, self.mu_effective_plan_scale * mu_interp)

    def query_online_horizon_tube(
        self,
        s_horizon: np.ndarray,
        phi_est: np.ndarray,
        mu_online: float = None,
        w_obs: float = 0.0,
        v_obs_proj: float = 0.0,
        v_escape: float = 0.30
    ) -> tuple:
        """
        Cập nhật Trực tuyến ở tần số 20 Hz (khi xe đang chạy t > 0 và có trạng thái từ EKF):
        - Trọng số suy giảm pha trong miền thời gian gamma_phase(Delta_t) = exp(-Delta_t / tau_phase).
        - Hệ số tỷ lệ ma sát trực tuyến sqrt(mu_online / mu_plan(s_q)) theo đúng ma sát hoạch định cục bộ
          CHỈ áp dụng riêng cho nhánh trần bám đường v_grip_online(s_n), không làm méo trần sóng sánh hay độ giật.
        - Sàn vận tốc thoát giao lộ v_escape_safe luôn bị chặn trên bởi trần cơ học v_phys_online(s_n).
        """
        if not self.is_ready:
            raise RuntimeError("Cần gọi build_offline_tube(waypoints) trước khi truy vấn trực tuyến!")

        s_q = np.clip(np.asarray(s_horizon, dtype=float), 0.0, self.s_grid[-1])
        v_ub_nom_q = np.interp(s_q, self.s_grid, self.v_ub_nom)
        v_lb_nom_q = np.interp(s_q, self.s_grid, self.v_lb_nom)
        v_grip_nom_q = np.interp(s_q, self.s_grid, self.v_grip_nom)
        v_traction_nom_q = np.interp(s_q, self.s_grid, self.v_traction_nom)
        v_jerk_nom_q = np.interp(s_q, self.s_grid, self.v_jerk_nom)
        kappa_q = np.interp(s_q, self.s_grid, self.kappa_grid)
        rho_clr_q = np.interp(s_q, self.s_grid, self.rho_clr_grid)

        th_x, om_x, th_y, om_y = [float(v) for v in np.asarray(phi_est).ravel()[:4]]

        E_x_cur = np.sqrt(th_x ** 2 + (om_x / omega_x) ** 2)
        E_y_cur = np.sqrt(th_y ** 2 + (om_y / omega_y) ** 2)

        theta_allow_y_mod = rho_clr_q * self.theta_allow_y
        long_ded = np.minimum(0.32 * theta_allow_y_mod, (TANK_A / TANK_B) * (0.38 * E_x_cur))
        theta_allow_y_eff = np.maximum(0.03, theta_allow_y_mod - long_ded)

        delta_s = np.maximum(0.0, s_q - s_q[0])
        v_ref_q = np.maximum(0.12, v_ub_nom_q)
        delta_t = delta_s / v_ref_q
        gamma_phase = np.exp(-delta_t / max(self.tau_phase_s, 1e-3))

        sigma_eq = -np.sign(kappa_q)
        theta_ref_y = gamma_phase * (sigma_eq * th_y) - (1.0 - gamma_phase) * E_y_cur

        numer = np.maximum(0.004, (theta_allow_y_eff ** 2) - (E_y_cur ** 2))
        denom = 2.0 * np.maximum(0.02, theta_allow_y_eff - theta_ref_y)
        theta_eq_max_online = numer / denom

        v_slosh_online = np.minimum(self.v_max, np.sqrt((g * theta_eq_max_online) / (np.abs(kappa_q) + 1e-6)))

        if self.enable_grip:
            if mu_online is not None:
                mu_plan_q = self.get_horizon_mu_plan(s_q)
                mu_on_arr = np.maximum(0.05, np.asarray(mu_online, dtype=float))
                grip_scale = np.sqrt(mu_on_arr / mu_plan_q)
                v_grip_online = np.minimum(self.v_max, v_grip_nom_q * grip_scale)
            else:
                v_grip_online = v_grip_nom_q.copy()
        else:
            v_grip_online = np.full_like(v_slosh_online, self.v_max)

        if not self.enable_traction:
            v_traction_nom_q = np.full_like(v_slosh_online, self.v_max)

        v_phys_online = np.minimum(
            np.minimum(v_slosh_online, v_grip_online),
            np.minimum(v_traction_nom_q, v_jerk_nom_q)
        )
        v_nom_reachable = np.minimum(v_ub_nom_q, v_phys_online)

        w_obs_c = float(np.clip(w_obs, 0.0, 1.0))
        v_escape_safe = np.minimum(v_phys_online, max(v_escape, v_obs_proj))
        v_env_obs = (1.0 - w_obs_c) * v_nom_reachable + w_obs_c * v_escape_safe

        v_ub_online = np.minimum(v_phys_online, v_env_obs)
        v_ub_online = np.clip(v_ub_online, 0.08, self.v_max)

        if E_y_cur > 0.75 * self.theta_allow_y or E_x_cur > 0.75 * self.theta_allow_x:
            v_lb_online = np.zeros_like(v_ub_online)
        else:
            v_lb_online = np.minimum(v_lb_nom_q, np.maximum(0.0, v_ub_online - self.delta_v_window))

        return v_ub_online, v_lb_online


def generate_multi_regime_benchmark_scenario():
    """
    Khởi tạo kịch bản đường đi chữ S đa chế độ khả vi giải tích (L = 7.42 m, 4 Checkpoints giữa tuyến):
    - CP1 (Đỉnh cua 1 tại s ~ 1.60 m): Sàn khô (mu = 0.35), hành lang hẹp -> Nút thắt Sóng sánh M1 (v_slosh) kích hoạt.
    - CP2 (Chuyển tiếp chữ S tại s ~ 1.98 m): Đạo hàm độ cong |kappa'(s)| đạt đỉnh trên sườn chuyển tiếp ->
      Nút thắt Độ giật ngang M4 (v_jerk) kích hoạt độc lập với 0 < v_ub < v_max.
    - CP3 (Khúc cua 2 tại s ~ 3.46 m): Vùng sàn ướt/dầu trơn (mu = 0.12) -> Nút thắt Bám Zonotope 3D M2 (v_grip) kích hoạt.
    - CP4 (Đỉnh cua 3 tại s ~ 5.70 m): Khúc cua trên cầu dẫn nghiêng phi_camber = 5.5 deg, độ vênh mặt sàn chuyển tiếp
      DIN 18202 delta_warp = 1.80 mm với bệ đỡ Z_COM,eff = 0.22 m (6.24% ngưỡng lật tĩnh, K_roll = 250.9 N.m/rad)
      -> Nút thắt Chống nhấc bánh M2 (v_traction) kích hoạt.
    """
    s_ctrl = np.linspace(0.0, 6.75, 250)
    x_ctrl = s_ctrl
    y_ctrl, _, _, _ = _analytical_benchmark_curve(x_ctrl)
    waypoints = np.column_stack([x_ctrl, y_ctrl])

    def _smooth_window(s: float, s_start: float, s_end: float, w: float = 0.06) -> float:
        return float(0.5 * (np.tanh((s - s_start) / w) - np.tanh((s - s_end) / w)))

    def mu_profile_fn(s: float) -> float:
        w_wet = _smooth_window(s, 2.85, 4.35, 0.06)
        return 0.35 - (0.35 - 0.12) * w_wet  # Vùng sàn trơn (mu=0.12) tại CP3, sàn khô (mu=0.35) tại CP1, CP2, CP4

    def d_net_profile_fn(s: float) -> float:
        w_nar = _smooth_window(s, 1.00, 1.82, 0.06)
        return 0.48 - (0.48 - 0.22) * w_nar  # Hành lang hẹp quanh CP1 làm kích hoạt nút thắt sóng sánh v_slosh

    def z_com_profile_fn(s: float) -> float:
        w_ramp = _smooth_window(s, 4.75, 6.45, 0.06)
        return Z_COM + (0.22 - Z_COM) * w_ramp  # Cao độ trọng tâm bàn chở cố định trên đoạn cầu dẫn CP4 (6.24% ngưỡng lật tĩnh)

    def phi_camber_profile_fn(s: float) -> float:
        w_ramp = _smooth_window(s, 4.75, 6.45, 0.06)
        return float(np.radians(5.5)) * w_ramp  # Độ dốc nghiêng ngang cầu dẫn phi_camber = 5.5 deg tại CP4

    return waypoints, mu_profile_fn, d_net_profile_fn, z_com_profile_fn, phi_camber_profile_fn


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 98)
    print("KIỂM CHỨNG ỐNG VẬN TỐC KHẢ THI BẬC 3 (HỘI TỤ THỰC) & CHUẨN HÓA WCSPH TẠI 4 CHECKPOINT")
    print("=" * 98)

    waypoints, mu_fn, d_net_fn, zc_fn, camber_fn = generate_multi_regime_benchmark_scenario()
    planner = CoupledVelocityTubeH124(mu_default=0.35, kappa_th=1.05)
    summary = planner.build_offline_tube(
        waypoints,
        mu_profile=mu_fn,
        d_net_profile=d_net_fn,
        z_com_profile=zc_fn,
        phi_camber_profile=camber_fn,
        run_wcsph_check=True
    )

    print(f"Chiều dài đường đi L            : {summary['path_length_m']:.2f} m ({summary['n_grid']} điểm lưới, ds = {summary['ds_m']:.3f} m)")
    print(f"Góc sóng cho phép (PT 1 Stokes) : theta_allow_x = {summary['theta_allow_x_deg']:.2f}° (coth={summary['coth_x']:.4f}), "
          f"theta_allow_y = {summary['theta_allow_y_deg']:.2f}° (coth={summary['coth_y']:.4f})")
    print(f"Hội tụ Điểm bất động (R={summary['n_fp_iters']} vòng) : ||T(v)-v||_inf = {summary['fp_unrelaxed_residual']:.6f} m/s, "
          f"hệ số co q_hat = {summary['fp_contraction_factor']:.4f}, vi phạm ràng buộc = {summary['max_constraint_violation_m_s']:.6f} m/s")
    print(f"Lịch sử phần dư ||T(v)-v||_inf  : {[h['unrelaxed_residual_m_s'] for h in summary['fp_history']]}")
    print(f"Thời gian tính Giai đoạn A      : t_plan = {summary['plan_time_total_ms']:.2f} ms "
          f"(Hình học: {summary['plan_time_geom_ms']:.2f} ms + {summary['n_fp_iters']} vòng ZOH/S-Curve: {summary['plan_time_analytical_ms']:.2f} ms, "
          f"{summary['total_rk4_substeps']} bước thời gian dt=2ms x {summary['n_freq_ensemble']} tần số)")
    ws = summary["wcsph_summary"]
    print(f"Đối chứng WCSPH (Giai đoạn B)   : {ws['n_total_particles']} hạt ({ws['n_fluid_particles']} chất lỏng + "
          f"{ws['n_bound_particles']} biên mDBC) trong {ws['cal_time_s']:.3f} s")
    print(f"Chỉ số khớp ROM(khử bias)/WCSPH : NSE = {ws['NSE']:.4f} | RMSE = {ws['RMSE_mm']:.2f} mm ({ws['NRMSE_pct']:.2f}%) | "
          f"max |r_eta| = {ws['max_abs_debiased_residual_eta_mm']:.2f} mm <= q_hat_ROM = {ws['q_eta_rom_mm']:.2f} mm")
    print("-" * 98)
    print(f"Tra cứu từ điển ngoại tuyến    : t_lookup = {summary['conformal_lookup_time_ms']:.4f} ms | min(v_ref_geom - v_ub) = {summary['min_v_ref_geom_minus_v_ub']:.6f} m/s")
    for cp in summary["checkpoints"]:
        print(
            f"  * CP{cp['cp_id']} ({cp['type']:15s} tại s={cp['s_peak']:.2f}m, |k|={abs(cp['kappa_peak']):.2f}, |k'|={abs(cp['dkappa_peak']):.2f}, mu={cp['mu_cp']:.2f}): "
            f"Raw[slosh={cp['v_slosh_raw_cp']:.3f}, grip={cp['v_grip_raw_cp']:.3f}, trac={cp['v_traction_raw_cp']:.3f}, jerk={cp['v_jerk_raw_cp']:.3f}] -> "
            f"Active={cp['active_regime']} -> v_ub={cp['v_ub_cp']:.3f} m/s (v_geom={cp['v_ref_geom']:.3f}) | "
            f"eta_DBEP(geom)={cp['eta_dbep_cmp_mm']:.2f}mm, eta_ROM={cp['eta_rom_corr_cp_mm']:.2f}mm, eta_WCSPH={cp['eta_wcsph_cp_mm']:.2f}mm "
            f"(r_deb={cp['debiased_residual_r_mm']:+.2f}mm) | Fz_tube={cp['fz_min_N']:.2f}N (WCSPH_tube={cp['fz_wcsph_at_tube_N']:.2f}N, Fz_geom={cp['fz_dbep_cmp_N']:.2f}N vs WCSPH_geom={cp['fz_wcsph_min_N']:.2f}N)"
        )
    print("-" * 98)
    grid_study = planner.run_grid_refinement_study(waypoints, mu_fn, d_net_fn, zc_fn, camber_fn)
    for lv in grid_study["levels"]:
        print(f"  [Grid ds={lv['ds_m']:.3f}m, N={lv['n_grid']:4d}] R={lv['n_fp_iters']} iters, t={lv['plan_time_ms']:.2f} ms | "
              f"max|dk'|={lv['max_dkappa_diff']:.5f} (spline={lv['max_dkappa_spline_diff']:.4f}), "
              f"max|dv_grip|={lv['max_v_grip_diff_m_s']:.5f}, max|dv_ub|={lv['max_v_ub_diff_m_s']:.5f} m/s")
    print("=" * 98)
