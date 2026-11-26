"""
dfe_mpcc_amr_controller.py
=============================================================================
BỘ ĐIỀU KHIỂN DYNAMIC FRICTION-ELLIPSE MPCC (DFE-MPCC) CHO XE AMR MECANUM (slosh3.xml)
MANG BÌNH CHẤT LỎNG TRÊN MẶT SÀN MA SÁT THẤP / BIẾN THIÊN.

Đặc điểm đóng góp lý thuyết (Major Theoretical Contribution):
1. Không gian trạng thái chuỗi tích phân bậc ba LTI chính xác ZOH.
2. Tích hợp Động lực học Chuyển dịch Tải trọng Pháp tuyến 4 Bánh xe Fz_i(t)
   dưới tác động của mô-men lật quán tính và lực phản kháng thủy động FSI.
3. Ràng buộc Elip Ma sát Động (Dynamic Friction Ellipse Constraints) cho từng
   bánh xe Mecanum thời gian thực: |f_i| <= mu * Fz_i(t).
4. Ràng buộc Chống Nhấc Bánh (Anti-Tipping Constraint): Fz_i >= Fz_min > 0.
5. Bảo đảm Tính Khả Thi Hồi Quy (Guaranteed Recursive Feasibility) qua biến bù
   chuẩn L2 phạt nặng.
6. Ràng buộc Slosh-ECBF bậc 3 hai trục và Obstacle-ECBF bậc 3.
=============================================================================
"""

import os
import sys
import time

_curr = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_curr) if os.path.basename(_curr) in ["controller", "solver"] else _curr
for _p in [_root, os.path.join(_root, "controller"), os.path.join(_root, "solver")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import casadi as cs
import numpy as np
from scipy.interpolate import CubicSpline

# Import thông số vật lý xe AMR và chất lỏng
from robot_model_amr_casadi import (
    build_amr_symbolic_model,
    PHI_MAX_X, PHI_MAX_Y,
    TANK_A, TANK_B, H_LIQUID,
    K_SLOSH_0_X, K_SLOSH_0_Y,
    OMEGA_MAX,
    L_X, L_Y, L_SUM, ROBOT_RADIUS,
    M_TOTAL, IZ_BASE, g, ZETA_SLOSH,
    m_0x, m_1x, lx, omega_x, hd_x,
    m_0y, m_1y, ly, omega_y, hd_y
)
from zonotope_wrench_set import compute_zonotope_facet_matrices
from mecanum_traction_dynamics import Z_COM, Z_SLOSH, Z_SLOSH_X, Z_SLOSH_Y

FZ_STATIC = (M_TOTAL * g) / 4.0  # 17.78 N mỗi bánh danh định tĩnh
FZ_MIN = 3.5        # Ngưỡng tải trọng pháp tuyến tối thiểu chống nhấc bánh (N)


class DFEMPCCAMRController:
    def __init__(self, N=15, dt=0.05, mu=0.35, solver_type='ipopt',
                 state_con_mode="skip0", mu_plan_scale=0.95, ipopt_max_iter=35,
                 h_lim_m=0.030, a_brake_max=0.65,
                 x_bounds=(-2.75, 8.5), y_bounds=(-2.75, 2.75)):
        """
        Khởi tạo bộ điều khiển DFE-MPCC tích hợp Miền Zonotope T1 (12 Mặt Biên)
        và Biên Nhám Mặt Đường Ngẫu Nhiên ISO 8608:2016.
        Parameters:
            N: Tầm nhìn dự báo (Horizon steps), mặc định 15 bước (0.75 s)
            dt: Chu kỳ lấy mẫu (s), mặc định 0.05 s (20 Hz)
            mu: Hệ số ma sát sàn danh định (có thể cập nhật trực tuyến qua P_mu)
            state_con_mode: 'skip0' (U1a - bỏ ràng buộc chỉ phụ thuộc trạng thái ở k=0) hoặc 'full'
            mu_plan_scale: Hệ số biên an toàn ma sát hoạch định T1 (P3 - mặc định 0.95)
            ipopt_max_iter: Giới hạn số vòng lặp IPOPT tối đa (U2 - mặc định 35)
            h_lim_m: Giới hạn mép thành tự do h_free (m), mặc định 0.030 m
            a_brake_max: Gia tốc phanh khẩn cấp cực đại đồng bộ với planner (m/s^2)
            x_bounds: Giới hạn tọa độ X của robot (m)
            y_bounds: Giới hạn tọa độ Y của robot (m)
        """
        self.N = N
        self.dt = dt
        self.default_mu = mu
        self.solver_type = solver_type
        self.state_con_mode = state_con_mode
        self.mu_plan_scale = float(mu_plan_scale)
        self.ipopt_max_iter = int(ipopt_max_iter)
        self.x_bounds = tuple(x_bounds)
        self.y_bounds = tuple(y_bounds)
        # Ngưỡng sóng dùng trong ràng buộc phanh dự phòng T2 (m). Mặc định 30 mm = h_free.
        self.h_lim_m = float(h_lim_m)
        self.a_brake_max = float(a_brake_max)

        # Ma trận 12 mặt biên Zonotope T1 trong không gian Wrench 3D: C_zono @ W <= M_reach @ f_bar
        self.C_zono, self.M_reach = compute_zonotope_facet_matrices(L_SUM)
        self.C_zono_dm = cs.DM(self.C_zono)
        self.M_reach_dm = cs.DM(self.M_reach)

        # Nạp nguồn lượng tử Conformal q_eta từ config hoặc mặc định
        calib_file = os.path.join(_root, "config", "conformal_calibration_q_eta.json")
        if os.path.exists(calib_file):
            try:
                import json
                with open(calib_file, "r", encoding="utf-8") as f:
                    cdata = json.load(f)
                    self.default_q_eta_m = float(cdata.get("recommended_q_eta_m", 0.00245))
                    if "q_eta_t2_total_mm" in cdata:
                        self.default_q_eta_t2_m = float(cdata["q_eta_t2_total_mm"]) / 1000.0
                    else:
                        self.default_q_eta_t2_m = self.default_q_eta_m
            except Exception:
                self.default_q_eta_m = 0.00245
                self.default_q_eta_t2_m = self.default_q_eta_m
        else:
            self.default_q_eta_m = 0.00245
            self.default_q_eta_t2_m = self.default_q_eta_m

        # Tiền tính toán ma trận truyền ZOH chính xác cho 2 trục sóng sánh và hệ số Stokes bậc 2
        self.Phi_sx, self.Gam_sx = self._compute_slosh_zoh_mats(omega_x, +1.0, self.dt)
        self.Phi_sy, self.Gam_sy = self._compute_slosh_zoh_mats(omega_y, -1.0, self.dt)
        self._stokes_ax = float((np.pi * TANK_A / 8.0) / np.tanh(np.pi * H_LIQUID / TANK_A))
        self._stokes_ay = float((np.pi * TANK_B / 8.0) / np.tanh(np.pi * H_LIQUID / TANK_B))
        self._stokes_ax_q = float(self._stokes_ax + 0.08 * TANK_A)
        self._stokes_ay_q = float(self._stokes_ay + 0.08 * TANK_B)
        
        self.nx = 12  # [q (3), v (3), a (3), s (1), vs (1), as (1)]
        self.nu = 4   # [j_amr (3), js (1)]
        self.tube_planner = None  # Mô-đun Ống vận tốc khả thi bậc 3 Krasnoselskii-Mann (None nếu chạy chế độ thường)
        
        # Tải mô hình CasADi SX cho AMR
        self.funcs = build_amr_symbolic_model()
        
        # 1. Ma trận động học LTI rời rạc hóa chính xác ZOH
        I3 = np.eye(3)
        Z3 = np.zeros((3, 3))
        dt_val = self.dt
        
        A_rob = np.block([
            [I3, dt_val * I3, 0.5 * (dt_val**2) * I3],
            [Z3, I3,          dt_val * I3          ],
            [Z3, Z3,          I3                   ]
        ])
        B_rob = np.block([
            [(dt_val**3 / 6.0) * I3],
            [0.5 * (dt_val**2) * I3],
            [dt_val * I3           ]
        ])
        
        A_s = np.array([
            [1.0, dt_val, 0.5 * dt_val**2],
            [0.0, 1.0,    dt_val         ],
            [0.0, 0.0,    1.0            ]
        ])
        B_s = np.array([
            [dt_val**3 / 6.0],
            [0.5 * dt_val**2],
            [dt_val         ]
        ])
        
        self.Ad = np.block([
            [A_rob, np.zeros((9, 3))],
            [np.zeros((3, 9)), A_s  ]
        ])
        self.Bd = np.block([
            [B_rob, np.zeros((9, 1))],
            [np.zeros((3, 3)), B_s  ]
        ])
        self.Ad_dm = cs.DM(self.Ad)
        self.Bd_dm = cs.DM(self.Bd)
        
        # 2. Trọng số hàm mục tiêu MPCC
        self.w_contour = 60.0        # Phạt lệch tâm đường (Contouring Error)
        self.w_lag = 50.0            # Phạt trễ dọc đường (Lag Error)
        self.w_progress = 6.0        # Thưởng vận tốc tiến độ vs
        self.w_heading = 30.0        # Bám góc tiếp tuyến đường đi
        self.w_u = 0.02              # Phạt giật Jerk
        self.w_v = 0.02              # Giảm vận tốc dư thừa
        self.w_a = 0.02              # Giảm gia tốc dư thừa
        self.w_slack_slosh = 2000.0  # Biến bù phạt Slosh ECBF
        self.w_slack_obs = 10000.0   # Biến bù phạt Obstacle ECBF
        self.w_slack_trac = 6000.0   # Biến bù phạt Elip ma sát động (Traction Slack)
        
        # Bộ đệm warm-start (primal + slack + dual lbx/g)
        self.x_traj_prev = np.zeros((self.nx, self.N + 1))
        self.u_traj_prev = np.zeros((self.nu, self.N))
        self._phi_traj_prev = np.zeros((4, self.N + 1), dtype=float)
        self._s_slosh_prev = np.zeros((4, self.N))
        self._s_obs_prev = np.zeros((1, self.N))
        self._s_zono_prev = np.zeros((12, self.N))
        self._s_tube_prev = np.zeros((1, self.N))
        self._s_term_prev = np.zeros((1, 1))
        self.x_opt_prev = None
        self.lam_x_prev = None
        self.lam_g_prev = None
        self._last_iter_count = 0
        self._nlp_solver_init = None
        self._nlp_solver_warm = None
        self._ng = 0
        self._nvars = 0

        self.spline_x = None
        self.spline_y = None
        self.spline_x_d1 = None
        self.spline_y_d1 = None
        self.path_length = 0.0
        self.is_path_ready = False

        # Bảng ánh xạ đồng bộ chiều dài cung giữa MPC (s_mpc) và Ống vận tốc (s_tube)
        self._s_mpc_map = None
        self._s_tube_map = None
        self.s_mapping_max_drift_m = 0.0

        # Xây dựng bài toán tối ưu
        self._build_solver()

    @staticmethod
    def _compute_slosh_zoh_mats(omega_val: float, sigma_val: float, dt_val: float):
        """Tính ma trận ZOH 2x2 chính xác cho dao động con lắc có cản (zeta = ZETA_SLOSH)."""
        sq = float(np.sqrt(max(1e-9, 1.0 - ZETA_SLOSH ** 2)))
        wd = float(omega_val * sq)
        ez = float(np.exp(-ZETA_SLOSH * omega_val * dt_val))
        cw = float(np.cos(wd * dt_val))
        sw = float(np.sin(wd * dt_val))
        z_sq = ZETA_SLOSH / sq
        phi_m = ez * np.array([
            [cw + z_sq * sw, sw / wd],
            [-(omega_val ** 2 / wd) * sw, cw - z_sq * sw],
        ], dtype=float)
        gam_v = (float(sigma_val) / g) * np.array([
            1.0 - ez * (cw + z_sq * sw),
            (omega_val ** 2 / wd) * ez * sw,
        ], dtype=float)
        return phi_m, gam_v

    def propagate_slosh_zoh(self, phi_est, a_body_seq):
        """Truyền trạng thái sóng [phi_x, dphi_x, phi_y, dphi_y] qua chuỗi gia tốc thân xe hằng từng bước."""
        st = np.asarray(phi_est, dtype=float).ravel()[:4].copy()
        sx = st[0:2]
        sy = st[2:4]
        for ab in np.asarray(a_body_seq, dtype=float):
            sx = self.Phi_sx @ sx + self.Gam_sx * float(ab[0])
            sy = self.Phi_sy @ sy + self.Gam_sy * float(ab[1])
        return np.array([sx[0], sx[1], sy[0], sy[1]], dtype=float)

    def reset_warm_start(self):
        """Đặt lại toàn bộ bộ đệm warm-start primal/slack/dual cho IPOPT."""
        self.x_traj_prev.fill(0.0)
        self.u_traj_prev.fill(0.0)
        self._phi_traj_prev.fill(0.0)
        self._s_slosh_prev.fill(0.0)
        self._s_obs_prev.fill(0.0)
        self._s_zono_prev.fill(0.0)
        self._s_tube_prev.fill(0.0)
        self._s_term_prev.fill(0.0)
        self.x_opt_prev = None
        self.lam_x_prev = None
        self.lam_g_prev = None
        self._last_iter_count = 0

    def _build_s_tube_mapping(self):
        """
        Khớp hình học điểm gần nhất (x, y) giữa tham số dây cung spline của MPC (s_mpc)
        và tham số độ dài cung mịn của Ống vận tốc (s_tube) bằng phép chiếu đơn điệu
        trong cửa sổ tiến nhằm tránh nhảy nhánh trên đường cong gấp / hành lang tự tiệm cận.
        """
        if not (
            self.is_path_ready
            and self.tube_planner is not None
            and getattr(self.tube_planner, "is_ready", False)
            and getattr(self.tube_planner, "x_grid", None) is not None
            and getattr(self.tube_planner, "s_grid", None) is not None
            and len(self.tube_planner.s_grid) >= 2
        ):
            self._s_mpc_map = None
            self._s_tube_map = None
            self.s_mapping_max_drift_m = 0.0
            return

        m_pts = max(int(len(self.tube_planner.s_grid)), 400)
        s_m = np.linspace(0.0, self.path_length, m_pts)
        pts_mpc = np.column_stack([self.spline_x(s_m), self.spline_y(s_m)])
        pts_tube = np.column_stack([self.tube_planner.x_grid, self.tube_planner.y_grid])
        s_tube_grid = np.asarray(self.tube_planner.s_grid, dtype=float)
        n_tube = len(s_tube_grid)
        ds_m = self.path_length / max(1, m_pts - 1)
        win_lookahead = max(0.45, 5.0 * ds_m)

        nearest_idx = np.empty(m_pts, dtype=int)
        cur_idx = 0
        for i in range(m_pts):
            s_cap = s_tube_grid[0] + max(0.50, 0.10 * float(s_tube_grid[-1])) if i == 0 else (s_tube_grid[cur_idx] + win_lookahead)
            j_hi = int(np.searchsorted(s_tube_grid, s_cap, side="right"))
            j_hi = max(cur_idx + 1, min(n_tube, j_hi))
            diff_w = pts_tube[cur_idx:j_hi] - pts_mpc[i]
            d2_w = diff_w[:, 0] ** 2 + diff_w[:, 1] ** 2
            cur_idx = cur_idx + int(np.argmin(d2_w))
            nearest_idx[i] = cur_idx

        s_t = np.asarray(s_tube_grid[nearest_idx], dtype=float).copy()
        s_t[0] = 0.0
        s_t[-1] = float(s_tube_grid[-1])
        s_t = np.maximum.accumulate(s_t)

        self._s_mpc_map = s_m
        self._s_tube_map = s_t
        self.s_mapping_max_drift_m = float(np.max(np.abs(s_t - s_m)))

    def map_s_mpc_to_tube(self, s_mpc):
        """Quy đổi tọa độ cung s của MPC sang s của Ống vận tốc H124."""
        is_scalar = np.isscalar(s_mpc)
        s_arr = np.asarray(s_mpc, dtype=float)
        if self._s_mpc_map is not None and self._s_tube_map is not None:
            mapped = np.interp(s_arr, self._s_mpc_map, self._s_tube_map)
        elif self.tube_planner is not None and getattr(self.tube_planner, "is_ready", False):
            mapped = np.clip(s_arr, 0.0, float(self.tube_planner.s_grid[-1]))
        else:
            mapped = s_arr
        return float(mapped) if is_scalar else mapped

    def set_path(self, waypoints):
        """Nạp quỹ đạo toàn cục và khớp Cubic Spline trơn C^2."""
        waypoints = np.asarray(waypoints)
        diffs = np.diff(waypoints, axis=0)
        dists = np.hypot(diffs[:, 0], diffs[:, 1])
        s_nodes = np.concatenate(([0.0], np.cumsum(dists)))
        self.path_length = float(s_nodes[-1])
        self.spline_x = CubicSpline(s_nodes, waypoints[:, 0])
        self.spline_y = CubicSpline(s_nodes, waypoints[:, 1])
        self.spline_x_d1 = self.spline_x.derivative()
        self.spline_y_d1 = self.spline_y.derivative()
        self.is_path_ready = True
        self.reset_warm_start()
        self._build_s_tube_mapping()
        print(f"[DFE-MPCC] Đã nạp thành công quỹ đạo tham chiếu: Chiều dài L = {self.path_length:.3f} m ({len(waypoints)} điểm).")

    def set_tube_planner(self, tube_planner):
        """Liên kết mô-đun Ống vận tốc khả thi bậc 3 Krasnoselskii-Mann (H124) vào bộ giải tối ưu NLP."""
        self.tube_planner = tube_planner
        self._build_s_tube_mapping()
        n_pts = len(tube_planner.s_grid) if getattr(tube_planner, "is_ready", False) else 0
        print(f"[DFE-MPCC] Đã tích hợp Ống vận tốc khả thi bậc 3 Krasnoselskii-Mann vào Solver ({n_pts} điểm lưới).")

    def get_path_reference(self, s_val):
        """Lấy vị trí [xr, yr], tiếp tuyến [tx, ty] và góc hướng theta_path tại độ dài cung s."""
        if not self.is_path_ready:
            return np.zeros(2), np.array([1.0, 0.0]), 0.0
        s_clamped = np.clip(s_val, 0.0, self.path_length)
        xr = float(self.spline_x(s_clamped))
        yr = float(self.spline_y(s_clamped))
        dxr = float(self.spline_x_d1(s_clamped))
        dyr = float(self.spline_y_d1(s_clamped))
        norm_d = np.hypot(dxr, dyr) + 1e-9
        tx = dxr / norm_d
        ty = dyr / norm_d
        th_path = np.arctan2(dyr, dxr)
        return np.array([xr, yr]), np.array([tx, ty]), th_path

    def _build_solver(self):
        """Xây dựng bài toán tối ưu DFE-MPCC qua CasADi SX + cs.nlpsol với cận biến lbx/ubx trực tiếp."""
        N = self.N
        nx = self.nx
        nu = self.nu
        k_start = 1 if self.state_con_mode == "skip0" else 0

        # Biến quyết định SX
        X = cs.SX.sym('X', nx, N + 1)
        U = cs.SX.sym('U', nu, N)
        Phi_s = cs.SX.sym('Phi_s', 4, N + 1)
        s_slosh = cs.SX.sym('s_slosh', 4, N)
        s_obs = cs.SX.sym('s_obs', 1, N)
        s_zono = cs.SX.sym('s_zono', 12, N)    # Biến bù cho 12 mặt biên Zonotope T1 (Exact L1 slack)
        s_tube = cs.SX.sym('s_tube', 1, N)     # Biến bù trần Ống vận tốc khả thi (Exact L1 slack)
        s_term_slosh = cs.SX.sym('s_term_slosh', 1, 1)

        # Tham số đầu vào cập nhật trực tuyến
        P_x0 = cs.SX.sym('P_x0', nx)
        P_path_p = cs.SX.sym('P_path_p', 2, N)
        P_path_t = cs.SX.sym('P_path_t', 2, N)
        P_path_sbar = cs.SX.sym('P_path_sbar', 1, N)
        P_th_path = cs.SX.sym('P_th_path', 1, N)
        P_phi_est = cs.SX.sym('P_phi_est', 4)
        # [ox, oy, oz, r_obs, d_safe, vox, voy]; vox = voy = 0 -> vật cản tĩnh (giống bản cũ)
        P_obs = cs.SX.sym('P_obs', 7)
        P_mu = cs.SX.sym('P_mu', 1, N)       # Hệ số ma sát hoạch định dọc chân trời dự báo
        P_q_eta = cs.SX.sym('P_q_eta', 1)    # Nguồn lượng tử Conformal an toàn q_eta (m) cho ràng buộc T2
        P_dfz_iso = cs.SX.sym('P_dfz_iso', 4) # Biến thiên tải pháp tuyến 4 bánh do mặt đường ISO 8608 (N)
        P_v_ub = cs.SX.sym('P_v_ub', 1, N)   # Trần Ống vận tốc khả thi bậc 3 (m/s) dọc chân trời dự báo
        P_v_lb = cs.SX.sym('P_v_lb', 1, N)   # Sàn Ống vận tốc khả thi bậc 3 (m/s)

        # Cận hộp trực tiếp trên biến quyết định (giảm kích thước ma trận KKT của IPOPT)
        lbX = np.full((nx, N + 1), -np.inf, dtype=float)
        ubX = np.full((nx, N + 1), np.inf, dtype=float)
        for k in range(k_start, N):
            lbX[0, k], ubX[0, k] = self.x_bounds[0], self.x_bounds[1]
            lbX[1, k], ubX[1, k] = self.y_bounds[0], self.y_bounds[1]
            lbX[9, k], ubX[9, k] = 0.0, 30.0
            lbX[5, k], ubX[5, k] = -1.2, 1.2
            lbX[6:8, k], ubX[6:8, k] = -1.2, 1.2
            lbX[8, k], ubX[8, k] = -2.5, 2.5
            lbX[11, k], ubX[11, k] = -1.2, 1.2

        lbU = np.zeros((nu, N), dtype=float)
        ubU = np.zeros((nu, N), dtype=float)
        lbU[0:2, :], ubU[0:2, :] = -4.0, 4.0
        lbU[2, :], ubU[2, :] = -6.0, 6.0
        lbU[3, :], ubU[3, :] = -8.0, 8.0

        # Đưa bù P_q_eta cho ECBF ra ngoài vòng lặp (P_q_eta hằng dọc chân trời)
        eta_ecbf_lim_x = 0.5 * TANK_A * PHI_MAX_X
        eta_ecbf_lim_y = 0.5 * TANK_B * PHI_MAX_Y
        excess_q_x = cs.fmax(0.0, eta_ecbf_lim_x + P_q_eta - self.h_lim_m)
        excess_q_y = cs.fmax(0.0, eta_ecbf_lim_y + P_q_eta - self.h_lim_m)
        dtheta_margin_x = cs.atan((2.0 * excess_q_x) / TANK_A)
        dtheta_margin_y = cs.atan((2.0 * excess_q_y) / TANK_B)
        q_eta_b3_margin = cs.vertcat(
            K_SLOSH_0_X * dtheta_margin_x,
            K_SLOSH_0_X * dtheta_margin_x,
            K_SLOSH_0_Y * dtheta_margin_y,
            K_SLOSH_0_Y * dtheta_margin_y,
        )

        g_list, lbg_list, ubg_list = [], [], []
        # Điều kiện đầu X[:, 0] == P_x0 và Phi_s[:, 0] == P_phi_est
        g_list.append(X[:, 0] - P_x0)
        lbg_list.append(np.zeros(nx, dtype=float))
        ubg_list.append(np.zeros(nx, dtype=float))
        g_list.append(Phi_s[:, 0] - P_phi_est)
        lbg_list.append(np.zeros(4, dtype=float))
        ubg_list.append(np.zeros(4, dtype=float))

        cost = 0.0
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        phi_x_k = P_phi_est[0]
        dphi_x_k = P_phi_est[1]
        phi_y_k = P_phi_est[2]
        dphi_y_k = P_phi_est[3]
        cos_phix = cs.cos(phi_x_k)
        cos_phiy = cs.cos(phi_y_k)
        M_grav_pend_x = m_1x * g * lx * cs.sin(phi_x_k)
        M_grav_pend_y = -m_1y * g * ly * cs.sin(phi_y_k)

        for k in range(N):
            # Động học LTI chính xác ZOH
            x_k = X[:, k]
            u_k = U[:, k]
            x_next = X[:, k + 1]
            g_list.append(x_next - (self.Ad_dm @ x_k + self.Bd_dm @ u_k))
            lbg_list.append(np.zeros(nx, dtype=float))
            ubg_list.append(np.zeros(nx, dtype=float))

            q_k = x_k[0:3]
            v_k = x_k[3:6]
            a_k = x_k[6:9]
            s_k = x_k[9]
            vs_k = x_k[10]
            u_amr = u_k[0:3]

            th_k = q_k[2]
            c_th = cs.cos(th_k)
            s_th = cs.sin(th_k)
            a_bx = c_th * a_k[0] + s_th * a_k[1]
            a_by = -s_th * a_k[0] + c_th * a_k[1]

            # Cải tiến U1a (skip0): Loại bỏ các ràng buộc chỉ phụ thuộc trạng thái tại k=0
            if k >= 1 or self.state_con_mode != "skip0":
                # 1. RÀNG BUỘC ỐNG VẬN TỐC KHẢ THI BẬC 3 (PT 18 Mục 2.6)
                v_ub_k = P_v_ub[0, k]
                v_lb_k = P_v_lb[0, k]
                g_list.append(vs_k - v_lb_k + s_tube[0, k])
                lbg_list.append(np.array([0.0]))
                ubg_list.append(np.array([np.inf]))
                g_list.append(vs_k - v_ub_k - s_tube[0, k])
                lbg_list.append(np.array([-np.inf]))
                ubg_list.append(np.array([0.0]))
                g_list.append(v_k[0]**2 + v_k[1]**2 - (v_ub_k + s_tube[0, k])**2)
                lbg_list.append(np.array([-np.inf]))
                ubg_list.append(np.array([0.0]))

                # 2. Ràng buộc vận tốc bánh xe Mecanum phi tuyến
                w_wheels = self.funcs['mecanum_wheels'](q_k, v_k)
                g_list.append(w_wheels)
                lbg_list.append(np.full(4, -OMEGA_MAX))
                ubg_list.append(np.full(4, OMEGA_MAX))

                # 3. RÀNG BUỘC ĐỘNG LỰC HỌC TẢI PHÁP TUYẾN & ELIP MA SÁT ĐỘNG (DFE) + ISO 8608
                # Lực phản kháng dao động sóng sánh thuần túy FSI (Convective Slosh Residual)
                # Khớp quy ước dấu robot_model_amr_casadi.py & mecanum_traction_dynamics.py
                ddphi_x_k = -omega_x**2 * phi_x_k - hd_x * dphi_x_k + (1.0 / lx) * a_bx
                ddphi_y_k = -omega_y**2 * phi_y_k - hd_y * dphi_y_k - (1.0 / ly) * a_by
                a_rel_pend_x = -lx * ddphi_x_k * cos_phix
                a_rel_pend_y =  ly * ddphi_y_k * cos_phiy
                F_slosh_bx = -m_1x * a_rel_pend_x
                F_slosh_by = -m_1y * a_rel_pend_y

                # Mô-men lật tổng hợp quanh điểm tiếp xúc sàn (gồm mô-men trọng lực con lắc lệch góc)
                M_pitch = M_TOTAL * a_bx * Z_COM - F_slosh_bx * Z_SLOSH_X + M_grav_pend_x
                M_roll  = M_TOTAL * a_by * Z_COM - F_slosh_by * Z_SLOSH_Y + M_grav_pend_y
                dFz_pitch = M_pitch / (4.0 * L_X)
                dFz_roll  = M_roll  / (4.0 * L_Y)

                Fz_1 = FZ_STATIC - dFz_pitch - dFz_roll + P_dfz_iso[0]  # FL
                Fz_2 = FZ_STATIC - dFz_pitch + dFz_roll + P_dfz_iso[1]  # FR
                Fz_3 = FZ_STATIC + dFz_pitch + dFz_roll + P_dfz_iso[2]  # RR
                Fz_4 = FZ_STATIC + dFz_pitch - dFz_roll + P_dfz_iso[3]  # RL
                Fz_vec = cs.vertcat(Fz_1, Fz_2, Fz_3, Fz_4)
                g_list.append(Fz_vec)
                lbg_list.append(np.full(4, FZ_MIN))
                ubg_list.append(np.full(4, np.inf))

                # Lực bệ xe yêu cầu & Zonotope T1 (12 mặt biên)
                W_cmd = cs.vertcat(M_TOTAL * a_bx - F_slosh_bx, M_TOTAL * a_by - F_slosh_by, IZ_BASE * a_k[2])
                f_bar_vec = (P_mu[0, k] * Fz_vec) * inv_sqrt2
                d_zono = self.M_reach_dm @ f_bar_vec
                g_list.append(self.C_zono_dm @ W_cmd - d_zono - s_zono[:, k])
                lbg_list.append(np.full(12, -np.inf))
                ubg_list.append(np.zeros(12))

            # 4. Sai số hình học MPCC
            p_bar = P_path_p[:, k]
            t_bar = P_path_t[:, k]
            s_bar = P_path_sbar[0, k]
            n_bar = cs.vertcat(-t_bar[1], t_bar[0])
            p_rob = cs.vertcat(q_k[0], q_k[1])
            e_k = p_rob - (p_bar + t_bar * (s_k - s_bar))
            eps_c = n_bar[0] * e_k[0] + n_bar[1] * e_k[1]
            eps_l = t_bar[0] * e_k[0] + t_bar[1] * e_k[1]

            # 5. Slosh-ECBF bậc 3 (Soft constraint)
            b3_slosh = self.funcs['slosh_cbf'](q_k, v_k, a_k, u_amr, P_phi_est)
            g_list.append(b3_slosh - q_eta_b3_margin + s_slosh[:, k])
            lbg_list.append(np.zeros(4))
            ubg_list.append(np.full(4, np.inf))

            # 6. Obstacle-ECBF bậc 3 (Soft constraint)
            t_k = k * self.dt
            obs_xy_k = P_obs[0:2] + P_obs[5:7] * t_k
            P_obs_k = cs.vertcat(obs_xy_k, P_obs[2:5])
            v_rel_k = cs.vertcat(v_k[0] - P_obs[5], v_k[1] - P_obs[6], v_k[2])
            b3_obs = self.funcs['obs_cbf'](q_k, v_rel_k, a_k, u_amr, P_obs_k)
            g_list.append(b3_obs + s_obs[0, k])
            lbg_list.append(np.array([0.0]))
            ubg_list.append(np.array([np.inf]))

            # Tích lũy hàm mục tiêu có thứ bậc chính xác (Exact Hierarchical Penalty)
            cost += self.w_contour * (eps_c ** 2)
            cost += self.w_lag * (eps_l ** 2)
            cost -= self.w_progress * vs_k
            cost += self.w_heading * ((q_k[2] - P_th_path[0, k]) ** 2)
            cost += self.w_u * cs.sumsqr(u_k)
            cost += self.w_v * cs.sumsqr(v_k)
            cost += self.w_a * cs.sumsqr(a_k)
            cost += self.w_slack_slosh * cs.sum1(s_slosh[:, k])
            cost += self.w_slack_obs * s_obs[0, k]
            cost += self.w_slack_trac * cs.sum1(s_zono[:, k])
            cost += 2000.0 * s_tube[0, k] + 10000.0 * (s_tube[0, k] ** 2)

            # Ràng buộc truyền trạng thái sóng ZOH thưa theo khối cho mọi k = 0..N-1
            phi_x_next = self.Phi_sx[0, 0] * Phi_s[0, k] + self.Phi_sx[0, 1] * Phi_s[1, k] + self.Gam_sx[0] * a_bx
            dphi_x_next = self.Phi_sx[1, 0] * Phi_s[0, k] + self.Phi_sx[1, 1] * Phi_s[1, k] + self.Gam_sx[1] * a_bx
            phi_y_next = self.Phi_sy[0, 0] * Phi_s[2, k] + self.Phi_sy[0, 1] * Phi_s[3, k] + self.Gam_sy[0] * a_by
            dphi_y_next = self.Phi_sy[1, 0] * Phi_s[2, k] + self.Phi_sy[1, 1] * Phi_s[3, k] + self.Gam_sy[1] * a_by
            g_list.append(Phi_s[:, k + 1] - cs.vertcat(phi_x_next, dphi_x_next, phi_y_next, dphi_y_next))
            lbg_list.append(np.zeros(4, dtype=float))
            ubg_list.append(np.zeros(4, dtype=float))

        # Chi phí điểm cuối (Terminal Cost)
        cost += 40.0 * (eps_c ** 2) + 40.0 * (eps_l ** 2) + 20.0 * ((X[2, N] - P_th_path[0, N - 1]) ** 2) - 6.0 * X[10, N]

        # 7. TẬP AN TOÀN CUỐI BẤT BIẾN PHANH DỪNG T2 TẠI k = N (LIFTED SPARSE ZOH + STOKES BẬC 2, 2 TRỤC X & Y)
        phi_x_N = Phi_s[0, N]
        dphi_x_N = Phi_s[1, N]
        phi_y_N = Phi_s[2, N]
        dphi_y_N = Phi_s[3, N]
        c_thN = cs.cos(X[2, N])
        s_thN = cs.sin(X[2, N])
        v_bx_N = c_thN * X[3, N] + s_thN * X[4, N]
        v_by_N = -s_thN * X[3, N] + c_thN * X[4, N]
        v_N_norm = cs.sqrt(v_bx_N**2 + v_by_N**2 + 0.08**2)
        a_brk_x_N = -self.a_brake_max * (v_bx_N / v_N_norm)
        a_brk_y_N = -self.a_brake_max * (v_by_N / v_N_norm)
        phi_eq_x_N = a_brk_x_N / g
        phi_eq_y_N = -a_brk_y_N / g
        phi_peak_brake_x = cs.sqrt(phi_eq_x_N**2 + 1e-6) + cs.sqrt((phi_x_N - phi_eq_x_N)**2 + (dphi_x_N / omega_x)**2 + 1e-6)
        phi_peak_brake_y = cs.sqrt(phi_eq_y_N**2 + 1e-6) + cs.sqrt((phi_y_N - phi_eq_y_N)**2 + (dphi_y_N / omega_y)**2 + 1e-6)
        eta_peak_x_m = 0.5 * TANK_A * phi_peak_brake_x + self._stokes_ax_q * (phi_peak_brake_x ** 2)
        eta_peak_y_m = 0.5 * TANK_B * phi_peak_brake_y + self._stokes_ay_q * (phi_peak_brake_y ** 2)
        h_free_m = self.h_lim_m
        g_list.append(eta_peak_x_m + 0.35 * eta_peak_y_m + P_q_eta - h_free_m - s_term_slosh[0, 0])
        lbg_list.append(np.array([-np.inf]))
        ubg_list.append(np.array([0.0]))
        g_list.append(eta_peak_y_m + 0.35 * eta_peak_x_m + P_q_eta - h_free_m - s_term_slosh[0, 0])
        lbg_list.append(np.array([-np.inf]))
        ubg_list.append(np.array([0.0]))
        cost += 8000.0 * s_term_slosh[0, 0] + 40000.0 * (s_term_slosh[0, 0] ** 2)

        w_vec = cs.vertcat(
            cs.vec(X), cs.vec(U), cs.vec(Phi_s),
            cs.vec(s_slosh), cs.vec(s_obs), cs.vec(s_zono), cs.vec(s_tube), s_term_slosh
        )
        p_vec = cs.vertcat(
            P_x0, cs.vec(P_path_p), cs.vec(P_path_t), cs.vec(P_path_sbar), cs.vec(P_th_path),
            P_phi_est, P_obs, cs.vec(P_mu), P_q_eta, P_dfz_iso, cs.vec(P_v_ub), cs.vec(P_v_lb)
        )
        g_vec = cs.vertcat(*g_list)
        self._lbg = np.concatenate(lbg_list)
        self._ubg = np.concatenate(ubg_list)
        self._n_slacks = 4 * N + N + 12 * N + N + 1
        lbPhi = np.full((4, N + 1), -np.inf, dtype=float)
        ubPhi = np.full((4, N + 1), np.inf, dtype=float)
        self._lbx = np.concatenate([
            lbX.ravel(order='F'), lbU.ravel(order='F'), lbPhi.ravel(order='F'),
            np.zeros(self._n_slacks, dtype=float),
        ])
        self._ubx = np.concatenate([
            ubX.ravel(order='F'), ubU.ravel(order='F'), ubPhi.ravel(order='F'),
            np.full(self._n_slacks, np.inf, dtype=float),
        ])
        self._nvars = int(w_vec.shape[0])
        self._ng = int(g_vec.shape[0])
        self._nX = nx * (N + 1)
        self._nU = nu * N
        self._nPhi = 4 * (N + 1)

        nlp_dict = {'x': w_vec, 'p': p_vec, 'f': cost, 'g': g_vec}
        ipopt_warm = {
            'max_iter': self.ipopt_max_iter,
            'print_level': 0,
            'sb': 'yes',
            'tol': 1e-2,
            'constr_viol_tol': 1e-4,
            'compl_inf_tol': 5e-4,
            'acceptable_tol': 5e-2,
            'acceptable_iter': 2,
            'mu_strategy': 'adaptive',
            'warm_start_init_point': 'yes',
            'warm_start_bound_push': 5e-6,
            'warm_start_slack_bound_push': 5e-6,
            'warm_start_mult_bound_push': 5e-6,
        }
        ipopt_init = dict(ipopt_warm)
        ipopt_init['warm_start_init_point'] = 'no'
        ipopt_init['max_iter'] = max(self.ipopt_max_iter, 25)
        self._nlp_solver_warm = cs.nlpsol(
            'dfe_mpcc_warm', 'ipopt', nlp_dict, {'print_time': False, 'error_on_fail': False, 'ipopt': ipopt_warm}
        )
        self._nlp_solver_init = cs.nlpsol(
            'dfe_mpcc_init', 'ipopt', nlp_dict, {'print_time': False, 'error_on_fail': False, 'ipopt': ipopt_init}
        )

    def solve_step(self, x_current, phi_est, obs_info=None, mu_floor=None, q_eta=None, dfz_iso=None, v_ub_horizon=None, v_lb_horizon=None, mu_horizon=None):
        """
        Thực thi giải một chu kỳ DFE-MPCC (20 Hz) nhúng Ống vận tốc khả thi bậc 3 (H124)
        và biên mặt đường ngẫu nhiên ISO 8608.
        """
        t_start = time.perf_counter()
        if obs_info is None:
            obs_info = np.array([10.0, 10.0, 0.5, 0.2, 0.1])
        obs_info = np.asarray(obs_info, dtype=float).ravel()
        if obs_info.size < 7:
            obs_info = np.concatenate([obs_info, np.zeros(7 - obs_info.size)])
        q_eta_apply = float(q_eta if q_eta is not None else self.default_q_eta_t2_m)
        if dfz_iso is None:
            dfz_iso_vec = np.zeros(4)
        else:
            dfz_iso_vec = np.asarray(dfz_iso, dtype=float).reshape(4)

        # 1. Trích xuất tham số tham chiếu dọc chuỗi s dự báo (vector hóa)
        s_cur = float(np.clip(x_current[9], 0.0, self.path_length))
        vs_cur = float(np.clip(x_current[10], 0.05, 0.75))
        vs_preview = max(vs_cur, 0.40)

        s_vec = np.clip(s_cur + np.arange(self.N, dtype=float) * self.dt * vs_preview, 0.0, self.path_length)
        path_sbar = s_vec.reshape(1, self.N)
        prev_h = float(x_current[2])
        th_path = np.zeros((1, self.N))
        if self.is_path_ready and self.spline_x_d1 is not None and self.spline_y_d1 is not None:
            px = self.spline_x(s_vec)
            py = self.spline_y(s_vec)
            dx = self.spline_x_d1(s_vec)
            dy = self.spline_y_d1(s_vec)
            nrm = np.hypot(dx, dy) + 1e-9
            path_p = np.vstack([px, py])
            path_t = np.vstack([dx / nrm, dy / nrm])
            hd = np.arctan2(dy, dx)
            for k in range(self.N):
                prev_h = prev_h + float(np.arctan2(np.sin(hd[k] - prev_h), np.cos(hd[k] - prev_h)))
                th_path[0, k] = prev_h
        else:
            path_p = np.zeros((2, self.N))
            path_t = np.vstack([np.ones(self.N), np.zeros(self.N)])
            th_path.fill(prev_h)

        s_preview_tube = self.map_s_mpc_to_tube(path_sbar[0, :])

        # 2. Nạp tham số ma sát & ống vận tốc
        if mu_horizon is not None:
            mu_plan_vec = np.asarray(mu_horizon, dtype=float).reshape(1, self.N)
        elif mu_floor is not None:
            if np.ndim(mu_floor) > 0:
                mu_plan_vec = np.asarray(mu_floor, dtype=float).reshape(1, self.N) * self.mu_plan_scale
            else:
                mu_plan_vec = np.full((1, self.N), float(mu_floor) * self.mu_plan_scale)
        elif self.tube_planner is not None and getattr(self.tube_planner, "is_ready", False) and getattr(self.tube_planner, "mu_grid", None) is not None:
            mu_plan_vec = self.tube_planner.get_horizon_mu_plan(s_preview_tube).reshape(1, self.N)
        else:
            mu_plan_vec = np.full((1, self.N), float(self.default_mu) * self.mu_plan_scale)
        mu_plan_vec = np.maximum(0.05, mu_plan_vec)

        if v_ub_horizon is not None:
            v_ub_vec = np.asarray(v_ub_horizon, dtype=float).reshape(1, self.N)
            v_lb_vec = np.asarray(v_lb_horizon, dtype=float).reshape(1, self.N) if v_lb_horizon is not None else np.zeros((1, self.N))
        elif self.tube_planner is not None and getattr(self.tube_planner, "is_ready", False):
            v_ub_interp = np.interp(s_preview_tube, self.tube_planner.s_grid, self.tube_planner.v_ub_nom)
            v_ub_vec = np.clip(v_ub_interp, 0.15, 0.70).reshape(1, self.N)
            v_lb_vec = np.zeros((1, self.N))
        else:
            v_ub_vec = np.full((1, self.N), 0.60)
            v_lb_vec = np.zeros((1, self.N))

        # Kẹp sàn ống v_lb theo gia tốc khả thi trong nón ma sát từ vận tốc tiến độ hiện tại
        # để sàn tiến độ (liveness) không bao giờ ép bộ giải vi phạm Zonotope ma sát (safety).
        vs_now_lb = float(np.clip(x_current[10], 0.0, 0.75))
        a_lb_ramp = np.minimum(0.55, 0.45 * mu_plan_vec * g)
        dv_lb_cum = np.concatenate(
            [np.zeros((1, 1), dtype=float), np.cumsum(a_lb_ramp[:, :-1] * self.dt, axis=1)],
            axis=1,
        )
        v_lb_vec = np.minimum(v_lb_vec, vs_now_lb + dv_lb_cum)

        p_val = np.concatenate([
            np.asarray(x_current, dtype=float).ravel(),
            path_p.ravel(order='F'),
            path_t.ravel(order='F'),
            path_sbar.ravel(order='F'),
            th_path.ravel(order='F'),
            np.asarray(phi_est, dtype=float).ravel(),
            obs_info[:7],
            mu_plan_vec.ravel(order='F'),
            np.array([q_eta_apply], dtype=float),
            dfz_iso_vec,
            v_ub_vec.ravel(order='F'),
            v_lb_vec.ravel(order='F'),
        ])

        # 3. Khởi tạo lạnh rollout ở bước đầu, hoặc warm-start dịch chuyển khi đủ thông tin đối ngẫu
        phi_vec = np.asarray(phi_est, dtype=float).ravel()
        has_primal_prev = bool(np.any(self.x_traj_prev != 0))
        has_warm_dual = (
            has_primal_prev
            and (self.x_opt_prev is not None)
            and (self.lam_x_prev is not None)
            and (self.lam_g_prev is not None)
        )
        if not has_primal_prev:
            x_init = np.zeros((self.nx, self.N + 1), dtype=float)
            x_init[:, 0] = np.asarray(x_current, dtype=float).ravel()
            for k in range(self.N):
                x_init[:, k + 1] = self.Ad @ x_init[:, k]
            u_init = np.zeros((self.nu, self.N), dtype=float)
        else:
            x_init = np.zeros((self.nx, self.N + 1), dtype=float)
            x_init[:, :-1] = self.x_traj_prev[:, 1:]
            x_init[:, -1] = self.Ad @ x_init[:, -2]
            u_init = np.zeros((self.nu, self.N), dtype=float)
            u_init[:, :-1] = self.u_traj_prev[:, 1:]

        phi_init = np.zeros((4, self.N + 1), dtype=float)
        if has_primal_prev and np.any(self._phi_traj_prev != 0):
            phi_init[:, :-1] = self._phi_traj_prev[:, 1:]
            th_nm2 = float(x_init[2, -2])
            c_nm2, s_nm2 = np.cos(th_nm2), np.sin(th_nm2)
            abx_nm2 = c_nm2 * float(x_init[6, -2]) + s_nm2 * float(x_init[7, -2])
            aby_nm2 = -s_nm2 * float(x_init[6, -2]) + c_nm2 * float(x_init[7, -2])
            phi_init[0:2, -1] = self.Phi_sx @ phi_init[0:2, -2] + self.Gam_sx * abx_nm2
            phi_init[2:4, -1] = self.Phi_sy @ phi_init[2:4, -2] + self.Gam_sy * aby_nm2
        else:
            phi_init[:, 0] = phi_vec[:4]
            for k in range(self.N):
                phi_init[0:2, k + 1] = self.Phi_sx @ phi_init[0:2, k]
                phi_init[2:4, k + 1] = self.Phi_sy @ phi_init[2:4, k]

        if has_warm_dual:
            slack_init = np.concatenate([
                self._s_slosh_prev.ravel(order='F'),
                self._s_obs_prev.ravel(order='F'),
                self._s_zono_prev.ravel(order='F'),
                self._s_tube_prev.ravel(order='F'),
                self._s_term_prev.ravel(order='F'),
            ])
            w0 = np.concatenate([
                x_init.ravel(order='F'), u_init.ravel(order='F'), phi_init.ravel(order='F'), slack_init
            ])
            res = self._nlp_solver_warm(
                x0=w0, p=p_val, lbx=self._lbx, ubx=self._ubx, lbg=self._lbg, ubg=self._ubg,
                lam_x0=self.lam_x_prev, lam_g0=self.lam_g_prev,
            )
            stats = self._nlp_solver_warm.stats()
        else:
            slack_init = np.full(self._n_slacks, 1e-3, dtype=float)
            w0 = np.concatenate([
                x_init.ravel(order='F'), u_init.ravel(order='F'), phi_init.ravel(order='F'), slack_init
            ])
            res = self._nlp_solver_init(
                x0=w0, p=p_val, lbx=self._lbx, ubx=self._ubx, lbg=self._lbg, ubg=self._ubg
            )
            stats = self._nlp_solver_init.stats()

        ret_status = str(stats.get('return_status', ''))
        is_ok = bool(stats.get('success', False)) or (ret_status == 'Solved_To_Acceptable_Level')
        iter_count = int(stats.get('iter_count', -1))

        w_sol = np.asarray(res['x'], dtype=float).ravel()
        nX, nU, nPhi, N = self._nX, self._nU, self._nPhi, self.N
        o0 = nX + nU + nPhi
        o1 = o0 + 4 * N
        o2 = o1 + N
        o3 = o2 + 12 * N
        o4 = o3 + N

        x_pred = w_sol[:nX].reshape(self.nx, N + 1, order='F')
        u_pred = w_sol[nX:nX + nU].reshape(self.nu, N, order='F')
        phi_pred = w_sol[nX + nU:nX + nU + nPhi].reshape(4, N + 1, order='F')
        s_sl_np = w_sol[o0:o1].reshape(4, N, order='F')
        s_ob_np = w_sol[o1:o2].reshape(1, N, order='F')
        s_zo_np = w_sol[o2:o3].reshape(12, N, order='F')
        s_tu_np = w_sol[o3:o4].reshape(1, N, order='F')
        s_te_np = w_sol[o4:o4 + 1].reshape(1, 1)
        lam_x_out = np.asarray(res['lam_x'], dtype=float).ravel()
        lam_g_out = np.asarray(res['lam_g'], dtype=float).ravel()

        xu_finite = bool(np.all(np.isfinite(x_pred)) and np.all(np.isfinite(u_pred)))
        all_finite = bool(
            xu_finite
            and np.all(np.isfinite(phi_pred))
            and np.all(np.isfinite(s_sl_np))
            and np.all(np.isfinite(s_ob_np))
            and np.all(np.isfinite(s_zo_np))
            and np.all(np.isfinite(s_tu_np))
            and np.all(np.isfinite(s_te_np))
            and np.all(np.isfinite(lam_x_out))
            and np.all(np.isfinite(lam_g_out))
        )
        if not all_finite:
            is_ok = False
            if not np.all(np.isfinite(x_pred)):
                x_pred = np.zeros((self.nx, N + 1), dtype=float)
            if not np.all(np.isfinite(u_pred)):
                u_pred = np.zeros((self.nu, N), dtype=float)
            if not np.all(np.isfinite(phi_pred)):
                phi_pred = np.zeros((4, N + 1), dtype=float)
            s_sl_np = np.zeros((4, N), dtype=float)
            s_ob_np = np.zeros((1, N), dtype=float)
            s_zo_np = np.zeros((12, N), dtype=float)
            s_tu_np = np.zeros((1, N), dtype=float)
            s_te_np = np.zeros((1, 1), dtype=float)
            lam_x_out = np.zeros(self._nvars, dtype=float)
            lam_g_out = np.zeros(self._ng, dtype=float)

        u_opt = u_pred[0:3, 0].copy()
        js_opt = float(u_pred[3, 0])
        slack_zono_val = float(np.max(s_zo_np))
        slack_tube_val = float(np.max(s_tu_np))

        if is_ok:
            status = 'optimal'
            self._s_slosh_prev = s_sl_np
            self._s_obs_prev = s_ob_np
            self._s_zono_prev = s_zo_np
            self._s_tube_prev = s_tu_np
            self._s_term_prev = s_te_np
            self.lam_x_prev = lam_x_out
            self.lam_g_prev = lam_g_out
            self.x_opt_prev = x_pred
            self.x_traj_prev = x_pred.copy()
            self.u_traj_prev = u_pred.copy()
            self._phi_traj_prev = phi_pred.copy()
        else:
            status = 'suboptimal_fallback'
            u_opt[0:2] = np.clip(u_opt[0:2], -4.0, 4.0)
            u_opt[2] = float(np.clip(u_opt[2], -6.0, 6.0))
            js_opt = float(np.clip(js_opt, -8.0, 8.0))
            self.lam_x_prev = None
            self.lam_g_prev = None
            self.x_opt_prev = None
            self._s_slosh_prev.fill(0.0)
            self._s_obs_prev.fill(0.0)
            self._s_zono_prev.fill(0.0)
            self._s_tube_prev.fill(0.0)
            self._s_term_prev.fill(0.0)
            self._phi_traj_prev.fill(0.0)
            if xu_finite:
                self.x_traj_prev = x_pred.copy()
                self.u_traj_prev = u_pred.copy()

        self._last_iter_count = iter_count

        t_solve_ms = (time.perf_counter() - t_start) * 1000.0
        
        extra_info = {
            'solve_time_ms': t_solve_ms,
            'status': status,
            'iter_count': iter_count,
            'slack_tube': slack_tube_val,
            'v_ub_preview': v_ub_vec.flatten(),
            'mu_plan_preview': mu_plan_vec.flatten(),
            'slack_zono_max': slack_zono_val,
            'slack_term_slosh': float(s_te_np[0, 0]),
            'phi_N_pred': phi_pred[:, -1].copy(),
        }
        
        return u_opt, js_opt, x_pred, u_pred, extra_info


if __name__ == "__main__":
    print("=" * 70)
    print("KIỂM TRA TỰ ĐỘNG BỘ ĐIỀU KHIỂN DFE-MPCC TÍCH HỢP ZONOTOPE T1 (UNIT TEST)")
    print("=" * 70)
    ctrl = DFEMPCCAMRController(N=15, dt=0.05, mu=0.12)
    path = np.array([
        [0.0, 0.0],
        [0.5, 0.2],
        [1.0, 0.8],
        [1.5, 1.2],
        [2.0, 1.2]
    ])
    ctrl.set_path(path)
    
    x0 = np.zeros(12)
    x0[9] = 0.0   # s
    x0[10] = 0.2  # vs
    phi0 = np.array([0.02, 0.0, 0.05, 0.0]) # góc sóng nhỏ
    
    u_opt, js_opt, x_pred, u_pred, info = ctrl.solve_step(x0, phi0, mu_floor=0.12)
    print(f"Status: {info['status']}")
    print(f"Thời gian giải: {info['solve_time_ms']:.2f} ms")
    print(f"Lệnh điều khiển jerk bệ: u_opt = {np.round(u_opt, 3)}")
    print(f"Jerk tiến độ ảo: js_opt = {js_opt:.3f}")
    print(f"Mức vi phạm Zonotope T1 (Slack L1): {info['slack_zono_max']:.6f}")
    print("=" * 70)
