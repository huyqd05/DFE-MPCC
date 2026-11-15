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
    TANK_A, TANK_B,
    K_SLOSH_0_X, K_SLOSH_0_Y,
    OMEGA_MAX,
    L_X, L_Y, L_SUM, ROBOT_RADIUS,
    M_TOTAL, IZ_BASE, g,
    m_0x, m_1x, lx, omega_x,
    m_0y, m_1y, ly, omega_y
)
from zonotope_wrench_set import compute_zonotope_facet_matrices
from mecanum_traction_dynamics import Z_COM, Z_SLOSH, Z_SLOSH_X, Z_SLOSH_Y

FZ_STATIC = (M_TOTAL * g) / 4.0  # 17.78 N mỗi bánh danh định tĩnh
FZ_MIN = 3.5        # Ngưỡng tải trọng pháp tuyến tối thiểu chống nhấc bánh (N)


class DFEMPCCAMRController:
    def __init__(self, N=15, dt=0.05, mu=0.35, solver_type='ipopt',
                 state_con_mode="skip0", mu_plan_scale=0.95, ipopt_max_iter=35,
                 h_lim_m=0.030, x_bounds=(-2.75, 8.5), y_bounds=(-2.75, 2.75)):
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
        # Có thể đặt thấp hơn (vd 18 mm) để ràng buộc sóng thực sự kích hoạt.
        self.h_lim_m = float(h_lim_m)

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
            except Exception:
                self.default_q_eta_m = 0.00245
        else:
            self.default_q_eta_m = 0.00245
        
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
        
        # Bộ đệm warm-start
        self.x_traj_prev = np.zeros((self.nx, self.N + 1))
        self.u_traj_prev = np.zeros((self.nu, self.N))
        
        self.spline_x = None
        self.spline_y = None
        self.path_length = 0.0
        self.is_path_ready = False
        
        # Xây dựng bài toán tối ưu
        self._build_solver()

    def set_path(self, waypoints):
        """Nạp quỹ đạo toàn cục và khớp Cubic Spline trơn C^2."""
        waypoints = np.asarray(waypoints)
        diffs = np.diff(waypoints, axis=0)
        dists = np.hypot(diffs[:, 0], diffs[:, 1])
        s_nodes = np.concatenate(([0.0], np.cumsum(dists)))
        self.path_length = float(s_nodes[-1])
        self.spline_x = CubicSpline(s_nodes, waypoints[:, 0])
        self.spline_y = CubicSpline(s_nodes, waypoints[:, 1])
        self.is_path_ready = True
        print(f"[DFE-MPCC] Đã nạp thành công quỹ đạo tham chiếu: Chiều dài L = {self.path_length:.3f} m ({len(waypoints)} điểm).")

    def set_tube_planner(self, tube_planner):
        """Liên kết mô-đun Ống vận tốc khả thi bậc 3 Krasnoselskii-Mann (H124) vào bộ giải tối ưu NLP."""
        self.tube_planner = tube_planner
        n_pts = len(tube_planner.s_grid) if getattr(tube_planner, "is_ready", False) else 0
        print(f"[DFE-MPCC] Đã tích hợp Ống vận tốc khả thi bậc 3 Krasnoselskii-Mann vào Solver ({n_pts} điểm lưới).")

    def get_path_reference(self, s_val):
        """Lấy vị trí [xr, yr], tiếp tuyến [tx, ty] và góc hướng theta_path tại độ dài cung s."""
        if not self.is_path_ready:
            return np.zeros(2), np.array([1.0, 0.0]), 0.0
        s_clamped = np.clip(s_val, 0.0, self.path_length)
        xr = float(self.spline_x(s_clamped))
        yr = float(self.spline_y(s_clamped))
        dxr = float(self.spline_x.derivative()(s_clamped))
        dyr = float(self.spline_y.derivative()(s_clamped))
        norm_d = np.hypot(dxr, dyr) + 1e-9
        tx = dxr / norm_d
        ty = dyr / norm_d
        th_path = np.arctan2(dyr, dxr)
        return np.array([xr, yr]), np.array([tx, ty]), th_path

    def _build_solver(self):
        """Xây dựng bài toán tối ưu DFE-MPCC qua CasADi Opti."""
        opti = cs.Opti()
        
        # Biến quyết định
        X = opti.variable(self.nx, self.N + 1)
        U = opti.variable(self.nu, self.N)
        s_slosh = opti.variable(4, self.N)
        s_obs = opti.variable(1, self.N)
        s_zono = opti.variable(12, self.N)    # Biến bù cho 12 mặt biên Zonotope T1 (Exact L1 slack)
        s_tube = opti.variable(1, self.N)     # Biến bù trần Ống vận tốc khả thi (Exact L1 slack)
        
        # Tham số đầu vào cập nhật trực tuyến
        P_x0 = opti.parameter(self.nx)
        P_path_p = opti.parameter(2, self.N)
        P_path_t = opti.parameter(2, self.N)
        P_path_sbar = opti.parameter(1, self.N)
        P_th_path = opti.parameter(1, self.N)
        P_phi_est = opti.parameter(4)
        # [ox, oy, oz, r_obs, d_safe, vox, voy]; vox = voy = 0 -> vật cản tĩnh (giống bản cũ)
        P_obs = opti.parameter(7)
        P_mu = opti.parameter(1)              # Hệ số ma sát sàn trực tuyến
        P_q_eta = opti.parameter(1)           # Nguồn lượng tử Conformal an toàn q_eta (m) cho ràng buộc T2
        P_dfz_iso = opti.parameter(4)         # Biến thiên tải pháp tuyến 4 bánh do mặt đường ISO 8608 (N)
        P_v_ub = opti.parameter(1, self.N)    # Trần Ống vận tốc khả thi bậc 3 (m/s) dọc chân trời dự báo
        P_v_lb = opti.parameter(1, self.N)    # Sàn Ống vận tốc khả thi bậc 3 (m/s)
        
        # Điều kiện đầu
        opti.subject_to(X[:, 0] == P_x0)
        
        cost = 0.0
        for k in range(self.N):
            # Biến bù không âm
            opti.subject_to(s_slosh[:, k] >= 0)
            opti.subject_to(s_obs[0, k] >= 0)
            opti.subject_to(s_zono[:, k] >= 0)
            opti.subject_to(s_tube[0, k] >= 0)
            
            # Động học LTI chính xác ZOH
            x_k = X[:, k]
            u_k = U[:, k]
            x_next = X[:, k + 1]
            opti.subject_to(x_next == self.Ad_dm @ x_k + self.Bd_dm @ u_k)
            
            q_k = x_k[0:3]
            v_k = x_k[3:6]
            a_k = x_k[6:9]
            s_k = x_k[9]
            vs_k = x_k[10]
            as_k = x_k[11]
            
            u_amr = u_k[0:3]
            js_k = u_k[3]
            
            # 1a. Ràng buộc đầu vào điều khiển (luôn áp dụng cho mọi k = 0..N-1)
            opti.subject_to(opti.bounded(-4.0, u_amr[0:2], 4.0))
            opti.subject_to(opti.bounded(-6.0, u_amr[2], 6.0))
            opti.subject_to(opti.bounded(-8.0, js_k, 8.0))
            
            # Cải tiến U1a (skip0): Loại bỏ các ràng buộc chỉ phụ thuộc trạng thái tại k=0
            # (vì X[:, 0] == P_x0 đã cố định hằng số, loại bỏ 27 ràng buộc dư thừa giúp tránh Restoration_Failed)
            if k >= 1 or self.state_con_mode != "skip0":
                # 1b. Ràng buộc trạng thái cơ bản
                opti.subject_to(opti.bounded(self.x_bounds[0], q_k[0], self.x_bounds[1]))
                opti.subject_to(opti.bounded(self.y_bounds[0], q_k[1], self.y_bounds[1]))
                opti.subject_to(opti.bounded(0.0, s_k, 30.0))
                opti.subject_to(opti.bounded(-1.2, v_k[2], 1.2))
                opti.subject_to(opti.bounded(-1.2, a_k[0:2], 1.2))
                opti.subject_to(opti.bounded(-2.5, a_k[2], 2.5))
                opti.subject_to(opti.bounded(-1.2, as_k, 1.2))

                # RÀNG BUỘC ỐNG VẬN TỐC KHẢ THI BẬC 3 (PT 18 Mục 2.6):
                # 1. Tốc độ tiến trình bám sát ống: v_lb(s_k) <= dot_s_k <= v_ub(s_k) + s_tube,k
                v_ub_k = P_v_ub[0, k]
                v_lb_k = P_v_lb[0, k]
                opti.subject_to(vs_k >= v_lb_k)
                opti.subject_to(vs_k <= v_ub_k + s_tube[0, k])

                # 2. Vận tốc hình học thực tế không vượt quá trần ống: ||[v_x, v_y]||_2 <= v_ub(s_k) + s_tube,k
                opti.subject_to(v_k[0]**2 + v_k[1]**2 <= (v_ub_k + s_tube[0, k])**2)
                
                # 2. Ràng buộc vận tốc bánh xe Mecanum phi tuyến
                w_wheels = self.funcs['mecanum_wheels'](q_k, v_k)
                opti.subject_to(opti.bounded(-OMEGA_MAX, w_wheels, OMEGA_MAX))
                
                # 3. RÀNG BUỘC ĐỘNG LỰC HỌC TẢI PHÁP TUYẾN & ELIP MA SÁT ĐỘNG (DFE) + ISO 8608
                th_k = q_k[2]
                c_th = cs.cos(th_k)
                s_th = cs.sin(th_k)
                
                # Gia tốc thân xe trong Body Frame
                a_bx = c_th * a_k[0] + s_th * a_k[1]
                a_by = -s_th * a_k[0] + c_th * a_k[1]
                
                # Lực phản kháng dao động sóng sánh thuần túy FSI (Convective Slosh Residual)
                phi_x_k = P_phi_est[0]
                phi_y_k = P_phi_est[2]
                ddphi_x_k = -omega_x**2 * phi_x_k + (1.0 / lx) * a_bx
                ddphi_y_k = -omega_y**2 * phi_y_k - (1.0 / ly) * a_by
                a_rel_pend_x = -lx * ddphi_x_k * cs.cos(phi_x_k)
                a_rel_pend_y =  ly * ddphi_y_k * cs.cos(phi_y_k)

                # Lực sóng sánh dư tác động lên bệ xe: F_slosh = -m_1 * a_rel
                F_slosh_bx = -m_1x * a_rel_pend_x
                F_slosh_by = -m_1y * a_rel_pend_y

                # Mô-men lật tổng hợp quanh điểm tiếp xúc sàn (M_total đã gồm toàn bộ 100% nước):
                M_pitch = M_TOTAL * a_bx * Z_COM - F_slosh_bx * Z_SLOSH_X
                M_roll  = M_TOTAL * a_by * Z_COM - F_slosh_by * Z_SLOSH_Y
                
                # Chuyển dịch tải trọng pháp tuyến lên 4 bánh xe + Nhiễu mặt đường ngẫu nhiên ISO 8608
                dFz_pitch = M_pitch / (4.0 * L_X)
                dFz_roll  = M_roll  / (4.0 * L_Y)
                
                Fz_1 = FZ_STATIC - dFz_pitch - dFz_roll + P_dfz_iso[0]  # FL
                Fz_2 = FZ_STATIC - dFz_pitch + dFz_roll + P_dfz_iso[1]  # FR
                Fz_3 = FZ_STATIC + dFz_pitch + dFz_roll + P_dfz_iso[2]  # RR
                Fz_4 = FZ_STATIC + dFz_pitch - dFz_roll + P_dfz_iso[3]  # RL
                
                # 3.1 Ràng buộc chống nhấc bánh (Anti-tipping: Fz >= Fz_min)
                opti.subject_to(Fz_1 >= FZ_MIN)
                opti.subject_to(Fz_2 >= FZ_MIN)
                opti.subject_to(Fz_3 >= FZ_MIN)
                opti.subject_to(Fz_4 >= FZ_MIN)
                
                # 3.2 Lực bệ xe yêu cầu trong Body Frame (M_TOTAL * a_b - F_slosh)
                F_cmd_bx = M_TOTAL * a_bx - F_slosh_bx
                F_cmd_by = M_TOTAL * a_by - F_slosh_by
                Tau_cmd_bz = IZ_BASE * a_k[2]
                
                # 3.3 Vector Wrench yêu cầu trên thân xe: W_cmd = [F_bx, F_by, Tau_bz]^T
                W_cmd = cs.vertcat(F_cmd_bx, F_cmd_by, Tau_cmd_bz)

                # 3.4 Giới hạn bám ma sát 4 bánh xe Mecanum với biên hoạch định P3 (mu_plan_scale = 0.95):
                # ||F_ground,i|| = sqrt(2)*|f_i| <= (mu_plan_scale * mu) * Fz_i
                inv_sqrt2 = 1.0 / np.sqrt(2.0)
                mu_plan = self.mu_plan_scale * P_mu
                f_bar_1 = (mu_plan * Fz_1) * inv_sqrt2
                f_bar_2 = (mu_plan * Fz_2) * inv_sqrt2
                f_bar_3 = (mu_plan * Fz_3) * inv_sqrt2
                f_bar_4 = (mu_plan * Fz_4) * inv_sqrt2
                f_bar_vec = cs.vertcat(f_bar_1, f_bar_2, f_bar_3, f_bar_4)

                # 3.5 RÀNG BUỘC MIỀN ZONOTOPE T1 (12 MẶT BIÊN TRONG KHÔNG GIAN WRENCH 3D):
                d_zono = self.M_reach_dm @ f_bar_vec
                opti.subject_to(self.C_zono_dm @ W_cmd <= d_zono + s_zono[:, k])
            
            # 4. Sai số hình học MPCC
            p_bar = P_path_p[:, k]
            t_bar = P_path_t[:, k]
            s_bar = P_path_sbar[0, k]
            n_bar = cs.vertcat(-t_bar[1], t_bar[0])
            p_rob = cs.vertcat(q_k[0], q_k[1])
            e_k = p_rob - (p_bar + t_bar * (s_k - s_bar))
            eps_c = n_bar[0] * e_k[0] + n_bar[1] * e_k[1]
            eps_l = t_bar[0] * e_k[0] + t_bar[1] * e_k[1]
            
            # 5. Slosh-ECBF bậc 3 (Soft constraint): sử dụng ngưỡng bảo toàn nội bộ PHI_MAX_X/Y
            # (~18.75 mm dọc, ~9.00 mm ngang đã có biên dự trữ dưới trần tràn 30 mm) khi P_phi_est giữ cố định trên chân trời.
            # Chỉ thu hẹp thêm góc ECBF nếu tổng ngưỡng ECBF + P_q_eta vượt quá trần tự do thực h_free = self.h_lim_m;
            # trong khi toàn bộ lượng tử bảo toàn P_q_eta vẫn được áp dụng đầy đủ ở ràng buộc đích T2 (eta_peak + P_q_eta <= h_free).
            b3_slosh = self.funcs['slosh_cbf'](q_k, v_k, a_k, u_amr, P_phi_est)
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
                K_SLOSH_0_Y * dtheta_margin_y
            )
            opti.subject_to(b3_slosh - q_eta_b3_margin + s_slosh[:, k] >= 0)
            
            # 6. Obstacle-ECBF bậc 3 (Soft constraint)
            # Vật cản di chuyển vận tốc hằng trên tầm dự báo: o(t_k) = o_0 + v_o * t_k.
            # Với h = |p - o|^2 - r^2, các đạo hàm dùng vận tốc tương đối (v - v_o).
            # Khi v_o = 0 biểu thức trùng với ECBF vật cản tĩnh trước đây.
            t_k = k * self.dt
            obs_xy_k = P_obs[0:2] + P_obs[5:7] * t_k
            P_obs_k = cs.vertcat(obs_xy_k, P_obs[2:5])
            v_rel_k = cs.vertcat(v_k[0] - P_obs[5], v_k[1] - P_obs[6], v_k[2])
            b3_obs = self.funcs['obs_cbf'](q_k, v_rel_k, a_k, u_amr, P_obs_k)
            opti.subject_to(b3_obs + s_obs[0, k] >= 0)
            
            # Tích lũy hàm mục tiêu có thứ bậc chính xác (Exact Hierarchical Penalty)
            cost += self.w_contour * (eps_c ** 2)
            cost += self.w_lag * (eps_l ** 2)
            cost -= self.w_progress * vs_k
            cost += self.w_heading * ((q_k[2] - P_th_path[0, k]) ** 2)
            cost += self.w_u * cs.sumsqr(u_k)
            cost += self.w_v * cs.sumsqr(v_k)
            cost += self.w_a * cs.sumsqr(a_k)
            # Exact L1 Slack Penalties (Kerrigan & Maciejowski 2000)
            cost += self.w_slack_slosh * cs.sum1(s_slosh[:, k])
            cost += self.w_slack_obs * s_obs[0, k]
            cost += self.w_slack_trac * cs.sum1(s_zono[:, k])
            cost += 2000.0 * s_tube[0, k] + 10000.0 * (s_tube[0, k] ** 2)
            
        # Chi phí điểm cuối (Terminal Cost)
        cost += 40.0 * (eps_c ** 2) + 40.0 * (eps_l ** 2) + 20.0 * ((X[2, self.N] - P_th_path[0, self.N - 1]) ** 2) - 6.0 * X[10, self.N]
        
        # 7. RÀNG BUỘC PHANH DỰ PHÒNG T2 TẠI BƯỚC CUỐI CHÂN TRỜI (k = N) CÓ BÙ LƯỢNG TỬ q_eta
        # Năng lượng con lắc dịch tâm phụ thuộc trực tiếp vào vận tốc & gia tốc cuối chân trời X[:, N]
        a_b_nom = 0.80
        phi_eq_x = a_b_nom / 9.81
        v_N_norm_sq = X[3, self.N]**2 + X[4, self.N]**2
        a_N_norm_sq = X[6, self.N]**2 + X[7, self.N]**2
        phi_x_N = cs.fabs(P_phi_est[0]) + 0.08 * cs.sqrt(a_N_norm_sq + 1e-4)
        dphi_x_N = cs.fabs(P_phi_est[1]) + 0.35 * cs.sqrt(v_N_norm_sq + 1e-4)
        omega_x_val = 13.84
        phi_peak_brake_x = phi_eq_x + cs.sqrt((phi_x_N - phi_eq_x)**2 + (dphi_x_N / omega_x_val)**2 + 1e-6)
        eta_peak_N_m = 0.5 * 0.150 * phi_peak_brake_x
        h_free_m = self.h_lim_m  # mặc định 30 mm
        s_term_slosh = opti.variable(1)
        opti.subject_to(s_term_slosh >= 0)
        opti.subject_to(eta_peak_N_m + P_q_eta <= h_free_m + s_term_slosh)
        cost += 5000.0 * s_term_slosh

        opti.minimize(cost)
        
        # Thiết lập bộ giải IPOPT (Cải tiến U2: giới hạn max_iter cắt đuôi trễ P95)
        opts_p = {'expand': True, 'print_time': False}
        opts_s = {
            'max_iter': self.ipopt_max_iter,
            'print_level': 0,
            'tol': 1e-2,
            'warm_start_init_point': 'yes',
            'acceptable_tol': 5e-2,
            'acceptable_iter': 4
        }
        opti.solver('ipopt', opts_p, opts_s)
        
        self.opti = opti
        self.X = X
        self.U = U
        self.s_slosh = s_slosh
        self.s_obs = s_obs
        self.s_zono = s_zono
        self.s_tube = s_tube
        self.s_term_slosh = s_term_slosh
        self.P_x0 = P_x0
        self.P_path_p = P_path_p
        self.P_path_t = P_path_t
        self.P_path_sbar = P_path_sbar
        self.P_th_path = P_th_path
        self.P_phi_est = P_phi_est
        self.P_obs = P_obs
        self.P_mu = P_mu
        self.P_q_eta = P_q_eta
        self.P_dfz_iso = P_dfz_iso
        self.P_v_ub = P_v_ub
        self.P_v_lb = P_v_lb

    def solve_step(self, x_current, phi_est, obs_info=None, mu_floor=None, q_eta=None, dfz_iso=None, v_ub_horizon=None, v_lb_horizon=None):
        """
        Thực thi giải một chu kỳ DFE-MPCC (20 Hz) nhúng Ống vận tốc khả thi bậc 3 (H124)
        và biên mặt đường ngẫu nhiên ISO 8608.
        """
        t_start = time.time()
        if obs_info is None:
            obs_info = np.array([10.0, 10.0, 0.5, 0.2, 0.1])
        obs_info = np.asarray(obs_info, dtype=float).ravel()
        if obs_info.size < 7:
            obs_info = np.concatenate([obs_info, np.zeros(7 - obs_info.size)])
        if mu_floor is None:
            mu_floor = self.default_mu
        q_eta_apply = float(q_eta if q_eta is not None else self.default_q_eta_m)
        if dfz_iso is None:
            dfz_iso_vec = np.zeros(4)
        else:
            dfz_iso_vec = np.asarray(dfz_iso, dtype=float).reshape(4)
            
        # 1. Trích xuất tham số tham chiếu dọc chuỗi s dự báo
        path_p = np.zeros((2, self.N))
        path_t = np.zeros((2, self.N))
        path_sbar = np.zeros((1, self.N))
        th_path = np.zeros((1, self.N))
        
        s_cur = float(np.clip(x_current[9], 0.0, self.path_length))
        vs_cur = float(np.clip(x_current[10], 0.05, 0.75))
        vs_preview = max(vs_cur, 0.40)
        
        prev_h = float(x_current[2])
        for k in range(self.N):
            s_k = float(np.clip(s_cur + k * self.dt * vs_preview, 0.0, self.path_length))
            p_ref, t_ref, heading = self.get_path_reference(s_k)
            path_p[:, k] = p_ref
            path_t[:, k] = t_ref
            path_sbar[0, k] = s_k
            
            d_th = np.arctan2(np.sin(heading - prev_h), np.cos(heading - prev_h))
            prev_h = prev_h + d_th
            th_path[0, k] = prev_h
            
        # 2. Nạp tham số vào bộ giải
        self.opti.set_value(self.P_x0, x_current)
        self.opti.set_value(self.P_path_p, path_p)
        self.opti.set_value(self.P_path_t, path_t)
        self.opti.set_value(self.P_path_sbar, path_sbar)
        self.opti.set_value(self.P_th_path, th_path)
        self.opti.set_value(self.P_phi_est, phi_est)
        self.opti.set_value(self.P_obs, obs_info)
        self.opti.set_value(self.P_mu, float(mu_floor))
        self.opti.set_value(self.P_q_eta, q_eta_apply)
        self.opti.set_value(self.P_dfz_iso, dfz_iso_vec)
        
        # Trích xuất trần vận tốc v_ub và sàn v_lb dọc theo chân trời dự báo N_p từ Ống Krasnoselskii-Mann
        if v_ub_horizon is not None:
            v_ub_vec = np.asarray(v_ub_horizon, dtype=float).reshape(1, self.N)
            v_lb_vec = np.asarray(v_lb_horizon, dtype=float).reshape(1, self.N) if v_lb_horizon is not None else np.zeros((1, self.N))
        elif self.tube_planner is not None and getattr(self.tube_planner, "is_ready", False):
            s_preview = path_sbar[0, :]
            v_ub_interp = np.interp(s_preview, self.tube_planner.s_grid, self.tube_planner.v_ub_nom)
            v_ub_vec = np.clip(v_ub_interp, 0.25, 0.65).reshape(1, self.N)
            v_lb_vec = np.zeros((1, self.N))
        else:
            v_ub_vec = np.full((1, self.N), 0.60)
            v_lb_vec = np.zeros((1, self.N))
            
        self.opti.set_value(self.P_v_ub, v_ub_vec)
        self.opti.set_value(self.P_v_lb, v_lb_vec)
        
        # 3. Warm start từ chu kỳ trước
        if np.any(self.x_traj_prev != 0):
            x_init = np.zeros((self.nx, self.N + 1))
            x_init[:, :-1] = self.x_traj_prev[:, 1:]
            x_init[:, -1] = self.x_traj_prev[:, -1]
            self.opti.set_initial(self.X, x_init)
            
            u_init = np.zeros((self.nu, self.N))
            u_init[:, :-1] = self.u_traj_prev[:, 1:]
            u_init[:, -1] = self.u_traj_prev[:, -1]
            self.opti.set_initial(self.U, u_init)
            
        # 4. Giải tối ưu NLP
        try:
            sol = self.opti.solve()
            x_pred = sol.value(self.X)
            u_pred = sol.value(self.U)
            u_opt = u_pred[0:3, 0]
            js_opt = float(u_pred[3, 0])
            slack_zono_val = float(np.max(sol.value(self.s_zono)))
            slack_tube_val = float(np.max(sol.value(self.s_tube)))
            status = 'optimal'
        except Exception:
            x_pred = self.opti.debug.value(self.X)
            u_pred = self.opti.debug.value(self.U)
            u_opt = u_pred[0:3, 0]
            js_opt = float(u_pred[3, 0])
            slack_zono_val = float(np.max(self.opti.debug.value(self.s_zono)))
            slack_tube_val = float(np.max(self.opti.debug.value(self.s_tube)))
            status = 'suboptimal_fallback'
            # Nghiệm debug của IPOPT có thể vi phạm hộp jerk; kẹp lại để lệnh áp dụng
            # luôn nằm trong giới hạn |j_xy| <= 4, |j_th| <= 6, |js| <= 8.
            u_opt = np.array(u_opt, dtype=float).copy()
            u_opt[0:2] = np.clip(u_opt[0:2], -4.0, 4.0)
            u_opt[2] = float(np.clip(u_opt[2], -6.0, 6.0))
            js_opt = float(np.clip(js_opt, -8.0, 8.0))
            
        self.x_traj_prev = x_pred.copy()
        self.u_traj_prev = u_pred.copy()
        
        t_solve_ms = (time.time() - t_start) * 1000.0
        
        extra_info = {
            'solve_time_ms': t_solve_ms,
            'status': status,
            'slack_tube': slack_tube_val,
            'v_ub_preview': v_ub_vec.flatten(),
            'slack_zono_max': slack_zono_val
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
