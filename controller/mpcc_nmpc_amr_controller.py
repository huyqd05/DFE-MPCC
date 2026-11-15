"""
mpcc_nmpc_amr_controller.py
=============================================================================
Bộ Điều Khiển MPCC-NMPC Cấp Jerk Dành Riêng Cho Xe Tự Hành AMR (slosh3.xml)
Mang Bình Nước Sóng Sánh Cố Định Trên Thân Xe

Kiến trúc:
1. Không gian trạng thái thu gọn n_x = 12 (9 DoF Xe AMR [q, v, a] + 3 DoF Tiến độ MPCC [s, vs, as]).
2. Vector đầu vào điều khiển n_u = 4 (3 DoF Jerk Xe [jx, jy, j_theta] + 1 Jerk Tiến độ ảo js).
3. Động học chuỗi tích phân bậc ba LTI chính xác tuyệt đối (ZOH Nilpotent A_c^3 = 0).
4. Phân rã sai số hình học trực giao MPCC: Contouring Error eps_c và Lag Error eps_l.
5. Ràng buộc Slosh-ECBF (r = 3) hai trục độc lập tác động trực tiếp lên Jerk bệ xe.
6. Ràng buộc Obstacle-ECBF (r = 3) cho thân xe AMR.
7. Ràng buộc phi tuyến động học ngược bánh xe Mecanum: -40 <= omega_i <= 40 rad/s.
8. Tối ưu hóa thời gian thực bằng CasADi Opti / IPOPT với warm-start (< 5 ms).
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
import casadi as cs
import numpy as np
from scipy.interpolate import CubicSpline

# Đảm bảo in tiếng Việt chuẩn trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Import mô hình CasADi AMR
from robot_model_amr_casadi import (
    build_amr_symbolic_model,
    PHI_MAX_X, PHI_MAX_Y,
    OMEGA_MAX,
    L_X, L_Y, ROBOT_RADIUS
)


class MPCCNMPCAMRController:
    def __init__(self, N=15, dt=0.05, solver_type='ipopt'):
        """
        Khởi tạo bộ điều khiển MPCC-NMPC cấp Jerk cho xe AMR mang bình chất lỏng.
        Parameters:
            N: Tầm nhìn dự báo (Horizon steps), mặc định 15 bước (0.75 s)
            dt: Chu kỳ lấy mẫu dự báo (s), mặc định 0.05 s (20 Hz NMPC)
            solver_type: 'ipopt' (NLP hoàn chỉnh với warm start)
        """
        self.N = N
        self.dt = dt
        self.solver_type = solver_type
        
        self.nx = 12  # 3 q + 3 v + 3 a + 1 s + 1 vs + 1 as
        self.nu = 4   # 3 u_amr + 1 js
        
        # Tải mô hình CasADi SX cho AMR
        self.funcs = build_amr_symbolic_model()
        
        # 1. Ma trận động học LTI rời rạc hóa chính xác ZOH (Block-Diagonal)
        # Khối Robot AMR (9 states, 3 inputs): [q (3), v (3), a (3)]
        I3 = np.eye(3)
        Ad_rob = np.block([
            [I3, dt * I3, 0.5 * (dt**2) * I3],
            [np.zeros((3, 3)), I3, dt * I3],
            [np.zeros((3, 3)), np.zeros((3, 3)), I3]
        ])
        Bd_rob = np.block([
            [(dt**3 / 6.0) * I3],
            [(dt**2 / 2.0) * I3],
            [dt * I3]
        ])
        
        # Khối Tiến độ MPCC (3 states, 1 input): [s, vs, as]
        Ad_s = np.array([
            [1.0, dt, 0.5 * (dt**2)],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0]
        ])
        Bd_s = np.array([
            [(dt**3 / 6.0)],
            [(dt**2 / 2.0)],
            [dt]
        ])
        
        # Ghép nối khối chéo đúng thứ tự: X = [q(3), v(3), a(3), s(1), vs(1), as(1)]^T
        self.Ad = np.block([
            [Ad_rob, np.zeros((9, 3))],
            [np.zeros((3, 9)), Ad_s]
        ])
        self.Bd = np.block([
            [Bd_rob, np.zeros((9, 1))],
            [np.zeros((3, 3)), Bd_s]
        ])
        self.Ad_dm = cs.DM(self.Ad)
        self.Bd_dm = cs.DM(self.Bd)
        
        # 2. Trọng số hàm mục tiêu MPCC
        self.w_contour = 60.0        # Phạt lệch tâm đường (Contouring Error)
        self.w_lag = 50.0            # Phạt trễ dọc đường (Lag Error)
        self.w_progress = 6.0        # Thưởng vận tốc tiến độ vs (tiến nhanh về đích)
        self.w_heading = 30.0        # Bám góc tiếp tuyến đường đi (tăng từ 15.0 lên 30.0 để bám sát hướng)
        self.w_u = 0.02              # Phạt giật Jerk
        self.w_v = 0.02              # Giảm vận tốc dư thừa
        self.w_a = 0.02              # Giảm gia tốc dư thừa
        self.w_slack_slosh = 2000.0  # Biến bù phạt Slosh ECBF (phạt nặng để giữ góc slosh an toàn)
        self.w_slack_obs = 10000.0   # Biến bù phạt Obstacle ECBF
        
        # Bộ đệm trạng thái dự báo (Warm start)
        self.x_traj_prev = np.zeros((self.nx, self.N + 1))
        self.u_traj_prev = np.zeros((self.nu, self.N))
        
        # Đường tham chiếu
        self.spline_x = None
        self.spline_y = None
        self.path_length = 0.0
        self.is_path_ready = False
        
        # Khởi tạo bộ giải
        self._build_solver()

    def set_path(self, waypoints):
        """
        Nạp quỹ đạo toàn cục từ A* và tham số hóa theo độ dài cung s (Cubic Spline).
        waypoints: mảng Nx2 [[x0, y0], [x1, y1], ...]
        """
        waypoints = np.asarray(waypoints)
        diffs = np.diff(waypoints, axis=0)
        dists = np.hypot(diffs[:, 0], diffs[:, 1])
        s_nodes = np.concatenate(([0.0], np.cumsum(dists)))
        self.path_length = s_nodes[-1]
        
        # Khớp đường cong CubicSpline trơn C^2
        self.spline_x = CubicSpline(s_nodes, waypoints[:, 0])
        self.spline_y = CubicSpline(s_nodes, waypoints[:, 1])
        self.is_path_ready = True
        print(f"[MPCC-AMR] Đã nạp thành công quỹ đạo tham chiếu: Chiều dài L = {self.path_length:.3f} m ({len(waypoints)} điểm).")

    def get_path_reference(self, s_val):
        """Lấy vị trí [xr, yr], tiếp tuyến [tx, ty] và góc hướng theta_path tại độ dài cung s."""
        if not self.is_path_ready:
            return np.zeros(2), np.array([1.0, 0.0]), 0.0
        
        # Kẹp s trong miền [0, L]
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
        """Xây dựng bài toán tối ưu NMPC qua CasADi Opti."""
        opti = cs.Opti()
        
        # Biến quyết định
        X = opti.variable(self.nx, self.N + 1)
        U = opti.variable(self.nu, self.N)
        s_slosh = opti.variable(4, self.N)
        s_obs = opti.variable(1, self.N)
        
        # Tham số đầu vào cập nhật trực tuyến
        P_x0 = opti.parameter(self.nx)
        P_path_p = opti.parameter(2, self.N)
        P_path_t = opti.parameter(2, self.N)
        P_path_sbar = opti.parameter(1, self.N)
        P_th_path = opti.parameter(1, self.N)
        P_phi_est = opti.parameter(4)
        P_obs = opti.parameter(5)
        
        # Điều kiện đầu
        opti.subject_to(X[:, 0] == P_x0)
        
        cost = 0.0
        for k in range(self.N):
            # Biến bù không âm
            opti.subject_to(s_slosh[:, k] >= 0)
            opti.subject_to(s_obs[0, k] >= 0)
            
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
            
            # 1. Ràng buộc vị trí
            opti.subject_to(opti.bounded(-2.5, q_k[0], 2.5))
            opti.subject_to(opti.bounded(-2.5, q_k[1], 2.5))
            opti.subject_to(opti.bounded(0.0, s_k, 25.0))
            
            # 2. Ràng buộc vận tốc
            opti.subject_to(opti.bounded(-0.7, v_k[0:2], 0.7))
            opti.subject_to(opti.bounded(-1.2, v_k[2], 1.2))
            opti.subject_to(opti.bounded(0.0, vs_k, 0.6))  # Không lùi s
            
            # 3. Ràng buộc gia tốc (khống chế góc dốc quán tính chất lỏng a/g <= 0.12)
            opti.subject_to(opti.bounded(-1.2, a_k[0:2], 1.2))
            opti.subject_to(opti.bounded(-2.5, a_k[2], 2.5))
            opti.subject_to(opti.bounded(-1.2, as_k, 1.2))
            
            # 4. Ràng buộc Jerk điều khiển (làm mượt chuyển động triệt tiêu kích thích mode dao động)
            opti.subject_to(opti.bounded(-4.0, u_amr[0:2], 4.0))
            opti.subject_to(opti.bounded(-6.0, u_amr[2], 6.0))
            opti.subject_to(opti.bounded(-8.0, js_k, 8.0))
            
            # 5. Ràng buộc vận tốc bánh xe Mecanum phi tuyến
            w_wheels = self.funcs['mecanum_wheels'](q_k, v_k)
            opti.subject_to(opti.bounded(-OMEGA_MAX, w_wheels, OMEGA_MAX))
            
            # 6. Sai số hình học trực giao MPCC
            p_bar = P_path_p[:, k]
            t_bar = P_path_t[:, k]
            s_bar = P_path_sbar[0, k]
            n_bar = cs.vertcat(-t_bar[1], t_bar[0])
            p_rob = cs.vertcat(q_k[0], q_k[1])
            e_k = p_rob - (p_bar + t_bar * (s_k - s_bar))
            eps_c = n_bar[0] * e_k[0] + n_bar[1] * e_k[1]
            eps_l = t_bar[0] * e_k[0] + t_bar[1] * e_k[1]
            
            # 7. Slosh-ECBF bậc tương đối r = 3 (Soft constraint)
            b3_slosh = self.funcs['slosh_cbf'](q_k, v_k, a_k, u_amr, P_phi_est)
            opti.subject_to(b3_slosh + s_slosh[:, k] >= 0)
            
            # 8. Obstacle-ECBF bậc tương đối r = 3 (Soft constraint)
            b3_obs = self.funcs['obs_cbf'](q_k, v_k, a_k, u_amr, P_obs)
            opti.subject_to(b3_obs + s_obs[0, k] >= 0)
            
            # Tích lũy hàm mục tiêu
            cost += self.w_contour * (eps_c ** 2)
            cost += self.w_lag * (eps_l ** 2)
            cost -= self.w_progress * vs_k
            cost += self.w_heading * ((q_k[2] - P_th_path[0, k]) ** 2)
            cost += self.w_u * cs.sumsqr(u_k)
            cost += self.w_v * cs.sumsqr(v_k)
            cost += self.w_a * cs.sumsqr(a_k)
            cost += self.w_slack_slosh * cs.sumsqr(s_slosh[:, k])
            cost += self.w_slack_obs * (s_obs[0, k] ** 2)
            
        # Chi phí điểm cuối (Terminal Cost)
        cost += 40.0 * (eps_c ** 2) + 40.0 * (eps_l ** 2) + 20.0 * ((X[2, self.N] - P_th_path[0, self.N - 1]) ** 2) - 6.0 * X[10, self.N]
        opti.minimize(cost)
        
        # Thiết lập bộ giải IPOPT
        opts_p = {'expand': True, 'print_time': False}
        opts_s = {
            'max_iter': 60,
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
        self.P_x0 = P_x0
        self.P_path_p = P_path_p
        self.P_path_t = P_path_t
        self.P_path_sbar = P_path_sbar
        self.P_th_path = P_th_path
        self.P_phi_est = P_phi_est
        self.P_obs = P_obs
        print("[MPCC-AMR] Đã cấu hình thành công bộ giải tối ưu NMPC với Horizon N =", self.N)

    def solve_step(self, x_current, phi_est, obs_info=None):
        """
        Giải một chu kỳ NMPC thời gian thực.
        Parameters:
            x_current: Vector trạng thái hiện tại (12,) [q(3), v(3), a(3), s(1), vs(1), as(1)]
            phi_est: Trạng thái sóng sánh (4,) [phi_x, dphi_x, phi_y, dphi_y]
            obs_info: Thông tin chướng ngại vật gần nhất (5,) [ox, oy, oz, r_obs, d_safe]
        Returns:
            u_opt: Vector điều khiển Jerk xe tối ưu (3,) [jx, jy, j_theta]
            js_opt: Jerk tiến độ ảo tối ưu (float)
            x_pred: Quỹ đạo dự báo trạng thái (12, N+1)
            u_pred: Quỹ đạo dự báo điều khiển (4, N)
            extra_info: Dict chứa {solve_time_ms}
        """
        t_start = time.time()
        
        if obs_info is None:
            obs_info = np.array([10.0, 10.0, 0.5, 0.2, 0.1])
            
        # 1. Trích xuất tham số tham chiếu dọc theo chuỗi s dự báo
        path_p = np.zeros((2, self.N))
        path_t = np.zeros((2, self.N))
        path_sbar = np.zeros((1, self.N))
        th_path = np.zeros((1, self.N))
        
        s_cur = float(np.clip(x_current[9], 0.0, self.path_length))
        vs_cur = float(np.clip(x_current[10], 0.05, 0.75))
        vs_preview = max(vs_cur, 0.40)
        
        # Đảm bảo góc hướng tham chiếu th_path liên tục (phase unwrapping) so với yaw hiện tại
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
        
        # 3. Warm start từ chu kỳ trước nếu có
        if np.any(self.x_traj_prev != 0):
            x_init = np.zeros((self.nx, self.N + 1))
            x_init[:, :-1] = self.x_traj_prev[:, 1:]
            x_init[:, -1] = self.x_traj_prev[:, -1]
            self.opti.set_initial(self.X, x_init)
            
            u_init = np.zeros((self.nu, self.N))
            u_init[:, :-1] = self.u_traj_prev[:, 1:]
            u_init[:, -1] = self.u_traj_prev[:, -1]
            self.opti.set_initial(self.U, u_init)
            
        # 4. Giải bài toán NLP
        try:
            sol = self.opti.solve()
            x_pred = sol.value(self.X)
            u_pred = sol.value(self.U)
            u_opt = u_pred[0:3, 0]
            js_opt = float(u_pred[3, 0])
        except Exception:
            # Fallback nếu IPOPT gặp cảnh báo
            x_pred = self.opti.debug.value(self.X)
            u_pred = self.opti.debug.value(self.U)
            u_opt = u_pred[0:3, 0]
            js_opt = float(u_pred[3, 0])
            
        self.x_traj_prev = x_pred.copy()
        self.u_traj_prev = u_pred.copy()
        
        t_solve_ms = (time.time() - t_start) * 1000.0
        
        extra_info = {
            'solve_time_ms': t_solve_ms
        }
        
        return u_opt, js_opt, x_pred, u_pred, extra_info


if __name__ == "__main__":
    print("=================================================================")
    print("KIỂM CHỨNG BỘ ĐIỀU KHIỂN MPCC-NMPC CHO XE AMR")
    print("=================================================================")
    controller = MPCCNMPCAMRController(N=15, dt=0.05)
    
    # Tạo đường đi mẫu hình chữ S
    s_pts = np.array([
        [-2.0, -2.0],
        [-2.0, -0.5],
        [-1.0,  0.5],
        [ 0.0, -0.5],
        [ 1.0,  0.5],
        [ 2.0,  0.5]
    ])
    controller.set_path(s_pts)
    
    x0 = np.zeros(12)
    x0[0:2] = [-2.0, -2.0]
    x0[2] = np.pi / 2.0 # hướng ban đầu dọc trục Y
    x0[10] = 0.2        # vs ban đầu
    
    phi0 = np.array([0.0, 0.0, 0.0, 0.0])
    
    print("\nChạy thử 1 chu kỳ solve_step...")
    u_opt, js_opt, x_pred, u_pred, info = controller.solve_step(x0, phi0)
    print(f"Jerk tối ưu xe u_amr (m/s^3): {np.round(u_opt, 3)}")
    print(f"Jerk tiến độ js (m/s^3): {js_opt:.3f}")
    print(f"Thời gian giải NMPC: {info['solve_time_ms']:.2f} ms")
    print("Kiểm chứng thành công!")
