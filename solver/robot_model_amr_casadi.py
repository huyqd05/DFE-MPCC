"""
robot_model_amr_casadi.py
=============================================================================
Mô hình Động học & Động lực học Tích hợp CasADi SX cho Xe AMR 3-DoF
mang Bình Nước Sóng Sánh Cố Định (Housner DMTP) trong mô hình slosh3.xml.

Đặc điểm kiến trúc:
1. AMR toàn hướng Mecanum 3-DoF (x_b, y_b, theta_b).
2. Bình chất lỏng gắn cố định trên sàn xe (không dùng cánh tay/gimbal).
3. Động học LTI chuỗi tích phân bậc ba rời rạc hóa chính xác tuyệt đối (ZOH Nilpotent).
4. Phân rã sai số MPCC: Contouring Error eps_c và Lag Error eps_l.
5. Slosh-ECBF bậc tương đối r = 3 theo 2 trục X và Y độc lập, khống chế trực tiếp Jerk bệ xe.
6. Obstacle-ECBF bậc tương đối r = 3 cho thân xe AMR.
7. Động học ngược 4 bánh xe Mecanum omega_i(q, v).
8. Lực phản kháng thủy động FSI và tọa độ ZMP giám sát ổn định lật.
=============================================================================
"""

import os
import sys
import time
import casadi as cs
import numpy as np

# Đảm bảo in tiếng Việt trên console Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ===========================================================================
# THÔNG SỐ VẬT LÝ VÀ HÌNH HỌC TỪ SLOSH3.XML
# ===========================================================================
g = 9.81

# 1. Bệ xe AMR (Mecanum Chassis - Không tải 5.5 kg)
M_BASE = 5.5            # Khối lượng khung xe không tải (kg) (khoảng 5-6 kg)
IZ_BASE = 0.045         # Mô-men quán tính quay trục Z (kg.m^2)
Z_CART = 0.06           # Cao độ sàn xe trong World Frame (m)
R_W = 0.04              # Bán kính bánh xe Mecanum (m)
L_X = 0.12              # Nửa chiều dài cơ sở bánh xe theo X (m)
L_Y = 0.14              # Nửa chiều rộng cơ sở bánh xe theo Y (m)
L_SUM = L_X + L_Y       # 0.26 m
OMEGA_MAX = 40.0        # Giới hạn tốc độ góc bánh xe (rad/s) (~380 rpm)
ROBOT_RADIUS = 0.24     # Bán kính hình tròn bao an toàn cho xe AMR (m)

# 2. Bình nước sóng sánh (Housner DMTP: Lòng trong 150 x 100 x 110 mm)
TANK_A = 0.150          # Chiều dài trong bình theo X (m) (150 mm)
TANK_B = 0.100          # Chiều rộng trong bình theo Y (m) (100 mm)
TANK_H = 0.110          # Chiều cao toàn bộ thành bình (m) (110 mm)
H_LIQUID = 0.080        # Chiều cao mực nước tĩnh (m) (80 mm ứng với 1.2 kg nước)
M_LIQUID = 1.200        # Khối lượng nước (kg) (1200 g)
M_SHELL = 0.550         # Khối lượng vỏ bình mica + cảm biến (kg) (550 g)
M_PAYLOAD = M_SHELL + M_LIQUID # 1.750 kg (khoảng 1.7 - 1.8 kg tải trên)
M_TOTAL = M_BASE + M_PAYLOAD   # 5.5 + 1.75 = 7.250 kg
Z_COM = 0.0555                 # Cao độ khối tâm danh định của toàn hệ từ mặt đất z=0 (khớp MuJoCo 0.0555 m)
ZETA_SLOSH = 0.015      # Tỉ số cản dao động mặt nước (1.5%)

# Tính tham số con lắc Housner tương đương trục X (a = 150 mm)
xi_x = np.pi * H_LIQUID / TANK_A
tanh_x = float(np.tanh(xi_x))
omega_x = float(np.sqrt((np.pi * g / TANK_A) * tanh_x))   # ~13.84 rad/s (~2.20 Hz)
lx = float(TANK_A / (np.pi * tanh_x))                     # ~0.0512 m (51.2 mm)
m_1x = float(M_LIQUID * (8.0 * TANK_A / (np.pi**3 * H_LIQUID)) * tanh_x) # ~0.5412 kg
m_0x = M_LIQUID - m_1x                                    # ~0.6588 kg
beta_x = 1.0 / lx
k_th_x = omega_x**2
hd_x = float(2.0 * ZETA_SLOSH * omega_x)

# Tính tham số con lắc Housner tương đương trục Y (b = 100 mm)
xi_y = np.pi * H_LIQUID / TANK_B
tanh_y = float(np.tanh(xi_y))
omega_y = float(np.sqrt((np.pi * g / TANK_B) * tanh_y))   # ~17.44 rad/s (~2.78 Hz)
ly = float(TANK_B / (np.pi * tanh_y))                     # ~0.0323 m (32.3 mm)
m_1y = float(M_LIQUID * (8.0 * TANK_B / (np.pi**3 * H_LIQUID)) * tanh_y) # ~0.3820 kg
m_0y = M_LIQUID - m_1y                                    # ~0.8180 kg
beta_y = 1.0 / ly
k_th_y = omega_y**2
hd_y = float(2.0 * ZETA_SLOSH * omega_y)

# Giới hạn góc sóng sánh an toàn (rad)
# Mép thành tự do: Freeboard = TANK_H - H_LIQUID = 110.0 - 80.0 = 30.0 mm
PHI_MAX_X = 0.25        # rad (~14.3 deg, sóng dâng eta_x ~ 19.1 mm << mép thành 30 mm)
PHI_MAX_Y = 0.18        # rad (~10.3 deg, sóng dâng eta_y ~ 9.3 mm << mép thành 30 mm)

# Cực cho Exponential CBF bậc 3 (p1, p2, p3 > 0)
POLE_SLOSH_X = (16.0, 16.0, 16.0)
K_SLOSH_2_X = POLE_SLOSH_X[0] + POLE_SLOSH_X[1] + POLE_SLOSH_X[2]
K_SLOSH_1_X = (POLE_SLOSH_X[0]*POLE_SLOSH_X[1] + POLE_SLOSH_X[1]*POLE_SLOSH_X[2] + POLE_SLOSH_X[2]*POLE_SLOSH_X[0])
K_SLOSH_0_X = POLE_SLOSH_X[0] * POLE_SLOSH_X[1] * POLE_SLOSH_X[2]

POLE_SLOSH_Y = (20.0, 20.0, 20.0)
K_SLOSH_2_Y = POLE_SLOSH_Y[0] + POLE_SLOSH_Y[1] + POLE_SLOSH_Y[2]
K_SLOSH_1_Y = (POLE_SLOSH_Y[0]*POLE_SLOSH_Y[1] + POLE_SLOSH_Y[1]*POLE_SLOSH_Y[2] + POLE_SLOSH_Y[2]*POLE_SLOSH_Y[0])
K_SLOSH_0_Y = POLE_SLOSH_Y[0] * POLE_SLOSH_Y[1] * POLE_SLOSH_Y[2]

POLE_OBS = (6.0, 8.0, 10.0)
K_OBS_2 = POLE_OBS[0] + POLE_OBS[1] + POLE_OBS[2]
K_OBS_1 = (POLE_OBS[0]*POLE_OBS[1] + POLE_OBS[1]*POLE_OBS[2] + POLE_OBS[2]*POLE_OBS[0])
K_OBS_0 = POLE_OBS[0] * POLE_OBS[1] * POLE_OBS[2]


# ===========================================================================
# XÂY DỰNG MÔ HÌNH SYMBOLIC CASADI CHO XE AMR 3-DOF
# ===========================================================================
def build_amr_symbolic_model():
    """
    Xây dựng toàn bộ các biểu thức giải tích và tạo hàm CasADi cho AMR mang bình cố định.
    Returns: dictionary chứa các CasADi Functions.
    """
    # -------------------------------------------------------------
    # 1. BIẾN SYMBOLIC TRẠNG THÁI VÀ ĐIỀU KHIỂN
    # -------------------------------------------------------------
    # Biến khớp AMR 3-DoF: [x_b, y_b, theta_b]^T
    q = cs.SX.sym('q', 3)
    v = cs.SX.sym('v', 3)       # Vận tốc World Frame [vx, vy, omega_z]
    a = cs.SX.sym('a', 3)       # Gia tốc khớp [ax, ay, alpha_z]
    u = cs.SX.sym('u', 3)       # Jerk điều khiển [jx, jy, j_theta]

    # Trạng thái sóng sánh ước lượng từ EKF: [phi_x, dphi_x, phi_y, dphi_y]
    phi_est = cs.SX.sym('phi_est', 4)

    # Biến tiến độ MPCC: s (quãng đường), v_s (vận tốc), a_s (gia tốc), j_s (jerk tiến độ ảo)
    s_prog = cs.SX.sym('s_prog', 1)
    vs_prog = cs.SX.sym('vs_prog', 1)
    as_prog = cs.SX.sym('as_prog', 1)
    js_prog = cs.SX.sym('js_prog', 1)

    # Điểm tham chiếu MPCC dọc đường cong: gamma(s) = [xr, yr]^T và dgamma(s) = [dxr, dyr]^T
    gamma_s = cs.SX.sym('gamma_s', 2)
    dgamma_s = cs.SX.sym('dgamma_s', 2)

    # Chướng ngại vật: [ox, oy, oz, r_obs, d_safe]
    obs_param = cs.SX.sym('obs_param', 5)

    # -------------------------------------------------------------
    # 2. VỊ TRÍ VÀ HỆ QUY CHIẾU THÂN XE (BASE / TANK FRAME)
    # -------------------------------------------------------------
    xb, yb, th_b = q[0], q[1], q[2]
    pos_base = cs.vertcat(xb, yb)

    c_th, s_th = cs.cos(th_b), cs.sin(th_b)
    # Ma trận quay từ World Frame sang Body Frame (R_world_to_body = R^T):
    # a_body = R^T * a_world
    a_bx = c_th * a[0] + s_th * a[1]
    a_by = -s_th * a[0] + c_th * a[1]
    a_body = cs.vertcat(a_bx, a_by)

    # Đạo hàm của gia tốc thân xe theo thời gian: dot(a_body)
    # d/dt (a_body) = d/dt(R^T) * a_world + R^T * u_world
    # d/dt(c_th) = -s_th * v[2], d/dt(s_th) = c_th * v[2]
    dot_a_bx = -s_th * v[2] * a[0] + c_th * u[0] + c_th * v[2] * a[1] + s_th * u[1]
    dot_a_by = -c_th * v[2] * a[0] - s_th * u[0] - s_th * v[2] * a[1] + c_th * u[1]
    dot_a_body = cs.vertcat(dot_a_bx, dot_a_by)

    # -------------------------------------------------------------
    # 3. ĐỘNG HỌC NGƯỢC 4 BÁNH XE MECANUM
    # -------------------------------------------------------------
    v_bx = c_th * v[0] + s_th * v[1]
    v_by = -s_th * v[0] + c_th * v[1]
    w_bz = v[2]

    w1 = (1.0 / R_W) * (v_bx - v_by - L_SUM * w_bz)  # FL
    w2 = (1.0 / R_W) * (v_bx + v_by + L_SUM * w_bz)  # FR
    w3 = (1.0 / R_W) * (v_bx - v_by + L_SUM * w_bz)  # RR
    w4 = (1.0 / R_W) * (v_bx + v_by - L_SUM * w_bz)  # RL
    omega_mecanum = cs.vertcat(w1, w2, w3, w4)

    # -------------------------------------------------------------
    # 4. SAI SỐ HÌNH HỌC MPCC (CONTOURING & LAG ERRORS)
    # -------------------------------------------------------------
    norm_dgamma = cs.sqrt(cs.sumsqr(dgamma_s) + 1e-12)
    t_tangent = dgamma_s / norm_dgamma                      # Vector tiếp tuyến đơn vị [tx, ty]^T
    n_normal = cs.vertcat(-t_tangent[1], t_tangent[0])      # Vector pháp tuyến đơn vị [-ty, tx]^T

    err_xy = pos_base - gamma_s
    eps_contour = n_normal[0] * err_xy[0] + n_normal[1] * err_xy[1]
    eps_lag = t_tangent[0] * err_xy[0] + t_tangent[1] * err_xy[1]
    mpcc_errors = cs.vertcat(eps_contour, eps_lag)

    # -------------------------------------------------------------
    # 5. SLOSH-ECBF BẬC TƯƠNG ĐỐI r = 3 TRÊN HỆ THÂN XE
    # -------------------------------------------------------------
    phi_x = phi_est[0]
    dphi_x = phi_est[1]
    phi_y = phi_est[2]
    dphi_y = phi_est[3]

    # Động lực học con lắc Housner kích thích bởi gia tốc thân xe:
    # Với slosh3.xml: joint slosh_x quay quanh trục +Y -> khi a_bx > 0 thì ddphi_x > 0 (+beta_x)
    # joint slosh_y quay quanh trục +X -> khi a_by > 0 thì ddphi_y < 0 (-beta_y)
    ddphi_x = -k_th_x * phi_x - hd_x * dphi_x + beta_x * a_bx
    ddphi_y = -k_th_y * phi_y - hd_y * dphi_y - beta_y * a_by

    dddphi_x = -k_th_x * dphi_x - hd_x * ddphi_x + beta_x * dot_a_bx
    dddphi_y = -k_th_y * dphi_y - hd_y * ddphi_y - beta_y * dot_a_by

    # Ràng buộc r = 3 trục X:
    B3_x_pos = -dddphi_x + K_SLOSH_2_X * (-ddphi_x) + K_SLOSH_1_X * (-dphi_x) + K_SLOSH_0_X * (PHI_MAX_X - phi_x)
    B3_x_neg = dddphi_x + K_SLOSH_2_X * ddphi_x + K_SLOSH_1_X * dphi_x + K_SLOSH_0_X * (PHI_MAX_X + phi_x)

    # Ràng buộc r = 3 trục Y:
    B3_y_pos = -dddphi_y + K_SLOSH_2_Y * (-ddphi_y) + K_SLOSH_1_Y * (-dphi_y) + K_SLOSH_0_Y * (PHI_MAX_Y - phi_y)
    B3_y_neg = dddphi_y + K_SLOSH_2_Y * ddphi_y + K_SLOSH_1_Y * dphi_y + K_SLOSH_0_Y * (PHI_MAX_Y + phi_y)

    slosh_cbf_vec = cs.vertcat(B3_x_pos, B3_x_neg, B3_y_pos, B3_y_neg)

    # -------------------------------------------------------------
    # 6. OBSTACLE-ECBF BẬC TƯƠNG ĐỐI r = 3 CHO THÂN XE AMR
    # -------------------------------------------------------------
    obs_xy = obs_param[0:2]
    r_obs = obs_param[3]
    d_safe = obs_param[4]
    r_total = ROBOT_RADIUS + r_obs + d_safe

    diff_obs = pos_base - obs_xy
    h_obs = cs.sumsqr(diff_obs) - r_total**2

    dot_h_obs = 2.0 * cs.dot(diff_obs, v[0:2])
    ddot_h_obs = 2.0 * cs.sumsqr(v[0:2]) + 2.0 * cs.dot(diff_obs, a[0:2])
    dddot_h_obs = 6.0 * cs.dot(v[0:2], a[0:2]) + 2.0 * cs.dot(diff_obs, u[0:2])

    B3_obs = dddot_h_obs + K_OBS_2 * ddot_h_obs + K_OBS_1 * dot_h_obs + K_OBS_0 * h_obs

    # -------------------------------------------------------------
    # 7. LỰC PHẢN KHÁNG FSI VÀ TỌA ĐỘ ZMP (CHUẨN HÓA KHÔNG TÍNH TRÙNG QUÁN TÍNH)
    # -------------------------------------------------------------
    # Gia tốc tương đối của quả cầu con lắc so với bệ xe (Body frame):
    # Trục X: x_pend = -lx * sin(phi_x) => a_rel_x = -lx*(ddphi_x*cos(phi_x) - dphi_x^2*sin(phi_x))
    # Trục Y: y_pend = +ly * sin(phi_y) => a_rel_y = +ly*(ddphi_y*cos(phi_y) - dphi_y^2*sin(phi_y))
    a_rel_pend_x = -lx * (ddphi_x * cs.cos(phi_x) - dphi_x**2 * cs.sin(phi_x))
    a_rel_pend_y =  ly * (ddphi_y * cs.cos(phi_y) - dphi_y**2 * cs.sin(phi_y))

    # Lực phản kháng dao động sóng sánh thuần túy (Convective Sloshing Residual Force):
    # F_slosh = -m_1 * a_rel_pend (tác dụng lên bệ xe)
    F_slosh_bx = -m_1x * a_rel_pend_x  # = +m_1x * lx * (ddphi_x*cos(phi_x) - dphi_x^2*sin(phi_x))
    F_slosh_by = -m_1y * a_rel_pend_y  # = -m_1y * ly * (ddphi_y*cos(phi_y) - dphi_y^2*sin(phi_y))
    F_slosh_body = cs.vertcat(F_slosh_bx, F_slosh_by)

    # Chuyển F_slosh về World Frame:
    F_slosh_x = c_th * F_slosh_bx - s_th * F_slosh_by
    F_slosh_y = s_th * F_slosh_bx + c_th * F_slosh_by
    F_slosh_world = cs.vertcat(F_slosh_x, F_slosh_y)

    # Lực thủy động toàn phần tác động lên bình (Total Hydrodynamic Force on Tank):
    # F_total_fluid = -M_liquid * a_body + F_slosh_body
    F_total_fluid_bx = -M_LIQUID * a_bx + F_slosh_bx
    F_total_fluid_by = -M_LIQUID * a_by + F_slosh_by
    F_total_fluid_x = c_th * F_total_fluid_bx - s_th * F_total_fluid_by
    F_total_fluid_y = s_th * F_total_fluid_bx + c_th * F_total_fluid_by
    F_total_fluid_world = cs.vertcat(F_total_fluid_x, F_total_fluid_y)

    # Tọa độ ZMP giám sát ổn định lật trên sàn (không tính trùng khối lượng nước):
    # Cao độ tâm áp lực FSI từ lý thuyết Housner h1* (bao gồm áp suất đáy bình và cao độ sàn z_floor = 0.052 m):
    # Với h = 80mm: h1*_x = 0.0658m, h1*_y = 0.0581m; z_floor = 0.052m
    z_slosh_x = 0.052 + 0.0658  # 0.1178 m
    z_slosh_y = 0.052 + 0.0581  # 0.1101 m
    F_z_total = M_TOTAL * g
    # Mô-men lật tổng hợp quanh điểm tiếp xúc sàn (bao gồm mô-men quán tính, FSI và trọng lực con lắc lệch trục):
    # - Trục X: con lắc x quay quanh +Y, x_pend = -lx*sin(phi_x). Khi a_bx > 0 thì phi_x > 0 => số hạng + m_1x*g*lx*sin(phi_x)
    # - Trục Y: con lắc y quay quanh +X, y_pend = +ly*sin(phi_y). Khi a_by > 0 thì phi_y < 0 => số hạng - m_1y*g*ly*sin(phi_y)
    M_pitch_ground = M_TOTAL * a_bx * Z_COM - F_slosh_bx * z_slosh_x + m_1x * g * lx * cs.sin(phi_x)
    M_roll_ground  = M_TOTAL * a_by * Z_COM - F_slosh_by * z_slosh_y - m_1y * g * ly * cs.sin(phi_y)
    x_zmp = xb - M_pitch_ground / (F_z_total + 1e-6)
    y_zmp = yb - M_roll_ground  / (F_z_total + 1e-6)
    zmp_coords = cs.vertcat(x_zmp, y_zmp)

    # -------------------------------------------------------------
    # 8. ĐÓNG GÓI HÀM CASADI
    # -------------------------------------------------------------
    fn_pos_base = cs.Function('pos_base', [q], [pos_base], ['q'], ['pos_base'])
    fn_body_acc = cs.Function('body_acc', [q, a], [a_body], ['q', 'a'], ['a_body'])
    fn_body_jerk = cs.Function('body_jerk', [q, v, a, u], [dot_a_body], ['q', 'v', 'a', 'u'], ['dot_a_body'])
    fn_mecanum = cs.Function('mecanum_wheels', [q, v], [omega_mecanum], ['q', 'v'], ['omega_mecanum'])
    fn_mpcc_err = cs.Function('mpcc_err', [q, gamma_s, dgamma_s], [mpcc_errors],
                              ['q', 'gamma_s', 'dgamma_s'], ['mpcc_errors'])
    fn_slosh_cbf = cs.Function('slosh_cbf', [q, v, a, u, phi_est], [slosh_cbf_vec],
                               ['q', 'v', 'a', 'u', 'phi_est'], ['slosh_cbf_vec'])
    fn_obs_cbf = cs.Function('obs_cbf', [q, v, a, u, obs_param], [B3_obs],
                             ['q', 'v', 'a', 'u', 'obs_param'], ['B3_obs'])
    fn_slosh_forces = cs.Function('slosh_forces', [q, v, a, phi_est], [F_slosh_world, F_slosh_body],
                                  ['q', 'v', 'a', 'phi_est'], ['F_slosh_world', 'F_slosh_body'])
    fn_fsi = cs.Function('fsi_forces', [q, v, a, phi_est], [F_total_fluid_world, zmp_coords],
                         ['q', 'v', 'a', 'phi_est'], ['F_total_fluid_world', 'zmp_coords'])

    functions_dict = {
        'pos_base': fn_pos_base,
        'body_acc': fn_body_acc,
        'body_jerk': fn_body_jerk,
        'mecanum': fn_mecanum,
        'mecanum_wheels': fn_mecanum,
        'mpcc_err': fn_mpcc_err,
        'slosh_cbf': fn_slosh_cbf,
        'obs_cbf': fn_obs_cbf,
        'slosh_forces': fn_slosh_forces,
        'fsi_forces': fn_fsi
    }

    return functions_dict


if __name__ == "__main__":
    print("=================================================================")
    print("KIỂM CHỨNG MÔ HÌNH SYMBOLIC CASADI CHO XE AMR (slosh3.xml)")
    print("=================================================================")
    funcs = build_amr_symbolic_model()
    print("Tạo thành công các hàm CasADi SX:")
    for name, f in funcs.items():
        print(f"  - {name}: {f.n_in()} inputs -> {f.n_out()} outputs")

    # Kiểm tra số học mẫu
    q_test = np.array([0.5, -0.2, 0.3])
    v_test = np.array([0.4, 0.1, 0.2])
    a_test = np.array([0.2, -0.1, 0.05])
    u_test = np.array([0.5, 0.2, 0.1])
    phi_test = np.array([0.02, 0.01, -0.01, 0.005])

    w_wheels = funcs['mecanum_wheels'](q_test, v_test).full().ravel()
    print("\nTốc độ góc bánh xe Mecanum (rad/s):", np.round(w_wheels, 2))

    b3_slosh = funcs['slosh_cbf'](q_test, v_test, a_test, u_test, phi_test).full().ravel()
    print("Giá trị Slosh-ECBF bậc 3 [X+, X-, Y+, Y-]:", np.round(b3_slosh, 2))
    print("Mô hình AMR CasADi đã sẵn sàng sử dụng!")
