"""
mecanum_traction_dynamics.py
=============================================================================
Mô đun Tính toán Động lực học Phân bố Tải trọng Pháp tuyến (Normal Forces),
Giới hạn Bám Ma sát Coulomb (Traction Limits) và Trượt Vi mô (Wheel Slip)
cho Xe Tự Hành AMR Mecanum 4 Bánh Mang Bình Chất Lỏng (Housner DMTP).

Phục vụ Kịch bản Nghiên cứu Test 3:
"Stress-Testing Baseline MPCC under Dynamic Wheel Load Transfer & Traction Limits"
=============================================================================
"""

import numpy as np

# Thông số hình học và cơ học khung xe Mecanum (từ slosh3.xml & robot_model_amr_casadi.py)
L_X = 0.12              # Nửa khoảng cách cơ sở bánh xe theo X (m) (2*Lx = 0.24m)
L_Y = 0.14              # Nửa khoảng cách cơ sở bánh xe theo Y (m) (2*Ly = 0.28m)
L_SUM = L_X + L_Y       # 0.26 m
R_W = 0.04              # Bán kính bánh xe Mecanum (m)
G = 9.81                # Gia tốc trọng trường (m/s^2)

# Khối lượng và kích thước hệ thống AMR + Bình chất lỏng
M_TOTAL = 7.250         # Tổng khối lượng toàn hệ (kg)
M_CART = 5.500          # Khối lượng xe cơ sở (kg)
M_SHELL = 0.550         # Khối lượng vỏ bình mica (kg)
M_LIQUID = 1.200        # Khối lượng nước (kg)

# Cao độ gốc tọa độ tính từ mặt đất tiếp xúc bánh xe (z = 0):
# - Bệ xe cart đặt tại z_cart = 0.040 m (bán kính bánh xe R_w = 0.040 m)
# - Đáy bình đặt tại z_floor = z_cart + 0.012 = 0.052 m
Z_CART = 0.040          # m (khối tâm bệ xe)
Z_FLOOR = 0.052         # m (mặt đáy bình chất lỏng)

# Cao độ khối tâm danh định của toàn hệ thống AMR + Bình + Nước (z = 0 tại mặt sàn):
# Trích xuất trực tiếp từ cây động học MuJoCo model (data.subtree_com[0] tại phi = 0):
# Z_COM = 0.05549 m ≈ 0.0555 m. Khối tâm hình học tổng thể là duy nhất cho toàn xe.
Z_COM = 0.0555          # m (khối tâm toàn hệ từ mặt sàn z = 0)
Z_COM_X = Z_COM         # Thống nhất duy nhất một giá trị vật lý
Z_COM_Y = Z_COM

# Cao độ hiệu dụng tâm áp lực FSI từ lý thuyết Housner h1* (bao gồm áp suất đáy bình):
# h = 80mm: h1*_x = 0.0658m, h1*_y = 0.0581m
Z_SLOSH_X = Z_FLOOR + 0.0658  # 0.1178 m
Z_SLOSH_Y = Z_FLOOR + 0.0581  # 0.1101 m
Z_SLOSH = (Z_SLOSH_X + Z_SLOSH_Y) / 2.0 # 0.1140 m

# Tham số con lắc Housner DMTP danh định (mực nước h = 80 mm)
M_1X = 0.5412           # Khối lượng dao động trục X (kg)
M_1Y = 0.3820           # Khối lượng dao động trục Y (kg)
L_P_X = 0.0512          # Chiều dài con lắc trục X (m)
L_P_Y = 0.0323          # Chiều dài con lắc trục Y (m)

# Ma trận phân phối lực Mecanum B (3 x 4)
B_MECANUM = np.array([
    [1.0,  1.0,  1.0,  1.0],
    [-1.0, 1.0, -1.0,  1.0],
    [-L_SUM, L_SUM, L_SUM, -L_SUM]
])

# Vị trí tâm 4 bánh xe trong Body Frame (m)
# Bánh 1: Front-Left (FL), Bánh 2: Front-Right (FR)
# Bánh 3: Rear-Right (RR),  Bánh 4: Rear-Left (RL)
WHEEL_POS = np.array([
    [ L_X,  L_Y],   # FL
    [ L_X, -L_Y],   # FR
    [-L_X, -L_Y],   # RR
    [-L_X,  L_Y]    # RL
])


def compute_wheel_friction_limits(Fz_wheels, mu=0.35):
    """Giới hạn bám ma sát 2D tiếp xúc mặt sàn: F_max = mu * Fz."""
    return mu * np.asarray(Fz_wheels)


def compute_actuator_force_limits_1d(Fz_wheels, mu=0.35):
    """Giới hạn lực con lăn bánh xe 1D: f_bar = (mu / sqrt(2)) * Fz."""
    return (mu / np.sqrt(2.0)) * np.asarray(Fz_wheels)


def compute_dynamic_normal_loads(m_total, a_body, f_slosh_body=None, phi_body=None, z_com=None, z_slosh=None, fz_min=0.889):
    """
    Tính toán tải trọng pháp tuyến thời gian thực tác dụng lên 4 bánh xe Mecanum.
    Chuẩn hóa theo nguyên lý D'Alembert & Housner DMTP:
    1. Trọng lượng tĩnh danh định: Fz_static = M_total * g / 4 (giả thiết khung xe đối xứng, sàn phẳng C2).
    2. Mô-men lật do quán tính toàn hệ: M_inertial = M_total * a_body * z_com (Z_COM = 0.0555 m).
    3. Mô-men lật do phản lực sóng sánh dư FSI: M_slosh = -F_slosh * z_slosh (z_slosh tính từ chiều cao Housner h1* và đáy bình z_floor = 0.052 m).
    4. Mô-men lật do trọng lực quả nặng con lắc bị lệch góc: M_grav_pend = m1 * g * l * sin(phi).
    5. Bảo toàn tổng tải trọng sum(Fz,i) = M_total * g ngay cả khi có bánh chạm ngưỡng Fz_min.

    Parameters:
        m_total: Tổng khối lượng xe + vỏ bình + 100% chất lỏng (kg) (danh định 7.25 kg).
        a_body: Vector gia tốc thân xe [a_bx, a_by] trong Body frame (m/s^2).
        f_slosh_body: Vector lực sóng sánh dư FSI tác dụng lên thân xe [Fx, Fy] (N).
        phi_body: Góc lắc con lắc [phi_x, phi_y] (rad). Mặc định None (coi phi = 0).
        z_com: Cao độ trọng tâm danh định của toàn hệ (m) (mặc định Z_COM = 0.0555 m).
        z_slosh: Cao độ tâm áp lực sóng sánh FSI (m), có thể là float hoặc tuple (zs_x, zs_y).
        fz_min: Ngưỡng tải trọng pháp tuyến tối thiểu chống nhấc bánh hoàn toàn (N).

    Returns:
        Fz: Mảng 4 phần tử [Fz_FL, Fz_FR, Fz_RR, Fz_RL] (N) bảo toàn tổng trọng lượng.
        M_tip: Tuple (M_pitch, M_roll) mô-men lật dọc và lật ngang toàn phần (N.m).
    """
    if z_com is None:
        zc_x, zc_y = Z_COM_X, Z_COM_Y
    elif isinstance(z_com, (tuple, list, np.ndarray)):
        zc_x, zc_y = float(z_com[0]), float(z_com[1])
    else:
        zc_x = zc_y = float(z_com)

    if f_slosh_body is None:
        f_slosh_body = np.zeros(2)
    else:
        f_slosh_body = np.asarray(f_slosh_body)

    if phi_body is None:
        phi_body = np.zeros(2)
    else:
        phi_body = np.asarray(phi_body)

    if z_slosh is None:
        zs_x, zs_y = Z_SLOSH_X, Z_SLOSH_Y
    elif isinstance(z_slosh, (tuple, list, np.ndarray)):
        zs_x, zs_y = z_slosh[0], z_slosh[1]
    else:
        zs_x = zs_y = float(z_slosh)

    # 1. Trọng lượng tĩnh chia đều 4 bánh (giả thiết khung đàn hồi đối xứng, sàn phẳng tuyệt đối)
    Fz_static = (m_total * G) / 4.0

    # 2. Mô-men trọng lực bổ sung do quả nặng con lắc lệch khỏi trục đứng:
    # m1x ~ 0.5412 kg, lx ~ 0.0512 m -> m1x * g * lx ~ 0.2718 N.m
    # m1y ~ 0.3820 kg, ly ~ 0.0323 m -> m1y * g * ly ~ 0.1210 N.m
    # Quy ước trục slosh3.xml:
    # - Trục Y: joint slosh_y quay quanh +X -> y_pend = +l_y * sin(phi_y).
    #   Khi a_by > 0 (xe cua trái), nước dồn sang phải (y < 0 => phi_y < 0).
    #   Trọng lượng nước dồn sang phải phải làm TĂNG tải bánh phải (FR, RR) => yêu cầu M_roll > 0.
    #   Do phi_y < 0, số hạng mô-men trọng lực BẮT BUỘC mang dấu TRỪ: - m1y * g * ly * sin(phi_y).
    # - Trục X: joint slosh_x quay quanh +Y -> x_pend = -l_x * sin(phi_x).
    #   Khi a_bx > 0 (tăng tốc tới), nước dồn ra sau (x < 0 => phi_x > 0).
    #   Trọng lượng dồn ra sau làm TĂNG tải bánh sau (RR, RL) => yêu cầu M_pitch > 0.
    #   Do phi_x > 0, số hạng mô-men mang dấu CỘNG: + m1x * g * lx * sin(phi_x).
    M_grav_pend_x = 0.5412 * G * 0.0512 * np.sin(phi_body[0])
    M_grav_pend_y = - 0.3820 * G * 0.0323 * np.sin(phi_body[1])

    # 3. Mô-men lật dọc (Pitch Moment quanh trục Y tiếp xúc sàn):
    # a_bx > 0 đẩy lùi sau -> đè bánh sau, nhấc bánh trước
    M_pitch = m_total * a_body[0] * zc_x - f_slosh_body[0] * zs_x + M_grav_pend_x

    # 4. Mô-men lật ngang (Roll Moment quanh trục X tiếp xúc sàn):
    # a_by > 0 đẩy sang phải -> đè bánh phải, nhấc bánh trái
    M_roll = m_total * a_body[1] * zc_y - f_slosh_body[1] * zs_y + M_grav_pend_y

    # 5. Phân bổ tuyến tính lên 4 bánh xe:
    dFz_pitch = M_pitch / (4.0 * L_X)
    dFz_roll  = M_roll  / (4.0 * L_Y)

    Fz_raw = np.array([
        Fz_static - dFz_pitch - dFz_roll, # FL (Trước-Trái)
        Fz_static - dFz_pitch + dFz_roll, # FR (Trước-Phải)
        Fz_static + dFz_pitch + dFz_roll, # RR (Sau-Phải)
        Fz_static + dFz_pitch - dFz_roll  # RL (Sau-Trái)
    ])

    # 6. Kẹp ngưỡng chặn dưới Fz_min và tái phân bổ để bảo toàn tổng lực Fz_total = m_total * G
    Fz = Fz_raw.copy()
    if np.any(Fz < fz_min):
        clipped_mask = Fz < fz_min
        Fz[clipped_mask] = fz_min
        # Lượng thiếu hụt cần bù trừ
        excess = np.sum(Fz) - (m_total * G)
        if excess > 0 and np.any(~clipped_mask):
            unclipped_count = np.sum(~clipped_mask)
            Fz[~clipped_mask] -= excess / unclipped_count
            # Đảm bảo không bị kéo lùi dưới fz_min
            Fz = np.maximum(fz_min, Fz)

    return Fz, (M_pitch, M_roll)


def allocate_chassis_forces_to_wheels(f_cmd_body):
    """
    Phân bổ lực yêu cầu thân xe [F_bx, F_by, Tau_bz] thành lực kéo tiếp tuyến
    yêu cầu của 4 bánh xe Mecanum (con lăn nghiêng 45 độ) qua ma trận Moore-Penrose Pseudoinverse.

    Quan hệ động lực học:
        F_body = B * f_wheels
        với B = [ 1,  1,  1,  1 ]
                [-1,  1, -1,  1 ]
                [-L,  L,  L, -L ] (với L = L_X + L_Y)

    Parameters:
        f_cmd_body: [F_bx, F_by, Tau_bz] (N, N, N.m).

    Returns:
        f_wheels_cmd: Mảng 4 phần tử [f1, f2, f3, f4] (N) cho 4 bánh xe.
    """
    # Ma trận phân phối hình học Mecanum B (3 x 4)
    B = np.array([
        [1.0,  1.0,  1.0,  1.0],
        [-1.0, 1.0, -1.0,  1.0],
        [-L_SUM, L_SUM, L_SUM, -L_SUM]
    ])

    # Ma trận giả nghịch đảo tối thiểu chuẩn bậc hai (Pseudoinverse)
    B_pinv = np.linalg.pinv(B)
    f_wheels_cmd = B_pinv @ f_cmd_body
    return f_wheels_cmd


def evaluate_traction_and_slip(f_wheels_cmd, Fz_wheels, mu=0.35):
    """
    Kiểm tra giới hạn bám ma sát Coulomb của từng bánh xe Mecanum và mô phỏng
    sự bão hòa lực tiếp xúc (Traction Saturation / Micro-slip).

    Parameters:
        f_wheels_cmd: Lực tiếp tuyến yêu cầu từ controller [f1, f2, f3, f4] (N).
        Fz_wheels: Tải trọng pháp tuyến tức thời [Fz1, Fz2, Fz3, Fz4] (N).
        mu: Hệ số ma sát bám giữa con lăn Mecanum và mặt sàn (danh định 0.35, sàn ướt 0.25).

    Returns:
        f_wheels_act: Lực tiếp tuyến thực tế sinh ra sau khi bị bão hòa bởi giới hạn bám (N).
        f_chassis_act: Lực thực tế truyền lên thân xe [F_bx_act, F_by_act, Tau_bz_act] (N, N, N.m).
        slip_metrics: Dict chứa {slip_ratios, max_slip, is_slipping, traction_margins}.
    """
    # 1. Giới hạn bám ma sát cực đại cho từng bánh xe Mecanum (con lăn nghiêng 45 độ):
    # Lực tiếp đất mặt đường: ||F_ground,i|| = sqrt(fx^2 + fy^2) = sqrt(2) * |f_i| <= mu * Fz_i
    # => |f_i| <= (mu / sqrt(2)) * Fz_wheels (Chuẩn hóa ma sát thực tế)
    f_max_traction = (mu / np.sqrt(2.0)) * Fz_wheels

    # 2. Bão hòa lực bánh xe (Traction Saturation):
    f_wheels_act = np.clip(f_wheels_cmd, -f_max_traction, f_max_traction)

    # 3. Tính độ trượt bánh (Slip Magnitude & Slip Ratio):
    f_demand_abs = np.abs(f_wheels_cmd)
    traction_deficit = np.maximum(0.0, f_demand_abs - f_max_traction)
    slip_ratios = traction_deficit / (f_demand_abs + 1e-6) # 0.0 = không trượt, >0.0 = đang trượt
    is_slipping = (traction_deficit > 1e-3)
    max_slip = float(np.max(slip_ratios))

    # 4. Dự trữ bám (Traction Margin) còn lại của từng bánh (N):
    traction_margins = f_max_traction - f_demand_abs

    # 5. Lực thực tế tác dụng lên thân xe từ 4 bánh xe:
    B = np.array([
        [1.0,  1.0,  1.0,  1.0],
        [-1.0, 1.0, -1.0,  1.0],
        [-L_SUM, L_SUM, L_SUM, -L_SUM]
    ])
    f_chassis_act = B @ f_wheels_act

    slip_metrics = {
        'f_max_traction': f_max_traction,
        'traction_deficit': traction_deficit,
        'slip_ratios': slip_ratios,
        'is_slipping': is_slipping,
        'max_slip': max_slip,
        'traction_margins': traction_margins,
        'any_slip': np.any(is_slipping)
    }

    return f_wheels_act, f_chassis_act, slip_metrics


def compute_feasible_alpha_interval(W, f_bar, L=L_SUM):
    """
    Tính khoảng giá trị khả thi [alpha_lo, alpha_hi] cho biến không gian null alpha
    của ma trận Mecanum B sao cho mọi lực bánh xe thỏa mãn:
        |f_i| <= f_bar_i  (i = 1, 2, 3, 4)
    với f_i = f_pinv_i + alpha * n_i và vector null-space n = [1, 1, -1, -1]^T.

    Xét đầy đủ 4 bánh xe và đổi chiều bất đẳng thức khi n_i < 0:
    - Bánh 1, 2 (n_i = +1 > 0):
        alpha >= (-f_bar_i - f_pinv_i) / n_i
        alpha <= ( f_bar_i - f_pinv_i) / n_i
    - Bánh 3, 4 (n_i = -1 < 0): Chia cho số âm đảo chiều:
        alpha >= ( f_bar_i - f_pinv_i) / n_i
        alpha <= (-f_bar_i - f_pinv_i) / n_i

    Returns:
        alpha_lo (float): Cận dưới lớn nhất.
        alpha_hi (float): Cận trên nhỏ nhất.
        is_empty (bool): True nếu alpha_lo > alpha_hi (Wrench nằm ngoài Zonotope).
    """
    B = np.array([
        [1.0,  1.0,  1.0,  1.0],
        [-1.0, 1.0, -1.0,  1.0],
        [-L, L, L, -L]
    ], dtype=float)
    f_pinv = np.linalg.pinv(B) @ np.asarray(W, dtype=float)
    n_vec = np.array([1.0, 1.0, -1.0, -1.0], dtype=float)

    lo_bounds = []
    hi_bounds = []
    for i in range(4):
        if n_vec[i] > 0.0:
            lo_bounds.append((-f_bar[i] - f_pinv[i]) / n_vec[i])
            hi_bounds.append(( f_bar[i] - f_pinv[i]) / n_vec[i])
        else:
            lo_bounds.append(( f_bar[i] - f_pinv[i]) / n_vec[i])
            hi_bounds.append((-f_bar[i] - f_pinv[i]) / n_vec[i])

    alpha_lo = float(np.max(lo_bounds))
    alpha_hi = float(np.min(hi_bounds))
    is_empty = alpha_lo > alpha_hi
    return alpha_lo, alpha_hi, f_pinv, n_vec, is_empty


def allocate_wheel_forces_clamped(W, f_bar, Fz=None, mu=None, L=L_SUM):
    """
    Phân bổ lực 4 bánh xe từ Wrench thân xe W với bảo đảm ràng buộc ma sát hộp.
    Tính alpha* theo nghiệm bình phương tối thiểu có trọng số (weighted least-squares)
    hoặc unweighted, sau đó kẹp vào đoạn khả thi [alpha_lo, alpha_hi].
    """
    alpha_lo, alpha_hi, f_pinv, n_vec, is_empty = compute_feasible_alpha_interval(W, f_bar, L)
    if Fz is not None and mu is not None:
        weights = 1.0 / np.maximum(1e-4, (mu * np.asarray(Fz))**2)
        denom = float(np.sum(weights * (n_vec**2)))
        numer = float(np.sum(weights * n_vec * f_pinv))
        alpha_star = -numer / denom if denom > 1e-12 else 0.0
    else:
        alpha_star = 0.0

    if not is_empty:
        alpha_clamped = float(np.clip(alpha_star, alpha_lo, alpha_hi))
    else:
        alpha_clamped = (alpha_lo + alpha_hi) / 2.0

    f_alloc = f_pinv + alpha_clamped * n_vec
    return f_alloc, alpha_clamped, (alpha_lo, alpha_hi, alpha_star)

