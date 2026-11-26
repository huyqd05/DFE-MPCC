"""
mecanum_nullspace_allocation.py
=============================================================================
ĐÓNG GÓP LÝ THUYẾT: BỔ ĐỀ PHÂN BỔ KHÔNG GIAN NULL CÓ LỜI GIẢI ĐÓNG O(1)
(Closed-Form Null-Space Optimal Allocation for Omnidirectional Mecanum Robots)

Cung cấp:
1. Ma trận cấu hình lực Mecanum B (3 x 4) và vector cơ sở không gian null n.
2. Công thức giải tích đóng duy nhất alpha*(w, Fz, mu) tối ưu hóa biên ma sát
   trong thời gian O(1) (< 1 microsecond, không cần solver số học).
3. So sánh định lượng giữa Pseudoinverse thuần túy (pinv) và Null-space Allocation.
=============================================================================
"""

import numpy as np
import time
import sys

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Kích thước hình học robot AMR (slosh3.xml)
L_X = 0.12              # Nửa khoảng cách cơ sở X (m) (chiều dài cơ sở 0.24m)
L_Y = 0.14              # Nửa khoảng cách cơ sở Y (m) (chiều rộng cơ sở 0.28m)
L_SUM = L_X + L_Y       # 0.26 m
R_W = 0.04              # Bán kính bánh xe (m)
G = 9.81                # Gia tốc trọng trường (m/s^2)

# Ma trận phân phối lực thân xe B (3 x 4):
# [Fx, Fy, Mz]^T = B * [f1, f2, f3, f4]^T
# Wheel 1: FL, Wheel 2: FR, Wheel 3: RR, Wheel 4: RL
B_MAT = np.array([
    [ 1.0,    1.0,    1.0,    1.0],
    [-1.0,    1.0,   -1.0,    1.0],
    [-L_SUM,  L_SUM,  L_SUM, -L_SUM]
], dtype=np.float64)

# Ma trận giả nghịch đảo Moore-Penrose (3 x 4)^+ -> (4 x 3)
B_PINV = np.linalg.pinv(B_MAT)

# Vector cơ sở chuẩn hóa của không gian Null: ker(B) = span(NULL_VEC)
# Thỏa mãn chính xác B_MAT @ NULL_VEC = 0
# Bánh trước (FL, FR) mang dấu dương, bánh sau (RR, RL) mang dấu âm
NULL_VEC = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float64)


def compute_optimal_nullspace_alpha(f_pinv, Fz_wheels, mu_wheels):
    """
    Tính toán nghiệm giải tích đóng của hệ số phân bổ không gian null alpha*.
    
    Bài toán tối ưu lồi 1 biến:
        min_alpha sum_{i=1}^4 ( (f_pinv_i + alpha * n_i) / (mu_i * Fz_i) )^2
        
    Đạo hàm dJ/dalpha = 0 dẫn đến nghiệm giải tích duy nhất:
        alpha* = - [ sum_{i=1}^4 (n_i * f_pinv_i) / (mu_i * Fz_i)^2 ] 
                 / [ sum_{i=1}^4 (n_i^2) / (mu_i * Fz_i)^2 ]

    Parameters:
        f_pinv: Vector 4 phần tử [f1, f2, f3, f4] lực bánh từ B_PINV @ w (N).
        Fz_wheels: Tải trọng pháp tuyến 4 bánh [Fz1, Fz2, Fz3, Fz4] (N).
        mu_wheels: Hệ số ma sát từng bánh (scalar hoặc vector 4 phần tử).

    Returns:
        alpha_star: Giá trị vô hướng tối ưu (N).
    """
    if np.isscalar(mu_wheels):
        mu_eff = np.full(4, mu_wheels, dtype=np.float64)
    else:
        mu_eff = np.asarray(mu_wheels, dtype=np.float64)

    # Ngưỡng an toàn chống chia cho 0 nếu có bánh bị nhấc bổng hoàn toàn
    Fz_safe = np.maximum(Fz_wheels, 0.5)
    f_max_cap = mu_eff * Fz_safe
    inv_cap_sq = 1.0 / (f_max_cap ** 2)

    # Tử số và mẫu số của công thức giải tích đóng
    numerator = np.sum(NULL_VEC * f_pinv * inv_cap_sq)
    denominator = np.sum((NULL_VEC ** 2) * inv_cap_sq)

    if denominator < 1e-12:
        return 0.0

    alpha_star = - (numerator / denominator)
    return float(alpha_star)


def allocate_chassis_wrench(w_cmd, Fz_wheels, mu=0.12, use_nullspace=True):
    """
    Phân bổ Wrench thân xe w_cmd = [Fx, Fy, Mz]^T thành 4 lực tiếp tuyến bánh xe.
    
    Parameters:
        w_cmd: [Fx, Fy, Mz] wrench thân xe yêu cầu (N, N, N.m).
        Fz_wheels: Tải trọng pháp tuyến tức thời 4 bánh xe (N).
        mu: Hệ số ma sát mặt sàn (mặc định 0.12).
        use_nullspace: True = dùng alpha* tối ưu; False = dùng Moore-Penrose thuần túy.

    Returns:
        f_wheels: Vector 4 lực tiếp xúc [f1, f2, f3, f4] (N).
        info: Dict chứa {alpha, f_pinv, max_friction_usage_pinv, max_friction_usage_opt, margin_gain_pct}.
    """
    w_cmd = np.asarray(w_cmd, dtype=np.float64)
    f_pinv = B_PINV @ w_cmd

    if not use_nullspace:
        f_opt = f_pinv.copy()
        alpha = 0.0
    else:
        alpha = compute_optimal_nullspace_alpha(f_pinv, Fz_wheels, mu)
        f_opt = f_pinv + alpha * NULL_VEC

    # Đánh giá chỉ số sử dụng ma sát cực đại (Friction Utilization Ratio)
    # rho_i = |f_i| / (mu * Fz_i)  (rho <= 1.0 là an toàn, > 1.0 là trượt bánh)
    Fz_safe = np.maximum(Fz_wheels, 0.1)
    f_cap = mu * Fz_safe

    rho_pinv = np.max(np.abs(f_pinv) / f_cap)
    rho_opt  = np.max(np.abs(f_opt)  / f_cap)
    margin_gain_pct = (rho_pinv - rho_opt) * 100.0

    info = {
        'alpha': alpha,
        'f_pinv': f_pinv,
        'f_opt': f_opt,
        'rho_pinv': float(rho_pinv),
        'rho_opt': float(rho_opt),
        'margin_gain_pct': float(margin_gain_pct)
    }

    return f_opt, info


def allocate_mecanum_wheel_forces_guaranteed(
    W_cmd: np.ndarray,
    f_bar: np.ndarray,
    lx: float = L_X,
    ly: float = L_Y,
    weights_w: np.ndarray = None,
    wheel_order: str = "FL_FR_RR_RL",
) -> dict:
    """
    Khôi phục lực vô hướng 4 bánh f_w từ wrench lệnh W_cmd = [Fx, Fy, Mz]^T
    sao cho |f_w| <= f_bar[w] với mọi w ∈ {0, 1, 2, 3} bất cứ khi nào W_cmd nằm trong Zonotope.
    Mặc định dùng thứ tự 4 bánh thống nhất toàn codebase: [FL, FR, RR, RL].
    (Khắc phục triệt để Lỗi A.8 - Không nhân trùng 1/sqrt(2) ở f_bar).
    """
    c45 = 1.0 / np.sqrt(2.0)
    L_sum = lx + ly
    # Ma trận phân bổ G (3x4) chứa duy nhất 1 lần hệ số 1/sqrt(2)
    if wheel_order == "FL_FR_RR_RL":
        G = c45 * np.array([
            [ 1.0,   1.0,   1.0,   1.0],
            [-1.0,   1.0,  -1.0,   1.0],
            [-L_sum, L_sum, L_sum, -L_sum],
        ])
    elif wheel_order == "FL_FR_RL_RR":
        G = c45 * np.array([
            [ 1.0,   1.0,    1.0,   1.0],
            [-1.0,   1.0,    1.0,  -1.0],
            [-L_sum, L_sum, -L_sum, L_sum],
        ])
    else:
        raise ValueError(f"Unsupported wheel_order: {wheel_order}")
    # Véc-tơ cơ sở không gian rỗng (Null-space generator) của G: G @ v_null = 0
    # Cả hai thứ tự [FL, FR, RR, RL] và [FL, FR, RL, RR] đều có hàng trước (+), hàng sau (-)
    v_null = np.array([1.0, 1.0, -1.0, -1.0])

    W_cmd = np.asarray(W_cmd, dtype=np.float64)
    f_bar = np.asarray(f_bar, dtype=np.float64)

    # Nghiệm riêng tối thiểu chuẩn L2 (hoặc có trọng số)
    if weights_w is None:
        f_p = np.linalg.pinv(G) @ W_cmd
        alpha_l2 = 0.0
    else:
        W_inv = np.diag(1.0 / np.maximum(weights_w, 1e-6))
        f_p = W_inv @ G.T @ np.linalg.inv(G @ W_inv @ G.T) @ W_cmd
        alpha_l2 = 0.0

    # Tìm khoảng hợp lệ [L_w, U_w] cho tham số alpha sao cho: -f_bar[w] <= f_p[w] + alpha * v_null[w] <= f_bar[w]
    L_bounds = np.empty(4)
    U_bounds = np.empty(4)
    for w in range(4):
        if v_null[w] > 0:
            L_bounds[w] = -f_bar[w] - f_p[w]
            U_bounds[w] =  f_bar[w] - f_p[w]
        else:
            L_bounds[w] = -f_bar[w] + f_p[w]
            U_bounds[w] =  f_bar[w] + f_p[w]

    alpha_min = float(np.max(L_bounds))
    alpha_max = float(np.min(U_bounds))

    # Nếu W_cmd nằm trong Zonotope (s_zono == 0), chắc chắn alpha_min <= alpha_max
    is_zonotope_feasible = (alpha_min <= alpha_max + 1e-7)
    alpha_star = float(np.clip(alpha_l2, alpha_min, alpha_max)) if is_zonotope_feasible else 0.5 * (alpha_min + alpha_max)
    f_star = f_p + alpha_star * v_null

    return {
        "f_wheel_scalar": f_star,
        "alpha_star": alpha_star,
        "alpha_interval": (alpha_min, alpha_max),
        "is_zonotope_feasible": is_zonotope_feasible,
        "max_violation": float(np.max(np.maximum(0.0, np.abs(f_star) - f_bar)))
    }



if __name__ == "__main__":
    print("=" * 70)
    print("KIỂM TRA BỔ ĐỀ PHÂN BỔ KHÔNG GIAN NULL CHO BÁNH MECANUM (UNIT TEST)")
    print("=" * 70)

    # 1. Kiểm tra tính triệt tiêu của không gian Null: B @ n = 0
    res_null = B_MAT @ NULL_VEC
    print(f"1. Kiểm tra B @ NULL_VEC: {res_null}")
    assert np.allclose(res_null, 0.0), "LỖI: NULL_VEC không thuộc không gian null của B!"
    print("   -> CHÍNH XÁC: Vector không gian null triệt tiêu hoàn toàn lực thân xe!")

    # 2. Giả lập kịch bản uốn lượn cua gắt (Chicane):
    # Xe cần lực kéo rẽ cua: Fx = 1.0 N, Fy = 2.5 N, Mz = 0.15 N.m
    w_test = np.array([1.0, 2.5, 0.15])
    
    # Do sóng sánh chất lỏng và gia tốc ngang, tải trọng dồn mạnh về bánh phải,
    # làm 2 bánh trái bị dỡ tải nặng nề (Fz danh định = 17.78 N):
    # FL: 14.5 N (bị dỡ tải), FR: 21.0 N (bị đè tải), RR: 21.0 N, RL: 14.5 N
    Fz_dynamic = np.array([14.5, 21.0, 21.0, 14.5])
    mu_low = 0.12 # Sàn trơn

    t0 = time.perf_counter()
    N_ITER = 10000
    for _ in range(N_ITER):
        f_res, info = allocate_chassis_wrench(w_test, Fz_dynamic, mu=mu_low, use_nullspace=True)
    t_elapsed = (time.perf_counter() - t0) / N_ITER * 1e6 # microseconds

    f_pinv_only, info_pinv = allocate_chassis_wrench(w_test, Fz_dynamic, mu=mu_low, use_nullspace=False)

    print("\n2. Kịch bản Chicane trên sàn trơn (mu = 0.12):")
    print(f"   Wrench yêu cầu thân xe: Fx = {w_test[0]} N, Fy = {w_test[1]} N, Mz = {w_test[2]} N.m")
    print(f"   Tải trọng pháp tuyến 4 bánh Fz: {Fz_dynamic} N")
    print(f"   Giới hạn bám tối đa mu*Fz:      {np.round(mu_low * Fz_dynamic, 3)} N")
    print("-" * 70)
    print(f"   Lực bánh dùng Pseudoinverse thuần túy (pinv):")
    print(f"   f_pinv = {np.round(info_pinv['f_pinv'], 3)} N")
    print(f"   Tỷ số sử dụng ma sát cực đại (rho_pinv): {info_pinv['rho_pinv']:.4f} "
          f"({'TRƯỢT BÁNH!' if info_pinv['rho_pinv'] > 1.0 else 'Bám'})")
    print("-" * 70)
    print(f"   Lực bánh dùng Tối ưu Không gian Null (alpha*):")
    print(f"   alpha* tính được = {info['alpha']:.4f} N")
    print(f"   f_opt  = {np.round(info['f_opt'], 3)} N")
    print(f"   Tỷ số sử dụng ma sát cực đại (rho_opt):  {info['rho_opt']:.4f} "
          f"({'TRƯỢT BÁNH!' if info['rho_opt'] > 1.0 else 'AN TOÀN BÁM ĐƯỜNG!'})")
    print(f"   ==> Mức độ cải thiện biên ma sát: {info['margin_gain_pct']:.2f}%")
    print(f"   ==> Thời gian tính toán nghiệm đóng: {t_elapsed:.3f} microseconds (O(1))!")
    print("=" * 70)
