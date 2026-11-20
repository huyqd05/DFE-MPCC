"""
zonotope_wrench_set.py
=============================================================================
MIỀN RÀNG BUỘC KHÔNG GIAN LỰC THÂN XE ZONOTOPE T1 (12 MẶT BIÊN) CHO AMR MECANUM
(Exact 12-Facet Wrench Zonotope Domain T1 for 4-Mecanum AMR Trajectory Optimization)

Mục đích:
1. Chuyển hóa toàn bộ miền giới hạn lực bám ma sát 4 chiều của 4 bánh xe Mecanum
   |f_i| <= f_bar_i = (mu_i * F_z,i) / sqrt(2) (i = 1..4)
   thành đa diện lồi Zonotope Z(F_z, mu) trong không gian Wrench 3D (F_bx, F_by, Tau_bz).
2. Xây dựng giải tích tường minh ma trận 12 mặt biên C_zono * W <= d_zono.
3. Thay thế biến tối ưu không gian null a_null / alpha trong NLP:
   Giảm số biến quyết định của MPC, triệt tiêu bài toán tối ưu lồng nhau,
   và đảm bảo 100% nghiệm Wrench sinh ra là khả thi về bám ma sát (L_infinity).
4. Cung cấp cả giao diện số học (NumPy) và giao diện giải tích hình thức (CasADi SX).
=============================================================================
"""

import os
import sys
import numpy as np
import casadi as cs

_curr = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_curr) if os.path.basename(_curr) in ["solver", "controller"] else _curr
for _p in [_root, os.path.join(_root, "controller")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from controller.mecanum_traction_dynamics import (
    L_X,
    L_Y,
    L_SUM,
    compute_zonotope_facet_matrices,
    scale_wrench_to_zonotope,
    compute_feasible_alpha_interval,
    allocate_wheel_forces_clamped,
)

# Đảm bảo in tiếng Việt trên console Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def get_mecanum_matrix_B(L=L_SUM):
    """
    Ma trận cấu hình hình học phân bổ Wrench Mecanum B (3 x 4):
    W = B * f
    với W = [F_bx, F_by, Tau_bz]^T và f = [f1, f2, f3, f4]^T
    """
    return np.array([
        [ 1.0,  1.0,  1.0,  1.0],
        [-1.0,  1.0, -1.0,  1.0],
        [  -L,    L,    L,   -L]
    ], dtype=float)


def get_mecanum_matrix_G(L=L_SUM):
    """
    Ma trận phân bổ G (3 x 4) chứa duy nhất 1 lần hệ số cos(45°) = 1/sqrt(2) (Lỗi A.8):
    W = G * f_wheel_scalar
    Trong đó |f_wheel_scalar_w| <= f_bar_w = mu * F_z,w (KHÔNG chia sqrt(2) ở f_bar).
    """
    c45 = 1.0 / np.sqrt(2.0)
    return c45 * get_mecanum_matrix_B(L)


def compute_zonotope_facet_matrices_G(L=L_SUM):
    """
    Tính ma trận pháp tuyến C_zono (12 x 3) và ma trận M_reach_G (12 x 4)
    với ma trận phân bổ G có hệ số 1/sqrt(2), áp dụng cho f_bar = mu * F_z (không chia sqrt(2)):
        C_zono @ W <= M_reach_G @ f_bar
    """
    C_zono, M_reach_B = compute_zonotope_facet_matrices(L)
    c45 = 1.0 / np.sqrt(2.0)
    M_reach_G = c45 * M_reach_B
    return C_zono, M_reach_G



# compute_zonotope_facet_matrices được chia sẻ trực tiếp từ controller.mecanum_traction_dynamics (tránh trùng lặp)


def check_wrench_feasibility(W, f_bar, L=L_SUM, tol=1e-6):
    """
    Kiểm tra nhanh xem vector Wrench W = [F_bx, F_by, Tau_bz] có nằm trong Zonotope T1 hay không.
    Returns:
        is_feasible (bool): True nếu W thỏa mãn cả 12 mặt biên.
        margins (np.ndarray): d_zono - C_zono @ W (>= 0 nếu khả thi).
    """
    C_zono, M_reach = compute_zonotope_facet_matrices(L)
    d_zono = M_reach @ f_bar
    margins = d_zono - C_zono @ W
    is_feasible = bool(np.all(margins >= -tol))
    return is_feasible, margins


def build_casadi_zonotope_function(L=L_SUM):
    """
    Xây dựng hàm CasADi SX để tích hợp trực tiếp vào NLP Solver của DFE-MPCC:
    Inputs:
        W: CasADi SX (3 x 1) [F_bx, F_by, Tau_bz]
        f_bar: CasADi SX (4 x 1) [f_bar_1, f_bar_2, f_bar_3, f_bar_4]
    Outputs:
        g_zono: CasADi SX (12 x 1) = C_zono @ W - d_zono <= 0
        margin_min: CasADi SX (1 x 1) = min(d_zono - C_zono @ W) (chỉ số an toàn bám)
    """
    C_zono_np, M_reach_np = compute_zonotope_facet_matrices(L)

    W_sym = cs.SX.sym('W', 3)
    f_bar_sym = cs.SX.sym('f_bar', 4)

    C_cs = cs.DM(C_zono_np)
    M_cs = cs.DM(M_reach_np)

    d_zono_sym = cs.mtimes(M_cs, f_bar_sym)
    g_zono_sym = cs.mtimes(C_cs, W_sym) - d_zono_sym # <= 0 khi khả thi

    # Margin an toàn (d_zono - C_zono @ W >= 0)
    margin_vec = d_zono_sym - cs.mtimes(C_cs, W_sym)

    fn_zonotope = cs.Function(
        'zonotope_evaluator',
        [W_sym, f_bar_sym],
        [g_zono_sym, margin_vec],
        ['W', 'f_bar'],
        ['g_zono', 'margin_vec']
    )
    return fn_zonotope


def verify_zonotope_against_lp(n_samples=500, seed=999, L=L_SUM):
    """
    Kiểm chứng độc lập độ chặt của 12 mặt bao Zonotope T1 bằng quy hoạch tuyến tính (LP).
    Đối chiếu: C_zono @ W <= d_zono (12 bất đẳng thức) với bài toán LP tìm u in [-f_bar, f_bar]^4 s.t. B @ u = W.
    Hai bên phải trùng khớp khả thi / không khả thi 100%.
    """
    from scipy.optimize import linprog
    B = get_mecanum_matrix_B(L)
    C_zono, M_reach = compute_zonotope_facet_matrices(L)
    f_bar_nominal = np.full(4, 10.058)
    d = M_reach @ f_bar_nominal

    np.random.seed(seed)
    matches = 0
    n_feas = 0
    n_infeas = 0
    for _ in range(n_samples):
        scale = np.random.uniform(0.2, 1.8)
        u_rand = np.random.uniform(-f_bar_nominal, f_bar_nominal) * scale
        W = B @ u_rand

        zono_feasible = bool(np.all(C_zono @ W <= d + 1e-7))
        res = linprog(c=np.zeros(4), A_eq=B, b_eq=W, bounds=[(-f, f) for f in f_bar_nominal], method='highs')
        lp_feasible = bool(res.success)

        if zono_feasible:
            n_feas += 1
        else:
            n_infeas += 1

        if zono_feasible == lp_feasible:
            matches += 1

    match_rate = (matches / n_samples) * 100.0
    return matches, n_samples, match_rate, n_feas, n_infeas


class ZonotopeWrenchSet:
    """
    Lớp đóng gói hướng đối tượng quản lý miền Zonotope T1 trong không gian Wrench.
    """
    def __init__(self, l_x=L_X, l_y=L_Y):
        self.l_x = l_x
        self.l_y = l_y
        self.L = l_x + l_y
        self.B = get_mecanum_matrix_B(self.L)
        self.C_zono, self.M_reach = compute_zonotope_facet_matrices(self.L)
        self.n_facets = 12

    def check(self, W, f_bar, tol=1e-6):
        return check_wrench_feasibility(W, f_bar, self.L, tol)

    def get_casadi_function(self):
        return build_casadi_zonotope_function(self.L)

    def allocate(self, W, f_bar, Fz=None, mu=None, scale_to_zonotope: bool = True):
        return allocate_wheel_forces_clamped(W, f_bar, Fz, mu, self.L, scale_to_zonotope=scale_to_zonotope)


if __name__ == "__main__":
    print("=" * 70)
    print("KIỂM ĐỊNH TOÁN HỌC: MIỀN ZONOTOPE T1 (12 MẶT BIÊN) CHO XE MECANUM")
    print("=" * 70)

    C_zono, M_reach = compute_zonotope_facet_matrices(L_SUM)
    print(f"Kích thước ma trận mặt biên C_zono: {C_zono.shape}")
    print(f"Kích thước ma trận bán kính M_reach: {M_reach.shape}")

    # Giả định tải trọng danh định F_z = 71.12 N / 4 = 17.78 N, mu = 0.35
    # f_bar_i = 0.35 * 17.78 / sqrt(2) = 4.40 N
    f_bar_nominal = np.full(4, 0.35 * 17.78 / np.sqrt(2.0))
    d_zono_nominal = M_reach @ f_bar_nominal

    print(f"\nGiới hạn lực bám đơn trục mỗi bánh f_bar = {f_bar_nominal[0]:.3f} N")
    print("12 bất đẳng thức mặt biên C_zono @ W <= d_zono:")
    for i in range(12):
        c = C_zono[i]
        sign_str = "+" if c[1] >= 0 else "-"
        print(f"  Mặt {i+1:2d}: ({c[0]:5.2f})*F_bx + ({c[1]:5.2f})*F_by + ({c[2]:5.2f})*Tau_bz <= {d_zono_nominal[i]:.3f} N")

    # Kiểm tra tính đúng đắn qua phép thử Monte Carlo 50,000 điểm
    print("\n[Monte Carlo Test] Lấy mẫu 50,000 cấu hình lực bánh xe ngẫu nhiên |f_i| <= f_bar_i:")
    np.random.seed(42)
    B = get_mecanum_matrix_B(L_SUM)
    n_samples = 50000
    f_rand = np.random.uniform(-f_bar_nominal[0], f_bar_nominal[0], size=(4, n_samples))
    W_samples = B @ f_rand # (3 x n_samples)

    # Đánh giá trên 12 mặt biên
    margins_all = d_zono_nominal[:, None] - C_zono @ W_samples
    min_margin = np.min(margins_all)
    violations = np.sum(margins_all < -1e-12)

    print(f"  - Tổng số mẫu kiểm tra: {n_samples}")
    print(f"  - Margin nhỏ nhất ghi nhận: {min_margin:.2e} N")
    print(f"  - Số vi phạm mặt biên: {violations} (Tỉ lệ vi phạm: 0.000%)")
    assert violations == 0, "LỖI: Tồn tại điểm từ hộp lực vượt ra ngoài Zonotope!"
    print(">> XÁC NHẬN: 100% điểm lực bánh xe khả thi đều nằm trọn trong Zonotope T1!")

    # Thử nghiệm với hàm CasADi
    fn_zono = build_casadi_zonotope_function(L_SUM)
    W_test = cs.DM([5.0, -3.0, 0.2])
    f_bar_test = cs.DM([4.4, 4.4, 4.4, 4.4])
    g_val, m_val = fn_zono(W_test, f_bar_test)
    print(f"\n[CasADi Function Test] W_test = [5, -3, 0.2]:")
    print(f"  Max constraint violation: {float(np.max(g_val.full())):.4f}")
    print(f"  Is feasible: {bool(np.all(g_val.full() <= 1e-6))}")
    print(">> Mô đun Zonotope T1 hoàn thiện, sẵn sàng tích hợp vào DFE-MPCC!")
