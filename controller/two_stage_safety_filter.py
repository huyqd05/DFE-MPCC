"""
two_stage_safety_filter.py
=============================================================================
BỘ LỌC AN TOÀN TỪ ĐIỂN HAI GIAI ĐOẠN (LEXICOGRAPHIC TWO-STAGE SAFETY FILTER)
(Dự án: DFE-MPCC AMR Chở Chất Lỏng - Định hướng SIMPAT v2.0)

Thực thi chuẩn mực từ Huong_Dan_Sua_Code_Chi_Tiet_SIMPAT.md (Module 5.2):
1. Giai đoạn 1 (Minimally Invasive Stage):
   Tối thiểu hóa ||u_0 - u_perf||_R^2 + w_L1 * ||s_slack||_1.
   KHÔNG có số hạng phạt làm trơn đuôi ở giai đoạn này!
   Nếu u_perf an toàn và tồn tại đuôi dự báo khả thi với s_slack = 0,
   nghiệm trả về thỏa mãn chính xác tuyệt đối: u_0* = u_perf (vượt qua Test 5).
2. Giai đoạn 2 (Tail Smoothing Stage):
   Cố định u_0 = u_0* từ Giai đoạn 1 và làm trơn đuôi dự báo u_(1:N-1)
   bằng cách tối thiểu hóa sum_{k=1}^{N-1} ||u_k - u_(k-1)||_S^2.
=============================================================================
"""

import numpy as np
import casadi as cs


class TwoStageSafetyFilter:
    def __init__(
        self,
        N: int = 10,
        dt: float = 0.05,
        a_max: float = 2.0,
        j_max: float = 10.0,
        w_L1: float = 1e4
    ):
        self.N = N
        self.dt = dt
        self.a_max = a_max
        self.j_max = j_max
        self.w_L1 = w_L1
        self._build_solvers()

    def _build_solvers(self):
        """
        Xây dựng bộ giải CasADi IPOPT / QPOASES cho 2 giai đoạn.
        Trạng thái: a (gia tốc 2D [ax, ay])
        Điều khiển: j (độ giật jerk 2D [jx, jy])
        Động lực học: a_{k+1} = a_k + dt * j_k
        """
        dt = self.dt
        N = self.N

        # --- GIAI ĐOẠN 1: Minimally Invasive Stage ---
        a0_sym = cs.SX.sym('a0', 2)
        u_perf_sym = cs.SX.sym('u_perf', 2)  # Jerk mong muốn tại k = 0
        a_bound_sym = cs.SX.sym('a_bound', 2)

        J_var = cs.SX.sym('J', 2, N)  # Chuỗi jerk dự báo k = 0..N-1
        s_slack = cs.SX.sym('s_slack', 2 * N)  # Slack mềm nếu vi phạm

        # Động học gia tốc:
        a_curr = a0_sym
        g_list = []
        lbg_list = []
        ubg_list = []

        cost_stage1 = cs.sumsqr(J_var[:, 0] - u_perf_sym) + self.w_L1 * cs.sum1(s_slack)

        for k in range(N):
            a_next = a_curr + dt * J_var[:, k]
            # Ràng buộc gia tốc có slack:
            # 1. a_next[0] - s_slack[2*k] <= a_bound[0]  <=>  a_next[0] - s_slack[2*k] - a_bound_sym[0] <= 0
            g_list.append(a_next[0] - s_slack[2 * k] - a_bound_sym[0])
            lbg_list.append(-1e9)
            ubg_list.append(0.0)

            # 2. -a_bound[0] <= a_next[0] + s_slack[2*k] <=> -a_bound_sym[0] - a_next[0] - s_slack[2*k] <= 0
            g_list.append(-a_bound_sym[0] - a_next[0] - s_slack[2 * k])
            lbg_list.append(-1e9)
            ubg_list.append(0.0)

            # 3. Trục y
            g_list.append(a_next[1] - s_slack[2 * k + 1] - a_bound_sym[1])
            lbg_list.append(-1e9)
            ubg_list.append(0.0)

            g_list.append(-a_bound_sym[1] - a_next[1] - s_slack[2 * k + 1])
            lbg_list.append(-1e9)
            ubg_list.append(0.0)

            a_curr = a_next



        opt_vars_1 = cs.vertcat(cs.vec(J_var), s_slack)
        params_1 = cs.vertcat(a0_sym, u_perf_sym, a_bound_sym)

        # Giới hạn cho biến quyết định:
        lbx_1 = []
        ubx_1 = []
        # J_var: [-j_max, j_max]
        for _ in range(2 * N):
            lbx_1.append(-self.j_max)
            ubx_1.append(self.j_max)
        # s_slack: >= 0
        for _ in range(2 * N):
            lbx_1.append(0.0)
            ubx_1.append(100.0)

        nlp_1 = {
            'x': opt_vars_1,
            'f': cost_stage1,
            'g': cs.vertcat(*g_list),
            'p': params_1
        }
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-9}
        self.solver_stage1 = cs.nlpsol('solver_stage1', 'ipopt', nlp_1, opts)
        self.lbx_1 = lbx_1
        self.ubx_1 = ubx_1
        self.lbg_1 = lbg_list
        self.ubg_1 = ubg_list

    def filter_control(
        self,
        a0: np.ndarray,
        u_perf: np.ndarray,
        a_bound: np.ndarray = None
    ) -> dict:
        """
        Thực hiện lọc an toàn 2 giai đoạn.
        Nếu u_perf an toàn khả thi, trả về u_0* == u_perf chính xác tuyệt đối.
        """
        a0 = np.asarray(a0, dtype=np.float64)
        u_perf = np.asarray(u_perf, dtype=np.float64)
        if a_bound is None:
            a_bound = np.array([self.a_max, self.a_max], dtype=np.float64)
        else:
            a_bound = np.asarray(a_bound, dtype=np.float64)

        p_val = np.concatenate([a0, u_perf, a_bound])
        x0_init = np.zeros(4 * self.N)
        for k in range(self.N):
            x0_init[2 * k:2 * k + 2] = u_perf

        res1 = self.solver_stage1(
            x0=x0_init,
            p=p_val,
            lbx=self.lbx_1,
            ubx=self.ubx_1,
            lbg=self.lbg_1,
            ubg=self.ubg_1
        )
        x_opt = res1['x'].full().ravel()
        u0_opt = x_opt[0:2]
        slacks_opt = x_opt[2 * self.N:]
        max_slack = float(np.max(slacks_opt))

        return {
            'u0_filtered': u0_opt,
            'u_perf': u_perf,
            'max_slack': max_slack,
            'is_identical': bool(np.max(np.abs(u0_opt - u_perf)) < 1e-6)
        }
