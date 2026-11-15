"""
sph_navier_stokes_engine.py
=============================================================================
LÕI MÔ PHỎNG THỦY ĐỘNG LỰC HỌC HẠT WCSPH (WEAKLY COMPRESSIBLE SPH)
CHUẨN NAVIER-STOKES CÓ KHUẾCH TÁN MẬT ĐỘ (delta-SPH) CHO XE TỰ HÀNH AMR

Các chuẩn mực kỹ thuật CFD được thực thi:
1. Sơ đồ tích phân thời gian: Symplectic Velocity-Verlet bậc 2 (Kick-Drift-Kick)
   kết hợp bước thời gian thích nghi theo điều kiện CFL nghiêm ngặt:
   dt <= C_cfl * min( h / (c_0 + |v|_max), sqrt(h / |a|_max) ) (~ 50 - 75 us).
2. Phương trình liên tục có số hạng khuếch tán mật độ delta-SPH
   (Molteni & Colagrossi 2009, delta = 0.1) khử nhiễu áp suất tần số cao.
3. Phương trình động lượng bảo toàn đối xứng + Độ nhớt nhân tạo Monaghan
   Pi_ij (tham số alpha hiệu chỉnh từ thí nghiệm tắt dần tự do).
4. Phương trình trạng thái Tait (gamma = 7, c_0 = 15 m/s >= 10*max(v_max, sqrt(g*h_l))
   đảm bảo dao động mật độ < 1%).
5. Điều kiện biên mDBC (3 lớp hạt biên ngoại suy áp suất thủy tĩnh).
   Khóa biên hình học chỉ đặt ở mép vật lý ngoài cùng như "lưới an toàn" (Safety Net)
   kèm bộ đếm số lần kích hoạt (clamp_count) ở mỗi bước tích phân.
6. Trường gia tốc hiệu dụng tổng quát trong hệ quy chiếu gắn thân xe:
   Bao gồm gia tốc tịnh tiến (a_y, a_z), trọng trường nghiêng theo góc roll phi,
   lực quán tính tiếp tuyến Euler (-alpha_x x r), lực ly tâm (-omega_x x (omega_x x r))
   và lực Coriolis (-2 * omega_x x v_rel).
=============================================================================
"""

import numpy as np
from scipy.spatial import cKDTree
import time


class WCSPH2DSolver:
    def __init__(
        self,
        tank_width=0.100,       # b: Chiều rộng bình theo phương Y (m)
        tank_height=0.110,      # H: Chiều cao bình theo phương Z (m)
        liquid_height=0.080,    # h_l: Mực nước tĩnh danh định (m)
        tank_length=0.150,      # L_x: Chiều dài bình theo phương X (m)
        dp=0.0035,              # dp: Khoảng cách hạt ban đầu (m)
        rho_0=1000.0,           # Khối lượng riêng danh định (kg/m^3)
        c_0=15.0,               # Vận tốc âm thanh nhân tạo (m/s): >= 10*sqrt(g*h_l) ≈ 8.86 m/s
        gamma=7.0,              # Số mũ phương trình trạng thái Tait
        alpha_visc=0.05,        # Hệ số nhớt nhân tạo Monaghan alpha
        beta_visc=0.0,          # Hệ số nhớt nhân tạo bậc hai beta
        delta_sph=0.10,         # Hệ số khuếch tán mật độ Molteni-Colagrossi (delta-SPH)
        cfl_coeff=0.22,         # Hệ số an toàn CFL (dt <= 0.22 * h / (c_0 + v_max))
        g=9.81
    ):
        self.tank_width = tank_width
        self.tank_height = tank_height
        self.liquid_height = liquid_height
        self.tank_length = tank_length
        self.dp = dp
        self.rho_0 = rho_0
        self.c_0 = c_0
        self.gamma = gamma
        self.alpha_visc = alpha_visc
        self.beta_visc = beta_visc
        self.delta_sph = delta_sph
        self.cfl_coeff = cfl_coeff
        self.g = g

        # Bán kính làm trơn Kernel Wendland C2
        self.h = 1.35 * dp
        self.r_cut = 2.0 * self.h
        self.r_skin = 2.60 * self.h  # Lớp đệm Verlet skin = 0.60 * h cho tích phân cửa sổ dài (chỉ lọc theo r_cut)

        # Hằng số phương trình trạng thái Tait
        self.B_tait = (rho_0 * c_0**2) / gamma

        # Hệ số chuẩn hóa Kernel Wendland C2 trong 2D: alpha_d = 7 / (4 * pi * h^2)
        self.alpha_d_2d = 7.0 / (4.0 * np.pi * self.h**2)

        # 1. Khởi tạo lưới hạt chất lỏng
        n_y = max(1, int(np.round(tank_width / dp)))
        n_z = max(1, int(np.round(liquid_height / dp)))
        dy_actual = tank_width / n_y
        dz_actual = liquid_height / n_z

        y_fluid = np.linspace(-tank_width / 2.0 + dy_actual / 2.0, tank_width / 2.0 - dy_actual / 2.0, n_y)
        z_fluid = np.linspace(dz_actual / 2.0, liquid_height - dz_actual / 2.0, n_z)
        yy_f, zz_f = np.meshgrid(y_fluid, z_fluid)
        self.pos_fluid = np.column_stack((yy_f.flatten(), zz_f.flatten()))
        self.n_fluid = len(self.pos_fluid)

        # Khối lượng 2D mỗi hạt (kg/m) và 3D quy đổi theo chiều dài L_x (kg)
        self.area_p = (tank_width * liquid_height) / self.n_fluid
        self.mass_2d = rho_0 * self.area_p
        self.mass_p = self.mass_2d * self.tank_length
        self.total_mass = self.n_fluid * self.mass_p

        # 2. Khởi tạo hạt biên thành bình mDBC (3 lớp hạt)
        n_layers = 3
        b_particles = []

        # Đáy bình (z < 0)
        y_bot = np.arange(-tank_width / 2.0 - n_layers * dp + dp / 2.0,
                          tank_width / 2.0 + n_layers * dp, dp)
        for l in range(1, n_layers + 1):
            z_val = -(l - 0.5) * dp
            for y_val in y_bot:
                b_particles.append([y_val, z_val])

        # Vách trái (y < -tank_width/2) & Vách phải (y > tank_width/2)
        z_wall = np.arange(dp / 2.0, tank_height + n_layers * dp, dp)
        for l in range(1, n_layers + 1):
            y_left = -tank_width / 2.0 - (l - 0.5) * dp
            y_right = tank_width / 2.0 + (l - 0.5) * dp
            for z_val in z_wall:
                b_particles.append([y_left, z_val])
                b_particles.append([y_right, z_val])

        self.pos_bound = np.array(b_particles)
        self.n_bound = len(self.pos_bound)
        self.n_total = self.n_fluid + self.n_bound

        # Mảng trạng thái tổng hợp
        self.pos = np.vstack((self.pos_fluid, self.pos_bound))
        self.vel = np.zeros_like(self.pos)
        self.rho = np.full(self.n_total, rho_0)
        self.p = np.zeros(self.n_total)

        self.is_fluid = np.zeros(self.n_total, dtype=bool)
        self.is_fluid[:self.n_fluid] = True

        # 3. Khởi tạo phân bố áp suất và mật độ thủy tĩnh tại t = 0
        z_f = self.pos_fluid[:, 1]
        p_hydro = np.maximum(rho_0 * g * (liquid_height - z_f), 0.0)
        self.p[self.is_fluid] = p_hydro
        self.rho[self.is_fluid] = rho_0 * (p_hydro / self.B_tait + 1.0)**(1.0 / gamma)

        # Bộ đệm lưu gia tốc và đạo hàm mật độ cho tích phân Velocity-Verlet
        self.acc = np.zeros_like(self.pos)
        self.drho_dt = np.zeros(self.n_total)
        self.f_wall_2d = np.zeros(2)
        self.m_roll_3d = 0.0
        self._initialized_forces = False

        # Bộ đệm danh sách lân cận Verlet
        self._cached_pairs = None
        self._tree_pos_ref = None
        self._steps_since_tree = 0
        self._tree_rebuild_interval = 3

        # Thống kê lưới an toàn biên hình học (Safety Net Counter)
        self.total_clamp_activations = 0
        self.step_clamp_count = 0
        self.total_steps = 0
        self.time = 0.0

    def compute_cfl_dt(self):
        """
        Tính bước thời gian thích nghi thỏa mãn điều kiện CFL:
        dt <= C_cfl * min( h / (c_0 + |v|_max), sqrt(h / |a|_max) )
        """
        v_fluid = self.vel[:self.n_fluid]
        v_max = float(np.max(np.linalg.norm(v_fluid, axis=1))) if self.n_fluid > 0 else 0.0
        dt_acoustic = self.h / (self.c_0 + v_max)

        a_fluid = self.acc[:self.n_fluid]
        a_max = float(np.max(np.linalg.norm(a_fluid, axis=1))) if self.n_fluid > 0 else self.g
        dt_force = np.sqrt(self.h / max(a_max, self.g))

        return self.cfl_coeff * min(dt_acoustic, dt_force)

    def _compute_non_inertial_g_eff(self, pos_f, vel_f, a_y_tank, a_z_tank=0.0,
                                    roll_angle=0.0, roll_rate=0.0, roll_acc=0.0):
        """
        Tính trường gia tốc hiệu dụng g_eff(r_i, v_i) trong hệ quy chiếu gắn chặt với bình:
        g_eff = R^T * g - a_tank - alpha x r - omega x (omega x r) - 2 * omega x v_rel
        """
        g_y_base = -a_y_tank - self.g * np.sin(roll_angle)
        g_z_base = -a_z_tank - self.g * np.cos(roll_angle)

        if abs(roll_rate) < 1e-9 and abs(roll_acc) < 1e-9:
            g_eff = np.empty_like(pos_f)
            g_eff[:, 0] = g_y_base
            g_eff[:, 1] = g_z_base
            return g_eff

        y_i = pos_f[:, 0]
        z_i = pos_f[:, 1]
        vy_i = vel_f[:, 0]
        vz_i = vel_f[:, 1]

        # Euler (-alpha x r) + Ly tâm (-omega x (omega x r)) + Coriolis (-2 omega x v)
        g_eff_y = g_y_base + roll_acc * z_i + (roll_rate**2) * y_i + 2.0 * roll_rate * vz_i
        g_eff_z = g_z_base - roll_acc * y_i + (roll_rate**2) * z_i - 2.0 * roll_rate * vy_i
        return np.column_stack((g_eff_y, g_eff_z))

    def _evaluate_derivatives(self, a_y_tank, a_z_tank=0.0,
                              roll_angle=0.0, roll_rate=0.0, roll_acc=0.0):
        """
        Tính toán vế phải của hệ phương trình Navier-Stokes + delta-SPH:
        - drho_dt (Phương trình liên tục + Khuếch tán mật độ Molteni-Colagrossi)
        - acc (Phương trình động lượng + Độ nhớt nhân tạo Monaghan + Phản lực biên)
        - Tích phân lực thành bình và mô-men lật thủy động M_roll
        """
        n_tot = self.n_total
        n_f = self.n_fluid

        # 1. Cập nhật danh sách lân cận (Verlet list dựa trên dịch chuyển cực đại so với skin)
        need_rebuild = (self._cached_pairs is None or self._tree_pos_ref is None)
        if not need_rebuild and n_f > 0:
            disp = self.pos[:n_f] - self._tree_pos_ref
            max_disp_sq = float(np.max(disp[:, 0] ** 2 + disp[:, 1] ** 2))
            skin_lim = 0.45 * (self.r_skin - self.r_cut)
            if max_disp_sq >= skin_lim * skin_lim:
                need_rebuild = True
        if need_rebuild:
            tree = cKDTree(self.pos)
            raw_pairs = tree.query_pairs(self.r_skin, output_type='ndarray')
            # Hạt biên đứng yên (n_f..n_total-1), loại bỏ toàn bộ cặp boundary-boundary
            self._cached_pairs = raw_pairs[raw_pairs[:, 0] < n_f] if len(raw_pairs) > 0 else raw_pairs
            self._tree_pos_ref = np.copy(self.pos[:n_f])
            self._steps_since_tree = 0
        self._steps_since_tree += 1

        pairs = self._cached_pairs
        if len(pairs) == 0:
            self.drho_dt.fill(0.0)
            self.acc.fill(0.0)
            return

        i_idx = pairs[:, 0]
        j_idx = pairs[:, 1]

        r_ij = self.pos[i_idx] - self.pos[j_idx]
        dist_sq = r_ij[:, 0]**2 + r_ij[:, 1]**2
        mask_cut = (dist_sq > 1e-18) & (dist_sq <= self.r_cut**2)

        i_idx = i_idx[mask_cut]
        j_idx = j_idx[mask_cut]
        r_ij = r_ij[mask_cut]
        dist_sq = dist_sq[mask_cut]
        dist = np.sqrt(dist_sq)

        # 2. Hàm Kernel Wendland C2 và đạo hàm
        q = dist / self.h
        one_minus_half_q = 1.0 - 0.5 * q
        omhq3 = one_minus_half_q**3
        W_pair = self.alpha_d_2d * (omhq3 * one_minus_half_q) * (1.0 + 2.0 * q)

        # dW/dr = alpha_d * (-5/h) * q * (1 - q/2)^3 => (1/r)*(dW/dr) = alpha_d * (-5/h^2) * (1 - q/2)^3
        grad_factor = (self.alpha_d_2d * (-5.0 / (self.h**2))) * omhq3
        grad_W_y = r_ij[:, 0] * grad_factor
        grad_W_z = r_ij[:, 1] * grad_factor

        # 3. Phương trình trạng thái Tait cho hạt chất lỏng
        rho_ratio = np.clip(self.rho[:n_f] / self.rho_0, 0.95, 1.10)
        p_f = self.B_tait * (rho_ratio**self.gamma - 1.0)
        self.p[:n_f] = np.clip(p_f, 0.0, 25000.0)

        # Trường gia tốc hiệu dụng tại tâm khối chất lỏng để ngoại suy biên mDBC
        g_eff_mean_y = -a_y_tank - self.g * np.sin(roll_angle)
        g_eff_mean_z = -a_z_tank - self.g * np.cos(roll_angle)

        # 4. Ngoại suy áp suất và mật độ cho hạt biên mDBC
        # Do hạt chất lỏng có chỉ số [0..n_f-1] và hạt biên [n_f..n_tot-1],
        # với i_idx < j_idx thì cặp fluid-boundary luôn có i_idx < n_f và j_idx >= n_f
        fb_mask = (i_idx < n_f) & (j_idx >= n_f)
        if np.any(fb_mask):
            f_id = i_idx[fb_mask]
            b_id = j_idx[fb_mask]
            w_fb = W_pair[fb_mask]
            # r_b - r_f = - r_ij[fb_mask]
            dr_bf_y = -r_ij[fb_mask, 0]
            dr_bf_z = -r_ij[fb_mask, 1]
            hydro_corr = self.rho_0 * (g_eff_mean_y * dr_bf_y + g_eff_mean_z * dr_bf_z)
            p_extrap = np.maximum(self.p[f_id] + hydro_corr, 0.0)

            b_num = np.bincount(b_id, weights=w_fb * p_extrap, minlength=n_tot)
            b_den = np.bincount(b_id, weights=w_fb, minlength=n_tot)

            valid_b = b_den[n_f:] > 1e-12
            p_b_new = np.zeros(self.n_bound)
            p_b_new[valid_b] = b_num[n_f:][valid_b] / b_den[n_f:][valid_b]
            self.p[n_f:] = p_b_new
            self.rho[n_f:] = self.rho_0 * (p_b_new / self.B_tait + 1.0)**(1.0 / self.gamma)
        else:
            self.p[n_f:] = 0.0
            self.rho[n_f:] = self.rho_0

        # 5. Phương trình liên tục + Khuếch tán mật độ delta-SPH (Molteni & Colagrossi 2009)
        v_ij_y = self.vel[i_idx, 0] - self.vel[j_idx, 0]
        v_ij_z = self.vel[i_idx, 1] - self.vel[j_idx, 1]
        v_dot_gradW = v_ij_y * grad_W_y + v_ij_z * grad_W_z

        cont_term = self.mass_2d * v_dot_gradW
        drho_i = np.bincount(i_idx, weights=cont_term, minlength=n_tot)
        drho_j = np.bincount(j_idx, weights=cont_term, minlength=n_tot)
        self.drho_dt = drho_i + drho_j

        # Số hạng khuếch tán delta-SPH giữa các hạt chất lỏng (fluid-fluid pairs)
        if self.delta_sph > 0.0:
            ff_mask = (i_idx < n_f) & (j_idx < n_f)
            if np.any(ff_mask):
                fi = i_idx[ff_mask]
                fj = j_idx[ff_mask]
                # r_ij . grad_W_ij = dist_sq * grad_factor
                r_dot_gradW = dist_sq[ff_mask] * grad_factor[ff_mask]
                rho_bar_ff = 0.5 * (self.rho[fi] + self.rho[fj])
                # diff_ij = 2 * delta * h * c_0 * m_2d * ((rho_i - rho_j) / rho_bar) * (r_ij . grad_W) / (r^2 + 0.01*h^2)
                # Lưu ý: r_dot_gradW < 0, nên khi rho_i > rho_j thì diff_term < 0 (làm giảm đỉnh mật độ rho_i)
                diff_term = (2.0 * self.delta_sph * self.h * self.c_0 * self.mass_2d *
                             (self.rho[fi] - self.rho[fj]) / rho_bar_ff *
                             r_dot_gradW / (dist_sq[ff_mask] + 0.01 * self.h**2))
                self.drho_dt += np.bincount(fi, weights=diff_term, minlength=n_tot)
                self.drho_dt -= np.bincount(fj, weights=diff_term, minlength=n_tot)

        # 6. Phương trình động lượng Navier-Stokes + Độ nhớt nhân tạo Monaghan Pi_ij
        p_rho2 = self.p / (self.rho * self.rho)
        pressure_factor = -(p_rho2[i_idx] + p_rho2[j_idx])

        v_dot_r = v_ij_y * r_ij[:, 0] + v_ij_z * r_ij[:, 1]
        visc_factor = np.zeros_like(pressure_factor)
        approaching = v_dot_r < 0.0
        if np.any(approaching):
            mu_ij = (self.h * v_dot_r[approaching]) / (dist_sq[approaching] + 0.01 * self.h**2)
            rho_bar = 0.5 * (self.rho[i_idx[approaching]] + self.rho[j_idx[approaching]])
            # -Pi_ij (mu_ij < 0 when approaching) must match pressure_factor (< 0) to repel & damp
            visc_factor[approaching] = (self.alpha_visc * self.c_0 * mu_ij - self.beta_visc * mu_ij**2) / rho_bar

        # Phương trình động lượng bảo toàn đối xứng SPH (mDBC áp suất ngoại suy)
        total_factor = self.mass_2d * (pressure_factor + visc_factor)
        f_pair_y = total_factor * grad_W_y
        f_pair_z = total_factor * grad_W_z

        acc_y = (np.bincount(i_idx, weights=f_pair_y, minlength=n_tot) -
                 np.bincount(j_idx, weights=f_pair_y, minlength=n_tot))
        acc_z = (np.bincount(i_idx, weights=f_pair_z, minlength=n_tot) -
                 np.bincount(j_idx, weights=f_pair_z, minlength=n_tot))

        # Thêm đệm biên trơn (Soft boundary cushion) phòng ngừa xâm lấn biên tại d < 0.28 dp
        b_half = self.tank_width / 2.0
        pos_f = self.pos[:n_f]
        d_left = pos_f[:, 0] - (-b_half)
        d_right = b_half - pos_f[:, 0]
        d_bot = pos_f[:, 1]
        thresh = 0.28 * self.dp
        k_rep = 0.25 * (self.c_0**2) / self.dp

        rep_y = np.zeros(n_f)
        rep_z = np.zeros(n_f)
        mask_l = (d_left > 0) & (d_left < thresh)
        mask_r = (d_right > 0) & (d_right < thresh)
        mask_b = (d_bot > 0) & (d_bot < thresh)

        if np.any(mask_l):
            rep_y[mask_l] += k_rep * ((thresh - d_left[mask_l]) / thresh)**2
        if np.any(mask_r):
            rep_y[mask_r] -= k_rep * ((thresh - d_right[mask_r]) / thresh)**2
        if np.any(mask_b):
            rep_z[mask_b] += k_rep * ((thresh - d_bot[mask_b]) / thresh)**2

        # Cộng trường gia tốc hiệu dụng phi quán tính cho các hạt chất lỏng
        g_eff_f = self._compute_non_inertial_g_eff(
            self.pos[:n_f], self.vel[:n_f],
            a_y_tank, a_z_tank, roll_angle, roll_rate, roll_acc
        )
        self.acc[:n_f, 0] = acc_y[:n_f] + g_eff_f[:, 0] + rep_y
        self.acc[:n_f, 1] = acc_z[:n_f] + g_eff_f[:, 1] + rep_z
        self.acc[n_f:, :] = 0.0

        # 7. Tích phân lực tác dụng lên thành bình theo định luật III Newton
        # Lực chất lỏng tác dụng lên thành bình = - Lực thành bình tác dụng lên chất lỏng
        if np.any(fb_mask):
            self.f_wall_2d[0] = -self.mass_2d * float(np.sum(f_pair_y[fb_mask]))
            self.f_wall_2d[1] = -self.mass_2d * float(np.sum(f_pair_z[fb_mask]))
        else:
            self.f_wall_2d[0] = 0.0
            self.f_wall_2d[1] = 0.0

    @staticmethod
    def _fast_percentile_90(arr: np.ndarray, default_val: float) -> float:
        m = len(arr)
        if m == 0:
            return float(default_val)
        if m == 1:
            return float(arr[0])
        s = np.sort(arr)
        pos = 0.90 * (m - 1)
        lo = int(pos)
        hi = min(m - 1, lo + 1)
        frac = pos - lo
        return float(s[lo] + frac * (s[hi] - s[lo]))

    def get_free_surface_metrics(self) -> dict:
        """Đo trực tiếp chiều cao mặt thoáng từ vị trí hạt tự do (vách biên & độ nghiêng khối tâm)."""
        pos_f = self.pos[:self.n_fluid]
        b_half = self.tank_width / 2.0
        strip = 2.0 * self.dp
        right_zone = pos_f[:, 0] < (-b_half + strip)
        left_zone = pos_f[:, 0] > (b_half - strip)
        eta_right = self._fast_percentile_90(pos_f[right_zone, 1], self.liquid_height)
        eta_left = self._fast_percentile_90(pos_f[left_zone, 1], self.liquid_height)
        eta_wall_mm = 0.5 * abs(eta_right - eta_left) * 1000.0
        com_horiz = float(np.mean(pos_f[:, 0]))
        eta_cm_mm = 6.0 * self.liquid_height * abs(com_horiz) / max(self.tank_width, 1e-6) * 1000.0
        return {
            "eta_left": eta_left,
            "eta_right": eta_right,
            "eta_wall_mm": eta_wall_mm,
            "eta_cm_mm": eta_cm_mm,
            "eta_direct_mm": max(eta_wall_mm, eta_cm_mm),
        }

    def step(self, dt, a_y_tank, a_z_tank=0.0, roll_angle=0.0, roll_rate=0.0, roll_acc=0.0, compute_metrics: bool = True):
        """
        Thực hiện 1 bước tích phân Symplectic Velocity-Verlet bậc 2 (Kick-Drift-Kick).
        """
        n_f = self.n_fluid

        # Khởi tạo gia tốc ở bước đầu tiên nếu chưa có
        if not self._initialized_forces:
            self._evaluate_derivatives(a_y_tank, a_z_tank, roll_angle, roll_rate, roll_acc)
            self._initialized_forces = True

        # --- BƯỚC 1 (Half-step Kick & Full-step Drift) ---
        vel_half = self.vel[:n_f] + 0.5 * dt * self.acc[:n_f]
        rho_half = np.clip(self.rho[:n_f] + 0.5 * dt * self.drho_dt[:n_f], 0.92 * self.rho_0, 1.10 * self.rho_0)

        self.pos[:n_f] += dt * vel_half
        self.vel[:n_f] = vel_half
        self.rho[:n_f] = rho_half

        # --- BỘ ĐẾM VI PHẠM BIÊN HÌNH HỌC (Safety Monitor) ---
        b_half = self.tank_width / 2.0
        margin = 0.05 * self.dp
        idx_l = self.pos[:n_f, 0] < (-b_half + margin)
        idx_r = self.pos[:n_f, 0] > (b_half - margin)
        idx_b = self.pos[:n_f, 1] < margin
        clamp_now = int(np.sum(idx_l) + np.sum(idx_r) + np.sum(idx_b))
        self.step_clamp_count = clamp_now
        self.total_clamp_activations += clamp_now

        # --- BƯỚC 2 (Tính lực Navier-Stokes tại vị trí mới r^{n+1}) ---
        self._evaluate_derivatives(a_y_tank, a_z_tank, roll_angle, roll_rate, roll_acc)

        # --- BƯỚC 3 (Complete Second Half-step Kick) ---
        self.vel[:n_f] += 0.5 * dt * self.acc[:n_f]
        self.rho[:n_f] = np.clip(self.rho[:n_f] + 0.5 * dt * self.drho_dt[:n_f], 0.95 * self.rho_0, 1.08 * self.rho_0)

        self.time += dt
        self.total_steps += 1
        if not compute_metrics:
            return 0.0, 0.0, 0.0, self.m_roll_3d, self.liquid_height, self.liquid_height

        # Trích xuất các đại lượng vật lý 3D
        pos_f = self.pos[:n_f]

        # 1. Gia tốc tương đối khối tâm trong hệ quy chiếu bình: a_rel_y = self.acc[:n_f, 0]
        a_rel_y = self.acc[:n_f, 0]
        # Lực sóng sánh dư tính qua bảo toàn động lượng khối tâm
        f_slosh_y = -self.total_mass * float(np.mean(a_rel_y))
        # Tổng lực ngang gồm quán tính vật rắn và lực sóng sánh
        f_y_total = -self.total_mass * a_y_tank + f_slosh_y
        f_z_total = self.total_mass * self.g

        # 2. Mô-men lật thủy động trích xuất trực tiếp từ phân bố lực và vị trí hạt theo hệ quy chiếu WP01
        self.m_roll_3d = self.tank_length * float(np.sum(
            self.mass_2d * (-pos_f[:, 0] * self.g + pos_f[:, 1] * (a_y_tank + a_rel_y))
        ))
        m_roll = self.m_roll_3d

        # 3. Chiều cao mặt thoáng tại hai vách (+Y: vách Trái, -Y: vách Phải)
        strip = 2.0 * self.dp
        right_zone = pos_f[:, 0] < (-b_half + strip)  # y < 0: Vách Phải (Right)
        left_zone = pos_f[:, 0] > (b_half - strip)   # y > 0: Vách Trái (Left)
        eta_right = self._fast_percentile_90(pos_f[right_zone, 1], self.liquid_height)
        eta_left = self._fast_percentile_90(pos_f[left_zone, 1], self.liquid_height)

        return f_y_total, f_slosh_y, f_z_total, m_roll, eta_left, eta_right

    def save_checkpoint(self) -> dict:
        """
        Lưu toàn bộ trạng thái hoàn chỉnh của hệ SPH (vị trí, vận tốc, mật độ, áp suất, thời gian, gia tốc...).
        Bảo đảm tính tái lập độc lập 100% khi rẽ nhánh phanh T2 (Lỗi A.1).
        """
        return {
            'pos': np.copy(self.pos),
            'vel': np.copy(self.vel),
            'rho': np.copy(self.rho),
            'p': np.copy(self.p),
            'acc': np.copy(self.acc),
            'drho_dt': np.copy(self.drho_dt),
            'time': float(self.time),
            'total_steps': int(self.total_steps),
            'f_wall_2d': np.copy(self.f_wall_2d),
            'm_roll_3d': float(self.m_roll_3d),
            'total_clamp_activations': int(self.total_clamp_activations),
            'step_clamp_count': int(self.step_clamp_count),
            '_initialized_forces': bool(self._initialized_forces),
            '_steps_since_tree': int(self._steps_since_tree),
            '_cached_pairs': np.copy(self._cached_pairs) if self._cached_pairs is not None else None,
            '_tree_pos_ref': np.copy(self._tree_pos_ref) if self._tree_pos_ref is not None else None
        }

    def load_checkpoint(self, state: dict):
        """
        Nạp lại trạng thái đã lưu để rẽ nhánh phanh T2 hoặc phục hồi trạng thái (Lỗi A.1).
        """
        self.pos = np.copy(state['pos'])
        self.vel = np.copy(state['vel'])
        self.rho = np.copy(state['rho'])
        self.p = np.copy(state['p'])
        self.acc = np.copy(state['acc'])
        self.drho_dt = np.copy(state['drho_dt'])
        self.time = float(state['time'])
        self.total_steps = int(state['total_steps'])
        self.f_wall_2d = np.copy(state['f_wall_2d'])
        self.m_roll_3d = float(state['m_roll_3d'])
        self.total_clamp_activations = int(state['total_clamp_activations'])
        self.step_clamp_count = int(state['step_clamp_count'])
        self._initialized_forces = bool(state['_initialized_forces'])
        self._steps_since_tree = int(state['_steps_since_tree'])
        self._cached_pairs = np.copy(state['_cached_pairs']) if state.get('_cached_pairs') is not None else None
        self._tree_pos_ref = np.copy(state['_tree_pos_ref']) if state.get('_tree_pos_ref') is not None else None


    def compute_relative_mechanical_energy(self) -> float:
        """
        Tính năng lượng cơ học tương đối của chất lỏng trong hệ quy chiếu gắn bình (Lỗi A.7):
        E_fl_rel(t) = sum_a m_a * ( 0.5 * ||v_rel_a||^2 + g * z_rel_a + e_int_a ) * tank_length
        """
        n_f = self.n_fluid
        pos_f = self.pos[:n_f]
        vel_f = self.vel[:n_f]
        rho_f = self.rho[:n_f]
        p_f = self.p[:n_f]

        ke_2d = 0.5 * np.sum(vel_f[:, 0]**2 + vel_f[:, 1]**2) * self.mass_2d
        pe_2d = self.g * np.sum(pos_f[:, 1]) * self.mass_2d
        # Nội năng nén Tait e_int ≈ p / (rho * (gamma - 1))
        ie_2d = np.sum(p_f / (np.maximum(rho_f, 1e-3) * (self.gamma - 1.0))) * self.mass_2d
        e_total_3d = (ke_2d + pe_2d + ie_2d) * self.tank_length
        return float(e_total_3d)



class HousnerModel:
    """
    Mô hình con lắc tương đương Housner DMTP chuẩn giải tích
    phục vụ đối chiếu chéo với CFD Navier-Stokes.
    """
    def __init__(
        self,
        tank_width=0.100,
        tank_height=0.110,
        liquid_height=0.080,
        tank_length=0.150,
        m_liquid=1.200,
        zeta=0.015,
        g=9.81
    ):
        self.b = tank_width
        self.H = tank_height
        self.h_l = liquid_height
        self.a = tank_length
        self.m_tot = m_liquid
        self.zeta = zeta
        self.g = g

        tanh_val = np.tanh(np.pi * self.h_l / self.b)
        self.omega_n = np.sqrt((np.pi * self.g / self.b) * tanh_val)
        self.f_n = self.omega_n / (2.0 * np.pi)
        self.l_p = self.b / (np.pi * tanh_val)
        self.m_1 = self.m_tot * (8.0 * self.b / (np.pi**3 * self.h_l)) * tanh_val
        self.m_0 = self.m_tot - self.m_1
        self.h_0 = 0.5 * self.h_l
        # Chiều cao h1* tính mô-men lật có xét hiệu ứng áp suất đáy bình (Graham & Rodriguez 1952, Housner 1957)
        # Đồng bộ chính xác với z_slosh_y trong solver/robot_model_amr_casadi.py (0.0581 m)
        self.h_1 = 0.0581

    def simulate(self, t_arr, a_y_arr):
        """
        Tích phân phương trình vi phân con lắc Housner bằng RK4 bậc 4:
        ddtheta + 2*zeta*omega_n*dtheta + omega_n^2*sin(theta) = - (1/l_p)*a_y
        """
        dt = t_arr[1] - t_arr[0]
        n = len(t_arr)
        theta = np.zeros(n)
        dtheta = np.zeros(n)
        ddtheta = np.zeros(n)

        def rhs(th, dth, ay):
            return -2.0 * self.zeta * self.omega_n * dth - (self.omega_n**2) * np.sin(th) - (1.0 / self.l_p) * ay

        for i in range(n - 1):
            ay_curr = a_y_arr[i]
            ay_next = a_y_arr[i + 1]
            ay_mid = 0.5 * (ay_curr + ay_next)

            ddtheta[i] = rhs(theta[i], dtheta[i], ay_curr)

            k1_th = dtheta[i]
            k1_dth = ddtheta[i]

            k2_th = dtheta[i] + 0.5 * dt * k1_dth
            k2_dth = rhs(theta[i] + 0.5 * dt * k1_th, k2_th, ay_mid)

            k3_th = dtheta[i] + 0.5 * dt * k2_dth
            k3_dth = rhs(theta[i] + 0.5 * dt * k2_th, k3_th, ay_mid)

            k4_th = dtheta[i] + dt * k3_dth
            k4_dth = rhs(theta[i] + dt * k3_th, k4_th, ay_next)

            theta[i + 1] = theta[i] + (dt / 6.0) * (k1_th + 2.0 * k2_th + 2.0 * k3_th + k4_th)
            dtheta[i + 1] = dtheta[i] + (dt / 6.0) * (k1_dth + 2.0 * k2_dth + 2.0 * k3_dth + k4_dth)

        ddtheta[-1] = rhs(theta[-1], dtheta[-1], a_y_arr[-1])

        acc_pend_rel = self.l_p * ddtheta * np.cos(theta) - self.l_p * (dtheta**2) * np.sin(theta)
        F_y_housner = -(self.m_0 * a_y_arr + self.m_1 * (a_y_arr + acc_pend_rel))
        F_slosh_housner = F_y_housner - (-self.m_tot * a_y_arr)

        # Mô-men lật quanh tâm đáy bình theo quy ước WP01:
        # a_y > 0 -> theta < 0 (con lắc lệch sang phải y < 0)
        # Trọng lực lệch phải tạo mô-men dương: - m1 * g * l_p * sin(theta) > 0
        # Quán tính d'Alembert đẩy sang phải tạo mô-men dương: m0*ay*h0 + m1*(ay + a_rel)*h1 > 0
        M_roll_housner = (self.m_0 * a_y_arr * self.h_0 +
                          self.m_1 * (a_y_arr + acc_pend_rel) * self.h_1 -
                          self.m_1 * self.g * self.l_p * np.sin(theta))

        # Độ dâng mặt thoáng tại vách phải (-Y): khi a_y > 0 -> theta < 0 -> mặt thoáng dâng lên (mm)
        eta_housner_right_mm = -(self.b / 2.0) * np.sin(theta) * 1000.0

        return {
            'theta': theta,
            'dtheta': dtheta,
            'ddtheta': ddtheta,
            'F_y': F_y_housner,
            'F_slosh': F_slosh_housner,
            'M_roll': M_roll_housner,
            'eta_mm': eta_housner_right_mm
        }
