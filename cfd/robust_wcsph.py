import numpy as np
import scipy.signal
from scipy.spatial import cKDTree

class RobustWCSPH2D:
    def __init__(
        self,
        tank_width=0.100,       # b = 100 mm
        tank_height=0.110,      # H = 110 mm
        liquid_height=0.080,    # h_l = 80 mm
        tank_length=0.150,      # L_x = 150 mm
        dp=0.0035,              # 3.5 mm
        rho_0=1000.0,
        c_0=10.0,
        gamma=7.0,
        alpha_visc=0.05,
        delta_sph=0.10,
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
        self.delta_sph = delta_sph
        self.g = g

        self.h = 1.35 * dp
        self.r_cut = 2.0 * self.h
        self.B_tait = (rho_0 * c_0**2) / gamma
        self.alpha_d = 7.0 / (4.0 * np.pi * self.h**2)

        # 1. Hạt chất lỏng
        n_y = max(1, int(np.round(tank_width / dp)))
        n_z = max(1, int(np.round(liquid_height / dp)))
        dy = tank_width / n_y
        dz = liquid_height / n_z

        y_f = np.linspace(-tank_width / 2.0 + dy / 2.0, tank_width / 2.0 - dy / 2.0, n_y)
        z_f = np.linspace(dz / 2.0, liquid_height - dz / 2.0, n_z)
        yy, zz = np.meshgrid(y_f, z_f)
        self.pos_f = np.column_stack((yy.flatten(), zz.flatten()))
        self.n_fluid = len(self.pos_f)

        self.area_p = (tank_width * liquid_height) / self.n_fluid
        self.mass_2d = rho_0 * self.area_p
        self.mass_3d = self.mass_2d * tank_length
        self.total_mass = self.n_fluid * self.mass_3d

        # 2. Hạt biên mDBC
        n_layers = 3
        b_pts = []
        y_bot = np.arange(-tank_width / 2.0 - n_layers * dp + dp / 2.0, tank_width / 2.0 + n_layers * dp, dp)
        for l in range(1, n_layers + 1):
            zv = -(l - 0.5) * dp
            for yv in y_bot:
                b_pts.append([yv, zv])
        z_wall = np.arange(dp / 2.0, tank_height + n_layers * dp, dp)
        for l in range(1, n_layers + 1):
            y_l = -tank_width / 2.0 - (l - 0.5) * dp
            y_r = tank_width / 2.0 + (l - 0.5) * dp
            for zv in z_wall:
                b_pts.append([y_l, zv])
                b_pts.append([y_r, zv])

        self.pos_b = np.array(b_pts)
        self.n_bound = len(self.pos_b)
        self.n_total = self.n_fluid + self.n_bound

        self.pos = np.vstack((self.pos_f, self.pos_b))
        self.vel = np.zeros_like(self.pos)
        self.rho = np.full(self.n_total, rho_0)
        self.p = np.zeros(self.n_total)

        p_hydro = np.maximum(rho_0 * g * (liquid_height - self.pos_f[:, 1]), 0.0)
        self.p[:self.n_fluid] = p_hydro
        self.rho[:self.n_fluid] = rho_0 * (p_hydro / self.B_tait + 1.0)**(1.0 / gamma)

        self.acc = np.zeros_like(self.pos)
        self.drho_dt = np.zeros(self.n_total)
        self.time = 0.0
        self.total_steps = 0

    def _eval_forces(self, ay):
        n_f = self.n_fluid
        n_tot = self.n_total

        tree = cKDTree(self.pos)
        pairs = tree.query_pairs(self.r_cut, output_type='ndarray')
        if len(pairs) == 0:
            self.acc.fill(0.0)
            self.drho_dt.fill(0.0)
            return

        i_idx = pairs[:, 0]
        j_idx = pairs[:, 1]
        r_ij = self.pos[i_idx] - self.pos[j_idx]
        dist_sq = r_ij[:, 0]**2 + r_ij[:, 1]**2
        dist = np.sqrt(dist_sq)

        q = dist / self.h
        omhq = 1.0 - 0.5 * q
        omhq3 = omhq**3
        W_pair = self.alpha_d * (omhq3 * omhq) * (1.0 + 2.0 * q)
        grad_factor = (self.alpha_d * (-5.0 / (self.h**2))) * omhq3
        grad_W_y = r_ij[:, 0] * grad_factor
        grad_W_z = r_ij[:, 1] * grad_factor

        rho_ratio = np.clip(self.rho[:n_f] / self.rho_0, 0.95, 1.08)
        self.p[:n_f] = np.maximum(self.B_tait * (rho_ratio**self.gamma - 1.0), 0.0)

        # mDBC extrapolation
        fb_mask = (i_idx < n_f) & (j_idx >= n_f)
        if np.any(fb_mask):
            fid = i_idx[fb_mask]
            bid = j_idx[fb_mask]
            w_fb = W_pair[fb_mask]
            dr_bf_y = -r_ij[fb_mask, 0]
            dr_bf_z = -r_ij[fb_mask, 1]
            hydro_corr = self.rho_0 * (-ay * dr_bf_y - self.g * dr_bf_z)
            p_extrap = np.maximum(self.p[fid] + hydro_corr, 0.0)

            b_num = np.bincount(bid, weights=w_fb * p_extrap, minlength=n_tot)
            b_den = np.bincount(bid, weights=w_fb, minlength=n_tot)
            val_b = b_den[n_f:] > 1e-12
            p_b = np.zeros(self.n_bound)
            p_b[val_b] = b_num[n_f:][val_b] / b_den[n_f:][val_b]
            self.p[n_f:] = p_b
            self.rho[n_f:] = self.rho_0 * (p_b / self.B_tait + 1.0)**(1.0 / self.gamma)

        # Continuity
        v_ij_y = self.vel[i_idx, 0] - self.vel[j_idx, 0]
        v_ij_z = self.vel[i_idx, 1] - self.vel[j_idx, 1]
        v_dot_gradW = v_ij_y * grad_W_y + v_ij_z * grad_W_z
        cont_term = self.mass_2d * v_dot_gradW
        self.drho_dt = (np.bincount(i_idx, weights=cont_term, minlength=n_tot) +
                        np.bincount(j_idx, weights=cont_term, minlength=n_tot))

        # delta-SPH
        if self.delta_sph > 0.0:
            ff_mask = (i_idx < n_f) & (j_idx < n_f)
            if np.any(ff_mask):
                fi = i_idx[ff_mask]
                fj = j_idx[ff_mask]
                r_dot_gradW = dist_sq[ff_mask] * grad_factor[ff_mask]
                rho_bar = 0.5 * (self.rho[fi] + self.rho[fj])
                diff = (2.0 * self.delta_sph * self.h * self.c_0 * self.mass_2d *
                        (self.rho[fi] - self.rho[fj]) / rho_bar *
                        r_dot_gradW / (dist_sq[ff_mask] + 0.01 * self.h**2))
                self.drho_dt += np.bincount(fi, weights=diff, minlength=n_tot)
                self.drho_dt -= np.bincount(fj, weights=diff, minlength=n_tot)

        # Momentum
        p_rho2_i = self.p[i_idx] / (self.rho[i_idx]**2)
        p_rho2_j = self.p[j_idx] / (self.rho[j_idx]**2)
        p_factor = -(p_rho2_i + p_rho2_j)

        v_dot_r = v_ij_y * r_ij[:, 0] + v_ij_z * r_ij[:, 1]
        visc_factor = np.zeros_like(p_factor)
        appr = v_dot_r < 0.0
        if np.any(appr):
            mu_ij = (self.h * v_dot_r[appr]) / (dist_sq[appr] + 0.01 * self.h**2)
            rho_b = 0.5 * (self.rho[i_idx[appr]] + self.rho[j_idx[appr]])
            visc_factor[appr] = self.alpha_visc * self.c_0 * mu_ij / rho_b

        total_factor = self.mass_2d * (p_factor + visc_factor)
        f_pair_y = total_factor * grad_W_y
        f_pair_z = total_factor * grad_W_z

        acc_y = (np.bincount(i_idx, weights=f_pair_y, minlength=n_tot) -
                 np.bincount(j_idx, weights=f_pair_y, minlength=n_tot))
        acc_z = (np.bincount(i_idx, weights=f_pair_z, minlength=n_tot) -
                 np.bincount(j_idx, weights=f_pair_z, minlength=n_tot))

        # Soft physical cushion at wall boundary (d < 0.8 dp)
        b_half = self.tank_width / 2.0
        pos_f = self.pos[:n_f]
        d_left = pos_f[:, 0] - (-b_half)
        d_right = b_half - pos_f[:, 0]
        d_bot = pos_f[:, 1]
        thresh = 0.28 * self.dp
        
        rep_y = np.zeros(n_f)
        rep_z = np.zeros(n_f)
        k_rep = 0.25 * (self.c_0**2) / (self.dp)

        mask_l = (d_left > 0) & (d_left < thresh)
        mask_r = (d_right > 0) & (d_right < thresh)
        mask_b = (d_bot > 0) & (d_bot < thresh)

        if np.any(mask_l):
            rep_y[mask_l] += k_rep * ((thresh - d_left[mask_l]) / thresh)**2
        if np.any(mask_r):
            rep_y[mask_r] -= k_rep * ((thresh - d_right[mask_r]) / thresh)**2
        if np.any(mask_b):
            rep_z[mask_b] += k_rep * ((thresh - d_bot[mask_b]) / thresh)**2

        self.acc[:n_f, 0] = acc_y[:n_f] - ay + rep_y
        self.acc[:n_f, 1] = acc_z[:n_f] - self.g + rep_z
        self.acc[n_f:] = 0.0

    def step(self, dt, ay):
        n_f = self.n_fluid
        if self.total_steps == 0:
            self._eval_forces(ay)

        self.vel[:n_f] += 0.5 * dt * self.acc[:n_f]
        self.rho[:n_f] = np.clip(self.rho[:n_f] + 0.5 * dt * self.drho_dt[:n_f], 0.95 * self.rho_0, 1.08 * self.rho_0)
        self.pos[:n_f] += dt * self.vel[:n_f]

        self._eval_forces(ay)

        self.vel[:n_f] += 0.5 * dt * self.acc[:n_f]
        self.rho[:n_f] = np.clip(self.rho[:n_f] + 0.5 * dt * self.drho_dt[:n_f], 0.95 * self.rho_0, 1.08 * self.rho_0)

        # Trích xuất trọng tâm khối chất lỏng y_cm
        pos_f = self.pos[:n_f]
        y_cm = float(np.mean(pos_f[:, 0]))

        # Chiều cao mặt thoáng sóng sánh
        b_half = self.tank_width / 2.0
        # 1.5 dp slice ở hai bên vách
        idx_l = pos_f[:, 0] < (-b_half + 2.0 * self.dp)
        idx_r = pos_f[:, 0] > (b_half - 2.0 * self.dp)
        zl = float(np.percentile(pos_f[idx_l, 1], 90)) if np.any(idx_l) else self.liquid_height
        zr = float(np.percentile(pos_f[idx_r, 1], 90)) if np.any(idx_r) else self.liquid_height
        # Chiều cao tương đối so với mực nước tĩnh h_l:
        # Khi sóng dâng bên phải, zr > h_l, zl < h_l => eta = 0.5 * (zr - zl)
        eta_slosh_mm = 0.5 * (zr - zl) * 1000.0

        self.time += dt
        self.total_steps += 1
        return y_cm, eta_slosh_mm
