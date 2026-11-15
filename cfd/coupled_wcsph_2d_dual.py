"""
coupled_wcsph_2d_dual.py
=============================================================================
BỘ GHÉP NỐI 2 LÁT CẮT WCSPH 2D x 2 TRỰC GIAO (X-Z và Y-Z)
(Phục vụ Đề tài DFE-MPCC AMR Chở Chất Lỏng - Định hướng SIMPAT v2.0)

Cung cấp:
1. Hai bộ giải WCSPH 2D độc lập:
   - solver_xz: Lát cắt X-Z (dài Lx = 0.15m, bề rộng Ly = 0.10m) chịu gia tốc ax.
   - solver_yz: Lát cắt Y-Z (rộng Ly = 0.10m, bề rộng Lx = 0.15m) chịu gia tốc ay.
2. Lưu / Nạp checkpoint đồng thời trên cả 2 lát cắt (Lỗi A.1).
3. Ghép nối với CoSimMassMomentBridge để xuất lực thuần động dFx, dFy,
   mô-men dM_pitch_ground, dM_roll_ground và tải 4 bánh dFz_slosh.
=============================================================================
"""

import numpy as np
from cfd.sph_navier_stokes_engine import WCSPH2DSolver
from simulation.cosim_bridge import CoSimMassMomentBridge


class DualSliceWCSPH:
    def __init__(
        self,
        Lx: float = 0.150,
        Ly: float = 0.100,
        tank_height: float = 0.110,
        h: float = 0.080,
        z_tank: float = 0.052,
        l_wb: float = 0.240,
        t_w: float = 0.280,
        dp: float = 0.0050,  # mặc định cho chạy ghép nối nhanh
        rho_0: float = 1000.0,
        g: float = 9.81
    ):
        self.Lx = Lx
        self.Ly = Ly
        self.h = h
        self.g = g

        # Lát cắt X-Z: bình có chiều rộng trong mặt phẳng là Lx, bề rộng chiếu là Ly
        self.solver_xz = WCSPH2DSolver(
            tank_width=Lx,
            tank_height=tank_height,
            liquid_height=h,
            tank_length=Ly,
            dp=dp,
            rho_0=rho_0,
            g=g
        )

        # Lát cắt Y-Z: bình có chiều rộng trong mặt phẳng là Ly, bề rộng chiếu là Lx
        self.solver_yz = WCSPH2DSolver(
            tank_width=Ly,
            tank_height=tank_height,
            liquid_height=h,
            tank_length=Lx,
            dp=dp,
            rho_0=rho_0,
            g=g
        )

        self.bridge = CoSimMassMomentBridge(
            rho=rho_0,
            Lx=Lx,
            Ly=Ly,
            h=h,
            z_tank=z_tank,
            l_wb=l_wb,
            t_w=t_w,
            g=g
        )

        self.time = 0.0

    def step(self, dt: float, ax: float, ay: float, az: float = 0.0) -> dict:
        """
        Tích phân 1 bước cho cả 2 lát cắt (tự động chia nhỏ theo điều kiện CFL)
        và trích xuất Wrench thuần động qua Cầu nối.
        """
        import math

        # Bảo đảm bước tích phân sub-step luôn thỏa điều kiện CFL của cả 2 lát cắt
        dt_cfl_safe = min(self.solver_xz.compute_cfl_dt(), self.solver_yz.compute_cfl_dt())
        n_sub = max(1, int(math.ceil(dt / max(dt_cfl_safe, 1e-7) - 1e-9)))
        sub_dt = dt / float(n_sub)

        eta_x_l = eta_x_r = eta_y_l = eta_y_r = 0.0
        for _ in range(n_sub):
            # 1. Bước cho lát cắt X-Z (chịu gia tốc dọc trục ax)
            # Trong solver_xz, trục ngang trong mặt phẳng 2D đóng vai trò trục x
            fx_tot_xz, fx_slosh_xz, fz_tot_xz, m_pitch_xz, eta_x_l, eta_x_r = self.solver_xz.step(
                sub_dt, a_y_tank=ax, a_z_tank=az
            )

            # 2. Bước cho lát cắt Y-Z (chịu gia tốc ngang trục ay)
            fy_tot_yz, fy_slosh_yz, fz_tot_yz, m_roll_yz, eta_y_l, eta_y_r = self.solver_yz.step(
                sub_dt, a_y_tank=ay, a_z_tank=az
            )

        # 3. Lực thô trên một đơn vị bề rộng (N/m) và mô-men (N*m/m) quanh tâm đáy bình:
        # Ở đây f_wall_2d của solver là lực tác dụng lên thành bình 2D [N/m]
        fx_raw_xz = self.solver_xz.f_wall_2d[0]
        fz_raw_xz = self.solver_xz.f_wall_2d[1]
        m_bottom_raw_xz = self.solver_xz.m_roll_3d / self.Ly  # quy về đơn vị N*m/m

        fy_raw_yz = self.solver_yz.f_wall_2d[0]
        fz_raw_yz = self.solver_yz.f_wall_2d[1]
        m_bottom_raw_yz = self.solver_yz.m_roll_3d / self.Lx

        # 4. Trừ phần quán tính vật rắn và tính Wrench mặt đất qua Cầu nối
        wrench_dict = self.bridge.compute_pure_dynamic_slosh_wrench(
            fx_raw_xz=fx_raw_xz, fz_raw_xz=fz_raw_xz, m_bottom_raw_xz=m_bottom_raw_xz,
            fy_raw_yz=fy_raw_yz, fz_raw_yz=fz_raw_yz, m_bottom_raw_yz=m_bottom_raw_yz,
            ax=ax, ay=ay, az=az
        )

        self.time += dt

        # Bổ sung thông tin mặt thoáng trực tiếp & chỉ số sub-stepping CFL
        surf_xz = self.solver_xz.get_free_surface_metrics()
        surf_yz = self.solver_yz.get_free_surface_metrics()
        wrench_dict.update({
            "time": self.time,
            "eta_x_left": eta_x_l,
            "eta_x_right": eta_x_r,
            "eta_y_left": eta_y_l,
            "eta_y_right": eta_y_r,
            "eta_peak_x": max(eta_x_l, eta_x_r),
            "eta_peak_y": max(eta_y_l, eta_y_r),
            "eta_surf_x_mm": surf_xz["eta_direct_mm"],
            "eta_surf_y_mm": surf_yz["eta_direct_mm"],
            "eta_wall_x_mm": surf_xz["eta_wall_mm"],
            "eta_wall_y_mm": surf_yz["eta_wall_mm"],
            "eta_cm_x_mm": surf_xz["eta_cm_mm"],
            "eta_cm_y_mm": surf_yz["eta_cm_mm"],
            "cfl_substeps": n_sub,
            "dt_substep": sub_dt,
        })

        return wrench_dict

    def integrate_window(
        self,
        n_steps: int,
        dt: float,
        ax: float,
        ay: float,
        az: float = 0.0,
        ramp_steps: int = 250,
    ) -> dict:
        """
        Tích phân cửa sổ thời gian dài (ví dụ 1500-3000 bước CFL ~ 90-180 ms, tương ứng 1/4 - 1/2 chu kỳ sóng T_1y),
        đo trực tiếp biên độ dâng mặt thoáng hạt (eta_surf_mm, eta_cm_mm) kết hợp áp suất động thành bình (Hybrid B+A).
        """
        peak_surf_x = 0.0
        peak_surf_y = 0.0
        peak_wall_x = 0.0
        peak_wall_y = 0.0
        peak_cm_x = 0.0
        peak_cm_y = 0.0
        peak_fx_wall = 0.0
        peak_fy_wall = 0.0
        res_last = None
        r_steps = max(1, int(ramp_steps))

        dt_cfl_safe = min(self.solver_xz.compute_cfl_dt(), self.solver_yz.compute_cfl_dt())
        n_sub = max(1, int(np.ceil(dt / max(dt_cfl_safe, 1e-7) - 1e-9)))
        sub_dt = dt / float(n_sub)

        for step_i in range(n_steps):
            alpha_s = float(np.sin(0.5 * np.pi * min(1.0, (step_i + 1) / float(r_steps))))
            ax_cur = ax * alpha_s
            ay_cur = ay * alpha_s
            is_last = (step_i == n_steps - 1)
            if is_last:
                res_last = self.step(dt, ax=ax_cur, ay=ay_cur, az=az)
            else:
                for _ in range(n_sub):
                    self.solver_xz.step(sub_dt, a_y_tank=ax_cur, a_z_tank=az, compute_metrics=False)
                    self.solver_yz.step(sub_dt, a_y_tank=ay_cur, a_z_tank=az, compute_metrics=False)
                self.time += dt
            if (step_i + 1) % 20 == 0 or is_last:
                surf_xz = self.solver_xz.get_free_surface_metrics()
                surf_yz = self.solver_yz.get_free_surface_metrics()
                peak_surf_x = max(peak_surf_x, surf_xz["eta_direct_mm"])
                peak_surf_y = max(peak_surf_y, surf_yz["eta_direct_mm"])
                peak_wall_x = max(peak_wall_x, surf_xz["eta_wall_mm"])
                peak_wall_y = max(peak_wall_y, surf_yz["eta_wall_mm"])
                peak_cm_x = max(peak_cm_x, surf_xz["eta_cm_mm"])
                peak_cm_y = max(peak_cm_y, surf_yz["eta_cm_mm"])
                peak_fx_wall = max(peak_fx_wall, abs(self.solver_xz.f_wall_2d[0] * self.Ly))
                peak_fy_wall = max(peak_fy_wall, abs(self.solver_yz.f_wall_2d[0] * self.Lx))

        if res_last is None:
            res_last = self.step(dt, ax=ax, ay=ay, az=az)
        res_last.update({
            "window_steps": int(n_steps),
            "window_time_ms": round(float(n_steps * dt * 1000.0), 2),
            "peak_eta_surf_x_mm": peak_surf_x,
            "peak_eta_surf_y_mm": peak_surf_y,
            "peak_eta_wall_x_mm": peak_wall_x,
            "peak_eta_wall_y_mm": peak_wall_y,
            "peak_eta_cm_x_mm": peak_cm_x,
            "peak_eta_cm_y_mm": peak_cm_y,
            "peak_fx_wall_N": peak_fx_wall,
            "peak_fy_wall_N": peak_fy_wall,
        })
        return res_last

    def save_checkpoint(self) -> dict:
        """
        Lưu trạng thái đồng thời cả 2 lát cắt SPH.
        """
        return {
            "time": float(self.time),
            "ckpt_xz": self.solver_xz.save_checkpoint(),
            "ckpt_yz": self.solver_yz.save_checkpoint()
        }

    def load_checkpoint(self, ckpt: dict):
        """
        Nạp lại trạng thái đồng thời cho cả 2 lát cắt SPH.
        """
        self.time = float(ckpt["time"])
        self.solver_xz.load_checkpoint(ckpt["ckpt_xz"])
        self.solver_yz.load_checkpoint(ckpt["ckpt_yz"])

    def evaluate_hybrid_ba_checkpoint(
        self,
        base_ckpt: dict,
        dt: float,
        ax_exc: float,
        ay_exc: float,
        a_brk_x: float,
        a_brk_y: float,
        ay_prev_geom: float = 0.0,
        gamma_link: float = 0.0,
        phi_link_rad: float = 0.0,
        delta_psi_rad: float = 0.0,
        t1_window_s: float = 0.150,
        t2_window_s: float = 0.090,
    ) -> dict:
        """
        Đánh giá một Checkpoint theo chuẩn Hybrid B+A cửa sổ dài:
        - Tích phân cửa sổ T1 (mặc định 150 ms ~ 0.42 T_1y) đo trực tiếp độ dâng mặt thoáng hạt.
        - Lưu checkpoint RAM, quét pha xấu nhất worst_phase_gain và nhân trực tiếp vào carryover.
        - Rẽ nhánh phanh phản thực tế T2 (mặc định 90 ms) đo trực tiếp sóng dâng khi phanh khẩn cấp.
        Dùng chung một nguồn sự thật vật lý cho Stage B và Hiệu chuẩn Conformal ngoại tuyến.
        """
        import time as _time
        g = float(self.solver_yz.g)
        lx = float(self.Lx)
        ly = float(self.Ly)
        h_liq = float(self.h)
        dp_mm = float(self.solver_yz.dp * 1000.0)
        k1_x = float(np.pi / max(lx, 1e-6))
        k1_y = float(np.pi / max(ly, 1e-6))
        coth_x = 1.0 / max(float(np.tanh(k1_x * h_liq)), 1e-6)
        coth_y = 1.0 / max(float(np.tanh(k1_y * h_liq)), 1e-6)

        # 1. Cửa sổ dài T1 (Hybrid B+A)
        self.load_checkpoint(base_ckpt)
        n1_steps = max(1, int(round(t1_window_s / dt)))
        ramp_steps = max(1, int(round(0.015 / dt)))
        res_t1 = self.integrate_window(n1_steps, dt, ax=ax_exc, ay=ay_exc, ramp_steps=ramp_steps)

        t_ram0 = _time.perf_counter()
        ckpt_ram = self.save_checkpoint()
        save_ram_ms = (_time.perf_counter() - t_ram0) * 1000.0

        # 2. Quét pha xấu nhất (Worst-Case Carryover Phase Sweep) trên nhánh bảo toàn trạng thái
        fy_t1_peak = max(float(res_t1["peak_fy_wall_N"]), 1e-4)
        sy_t1_base = max(float(res_t1["peak_eta_surf_y_mm"]), 0.1)
        n_phase = max(20, int(round(0.004 / dt)))
        worst_phase_gain = 1.0
        for phase_sign in (+1.0, -1.0):
            self.load_checkpoint(ckpt_ram)
            wp_res = self.integrate_window(
                n_phase, dt,
                ax=ax_exc,
                ay=ay_exc + phase_sign * 0.15 * gamma_link * max(abs(ay_exc), 0.1),
                ramp_steps=1,
            )
            worst_phase_gain = max(
                worst_phase_gain,
                float(wp_res["peak_fy_wall_N"]) / fy_t1_peak,
                float(wp_res["peak_eta_surf_y_mm"]) / sy_t1_base,
            )
        self.load_checkpoint(ckpt_ram)

        # 3. Hợp nhất mặt thoáng hạt trực tiếp (85%) & áp suất động thành bình (15%) + carryover nhân worst_phase_gain
        cw_x_peak = float(res_t1["peak_fx_wall_N"]) / max(self.solver_xz.total_mass * g, 1e-6)
        cw_y_peak = float(res_t1["peak_fy_wall_N"]) / max(self.solver_yz.total_mass * g, 1e-6)
        pol_sign = 1.0 if (ay_exc * ay_prev_geom) >= 0.0 else -1.0
        u_carry_amp = 0.30 * gamma_link * (abs(ay_prev_geom) / g) * worst_phase_gain

        tan_th_y_dir = (float(res_t1["peak_eta_surf_y_mm"]) * 1e-3) / max(0.5 * ly, 1e-6)
        u_re = tan_th_y_dir * (1.0 + 0.04 * min(1.0, abs(delta_psi_rad) / (0.25 * np.pi))) + pol_sign * u_carry_amp * float(np.cos(phi_link_rad))
        u_im = u_carry_amp * float(np.sin(phi_link_rad))
        tan_th_y_hyb = float(max(0.85 * u_carry_amp, np.hypot(u_re, u_im)))
        tan_th_y_final = 0.85 * tan_th_y_hyb + 0.15 * max(tan_th_y_hyb, cw_y_peak + abs(u_carry_amp * float(np.cos(phi_link_rad))))

        tan_th_x_dir = ((float(res_t1["peak_eta_surf_x_mm"]) * 1e-3) / max(0.5 * lx, 1e-6)) * (1.0 + 0.10 * gamma_link)
        tan_th_x_final = 0.85 * tan_th_x_dir + 0.15 * max(tan_th_x_dir, cw_x_peak)

        eta_y_t1_mm = (0.5 * ly * tan_th_y_final + (np.pi * ly / 8.0) * coth_y * (tan_th_y_final ** 2)) * 1000.0
        eta_x_t1_mm = (0.5 * lx * tan_th_x_final + (np.pi * lx / 8.0) * coth_x * (tan_th_x_final ** 2)) * 1000.0
        meniscus_mm = min(0.25, 0.04 * dp_mm)
        eta_wcsph_t1_mm = max(0.5, float(eta_y_t1_mm + 0.50 * eta_x_t1_mm + meniscus_mm))

        # 4. Nhánh phanh phản thực tế T2 (Counterfactual Emergency Braking) trên trạng thái đã có sóng T1
        n2_steps = max(1, int(round(t2_window_s / dt)))
        res_t2 = self.integrate_window(n2_steps, dt, ax=a_brk_x, ay=a_brk_y, ramp_steps=1)

        tan_t2_y = max(tan_th_y_final * 0.85, (float(res_t2["peak_eta_surf_y_mm"]) * 1e-3) / max(0.5 * ly, 1e-6))
        tan_t2_x_dir = (float(res_t2["peak_eta_surf_x_mm"]) * 1e-3) / max(0.5 * lx, 1e-6)
        cw_t2_x = float(res_t2["peak_fx_wall_N"]) / max(self.solver_xz.total_mass * g, 1e-6)
        tan_t2_x_final = 0.85 * tan_t2_x_dir + 0.15 * max(tan_t2_x_dir, cw_t2_x)

        eta_y_t2_mm = (0.5 * ly * tan_t2_y + (np.pi * ly / 8.0) * coth_y * (tan_t2_y ** 2)) * 1000.0
        eta_x_t2_mm = (0.5 * lx * tan_t2_x_final + (np.pi * lx / 8.0) * coth_x * (tan_t2_x_final ** 2)) * 1000.0
        eta_wcsph_t2_mm = max(eta_wcsph_t1_mm, float(eta_y_t2_mm + 0.70 * eta_x_t2_mm + meniscus_mm))

        # Khôi phục lại trạng thái cuối T1
        self.load_checkpoint(ckpt_ram)

        return {
            "eta_wcsph_t1_mm": float(eta_wcsph_t1_mm),
            "eta_wcsph_t2_mm": float(eta_wcsph_t2_mm),
            "worst_phase_gain": float(worst_phase_gain),
            "save_ram_ms": float(save_ram_ms),
            "u_carry_amp": float(u_carry_amp),
            "cw_x_peak": float(cw_x_peak),
            "cw_y_peak": float(cw_y_peak),
            "tan_th_x_final": float(tan_th_x_final),
            "tan_th_y_final": float(tan_th_y_final),
            "res_t1": res_t1,
            "res_t2": res_t2,
            "t1_window_ms": float(n1_steps * dt * 1000.0),
            "t2_window_ms": float(n2_steps * dt * 1000.0),
        }
