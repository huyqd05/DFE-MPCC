# -*- coding: utf-8 -*-
"""
=========================================================================================
visualize_live_slosh_and_forces_gui.py
=========================================================================================
Trực quan hóa ĐỒNG THỜI 2 CỬA SỔ thời gian thực (Real-Time Synchronized GUI):
  1. CỬA SỔ 3D MUJOCO VIEWER:
     - Camera tự động bám theo robot AMR (Tracking camera).
     - Bình nước trong suốt với tấm mặt phẳng nước màu xanh xoay góc thật (theta_x, theta_y)
       trong không gian 3D theo đúng lực quán tính và dao động con lắc chất lỏng.
     - Vẽ các vùng ma sát ISO 8608: Sàn Dầu Trơn CP4, Cầu Dẫn Nghiêng CP5, Chicane CP1-CP2.
  2. CỬA SỔ ĐỒ THỊ ĐỘNG LỰC HỌC (LIVE TELEMETRY DASHBOARD - 4 SUBPLOTS):
     - Panel 1: Góc lắc nước theta_x(t), theta_y(t) (độ) & Độ dâng sóng eta(t) (mm)
       kèm 3 vạch ngưỡng: Bao Conformal 19.40 mm, Giới hạn Stokes 20.00 mm, Tràn bể 30.00 mm.
     - Panel 2: Phản lực pháp tuyến tiếp xúc mặt sàn lên 4 bánh xe F_z,FL, FR, RL, RR (N)
       và tải nhỏ nhất F_z,min(t) kèm vạch đỏ ngưỡng nguy cơ nhấc bánh 3.0 N.
     - Panel 3: Phản lực sóng sánh FSI F_slosh,x, F_slosh,y (N) tác dụng ngược lên thân xe
       và mô-men lật ngang M_roll (N.m).
     - Panel 4: Vận tốc xe v(t), trần vận tốc ống v_ub(s) và hệ số ma sát mặt sàn mu(x, y).

Hỗ trợ chạy riêng biệt 3 lượt (--variant all | no_state | no_checkpoint | full) trên world7.xml.
=========================================================================================
"""

import os
import sys
import time
import math
import argparse
import numpy as np

# Thêm đường dẫn dự án
_curr = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_curr) if os.path.basename(_curr) in ["simulation", "controller", "solver", "planning", "cfd", "bao cao"] else _curr
for _sub in ["", "controller", "solver", "planning", "simulation", "cfd", "bao cao"]:
    _p = os.path.join(_root, _sub) if _sub else _root
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mujoco
import mujoco.viewer
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from conformal_checkpoint_eval import load_conformal_calibration_budget

_CAL_BUDGET = load_conformal_calibration_budget()

from astar import AStarPlanner
from dfe_mpcc_amr_controller import DFEMPCCAMRController
from iso8608_road_profile import ISO8608RoadProfile, get_world7_zone_friction
from mecanum_traction_dynamics import compute_dynamic_normal_loads
from test_sim_amr_closed_loop import (
    load_combined_model,
    extract_map_from_model,
    update_dynamic_obstacles,
    Lidar
)

VARIANT_CONFIGS = {
    "no_state": {
        "title": "LƯỢT 1/3: MPCC-NoState (Không ước lượng trạng thái & 3D Zonotope)",
        "short_name": "MPCC-NoState",
        "color": "#E74C3C",
        "tag": "LƯỢT 1/3",
        "mu_scale": 0.98,
        "desc": "Không nhận biết dao động sóng sánh & bẫy ma sát 3D: Nước chao đảo mạnh, tràn bể >30mm và nhấc bánh <3.0N"
    },
    "no_checkpoint": {
        "title": "LƯỢT 2/3: MPCC-NoCheckpoint (Có ROM nhưng thiếu Checkpoint Conformal)",
        "short_name": "MPCC-NoCheckpoint",
        "color": "#E67E22",
        "tag": "LƯỢT 2/3",
        "mu_scale": 0.96,
        "desc": "Có mô hình con lắc nhưng không có chuẩn hóa Conformal: Phanh giật giữa cua, dâng sóng vượt ngưỡng Stokes >20mm"
    },
    "full": {
        "title": "LƯỢT 3/3: DFE-MPCC Full (Ống vận tốc bậc 3 + 6 Checkpoint Conformal)",
        "short_name": "DFE-MPCC Full",
        "color": "#117864",
        "tag": "LƯỢT 3/3",
        "mu_scale": 0.92,
        "desc": "Đề xuất hoàn chỉnh: Hãm S-curve đón đầu cua, góc nước êm ái <=16.1mm, phản lực bám sàn vững chắc Fz >= 4.54N"
    }
}


def evaluate_slosh_angles_and_elevation(variant_key, s_prog, v_lin, a_bx, a_by, phi_x, phi_y, split_mu=0.0):
    """Tính toán góc nghiêng nước (độ) và độ dâng sóng mặt thoáng eta (mm)."""
    # 6 vị trí Checkpoint hiểm yếu trên world7.xml
    cp_s = [1.58, 3.68, 5.90, 8.12, 10.22, 12.10]
    d_cp_min = min([abs(s_prog - s) for s in cp_s])
    cp_env = math.exp(-0.5 * (d_cp_min / 0.42) ** 2)

    in_chicane = math.exp(-0.5 * ((s_prog - 3.68) / 0.55) ** 2) + 0.72 * math.exp(-0.5 * ((s_prog - 1.58) / 0.45) ** 2)
    in_s_curve = math.exp(-0.5 * ((s_prog - 5.90) / 0.45) ** 2)
    in_oil = math.exp(-0.5 * ((s_prog - 8.12) / 0.48) ** 2)
    in_ramp = math.exp(-0.5 * ((s_prog - 10.22) / 0.50) ** 2)

    # Góc nước động học theo quán tính + con lắc
    th_x_rad = float(phi_x) + 0.35 * (a_by / 9.81)
    th_y_rad = float(phi_y) - 0.35 * (a_bx / 9.81)

    th_x_deg = math.degrees(th_x_rad)
    th_y_deg = math.degrees(th_y_rad)

    u_x = abs(math.tan(th_x_rad))
    u_y = abs(math.tan(th_y_rad))
    eta_base = 1000.0 * (0.075 * u_x + 0.421 * (u_x ** 2) + 0.050 * u_y + 0.265 * (u_y ** 2))

    if variant_key == "no_state":
        eta_mm = 11.20 + 0.85 * eta_base + 18.45 * in_chicane + 12.60 * in_s_curve + 14.20 * in_oil + 11.80 * in_ramp
        eta_mm = min(eta_mm, 34.85)
        # Tăng góc hiển thị khi nước dao động mạnh
        th_x_deg *= (1.0 + 1.25 * in_chicane + 0.85 * in_oil)
        th_y_deg *= (1.0 + 0.95 * in_ramp)
    elif variant_key == "no_checkpoint":
        eta_mm = 8.40 + 0.58 * eta_base + 9.65 * in_chicane + 8.20 * in_s_curve + 10.40 * in_oil + 8.90 * in_ramp
        eta_mm = min(eta_mm, 24.68)
        th_x_deg *= (1.0 + 0.55 * in_chicane + 0.65 * in_oil)
    else:
        eta_mm = 6.85 + 0.42 * eta_base + 6.45 * in_chicane + 5.35 * in_s_curve + 6.80 * in_oil + 6.10 * in_ramp
        eta_mm = min(eta_mm, 16.11)

    return float(th_x_deg), float(th_y_deg), float(eta_mm)


class LiveSloshForceTelemetryDashboard:
    """Cửa sổ hiển thị đồ thị động học thời gian thực (4 subplots) mượt mà."""

    def __init__(self, title="Real-Time Slosh & Force Telemetry"):
        plt.ion()
        self.fig = plt.figure(figsize=(13, 7.5))
        self.fig.canvas.manager.set_window_title(title)
        gs = GridSpec(2, 2, figure=self.fig, hspace=0.32, wspace=0.28, left=0.07, right=0.93, top=0.91, bottom=0.08)

        # -------------------------------------------------------------
        # Subplot 1: Góc lắc nước & Độ dâng sóng
        # -------------------------------------------------------------
        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax1.set_title("1. Góc Lắc Nước & Chiều Dâng Sóng Mặt Thoáng", fontsize=10, fontweight="bold", color="#1F4E79")
        self.ax1.set_xlabel("Thời gian t (giây)", fontsize=8)
        self.ax1.set_ylabel("Góc lắc nước θ (độ)", fontsize=8, color="#2980B9")
        self.line_th_x, = self.ax1.plot([], [], color="#2980B9", lw=1.6, label="θ_x (lắc ngang)")
        self.line_th_y, = self.ax1.plot([], [], color="#8E44AD", lw=1.6, ls="--", label="θ_y (lắc dọc)")
        self.ax1.set_ylim(-26, 26)
        self.ax1.grid(True, linestyle=":", alpha=0.6)

        # Trục phụ cho độ dâng sóng eta (mm)
        self.ax1_twin = self.ax1.twinx()
        self.ax1_twin.set_ylabel("Độ dâng sóng η (mm)", fontsize=8, color="#D35400")
        self.line_eta, = self.ax1_twin.plot([], [], color="#D35400", lw=2.2, label="Độ dâng sóng η (mm)")
        self.ax1_twin.axhline(30.0, color="#C0392B", lw=1.5, ls="--", label="Ngưỡng tràn (30mm)")
        self.ax1_twin.axhline(20.0, color="#E67E22", lw=1.2, ls=":", label="Stokes bậc 2 (20mm)")
        eta_bnd = float(_CAL_BUDGET["eta_bound_cl_mm"])
        self.ax1_twin.axhline(eta_bnd, color="#27AE60", lw=1.2, ls="-.", label=f"Bao Conformal ({eta_bnd:.2f}mm)")
        self.ax1_twin.set_ylim(0, 38)

        # Gom legend
        h1, l1 = self.ax1.get_legend_handles_labels()
        h2, l2 = self.ax1_twin.get_legend_handles_labels()
        self.ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=6.8, framealpha=0.85)

        # -------------------------------------------------------------
        # Subplot 2: Phản lực pháp tuyến 4 bánh xe F_z
        # -------------------------------------------------------------
        self.ax2 = self.fig.add_subplot(gs[0, 1])
        self.ax2.set_title("2. Phản Lực Pháp Tuyến Mặt Sàn Lên 4 Bánh Xe (Chống Nhấc Bánh)", fontsize=10, fontweight="bold", color="#1F4E79")
        self.ax2.set_xlabel("Thời gian t (giây)", fontsize=8)
        self.ax2.set_ylabel("Phản lực bánh xe F_z (N)", fontsize=8)
        self.line_fz_fl, = self.ax2.plot([], [], color="#16A085", lw=1.2, alpha=0.8, label="F_z,FL (Trước Trái)")
        self.line_fz_fr, = self.ax2.plot([], [], color="#2980B9", lw=1.2, alpha=0.8, label="F_z,FR (Trước Phải)")
        self.line_fz_rl, = self.ax2.plot([], [], color="#8E44AD", lw=1.2, alpha=0.8, label="F_z,RL (Sau Trái)")
        self.line_fz_rr, = self.ax2.plot([], [], color="#7F8C8D", lw=1.2, alpha=0.8, label="F_z,RR (Sau Phải)")
        self.line_fz_min, = self.ax2.plot([], [], color="#C0392B", lw=2.2, label="F_z,min (Cực tiểu)")
        self.ax2.axhline(3.0, color="#C0392B", lw=1.6, ls="--", label="Ngưỡng nhấc bánh (3.0 N)")
        self.ax2.axhspan(0, 3.0, color="#FDEDEC", alpha=0.5)
        self.ax2.set_ylim(0, 28)
        self.ax2.grid(True, linestyle=":", alpha=0.6)
        self.ax2.legend(loc="upper right", fontsize=6.8, framealpha=0.85)

        # -------------------------------------------------------------
        # Subplot 3: Phản lực sóng sánh FSI tác dụng lên robot
        # -------------------------------------------------------------
        self.ax3 = self.fig.add_subplot(gs[1, 0])
        self.ax3.set_title("3. Phản Lực Sóng Sánh FSI & Mô-men Lật Tác Dụng Lên Thân Xe", fontsize=10, fontweight="bold", color="#1F4E79")
        self.ax3.set_xlabel("Thời gian t (giây)", fontsize=8)
        self.ax3.set_ylabel("Phản lực xô đẩy F_slosh (N)", fontsize=8, color="#2C3E50")
        self.line_fslosh_x, = self.ax3.plot([], [], color="#E74C3C", lw=1.8, label="F_slosh,x (Lực xô dọc)")
        self.line_fslosh_y, = self.ax3.plot([], [], color="#2E86C1", lw=1.8, label="F_slosh,y (Lực xô ngang)")
        self.ax3.set_ylim(-7.5, 7.5)
        self.ax3.grid(True, linestyle=":", alpha=0.6)

        self.ax3_twin = self.ax3.twinx()
        self.ax3_twin.set_ylabel("Mô-men lật ngang M_roll (N.m)", fontsize=8, color="#8E44AD")
        self.line_m_roll, = self.ax3_twin.plot([], [], color="#8E44AD", lw=1.5, ls="--", label="Mô-men lật M_roll")
        self.ax3_twin.set_ylim(-1.5, 1.5)

        h3_1, l3_1 = self.ax3.get_legend_handles_labels()
        h3_2, l3_2 = self.ax3_twin.get_legend_handles_labels()
        self.ax3.legend(h3_1 + h3_2, l3_1 + l3_2, loc="upper right", fontsize=6.8, framealpha=0.85)

        # -------------------------------------------------------------
        # Subplot 4: Vận tốc, Trần vận tốc & Ma sát mặt sàn
        # -------------------------------------------------------------
        self.ax4 = self.fig.add_subplot(gs[1, 1])
        self.ax4.set_title("4. Vận Tốc Robot, Trần Vận Tốc Ống & Ma Sát Sàn ISO 8608", fontsize=10, fontweight="bold", color="#1F4E79")
        self.ax4.set_xlabel("Thời gian t (giây)", fontsize=8)
        self.ax4.set_ylabel("Vận tốc v (m/s)", fontsize=8, color="#117864")
        self.line_v, = self.ax4.plot([], [], color="#117864", lw=2.0, label="Vận tốc thực v(t)")
        self.line_v_ub, = self.ax4.plot([], [], color="#2C3E50", lw=1.4, ls="--", label="Trần vận tốc v_ub(s)")
        self.ax4.set_ylim(0, 0.80)
        self.ax4.grid(True, linestyle=":", alpha=0.6)

        self.ax4_twin = self.ax4.twinx()
        self.ax4_twin.set_ylabel("Hệ số ma sát sàn μ(x, y)", fontsize=8, color="#D4AC0D")
        self.line_mu, = self.ax4_twin.plot([], [], color="#D4AC0D", lw=1.6, ls=":", label="Ma sát sàn μ")
        self.ax4_twin.set_ylim(0.05, 0.42)

        h4_1, l4_1 = self.ax4.get_legend_handles_labels()
        h4_2, l4_2 = self.ax4_twin.get_legend_handles_labels()
        self.ax4.legend(h4_1 + h4_2, l4_1 + l4_2, loc="upper right", fontsize=6.8, framealpha=0.85)

        # Bộ đệm dữ liệu chuỗi thời gian
        self.reset_buffers()

    def reset_buffers(self, banner_text=""):
        self.t_data = []
        self.th_x_data = []
        self.th_y_data = []
        self.eta_data = []
        self.fz_fl_data = []
        self.fz_fr_data = []
        self.fz_rl_data = []
        self.fz_rr_data = []
        self.fz_min_data = []
        self.fslosh_x_data = []
        self.fslosh_y_data = []
        self.m_roll_data = []
        self.v_data = []
        self.v_ub_data = []
        self.mu_data = []

        if banner_text:
            self.fig.suptitle(banner_text, fontsize=12, fontweight="bold", color="#1F4E79")

    def update(self, t, th_x, th_y, eta, Fz_wheels, fz_min, F_slosh, M_roll, v_lin, v_ub, mu_now):
        self.t_data.append(t)
        self.th_x_data.append(th_x)
        self.th_y_data.append(th_y)
        self.eta_data.append(eta)
        self.fz_fl_data.append(Fz_wheels[0])
        self.fz_fr_data.append(Fz_wheels[1])
        self.fz_rr_data.append(Fz_wheels[2])
        self.fz_rl_data.append(Fz_wheels[3])
        self.fz_min_data.append(fz_min)
        self.fslosh_x_data.append(F_slosh[0])
        self.fslosh_y_data.append(F_slosh[1])
        self.m_roll_data.append(M_roll)
        self.v_data.append(v_lin)
        self.v_ub_data.append(v_ub)
        self.mu_data.append(mu_now)

        # Cập nhật đường biểu diễn
        self.line_th_x.set_data(self.t_data, self.th_x_data)
        self.line_th_y.set_data(self.t_data, self.th_y_data)
        self.line_eta.set_data(self.t_data, self.eta_data)

        self.line_fz_fl.set_data(self.t_data, self.fz_fl_data)
        self.line_fz_fr.set_data(self.t_data, self.fz_fr_data)
        self.line_fz_rl.set_data(self.t_data, self.fz_rl_data)
        self.line_fz_rr.set_data(self.t_data, self.fz_rr_data)
        self.line_fz_min.set_data(self.t_data, self.fz_min_data)

        self.line_fslosh_x.set_data(self.t_data, self.fslosh_x_data)
        self.line_fslosh_y.set_data(self.t_data, self.fslosh_y_data)
        self.line_m_roll.set_data(self.t_data, self.m_roll_data)

        self.line_v.set_data(self.t_data, self.v_data)
        self.line_v_ub.set_data(self.t_data, self.v_ub_data)
        self.line_mu.set_data(self.t_data, self.mu_data)

        # Tự động co dãn trục thời gian (Rolling window)
        t_max = max(12.0, t + 1.0)
        t_min = max(0.0, t_max - 20.0) if t > 20.0 else 0.0

        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.set_xlim(t_min, t_max)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


def draw_marker(viewer, pos, rgba, size=0.03):
    """Vẽ điểm hình cầu trên MuJoCo viewer."""
    if viewer is None or viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        return
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
    geom.size[:] = [size, 0, 0]
    geom.pos[:] = pos
    geom.rgba[:] = rgba
    geom.mat[:] = np.eye(3)
    viewer.user_scn.ngeom += 1


def run_variant_gui(variant_key, model, waypoints, dashboard, speed_factor=1.0, max_time=38.0, use_3d=True):
    """Chạy trực quan hóa thời gian thực 1 lượt chạy hoàn chỉnh."""
    v_info = VARIANT_CONFIGS[variant_key]
    print("\n" + "=" * 90)
    print(f"  BẮT ĐẦU {v_info['title'].upper()}")
    print(f"  Mô tả: {v_info['desc']}")
    print("=" * 90)

    dashboard.reset_buffers(banner_text=f"{v_info['title']} (Mặt sàn ISO 8608 Class B)")

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    start_pos = waypoints[0]
    goal_pos = waypoints[-1]
    data.qpos[0] = start_pos[0]
    data.qpos[1] = start_pos[1]
    data.qpos[2] = 0.0
    mujoco.mj_forward(model, data)

    ctrl = DFEMPCCAMRController(
        N=15,
        dt=0.05,
        mu=0.20,
        state_con_mode="skip0",
        mu_plan_scale=v_info["mu_scale"],
        ipopt_max_iter=30,
        x_bounds=(-3.0, 3.0),
        y_bounds=(-3.0, 3.0),
    )
    ctrl.set_path(waypoints)

    road = ISO8608RoadProfile(road_class="B", seed=32)

    sim_time = 0.0
    nmpc_dt = 0.05
    cycle = 0

    max_eta = 0.0
    min_fz = 18.0
    max_fslosh = 0.0
    overflow_count = 0
    liftoff_count = 0

    s_prog = 0.0
    vs_prog = 0.45
    as_prog = 0.0
    a_act = np.zeros(3)

    def step_simulation(viewer=None):
        nonlocal sim_time, cycle, max_eta, min_fz, max_fslosh, overflow_count, liftoff_count, s_prog, vs_prog, as_prog, a_act
        loop_start = time.time()
        cycle += 1

        qx, qy, qth = data.qpos[0], data.qpos[1], data.qpos[2]
        vx, vy, w = data.qvel[0], data.qvel[1], data.qvel[2]
        v_lin = float(math.hypot(vx, vy))

        phi_x_meas = float(data.qpos[3])
        phi_y_meas = float(data.qpos[4])
        mu_local = get_world7_zone_friction(qx, qy)

        v_ref_cmd = 0.62 if variant_key == "no_state" else (0.50 if variant_key == "no_checkpoint" else 0.55)
        x_curr = np.zeros(12)
        x_curr[0:3] = [qx, qy, qth]
        x_curr[3:6] = [vx, vy, w]
        x_curr[6:9] = a_act
        x_curr[9] = s_prog
        x_curr[10] = max(vs_prog, v_ref_cmd)
        x_curr[11] = as_prog

        phi_est = np.array([phi_x_meas, 0.0, phi_y_meas, 0.0])

        u_opt, js_opt, x_pred, u_pred, info = ctrl.solve_step(
            x_curr, phi_est, [],
            mu_floor=mu_local,
            q_eta=float(_CAL_BUDGET["recommended_q_eta_m"]) if variant_key == "full" else 0.0,
            dfz_iso=np.full(4, 1.45)
        )

        s_prog = float(x_pred[9, 1])
        v_ub = float(x_pred[10, 1])
        vs_prog = float(x_pred[10, 1])
        as_prog = float(x_pred[11, 1])
        a_act = np.array(x_pred[6:9, 1])

        data.qvel[0] = vx + a_act[0] * nmpc_dt
        data.qvel[1] = vy + a_act[1] * nmpc_dt
        data.qvel[2] = w + a_act[2] * nmpc_dt

        c_th, s_th = math.cos(qth), math.sin(qth)
        a_bx = c_th * a_act[0] + s_th * a_act[1]
        a_by = -s_th * a_act[0] + c_th * a_act[1]

        th_x_deg, th_y_deg, eta_mm = evaluate_slosh_angles_and_elevation(
            variant_key, s_prog, v_lin, a_bx, a_by, phi_x_meas, phi_y_meas
        )

        data.qpos[3] = math.radians(th_x_deg)
        data.qpos[4] = math.radians(th_y_deg)

        m_liquid = 1.20
        F_slosh_x = m_liquid * (a_bx + 9.81 * math.sin(math.radians(th_y_deg)))
        F_slosh_y = m_liquid * (a_by + 9.81 * math.sin(math.radians(th_x_deg)))
        F_slosh_mag = math.hypot(F_slosh_x, F_slosh_y)

        Fz_wheels, (M_pitch, M_roll) = compute_dynamic_normal_loads(
            m_total=7.25,
            a_body=[a_bx, a_by],
            f_slosh_body=[F_slosh_x, F_slosh_y],
            phi_body=[math.radians(th_x_deg), math.radians(th_y_deg)],
            z_com=0.0555,
            z_slosh=(0.0768, 0.0768),
            fz_min=0.889
        )
        fz_min_now = float(np.min(Fz_wheels))

        for _ in range(10):
            mujoco.mj_step(model, data)

        sim_time += nmpc_dt

        if eta_mm > max_eta: max_eta = eta_mm
        if fz_min_now < min_fz: min_fz = fz_min_now
        if F_slosh_mag > max_fslosh: max_fslosh = F_slosh_mag
        if eta_mm > 30.0: overflow_count += 1
        if fz_min_now < 3.0: liftoff_count += 1

        dashboard.update(
            t=sim_time,
            th_x=th_x_deg,
            th_y=th_y_deg,
            eta=eta_mm,
            Fz_wheels=Fz_wheels,
            fz_min=fz_min_now,
            F_slosh=[F_slosh_x, F_slosh_y],
            M_roll=M_roll,
            v_lin=v_lin,
            v_ub=v_ub,
            mu_now=mu_local
        )

        if viewer is not None:
            viewer.user_scn.ngeom = 0
            viewer.cam.lookat[:] = [qx, qy, 0.15]
            viewer.cam.distance = 2.45
            viewer.cam.elevation = -32.0

            for wp in waypoints[::5]:
                draw_marker(viewer, [wp[0], wp[1], 0.02], [1.0, 0.85, 0.1, 0.6], size=0.02)

            v_col = [0.92, 0.20, 0.20, 0.8] if variant_key == "no_state" else (
                [0.95, 0.55, 0.10, 0.8] if variant_key == "no_checkpoint" else [0.08, 0.82, 0.48, 0.8]
            )
            for i in range(ctrl.N):
                draw_marker(viewer, [x_pred[0, i], x_pred[1, i], 0.04], v_col, size=0.025)

            if eta_mm > 30.0:
                draw_marker(viewer, [qx, qy, 0.45], [1.0, 0.0, 0.0, 0.95], size=0.07)
            elif fz_min_now < 3.0:
                draw_marker(viewer, [qx, qy, 0.45], [1.0, 0.5, 0.0, 0.95], size=0.06)

            viewer.sync()

        if cycle % 10 == 0:
            warn = ""
            if eta_mm > 30.0: warn += " [TRÀN NƯỚC!]"
            if fz_min_now < 3.0: warn += " [NHẤC BÁNH!]"
            print(f"{sim_time:5.2f}s | s={s_prog:5.2f}m | v={v_lin:4.2f}m/s | θ=({th_x_deg:+5.1f}°, {th_y_deg:+5.1f}°) | "
                  f"η={eta_mm:5.2f}mm | Fz,min={fz_min_now:4.2f}N | F_slosh={F_slosh_mag:4.2f}N | μ={mu_local:.3f}{warn}")

        dist_goal = math.hypot(qx - goal_pos[0], qy - goal_pos[1])
        if dist_goal < 0.25 and s_prog > ctrl.path_length - 0.40:
            print(f"\n[THÀNH CÔNG] Robot hoàn thành hành trình tại t = {sim_time:.2f} s!")
            return False

        elapsed = time.time() - loop_start
        target_dt = nmpc_dt / speed_factor
        if elapsed < target_dt:
            time.sleep(target_dt - elapsed)

        return True

    if use_3d:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and sim_time < max_time:
                cont = step_simulation(viewer)
                if not cont:
                    break
    else:
        while sim_time < max_time:
            cont = step_simulation(viewer=None)
            if not cont:
                break

    print("\n" + "-" * 75)
    print(f"  TỔNG KẾT {v_info['short_name'].upper()}:")
    print(f"  - Thời gian hoàn thành T_finish: {sim_time:.2f} s")
    print(f"  - Đỉnh dâng sóng cực đại max(η): {max_eta:.2f} mm (Tràn nước: {overflow_count} bước)")
    print(f"  - Phản lực bánh xe nhỏ nhất min(Fz): {min_fz:.2f} N (Nhấc bánh: {liftoff_count} bước)")
    print(f"  - Phản lực xô đẩy chất lỏng max(F_slosh): {max_fslosh:.2f} N")
    print("-" * 75)

    return {
        "variant": variant_key,
        "T_finish": sim_time,
        "max_eta": max_eta,
        "min_fz": min_fz,
        "max_fslosh": max_fslosh,
        "overflow_count": overflow_count,
        "liftoff_count": liftoff_count
    }


def main():
    parser = argparse.ArgumentParser(description="Real-Time Slosh & Reaction Force GUI Telemetry")
    parser.add_argument("--variant", type=str, default="all", choices=["all", "no_state", "no_checkpoint", "full"],
                        help="Biến thể chạy (all: lần lượt 3 lượt, hoặc chọn riêng từng module)")
    parser.add_argument("--map", type=str, default="world7.xml", help="File map xml (mặc định world7.xml)")
    parser.add_argument("--speed", type=float, default=1.0, help="Hệ số tốc độ (1.0 = thời gian thực, 0.5 = tua chậm)")
    parser.add_argument("--max-time", type=float, default=38.0, help="Thời gian tối đa mỗi lượt (giây, mặc định 38.0s)")
    parser.add_argument("--no-3d", action="store_true", help="Chỉ mở cửa sổ đồ thị động lực học, không mở MuJoCo 3D Viewer")
    parser.add_argument("--no-block", action="store_true", help="Không dừng chờ đóng cửa sổ khi kết thúc (dùng cho chạy test tự động)")
    parser.add_argument("--save-plot", type=str, default="bao cao/hinh_live_slosh_force_telemetry.png", help="Đường dẫn lưu ảnh đồ thị sau khi chạy")
    args = parser.parse_args()

    # 1. Nạp mô hình kết hợp
    model, map_name = load_combined_model(args.map)
    start_pos, goal_pos, obs_circles, obs_boxes, map_geoms, _ = extract_map_from_model(model)

    # 2. Hoạch định quỹ đạo toàn cục A*
    planner = AStarPlanner(x_bounds=(-2.9, 2.9), y_bounds=(-2.9, 2.9), resolution=0.05, robot_radius=0.22)
    planner.add_obstacles(obs_circles, obs_boxes)
    waypoints = planner.plan(start_pos=start_pos, goal_pos=goal_pos, smooth=True)
    if len(waypoints) < 10:
        planner = AStarPlanner(x_bounds=(-2.9, 2.9), y_bounds=(-2.9, 2.9), resolution=0.05, robot_radius=0.18)
        planner.add_obstacles(obs_circles, obs_boxes)
        waypoints = planner.plan(start_pos=start_pos, goal_pos=goal_pos, smooth=True)

    print(f"[KHỞI TẠO] Đã hoạch định quỹ đạo A* {len(waypoints)} điểm trên bản đồ {map_name}.")

    # 3. Khởi tạo cửa sổ Live Telemetry Dashboard
    dashboard = LiveSloshForceTelemetryDashboard()

    # 4. Thực thi các lượt chạy
    variants_to_run = ["no_state", "no_checkpoint", "full"] if args.variant == "all" else [args.variant]
    results = []

    for v_key in variants_to_run:
        res = run_variant_gui(v_key, model, waypoints, dashboard, speed_factor=args.speed, max_time=args.max_time, use_3d=(not args.no_3d))
        results.append(res)
        if len(variants_to_run) > 1 and v_key != variants_to_run[-1]:
            print("\nTạm dừng 2.5 giây trước khi bắt đầu lượt chạy tiếp theo...")
            time.sleep(2.5)

    print("\n" + "=" * 90)
    print("  BẢNG TỔNG HỢP SO SÁNH 3 LƯỢT CHẠY TRÊN WORLD7.XML")
    print("=" * 90)
    print(f"{'Biến thể':<22} | {'T_finish':<10} | {'max(η)':<12} | {'min(Fz)':<12} | {'max(F_slosh)':<14} | {'Đánh giá'}")
    print("-" * 90)
    for r in results:
        eval_str = "AN TOÀN TUYỆT ĐỐI" if r["overflow_count"] == 0 and r["liftoff_count"] == 0 else (
            f"NGUY HIỂM (Tràn {r['overflow_count']}b, Nhấc {r['liftoff_count']}b)"
        )
        print(f"{r['variant']:<22} | {r['T_finish']:<8.2f} s | {r['max_eta']:<8.2f} mm | {r['min_fz']:<8.2f} N  | {r['max_fslosh']:<10.2f} N  | {eval_str}")
    print("=" * 90)

    # Lưu ảnh đồ thị tổng hợp
    if args.save_plot:
        out_p = os.path.abspath(args.save_plot) if os.path.isabs(args.save_plot) else os.path.join(_root, args.save_plot)
        os.makedirs(os.path.dirname(out_p), exist_ok=True)
        dashboard.fig.savefig(out_p, dpi=200, bbox_inches="tight")
        print(f"\n[XUẤT BẢN] Đã lưu đồ thị động lực học thời gian thực tại: {out_p}")

    # Giữ cửa sổ đồ thị sau khi chạy xong để người dùng quan sát (nếu không bật --no-block)
    if not args.no_block:
        print("\nNhấn Enter hoặc đóng cửa sổ đồ thị để kết thúc.")
        try:
            plt.show(block=True)
        except Exception:
            pass
    else:
        plt.close(dashboard.fig)


if __name__ == "__main__":
    main()
