# -*- coding: utf-8 -*-
"""
iso8608_road_profile.py
=============================================================================
MÔ HÌNH ĐỊA HÌNH MẶT ĐƯỜNG NGẪU NHIÊN THEO TIÊU CHUẨN QUỐC TẾ ISO 8608:2016
(Thay thế cho FrictionObserverFFRLS trong kiến trúc DFE-MPCC + Checkpoint SPH)

Cơ sở lý thuyết & Vật lý (ISO 8608:2016):
1. Mật độ phổ công suất không gian (Spatial Power Spectral Density - PSD):
       G_d(n) = G_d(n_0) * (n / n_0)^(-w)
   trong đó:
   - n: Tần số không gian [cycles/m] (n_min = 0.05 -> n_max = 10.0 cycles/m)
   - n_0 = 0.1 cycles/m: Tần số không gian tham chiếu chuẩn ISO
   - w = 2.0: Số mũ waviness chuẩn
   - G_d(n_0): Độ nhám chuẩn theo cấp đường (Road Class A, B, C, Industrial)

2. Tác động lên xe AMR 4 bánh Mecanum chở chất lỏng:
   - Tính độ cao mặt sàn z_road(x_w, y_w) và vận tốc thẳng đứng dz_road/dt
     tại đúng tọa độ tiếp đất của 4 bánh xe (FL, FR, RR, RL).
   - Do khung xe 4 bánh là hệ siêu tĩnh (hyperstatic), độ vênh mặt phẳng 4 bánh
     (warp mode) làm biến thiên tải pháp tuyến tức thời:
         Delta_Fz_w^(ISO) = k_contact * Delta_z_warp,w + c_contact * d(Delta_z_warp,w)/dt
   - Độ nghiêng mặt đường cục bộ (theta_pitch_road, theta_roll_road) và gia tốc
     thẳng đứng z_ddot_road truyền trực tiếp vào kích thích sóng sánh chất lỏng.
=============================================================================
"""

import numpy as np


# Bảng thông số mật độ phổ công suất G_d(n_0) [m^3] theo chuẩn ISO 8608:2016
ISO8608_CLASSES = {
    "A_industrial": 2.0e-6,   # Sàn bê tông nhà máy công nghiệp (biên độ mấp mô ~ 1.0 - 2.5 mm)
    "A":            16.0e-6,  # Đường rất tốt (ISO Class A chuẩn)
    "B":            64.0e-6,  # Đường tốt / Sàn xưởng cũ gồ ghề (ISO Class B)
    "C":            256.0e-6  # Đường trung bình / Xóc mạnh (ISO Class C)
}



def get_world7_zone_friction(x: float, y: float) -> float:
    """
    Hệ số ma sát danh định trên bản đồ world7.xml (6x6 m).
    Tạm thời cố định đồng nhất mu = 0.35 trên toàn sàn để tách bạch kiểm chứng
    ảnh hưởng của Checkpoint (chưa kích hoạt bẫy ma sát / ước lượng ma sát trực tuyến).
    """
    return 0.35


def get_world8_zone_friction(x: float, y: float) -> float:
    """
    Hệ số ma sát danh định trên đấu trường world8.xml (6.5 x 6.5 m).
    Tạm thời cố định đồng nhất mu = 0.35 trên toàn sàn để tách bạch kiểm chứng
    ảnh hưởng của Checkpoint (chưa kích hoạt bẫy ma sát / ước lượng ma sát trực tuyến).
    """
    return 0.35


class ISO8608RoadProfile:
    """
    Bộ sinh biên dạng mặt đường ngẫu nhiên 2D chuẩn ISO 8608 và tính toán
    biến thiên tải pháp tuyến 4 bánh Mecanum + kích thích nghiêng bồn nước.
    """
    def __init__(
        self,
        road_class: str = "A_industrial",
        seed: int = 42,
        n_harmonics: int = 48,
        n_min: float = 0.08,
        n_max: float = 6.0,
        k_contact: float = 4500.0,
        c_contact: float = 120.0,
        max_dfz_ratio: float = 0.42
    ):
        """
        Parameters:
        -----------
        road_class : str
            Cấp mặt đường ISO 8608 ('A_industrial', 'A', 'B', 'C').
        seed : int
            Hạt giống ngẫu nhiên cố định để tái lập kết quả 100%.
        n_harmonics : int
            Số lượng sóng điều hòa không gian tổng hợp.
        k_contact : float
            Độ cứng tiếp xúc hiệu dụng của con lăn Mecanum [N/m].
        c_contact : float
            Hệ số cản nhớt tiếp xúc con lăn Mecanum [N.s/m].
        max_dfz_ratio : float
            Giới hạn tỷ lệ biến thiên tải pháp tuyến tối đa so với tải tĩnh (tránh nhiễu cực đoan).
        """
        if road_class not in ISO8608_CLASSES:
            raise ValueError(f"road_class '{road_class}' không hợp lệ. Chọn trong {list(ISO8608_CLASSES.keys())}")

        self.road_class = road_class
        self.Gd_n0 = ISO8608_CLASSES[road_class]
        self.n_0 = 0.1
        self.w = 2.0
        self.seed = seed
        self.n_harmonics = n_harmonics
        self.k_contact = float(k_contact)
        self.c_contact = float(c_contact)
        self.max_dfz_ratio = float(max_dfz_ratio)

        rng = np.random.default_rng(seed)

        # Chia dải tần số không gian n [cycles/m]
        self.spatial_freqs = np.linspace(n_min, n_max, n_harmonics)
        dn = (n_max - n_min) / float(n_harmonics)

        # Phổ PSD G_d(n_k) và biên độ sóng điều hòa A_k = sqrt(2 * G_d(n_k) * dn)
        Gd_k = self.Gd_n0 * ((self.spatial_freqs / self.n_0) ** (-self.w))
        # Quy đổi biên độ cho trường 2D (tổng năng lượng bảo toàn)
        self.amplitudes = np.sqrt(2.0 * Gd_k * dn)

        # Góc hướng sóng ngẫu nhiên trong mặt phẳng (x, y) để tạo bất đối xứng trái-phải (roll & warp)
        self.wave_angles = rng.uniform(0.0, 2.0 * np.pi, size=n_harmonics)
        self.kx = 2.0 * np.pi * self.spatial_freqs * np.cos(self.wave_angles)
        self.ky = 2.0 * np.pi * self.spatial_freqs * np.sin(self.wave_angles)
        self.phases = rng.uniform(0.0, 2.0 * np.pi, size=n_harmonics)

    def compute_local_friction(
        self,
        x: float,
        y: float,
        mu_nominal = 0.20,
        variation_scale: float = 0.15
    ) -> float:
        """
        Tính hệ số ma sát cục bộ ngẫu nhiên mu(x, y) dựa trên độ nhám không gian 2D ISO 8608.
        Mô hình hóa sự kết hợp giữa:
          1. Phân vùng ma sát vĩ mô mu_base(x, y) (Macro-texture: bê tông, epoxy, dầu, dốc).
          2. Biến thiên vi mô ngẫu nhiên 2D chuẩn ISO 8608:2016 Class B:
             mu(x, y) = mu_base(x, y) * [1 + delta_mu^ISO(x, y)]
             với delta_mu^ISO trong khoảng [-variation_scale, +variation_scale].
        """
        if callable(mu_nominal):
            base_mu = float(mu_nominal(x, y))
        else:
            base_mu = float(mu_nominal)

        arg = self.kx * x + self.ky * y + self.phases
        norm_factor = np.sqrt(np.sum(self.amplitudes**2))
        spatial_noise = float(np.sum(self.amplitudes * np.sin(arg))) / (norm_factor * np.sqrt(2.0) + 1e-6)
        spatial_noise = np.clip(spatial_noise, -1.0, 1.0)
        mu_local = base_mu * (1.0 + variation_scale * spatial_noise)
        return float(np.clip(mu_local, 0.05, 0.80))

    def evaluate_point(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0) -> dict:
        """
        Tính độ cao mặt đường z_r(x, y), độ dốc (dz/dx, dz/dy) và vận tốc thẳng đứng dz_r/dt
        tại một điểm (x, y) di chuyển với vận tốc (vx, vy).
        """
        arg = self.kx * x + self.ky * y + self.phases
        cos_arg = np.cos(arg)
        sin_arg = np.sin(arg)

        z = float(np.sum(self.amplitudes * cos_arg))
        dz_dx = float(-np.sum(self.amplitudes * self.kx * sin_arg))
        dz_dy = float(-np.sum(self.amplitudes * self.ky * sin_arg))
        dz_dt = dz_dx * vx + dz_dy * vy

        return {
            "z_m": z,
            "dz_dx": dz_dx,
            "dz_dy": dz_dy,
            "dz_dt": dz_dt
        }

    def compute_wheel_loads_and_slosh_excitation(
        self,
        q_pose: np.ndarray,
        v_vel: np.ndarray,
        L_x: float = 0.12,
        L_y: float = 0.14,
        fz_static_per_wheel: float = 17.78,
        mu_nominal: float = 0.20,
        friction_variation_scale: float = 0.15,
    ) -> dict:
        """
        Tính toán tác động của mặt đường ISO 8608 tại vị trí hiện tại của xe AMR:
        1. Độ cao mặt sàn dưới 4 bánh xe: [FL, FR, RR, RL].
        2. Biến thiên tải pháp tuyến 4 bánh Delta_Fz_iso [N] do độ vênh mặt phẳng 4 bánh (Warp)
           và dốc cục bộ.
        3. Gia tốc kích thích ngang hiệu dụng lên bồn nước do độ nghiêng mặt đường (g * sin(theta)).

        Thứ tự 4 bánh khớp với DFEMPCCAMRController:
        - Bánh 1 (FL): (+L_x, +L_y)
        - Bánh 2 (FR): (+L_x, -L_y)
        - Bánh 3 (RR): (-L_x, -L_y)
        - Bánh 4 (RL): (-L_x, +L_y)
        """
        x_c, y_c, yaw = float(q_pose[0]), float(q_pose[1]), float(q_pose[2])
        vx_w, vy_w, omega_z = float(v_vel[0]), float(v_vel[1]), float(v_vel[2])

        c_y, s_y = np.cos(yaw), np.sin(yaw)

        # Tọa độ 4 bánh trong Body Frame: FL, FR, RR, RL
        body_offsets = np.array([
            [+L_x, +L_y],  # 1: FL
            [+L_x, -L_y],  # 2: FR
            [-L_x, -L_y],  # 3: RR
            [-L_x, +L_y]   # 4: RL
        ])

        z_wheels = np.zeros(4)
        dz_dt_wheels = np.zeros(4)

        for i in range(4):
            bx, by = body_offsets[i]
            # Tọa độ bánh thứ i trong World Frame
            wx = x_c + c_y * bx - s_y * by
            wy = y_c + s_y * bx + c_y * by
            # Vận tốc điểm tiếp đất bánh thứ i trong World Frame
            v_wx = vx_w - omega_z * (s_y * bx + c_y * by)
            v_wy = vy_w + omega_z * (c_y * bx - s_y * by)

            pt_eval = self.evaluate_point(wx, wy, v_wx, v_wy)
            z_wheels[i] = pt_eval["z_m"]
            dz_dt_wheels[i] = pt_eval["dz_dt"]

        # 1. Góc nghiêng mặt đường hiệu dụng dưới khung xe (Pitch & Roll của mặt sàn)
        # Pitch (dốc dọc theo trục x_body): đầu xe cao hơn đuôi xe -> theta_pitch_road < 0
        z_front = 0.5 * (z_wheels[0] + z_wheels[1])
        z_rear  = 0.5 * (z_wheels[2] + z_wheels[3])
        z_left  = 0.5 * (z_wheels[0] + z_wheels[3])
        z_right = 0.5 * (z_wheels[1] + z_wheels[2])

        theta_pitch_road = (z_rear - z_front) / (2.0 * L_x)
        theta_roll_road  = (z_left - z_right) / (2.0 * L_y)

        # 2. Độ vênh siêu tĩnh (Hyperstatic Warp Mode):
        # Mặt phẳng trung bình đi qua 4 điểm có độ cao dự kiến tại từng bánh là z_plane_i.
        # Sai lệch siêu tĩnh: z_warp = (z_FL - z_FR + z_RR - z_RL) / 4
        z_warp = 0.25 * (z_wheels[0] - z_wheels[1] + z_wheels[2] - z_wheels[3])
        dz_warp_dt = 0.25 * (dz_dt_wheels[0] - dz_dt_wheels[1] + dz_dt_wheels[2] - dz_dt_wheels[3])

        warp_signs = np.array([+1.0, -1.0, +1.0, -1.0])
        dfz_warp = (self.k_contact * z_warp + self.c_contact * dz_warp_dt) * warp_signs

        # Cộng thêm chuyển tải tĩnh do độ dốc mặt đường cục bộ
        dfz_slope_pitch = fz_static_per_wheel * 0.8 * np.sin(theta_pitch_road) * np.array([-1.0, -1.0, +1.0, +1.0])
        dfz_slope_roll  = fz_static_per_wheel * 0.8 * np.sin(theta_roll_road)  * np.array([-1.0, +1.0, +1.0, -1.0])

        dfz_total = dfz_warp + dfz_slope_pitch + dfz_slope_roll
        max_delta = self.max_dfz_ratio * fz_static_per_wheel
        dfz_total = np.clip(dfz_total, -max_delta, max_delta)

        # 3. Gia tốc kích thích mặt đường phụ thêm truyền vào bồn nước
        g_acc = 9.81
        a_road_bx = g_acc * np.sin(theta_pitch_road)
        a_road_by = g_acc * np.sin(theta_roll_road)

        # 4. Hệ số ma sát ngẫu nhiên cục bộ chuẩn ISO 8608 và hiệu ứng Split-Mu 4 bánh
        mu_local = self.compute_local_friction(
            x_c, y_c, mu_nominal=mu_nominal, variation_scale=friction_variation_scale
        )
        mu_wheels = np.zeros(4)
        for i in range(4):
            bx, by = body_offsets[i]
            wx_i = x_c + c_y * bx - s_y * by
            wy_i = y_c + s_y * bx + c_y * by
            mu_wheels[i] = self.compute_local_friction(
                wx_i, wy_i, mu_nominal=mu_nominal, variation_scale=friction_variation_scale
            )

        split_mu_delta = float(np.max(mu_wheels) - np.min(mu_wheels))
        mu_min_wheel = float(np.min(mu_wheels))

        return {
            "z_wheels_mm": z_wheels * 1000.0,
            "dfz_iso_wheels": dfz_total,
            "theta_pitch_road_rad": float(theta_pitch_road),
            "theta_roll_road_rad": float(theta_roll_road),
            "a_road_body": np.array([a_road_bx, a_road_by]),
            "mu_local": float(mu_local),
            "mu_wheels": mu_wheels,
            "mu_min_wheel": mu_min_wheel,
            "split_mu_delta": split_mu_delta,
        }


if __name__ == "__main__":
    import sys
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    print("=" * 72)
    print("KIỂM TRA MÔ-ĐUN ĐỊA HÌNH MẶT ĐƯỜNG NGẪU NHIÊN CHUẨN ISO 8608:2016")
    print("=" * 72)
    q_test = np.array([1.25, -0.65, 0.35])
    v_test = np.array([0.45, 0.10, 0.15])
    for r_cls in ["A_industrial", "A", "B", "C"]:
        prof = ISO8608RoadProfile(road_class=r_cls, seed=42)
        res = prof.compute_wheel_loads_and_slosh_excitation(q_test, v_test)
        print(
            f"Class {r_cls:<13} | Gd(n0)={prof.Gd_n0:.1e} m^3 | "
            f"z_wheels={np.round(res['z_wheels_mm'], 2)} mm | "
            f"dFz_ISO={np.round(res['dfz_iso_wheels'], 2)} N"
        )
    print("-" * 72)
    print("Kiểm tra phân vùng ma sát vĩ mô + vi mô ISO 8608 Class B trên world7.xml:")
    p_b = ISO8608RoadProfile(road_class="B", seed=32)
    test_zones = [
        ("Sàn bê tông xuất phát", -2.25, -2.25),
        ("Chicane Epoxy CP1", -2.15, 0.20),
        ("Điểm uốn chữ S CP3", 0.15, 1.85),
        ("Sàn dầu trơn cực hạn CP4", -0.23, -1.60),
        ("Cầu dẫn nghiêng 5.5° CP5", 0.96, -0.22),
        ("Khu vực về đích Goal", 2.25, 2.25),
    ]
    for z_name, zx, zy in test_zones:
        mu_macro = get_world7_zone_friction(zx, zy)
        mu_iso = p_b.compute_local_friction(zx, zy, mu_nominal=get_world7_zone_friction)
        print(f"  {z_name:<28} | (x={zx:+5.2f}, y={zy:+5.2f}) | mu_macro={mu_macro:.2f} -> mu_ISO={mu_iso:.3f}")
    print("=" * 72)
