"""
cosim_bridge.py
=============================================================================
CẦU NỐI ĐỒNG MÔ PHỎNG MUJOCO - WCSPH 2D x 2 & SỔ SÁCH KHỐI LƯỢNG / MÔ-MEN
(Dự án: DFE-MPCC AMR Chở Chất Lỏng - Định hướng SIMPAT v2.0)

Thực hiện chuẩn mực cơ học từ Huong_Dan_Sua_Code_Chi_Tiet_SIMPAT.md (Module 1):
1. Phân tách khối lượng tĩnh - động: Trừ tường minh phần quán tính vật rắn
   (frozen rigid fluid) khỏi lực/mô-men thô của 2 lát cắt WCSPH 2D trước khi
   truyền vào MuJoCo (Lỗi A.7, A.11).
2. Quy đổi mô-men lật về mặt phẳng tiếp xúc bánh xe với mặt đất (ground contact plane)
   thông qua cánh tay đòn z_tank.
3. Phân bổ tải pháp tuyến thuần động do sóng sánh dFz_slosh cho 4 bánh xe Mecanum.
4. Kiểm toán phần dư năng lượng số học E_res(t) trong hệ quy chiếu gắn bình.
=============================================================================
"""

import numpy as np


class CoSimMassMomentBridge:
    def __init__(
        self,
        rho: float = 1000.0,
        Lx: float = 0.150,      # Chiều dài bình dọc trục x [m] (150 mm)
        Ly: float = 0.100,      # Chiều rộng bình dọc trục y [m] (100 mm)
        h: float = 0.080,       # Mực nước tĩnh [m] (80 mm)
        z_tank: float = 0.052,  # Độ cao đáy bình so với mặt đất [m]
        l_wb: float = 0.240,    # Chiều dài cơ sở (wheelbase = 2 * L_X = 0.24 m)
        t_w: float = 0.280,     # Vết bánh xe (track width = 2 * L_Y = 0.28 m)
        g: float = 9.81
    ):
        self.rho = rho
        self.Lx = Lx
        self.Ly = Ly
        self.h = h
        self.z_tank = z_tank
        self.l_wb = l_wb
        self.t_w = t_w
        self.g = g

        # Khối lượng chất lỏng trên một đơn vị bề rộng cho từng lát cắt 2D [kg/m]
        # Lát cắt x-z có chiều dài Lx, bề rộng chiếu Ly: m_bar_xz = rho * Lx * h
        # Lát cắt y-z có chiều rộng Ly, bề rộng chiếu Lx: m_bar_yz = rho * Ly * h
        self.m_bar_xz = self.rho * self.Lx * self.h
        self.m_bar_yz = self.rho * self.Ly * self.h
        self.z_c = 0.5 * self.h  # Trọng tâm tĩnh của nước tính từ đáy bình [m]

        # Khối lượng chất lỏng 3D tổng cộng [kg]
        self.m_fl_total = self.rho * self.Lx * self.Ly * self.h

    def compute_pure_dynamic_slosh_wrench(
        self,
        fx_raw_xz: float, fz_raw_xz: float, m_bottom_raw_xz: float,
        fy_raw_yz: float, fz_raw_yz: float, m_bottom_raw_yz: float,
        ax: float, ay: float, az: float = 0.0
    ) -> dict:
        """
        Trừ thành phần quán tính vật rắn (frozen rigid fluid) khỏi lực/mô-men thô của 2 lát cắt WCSPH 2D
        và quy đổi mô-men lật về mặt đất tiếp xúc bánh xe (ground contact plane).

        Hệ trục thân xe chuẩn (WP01):
          - x hướng tới trước, y hướng sang trái, z hướng lên trên.
        Quy ước mô-men quanh đáy bình:
          - m_bottom_raw_xz > 0 khi tạo mô-men chúc đầu về trước (pitch, nose-down/up)
          - m_bottom_raw_yz > 0 khi tạo mô-men nghiêng sang phải (roll, dồn tải bánh phải)

        Parameters:
        -----------
        fx_raw_xz, fz_raw_xz, m_bottom_raw_xz : Lực [N/m] và mô-men [N*m/m] thô lát cắt x-z
        fy_raw_yz, fz_raw_yz, m_bottom_raw_yz : Lực [N/m] và mô-men [N*m/m] thô lát cắt y-z
        ax, ay, az : Gia tốc tịnh tiến bệ xe trong hệ quy chiếu thân xe [m/s^2]
        """
        # 1. Thành phần vật rắn trên một đơn vị bề rộng [N/m] và [N*m/m] quanh tâm đáy bình:
        # Lực quán tính vật rắn d'Alembert trong hệ gắn bình:
        # F_inertial_x = - m_bar_xz * ax đặt tại độ cao z_c
        # Lực này tạo mô-men quanh tâm đáy bình: + m_bar_xz * z_c * ax
        fx_rigid_xz = -self.m_bar_xz * ax
        m_bottom_rigid_xz = self.m_bar_xz * self.z_c * ax

        # Tương tự cho trục y: khi xe gia tốc sang trái ay > 0, lực quán tính đẩy sang phải:
        # Fy_inertial = - m_bar_yz * ay. Đặt tại độ cao z_c, tạo mô-men quanh trục x:
        # Làm nước dồn sang phải (y < 0) đè thêm bánh phải -> mô-men dồn tải bánh phải m_bottom > 0.
        # Ở đây theo quy ước m_bottom_raw_yz > 0 khi đè bánh phải:
        # Thành phần vật rắn: m_bottom_rigid_yz = self.m_bar_yz * self.z_c * ay
        fy_rigid_yz = -self.m_bar_yz * ay
        m_bottom_rigid_yz = self.m_bar_yz * self.z_c * ay

        # 2. Lực và mô-men thuần động do sóng sánh (Purely Dynamic Slosh Component) nhân bề rộng 3D
        dFx_slosh = self.Ly * (fx_raw_xz - fx_rigid_xz)
        dFy_slosh = self.Lx * (fy_raw_yz - fy_rigid_yz)

        dM_pitch_bottom = self.Ly * (m_bottom_raw_xz - m_bottom_rigid_xz)
        dM_roll_bottom  = self.Lx * (m_bottom_raw_yz - m_bottom_rigid_yz)

        # 3. Quy đổi mô-men về mặt đất tiếp xúc bánh xe (cộng cánh tay đòn z_tank * lực ngang)
        # Tại mặt đất (z = 0), đáy bình ở z = z_tank:
        # Lực ngang dFx_slosh tác dụng tại đáy bình tạo thêm cánh tay đòn z_tank
        dM_pitch_ground = dM_pitch_bottom + self.z_tank * dFx_slosh
        dM_roll_ground  = dM_roll_bottom  + self.z_tank * dFy_slosh

        # 4. Phân bổ độ lệch tải pháp tuyến thuần động do slosh cho 4 bánh xe:
        # Thứ tự bánh: w=0 (FL: trước-trái), w=1 (FR: trước-phải), w=2 (RL: sau-trái), w=3 (RR: sau-phải)
        # s_pitch = -1 cho bánh trước, +1 cho bánh sau (khi dM_pitch_ground > 0 làm dồn tải ra bánh sau)
        # s_roll  = -1 cho bánh trái, +1 cho bánh phải (khi dM_roll_ground > 0 làm dồn tải sang bánh phải)
        s_pitch = np.array([-1.0, -1.0, +1.0, +1.0])
        s_roll  = np.array([-1.0, +1.0, -1.0, +1.0])

        dFz_slosh_wheels = (
            s_pitch * (dM_pitch_ground / (2.0 * self.l_wb)) +
            s_roll  * (dM_roll_ground  / (2.0 * self.t_w))
        )

        return {
            "dFx_slosh": float(dFx_slosh),               # Lực ngang x thuần động [N]
            "dFy_slosh": float(dFy_slosh),               # Lực ngang y thuần động [N]
            "dM_pitch_bottom": float(dM_pitch_bottom),   # Mô-men pitch tại đáy bình [N*m]
            "dM_roll_bottom": float(dM_roll_bottom),     # Mô-men roll tại đáy bình [N*m]
            "dM_pitch_ground": float(dM_pitch_ground),   # Mô-men pitch tại mặt đất [N*m]
            "dM_roll_ground": float(dM_roll_ground),     # Mô-men roll tại mặt đất [N*m]
            "dFz_slosh_wheels": dFz_slosh_wheels         # Độ lệch tải pháp tuyến 4 bánh [N]
        }


def compute_energy_residual_noninertial(
    W_inertial_cum: float,
    E_fl_rel_curr: float,
    E_fl_rel_0: float,
    E_num_diss_cum: float
) -> float:
    """
    Công thức Phần dư Năng lượng E_res(t) trong Hệ Quy chiếu Gắn Bình (Lỗi A.7).
    E_res(t) = | W_inertial(t) - (E_fl_rel(t) - E_fl_rel(0)) - E_num_diss(t) |
    """
    delta_E_fl = E_fl_rel_curr - E_fl_rel_0
    return abs(W_inertial_cum - delta_E_fl - E_num_diss_cum)
