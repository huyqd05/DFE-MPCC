"""
simulation/generate_slosh_xmls.py
=============================================================================
Tự động sinh các tệp mô hình MuJoCo XML cho các mực nước khác nhau:
  - xml/slosh_h40.xml (h = 40 mm, M_liquid = 0.600 kg)
  - xml/slosh_h60.xml (h = 60 mm, M_liquid = 0.900 kg)
  - xml/slosh_h80.xml (h = 80 mm, M_liquid = 1.200 kg, danh định)
Bảo đảm tính bảo toàn khối lượng và nhất quán thông số vật lý 100%.
=============================================================================
"""

import os
import sys
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

_curr = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_curr) if os.path.basename(_curr) in ["simulation", "controller", "solver", "planning", "cfd", "bao cao"] else _curr
xml_dir = os.path.join(_root, "xml")
os.makedirs(xml_dir, exist_ok=True)

# Thông số hình học bình acrylic
TANK_A = 0.150      # Dài lòng trong X (m)
TANK_B = 0.100      # Rộng lòng trong Y (m)
TANK_H = 0.110      # Cao lòng trong Z (m)
M_CART = 5.500      # Khối lượng bệ xe (kg)
M_SHELL = 0.550     # Khối lượng vỏ bình + cảm biến (kg)
RHO_WATER = 1000.0  # kg/m^3
G = 9.81            # m/s^2
ZETA = 0.0150       # Hệ số cản dao động nước thực tế (1.50%)

WATER_LEVELS = [0.040, 0.060, 0.080]


def compute_housner_params(h):
    m_liq = RHO_WATER * TANK_A * TANK_B * h
    
    # Trục X (chiều dài a = 0.150 m)
    xi_x = np.pi * h / TANK_A
    tanh_x = np.tanh(xi_x)
    sinh_x = np.sinh(xi_x)
    cosh_x = np.cosh(xi_x)
    m_1x = m_liq * (8.0 * TANK_A / (np.pi**3 * h)) * tanh_x
    m_0x = m_liq - m_1x
    l_px = TANK_A / (np.pi * tanh_x)
    omega_nx = np.sqrt((np.pi * G / TANK_A) * tanh_x)
    c_x = 2.0 * ZETA * omega_nx * m_1x * (l_px**2)
    # Housner convective heights
    h1_w_x = h * (1.0 - (cosh_x - 1.0) / (xi_x * sinh_x))
    h1_star_x = h * (1.0 - (cosh_x - 2.0) / (xi_x * sinh_x))
    
    # Trục Y (chiều rộng b = 0.100 m)
    xi_y = np.pi * h / TANK_B
    tanh_y = np.tanh(xi_y)
    sinh_y = np.sinh(xi_y)
    cosh_y = np.cosh(xi_y)
    m_1y = m_liq * (8.0 * TANK_B / (np.pi**3 * h)) * tanh_y
    m_0y = m_liq - m_1y
    l_py = TANK_B / (np.pi * tanh_y)
    omega_ny = np.sqrt((np.pi * G / TANK_B) * tanh_y)
    c_y = 2.0 * ZETA * omega_ny * m_1y * (l_py**2)
    # Housner convective heights
    h1_w_y = h * (1.0 - (cosh_y - 1.0) / (xi_y * sinh_y))
    h1_star_y = h * (1.0 - (cosh_y - 2.0) / (xi_y * sinh_y))
    
    # Khối lượng tĩnh bù vào container trong MuJoCo
    m0_equiv = m_liq - (m_1x + m_1y)
    container_mass = M_SHELL + m0_equiv
    total_mass = M_CART + container_mass + m_1x + m_1y
    
    # Cao độ khớp quay con lắc trên bệ xe:
    # Đáy trong bình ở cao độ z = 0.012 m (mặt trên đáy mica 2mm).
    # Pivot con lắc nằm tại z_pivot = z_floor + h1_w + l_p (để quả cầu ở cao độ h1_w)
    # hoặc theo quy ước slosh3.xml: pos_pivot = h_pivot
    z_floor = 0.012
    h_piv_x = h1_w_x + l_px
    h_piv_y = h1_w_y + l_py
    
    return {
        'h': h,
        'm_liq': m_liq,
        'm_1x': m_1x,
        'm_0x': m_0x,
        'l_px': l_px,
        'omega_nx': omega_nx,
        'f_nx': omega_nx / (2.0 * np.pi),
        'c_x': c_x,
        'h1_w_x': h1_w_x,
        'h1_star_x': h1_star_x,
        'z_slosh_x': z_floor + h1_star_x,
        'h_piv_x': h_piv_x,
        
        'm_1y': m_1y,
        'm_0y': m_0y,
        'l_py': l_py,
        'omega_ny': omega_ny,
        'f_ny': omega_ny / (2.0 * np.pi),
        'c_y': c_y,
        'h1_w_y': h1_w_y,
        'h1_star_y': h1_star_y,
        'z_slosh_y': z_floor + h1_star_y,
        'h_piv_y': h_piv_y,
        
        'm0_equiv': m0_equiv,
        'container_mass': container_mass,
        'total_mass': total_mass
    }


def generate_xml_content(p):
    h_mm = int(p['h'] * 1000)
    xml_text = f"""<mujoco model="amr_sloshing_h{h_mm}mm">
    <!-- Tự động sinh từ simulation/generate_slosh_xmls.py
         Mực nước h = {h_mm} mm | M_liquid = {p['m_liq']:.4f} kg | Tổng khối lượng = {p['total_mass']:.4f} kg -->
    <compiler angle="radian" coordinate="local"/>
    <option gravity="0 0 -9.81" timestep="0.002" integrator="RK4"/>

    <default>
        <geom friction="0.8 0.01 0.001" solref="0.004 1" solimp="0.95 0.99 0.001"/>
        <joint damping="0.01"/>
    </default>

    <visual>
        <headlight ambient="0.4 0.4 0.4" diffuse="0.8 0.8 0.8" specular="0.1 0.1 0.1"/>
        <rgba haze="0.15 0.25 0.35 1"/>
    </visual>

    <worldbody>
        <!-- Mặt sàn phẳng kiểm chứng (C2 smooth floor) -->
        <geom name="ground" type="plane" size="10 10 0.1" rgba="0.92 0.93 0.95 1" contype="1" conaffinity="1"/>
        <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>

        <!-- KHUNG XE CƠ SỞ (CART BASE: 5.500 kg, kích thước 0.32 x 0.24 x 0.05 m) -->
        <body name="cart" pos="0 0 0.040">
            <inertial pos="0 0 0" mass="5.500" diaginertia="0.035 0.050 0.075"/>
            <!-- 3 Khớp bậc tự do phẳng (X, Y, Theta_Z) -->
            <joint name="slide_x" type="slide" axis="1 0 0" damping="2.0"/>
            <joint name="slide_y" type="slide" axis="0 1 0" damping="2.0"/>
            <joint name="hinge_z" type="hinge" axis="0 0 1" damping="1.2"/>

            <!-- Khung vỏ bảo vệ xe -->
            <geom type="box" size="0.16 0.12 0.025" rgba="0.25 0.28 0.32 0.85" contype="0" conaffinity="0"/>

            <!-- 4 Bánh xe Mecanum (con lăn tiếp xúc sàn R=0.04m, nửa khoảng cách Lx=0.12m, Ly=0.14m) -->
            <geom name="wheel_FL" type="cylinder" size="0.04 0.02" pos="0.12 0.14 -0.02" euler="1.5708 0 0" rgba="0.2 0.2 0.2 1" contype="0" conaffinity="0"/>
            <geom name="wheel_FR" type="cylinder" size="0.04 0.02" pos="0.12 -0.14 -0.02" euler="1.5708 0 0" rgba="0.2 0.2 0.2 1" contype="0" conaffinity="0"/>
            <geom name="wheel_RL" type="cylinder" size="0.04 0.02" pos="-0.12 0.14 -0.02" euler="1.5708 0 0" rgba="0.2 0.2 0.2 1" contype="0" conaffinity="0"/>
            <geom name="wheel_RR" type="cylinder" size="0.04 0.02" pos="-0.12 -0.14 -0.02" euler="1.5708 0 0" rgba="0.2 0.2 0.2 1" contype="0" conaffinity="0"/>

            <!-- BÌNH ACRYLIC: Kích thước trong 150 x 100 x 110 mm
                 Khối lượng container = vỏ mica (0.550 kg) + nước tĩnh bù m0_equiv ({p['m0_equiv']:.4f} kg) = {p['container_mass']:.4f} kg -->
            <body name="container" pos="0 0 0.012">
                <inertial pos="0 0 0.055" mass="{p['container_mass']:.4f}" diaginertia="0.001 0.0015 0.002"/>
                <site name="pivot_site" pos="0 0 {p['h']:.3f}" size="0.008" rgba="1 0 0 0"/>

                <!-- Đáy bình mica 2mm -->
                <geom type="box" size="0.077 0.052 0.001" pos="0 0 0" rgba="0.5 0.8 1 0.3" contype="0" conaffinity="0"/>
                <!-- Các thành bình mica -->
                <geom type="box" size="0.076 0.001 0.055" pos="0 -0.051 0.055" rgba="0.5 0.8 1 0.25" contype="0" conaffinity="0"/>
                <geom type="box" size="0.076 0.001 0.055" pos="0  0.051 0.055" rgba="0.5 0.8 1 0.25" contype="0" conaffinity="0"/>
                <geom type="box" size="0.001 0.051 0.055" pos="-0.076 0 0.055" rgba="0.5 0.8 1 0.25" contype="0" conaffinity="0"/>
                <geom type="box" size="0.001 0.051 0.055" pos=" 0.076 0 0.055" rgba="0.5 0.8 1 0.25" contype="0" conaffinity="0"/>

                <!-- Khối nước tĩnh trực quan -->
                <geom type="box" size="0.075 0.050 {p['h']/2.0:.4f}" pos="0 0 {p['h']/2.0:.4f}" rgba="0 0.5 1 0.25" contype="0" conaffinity="0"/>

                <!-- 1. CON LẮC HOUSNER X (lx = {p['l_px']:.4f} m, m1x = {p['m_1x']:.4f} kg, damping = {p['c_x']:.6f}) -->
                <body name="slosh_pendulum_x" pos="0 0 {p['h_piv_x']:.4f}">
                    <inertial pos="0 0 -{p['l_px']:.4f}" mass="{p['m_1x']:.4f}" diaginertia="2e-5 2e-5 2e-5"/>
                    <joint name="slosh_x" type="hinge" axis="0 1 0" damping="{p['c_x']:.6f}"/>
                    <geom type="cylinder" size="0.002 {p['l_px']/2.0:.4f}" pos="0 0 -{p['l_px']/2.0:.4f}" rgba="0.8 0.9 0.8 1" mass="0" contype="0" conaffinity="0"/>
                    <geom type="sphere" size="0.015" pos="0 0 -{p['l_px']:.4f}" rgba="0 0.9 1 1" mass="0" contype="0" conaffinity="0"/>
                </body>

                <!-- 2. CON LẮC HOUSNER Y (ly = {p['l_py']:.4f} m, m1y = {p['m_1y']:.4f} kg, damping = {p['c_y']:.6f}) -->
                <body name="slosh_pendulum_y" pos="0 0 {p['h_piv_y']:.4f}">
                    <inertial pos="0 0 -{p['l_py']:.4f}" mass="{p['m_1y']:.4f}" diaginertia="2e-5 2e-5 2e-5"/>
                    <joint name="slosh_y" type="hinge" axis="1 0 0" damping="{p['c_y']:.6f}"/>
                    <geom type="cylinder" size="0.002 {p['l_py']/2.0:.4f}" pos="0 0 -{p['l_py']/2.0:.4f}" rgba="0.9 0 0.8 0.8" mass="0" contype="0" conaffinity="0"/>
                    <geom type="sphere" size="0.013" pos="0 0 -{p['l_py']:.4f}" rgba="1 0 0.9 1" mass="0" contype="0" conaffinity="0"/>
                </body>
            </body>
        </body>
    </worldbody>

    <sensor>
        <force name="fluid_force" site="pivot_site"/>
    </sensor>

    <actuator>
        <motor name="motor_x" joint="slide_x" gear="1" ctrllimited="true" ctrlrange="-150 150"/>
        <motor name="motor_y" joint="slide_y" gear="1" ctrllimited="true" ctrlrange="-150 150"/>
        <motor name="motor_z" joint="hinge_z" gear="1" ctrllimited="true" ctrlrange="-50 50"/>
    </actuator>
</mujoco>
"""
    return xml_text


def main():
    print("=" * 80)
    print("TỰ ĐỘNG SINH CÁC TỆP MÔ HÌNH MUJOCO XML THEO MỰC NƯỚC (HOUSNER DMTP)")
    print("=" * 80)
    
    summary = []
    for h in WATER_LEVELS:
        p = compute_housner_params(h)
        h_mm = int(h * 1000)
        xml_path = os.path.join(xml_dir, f"slosh_h{h_mm}.xml")
        xml_content = generate_xml_content(p)
        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(xml_content)
        print(f">> Đã sinh: {xml_path}")
        print(f"   h = {h_mm} mm | M_liq = {p['m_liq']:.3f} kg | Container = {p['container_mass']:.4f} kg | M_tot = {p['total_mass']:.4f} kg")
        print(f"   X: m1x = {p['m_1x']:.4f} kg, lx = {p['l_px']*1000:.1f} mm, fx = {p['f_nx']:.3f} Hz, cx = {p['c_x']:.6f}, h1* = {p['h1_star_x']*1000:.1f} mm")
        print(f"   Y: m1y = {p['m_1y']:.4f} kg, ly = {p['l_py']*1000:.1f} mm, fy = {p['f_ny']:.3f} Hz, cy = {p['c_y']:.6f}, h1* = {p['h1_star_y']*1000:.1f} mm")
        summary.append(p)
        
        # Nếu là h = 80 mm, đồng bộ cập nhật vào slosh3.xml
        if h_mm == 80:
            slosh3_path = os.path.join(xml_dir, "slosh3.xml")
            with open(slosh3_path, "w", encoding="utf-8") as f:
                f.write(xml_content)
            print(f">> Đã đồng bộ slosh3.xml với slosh_h80.xml!")
            
    print("=" * 80)
    print("HOÀN TẤT SINH XML TỰ ĐỘNG!")
    print("=" * 80)
    return summary


if __name__ == "__main__":
    main()
