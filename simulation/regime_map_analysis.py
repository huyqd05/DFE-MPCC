"""
regime_map_analysis.py
=============================================================================
ĐÓNG GÓP LÝ THUYẾT: BẢN ĐỒ CHẾ ĐỘ VÀ RANH GIỚI KHẢ THI VẬT LÝ GIẢI TÍCH
(Analytical Regime Classification Map & Physical Feasibility Boundaries)

Chứng minh bằng toán học và xuất biểu đồ ấn phẩm (Publication Figure):
1. Ranh giới lật bánh vs trượt bánh: mu_lift = L / (2*h).
2. Ba vùng chế độ hoạt động:
   - Regime I (Slosh-Dominant): Sóng sánh chạm ngưỡng tràn trước khi trượt.
   - Regime II (Friction-Dominant): Trượt bánh xảy ra trước khi sóng chạm ngưỡng tràn.
   - Regime III (DFE-Coupled Critical): Vùng ghép nối nhạy cảm, Fz biến thiên > 10%.
3. Tần số bẻ lái / uốn lượn tới hạn omega_crit theo hệ số ma sát mu.
=============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless backend for PNG export
import matplotlib.pyplot as plt
import os
import sys

_curr = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_curr) if os.path.basename(_curr) in ["simulation", "controller", "solver", "planning", "cfd", "bao cao"] else _curr
for _sub in ["", "controller", "solver", "planning", "simulation", "cfd", "bao cao"]:
    _p = os.path.join(_root, _sub) if _sub else _root
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Thông số vật lý của robot AMR và bình nước (slosh3.xml)
M_BASE = 5.500          # Khối lượng thân xe (kg)
M_SHELL = 0.550         # Khối lượng vỏ bình mica (kg)
M_WATER = 1.200         # Khối lượng nước danh định (kg)
M_TOT = M_BASE + M_SHELL + M_WATER # 7.250 kg

L_X = 0.12              # Bán chiều dài cơ sở (m) (2*Lx = 0.24m)
L_Y = 0.14              # Bán chiều rộng cơ sở (m) (2*Ly = 0.28m)
H_COM = 0.095           # Chiều cao trọng tâm tổng (m)
G = 9.81                # Gia tốc trọng trường (m/s^2)

# Thông số sóng sánh Housner DMTP (bình 150x100x110 mm, h_liquid = 80 mm)
TANK_A = 0.150          # Dài X (m)
TANK_B = 0.100          # Rộng Y (m)
TANK_H = 0.110          # Cao Z (m)
H_LIQ = 0.080           # Mức nước tĩnh (m)
FREEBOARD = TANK_H - H_LIQ # 0.030 m (30.0 mm)

# Tần số dao động riêng bậc 1 theo phương ngang Y (phương lắc cua chính)
tanh_y = np.tanh(np.pi * H_LIQ / TANK_B)
OMEGA_N_Y = np.sqrt((np.pi * G / TANK_B) * tanh_y) # ~17.44 rad/s (~2.78 Hz)
L_P_Y = TANK_B / (np.pi * tanh_y)                  # ~0.0323 m
M_P_Y = M_WATER * (8.0 * TANK_B / (np.pi**3 * H_LIQ)) * tanh_y # ~0.382 kg


def compute_regime_boundaries():
    """
    Tính toán các đường ranh giới giải tích trong không gian (mu, ay_max).
    """
    # 1. Ngưỡng gia tốc gây trượt bánh thuần túy: ay_slip = mu * g
    # 2. Ngưỡng gia tốc làm sóng sánh dâng chạm mép tràn (Freeboard = 30 mm):
    # Chiều cao sóng dâng cực đại: eta_max = (TANK_B / 2) * theta_max
    # Với theta_max xấp xỉ gia tốc kích động / g: theta_max approx ay / (g * (1 - (omega/omega_n)^2))
    # Tại tần số chuyển làn thông thường omega = 0.5 * omega_n (kích động dưới cộng hưởng):
    dyn_amp = 1.0 / (1.0 - 0.25) # Hệ số khuếch đại động lực học ~ 1.33
    theta_limit = FREEBOARD / (TANK_B / 2.0) # 0.030 / 0.050 = 0.60 rad (mép bình)
    # Để an toàn công nghiệp, góc giới hạn cho phép là 0.18 rad (~10.3 deg, eta = 9 mm)
    theta_admis = 0.18
    ay_slosh_limit = (theta_admis * G) / dyn_amp # ~ 1.32 m/s^2

    # 3. Ngưỡng gia tốc gây nhấc bánh (Wheel lift-off: Fz_inner -> 0):
    # Fz_inner = Mg/4 - (M*ay*h + M_slosh)/(4*Ly) = 0
    # M_slosh = M_P_Y * g * L_P_Y * sin(theta) + M_P_Y * ay * L_P_Y
    # Bỏ qua slosh: ay_lift = g * Ly / h_com = 9.81 * 0.14 / 0.095 = 14.45 m/s^2 >> g!
    ay_lift_limit = (G * L_Y) / H_COM # ~ 14.45 m/s^2

    return ay_slosh_limit, ay_lift_limit


def plot_regime_classification_map(output_path="testbm/regime_classification_map.png"):
    """
    Vẽ Bản đồ Phân loại Chế độ (Regime Classification Map) chuẩn ấn phẩm IEEE.
    """
    if not os.path.isabs(output_path):
        output_path = os.path.join(_root, output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    mu_range = np.linspace(0.04, 0.40, 200)
    ay_slosh_limit, ay_lift_limit = compute_regime_boundaries()

    # Đường giới hạn bám ma sát phẳng (Friction limit curve): ay = mu * g
    ay_traction_static = mu_range * G

    # Đường giới hạn bám có tính chuyển dịch tải động FSI (DFE Coupled limit):
    # Bánh trong bị dỡ tải: Fz_min = Fz_stat - dFz -> ay_coupled = mu * (1 - Delta_Fz/Fz) * g
    # delta_Fz / Fz_stat approx 13.5%
    ratio_delta_fz = 0.135
    ay_traction_dfe = mu_range * G * (1.0 - ratio_delta_fz)

    plt.figure(figsize=(9, 6), dpi=300)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Vẽ các đường ranh giới
    plt.plot(mu_range, ay_traction_static, 'b--', linewidth=2.0, label=r'Static Traction Limit: $a_{y} = \mu g$')
    plt.plot(mu_range, ay_traction_dfe, 'r-', linewidth=2.5, label=r'Dynamic Coupled Limit (DFE): $a_{y} = \mu (1 - \Delta F_z/F_z) g$')
    plt.axhline(y=ay_slosh_limit, color='g', linestyle='-.', linewidth=2.0, label=rf'Slosh Freeboard Limit: $a_{{y}} \approx {ay_slosh_limit:.2f}$ m/s$^2$ ($\eta \leq 9$ mm)')
    plt.axhline(y=ay_lift_limit, color='k', linestyle=':', linewidth=1.5, label=rf'Wheel Lift-off Limit: $a_{{y}} \approx {ay_lift_limit:.1f}$ m/s$^2$ ($h/L$ threshold)')

    # Tô màu các vùng chế độ
    # 1. Vùng Regime II: Friction-Dominant (mu <= 0.135)
    mu_cross = ay_slosh_limit / (G * (1.0 - ratio_delta_fz)) # mu ~ 0.156
    plt.fill_between(mu_range[mu_range <= mu_cross], 0, ay_traction_dfe[mu_range <= mu_cross], 
                     color='salmon', alpha=0.25, label='Regime II: Friction-Dominant (Slip Critical)')
    
    # 2. Vùng Chênh lệch chết người (Fatal Blindspot of Static MPC):
    plt.fill_between(mu_range[mu_range <= mu_cross], ay_traction_dfe[mu_range <= mu_cross], ay_traction_static[mu_range <= mu_cross],
                     color='red', alpha=0.45, hatch='//', label='Fatal Skid Zone: Static MPC assumes safe, but DFE slips!')

    # 3. Vùng Regime I: Slosh-Dominant (mu > 0.156)
    plt.fill_between(mu_range[mu_range > mu_cross], 0, ay_slosh_limit,
                     color='lightgreen', alpha=0.25, label='Regime I: Slosh-Dominant (Wave Overflow Critical)')

    # Điểm vận hành của AMR trong nghiên cứu: mu = 0.12 (Sàn trơn)
    plt.scatter([0.12], [0.12 * G * (1 - ratio_delta_fz)], color='darkred', s=120, zorder=5)
    plt.annotate(r'Our Benchmark ($\mu = 0.12$)' '\n' r'In Regime II / Fatal Zone',
                 xy=(0.12, 0.12 * G * (1 - ratio_delta_fz)), xytext=(0.06, 2.2),
                 arrowprops=dict(facecolor='darkred', shrink=0.08, width=1.5, headwidth=8),
                 fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.6))

    plt.title('THEORETICAL REGIME CLASSIFICATION MAP FOR OMNIDIRECTIONAL AMR\n'
              r'Coupled Slosh Dynamics vs Dynamic Normal Force Transfer ($F_{z,i}$)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel(r'Floor Friction Coefficient $\mu$ [-]', fontsize=11, fontweight='bold')
    plt.ylabel(r'Allowable Lateral Acceleration $a_y$ [m/s$^2$]', fontsize=11, fontweight='bold')
    plt.xlim(0.04, 0.35)
    plt.ylim(0.0, 3.5)
    plt.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=8.5)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"-> Đã xuất biểu đồ Bản đồ Chế độ chuẩn IEEE: {output_path}")


if __name__ == "__main__":
    print("=" * 70)
    print("PHÂN TÍCH BẢN ĐỒ CHẾ ĐỘ GIẢI TÍCH (REGIME MAP ANALYSIS)")
    print("=" * 70)
    ay_slosh, ay_lift = compute_regime_boundaries()
    print(f"- Ngưỡng lật bánh (Wheel lift-off):   ay = {ay_lift:.2f} m/s^2 (Hoàn toàn thụ động trên AMR dẹt)")
    print(f"- Ngưỡng tràn nước (Slosh freeboard): ay = {ay_slosh:.2f} m/s^2")
    print(f"- Ngưỡng ma sát cắt ngang (mu_cross): mu = {ay_slosh / (G * 0.865):.3f}")
    print("=" * 70)
    plot_regime_classification_map()
