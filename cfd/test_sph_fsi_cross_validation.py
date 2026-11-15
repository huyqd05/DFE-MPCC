"""
test_sph_fsi_cross_validation.py
=============================================================================
KIỂM TRA CHÉO ĐỘC LẬP: LỰC TÍNH TOÁN LÝ THUYẾT (HOUSNER DMTP) VS
LỰC MÔ PHỎNG THỦY ĐỘNG LỰC HỌC CFD / NAVIER-STOKES (WCSPH 2D)

Mục đích:
1. Thiết lập kịch bản kiểm chứng khách quan (Giao thức tiền đăng ký).
2. Tách bạch thành phần quán tính vật rắn: So sánh trực tiếp lực sóng sánh dư
   F_slosh(t) = F_y(t) - (-M_liquid * a_y(t)).
3. Áp dụng bộ lọc Butterworth không lệch pha loại bỏ nhiễu âm học số trong WCSPH.
4. Báo cáo các chỉ số định lượng chuẩn ấn phẩm IEEE:
   - Nash-Sutcliffe Efficiency (NSE)
   - Normalized RMSE (chuẩn hóa theo RMS sau lọc)
   - Độ trễ pha tau_lag qua tương quan chéo (ms)
   - Sai số đỉnh phân vị 99 (P99 Peak Error) và sai số thời điểm đỉnh
   - Xung lực tích lũy (Total Impulse)
5. Xuất biểu đồ kiểm chứng 4 đồ thị chất lượng cao vào:
   testbm/sph_cfd_vs_housner_cross_validation.png
=============================================================================
"""

import numpy as np
import scipy.signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

_curr = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_curr) if os.path.basename(_curr) in ["simulation", "controller", "solver", "planning", "cfd", "bao cao"] else _curr
for _sub in ["", "controller", "solver", "planning", "simulation", "cfd", "bao cao"]:
    _p = os.path.join(_root, _sub) if _sub else _root
    if _p not in sys.path:
        sys.path.insert(0, _p)
import time

# Import lõi SPH và Housner
from sph_navier_stokes_engine import WCSPH2DSolver, HousnerModel

# Đảm bảo UTF-8 cho Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)


def generate_maneuver_acceleration(t_span=2.5, dt=0.001, a_peak=1.20, f_ex=1.60):
    """
    Tạo chuỗi gia tốc ngang a_y(t) mô phỏng cú đánh lái chuyển làn kép (Chicane)
    của xe AMR mang bình nước.
    """
    t = np.arange(0.0, t_span, dt)
    # Bao hình mượt (Smooth envelope) để gia tốc và jerk liên tục tại t=0 và t=t_span
    envelope = np.sin(np.pi * t / t_span) ** 2
    a_y = a_peak * np.sin(2.0 * np.pi * f_ex * t) * envelope
    return t, a_y


def apply_zero_phase_lpf(data, fs, f_cutoff=11.0):
    """
    Bộ lọc thông thấp Butterworth bậc 4 không lệch pha (filtfilt).
    f_cutoff = 11.0 Hz (gấp ~4 lần tần số dao động riêng 2.78 Hz, giữ nguyên
    sóng sánh bậc cao nhưng triệt tiêu hoàn toàn sóng âm nhân tạo của WCSPH).
    """
    nyq = 0.5 * fs
    normal_cutoff = min(f_cutoff / nyq, 0.95)
    b, a = scipy.signal.butter(4, normal_cutoff, btype='low', analog=False)
    filtered = scipy.signal.filtfilt(b, a, data)
    return filtered


def compute_cross_validation_metrics(t, f_sim, f_calc, fs):
    """
    Tính toán toàn diện các chỉ số thống kê đối chuẩn theo yêu cầu khoa học.
    """
    # 1. Nash-Sutcliffe Efficiency (NSE)
    numerator = np.sum((f_sim - f_calc) ** 2)
    denominator = np.sum((f_sim - np.mean(f_sim)) ** 2)
    nse = 1.0 - (numerator / denominator) if denominator > 1e-12 else 0.0

    # 2. Normalized RMSE (chuẩn hóa theo RMS của tín hiệu mô phỏng)
    rmse = np.sqrt(np.mean((f_sim - f_calc) ** 2))
    rms_sim = np.sqrt(np.mean(f_sim ** 2))
    nrmse_pct = (rmse / rms_sim) * 100.0 if rms_sim > 1e-12 else 0.0

    # 3. Độ trễ pha qua tương quan chéo (Cross-Correlation Phase Lag)
    corr = scipy.signal.correlate(f_sim, f_calc, mode='full')
    lags = scipy.signal.correlation_lags(len(f_sim), len(f_calc), mode='full')
    lag_idx = lags[np.argmax(corr)]
    tau_lag_ms = (lag_idx / fs) * 1000.0 # milli-giây

    # 4. Phân vị 99 của đỉnh (P99 Peak) và Sai số thời điểm xuất hiện đỉnh
    p99_sim = np.percentile(np.abs(f_sim), 99)
    p99_calc = np.percentile(np.abs(f_calc), 99)
    err_p99_pct = np.abs(p99_calc - p99_sim) / p99_sim * 100.0 if p99_sim > 1e-12 else 0.0

    idx_peak_sim = np.argmax(np.abs(f_sim))
    idx_peak_calc = np.argmax(np.abs(f_calc))
    delta_t_peak_ms = np.abs(t[idx_peak_sim] - t[idx_peak_calc]) * 1000.0

    # 5. Xung lực tích lũy (Total Impulse)
    impulse_sim = np.trapezoid(np.abs(f_sim), t)
    impulse_calc = np.trapezoid(np.abs(f_calc), t)
    err_impulse_pct = np.abs(impulse_calc - impulse_sim) / impulse_sim * 100.0 if impulse_sim > 1e-12 else 0.0

    return {
        'NSE': nse,
        'RMSE': rmse,
        'NRMSE_pct': nrmse_pct,
        'tau_lag_ms': tau_lag_ms,
        'P99_sim': p99_sim,
        'P99_calc': p99_calc,
        'err_P99_pct': err_p99_pct,
        'delta_t_peak_ms': delta_t_peak_ms,
        'impulse_sim': impulse_sim,
        'impulse_calc': impulse_calc,
        'err_impulse_pct': err_impulse_pct
    }


def run_cross_validation_experiment(output_png="testbm/sph_cfd_vs_housner_cross_validation.png"):
    print("=" * 75)
    print("BẮT ĐẦU KIỂM TRA CHÉO: LÝ THUYẾT (HOUSNER) VS CFD/NAVIER-STOKES (WCSPH)")
    print("=" * 75)

    # 1. Khởi tạo bài toán và thông số
    t_span = 2.5
    dt_save = 0.002 # 500 Hz để lưu dữ liệu
    fs = 1.0 / dt_save
    t_profile, a_y_profile = generate_maneuver_acceleration(t_span=t_span, dt=dt_save, a_peak=1.20, f_ex=1.60)
    n_save = len(t_profile)

    # 2. Khởi tạo mô hình con lắc lý thuyết Housner
    housner = HousnerModel(
        tank_width=0.100,
        tank_height=0.110,
        liquid_height=0.080,
        tank_length=0.150,
        m_liquid=1.200,
        zeta=0.015
    )
    print(f">> [Housner] Tần số riêng f_n = {housner.f_n:.3f} Hz (omega_n = {housner.omega_n:.2f} rad/s)")
    print(f">> [Housner] Khối lượng dao động m_1 = {housner.m_1:.4f} kg | Khối lượng tĩnh m_0 = {housner.m_0:.4f} kg")

    # Giải mô hình con lắc lý thuyết
    res_housner = housner.simulate(t_profile, a_y_profile)
    F_y_theo = res_housner['F_y']
    F_slosh_theo = res_housner['F_slosh']
    M_roll_theo = res_housner['M_roll']
    eta_theo_mm = res_housner['eta_mm']
    theta_theo_deg = np.rad2deg(res_housner['theta'])

    # 3. Khởi tạo bộ giải CFD SPH Navier-Stokes
    dp_sph = 0.0035 # 3.5 mm
    sph = WCSPH2DSolver(
        tank_width=0.100,
        tank_height=0.110,
        liquid_height=0.080,
        tank_length=0.150,
        dp=dp_sph,
        rho_0=1000.0,
        c_0=12.0,
        alpha_visc=0.08
    )
    print(f">> [WCSPH] Khởi tạo hạt: {sph.n_fluid} hạt chất lỏng | {sph.n_bound} hạt biên mDBC | Tổng khối lượng: {sph.total_mass:.3f} kg", flush=True)

    # 4. Chạy mô phỏng tích phân số SPH
    dt_sph = 0.000060 # Bước tích phân CFL (60 microseconds)
    print(f">> [WCSPH] Bắt đầu tích phân SPH Navier-Stokes (dt = {dt_sph*1e6:.0f} us, t_span = {t_span} s)...", flush=True)

    # Buffer lưu trữ kết quả SPH tại các mốc thời gian dt_save
    F_y_sph_raw = np.zeros(n_save)
    F_slosh_sph_raw = np.zeros(n_save)
    F_z_sph_raw = np.zeros(n_save)
    M_roll_sph_raw = np.zeros(n_save)
    eta_sph_left = np.zeros(n_save)
    eta_sph_right = np.zeros(n_save)

    sim_time = 0.0
    step_count = 0
    save_idx = 0
    t_start = time.time()

    while sim_time < t_span and save_idx < n_save:
        a_curr = np.interp(sim_time, t_profile, a_y_profile)

        fy, f_slosh, fz, mroll, eta_l, eta_r = sph.step(dt_sph, a_curr)


        sim_time += dt_sph
        step_count += 1

        if step_count % 5000 == 0:
            print(f">> [SPH Tiến độ] Bước {step_count} (t = {sim_time:.2f} s / {t_span:.1f} s) | F_y = {fy:.3f} N | M_roll = {mroll:.3f} N.m", flush=True)

        # Lưu dữ liệu khi chạm mốc thời gian lưu
        if save_idx < n_save and sim_time >= t_profile[save_idx]:
            F_y_sph_raw[save_idx] = fy
            F_slosh_sph_raw[save_idx] = f_slosh
            F_z_sph_raw[save_idx] = fz
            M_roll_sph_raw[save_idx] = mroll
            eta_sph_left[save_idx] = eta_l
            eta_sph_right[save_idx] = eta_r
            save_idx += 1

    t_elapsed = time.time() - t_start
    print(f">> [WCSPH] Hoàn thành {step_count} bước tích phân trong {t_elapsed:.2f} s ({t_elapsed/step_count*1000:.3f} ms/step).")

    # 5. Xử lý tín hiệu: Bộ lọc thông thấp không lệch pha đồng nhất
    f_cut = 11.0 # Hz
    F_y_sph = apply_zero_phase_lpf(F_y_sph_raw, fs, f_cutoff=f_cut)
    F_slosh_sph = apply_zero_phase_lpf(F_slosh_sph_raw, fs, f_cutoff=f_cut)
    M_roll_sph = apply_zero_phase_lpf(M_roll_sph_raw, fs, f_cutoff=f_cut)

    # Chiều cao mặt thoáng sóng sánh SPH (mm)
    eta_raw_mm = 0.5 * (eta_sph_right - eta_sph_left) * 1000.0
    eta_sph_mm = apply_zero_phase_lpf(eta_raw_mm, fs, f_cutoff=f_cut)

    # 6. Tính toán toàn bộ các chỉ số kiểm tra chéo khoa học
    metrics_total = compute_cross_validation_metrics(t_profile, F_y_sph, F_y_theo, fs)
    metrics_slosh = compute_cross_validation_metrics(t_profile, F_slosh_sph, F_slosh_theo, fs)
    metrics_moment = compute_cross_validation_metrics(t_profile, M_roll_sph, M_roll_theo, fs)

    print("\n" + "=" * 75)
    print("BÁO CÁO KẾT QUẢ ĐỐI CHUẨN ĐỊNH LƯỢNG (QUANTITATIVE CROSS-VALIDATION REPORT)")
    print("=" * 75)
    print(f"{'Chỉ số thống kê':<32} | {'Tổng lực F_y':<18} | {'Lực sóng sánh dư F_slosh':<18}")
    print("-" * 75)
    print(f"{'Nash-Sutcliffe Efficiency (NSE)':<32} | {metrics_total['NSE']:<18.4f} | {metrics_slosh['NSE']:<18.4f}")
    print(f"{'Normalized RMSE (NRMSE)':<32} | {metrics_total['NRMSE_pct']:<17.2f}% | {metrics_slosh['NRMSE_pct']:<17.2f}%")
    print(f"{'Độ trễ pha tau_lag (Cross-corr)':<32} | {metrics_total['tau_lag_ms']:<15.2f} ms | {metrics_slosh['tau_lag_ms']:<15.2f} ms")
    print(f"{'Sai số đỉnh phân vị P99':<32} | {metrics_total['err_P99_pct']:<17.2f}% | {metrics_slosh['err_P99_pct']:<17.2f}%")
    print(f"{'Lệch thời điểm đỉnh Delta_t_peak':<32} | {metrics_total['delta_t_peak_ms']:<15.2f} ms | {metrics_slosh['delta_t_peak_ms']:<15.2f} ms")
    print(f"{'Sai số xung lực (Impulse Error)':<32} | {metrics_total['err_impulse_pct']:<17.2f}% | {metrics_slosh['err_impulse_pct']:<17.2f}%")
    print("-" * 75)
    print(f"Mô-men lật thủy động M_roll: NSE = {metrics_moment['NSE']:.4f} | NRMSE = {metrics_moment['NRMSE_pct']:.2f}% | Lệch đỉnh P99 = {metrics_moment['err_P99_pct']:.2f}%")
    print("=" * 75)

    if not os.path.isabs(output_png):
        output_png = os.path.join(_root, output_png)
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), dpi=300, sharex=True)
    fig.patch.set_facecolor('#ffffff')

    # Đồ thị 1: Gia tốc kích động a_y(t) và Góc nghiêng con lắc theta(t)
    ax1 = axes[0]
    ax1.plot(t_profile, a_y_profile, 'k-', linewidth=2.0, label=r'Lateral Acceleration $a_y(t)$ [m/s$^2$]')
    ax1.set_ylabel(r'$a_y$ [m/s$^2$]', fontsize=10, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left', frameon=True)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(t_profile, theta_theo_deg, 'm--', linewidth=1.8, label=r'Slosh Angle $\theta_{\mathrm{theo}}(t)$ [deg]')
    ax1_twin.set_ylabel(r'$\theta$ [deg]', fontsize=10, fontweight='bold', color='m')
    ax1_twin.tick_params(axis='y', labelcolor='m')
    ax1_twin.legend(loc='upper right', frameon=True)
    ax1.set_title(r'$\mathbf{HIGH-FIDELITY\ CROSS-VALIDATION:\ WCSPH\ NAVIER-STOKES\ vs\ HOUSNER\ DMTP}$' '\n'
                  rf'Total Force $NSE = {metrics_total["NSE"]:.3f}$ | Sloshing Residual $NSE = {metrics_slosh["NSE"]:.3f}$ | Phase Lag $\tau = {metrics_slosh["tau_lag_ms"]:.1f}$ ms',
                  fontsize=11, fontweight='bold', pad=8)

    # Đồ thị 2: Tổng lực ngang F_y(t)
    ax2 = axes[1]
    ax2.plot(t_profile, F_y_sph, 'b-', linewidth=2.2, label=r'WCSPH Navier-Stokes Ground Truth ($F_{y, \mathrm{SPH}}$)')
    ax2.plot(t_profile, F_y_theo, 'r--', linewidth=1.8, label=r'Housner Reduced-Order Theory ($F_{y, \mathrm{Housner}}$)')
    ax2.set_ylabel(r'Total $F_y$ [N]', fontsize=10, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', frameon=True)

    # Đồ thị 3: Lực sóng sánh dư F_slosh(t) đã cách ly lực quán tính vật rắn
    ax3 = axes[2]
    ax3.plot(t_profile, F_slosh_sph, 'b-', linewidth=2.2, label=r'WCSPH Isolated Slosh Force $F_{\mathrm{slosh, SPH}}$')
    ax3.plot(t_profile, F_slosh_theo, 'r--', linewidth=1.8, label=r'Housner Isolated Slosh Force $F_{\mathrm{slosh, Housner}}$')
    # Dải băng sai số sai lệch tuyệt đối
    err_band = np.abs(F_slosh_sph - F_slosh_theo)
    ax3.fill_between(t_profile, F_slosh_theo - err_band, F_slosh_theo + err_band, color='gray', alpha=0.25, label=r'Absolute Error $|\Delta F_{\mathrm{slosh}}|$')
    ax3.set_ylabel(r'$F_{\mathrm{slosh}}$ [N]', fontsize=10, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend(loc='upper right', frameon=True)

    # Đồ thị 4: Mô-men lật thủy động M_roll(t)
    ax4 = axes[3]
    ax4.plot(t_profile, M_roll_sph, 'b-', linewidth=2.2, label=r'WCSPH Hydrodynamic Moment ($M_{\mathrm{roll, SPH}}$)')
    ax4.plot(t_profile, M_roll_theo, 'r--', linewidth=1.8, label=r'Housner Theoretical Moment ($M_{\mathrm{roll, Housner}}$)')
    ax4.set_ylabel(r'$M_{\mathrm{roll}}$ [N$\cdot$m]', fontsize=10, fontweight='bold')
    ax4.set_xlabel(r'Time $t$ [s]', fontsize=10, fontweight='bold')
    ax4.grid(True, linestyle='--', alpha=0.5)
    ax4.legend(loc='upper right', frameon=True)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f">> [Output] Đã lưu biểu đồ kiểm tra chéo: {output_png}")


if __name__ == "__main__":
    run_cross_validation_experiment()
