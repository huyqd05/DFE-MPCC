"""
test_sim_amr_closed_loop.py
=============================================================================
Kịch bản Kiểm Chứng & Benchmark Mô Phỏng Đóng Vòng Cho Xe AMR (slosh3.xml)
Mang Bình Nước Sóng Sánh Cố Định Bằng Bộ Điều Khiển MPCC-NMPC Cấp Jerk

Quy trình:
1. Đọc tự động bản đồ, vật cản, vị trí Start/Goal từ slosh3.xml.
2. Hoạch định quỹ đạo toàn cục không va chạm bằng A* Planner.
3. Chạy điều khiển hồi tiếp vòng kín MPCC-NMPC cấp Jerk với bộ giải IPOPT.
4. Điều khiển bám cấp thấp (Low-level Controller) bằng bù quán tính M(q),
   bù ma sát Coulomb và bộ điều khiển PD hồi tiếp ở tần số 200 Hz.
5. Thu thập số liệu benchmark (thời gian, góc slosh, sóng dâng, thời gian giải, độ chính xác).
6. Xuất biểu đồ 6 đồ thị kỹ thuật phục vụ so sánh benchmark với Mobile Manipulator.
=============================================================================
"""

import os
import sys
import time
import argparse

_curr = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_curr) if os.path.basename(_curr) in ["simulation", "controller", "solver", "planning", "cfd", "bao cao"] else _curr
for _sub in ["", "controller", "solver", "planning", "simulation", "cfd", "bao cao"]:
    _p = os.path.join(_root, _sub) if _sub else _root
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mujoco
try:
    import mujoco.viewer
except ImportError:
    pass
import numpy as np

from astar import AStarPlanner
from mpcc_nmpc_amr_controller import MPCCNMPCAMRController
from obstacle_injector import create_injected_simulation_model
from robot_model_amr_casadi import (
    PHI_MAX_X, PHI_MAX_Y,
    TANK_A, TANK_B, TANK_H, H_LIQUID,
    ROBOT_RADIUS
)

# Đảm bảo in tiếng Việt chuẩn trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


# =====================================================================
# THIẾT LẬP CẢM BIẾN LIDAR 2D & BẢN ĐỒ OCCUPANCY GRID (CHO MPCC)
# =====================================================================
LIDAR_SITE = "lidar_site"
LIDAR_DT = 0.10            # Chu kỳ quét LiDAR 10 Hz (mỗi 0.10 s)
N_BEAMS = 180              # 180 tia quét 360 độ (độ phân giải 2 độ)
LIDAR_RANGE = 4.0          # Tầm quét tối đa (m)
LIDAR_NOISE = 0.01         # Độ nhiễu Gauss của khoảng cách đo (m)
OBS_GROUP = 1              # Nhóm geom cảm biến có thể nhìn thấy (group=1: tường, vật cản)

GRID_RES = 0.10            # Kích thước ô lưới Occupancy Grid (m)
GRID_HALF = 3.0            # Lưới phủ dải [-3, 3] x [-3, 3] (6x6 m)
L_OCC, L_FREE = 1.0, -0.6  # Hệ số log-odds
L_MIN, L_MAX, L_THRESH = -3.0, 3.0, 0.7


class Lidar:
    def __init__(self, model, rng=None):
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, LIDAR_SITE)
        if self.site_id < 0:
            for fallback_site in ["lidar_site", "pivot_site", "cart_site"]:
                s_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, fallback_site)
                if s_id >= 0:
                    self.site_id = s_id
                    break
        if self.site_id < 0 and model.nsite > 0:
            self.site_id = 0
        if self.site_id < 0:
            raise ValueError(f"Site '{LIDAR_SITE}' không tìm thấy trong mô hình MuJoCo!")
        ang = np.linspace(0.0, 2 * np.pi, N_BEAMS, endpoint=False)
        self.dirs_local = np.column_stack((np.cos(ang), np.sin(ang), np.zeros(N_BEAMS)))
        self.geomgroup = np.zeros(6, dtype=np.uint8)
        self.geomgroup[OBS_GROUP] = 1
        self.geomid = np.zeros(1, dtype=np.int32)
        self.rng = rng if rng is not None else np.random.default_rng(0)

    def scan(self, model, data):
        """Quét 180 tia ray-casting từ site của LiDAR trên thân xe AMR"""
        origin = data.site_xpos[self.site_id].copy()
        R = data.site_xmat[self.site_id].reshape(3, 3)
        dirs = self.dirs_local @ R.T
        raw = np.full(N_BEAMS, LIDAR_RANGE)
        for i in range(N_BEAMS):
            d = mujoco.mj_ray(model, data, origin, dirs[i], self.geomgroup, 1, -1, self.geomid)
            if 0.0 <= d < LIDAR_RANGE:
                raw[i] = d
        hit = raw < LIDAR_RANGE
        meas = raw.copy()
        meas[hit] += self.rng.normal(0.0, LIDAR_NOISE, int(hit.sum()))
        return origin, dirs, meas, hit, raw


class OccupancyGrid:
    def __init__(self):
        self.n = int(round(2 * GRID_HALF / GRID_RES))
        self.L = np.zeros((self.n, self.n))
        self.s = np.arange(0.0, LIDAR_RANGE, GRID_RES * 0.5)

    def cell(self, xy):
        return (int(np.floor((xy[0] + GRID_HALF) / GRID_RES)), int(np.floor((xy[1] + GRID_HALF) / GRID_RES)))

    def _flat(self, xy):
        idx = np.floor((xy + GRID_HALF) / GRID_RES).astype(int)
        ok = (idx[:, 0] >= 0) & (idx[:, 0] < self.n) & (idx[:, 1] >= 0) & (idx[:, 1] < self.n)
        return idx[ok, 0] * self.n + idx[ok, 1]

    def update(self, origin, dirs, ranges, hit):
        o, d = origin[:2], dirs[:, :2]
        r_free = np.where(hit, ranges - GRID_RES, LIDAR_RANGE)
        m = self.s[None, :] < r_free[:, None]
        pts = o[None, None, :] + d[:, None, :] * self.s[None, :, None]
        free = np.unique(self._flat(pts[m]))
        occ = np.unique(self._flat(o[None, :] + d[hit] * ranges[hit, None])) if hit.any() else np.array([], dtype=int)
        free = np.setdiff1d(free, occ)
        Lf = self.L.reshape(-1)
        Lf[free] = np.clip(Lf[free] + L_FREE, L_MIN, L_MAX)
        Lf[occ] = np.clip(Lf[occ] + L_OCC, L_MIN, L_MAX)

    def occupied_points(self):
        ix, iy = np.nonzero(self.L > L_THRESH)
        return np.column_stack(((ix + 0.5) * GRID_RES - GRID_HALF, (iy + 0.5) * GRID_RES - GRID_HALF))


def draw_sphere(viewer, pos, rgba, size=0.015):
    if viewer is None or viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        return
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([size, 0, 0]), pos,
                        np.eye(3).flatten(), np.array(rgba, dtype=np.float32))
    viewer.user_scn.ngeom += 1


def select_lidar_obstacle(scan_pts, occ_pts, q_meas, v_meas, s_prog, controller):
    """
    Trích xuất điểm chướng ngại vật nguy hiểm nhất từ cảm biến LiDAR và Occupancy Grid
    để nạp vào hàm ràng buộc Obstacle-ECBF của MPCC.
    """
    candidate_pts = scan_pts if len(scan_pts) > 0 else occ_pts
    if len(candidate_pts) == 0:
        return np.array([10.0, 10.0, 0.5, 0.1, 0.1]), 100.0

    # Lấy các điểm tham chiếu phía trước dọc theo đường đi (trong tầm 1.2 m)
    s_start = min(s_prog + 0.05, controller.path_length)
    s_end = min(s_prog + 1.2, controller.path_length)
    forward_s = np.linspace(s_start, s_end, 8) if s_end > s_start else np.array([s_start])
    ref_ahead = np.array([controller.get_path_reference(s_i)[0] for s_i in forward_s])

    # Khoảng cách từ các điểm LiDAR đến robot và đến tim đường dự kiến phía trước
    diff = candidate_pts - q_meas[:2]
    d_rob = np.hypot(diff[:, 0], diff[:, 1])
    d_path = np.min(np.hypot(candidate_pts[:, None, 0] - ref_ahead[None, :, 0],
                             candidate_pts[:, None, 1] - ref_ahead[None, :, 1]), axis=1)

    # Chiếu vị trí điểm lên hướng vận tốc robot để xác định điểm ở phía trước mặt (p_ahead > 0)
    v_norm = np.hypot(v_meas[0], v_meas[1])
    if v_norm > 0.05:
        p_ahead = (diff[:, 0] * v_meas[0] + diff[:, 1] * v_meas[1]) / v_norm
    else:
        # Nếu robot đứng yên hoặc vận tốc rất nhỏ, chiếu theo hướng mũi xe (yaw)
        yaw = q_meas[2]
        p_ahead = diff[:, 0] * np.cos(yaw) + diff[:, 1] * np.sin(yaw)

    # 1. Ưu tiên cao nhất: Điểm LiDAR nằm chắn ngang tim đường phía trước (d_path < 0.20m và d_rob < 1.1m)
    # Đây là vật thể chắn ngay trên hành lang di chuyển dự kiến (vật cản động cắt ngang đường)
    mask_block = (d_rob < 1.1) & (d_path < 0.20) & (p_ahead > -0.05)
    if np.any(mask_block):
        threat_pts = candidate_pts[mask_block]
        threat_dists = d_rob[mask_block]
        best_idx = np.argmin(threat_dists)
        p_crit = threat_pts[best_idx]
        return np.array([p_crit[0], p_crit[1], 0.5, 0.06, 0.16]), threat_dists[best_idx]

    # 2. Ưu tiên thứ hai: Vật cản tiến sát trực diện phía trước hướng di chuyển (p_ahead > 0.08m và d_rob < 0.25m)
    # Loại trừ các vách tường bên hông chạy song song với thân xe
    mask_close = (d_rob < 0.25) & (p_ahead > 0.08)
    if np.any(mask_close):
        close_pts = candidate_pts[mask_close]
        close_dists = d_rob[mask_close]
        best_idx = np.argmin(close_dists)
        p_crit = close_pts[best_idx]
        return np.array([p_crit[0], p_crit[1], 0.5, 0.02, 0.04]), close_dists[best_idx]

    # Không có nguy cơ tức thời
    return np.array([10.0, 10.0, 0.5, 0.1, 0.1]), 100.0


def extract_map_from_model(model):
    """
    Tự động trích xuất Start, Goal, Tường và Chướng ngại vật từ MuJoCo model.
    """
    obs_circles, obs_boxes = [], []
    map_geoms = []
    start_pos = np.array([-2.0, -2.0])
    goal_pos = np.array([2.0, 0.5])

    for i in range(model.ngeom):
        # Chỉ quét các geom thuộc thế giới cố định (bodyid == 0)
        if model.geom_bodyid[i] != 0:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name in ['ground']:
            continue

        pos = model.geom_pos[i]
        size = model.geom_size[i]
        rgba = model.geom_rgba[i]
        gtype = model.geom_type[i]

        if name == 'start_marker':
            start_pos = pos[:2].copy()
        elif name == 'goal_marker':
            goal_pos = pos[:2].copy()
        elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
            obs_circles.append([float(pos[0]), float(pos[1]), float(size[0])])
            map_geoms.append(('cylinder', pos[:2].copy(), float(size[0]), rgba.copy(), name))
        elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
            obs_boxes.append([float(pos[0]), float(pos[1]), float(size[0]), float(size[1])])
            map_geoms.append(('box', pos[:2].copy(), size[:2].copy(), rgba.copy(), name))

    dyn_geoms = []
    for i in range(model.ngeom):
        b_id = model.geom_bodyid[i]
        b_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b_id)
        if b_name and 'dyn' in b_name.lower():
            gtype = model.geom_type[i]
            r = float(model.geom_size[i][0])
            rgba = model.geom_rgba[i]
            dyn_geoms.append((i, r, gtype, rgba.copy(), b_name))

    return start_pos, goal_pos, obs_circles, obs_boxes, map_geoms, dyn_geoms


def update_dynamic_obstacles(model, data, t, custom_dyn_configs=None):
    """
    Cập nhật vị trí và vận tốc các vật thể động theo thời gian t.
    Tự động nhận diện các khớp có tiền tố 'dyn_joint'.
    """
    dyn_configs = {
        "dyn_joint_1": {"amp": 0.55, "omega": 1.50, "phase": 0.0},
        "dyn_joint_2": {"amp": 0.45, "omega": 1.05, "phase": 0.5},
        "dyn_joint_3": {"amp": 0.50, "omega": 0.95, "phase": 1.0},
    }
    if custom_dyn_configs:
        dyn_configs.update(custom_dyn_configs)

    for j_name, cfg in dyn_configs.items():
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
        if j_id >= 0:
            q_adr = model.jnt_qposadr[j_id]
            v_adr = model.jnt_dofadr[j_id]
            amp = cfg.get("amp", 0.50)
            w = cfg.get("omega", 1.0)
            ph = cfg.get("phase", 0.0)
            data.qpos[q_adr] = amp * np.sin(w * t + ph)
            data.qvel[v_adr] = amp * w * np.cos(w * t + ph)


def load_combined_model(map_name=None):
    """
    Tự động đọc bản đồ (từ map_name, active_map.txt hoặc world.xml)
    và ghép nối với xe AMR (từ slosh3.xml) thành một MjModel hoàn chỉnh.
    """
    workspace_dir = _root

    # 1. Đọc tên map nếu chưa truyền vào
    if not map_name:
        active_txt = os.path.join(workspace_dir, "active_map.txt")
        if os.path.exists(active_txt):
            with open(active_txt, 'r', encoding='utf-8') as f:
                map_name = f.read().strip()
        if not map_name:
            map_name = "world.xml"

    def _resolve_xml(name):
        candidates = [
            name if os.path.isabs(name) else None,
            os.path.join(workspace_dir, name),
            os.path.join(workspace_dir, "xml", name),
            os.path.join(workspace_dir, "xml", os.path.basename(name))
        ]
        for c in candidates:
            if c and os.path.exists(c):
                return c
        return os.path.join(workspace_dir, "xml", name)

    map_path = _resolve_xml(map_name)
    robot_path = _resolve_xml("slosh3.xml")

    if not os.path.exists(map_path):
        print(f"[CẢNH BÁO] Không tìm thấy file map: {map_path}, sử dụng slosh3.xml mặc định.")
        return mujoco.MjModel.from_xml_path(robot_path), "slosh3.xml"


    # Nếu map là chính slosh3.xml thì tải trực tiếp
    if os.path.basename(map_path).lower() == "slosh3.xml":
        return mujoco.MjModel.from_xml_path(robot_path), "slosh3.xml"

    # Với tất cả các map khác (slosh_rect.XML, world5.xml,...):
    # Đảm bảo LUÔN LUÔN dùng mô hình xe AMR thống nhất từ slosh3.xml
    with open(map_path, 'r', encoding='utf-8') as f:
        map_txt = f.read()

    import xml.etree.ElementTree as ET
    map_root = ET.fromstring(map_txt)

    with open(robot_path, 'r', encoding='utf-8') as f:
        robot_txt = f.read()
    robot_root = ET.fromstring(robot_txt)

    wb_map = map_root.find('worldbody')
    wb_robot = robot_root.find('worldbody')
    cart_canonical = wb_robot.find('.//body[@name="cart"]')

    # Nếu file map đã có sẵn body cart cũ (như trong slosh_rect.XML), xóa bỏ để thay bằng cart từ slosh3.xml
    old_cart = wb_map.find('.//body[@name="cart"]') if wb_map is not None else None
    if old_cart is not None and wb_map is not None:
        wb_map.remove(old_cart)

    if cart_canonical is not None and wb_map is not None:
        wb_map.insert(0, cart_canonical)

    # Thay thế actuator, sensor, equality cũ bằng cấu hình chuẩn từ slosh3.xml
    for tag in ['sensor', 'actuator', 'equality']:
        old_elem = map_root.find(tag)
        if old_elem is not None:
            map_root.remove(old_elem)
        new_elem = robot_root.find(tag)
        if new_elem is not None:
            map_root.append(new_elem)

    combined_xml = ET.tostring(map_root, encoding='utf-8').decode('utf-8')
    model = mujoco.MjModel.from_xml_string(combined_xml)
    return model, os.path.basename(map_path)


def run_amr_closed_loop_test(max_time=30.0, gui=False, save_path=None, no_plot=False, map_name=None, inject_obs=None):
    print("=================================================================")
    print("MÔ PHỎNG ĐÓNG VÒNG & BENCHMARK XE AMR MANG BÌNH NƯỚC")
    print("=================================================================")

    # 1. Nạp mô hình kết hợp tự động (AMR + Map kích hoạt)
    model, active_map_name = load_combined_model(map_name)
    data = mujoco.MjData(model)

    dt_physics = model.opt.timestep # 0.005 s (200 Hz)
    nmpc_dt = 0.05                  # 0.05 s (20 Hz)
    substeps = int(round(nmpc_dt / dt_physics)) # 10 substeps

    print(f">> BẢN ĐỒ ĐANG KÍCH HOẠT: {active_map_name} <<")
    print(f"Mô hình MuJoCo: nv={model.nv}, nu={model.nu}, timestep={dt_physics}s")
    print(f"Chu kỳ NMPC: dt={nmpc_dt}s ({substeps} physics substeps/cycle).")

    # 2. Trích xuất bản đồ & Quy hoạch đường đi toàn cục bằng A*
    start_pos, goal_pos, obs_circles, obs_boxes, map_geoms, dyn_geoms = extract_map_from_model(model)
    print(f"Start: {start_pos} | Goal: {goal_pos} ({len(dyn_geoms)} vật thể động)")

    waypoints = []
    for r_test in [0.26, 0.24, 0.22, 0.20]:
        planner = AStarPlanner(x_bounds=(-2.9, 2.9), y_bounds=(-2.9, 2.9), resolution=0.05, robot_radius=r_test)
        planner.add_obstacles(obs_circles, obs_boxes)
        waypoints = planner.plan(start_pos=start_pos, goal_pos=goal_pos, smooth=True)
        if len(waypoints) > 5:
            print(f"A* đã tìm thấy quỹ đạo gồm {len(waypoints)} điểm (Bán kính robot: {r_test}m, {len(obs_circles)} trụ + {len(obs_boxes)} hộp).")
            break

    if len(waypoints) <= 5:
        raise RuntimeError(f"Không thể tìm thấy đường đi từ Start đến Goal trên bản đồ {active_map_name}!")

    # 2.1 Chèn vật cản sau khi A* sinh xong global path (khi kích hoạt cờ inject_obs hoặc bản đồ world5.xml)
    if inject_obs is None:
        inject_obs = (os.path.basename(active_map_name) == "world5.xml")

    dyn_configs = None
    if inject_obs:
        print("\n>> [POST-A* INJECTION] Đang chèn vật cản tĩnh & động đè lên đường đi toàn cục của A*... <<")
        model, dyn_configs, injected_info = create_injected_simulation_model(active_map_name, waypoints)
        data = mujoco.MjData(model)
        # Cập nhật lại danh sách đối tượng bản đồ để vẽ đồ thị và nhận diện
        start_pos, goal_pos, obs_circles, obs_boxes, map_geoms, dyn_geoms = extract_map_from_model(model)
        print(f">> Đã chèn thành công {len(injected_info)} vật cản lên quỹ đạo: {len(dyn_geoms)} vật thể động.")

    viewer = None
    if gui:
        try:
            viewer = mujoco.viewer.launch_passive(model, data)
            print(">> Đã khởi động giao diện đồ họa 3D (MuJoCo Viewer) <<")
        except Exception as e:
            print(f"Cảnh báo: Không thể khởi chạy 3D Viewer ({e}). Chuyển sang console mode.")
            viewer = None

    # 2.2 Khởi tạo bộ điều khiển MPCC-NMPC cho AMR (bám theo đường A* ban đầu)
    controller = MPCCNMPCAMRController(N=15, dt=nmpc_dt)
    controller.set_path(waypoints)

    # 3. Đặt trạng thái ban đầu
    mujoco.mj_resetData(model, data)
    update_dynamic_obstacles(model, data, 0.0, custom_dyn_configs=dyn_configs)
    data.qpos[0] = start_pos[0]
    data.qpos[1] = start_pos[1]
    p_init, t_init, heading_init = controller.get_path_reference(0.0)
    data.qpos[2] = heading_init
    mujoco.mj_forward(model, data)

    # Hệ số PD hồi tiếp cấp thấp (Low-level controller) cho AMR
    # Trục quay Z có quán tính Iz ~ 0.05 kg.m^2, tinh chỉnh Kp=15.0, Kd=2.0 tránh rung giật số học
    kp_diag = np.array([120.0, 120.0, 15.0])
    kd_diag = np.array([45.0,  45.0,   2.0])
    M_full = np.zeros((model.nv, model.nv))

    # Vector trạng thái nội tại
    a_act = np.zeros(3)
    s_prog = 0.0
    vs_prog = 0.2
    as_prog = 0.0

    # Khởi tạo cảm biến LiDAR 2D và bản đồ Occupancy Grid (cho MPCC)
    lidar = Lidar(model)
    grid = OccupancyGrid()
    scan_pts = np.zeros((0, 2))
    min_range = LIDAR_RANGE

    # Khởi tạo bộ đệm logging
    log = {
        't': [],
        'x': [],
        'y': [],
        'yaw': [],
        'yaw_ref': [],
        'vx': [],
        'vy': [],
        'v_lin': [],
        'vs': [],
        'ax': [],
        'ay': [],
        'phi_x': [],
        'phi_y': [],
        'eta_x_mm': [],
        'eta_y_mm': [],
        'fx': [],
        'fy': [],
        'tau_z': [],
        'solve_ms': [],
        'dyn_obs_pos': [],
        'min_range': [],
        'n_pts': []
    }

    sim_time = 0.0
    max_steps = int(max_time / nmpc_dt)
    goal_reached = False

    print(f"\nBắt đầu mô phỏng đóng vòng ({max_steps} chu kỳ NMPC, T_max = {max_time} s)...")
    t_start_sim = time.time()

    for cycle in range(max_steps):
        if viewer is not None and not viewer.is_running():
            print("\n>> Người dùng đã đóng cửa sổ 3D Viewer.")
            break

        # 1. Cập nhật vị trí vật thể động & Đọc trạng thái đo từ MuJoCo
        update_dynamic_obstacles(model, data, sim_time, custom_dyn_configs=dyn_configs)
        mujoco.mj_forward(model, data)

        q_meas = data.qpos[:3].copy()
        v_meas = data.qvel[:3].copy()
        
        # Con lắc slosh: khớp 3 (slosh_x) và khớp 4 (slosh_y) trong slosh3.xml
        phi_x_meas = float(data.qpos[3])
        dphi_x_meas = float(data.qvel[3])
        phi_y_meas = float(data.qpos[4])
        dphi_y_meas = float(data.qvel[4])
        phi_est = np.array([phi_x_meas, dphi_x_meas, phi_y_meas, dphi_y_meas])

        # Kiểm tra điều kiện về đích
        dist_to_goal = np.hypot(q_meas[0] - goal_pos[0], q_meas[1] - goal_pos[1])
        if dist_to_goal < 0.12 and s_prog > controller.path_length - 0.25:
            print(f"\n>>> ĐÃ ĐẾN ĐÍCH THÀNH CÔNG tại chu kỳ {cycle+1} (t = {sim_time:.2f} s)! Sai số vị trí = {dist_to_goal*100:.1f} cm <<<")
            goal_reached = True
            break

        # 2. Xây dựng vector trạng thái NMPC (12,)
        x_nmpc = np.zeros(12)
        x_nmpc[0:3] = q_meas
        x_nmpc[3:6] = v_meas
        x_nmpc[6:9] = a_act
        x_nmpc[9] = s_prog
        x_nmpc[10] = vs_prog
        x_nmpc[11] = as_prog

        # 3. Quét cảm biến LiDAR 2D và cập nhật bản đồ Occupancy Grid (cho MPCC)
        origin, dirs, meas, hit, raw = lidar.scan(model, data)
        grid.update(origin, dirs, meas, hit)
        scan_pts = origin[:2] + dirs[hit, :2] * meas[hit, None]
        min_range = float(raw.min())
        occ_pts = grid.occupied_points()

        # Nhận diện điểm chướng ngại vật nguy hiểm nhất từ LiDAR đưa vào Obstacle-ECBF của MPCC
        closest_obs, min_d = select_lidar_obstacle(scan_pts, occ_pts, q_meas, v_meas, s_prog, controller)

        # 4. Giải tối ưu NMPC
        u_opt, js_opt, x_pred, u_pred, info = controller.solve_step(x_nmpc, phi_est, closest_obs)

        # 5. Cập nhật tiến độ MPCC
        s_prog = float(x_pred[9, 1])
        vs_prog = float(x_pred[10, 1])
        as_prog = float(x_pred[11, 1])
        a_act = np.array(x_pred[6:9, 1])

        # 6. Chạy các bước mô phỏng vật lý MuJoCo với nội suy bậc 3 liên tục
        q0_cyc = q_meas.copy()
        v0_cyc = v_meas.copy()
        a0_cyc = a_act.copy()
        u0_cyc = u_opt[0:3].copy()

        for sub in range(substeps):
            tau_sub = (sub + 1) * dt_physics
            t_curr = sim_time + dt_physics
            update_dynamic_obstacles(model, data, t_curr, custom_dyn_configs=dyn_configs)

            q_target = q0_cyc + tau_sub * v0_cyc + 0.5 * (tau_sub**2) * a0_cyc + (tau_sub**3 / 6.0) * u0_cyc
            v_target = v0_cyc + tau_sub * a0_cyc + 0.5 * (tau_sub**2) * u0_cyc
            a_target = a0_cyc + tau_sub * u0_cyc

            mujoco.mj_fullM(model, data, M_full)
            F_ff = M_full[:3, :3] @ a_target

            # Bù ma sát Coulomb
            fric_ff = np.zeros(3)
            fric_ff[0:2] = 0.5 * np.tanh(v_target[0:2] / 0.02)
            fric_ff[2] = 0.2 * np.tanh(v_target[2] / 0.02)

            q_now = data.qpos[:3]
            v_now = data.qvel[:3]
            F_bias = data.qfrc_bias[:3].copy()

            err_q = q_target - q_now
            # Triệt tiêu bước nhảy pha 2*pi cho trục xoay Z, tránh xung lực mô-men gây hiện tượng tự xoay tròn
            err_q[2] = np.arctan2(np.sin(err_q[2]), np.cos(err_q[2]))
            err_v = v_target - v_now

            F_cmd = F_ff + fric_ff + kp_diag * err_q + kd_diag * err_v + F_bias

            # Giới hạn lực động cơ bệ xe
            F_cmd[0:2] = np.clip(F_cmd[0:2], -30.0, 30.0)
            F_cmd[2] = np.clip(F_cmd[2], -10.0, 10.0)

            data.ctrl[:3] = F_cmd
            mujoco.mj_step(model, data)
            sim_time += dt_physics

        if viewer is not None:
            viewer.user_scn.ngeom = 0
            viewer.cam.lookat[:] = [q_meas[0] * 0.5, q_meas[1] * 0.5, 0.1]

            # Điểm đường toàn cục A* (màu vàng nhạt)
            if len(waypoints) > 0:
                for pt in waypoints[::2]:
                    draw_sphere(viewer, np.array([pt[0], pt[1], 0.02]), [1.0, 1.0, 0.0, 0.6], size=0.025)

            # Điểm đích (màu xanh lá)
            draw_sphere(viewer, np.array([goal_pos[0], goal_pos[1], 0.05]), [0.0, 0.9, 0.2, 0.8], size=0.07)

            # Quỹ đạo dự đoán MPCC (màu xanh lục sáng)
            for i in range(controller.N):
                draw_sphere(viewer, np.array([x_pred[0, i], x_pred[1, i], 0.05]), [0.0, 1.0, 0.0, 0.8], size=0.02)

            # Tia LiDAR quét phản xạ (màu cyan)
            for p in scan_pts:
                draw_sphere(viewer, np.array([p[0], p[1], 0.35]), [0.0, 1.0, 1.0, 0.8], size=0.015)

            # Điểm chướng ngại vật nguy hiểm nhất đưa vào Obstacle-ECBF (màu vàng)
            if min_d < 50.0:
                draw_sphere(viewer, np.array([closest_obs[0], closest_obs[1], 0.40]), [1.0, 0.8, 0.0, 1.0], size=0.04)

            # Hiển thị bộ đếm giây trực quan trên màn hình 3D MuJoCo
            viewer.set_texts([
                (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, "Time:", f"{sim_time:5.2f} s")
            ])
            viewer.sync()
            time.sleep(0.002)

        # Logging dữ liệu
        _, _, heading_ref = controller.get_path_reference(s_prog)
        eta_x = 0.5 * TANK_A * np.tan(phi_x_meas) * 1000.0
        eta_y = 0.5 * TANK_B * np.tan(phi_y_meas) * 1000.0
        v_lin = float(np.hypot(v_meas[0], v_meas[1]))

        log['t'].append(sim_time)
        log['x'].append(q_meas[0])
        log['y'].append(q_meas[1])
        log['yaw'].append(q_meas[2])
        log['yaw_ref'].append(heading_ref)
        log['vx'].append(v_meas[0])
        log['vy'].append(v_meas[1])
        log['v_lin'].append(v_lin)
        log['vs'].append(vs_prog)
        log['ax'].append(a_act[0])
        log['ay'].append(a_act[1])
        log['phi_x'].append(phi_x_meas)
        log['phi_y'].append(phi_y_meas)
        log['eta_x_mm'].append(eta_x)
        log['eta_y_mm'].append(eta_y)
        log['fx'].append(F_cmd[0])
        log['fy'].append(F_cmd[1])
        log['tau_z'].append(F_cmd[2])
        log['solve_ms'].append(info['solve_time_ms'])
        log['dyn_obs_pos'].append([data.geom_xpos[g_id][:2].copy() for g_id, _, _, _, _ in dyn_geoms])
        log['min_range'].append(min_range)
        log['n_pts'].append(len(scan_pts))

        if (cycle + 1) % 10 == 0 or cycle == 0:
            print(f"Cycle {cycle+1:3d} (t={sim_time:.2f}s): Pos=[{q_meas[0]:.2f}, {q_meas[1]:.2f}] | s={s_prog:.2f}/{controller.path_length:.2f}m | vs={vs_prog:.2f}m/s | Slosh=[{np.degrees(phi_x_meas):.1f}°, {np.degrees(phi_y_meas):.1f}°] | LiDAR_min={min_range:.2f}m | Solve={info['solve_time_ms']:.1f}ms")

    if viewer is not None:
        viewer.sync()
        time.sleep(1.0)
        viewer.close()

    # Chuyển đổi dữ liệu log sang numpy
    for k in log:
        log[k] = np.array(log[k])

    # 4. TỔNG KẾT BENCHMARK
    print("\n=================================================================")
    print("KẾT QUẢ BENCHMARK TOÀN DIỆN - AMR WITH FIXED CONTAINER (slosh3.xml)")
    print("=================================================================")
    t_arr = log['t']
    phi_x_deg = np.rad2deg(log['phi_x']) if len(t_arr) > 0 else np.array([0.0])
    phi_y_deg = np.rad2deg(log['phi_y']) if len(t_arr) > 0 else np.array([0.0])
    eta_x_mm = log['eta_x_mm'] if len(t_arr) > 0 else np.array([0.0])
    eta_y_mm = log['eta_y_mm'] if len(t_arr) > 0 else np.array([0.0])
    solve_ms = log['solve_ms'] if len(t_arr) > 0 else np.array([0.0])

    max_phi_x = np.max(np.abs(phi_x_deg))
    max_phi_y = np.max(np.abs(phi_y_deg))
    rms_phi_x = np.sqrt(np.mean(phi_x_deg**2))
    rms_phi_y = np.sqrt(np.mean(phi_y_deg**2))

    max_eta_x = np.max(np.abs(eta_x_mm))
    max_eta_y = np.max(np.abs(eta_y_mm))
    freeboard_mm = (TANK_H - H_LIQUID) * 1000.0 # 30 mm (110 mm - 80 mm)

    avg_solve = np.mean(solve_ms)
    max_solve = np.max(solve_ms)
    p99_solve = np.percentile(solve_ms, 99) if len(solve_ms) > 0 else 0.0

    print(f"• Trạng thái hoàn thành: {'VỀ ĐÍCH THÀNH CÔNG' if goal_reached else 'CHƯA CHẠM ĐÍCH'}")
    print(f"• Tổng thời gian di chuyển: {sim_time:.2f} s")
    print(f"• Vận tốc trung bình: {np.mean(log['v_lin']):.2f} m/s | Cực đại: {np.max(log['v_lin']):.2f} m/s")
    print(f"• Góc sóng sánh cực đại: Phi_X = {max_phi_x:.2f}° (Limit ±{np.degrees(PHI_MAX_X):.1f}°), Phi_Y = {max_phi_y:.2f}° (Limit ±{np.degrees(PHI_MAX_Y):.1f}°)")
    print(f"• Góc sóng sánh RMS: Phi_X_rms = {rms_phi_x:.2f}°, Phi_Y_rms = {rms_phi_y:.2f}°")
    print(f"• Sóng dâng cực đại tại vách: Eta_X = {max_eta_x:.1f} mm, Eta_Y = {max_eta_y:.1f} mm (Mép an toàn: {freeboard_mm:.0f} mm)")
    print(f"• Tình trạng tràn chất lỏng: {'AN TOÀN TUYỆT ĐỐI (KHÔNG TRÀN)' if (max_eta_x < freeboard_mm and max_eta_y < freeboard_mm) else 'CẢNH BÁO TRÀN NƯỚC'}")
    min_lidar = np.min(log['min_range']) if len(log['min_range']) > 0 else 0.0
    mapped_cells = int((grid.L > L_THRESH).sum())
    print(f"• Cự ly quét LiDAR nhỏ nhất: {min_lidar:.2f} m")
    print(f"• Số ô lưới Occupancy Grid lập bản đồ: {mapped_cells} ô ({mapped_cells * (GRID_RES**2):.2f} m²)")
    print(f"• Thời gian giải NMPC: Trung bình = {avg_solve:.2f} ms | P99 = {p99_solve:.2f} ms | Cực đại = {max_solve:.2f} ms")
    print(f"• Tần số điều khiển thực tế tương đương: {1000.0 / avg_solve:.1f} Hz")
    print("=================================================================")

    # 5. Vẽ đồ thị kết quả
    workspace_dir = _root
    bm_folder = os.path.join(workspace_dir, "testbm")
    os.makedirs(bm_folder, exist_ok=True)

    if save_path:
        if not os.path.isabs(save_path):
            save_path = os.path.join(bm_folder, os.path.basename(save_path))
    else:
        map_tag = os.path.splitext(os.path.basename(active_map_name))[0]
        save_path = os.path.join(bm_folder, f"benchmark_amr_{map_tag}.png")

    if not no_plot:
        import matplotlib.pyplot as plt
        print("\n>> Đang tạo và hiển thị cửa sổ đồ thị kết quả (Matplotlib)...")
        fig = plot_amr_results(log, waypoints, start_pos, goal_pos, map_geoms, dyn_geoms=dyn_geoms, grid=grid, active_map_name=active_map_name, save_path=save_path)
        plt.show()
    elif save_path:
        plot_amr_results(log, waypoints, start_pos, goal_pos, map_geoms, dyn_geoms=dyn_geoms, grid=grid, active_map_name=active_map_name, save_path=save_path)


def plot_amr_results(log, waypoints, start_pos, goal_pos, map_geoms, dyn_geoms=None, grid=None, active_map_name="Arena", save_path=None):
    """
    Vẽ 6 đồ thị kỹ thuật benchmark chuyên sâu cho xe AMR mang bình nước.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    try:
        fig.canvas.manager.set_window_title(f"Benchmark NMPC MPCC - AMR Chứa Bình Nước ({active_map_name})")
    except Exception:
        pass

    ax1, ax2, ax3 = axes[0, 0], axes[0, 1], axes[0, 2]
    ax4, ax5, ax6 = axes[1, 0], axes[1, 1], axes[1, 2]

    t = log['t']
    total_time = float(t[-1]) if len(t) > 0 else 0.0

    fig.suptitle(f"BENCHMARK MPCC-NMPC: AMR MANG BÌNH CHẤT LỎNG (slosh3.xml)\nBản đồ: {active_map_name}   |   TỔNG THỜI GIAN DI CHUYỂN: {total_time:.2f} s",
                 fontsize=14, fontweight='bold', y=0.985)

    # 1. Bản đồ Occupancy Grid (LiDAR log-odds) & Quỹ đạo AMR (Bỏ nhãn ký hiệu vật cản khỏi legend)
    if grid is not None:
        ax1.imshow(np.clip(grid.L.T, L_MIN, L_MAX), origin="lower", cmap="gray_r", vmin=-2, vmax=3,
                   extent=[-GRID_HALF, GRID_HALF, -GRID_HALF, GRID_HALF], alpha=0.35)
    for gtype, pos, size, rgba, name in map_geoms:
        if gtype == 'box':
            ax1.add_patch(plt.Rectangle((pos[0] - size[0], pos[1] - size[1]), 2 * size[0], 2 * size[1],
                                        color=rgba[:3], alpha=0.85))
        elif gtype == 'cylinder':
            ax1.add_patch(plt.Circle((pos[0], pos[1]), size,
                                     color=rgba[:3], alpha=0.75))

    # Vẽ vật thể động và quỹ đạo chuyển động cắt ngang (không gán label vào legend)
    if dyn_geoms and len(dyn_geoms) > 0 and 'dyn_obs_pos' in log and len(log['dyn_obs_pos']) > 0:
        dyn_pts = np.array(log['dyn_obs_pos'])  # shape (T, N_dyn, 2)
        dyn_colors = ['magenta', 'cyan', 'gold']
        for j, (g_id, r_dyn, gtype, rgba, b_name) in enumerate(dyn_geoms):
            c = dyn_colors[j % len(dyn_colors)]
            if j < dyn_pts.shape[1]:
                ax1.plot(dyn_pts[:, j, 0], dyn_pts[:, j, 1], color=c, ls='--', lw=2.0, alpha=0.85)
                mid_idx = len(dyn_pts) // 2
                cur_x, cur_y = dyn_pts[mid_idx, j, 0], dyn_pts[mid_idx, j, 1]
                if gtype == mujoco.mjtGeom.mjGEOM_BOX:
                    ax1.add_patch(plt.Rectangle((cur_x - r_dyn, cur_y - r_dyn), 2 * r_dyn, 2 * r_dyn,
                                                color=c, alpha=0.6))
                else:
                    ax1.add_patch(plt.Circle((cur_x, cur_y), r_dyn, color=c, alpha=0.6))

    if waypoints is not None and len(waypoints) > 0:
        ax1.plot(waypoints[:, 0], waypoints[:, 1], 'y--', lw=2.0, label='A* Global Path')

    ax1.plot(log['x'], log['y'], 'b-', lw=2.2, label='AMR Path (MPCC-NMPC)')
    stride = max(1, len(log['x']) // 25)
    ax1.quiver(log['x'][::stride], log['y'][::stride],
               np.cos(log['yaw'][::stride]), np.sin(log['yaw'][::stride]),
               angles='xy', scale_units='xy', scale=7, color='darkorange', width=0.005, alpha=0.85, label='Heading')

    ax1.plot(start_pos[0], start_pos[1], 'mo', markersize=9, label='Start')
    ax1.plot(goal_pos[0], goal_pos[1], 'r*', markersize=14, label='Goal')
    ax1.set_xlim([-3.2, 3.2])
    ax1.set_ylim([-3.2, 3.2])
    ax1.set_aspect('equal')
    ax1.set_title(f'AMR Trajectory & Map ({active_map_name})\nTổng thời gian chạy: {total_time:.2f} s', fontsize=11, fontweight='bold')
    ax1.text(0.03, 0.94, f"Tổng thời gian: {total_time:.2f} s", transform=ax1.transAxes, fontsize=10, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.4", fc="#FFF2B2", ec="#FF8C00", lw=1.5, alpha=0.95))
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.legend(loc='upper right', fontsize=7)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # 2. Góc sóng sánh chất lỏng 2 trục
    phi_x_deg = np.rad2deg(log['phi_x'])
    phi_y_deg = np.rad2deg(log['phi_y'])
    lim_x_deg = np.degrees(PHI_MAX_X)
    lim_y_deg = np.degrees(PHI_MAX_Y)

    ax2.plot(t, phi_x_deg, 'c-', lw=2.0, label=r'Slosh $\phi_x$ (Longitudinal)')
    ax2.plot(t, phi_y_deg, 'm-', lw=2.0, label=r'Slosh $\phi_y$ (Transverse)')
    ax2.axhline(lim_x_deg, color='cyan', ls='--', lw=1.5, label=f'Limit X (±{lim_x_deg:.1f}°)')
    ax2.axhline(-lim_x_deg, color='cyan', ls='--', lw=1.5)
    ax2.axhline(lim_y_deg, color='magenta', ls='--', lw=1.5, label=f'Limit Y (±{lim_y_deg:.1f}°)')
    ax2.axhline(-lim_y_deg, color='magenta', ls='--', lw=1.5)
    ax2.set_title('Housner Slosh Angles (ECBF r=3 Constrained)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Slosh Angle (deg)')
    ax2.legend(loc='upper right', fontsize=7)
    ax2.grid(True, linestyle=':', alpha=0.6)

    # 3. Chiều cao sóng dâng tại vách bình & Ngưỡng tràn nước
    eta_max_x_mm = 0.5 * TANK_A * np.tan(PHI_MAX_X) * 1000.0
    eta_max_y_mm = 0.5 * TANK_B * np.tan(PHI_MAX_Y) * 1000.0
    freeboard_mm = (TANK_H - H_LIQUID) * 1000.0

    ax3.plot(t, log['eta_x_mm'], 'c-', lw=2.0, label=f'Wave X (Wall {TANK_A*1000:.0f}mm)')
    ax3.plot(t, log['eta_y_mm'], 'm-', lw=2.0, label=f'Wave Y (Wall {TANK_B*1000:.0f}mm)')
    ax3.axhline(eta_max_x_mm, color='cyan', ls='--', lw=1.4, label=f'Safe Limit X (±{eta_max_x_mm:.1f} mm)')
    ax3.axhline(-eta_max_x_mm, color='cyan', ls='--', lw=1.4)
    ax3.axhline(eta_max_y_mm, color='magenta', ls='--', lw=1.4, label=f'Safe Limit Y (±{eta_max_y_mm:.1f} mm)')
    ax3.axhline(-eta_max_y_mm, color='magenta', ls='--', lw=1.4)
    ax3.axhline(freeboard_mm, color='red', ls='-.', lw=1.6, label=f'Spillover Rim ({freeboard_mm:.0f} mm)')
    ax3.set_title('Wave Elevation at Tank Boundary Walls', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel(r'Wave Elevation $\eta$ (mm)')
    ax3.legend(loc='upper right', fontsize=7)
    ax3.grid(True, linestyle=':', alpha=0.6)

    # 4. Lực tịnh tiến bệ xe Fx, Fy & Mô-men xoay tau_z
    ax4.plot(t, log['fx'], 'c-', lw=1.8, label='Force Fx')
    ax4.plot(t, log['fy'], 'm-', lw=1.8, label='Force Fy')
    ax4.axhline(30.0, color='gray', ls='--', lw=1.0, label='Motor Force Limit (±30 N)')
    ax4.axhline(-30.0, color='gray', ls='--', lw=1.0)
    ax4.set_title('AMR Actuator Forces & Torque', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Force (N)')
    ax4.set_ylim([-35.0, 35.0])
    ax4.legend(loc='upper left', fontsize=7)
    ax4.grid(True, linestyle=':', alpha=0.6)

    ax4_twin = ax4.twinx()
    ax4_twin.plot(t, log['tau_z'], color='purple', ls='-.', lw=1.5, label=r'$\tau_z$')
    ax4_twin.set_ylabel(r'Yaw Torque $\tau_z$ (N.m)', color='purple')
    ax4_twin.tick_params(axis='y', labelcolor='purple')

    # 5. Vận tốc và Gia tốc bệ xe
    ax5.plot(t, log['v_lin'], 'b-', lw=2.0, label='Linear Velocity |v| (m/s)')
    ax5.plot(t, log['vs'], 'g--', lw=1.8, label='MPCC Progress vs (m/s)')
    ax5.set_title('AMR Velocities & MPCC Dynamic Speed Adaptation', fontsize=11, fontweight='bold')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Velocity (m/s)')
    ax5.legend(loc='upper left', fontsize=7)
    ax5.grid(True, linestyle=':', alpha=0.6)

    ax5_twin = ax5.twinx()
    a_net = np.hypot(log['ax'], log['ay'])
    ax5_twin.plot(t, a_net, color='crimson', ls=':', lw=1.4, label='Total Accel |a| (m/s²)')
    ax5_twin.set_ylabel('Acceleration (m/s²)', color='crimson')
    ax5_twin.tick_params(axis='y', labelcolor='crimson')

    # 6. Bám góc hướng và Thời gian giải NMPC
    yaw_deg = np.rad2deg(log['yaw'])
    yaw_ref_deg = np.rad2deg(log['yaw_ref'])
    ax6.plot(t, yaw_ref_deg, 'g--', lw=1.8, label=r'Target Heading $\psi_{ref}$')
    ax6.plot(t, yaw_deg, color='darkorange', lw=2.0, label=r'Actual Heading $\psi$')
    ax6.set_title('Heading Tracking & NMPC Solve Time', fontsize=11, fontweight='bold')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Heading (deg)')
    ax6.legend(loc='upper left', fontsize=7)
    ax6.grid(True, linestyle=':', alpha=0.6)

    ax6_twin = ax6.twinx()
    ax6_twin.plot(t, log['solve_ms'], color='teal', ls=':', lw=1.4, alpha=0.85, label='Solve Time (ms)')
    ax6_twin.axhline(50.0, color='red', ls='--', lw=1.0, alpha=0.7, label='Real-time Bound (50 ms)')
    ax6_twin.set_ylabel('Solve Time (ms)', color='teal')
    ax6_twin.tick_params(axis='y', labelcolor='teal')

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f">> Đã lưu thành công biểu đồ đồ thị kỹ thuật tại: {save_path}")
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiểm thử mô phỏng đóng vòng xe AMR mang bình nước")
    parser.add_argument("--max-time", type=float, default=25.0, help="Thời gian mô phỏng tối đa (giây)")
    parser.add_argument("--gui", action="store_true", help="Bật cửa sổ hiển thị đồ họa 3D MuJoCo Viewer")
    parser.add_argument("--no-plot", action="store_true", help="Không mở cửa sổ Matplotlib")
    parser.add_argument("--map", type=str, default=None, help="Tên file bản đồ (ví dụ world.xml, world2.xml, world5.xml)")
    parser.add_argument("--save-path", type=str, default=None, help="Đường dẫn lưu hình ảnh biểu đồ")
    parser.add_argument("--inject-obs", action="store_true", default=None, help="Chèn vật cản tĩnh & động đè lên đường đi A* sau khi sinh global path")
    parser.add_argument("--no-inject", action="store_true", help="Không chèn vật cản (kể cả trên world5.xml)")
    args = parser.parse_args()

    inject_flag = False if args.no_inject else (True if args.inject_obs else None)
    run_amr_closed_loop_test(max_time=args.max_time, gui=args.gui, save_path=args.save_path, no_plot=args.no_plot, map_name=args.map, inject_obs=inject_flag)
