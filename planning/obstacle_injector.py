"""
obstacle_injector.py
=============================================================================
Module Chèn Vật Cản Động & Tĩnh Sau Khi A* Hoạch Định Quỹ Đạo Toàn Cục
(Post-A* Obstacle Injection Mechanism for MPCC & NMPC Testing)

Quy trình:
1. A* quy hoạch đường đi trên bản đồ vách tường trống (world5.xml).
2. Sau khi có danh sách waypoints toàn cục, module này tính toán tọa độ chèn:
   - Vật cản tĩnh: Đè lên hoặc áp sát tim đường A*.
   - Vật cản động: Chuyển động tuần hoàn cắt ngang qua tim đường A*.
3. Tự động sinh XML MuJoCo hoàn chỉnh đã được chèn vật cản và nạp vào mô phỏng.
4. Cảm biến LiDAR 2D sẽ phát hiện vật cản thời gian thực và kích hoạt bộ điều
   khiển cục bộ (MPCC / NMPC) để né tránh an toàn.
=============================================================================
"""

import os
import sys
_curr = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_curr) if os.path.basename(_curr) in ["simulation", "controller", "solver", "planning", "cfd", "bao cao"] else _curr
for _sub in ["", "controller", "solver", "planning", "simulation", "cfd", "bao cao"]:
    _p = os.path.join(_root, _sub) if _sub else _root
    if _p not in sys.path:
        sys.path.insert(0, _p)
import xml.etree.ElementTree as ET
import numpy as np
import mujoco


def get_combined_xml_string(map_name="world5.xml"):
    """
    Ghép file bản đồ với xe AMR (slosh3.xml) thành chuỗi XML hợp nhất.
    """
    workspace_dir = _root
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

    with open(map_path, 'r', encoding='utf-8') as f:
        map_txt = f.read()
    with open(robot_path, 'r', encoding='utf-8') as f:
        robot_txt = f.read()

    map_root = ET.fromstring(map_txt)
    robot_root = ET.fromstring(robot_txt)

    wb_map = map_root.find('worldbody')
    wb_robot = robot_root.find('worldbody')
    cart = wb_robot.find('.//body[@name="cart"]')
    if cart is not None and wb_map is not None:
        wb_map.insert(0, cart)

    for tag in ['sensor', 'actuator', 'equality']:
        elem = robot_root.find(tag)
        if elem is not None:
            map_root.append(elem)

    return ET.tostring(map_root, encoding='utf-8').decode('utf-8')


def inject_obstacles_into_xml(base_xml_str, waypoints, configs=None):
    """
    Chèn vật cản tĩnh và động trực tiếp vào XML MuJoCo dựa trên waypoints của A*.

    Parameters:
        base_xml_str: Chuỗi XML MuJoCo gốc (đã ghép map và robot)
        waypoints: Mảng (N, 2) tọa độ đường đi toàn cục do A* sinh ra
        configs: Danh sách dict cấu hình vật cản chèn vào. Nếu None, dùng cấu hình mẫu.

    Returns:
        injected_xml: Chuỗi XML mới đã bổ sung các geom/body vật cản
        dyn_configs: Dict chứa thông số điều khiển khớp động cho các vật thể động
        injected_info: Danh sách thông tin vật cản phục vụ logging và vẽ đồ thị
    """
    root = ET.fromstring(base_xml_str)
    wb = root.find('worldbody')
    if wb is None:
        raise ValueError("Không tìm thấy thẻ 'worldbody' trong XML!")

    n_pts = len(waypoints)
    # Cấu hình mặc định nếu không truyền vào
    # Cấu hình mặc định nếu không truyền vào:
    # 1. Vật thể tĩnh 1: Trụ đỏ tại 22% quãng đường (Hành lang Tây), đặt sát mép đường (offset 0.35m) thu hẹp hành lang di chuyển.
    # 2. Vật thể động 2: Trụ tím trượt ngang trục X tại 78% quãng đường (Hành lang Đông), cắt ngang qua tim đường.
    if configs is None:
        configs = [
            {
                "type": "static_cylinder",
                "ratio": 0.22,
                "offset_lateral": 0.35,
                "radius": 0.14,
                "rgba": "0.90 0.20 0.20 1",
                "name": "injected_static_obs_1"
            },
            {
                "type": "static_cylinder",       # "static_cylinder" hoặc "static_box"
                "ratio": 0.6,                   # Vị trí theo % chiều dài đường đi A* (0.0 = Start, 1.0 = Goal)
                "offset_lateral": 0.3,           # 0.0 = ĐÈ CHÍNH GIỮA tim đường; >0 = Lệch trái; <0 = Lệch phải (m)
                "radius": 0.20,                  # Bán kính hình trụ (m)
                "rgba": "0.90 0.20 0.20 1",       # Màu sắc (R G B Alpha)
                "name": "my_static_obs"          # Tên geom
            },
            {
                "type": "dynamic_cylinder",
                "center_override": [2.0, 0.5],    # Trung tâm hành lang Đông (x in [1.0, 3.0]), cách đều 2 vách tường 1m
                "radius": 0.20,
                "amp": 0.45,                     # Dao động +-0.35m (x in [1.65, 2.35]) cắt ngang tim đường A*
                "omega": 0.95,                   # Tần số góc vừa phải (rad/s)
                "phase": 0.0,
                "slide_axis": "1 0 0",
                "rgba": "0.95 0.15 0.75 1",
                "name": "injected_dyn_obs_1"
            }
        ]

    dyn_configs = {}
    injected_info = []
    

    for idx, cfg in enumerate(configs):
        if "center_override" in cfg:
            pos_xy = np.array(cfg["center_override"], dtype=float)
        else:
            r_ratio = cfg.get("ratio", 0.5)
            pt_idx = int(np.clip(r_ratio * (n_pts - 1), 0, n_pts - 1))
            center_xy = waypoints[pt_idx].copy()

            # Tính tiếp tuyến và pháp tuyến đường đi để xác định hướng lệch tâm
            if pt_idx < n_pts - 1:
                tan_vec = waypoints[pt_idx + 1] - waypoints[pt_idx]
            else:
                tan_vec = waypoints[pt_idx] - waypoints[pt_idx - 1]
            tan_norm = np.hypot(tan_vec[0], tan_vec[1]) + 1e-6
            normal_vec = np.array([-tan_vec[1], tan_vec[0]]) / tan_norm

            offset_lat = cfg.get("offset_lateral", 0.0)
            pos_xy = center_xy + normal_vec * offset_lat
        z_pos = 0.50

        c_type = cfg.get("type", "static_cylinder")
        obs_name = cfg.get("name", f"injected_obs_{idx+1}")
        rgba = cfg.get("rgba", "0.85 0.2 0.2 1")

        if c_type == "static_cylinder":
            radius = cfg.get("radius", 0.22)
            geom_elem = ET.SubElement(wb, 'geom', {
                'class': 'obs',
                'name': obs_name,
                'type': 'cylinder',
                'size': f"{radius:.3f} 0.50",
                'pos': f"{pos_xy[0]:.3f} {pos_xy[1]:.3f} {z_pos:.3f}",
                'rgba': rgba
            })
            injected_info.append(('cylinder', pos_xy, radius, obs_name, False))

        elif c_type == "static_box":
            size_xy = cfg.get("size_xy", (0.18, 0.18))
            geom_elem = ET.SubElement(wb, 'geom', {
                'class': 'obs',
                'name': obs_name,
                'type': 'box',
                'size': f"{size_xy[0]:.3f} {size_xy[1]:.3f} 0.50",
                'pos': f"{pos_xy[0]:.3f} {pos_xy[1]:.3f} {z_pos:.3f}",
                'rgba': rgba
            })
            injected_info.append(('box', pos_xy, np.array(size_xy), obs_name, False))

        elif c_type in ["dynamic_cylinder", "dynamic_box"]:
            radius = cfg.get("radius", 0.22)
            joint_name = f"dyn_joint_inj_{idx+1}"
            slide_axis = cfg.get("slide_axis", "1 0 0")

            body_elem = ET.SubElement(wb, 'body', {
                'name': obs_name,
                'pos': f"{pos_xy[0]:.3f} {pos_xy[1]:.3f} {z_pos:.3f}"
            })
            ET.SubElement(body_elem, 'joint', {
                'name': joint_name,
                'type': 'slide',
                'axis': slide_axis,
                'damping': '1.0'
            })
            g_type = 'cylinder' if c_type == "dynamic_cylinder" else 'box'
            ET.SubElement(body_elem, 'geom', {
                'class': 'dyn_obs',
                'name': f"geom_{obs_name}",
                'type': g_type,
                'size': f"{radius:.3f} 0.48",
                'rgba': rgba
            })

            dyn_configs[joint_name] = {
                "amp": cfg.get("amp", 0.50),
                "omega": cfg.get("omega", 0.95),
                "phase": cfg.get("phase", 0.0),
                "axis": [float(x) for x in slide_axis.split()]
            }
            injected_info.append((g_type, pos_xy, radius, obs_name, True))

    injected_xml = ET.tostring(root, encoding='utf-8').decode('utf-8')
    return injected_xml, dyn_configs, injected_info


def create_injected_simulation_model(map_name, waypoints, configs=None):
    """
    Sinh MjModel hoàn chỉnh sau khi đã chèn vật cản đè lên đường A*.
    """
    base_xml = get_combined_xml_string(map_name)
    injected_xml, dyn_configs, injected_info = inject_obstacles_into_xml(base_xml, waypoints, configs)
    model = mujoco.MjModel.from_xml_string(injected_xml)
    return model, dyn_configs, injected_info
