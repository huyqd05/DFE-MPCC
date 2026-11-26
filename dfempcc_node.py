import os
import sys
import time
import numpy as np

# Nạp mã nguồn dự án (ưu tiên thư mục chứa script, fallback về /mnt/c/Users/LENOVO/Desktop/mujoco trên WSL)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(_SCRIPT_DIR, "controller", "dfe_mpcc_amr_controller.py")):
    PROJECT_DIR = _SCRIPT_DIR
else:
    PROJECT_DIR = "/mnt/c/Users/LENOVO/Desktop/mujoco"

for _sub in ["", "controller", "solver", "planning", "cfd", "simulation"]:
    _p = os.path.join(PROJECT_DIR, _sub) if _sub else PROJECT_DIR
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist, Vector3Stamped
    from nav_msgs.msg import Odometry, Path
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import Float64MultiArray
    HAS_RCLPY = True
except ImportError:
    HAS_RCLPY = False
    rclpy = None
    Node = object
    Twist = object
    Vector3Stamped = object
    Odometry = object
    Path = object
    LaserScan = object
    Float64MultiArray = object

from astar import AStarPlanner
from conformal_checkpoint_eval import load_conformal_calibration_budget
from dfe_mpcc_amr_controller import DFEMPCCAMRController
from friction_observer_ffrls import FrictionObserverFFRLS, OnlineFrictionFusion
from mecanum_nullspace_allocation import allocate_mecanum_wheel_forces_guaranteed
from mecanum_traction_dynamics import (
    L_P_X,
    L_P_Y,
    M_1X,
    M_1Y,
    M_TOTAL,
    Z_COM,
    Z_SLOSH_X,
    Z_SLOSH_Y,
    compute_dynamic_normal_loads,
    evaluate_traction_and_slip,
    scale_wrench_to_zonotope,
)
from robot_model_amr_casadi import IZ_BASE, L_X, L_Y, hd_x, hd_y, omega_x, omega_y
from semantic_friction_estimator import (
    SemanticFloorMap,
    SemanticFrictionEstimator,
    make_world_semantic_map,
)
from velocity_tube_h124 import CoupledVelocityTubeH124


def build_default_world_waypoints():
    """Hoạch định đường đi mẫu ban đầu cho world.xml bằng A*."""
    planner = AStarPlanner(
        x_bounds=(-3.0, 3.0), y_bounds=(-3.0, 3.0), resolution=0.05, robot_radius=0.25
    )
    obs_boxes = [
        [-1.0, -1.075, 0.05, 1.875],
        [1.0, 1.075, 0.05, 1.875],
        [0.60, 0.20, 0.35, 0.05],
        [2.45, -0.35, 0.45, 0.05],
    ]
    obs_circles = [
        [-2.30, -0.70, 0.28],
        [-1.50, 1.80, 0.32],
        [0.25, 2.05, 0.26],
        [-0.40, -1.90, 0.28],
        [2.40, 0.85, 0.28],
    ]
    planner.add_obstacles(obs_circles, obs_boxes)
    return planner.plan([-2.2, -2.2], [2.2, 2.2], smooth=True)


class DfeMpccPipeline:
    """
    Pipeline điều khiển DFE-MPCC hoàn chỉnh (độc lập ROS 2 để dễ kiểm thử & nhúng):
      - Tầng 0/A: Ống vận tốc khả thi bậc 3 Krasnoselskii-Mann (CoupledVelocityTubeH124)
      - Tầng 1:   Tiên nghiệm ma sát ngữ nghĩa Zero-Shot + Split Conformal (SemanticFrictionEstimator)
      - Tầng 2:   Adaptive FF-RLS + OnlineFrictionFusion (cùng phân bổ lực bánh không gian null O(1))
      - Tầng MPC: DFEMPCCAMRController với q_eta Conformal và ngân sách thực thi 20 Hz (mặc định N=8, max_iter=12)
    """

    def __init__(
        self,
        horizon_N: int = 8,
        dt: float = 0.05,
        ipopt_max_iter: int = 12,
        mu_default: float = 0.35,
        map_name: str = "generic_dry",
        x_bounds: tuple = (-3.6, 3.6),
        y_bounds: tuple = (-3.6, 3.6),
        waypoints=None,
    ):
        self.dt = float(dt)
        self.mu_default = float(mu_default)
        self.map_name = str(map_name)

        cal_budget = load_conformal_calibration_budget()
        self.q_eta_m = float(cal_budget.get("recommended_q_eta_m", 0.004226))

        cal_mu_path = os.path.join(PROJECT_DIR, "config", "conformal_calibration_mu_semantic.json")
        key = os.path.basename(self.map_name).lower()
        if any(k in key for k in ("world6", "world7", "world8", "world9", "map9")):
            sem_map = make_world_semantic_map(self.map_name)
        else:
            sem_map = SemanticFloorMap(zones=[], default_label="dry_concrete", name=f"semantic_{key}")

        self.sem_estimator = SemanticFrictionEstimator.from_calibration_or_default(
            cal_mu_path if os.path.exists(cal_mu_path) else None,
            semantic_map=sem_map,
        )

        self.controller = DFEMPCCAMRController(
            N=int(horizon_N),
            dt=self.dt,
            mu=self.mu_default,
            state_con_mode="skip0",
            ipopt_max_iter=int(ipopt_max_iter),
            x_bounds=x_bounds,
            y_bounds=y_bounds,
        )
        self.tube_planner = CoupledVelocityTubeH124(mu_default=self.mu_default, kappa_th=1.05)

        self.friction_observer = FrictionObserverFFRLS(
            mu_init=0.22, lam=None, adaptive_forgetting=True, rho_deadzone=0.06
        )
        self.friction_fusion = OnlineFrictionFusion(mu_min=0.08, mu_max=0.55)
        self.mu_rls_curr = 0.22
        self.bin_curr = 0

        self.s_prog = 0.0
        self.vs_prog = 0.2
        self.as_prog = 0.0
        self.a_act = np.zeros(3, dtype=float)
        self._prev_v_meas = None
        self._prev_a_cmd_body = np.zeros(2, dtype=float)

        init_wps = build_default_world_waypoints() if waypoints is None else waypoints
        self.set_waypoints(init_wps)

    def set_waypoints(self, waypoints) -> float:
        """Nạp quỹ đạo mới, tái dựng Ống vận tốc H124 + bản đồ ma sát Tầng 1 và đặt lại tiến độ."""
        wp_arr = np.asarray(waypoints, dtype=float)
        self.controller.set_path(wp_arr)
        self.tube_planner.build_offline_tube(
            wp_arr,
            friction_estimator=self.sem_estimator,
            friction_mode="conformal_lcb",
            run_wcsph_check=False,
        )
        self.controller.set_tube_planner(self.tube_planner)
        self.controller.mu_plan_scale = float(getattr(self.tube_planner, "mu_effective_plan_scale", 1.0))

        mu_init = float(np.asarray(self.tube_planner.get_horizon_mu_plan([0.0])).ravel()[0])
        self.friction_observer.reset(mu_init=mu_init)
        self.friction_fusion.reset()
        self.mu_rls_curr = mu_init
        self.bin_curr = 0

        self.s_prog = 0.0
        self.vs_prog = 0.2
        self.as_prog = 0.0
        self.a_act = np.zeros(3, dtype=float)
        self._prev_v_meas = None
        self._prev_a_cmd_body = np.zeros(2, dtype=float)
        return float(self.controller.path_length)

    def step(
        self,
        q_meas: np.ndarray,
        v_meas: np.ndarray,
        phi_est: np.ndarray,
        obs_info: np.ndarray = None,
        wheel_slips: np.ndarray = None,
    ) -> dict:
        """Thực thi một chu kỳ điều khiển 20 Hz DFE-MPCC + Tube H124 + Ma sát 2 tầng + Phân bổ lực 4 bánh."""
        q_arr = np.asarray(q_meas, dtype=float).reshape(3)
        v_arr = np.asarray(v_meas, dtype=float).reshape(3)
        phi_arr = np.asarray(phi_est, dtype=float).reshape(4)
        if obs_info is None:
            obs_info = np.array([10.0, 10.0, 0.5, 0.2, 0.1, 0.0, 0.0], dtype=float)

        # 1. Cập nhật Tầng 2 Adaptive FF-RLS từ sai lệch gia tốc thực tế (hoặc telemetry trượt 4 bánh)
        th = float(q_arr[2])
        c_th, s_th = float(np.cos(th)), float(np.sin(th))
        R_w2b_xy = np.array([[c_th, s_th], [-s_th, c_th]], dtype=float)

        if self._prev_v_meas is not None:
            a_world_meas = (v_arr - self._prev_v_meas) / max(self.dt, 1e-3)
            a_body_meas_xy = R_w2b_xy @ a_world_meas[:2]
        else:
            a_body_meas_xy = self._prev_a_cmd_body.copy()
        self._prev_v_meas = v_arr.copy()

        ddphi_x = -(omega_x ** 2) * phi_arr[0] - hd_x * phi_arr[1] + (1.0 / L_P_X) * self._prev_a_cmd_body[0]
        ddphi_y = -(omega_y ** 2) * phi_arr[2] - hd_y * phi_arr[3] - (1.0 / L_P_Y) * self._prev_a_cmd_body[1]
        f_slosh_prev = np.array([
            -M_1X * (-L_P_X * ddphi_x * np.cos(phi_arr[0])),
            -M_1Y * (L_P_Y * ddphi_y * np.cos(phi_arr[2])),
        ], dtype=float)
        W_cmd_prev_xy = M_TOTAL * self._prev_a_cmd_body - f_slosh_prev
        W_act_est_xy = M_TOTAL * a_body_meas_xy - f_slosh_prev

        Fz_prev, _ = compute_dynamic_normal_loads(
            m_total=M_TOTAL,
            a_body=self._prev_a_cmd_body,
            f_slosh_body=f_slosh_prev,
            phi_body=[phi_arr[0], phi_arr[2]],
            z_com=Z_COM,
            z_slosh=(Z_SLOSH_X, Z_SLOSH_Y),
            fz_min=0.889,
        )
        if wheel_slips is None:
            slip_den = max(float(np.linalg.norm(self._prev_a_cmd_body)), 0.25)
            est_slip = float(np.clip(np.linalg.norm(self._prev_a_cmd_body - a_body_meas_xy) / slip_den - 0.15, 0.0, 0.95))
            slips_obs = np.full(4, est_slip, dtype=float)
        else:
            slips_obs = np.asarray(wheel_slips, dtype=float).reshape(4)

        self.mu_rls_curr, _, self.bin_curr = self.friction_observer.update(
            W_act_est_xy, W_cmd_prev_xy, slips_obs, float(np.sum(Fz_prev)), M_TOTAL
        )

        # 2. Ánh xạ cung đường MPC <-> Tube H124 & hợp nhất ma sát chân trời Tầng 1 + Tầng 2
        x_nmpc = np.zeros(12, dtype=float)
        x_nmpc[0:3] = q_arr
        x_nmpc[3:6] = v_arr
        x_nmpc[6:9] = self.a_act
        x_nmpc[9] = self.s_prog
        x_nmpc[10] = self.vs_prog
        x_nmpc[11] = self.as_prog

        vs_prev_tube = max(float(np.clip(self.vs_prog, 0.05, 0.75)), 0.40)
        s_vec_mpc = np.clip(
            self.s_prog + np.arange(self.controller.N, dtype=float) * self.dt * vs_prev_tube,
            0.0,
            self.controller.path_length,
        )
        s_tube_h = self.controller.map_s_mpc_to_tube(s_vec_mpc)
        mu_plan_h = self.tube_planner.get_horizon_mu_plan(s_tube_h)
        mu_fuse_h = self.friction_fusion.fuse_horizon_mu(
            mu_plan_h, s_tube_h, self.mu_rls_curr, self.bin_curr, s_curr=float(s_tube_h[0])
        )
        v_ub_h, v_lb_h = self.tube_planner.query_online_horizon_tube(
            s_tube_h, phi_arr, mu_online=mu_fuse_h
        )
        v_ub_h = np.maximum(0.08, v_ub_h)
        v_lb_h = np.minimum(v_lb_h, v_ub_h)

        # 3. Giải NMPC DFE-MPCC với đầy đủ ràng buộc Zonotope T1, Ống H124 và T2 (q_eta)
        u_opt, js_opt, x_pred, u_pred, info = self.controller.solve_step(
            x_nmpc,
            phi_arr,
            obs_info,
            q_eta=self.q_eta_m,
            v_ub_horizon=v_ub_h,
            v_lb_horizon=v_lb_h,
            mu_horizon=mu_fuse_h,
        )

        self.s_prog = float(x_pred[9, 1])
        self.vs_prog = float(x_pred[10, 1])
        self.as_prog = float(x_pred[11, 1])
        self.a_act = np.array(x_pred[6:9, 1], dtype=float)

        # 4. Chuyển lệnh vận tốc & tính phân bổ lực 4 bánh guaranteed (O(1) Nullspace, [FL, FR, RR, RL])
        v_world_target = np.array(x_pred[3:6, 1], dtype=float)
        v_body_cmd = np.array([
            c_th * v_world_target[0] + s_th * v_world_target[1],
            -s_th * v_world_target[0] + c_th * v_world_target[1],
            v_world_target[2],
        ], dtype=float)

        a_cmd_body_xy = R_w2b_xy @ self.a_act[:2]
        self._prev_a_cmd_body = a_cmd_body_xy.copy()
        ddphi_x_now = -(omega_x ** 2) * phi_arr[0] - hd_x * phi_arr[1] + (1.0 / L_P_X) * a_cmd_body_xy[0]
        ddphi_y_now = -(omega_y ** 2) * phi_arr[2] - hd_y * phi_arr[3] - (1.0 / L_P_Y) * a_cmd_body_xy[1]
        f_slosh_now = np.array([
            -M_1X * (-L_P_X * ddphi_x_now * np.cos(phi_arr[0])),
            -M_1Y * (L_P_Y * ddphi_y_now * np.cos(phi_arr[2])),
        ], dtype=float)
        Fz_now, _ = compute_dynamic_normal_loads(
            m_total=M_TOTAL,
            a_body=a_cmd_body_xy,
            f_slosh_body=f_slosh_now,
            phi_body=[phi_arr[0], phi_arr[2]],
            z_com=Z_COM,
            z_slosh=(Z_SLOSH_X, Z_SLOSH_Y),
            fz_min=0.889,
        )
        W_cmd_now = np.array([
            M_TOTAL * a_cmd_body_xy[0] - f_slosh_now[0],
            M_TOTAL * a_cmd_body_xy[1] - f_slosh_now[1],
            IZ_BASE * self.a_act[2],
        ], dtype=float)
        f_bar_now = float(mu_fuse_h[0]) * Fz_now
        W_cmd_safe, zono_scale = scale_wrench_to_zonotope(W_cmd_now, f_bar_now / np.sqrt(2.0))
        wheel_alloc = allocate_mecanum_wheel_forces_guaranteed(
            W_cmd_safe, f_bar_now, lx=L_X, ly=L_Y, wheel_order="FL_FR_RR_RL"
        )
        _, _, traction_check = evaluate_traction_and_slip(
            wheel_alloc["f_wheel_scalar"] / np.sqrt(2.0), Fz_now, mu=float(mu_fuse_h[0])
        )

        return {
            "u_opt": u_opt,
            "js_opt": js_opt,
            "x_pred": x_pred,
            "u_pred": u_pred,
            "v_body_cmd": v_body_cmd,
            "mu_fuse_h": mu_fuse_h,
            "v_ub_h": v_ub_h,
            "v_lb_h": v_lb_h,
            "wheel_forces_4w": wheel_alloc["f_wheel_scalar"],
            "wheel_alloc_viol": wheel_alloc["max_violation"],
            "zonotope_scale": float(zono_scale),
            "traction_margins": traction_check["traction_margins"],
            "solver_info": info,
        }


class DfeMpccRos2Node(Node):
    def __init__(self):
        if not HAS_RCLPY:
            raise RuntimeError("rclpy không khả dụng trong môi trường Python hiện tại.")
        super().__init__('dfempcc_controller_node')

        self.declare_parameter('horizon_n', 8)
        self.declare_parameter('ipopt_max_iter', 12)
        self.declare_parameter('map_name', 'generic_dry')
        self.declare_parameter('mu_default', 0.35)

        n_hor = int(self.get_parameter('horizon_n').value)
        max_it = int(self.get_parameter('ipopt_max_iter').value)
        map_name = str(self.get_parameter('map_name').value)
        mu_def = float(self.get_parameter('mu_default').value)

        self.get_logger().info(
            f"Khởi tạo DFE-MPCC ROS 2 Pipeline (N={n_hor}, max_iter={max_it}, map={map_name})..."
        )
        self.pipeline = DfeMpccPipeline(
            horizon_N=n_hor,
            dt=0.05,
            ipopt_max_iter=max_it,
            mu_default=mu_def,
            map_name=map_name,
        )
        self.controller = self.pipeline.controller
        self.get_logger().info(
            f"Đã nạp quỹ đạo L={self.controller.path_length:.2f} m, "
            f"q_eta={self.pipeline.q_eta_m*1000.0:.2f} mm, {len(self.pipeline.tube_planner.checkpoints)} CP."
        )

        # Trạng thái đo đạc
        self.q_meas = np.array([-2.2, -2.2, 0.0], dtype=float)
        self.v_meas = np.zeros(3, dtype=float)
        self.phi_est = np.zeros(4, dtype=float)
        self._last_slosh_ts = None
        self._last_phi_xy = np.zeros(2, dtype=float)
        self.latest_scan = None
        self.latest_wheel_slips = None

        # Subscribers & Publishers
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.slosh_sub = self.create_subscription(Vector3Stamped, '/slosh/telemetry', self.slosh_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.path_sub = self.create_subscription(Path, '/plan', self.plan_callback, 10)
        self.slip_sub = self.create_subscription(
            Float64MultiArray, '/wheel_slip', self.wheel_slip_callback, 10
        )

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.wheel_pub = self.create_publisher(Float64MultiArray, '/dfempcc/wheel_forces', 10)

        # Timer vòng lặp điều khiển 20 Hz (dt = 0.05 s)
        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info(">>> DFE-MPCC Controller Node (Full Tube + Zero-Shot + FF-RLS) đã sẵn sàng!")

    def odom_callback(self, msg: Odometry):
        self.q_meas[0] = msg.pose.pose.position.x
        self.q_meas[1] = msg.pose.pose.position.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        self.q_meas[2] = 2.0 * np.arctan2(qz, qw)

        self.v_meas[0] = msg.twist.twist.linear.x
        self.v_meas[1] = msg.twist.twist.linear.y
        self.v_meas[2] = msg.twist.twist.angular.z

    def slosh_callback(self, msg: Vector3Stamped):
        now_t = time.monotonic()
        frame_id = getattr(getattr(msg, "header", None), "frame_id", "")
        if frame_id == "base_link":
            # run_ros2_rviz_headless_mujoco.py xuất [deg_x, deg_y, eta_mm] trên frame_id='base_link'
            px = float(np.radians(msg.vector.x))
            py = float(np.radians(msg.vector.y))
            z_raw = 0.0
        else:
            # bridge_node / run_ros2_sil_experiment xuất [rad_x, rad_y, dphi]
            px = float(msg.vector.x)
            py = float(msg.vector.y)
            z_raw = float(msg.vector.z)

        dpx_fd, dpy_fd = 0.0, 0.0
        if self._last_slosh_ts is not None:
            dt_cb = now_t - self._last_slosh_ts
            if 1e-4 < dt_cb < 0.25:
                dpx_fd = float(np.clip((px - self._last_phi_xy[0]) / dt_cb, -5.0, 5.0))
                dpy_fd = float(np.clip((py - self._last_phi_xy[1]) / dt_cb, -5.0, 5.0))

        if frame_id == "container":
            dpx = dpx_fd
            dpy = z_raw
        elif abs(z_raw) > 1e-12:
            dpx = z_raw
            dpy = dpy_fd
        else:
            dpx = dpx_fd
            dpy = dpy_fd

        self._last_slosh_ts = now_t
        self._last_phi_xy[0] = px
        self._last_phi_xy[1] = py
        self.phi_est[0] = px
        self.phi_est[1] = dpx
        self.phi_est[2] = py
        self.phi_est[3] = dpy

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg

    def wheel_slip_callback(self, msg: Float64MultiArray):
        if len(msg.data) >= 4:
            self.latest_wheel_slips = np.asarray(msg.data[:4], dtype=float)

    def plan_callback(self, msg: Path):
        if len(msg.poses) > 5:
            pts = [[p.pose.position.x, p.pose.position.y] for p in msg.poses]
            path_len = self.pipeline.set_waypoints(pts)
            self.get_logger().info(f"Đã cập nhật quỹ đạo /plan & Ống vận tốc H124 (L={path_len:.2f} m)!")

    def control_loop(self):
        obs_info = np.array([10.0, 10.0, 0.5, 0.2, 0.1, 0.0, 0.0], dtype=float)
        if self.latest_scan is not None:
            ranges = np.array(self.latest_scan.ranges)
            valid = (ranges > self.latest_scan.range_min) & (ranges < 1.5)
            if np.any(valid):
                idx = np.argmin(ranges[valid])
                min_r = ranges[valid][idx]
                angles = np.linspace(
                    self.latest_scan.angle_min, self.latest_scan.angle_max, len(ranges), endpoint=False
                )
                ang = angles[valid][idx]
                th = self.q_meas[2]
                ox = self.q_meas[0] + min_r * np.cos(th + ang)
                oy = self.q_meas[1] + min_r * np.sin(th + ang)
                obs_info = np.array([ox, oy, 0.5, 0.25, 0.10, 0.0, 0.0], dtype=float)

        try:
            step_out = self.pipeline.step(
                self.q_meas,
                self.v_meas,
                self.phi_est,
                obs_info=obs_info,
                wheel_slips=self.latest_wheel_slips,
            )
            v_body_cmd = step_out["v_body_cmd"]

            cmd = Twist()
            cmd.linear.x = float(v_body_cmd[0])
            cmd.linear.y = float(v_body_cmd[1])
            cmd.angular.z = float(v_body_cmd[2])
            self.cmd_pub.publish(cmd)

            wf_msg = Float64MultiArray()
            wf_msg.data = [float(v) for v in step_out["wheel_forces_4w"]]
            self.wheel_pub.publish(wf_msg)

        except Exception as e:
            self.get_logger().warn(f"DFE-MPCC lỗi bước giải: {e}")


def main(args=None):
    if not HAS_RCLPY:
        raise RuntimeError("Không thể khởi chạy DfeMpccRos2Node khi chưa cài đặt ROS 2 (rclpy).")
    rclpy.init(args=args)
    node = DfeMpccRos2Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
