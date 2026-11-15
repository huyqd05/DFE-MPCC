import os
import sys
import numpy as np

# Nạp mã nguồn dự án vào sys.path
PROJECT_DIR = "/mnt/c/Users/LENOVO/Desktop/mujoco"
for p in [PROJECT_DIR, f"{PROJECT_DIR}/controller", f"{PROJECT_DIR}/solver", f"{PROJECT_DIR}/planning"]:
    if p not in sys.path:
        sys.path.insert(0, p)

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3Stamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan

from dfe_mpcc_amr_controller import DFEMPCCAMRController
from mecanum_nullspace_allocation import allocate_mecanum_wheel_forces_guaranteed
from astar import AStarPlanner


class DfeMpccRos2Node(Node):
    def __init__(self):
        super().__init__('dfempcc_controller_node')
        
        # 1. Khởi tạo bộ điều khiển DFE-MPCC (20 Hz, N=15)
        self.get_logger().info("Khởi tạo bộ điều khiển DFE-MPCC với CasADi/IPOPT...")
        self.controller = DFEMPCCAMRController(
            N=15,
            dt=0.05,
            mu=0.35,
            state_con_mode="skip0",
            ipopt_max_iter=30
        )
        
        # 2. Hoạch định đường đi mẫu ban đầu từ (-2.2, -2.2) đến (2.2, 2.2) bằng A*
        self.get_logger().info("Hoạch định quỹ đạo tham chiếu toàn cục ban đầu...")
        planner = AStarPlanner(x_bounds=(-3.0, 3.0), y_bounds=(-3.0, 3.0), resolution=0.05, robot_radius=0.25)
        # Nạp chướng ngại vật cơ bản của world.xml
        obs_boxes = [
            [-1.0, -1.075, 0.05, 1.875],
            [1.0, 1.075, 0.05, 1.875],
            [0.60, 0.20, 0.35, 0.05],
            [2.45, -0.35, 0.45, 0.05]
        ]
        obs_circles = [
            [-2.30, -0.70, 0.28],
            [-1.50, 1.80, 0.32],
            [0.25, 2.05, 0.26],
            [-0.40, -1.90, 0.28],
            [2.40, 0.85, 0.28]
        ]
        planner.add_obstacles(obs_circles, obs_boxes)
        waypoints = planner.plan([-2.2, -2.2], [2.2, 2.2], smooth=True)
        self.controller.set_path(waypoints)
        self.get_logger().info(f"Đã nạp đường đi toàn cục thành công: L = {self.controller.path_length:.2f} m.")
        
        # 3. Trạng thái đo đạc
        self.q_meas = np.array([-2.2, -2.2, 0.0])
        self.v_meas = np.zeros(3)
        self.a_act = np.zeros(3)
        self.s_prog = 0.0
        self.vs_prog = 0.2
        self.as_prog = 0.0
        self.phi_est = np.zeros(4)
        self.latest_scan = None
        self.mu_floor = 0.35
        
        # 4. Subscribers & Publishers
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.slosh_sub = self.create_subscription(Vector3Stamped, '/slosh/telemetry', self.slosh_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.path_sub = self.create_subscription(Path, '/plan', self.plan_callback, 10)
        
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # 5. Timer vòng lặp điều khiển 20 Hz (dt = 0.05 s)
        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info(">>> DFE-MPCC Controller Node đã kích hoạt tại tần số 20 Hz!")

    def odom_callback(self, msg: Odometry):
        self.q_meas[0] = msg.pose.pose.position.x
        self.q_meas[1] = msg.pose.pose.position.y
        # Quaternion -> Yaw
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        self.q_meas[2] = 2.0 * np.arctan2(qz, qw)
        
        self.v_meas[0] = msg.twist.twist.linear.x
        self.v_meas[1] = msg.twist.twist.linear.y
        self.v_meas[2] = msg.twist.twist.angular.z

    def slosh_callback(self, msg: Vector3Stamped):
        self.phi_est[0] = msg.vector.x # phi_x
        self.phi_est[1] = msg.vector.z # dphi_x
        self.phi_est[2] = msg.vector.y # phi_y
        self.phi_est[3] = 0.0          # dphi_y

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg

    def plan_callback(self, msg: Path):
        # Khi Nav2 sinh đường toàn cục mới, cập nhật lại Cubic Spline cho MPCC
        if len(msg.poses) > 5:
            pts = [[p.pose.position.x, p.pose.position.y] for p in msg.poses]
            self.controller.set_path(pts)
            self.s_prog = 0.0
            self.get_logger().info("Đã cập nhật quỹ đạo mới từ Nav2 /plan!")

    def control_loop(self):
        # 1. Tạo vector trạng thái 12 biến
        x_nmpc = np.zeros(12)
        x_nmpc[0:3] = self.q_meas
        x_nmpc[3:6] = self.v_meas
        x_nmpc[6:9] = self.a_act
        x_nmpc[9] = self.s_prog
        x_nmpc[10] = self.vs_prog
        x_nmpc[11] = self.as_prog
        
        # 2. Xử lý vật cản gần nhất từ LiDAR cho Obstacle-ECBF
        obs_info = np.array([10.0, 10.0, 0.5, 0.2, 0.1, 0.0, 0.0])
        if self.latest_scan is not None:
            ranges = np.array(self.latest_scan.ranges)
            valid = (ranges > self.latest_scan.range_min) & (ranges < 1.5)
            if np.any(valid):
                idx = np.argmin(ranges[valid])
                min_r = ranges[valid][idx]
                angles = np.linspace(self.latest_scan.angle_min, self.latest_scan.angle_max, len(ranges), endpoint=False)
                ang = angles[valid][idx]
                th = self.q_meas[2]
                ox = self.q_meas[0] + min_r * np.cos(th + ang)
                oy = self.q_meas[1] + min_r * np.sin(th + ang)
                obs_info = np.array([ox, oy, 0.5, 0.25, 0.10, 0.0, 0.0])

        # 3. Giải bài toán DFE-MPCC (IPOPT)
        try:
            u_opt, js_opt, x_pred, u_pred, info = self.controller.solve_step(
                x_nmpc, self.phi_est, obs_info, mu_floor=self.mu_floor
            )
            
            # Cập nhật tiến độ dự báo cho chu kỳ sau
            self.s_prog = float(x_pred[9, 1])
            self.vs_prog = float(x_pred[10, 1])
            self.as_prog = float(x_pred[11, 1])
            self.a_act = np.array(x_pred[6:9, 1])
            
            # Vận tốc mục tiêu tiếp theo trong World Frame
            v_world_target = np.array(x_pred[3:6, 1])
            
            # Chuyển vận tốc sang Body Frame để gửi lệnh /cmd_vel
            th = self.q_meas[2]
            c_th, s_th = np.cos(th), np.sin(th)
            R_w2b = np.array([
                [ c_th, s_th, 0.0],
                [-s_th, c_th, 0.0],
                [ 0.0,  0.0,  1.0]
            ])
            v_body_cmd = R_w2b @ v_world_target
            
            # 4. Xuất bản /cmd_vel
            cmd = Twist()
            cmd.linear.x = float(v_body_cmd[0])
            cmd.linear.y = float(v_body_cmd[1])
            cmd.angular.z = float(v_body_cmd[2])
            self.cmd_pub.publish(cmd)
            
        except Exception as e:
            self.get_logger().warn(f"DFE-MPCC lỗi bước giải: {e}")


def main(args=None):
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
