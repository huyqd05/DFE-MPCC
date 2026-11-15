import sys
import os
import rclpy

print(f"Python version: {sys.version}")

rclpy.init()

try:
    from amr_slosh_bridge.bridge_node import MujocoRos2Bridge
    print("Khởi tạo MujocoRos2Bridge...")
    bridge = MujocoRos2Bridge()
    print("[THÀNH CÔNG] MujocoRos2Bridge đã khởi tạo xong và nạp thành công XML MuJoCo!")
    print(f"  - Model nq: {bridge.model.nq}, nv: {bridge.model.nv}, ngeom: {bridge.model.ngeom}")
    print(f"  - Vị trí robot ban đầu: qpos[0:3] = {bridge.data.qpos[:3]}")
    # Chạy thử 1 physics step
    bridge.physics_step()
    print("[THÀNH CÔNG] physics_step() chạy 1 chu kỳ hoàn tất không có lỗi!")
    bridge.destroy_node()
except Exception as e:
    print(f"[LỖI] MujocoRos2Bridge thất bại: {e}")
    import traceback
    traceback.print_exc()

try:
    from amr_slosh_bridge.dfempcc_node import DfeMpccRos2Node
    print("\nKhởi tạo DfeMpccRos2Node...")
    node = DfeMpccRos2Node()
    print("[THÀNH CÔNG] DfeMpccRos2Node đã khởi tạo xong bộ điều khiển DFE-MPCC!")
    print(f"  - Chiều dài đường tham chiếu: {node.controller.path_length:.2f} m")
    print(f"  - Trạng thái đo ban đầu: q = {node.q_meas}")
    # Chạy thử 1 control step
    node.control_loop()
    print("[THÀNH CÔNG] control_loop() giải 1 chu kỳ IPOPT hoàn tất không có lỗi!")
    node.destroy_node()
except Exception as e:
    print(f"[LỖI] DfeMpccRos2Node thất bại: {e}")
    import traceback
    traceback.print_exc()

rclpy.shutdown()
print("\n>>> KIỂM TRA TOÀN DIỆN ROS 2 HOÀN TẤT!")
