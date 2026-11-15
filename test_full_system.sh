#!/bin/bash
set -e
source /opt/ros/jazzy/setup.bash
source ~/amr_ws/install/setup.bash

echo "=========================================================="
echo "CHẠY THỬ NGHIỆM ĐỒNG BỘ: MUJOCO BRIDGE + DFE-MPCC NODE"
echo "=========================================================="

# 1. Khởi động MuJoCo Bridge Node
ros2 run amr_slosh_bridge bridge_node &
BRIDGE_PID=$!
sleep 2

# 2. Khởi động DFE-MPCC Controller Node
ros2 run amr_slosh_bridge dfempcc_node &
CTRL_PID=$!
sleep 3

echo ""
echo "=== KIỂM TRA TOPIC LỆNH /cmd_vel PHÁT TỪ DFE-MPCC ==="
ros2 topic echo /cmd_vel --once

echo ""
echo "=== KIỂM TRA TỌA ĐỘ XE SAU KHI DFE-MPCC ĐIỀU KHIỂN ==="
ros2 topic echo /odom --once

# Dọn dẹp tiến trình
kill $CTRL_PID 2>/dev/null || true
kill $BRIDGE_PID 2>/dev/null || true
wait $CTRL_PID 2>/dev/null || true
wait $BRIDGE_PID 2>/dev/null || true

echo ""
echo "=========================================================="
echo ">>> CHU TRÌNH VÒNG KÍN ROS 2 + MUJOCO ĐÃ CHẠY THÀNH CÔNG!"
echo "=========================================================="
