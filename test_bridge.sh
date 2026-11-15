#!/bin/bash
set -e
source /opt/ros/jazzy/setup.bash
source ~/amr_ws/install/setup.bash

echo ">>> KHỞI ĐỘNG BRIDGE NODE TRONG NỀN..."
ros2 run amr_slosh_bridge bridge_node &
PID=$!

sleep 2

echo ""
echo "=== DANH SÁCH TOPIC ROS 2 ĐANG XUẤT BẢN ==="
ros2 topic list

echo ""
echo "=== KIỂM TRA MỘT THÔNG ĐIỆP ODOMETRY ==="
ros2 topic echo /odom --once

echo ""
echo "=== KIỂM TRA MỘT THÔNG ĐIỆP SLOSH TELEMETRY ==="
ros2 topic echo /slosh/telemetry --once

kill $PID 2>/dev/null || true
wait $PID 2>/dev/null || true
echo ">>> KIỂM THỬ THÀNH CÔNG RỰC RỠ!"
