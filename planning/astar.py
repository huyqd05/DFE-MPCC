import numpy as np
import heapq

class AStarPlanner:
    def __init__(self, x_bounds=(-3.0, 3.0), y_bounds=(-3.0, 3.0), resolution=0.05, robot_radius=0.25):
        """
        x_bounds, y_bounds: Giới hạn bản đồ (mét)
        resolution: Kích thước mỗi ô lưới (5 cm = 0.05 m)
        robot_radius: Bán kính an toàn của robot (để phóng to vật cản, tránh cọ tường)
        """
        self.xmin, self.xmax = x_bounds
        self.ymin, self.ymax = y_bounds
        self.res = resolution
        self.robot_radius = robot_radius
        
        self.nx = int(np.round((self.xmax - self.xmin) / self.res)) + 1
        self.ny = int(np.round((self.ymax - self.ymin) / self.res)) + 1
        self.grid = np.zeros((self.nx, self.ny), dtype=bool)

    def to_grid(self, x, y):
        """Chuyển toạ độ thực (mét) sang chỉ số lưới (i, j)"""
        ix = int(np.round((x - self.xmin) / self.res))
        iy = int(np.round((y - self.ymin) / self.res))
        return ix, iy

    def to_world(self, ix, iy):
        """Chuyển chỉ số lưới (i, j) về toạ độ thực (x, y)"""
        x = self.xmin + ix * self.res
        y = self.ymin + iy * self.res
        return x, y

    def add_obstacles(self, obs_circles, obs_boxes):
        """
        Nạp vật cản và phóng to (inflation) theo kích thước robot:
        - obs_circles: danh sách [x, y, radius] (ví dụ hình trụ)
        - obs_boxes: danh sách [x, y, half_w, half_h] (ví dụ bức tường)
        """
        for i in range(self.nx):
            for j in range(self.ny):
                x, y = self.to_world(i, j)
                
                # Kiểm tra va chạm với vật cản tròn
                for ox, oy, r in obs_circles:
                    if np.hypot(x - ox, y - oy) <= (r + self.robot_radius):
                        self.grid[i, j] = True
                        break
                        
                # Kiểm tra va chạm với vật cản hình hộp (tường)
                if not self.grid[i, j]:
                    for bx, by, hw, hh in obs_boxes:
                        if (bx - hw - self.robot_radius <= x <= bx + hw + self.robot_radius and
                            by - hh - self.robot_radius <= y <= by + hh + self.robot_radius):
                            self.grid[i, j] = True
                            break

    def plan(self, start_pos, goal_pos, smooth=True):
        """Thuật toán tìm đường A* (hỗ trợ làm mượt đường tự động với smooth=True)"""
        sx, sy = self.to_grid(start_pos[0], start_pos[1])
        gx, gy = self.to_grid(goal_pos[0], goal_pos[1])

        # Kiểm tra nếu Goal rơi vào vùng cấm (tường / vật cản), tự động tìm ô tự do gần nhất
        if self.grid[gx, gy]:
            print(f"[AStar Warning] Goal {goal_pos} is inside obstacle/wall margin! Snapping to nearest free cell...")
            best_d, best_cell = float('inf'), (gx, gy)
            for r in range(1, 15):
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        nx, ny = gx + dx, gy + dy
                        if 0 <= nx < self.nx and 0 <= ny < self.ny and not self.grid[nx, ny]:
                            d = dx * dx + dy * dy
                            if d < best_d:
                                best_d = d
                                best_cell = (nx, ny)
                if best_d < float('inf'):
                    gx, gy = best_cell
                    snapped_pos = self.to_world(gx, gy)
                    print(f"[AStar] Snapped goal to nearest free cell: ({snapped_pos[0]:.2f}, {snapped_pos[1]:.2f})")
                    break

        # 8 hướng di chuyển (ngang, dọc, chéo)
        motions = [
            (1, 0, 1.0), (0, 1, 1.0), (-1, 0, 1.0), (0, -1, 1.0),
            (1, 1, np.sqrt(2)), (-1, 1, np.sqrt(2)), (-1, -1, np.sqrt(2)), (1, -1, np.sqrt(2))
        ]
        
        open_set = []
        heapq.heappush(open_set, (0.0, (sx, sy)))
        
        came_from = {}
        cost_so_far = {(sx, sy): 0.0}
        
        reached_goal = False
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == (gx, gy) or np.hypot(current[0] - gx, current[1] - gy) <= 1:
                gx, gy = current
                reached_goal = True
                break
                
            for dx, dy, cost in motions:
                neighbor = (current[0] + dx, current[1] + dy)
                if not (0 <= neighbor[0] < self.nx and 0 <= neighbor[1] < self.ny):
                    continue
                if self.grid[neighbor[0], neighbor[1]]:
                    continue
                    
                new_cost = cost_so_far[current] + cost
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    heuristic = np.hypot(neighbor[0] - gx, neighbor[1] - gy)
                    priority = new_cost + heuristic
                    heapq.heappush(open_set, (priority, neighbor))
                    came_from[neighbor] = current
                    
        if not reached_goal and (sx, sy) != (gx, gy):
            print(f"[AStar Error] Could not find path from {start_pos} to {goal_pos}! Please check coordinates.")
            return np.array([list(self.to_world(sx, sy))])

        # Tái tạo lại đường đi từ Đích về Xuất phát
        path = []
        curr = (gx, gy)
        while curr in came_from:
            path.append(list(self.to_world(curr[0], curr[1])))
            curr = came_from[curr]
        path.append(list(self.to_world(sx, sy)))
        path.reverse()
        raw_path = np.array(path)

        if smooth and len(raw_path) > 2:
            return self.smooth_path(raw_path)
        return raw_path

    def smooth_path(self, path, weight_data=0.15, weight_smooth=0.70, tolerance=0.0005, max_iter=300):
        """
        Làm mượt đường đi A* bằng thuật toán Gradient Descent (Elastic Band):
        - weight_data: Trọng số kéo về đường ban đầu của A* (tránh lệch khỏi hành lang an toàn)
        - weight_smooth: Trọng số kéo căng các đỉnh nhọn, triệt tiêu góc bẻ 90 và 45 độ
        - Kiểm tra an toàn: Đảm bảo các điểm sau khi làm mượt không va chạm với vật cản / biên
        """
        if len(path) <= 2:
            return np.array(path)

        newpath = np.copy(path).astype(float)
        change = tolerance
        it = 0
        while change >= tolerance and it < max_iter:
            change = 0.0
            it += 1
            for i in range(1, len(path) - 1):
                old_val = np.copy(newpath[i])
                candidate = newpath[i] + weight_data * (path[i] - newpath[i]) + \
                            weight_smooth * (newpath[i - 1] + newpath[i + 1] - 2.0 * newpath[i])

                # Kiểm tra an toàn: chỉ nhận điểm mới nếu không rơi vào ô vật cản
                ix, iy = self.to_grid(candidate[0], candidate[1])
                if 0 <= ix < self.nx and 0 <= iy < self.ny and not self.grid[ix, iy]:
                    newpath[i] = candidate
                    change += np.linalg.norm(candidate - old_val)

        return newpath