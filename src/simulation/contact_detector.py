from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

# 可选：Numba 加速
try:
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False


@dataclass
class NeighborParams:
    r: float
    skin: float
    cell_size: float | None = None
    Lx: float | None = None
    Ly: float | None = None


# ============================================================
# Numba 加速版邻域搜索（周期边界 + 5×5 邻域）
# ============================================================

if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _numba_minimal_image(dx: float, dy: float, Lx: float, Ly: float):
        """最小镜像位移（几何判断）"""
        half_x = 0.5 * Lx
        half_y = 0.5 * Ly
        if dx > half_x:
            dx -= Lx
        elif dx < -half_x:
            dx += Lx
        if dy > half_y:
            dy -= Ly
        elif dy < -half_y:
            dy += Ly
        return dx, dy

    @njit(cache=True)
    def _numba_wrap_periodic(x: np.ndarray, y: np.ndarray, Lx: float, Ly: float):
        """将坐标包裹到 [0, L)"""
        n = x.shape[0]
        xp = np.empty(n, dtype=np.float64)
        yp = np.empty(n, dtype=np.float64)
        half_Lx = 0.5 * Lx
        half_Ly = 0.5 * Ly
        for i in range(n):
            xp[i] = (x[i] + half_Lx) % Lx
            yp[i] = (y[i] + half_Ly) % Ly
        return xp, yp

    @njit(cache=True)
    def _numba_build_cell_structure(xp: np.ndarray, yp: np.ndarray, Nx: int, Ny: int, Lx: float, Ly: float):
        """构建格子结构用于快速邻域查询"""
        n = xp.shape[0]
        num_cells = Nx * Ny

        # 计算每个粒子的格子索引
        cell_indices = np.empty(n, dtype=np.int64)
        cell_x = Lx / Nx
        cell_y = Ly / Ny
        for i in range(n):
            gx = int(xp[i] / cell_x) % Nx
            gy = int(yp[i] / cell_y) % Ny
            cell_indices[i] = gx * Ny + gy

        # 计算每个格子的粒子数
        cell_count = np.zeros(num_cells, dtype=np.int64)
        for i in range(n):
            cell_count[cell_indices[i]] += 1

        # 计算每个格子的起始位置
        cell_start = np.zeros(num_cells, dtype=np.int64)
        for c in range(1, num_cells):
            cell_start[c] = cell_start[c - 1] + cell_count[c - 1]

        # 按格子排序粒子索引
        sorted_ids = np.empty(n, dtype=np.int64)
        cell_offset = np.zeros(num_cells, dtype=np.int64)
        for i in range(n):
            c = cell_indices[i]
            sorted_ids[cell_start[c] + cell_offset[c]] = i
            cell_offset[c] += 1

        return cell_indices, cell_start, cell_count, sorted_ids

    @njit(cache=True)
    def _numba_find_contacts_periodic(
        xp: np.ndarray, yp: np.ndarray,
        cell_start: np.ndarray, cell_count: np.ndarray, sorted_ids: np.ndarray,
        Nx: int, Ny: int, Lx: float, Ly: float, thresh: float, max_pairs: int
    ):
        """Numba 邻域搜索（周期边界 + 5×5 邻域）"""
        n = xp.shape[0]
        pair_i = np.empty(max_pairs, dtype=np.int64)
        pair_j = np.empty(max_pairs, dtype=np.int64)
        pair_dx = np.empty(max_pairs, dtype=np.float64)
        pair_dy = np.empty(max_pairs, dtype=np.float64)
        pair_d = np.empty(max_pairs, dtype=np.float64)

        count = 0
        thresh_sq = thresh * thresh
        cell_x = Lx / Nx
        cell_y = Ly / Ny

        for i in range(n):
            xi = xp[i]
            yi = yp[i]
            gx_i = int(xi / cell_x) % Nx
            gy_i = int(yi / cell_y) % Ny

            # 5×5 邻域扫描
            for dx_cell in range(-2, 3):
                for dy_cell in range(-2, 3):
                    ng_x = (gx_i + dx_cell) % Nx
                    ng_y = (gy_i + dy_cell) % Ny
                    cell_idx = ng_x * Ny + ng_y

                    start = cell_start[cell_idx]
                    cnt = cell_count[cell_idx]

                    for k in range(start, start + cnt):
                        j = sorted_ids[k]
                        if j <= i:
                            continue

                        dx = xp[j] - xi
                        dy = yp[j] - yi
                        dx, dy = _numba_minimal_image(dx, dy, Lx, Ly)

                        d_sq = dx * dx + dy * dy
                        if d_sq <= thresh_sq:
                            if count < max_pairs:
                                pair_i[count] = i
                                pair_j[count] = j
                                pair_dx[count] = dx
                                pair_dy[count] = dy
                                pair_d[count] = math.sqrt(d_sq)
                                count += 1

        return pair_i[:count], pair_j[:count], pair_dx[:count], pair_dy[:count], pair_d[:count]

    # ============================================================
    # 并行版本：使用 prange 进行多线程加速
    # ============================================================

    @njit(cache=True, parallel=True)
    def _numba_count_neighbors_parallel(
        xp: np.ndarray, yp: np.ndarray,
        cell_start: np.ndarray, cell_count: np.ndarray, sorted_ids: np.ndarray,
        Nx: int, Ny: int, Lx: float, Ly: float, thresh: float
    ):
        """第一阶段：并行统计每个粒子的邻居数量"""
        n = xp.shape[0]
        neighbor_counts = np.zeros(n, dtype=np.int64)
        thresh_sq = thresh * thresh
        cell_x = Lx / Nx
        cell_y = Ly / Ny

        for i in prange(n):
            xi = xp[i]
            yi = yp[i]
            gx_i = int(xi / cell_x) % Nx
            gy_i = int(yi / cell_y) % Ny
            local_count = 0

            for dx_cell in range(-2, 3):
                for dy_cell in range(-2, 3):
                    ng_x = (gx_i + dx_cell) % Nx
                    ng_y = (gy_i + dy_cell) % Ny
                    cell_idx = ng_x * Ny + ng_y

                    start = cell_start[cell_idx]
                    cnt = cell_count[cell_idx]

                    for k in range(start, start + cnt):
                        j = sorted_ids[k]
                        if j <= i:
                            continue

                        dx = xp[j] - xi
                        dy = yp[j] - yi
                        dx, dy = _numba_minimal_image(dx, dy, Lx, Ly)

                        d_sq = dx * dx + dy * dy
                        if d_sq <= thresh_sq:
                            local_count += 1

            neighbor_counts[i] = local_count

        return neighbor_counts

    @njit(cache=True, parallel=True)
    def _numba_find_contacts_parallel(
        xp: np.ndarray, yp: np.ndarray,
        cell_start: np.ndarray, cell_count: np.ndarray, sorted_ids: np.ndarray,
        Nx: int, Ny: int, Lx: float, Ly: float, thresh: float,
        pair_offsets: np.ndarray, total_pairs: int
    ):
        """第二阶段：并行填充邻居对数据"""
        n = xp.shape[0]
        pair_i = np.empty(total_pairs, dtype=np.int64)
        pair_j = np.empty(total_pairs, dtype=np.int64)
        pair_dx = np.empty(total_pairs, dtype=np.float64)
        pair_dy = np.empty(total_pairs, dtype=np.float64)
        pair_d = np.empty(total_pairs, dtype=np.float64)

        thresh_sq = thresh * thresh
        cell_x = Lx / Nx
        cell_y = Ly / Ny

        for i in prange(n):
            xi = xp[i]
            yi = yp[i]
            gx_i = int(xi / cell_x) % Nx
            gy_i = int(yi / cell_y) % Ny
            write_pos = pair_offsets[i]

            for dx_cell in range(-2, 3):
                for dy_cell in range(-2, 3):
                    ng_x = (gx_i + dx_cell) % Nx
                    ng_y = (gy_i + dy_cell) % Ny
                    cell_idx = ng_x * Ny + ng_y

                    start = cell_start[cell_idx]
                    cnt = cell_count[cell_idx]

                    for k in range(start, start + cnt):
                        j = sorted_ids[k]
                        if j <= i:
                            continue

                        dx = xp[j] - xi
                        dy = yp[j] - yi
                        dx, dy = _numba_minimal_image(dx, dy, Lx, Ly)

                        d_sq = dx * dx + dy * dy
                        if d_sq <= thresh_sq:
                            pair_i[write_pos] = i
                            pair_j[write_pos] = j
                            pair_dx[write_pos] = dx
                            pair_dy[write_pos] = dy
                            pair_d[write_pos] = math.sqrt(d_sq)
                            write_pos += 1

        return pair_i, pair_j, pair_dx, pair_dy, pair_d


def find_contacts_numba(x: np.ndarray, y: np.ndarray, prm: NeighborParams, use_parallel: bool = False) -> List[Tuple[int, int, float, float, float]]:
    """Numba 加速的邻域搜索（仅周期边界）
    
    Args:
        x, y: 粒子坐标
        prm: 邻域参数
        use_parallel: 是否启用多线程并行（默认 False）
    """
    if not _NUMBA_AVAILABLE:
        return find_contacts(x, y, prm)

    r, skin = float(prm.r), float(prm.skin)
    thresh = 2.0 * r + max(0.0, skin)
    cell = float(prm.cell_size or thresh)
    Lx = float(prm.Lx or 0.0)
    Ly = float(prm.Ly or 0.0)

    if Lx <= 0.0 or Ly <= 0.0:
        # 非周期回退到 Python 版本
        return find_contacts(x, y, prm)

    n = x.shape[0]
    Nx = max(1, int(math.ceil(Lx / cell)))
    Ny = max(1, int(math.ceil(Ly / cell)))

    # 包裹坐标
    xp, yp = _numba_wrap_periodic(x, y, Lx, Ly)

    # 构建格子结构
    cell_indices, cell_start, cell_count, sorted_ids = _numba_build_cell_structure(xp, yp, Nx, Ny, Lx, Ly)

    if use_parallel:
        # 并行版本：两阶段策略
        # 第一阶段：统计每个粒子的邻居数
        neighbor_counts = _numba_count_neighbors_parallel(
            xp, yp, cell_start, cell_count, sorted_ids, Nx, Ny, Lx, Ly, thresh
        )
        
        # 计算每个粒子的写入偏移量（前缀和）
        pair_offsets = np.zeros(n, dtype=np.int64)
        total_pairs = 0
        for i in range(n):
            pair_offsets[i] = total_pairs
            total_pairs += neighbor_counts[i]
        
        if total_pairs == 0:
            return []
        
        # 第二阶段：并行填充邻居对
        pair_i, pair_j, pair_dx, pair_dy, pair_d = _numba_find_contacts_parallel(
            xp, yp, cell_start, cell_count, sorted_ids, Nx, Ny, Lx, Ly, thresh,
            pair_offsets, int(total_pairs)
        )
    else:
        # 串行版本
        max_pairs = min(n * n // 2, n * 50)
        pair_i, pair_j, pair_dx, pair_dy, pair_d = _numba_find_contacts_periodic(
            xp, yp, cell_start, cell_count, sorted_ids, Nx, Ny, Lx, Ly, thresh, max_pairs
        )

    # 转换为 Python 列表
    pairs = []
    for k in range(len(pair_i)):
        pairs.append((int(pair_i[k]), int(pair_j[k]), float(pair_dx[k]), float(pair_dy[k]), float(pair_d[k])))
    return pairs


# ============================================================
# 原始 Python 版本（作为回退）
# ============================================================

def _wrap_pos_periodic(x: np.ndarray, y: np.ndarray, Lx: float, Ly: float) -> tuple[np.ndarray, np.ndarray]:
    """将坐标包裹到域内 [0, L)"""
    xp = (x + 0.5 * Lx) % Lx
    yp = (y + 0.5 * Ly) % Ly
    return xp, yp


def build_grid_periodic(x: np.ndarray, y: np.ndarray, cell: float, Lx: float, Ly: float) -> tuple[Dict[Tuple[int, int], List[int]], int, int]:
    """周期网格：索引取模，保证边界邻居被覆盖"""
    Nx = max(1, int(math.ceil(Lx / cell)))
    Ny = max(1, int(math.ceil(Ly / cell)))
    xp, yp = _wrap_pos_periodic(x, y, Lx, Ly)
    grid: Dict[Tuple[int, int], List[int]] = {}
    for i, (xi, yi) in enumerate(zip(xp, yp)):
        gx = int(math.floor(xi / cell)) % Nx
        gy = int(math.floor(yi / cell)) % Ny
        grid.setdefault((gx, gy), []).append(i)
    return grid, Nx, Ny


def _minimal_image(dx: float, dy: float, Lx: float, Ly: float) -> tuple[float, float]:
    """最小镜像位移（几何判断）"""
    if Lx > 0.0 and Ly > 0.0:
        half_x = 0.5 * Lx
        half_y = 0.5 * Ly
        if dx > half_x:
            dx -= Lx
        elif dx < -half_x:
            dx += Lx
        if dy > half_y:
            dy -= Ly
        elif dy < -half_y:
            dy += Ly
    return dx, dy


def find_contacts(x: np.ndarray, y: np.ndarray, prm: NeighborParams) -> List[Tuple[int, int, float, float, float]]:
    """
    Python 版邻域搜索（作为回退）
    返回候选对列表: (i, j, dx, dy, dist)
    threshold = 2r + skin
    """
    r, skin = float(prm.r), float(prm.skin)
    thresh = 2.0 * r + max(0.0, skin)
    cell = float(prm.cell_size or thresh)
    Lx = float(prm.Lx or 0.0)
    Ly = float(prm.Ly or 0.0)

    if Lx <= 0.0 or Ly <= 0.0:
        # 非周期回退到非包裹网格
        grid: Dict[Tuple[int, int], List[int]] = {}
        for i, (xi, yi) in enumerate(zip(x, y)):
            gx = int(math.floor(xi / cell))
            gy = int(math.floor(yi / cell))
            grid.setdefault((gx, gy), []).append(i)
        Nx = Ny = None
        xp, yp = x, y
    else:
        # 周期模式：包裹坐标
        xp, yp = _wrap_pos_periodic(x, y, Lx, Ly)
        grid, Nx, Ny = build_grid_periodic(x, y, cell, Lx, Ly)

    pairs: List[Tuple[int, int, float, float, float]] = []

    for (gx, gy), idxs in grid.items():
        for i in idxs:
            xi = xp[i]
            yi = yp[i]
            # 5×5 邻域确保覆盖（解决边界对齐问题）
            for dx_cell in (-2, -1, 0, 1, 2):
                for dy_cell in (-2, -1, 0, 1, 2):
                    if Lx > 0.0 and Ly > 0.0:
                        ng_x = (gx + dx_cell) % Nx
                        ng_y = (gy + dy_cell) % Ny
                        neigh = grid.get((ng_x, ng_y))
                    else:
                        neigh = grid.get((gx + dx_cell, gy + dy_cell))
                    if not neigh:
                        continue
                    for j in neigh:
                        if j <= i:
                            continue
                        dx = xp[j] - xi
                        dy = yp[j] - yi
                        if Lx > 0.0 and Ly > 0.0:
                            dx, dy = _minimal_image(dx, dy, Lx, Ly)
                        d = math.hypot(dx, dy)
                        if d <= thresh:
                            pairs.append((i, j, dx, dy, d))
    return pairs

def connected_components_from_contacts(n: int, contacts_2r: List[Tuple[int,int]]) -> List[List[int]]:
    adj: List[List[int]] = [[] for _ in range(n)]
    for (i, j) in contacts_2r:
        adj[i].append(j)
        adj[j].append(i)
    vis = [False]*n
    comps: List[List[int]] = []
    for i in range(n):
        if vis[i]:
            continue
        stack = [i]; vis[i] = True
        comp = [i]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not vis[v]:
                    vis[v] = True
                    stack.append(v)
                    comp.append(v)
        comps.append(comp)
    return comps
