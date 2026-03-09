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
# Numba 加速版邻域搜索（周期边界 + 3×3 邻域）
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
        """Numba 邻域搜索（周期边界 + 3×3 邻域，要求 cell_size >= thresh）"""
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

            # 3×3 邻域扫描（cell_size >= thresh 保证无漏检）
            for dx_cell in range(-1, 2):
                for dy_cell in range(-1, 2):
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
        """第一阶段：并行统计每个粒子的邻居数量（3×3 邻域）"""
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

            for dx_cell in range(-1, 2):
                for dy_cell in range(-1, 2):
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
        """第二阶段：并行填充邻居对数据（3×3 邻域）"""
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

            for dx_cell in range(-1, 2):
                for dy_cell in range(-1, 2):
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
    # 确保 cell_size >= thresh，3×3 邻域才不会漏检
    cell = max(cell, thresh)
    Lx = float(prm.Lx or 0.0)
    Ly = float(prm.Ly or 0.0)

    if Lx <= 0.0 or Ly <= 0.0:
        # 非周期回退到 Python 版本
        return find_contacts(x, y, prm)

    n = x.shape[0]
    # Nx 用 floor 而非 ceil，保证实际格子尺寸 Lx/Nx >= cell >= thresh，3×3 扫描才不会漏检
    Nx = max(1, int(Lx / cell))
    Ny = max(1, int(Ly / cell))

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
    Nx = max(1, int(Lx / cell))
    Ny = max(1, int(Ly / cell))
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
    # 确保 cell_size >= thresh，3×3 邻域才不会漏检
    cell = max(cell, thresh)
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
            # 3×3 邻域扫描（cell_size >= thresh 保证无漏检）
            for dx_cell in (-1, 0, 1):
                for dy_cell in (-1, 0, 1):
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


# ============================================================
# 自适应网格：四叉树 (Python 回退路径)
# ============================================================

class _QuadTreeNode:
    """四叉树节点：密集区域自动细分，稀疏区域保留大格子。"""
    __slots__ = ('x0', 'y0', 'x1', 'y1', 'depth', 'particles', 'children')

    def __init__(self, x0: float, y0: float, x1: float, y1: float, depth: int = 0):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.depth = depth
        self.particles: List[int] = []
        self.children: List[_QuadTreeNode] | None = None

    def _subdivide(self) -> None:
        mx = 0.5 * (self.x0 + self.x1)
        my = 0.5 * (self.y0 + self.y1)
        d1 = self.depth + 1
        self.children = [
            _QuadTreeNode(self.x0, my, mx, self.y1, d1),   # NW
            _QuadTreeNode(mx, my, self.x1, self.y1, d1),    # NE
            _QuadTreeNode(self.x0, self.y0, mx, my, d1),    # SW
            _QuadTreeNode(mx, self.y0, self.x1, my, d1),    # SE
        ]


def _qtree_insert(node: _QuadTreeNode, idx: int, px: float, py: float,
                  xp: np.ndarray, yp: np.ndarray,
                  max_per_leaf: int, min_size: float, max_depth: int) -> None:
    """向四叉树叶子插入颗粒，超阈值时自动细分。"""
    if node.children is not None:
        _qtree_insert_child(node, idx, px, py, xp, yp, max_per_leaf, min_size, max_depth)
        return
    node.particles.append(idx)
    w = node.x1 - node.x0
    h = node.y1 - node.y0
    if (len(node.particles) > max_per_leaf
            and node.depth < max_depth
            and w > 2.0 * min_size
            and h > 2.0 * min_size):
        node._subdivide()
        old = node.particles
        node.particles = []
        for pi in old:
            _qtree_insert_child(node, pi, xp[pi], yp[pi], xp, yp, max_per_leaf, min_size, max_depth)


def _qtree_insert_child(node: _QuadTreeNode, idx: int, px: float, py: float,
                         xp: np.ndarray, yp: np.ndarray,
                         max_per_leaf: int, min_size: float, max_depth: int) -> None:
    mx = 0.5 * (node.x0 + node.x1)
    my = 0.5 * (node.y0 + node.y1)
    if px < mx:
        child = node.children[0] if py >= my else node.children[2]
    else:
        child = node.children[1] if py >= my else node.children[3]
    _qtree_insert(child, idx, px, py, xp, yp, max_per_leaf, min_size, max_depth)


def _qtree_query(node: _QuadTreeNode, qx0: float, qy0: float, qx1: float, qy1: float,
                 result: List[int]) -> None:
    """范围查询：返回所有与查询框重叠的叶子中的颗粒索引。"""
    if qx1 <= node.x0 or qx0 >= node.x1 or qy1 <= node.y0 or qy0 >= node.y1:
        return
    if node.children is None:
        result.extend(node.particles)
    else:
        for child in node.children:
            _qtree_query(child, qx0, qy0, qx1, qy1, result)


def _periodic_query_regions(xi: float, yi: float, thresh: float,
                            Lx: float, Ly: float) -> List[Tuple[float, float, float, float]]:
    """根据周期边界生成查询矩形列表 (x0, y0, x1, y1)。"""
    x_lo, x_hi = xi - thresh, xi + thresh
    if x_hi - x_lo >= Lx:
        x_ranges = [(0.0, Lx)]
    elif x_lo < 0.0:
        x_ranges = [(0.0, x_hi), (x_lo + Lx, Lx)]
    elif x_hi > Lx:
        x_ranges = [(x_lo, Lx), (0.0, x_hi - Lx)]
    else:
        x_ranges = [(x_lo, x_hi)]

    y_lo, y_hi = yi - thresh, yi + thresh
    if y_hi - y_lo >= Ly:
        y_ranges = [(0.0, Ly)]
    elif y_lo < 0.0:
        y_ranges = [(0.0, y_hi), (y_lo + Ly, Ly)]
    elif y_hi > Ly:
        y_ranges = [(y_lo, Ly), (0.0, y_hi - Ly)]
    else:
        y_ranges = [(y_lo, y_hi)]

    return [(xl, yl, xh, yh) for (xl, xh) in x_ranges for (yl, yh) in y_ranges]


def find_contacts_adaptive(x: np.ndarray, y: np.ndarray, prm: NeighborParams,
                           max_per_leaf: int = 8) -> List[Tuple[int, int, float, float, float]]:
    """四叉树自适应邻域搜索（Python 路径，支持周期边界）。

    密集区域自动细分为小格子，稀疏区域保留大格子。
    """
    r, skin = float(prm.r), float(prm.skin)
    thresh = 2.0 * r + max(0.0, skin)
    Lx = float(prm.Lx or 0.0)
    Ly = float(prm.Ly or 0.0)
    periodic = Lx > 0.0 and Ly > 0.0

    n = x.shape[0]
    if n == 0:
        return []

    if periodic:
        xp, yp = _wrap_pos_periodic(x, y, Lx, Ly)
        root = _QuadTreeNode(0.0, 0.0, Lx, Ly)
    else:
        xp, yp = x, y
        x_min = float(np.min(x)) - thresh
        x_max = float(np.max(x)) + thresh
        y_min = float(np.min(y)) - thresh
        y_max = float(np.max(y)) + thresh
        root = _QuadTreeNode(x_min, y_min, x_max, y_max)

    for i in range(n):
        _qtree_insert(root, i, xp[i], yp[i], xp, yp, max_per_leaf, thresh, 20)

    pairs: List[Tuple[int, int, float, float, float]] = []
    for i in range(n):
        xi, yi = float(xp[i]), float(yp[i])

        if periodic:
            regions = _periodic_query_regions(xi, yi, thresh, Lx, Ly)
        else:
            regions = [(xi - thresh, yi - thresh, xi + thresh, yi + thresh)]

        seen: set = set()
        for (qx0, qy0, qx1, qy1) in regions:
            candidates: List[int] = []
            _qtree_query(root, qx0, qy0, qx1, qy1, candidates)
            for j in candidates:
                if j <= i or j in seen:
                    continue
                seen.add(j)
                dx = xp[j] - xi
                dy = yp[j] - yi
                if periodic:
                    dx, dy = _minimal_image(dx, dy, Lx, Ly)
                d = math.hypot(dx, dy)
                if d <= thresh:
                    pairs.append((i, j, dx, dy, d))

    return pairs


# ============================================================
# 自适应网格：空间哈希 (Numba 加速路径)
# ============================================================
# 用哈希表替代固定 Nx×Ny 数组，存储量 O(n) 而非 O(Lx*Ly/cell²)。
# 当域远大于颗粒分布区域时，内存和缓存性能大幅提升。

if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _numba_spatial_hash(gx: int, gy: int, table_size: int) -> int:
        """空间哈希：将格子坐标映射到紧凑表索引。"""
        h = np.int64(gx) * np.int64(73856093) + np.int64(gy) * np.int64(19349663)
        h = h % np.int64(table_size)
        if h < 0:
            h += np.int64(table_size)
        return h

    @njit(cache=True)
    def _numba_build_hash_grid(xp: np.ndarray, yp: np.ndarray,
                                Nx: int, Ny: int, Lx: float, Ly: float,
                                table_size: int):
        """构建空间哈希网格（紧凑存储）。

        使用 Lx/Nx 和 Ly/Ny 做格子索引（与固定网格完全一致），
        仅改变存储结构为 O(n) 的哈希表。
        """
        n = xp.shape[0]
        gx_arr = np.empty(n, dtype=np.int64)
        gy_arr = np.empty(n, dtype=np.int64)
        hashes = np.empty(n, dtype=np.int64)

        # 与固定网格 _numba_build_cell_structure 完全相同的格子尺寸
        cell_x = Lx / Nx
        cell_y = Ly / Ny
        for i in range(n):
            gx = int(xp[i] / cell_x) % Nx
            gy = int(yp[i] / cell_y) % Ny
            gx_arr[i] = gx
            gy_arr[i] = gy
            hashes[i] = _numba_spatial_hash(gx, gy, table_size)

        hash_count = np.zeros(table_size, dtype=np.int64)
        for i in range(n):
            hash_count[hashes[i]] += 1

        hash_start = np.zeros(table_size, dtype=np.int64)
        for h in range(1, table_size):
            hash_start[h] = hash_start[h - 1] + hash_count[h - 1]

        sorted_ids = np.empty(n, dtype=np.int64)
        offsets = np.zeros(table_size, dtype=np.int64)
        for i in range(n):
            h = hashes[i]
            sorted_ids[hash_start[h] + offsets[h]] = i
            offsets[h] += 1

        return hash_start, hash_count, sorted_ids, gx_arr, gy_arr

    @njit(cache=True)
    def _numba_find_contacts_hash(
        xp: np.ndarray, yp: np.ndarray,
        hash_start: np.ndarray, hash_count: np.ndarray, sorted_ids: np.ndarray,
        gx_arr: np.ndarray, gy_arr: np.ndarray,
        Nx: int, Ny: int, Lx: float, Ly: float,
        table_size: int, thresh: float, max_pairs: int
    ):
        """空间哈希邻域搜索（串行，周期边界）。"""
        n = xp.shape[0]
        pair_i = np.empty(max_pairs, dtype=np.int64)
        pair_j = np.empty(max_pairs, dtype=np.int64)
        pair_dx = np.empty(max_pairs, dtype=np.float64)
        pair_dy = np.empty(max_pairs, dtype=np.float64)
        pair_d = np.empty(max_pairs, dtype=np.float64)

        count = 0
        thresh_sq = thresh * thresh

        for i in range(n):
            xi = xp[i]
            yi = yp[i]
            gxi = gx_arr[i]
            gyi = gy_arr[i]

            for dx_cell in range(-1, 2):
                for dy_cell in range(-1, 2):
                    ng_x = (gxi + dx_cell) % Nx
                    ng_y = (gyi + dy_cell) % Ny
                    h = _numba_spatial_hash(ng_x, ng_y, table_size)

                    start = hash_start[h]
                    cnt = hash_count[h]

                    for k in range(start, start + cnt):
                        j = sorted_ids[k]
                        if j <= i:
                            continue
                        # 过滤哈希碰撞：验证实际格子坐标
                        if gx_arr[j] != ng_x or gy_arr[j] != ng_y:
                            continue

                        dxij = xp[j] - xi
                        dyij = yp[j] - yi
                        dxij, dyij = _numba_minimal_image(dxij, dyij, Lx, Ly)

                        d_sq = dxij * dxij + dyij * dyij
                        if d_sq <= thresh_sq:
                            if count < max_pairs:
                                pair_i[count] = i
                                pair_j[count] = j
                                pair_dx[count] = dxij
                                pair_dy[count] = dyij
                                pair_d[count] = math.sqrt(d_sq)
                                count += 1

        return pair_i[:count], pair_j[:count], pair_dx[:count], pair_dy[:count], pair_d[:count]

    @njit(cache=True, parallel=True)
    def _numba_count_neighbors_hash_parallel(
        xp: np.ndarray, yp: np.ndarray,
        hash_start: np.ndarray, hash_count: np.ndarray, sorted_ids: np.ndarray,
        gx_arr: np.ndarray, gy_arr: np.ndarray,
        Nx: int, Ny: int, Lx: float, Ly: float,
        table_size: int, thresh: float
    ):
        """空间哈希并行统计邻居数（第一阶段）。"""
        n = xp.shape[0]
        neighbor_counts = np.zeros(n, dtype=np.int64)
        thresh_sq = thresh * thresh

        for i in prange(n):
            xi = xp[i]
            yi = yp[i]
            gxi = gx_arr[i]
            gyi = gy_arr[i]
            local_count = 0

            for dx_cell in range(-1, 2):
                for dy_cell in range(-1, 2):
                    ng_x = (gxi + dx_cell) % Nx
                    ng_y = (gyi + dy_cell) % Ny
                    h = _numba_spatial_hash(ng_x, ng_y, table_size)

                    start = hash_start[h]
                    cnt = hash_count[h]

                    for k in range(start, start + cnt):
                        j = sorted_ids[k]
                        if j <= i:
                            continue
                        if gx_arr[j] != ng_x or gy_arr[j] != ng_y:
                            continue

                        dxij = xp[j] - xi
                        dyij = yp[j] - yi
                        dxij, dyij = _numba_minimal_image(dxij, dyij, Lx, Ly)

                        d_sq = dxij * dxij + dyij * dyij
                        if d_sq <= thresh_sq:
                            local_count += 1

            neighbor_counts[i] = local_count

        return neighbor_counts

    @njit(cache=True, parallel=True)
    def _numba_find_contacts_hash_parallel(
        xp: np.ndarray, yp: np.ndarray,
        hash_start: np.ndarray, hash_count: np.ndarray, sorted_ids: np.ndarray,
        gx_arr: np.ndarray, gy_arr: np.ndarray,
        Nx: int, Ny: int, Lx: float, Ly: float,
        table_size: int, thresh: float,
        pair_offsets: np.ndarray, total_pairs: int
    ):
        """空间哈希并行填充邻居对（第二阶段）。"""
        n = xp.shape[0]
        pair_i = np.empty(total_pairs, dtype=np.int64)
        pair_j = np.empty(total_pairs, dtype=np.int64)
        pair_dx = np.empty(total_pairs, dtype=np.float64)
        pair_dy = np.empty(total_pairs, dtype=np.float64)
        pair_d = np.empty(total_pairs, dtype=np.float64)

        thresh_sq = thresh * thresh

        for i in prange(n):
            xi = xp[i]
            yi = yp[i]
            gxi = gx_arr[i]
            gyi = gy_arr[i]
            write_pos = pair_offsets[i]

            for dx_cell in range(-1, 2):
                for dy_cell in range(-1, 2):
                    ng_x = (gxi + dx_cell) % Nx
                    ng_y = (gyi + dy_cell) % Ny
                    h = _numba_spatial_hash(ng_x, ng_y, table_size)

                    start = hash_start[h]
                    cnt = hash_count[h]

                    for k in range(start, start + cnt):
                        j = sorted_ids[k]
                        if j <= i:
                            continue
                        if gx_arr[j] != ng_x or gy_arr[j] != ng_y:
                            continue

                        dxij = xp[j] - xi
                        dyij = yp[j] - yi
                        dxij, dyij = _numba_minimal_image(dxij, dyij, Lx, Ly)

                        d_sq = dxij * dxij + dyij * dyij
                        if d_sq <= thresh_sq:
                            pair_i[write_pos] = i
                            pair_j[write_pos] = j
                            pair_dx[write_pos] = dxij
                            pair_dy[write_pos] = dyij
                            pair_d[write_pos] = math.sqrt(d_sq)
                            write_pos += 1

        return pair_i, pair_j, pair_dx, pair_dy, pair_d


def find_contacts_numba_adaptive(x: np.ndarray, y: np.ndarray, prm: NeighborParams,
                                  use_parallel: bool = False) -> List[Tuple[int, int, float, float, float]]:
    """Numba 空间哈希自适应邻域搜索（周期边界）。

    用 O(n) 大小的哈希表替代 O(Nx*Ny) 的固定数组，
    当域远大于颗粒分布区域时大幅节省内存和提升缓存性能。
    """
    if not _NUMBA_AVAILABLE:
        return find_contacts_adaptive(x, y, prm)

    r, skin = float(prm.r), float(prm.skin)
    thresh = 2.0 * r + max(0.0, skin)
    cell = float(prm.cell_size or thresh)
    cell = max(cell, thresh)
    Lx = float(prm.Lx or 0.0)
    Ly = float(prm.Ly or 0.0)

    if Lx <= 0.0 or Ly <= 0.0:
        return find_contacts_adaptive(x, y, prm)

    n = x.shape[0]
    if n == 0:
        return []

    # Nx 用 floor 而非 ceil，保证实际格子尺寸 Lx/Nx >= cell >= thresh
    Nx = max(1, int(Lx / cell))
    Ny = max(1, int(Ly / cell))

    # 哈希表大小与颗粒数成正比，而非域面积
    table_size = max(16, 2 * n + 1)

    xp, yp = _numba_wrap_periodic(x, y, Lx, Ly)
    hash_start, hash_count, sorted_ids, gx_arr, gy_arr = _numba_build_hash_grid(
        xp, yp, Nx, Ny, Lx, Ly, table_size
    )

    if use_parallel:
        neighbor_counts = _numba_count_neighbors_hash_parallel(
            xp, yp, hash_start, hash_count, sorted_ids, gx_arr, gy_arr,
            Nx, Ny, Lx, Ly, table_size, thresh
        )
        pair_offsets = np.zeros(n, dtype=np.int64)
        total_pairs = 0
        for i in range(n):
            pair_offsets[i] = total_pairs
            total_pairs += neighbor_counts[i]
        if total_pairs == 0:
            return []
        pair_i, pair_j, pair_dx, pair_dy, pair_d = _numba_find_contacts_hash_parallel(
            xp, yp, hash_start, hash_count, sorted_ids, gx_arr, gy_arr,
            Nx, Ny, Lx, Ly, table_size, thresh,
            pair_offsets, int(total_pairs)
        )
    else:
        max_pairs = min(n * n // 2, n * 50)
        pair_i, pair_j, pair_dx, pair_dy, pair_d = _numba_find_contacts_hash(
            xp, yp, hash_start, hash_count, sorted_ids, gx_arr, gy_arr,
            Nx, Ny, Lx, Ly, table_size, thresh, max_pairs
        )

    pairs = []
    for k in range(len(pair_i)):
        pairs.append((int(pair_i[k]), int(pair_j[k]), float(pair_dx[k]), float(pair_dy[k]), float(pair_d[k])))
    return pairs
