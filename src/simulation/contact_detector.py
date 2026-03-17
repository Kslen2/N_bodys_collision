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
    Lz: float | None = None


# ============================================================
# Numba 加速版邻域搜索（周期边界 + 3×3 邻域）
# ============================================================

if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _numba_minimal_image(dx: float, dy: float, dz: float,
                             Lx: float, Ly: float, Lz: float):
        """最小镜像位移（三维周期边界）"""
        half_x = 0.5 * Lx
        half_y = 0.5 * Ly
        half_z = 0.5 * Lz
        if dx > half_x:
            dx -= Lx
        elif dx < -half_x:
            dx += Lx
        if dy > half_y:
            dy -= Ly
        elif dy < -half_y:
            dy += Ly
        if dz > half_z:
            dz -= Lz
        elif dz < -half_z:
            dz += Lz
        return dx, dy, dz

    @njit(cache=True, parallel=True)
    def _numba_wrap_periodic(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                             Lx: float, Ly: float, Lz: float):
        """将坐标包裹到 [0, L)（三维）"""
        n = x.shape[0]
        xp = np.empty(n, dtype=np.float64)
        yp = np.empty(n, dtype=np.float64)
        zp = np.empty(n, dtype=np.float64)
        half_Lx = 0.5 * Lx
        half_Ly = 0.5 * Ly
        half_Lz = 0.5 * Lz
        for i in prange(n):
            xp[i] = (x[i] + half_Lx) % Lx
            yp[i] = (y[i] + half_Ly) % Ly
            zp[i] = (z[i] + half_Lz) % Lz
        return xp, yp, zp

    @njit(cache=True)
    def _numba_build_cell_structure(xp: np.ndarray, yp: np.ndarray, zp: np.ndarray,
                                    Nx: int, Ny: int, Nz: int,
                                    Lx: float, Ly: float, Lz: float):
        """构建三维格子结构用于快速邻域查询"""
        n = xp.shape[0]
        num_cells = Nx * Ny * Nz

        cell_indices = np.empty(n, dtype=np.int64)
        cell_x = Lx / Nx
        cell_y = Ly / Ny
        cell_z = Lz / Nz
        for i in range(n):
            gx = int(xp[i] / cell_x) % Nx
            gy = int(yp[i] / cell_y) % Ny
            gz = int(zp[i] / cell_z) % Nz
            cell_indices[i] = gx * (Ny * Nz) + gy * Nz + gz

        cell_count = np.zeros(num_cells, dtype=np.int64)
        for i in range(n):
            cell_count[cell_indices[i]] += 1

        cell_start = np.zeros(num_cells, dtype=np.int64)
        for c in range(1, num_cells):
            cell_start[c] = cell_start[c - 1] + cell_count[c - 1]

        sorted_ids = np.empty(n, dtype=np.int64)
        cell_offset = np.zeros(num_cells, dtype=np.int64)
        for i in range(n):
            c = cell_indices[i]
            sorted_ids[cell_start[c] + cell_offset[c]] = i
            cell_offset[c] += 1

        return cell_indices, cell_start, cell_count, sorted_ids

    @njit(cache=True)
    def _numba_find_contacts_periodic(
        xp: np.ndarray, yp: np.ndarray, zp: np.ndarray,
        cell_start: np.ndarray, cell_count: np.ndarray, sorted_ids: np.ndarray,
        Nx: int, Ny: int, Nz: int,
        Lx: float, Ly: float, Lz: float, thresh: float, max_pairs: int
    ):
        """Numba 邻域搜索（三维周期边界 + 3×3×3 邻域）"""
        n = xp.shape[0]
        pair_i = np.empty(max_pairs, dtype=np.int64)
        pair_j = np.empty(max_pairs, dtype=np.int64)
        pair_dx = np.empty(max_pairs, dtype=np.float64)
        pair_dy = np.empty(max_pairs, dtype=np.float64)
        pair_dz = np.empty(max_pairs, dtype=np.float64)
        pair_d = np.empty(max_pairs, dtype=np.float64)

        count = 0
        thresh_sq = thresh * thresh
        cell_x = Lx / Nx
        cell_y = Ly / Ny
        cell_z = Lz / Nz

        for i in range(n):
            xi = xp[i]
            yi = yp[i]
            zi = zp[i]
            gx_i = int(xi / cell_x) % Nx
            gy_i = int(yi / cell_y) % Ny
            gz_i = int(zi / cell_z) % Nz

            for dx_cell in range(-1, 2):
                for dy_cell in range(-1, 2):
                    for dz_cell in range(-1, 2):
                        ng_x = (gx_i + dx_cell) % Nx
                        ng_y = (gy_i + dy_cell) % Ny
                        ng_z = (gz_i + dz_cell) % Nz
                        cell_idx = ng_x * (Ny * Nz) + ng_y * Nz + ng_z

                        start = cell_start[cell_idx]
                        cnt = cell_count[cell_idx]

                        for k in range(start, start + cnt):
                            j = sorted_ids[k]
                            if j <= i:
                                continue

                            dx = xp[j] - xi
                            dy = yp[j] - yi
                            dz = zp[j] - zi
                            dx, dy, dz = _numba_minimal_image(dx, dy, dz, Lx, Ly, Lz)

                            d_sq = dx * dx + dy * dy + dz * dz
                            if d_sq <= thresh_sq:
                                if count < max_pairs:
                                    pair_i[count] = i
                                    pair_j[count] = j
                                    pair_dx[count] = dx
                                    pair_dy[count] = dy
                                    pair_dz[count] = dz
                                    pair_d[count] = math.sqrt(d_sq)
                                    count += 1

        return pair_i[:count], pair_j[:count], pair_dx[:count], pair_dy[:count], pair_dz[:count], pair_d[:count]

    # ============================================================
    # 并行版本：使用 prange 进行多线程加速
    # ============================================================

    @njit(cache=True, parallel=True)
    def _numba_count_neighbors_parallel(
        xp: np.ndarray, yp: np.ndarray, zp: np.ndarray,
        cell_start: np.ndarray, cell_count: np.ndarray, sorted_ids: np.ndarray,
        Nx: int, Ny: int, Nz: int,
        Lx: float, Ly: float, Lz: float, thresh: float
    ):
        """第一阶段：并行统计每个粒子的邻居数量（3×3×3 邻域）"""
        n = xp.shape[0]
        neighbor_counts = np.zeros(n, dtype=np.int64)
        thresh_sq = thresh * thresh
        cell_x = Lx / Nx
        cell_y = Ly / Ny
        cell_z = Lz / Nz

        for i in prange(n):
            xi = xp[i]
            yi = yp[i]
            zi = zp[i]
            gx_i = int(xi / cell_x) % Nx
            gy_i = int(yi / cell_y) % Ny
            gz_i = int(zi / cell_z) % Nz
            local_count = 0

            for dx_cell in range(-1, 2):
                for dy_cell in range(-1, 2):
                    for dz_cell in range(-1, 2):
                        ng_x = (gx_i + dx_cell) % Nx
                        ng_y = (gy_i + dy_cell) % Ny
                        ng_z = (gz_i + dz_cell) % Nz
                        cell_idx = ng_x * (Ny * Nz) + ng_y * Nz + ng_z

                        start = cell_start[cell_idx]
                        cnt = cell_count[cell_idx]

                        for k in range(start, start + cnt):
                            j = sorted_ids[k]
                            if j <= i:
                                continue

                            dx = xp[j] - xi
                            dy = yp[j] - yi
                            dz = zp[j] - zi
                            dx, dy, dz = _numba_minimal_image(dx, dy, dz, Lx, Ly, Lz)

                            d_sq = dx * dx + dy * dy + dz * dz
                            if d_sq <= thresh_sq:
                                local_count += 1

            neighbor_counts[i] = local_count

        return neighbor_counts

    @njit(cache=True, parallel=True)
    def _numba_find_contacts_parallel(
        xp: np.ndarray, yp: np.ndarray, zp: np.ndarray,
        cell_start: np.ndarray, cell_count: np.ndarray, sorted_ids: np.ndarray,
        Nx: int, Ny: int, Nz: int,
        Lx: float, Ly: float, Lz: float, thresh: float,
        pair_offsets: np.ndarray, total_pairs: int
    ):
        """第二阶段：并行填充邻居对数据（3×3×3 邻域）"""
        n = xp.shape[0]
        pair_i = np.empty(total_pairs, dtype=np.int64)
        pair_j = np.empty(total_pairs, dtype=np.int64)
        pair_dx = np.empty(total_pairs, dtype=np.float64)
        pair_dy = np.empty(total_pairs, dtype=np.float64)
        pair_dz = np.empty(total_pairs, dtype=np.float64)
        pair_d = np.empty(total_pairs, dtype=np.float64)

        thresh_sq = thresh * thresh
        cell_x = Lx / Nx
        cell_y = Ly / Ny
        cell_z = Lz / Nz

        for i in prange(n):
            xi = xp[i]
            yi = yp[i]
            zi = zp[i]
            gx_i = int(xi / cell_x) % Nx
            gy_i = int(yi / cell_y) % Ny
            gz_i = int(zi / cell_z) % Nz
            write_pos = pair_offsets[i]

            for dx_cell in range(-1, 2):
                for dy_cell in range(-1, 2):
                    for dz_cell in range(-1, 2):
                        ng_x = (gx_i + dx_cell) % Nx
                        ng_y = (gy_i + dy_cell) % Ny
                        ng_z = (gz_i + dz_cell) % Nz
                        cell_idx = ng_x * (Ny * Nz) + ng_y * Nz + ng_z

                        start = cell_start[cell_idx]
                        cnt = cell_count[cell_idx]

                        for k in range(start, start + cnt):
                            j = sorted_ids[k]
                            if j <= i:
                                continue

                            dx = xp[j] - xi
                            dy = yp[j] - yi
                            dz = zp[j] - zi
                            dx, dy, dz = _numba_minimal_image(dx, dy, dz, Lx, Ly, Lz)

                            d_sq = dx * dx + dy * dy + dz * dz
                            if d_sq <= thresh_sq:
                                pair_i[write_pos] = i
                                pair_j[write_pos] = j
                                pair_dx[write_pos] = dx
                                pair_dy[write_pos] = dy
                                pair_dz[write_pos] = dz
                                pair_d[write_pos] = math.sqrt(d_sq)
                                write_pos += 1

        return pair_i, pair_j, pair_dx, pair_dy, pair_dz, pair_d


def find_contacts_numba_arrays(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    prm: NeighborParams,
    use_parallel: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Numba 邻域搜索（数组输出，避免构造 Python tuple/list）。"""
    if not _NUMBA_AVAILABLE:
        pairs = find_contacts(x, y, z, prm)
        m = len(pairs)
        if m == 0:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
            )
        pi = np.empty(m, dtype=np.int64)
        pj = np.empty(m, dtype=np.int64)
        pdx = np.empty(m, dtype=np.float64)
        pdy = np.empty(m, dtype=np.float64)
        pdz = np.empty(m, dtype=np.float64)
        pd = np.empty(m, dtype=np.float64)
        for k, (i, j, dx, dy, dz, d) in enumerate(pairs):
            pi[k] = int(i)
            pj[k] = int(j)
            pdx[k] = float(dx)
            pdy[k] = float(dy)
            pdz[k] = float(dz)
            pd[k] = float(d)
        return pi, pj, pdx, pdy, pdz, pd

    r, skin = float(prm.r), float(prm.skin)
    thresh = 2.0 * r + max(0.0, skin)
    cell = float(prm.cell_size or thresh)
    cell = max(cell, thresh)
    Lx = float(prm.Lx or 0.0)
    Ly = float(prm.Ly or 0.0)
    Lz = float(prm.Lz or 0.0)

    if Lx <= 0.0 or Ly <= 0.0 or Lz <= 0.0:
        pairs = find_contacts(x, y, z, prm)
        m = len(pairs)
        if m == 0:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
            )
        pi = np.empty(m, dtype=np.int64)
        pj = np.empty(m, dtype=np.int64)
        pdx = np.empty(m, dtype=np.float64)
        pdy = np.empty(m, dtype=np.float64)
        pdz = np.empty(m, dtype=np.float64)
        pd = np.empty(m, dtype=np.float64)
        for k, (i, j, dx, dy, dz, d) in enumerate(pairs):
            pi[k] = int(i)
            pj[k] = int(j)
            pdx[k] = float(dx)
            pdy[k] = float(dy)
            pdz[k] = float(dz)
            pd[k] = float(d)
        return pi, pj, pdx, pdy, pdz, pd

    n = x.shape[0]
    Nx = max(1, int(Lx / cell))
    Ny = max(1, int(Ly / cell))
    Nz = max(1, int(Lz / cell))

    xp, yp, zp = _numba_wrap_periodic(x, y, z, Lx, Ly, Lz)
    cell_indices, cell_start, cell_count, sorted_ids = _numba_build_cell_structure(xp, yp, zp, Nx, Ny, Nz, Lx, Ly, Lz)

    if use_parallel:
        neighbor_counts = _numba_count_neighbors_parallel(
            xp, yp, zp, cell_start, cell_count, sorted_ids, Nx, Ny, Nz, Lx, Ly, Lz, thresh
        )
        pair_offsets = np.empty(n, dtype=np.int64)
        pair_offsets[0] = 0
        if n > 1:
            np.cumsum(neighbor_counts[:-1], out=pair_offsets[1:])
        total_pairs = int(pair_offsets[-1] + neighbor_counts[-1]) if n > 0 else 0
        if total_pairs == 0:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
            )
        pair_i, pair_j, pair_dx, pair_dy, pair_dz, pair_d = _numba_find_contacts_parallel(
            xp, yp, zp, cell_start, cell_count, sorted_ids, Nx, Ny, Nz, Lx, Ly, Lz, thresh,
            pair_offsets, int(total_pairs)
        )
    else:
        max_pairs = min(n * n // 2, n * 50)
        pair_i, pair_j, pair_dx, pair_dy, pair_dz, pair_d = _numba_find_contacts_periodic(
            xp, yp, zp, cell_start, cell_count, sorted_ids, Nx, Ny, Nz, Lx, Ly, Lz, thresh, max_pairs
        )

    return pair_i, pair_j, pair_dx, pair_dy, pair_dz, pair_d


def find_contacts_numba(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                        prm: NeighborParams, use_parallel: bool = False) -> List[Tuple[int, int, float, float, float, float]]:
    """Numba 加速的三维邻域搜索（周期边界 + 3×3×3 邻域）"""
    pair_i, pair_j, pair_dx, pair_dy, pair_dz, pair_d = find_contacts_numba_arrays(
        x, y, z, prm, use_parallel=use_parallel
    )
    pairs = []
    for k in range(len(pair_i)):
        pairs.append((int(pair_i[k]), int(pair_j[k]), float(pair_dx[k]), float(pair_dy[k]), float(pair_dz[k]), float(pair_d[k])))
    return pairs


# ============================================================
# 原始 Python 版本（三维，作为回退）
# ============================================================

def _wrap_pos_periodic(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                       Lx: float, Ly: float, Lz: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """将坐标包裹到域内 [0, L)"""
    xp = (x + 0.5 * Lx) % Lx
    yp = (y + 0.5 * Ly) % Ly
    zp = (z + 0.5 * Lz) % Lz
    return xp, yp, zp


def build_grid_periodic(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                        cell: float, Lx: float, Ly: float, Lz: float) -> tuple[Dict[Tuple[int, int, int], List[int]], int, int, int]:
    """三维周期网格"""
    Nx = max(1, int(Lx / cell))
    Ny = max(1, int(Ly / cell))
    Nz = max(1, int(Lz / cell))
    xp, yp, zp = _wrap_pos_periodic(x, y, z, Lx, Ly, Lz)
    grid: Dict[Tuple[int, int, int], List[int]] = {}
    for i, (xi, yi, zi) in enumerate(zip(xp, yp, zp)):
        gx = int(math.floor(xi / cell)) % Nx
        gy = int(math.floor(yi / cell)) % Ny
        gz = int(math.floor(zi / cell)) % Nz
        grid.setdefault((gx, gy, gz), []).append(i)
    return grid, Nx, Ny, Nz


def _minimal_image(dx: float, dy: float, dz: float,
                   Lx: float, Ly: float, Lz: float) -> tuple[float, float, float]:
    """最小镜像位移（三维）"""
    if Lx > 0.0 and Ly > 0.0 and Lz > 0.0:
        half_x = 0.5 * Lx
        half_y = 0.5 * Ly
        half_z = 0.5 * Lz
        if dx > half_x:
            dx -= Lx
        elif dx < -half_x:
            dx += Lx
        if dy > half_y:
            dy -= Ly
        elif dy < -half_y:
            dy += Ly
        if dz > half_z:
            dz -= Lz
        elif dz < -half_z:
            dz += Lz
    return dx, dy, dz


def find_contacts(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                  prm: NeighborParams) -> List[Tuple[int, int, float, float, float, float]]:
    """
    三维 Python 版邻域搜索（作为回退）
    返回候选对列表: (i, j, dx, dy, dz, dist)
    """
    r, skin = float(prm.r), float(prm.skin)
    thresh = 2.0 * r + max(0.0, skin)
    cell = float(prm.cell_size or thresh)
    cell = max(cell, thresh)
    Lx = float(prm.Lx or 0.0)
    Ly = float(prm.Ly or 0.0)
    Lz = float(prm.Lz or 0.0)

    if Lx <= 0.0 or Ly <= 0.0 or Lz <= 0.0:
        grid: Dict[Tuple[int, int, int], List[int]] = {}
        for i, (xi, yi, zi) in enumerate(zip(x, y, z)):
            gx = int(math.floor(xi / cell))
            gy = int(math.floor(yi / cell))
            gz = int(math.floor(zi / cell))
            grid.setdefault((gx, gy, gz), []).append(i)
        Nx = Ny = Nz = None
        xp, yp, zp = x, y, z
    else:
        xp, yp, zp = _wrap_pos_periodic(x, y, z, Lx, Ly, Lz)
        grid, Nx, Ny, Nz = build_grid_periodic(x, y, z, cell, Lx, Ly, Lz)

    pairs: List[Tuple[int, int, float, float, float, float]] = []

    for (gx, gy, gz), idxs in grid.items():
        for i in idxs:
            xi = xp[i]
            yi = yp[i]
            zi = zp[i]
            for dx_cell in (-1, 0, 1):
                for dy_cell in (-1, 0, 1):
                    for dz_cell in (-1, 0, 1):
                        if Lx > 0.0 and Ly > 0.0 and Lz > 0.0:
                            ng_x = (gx + dx_cell) % Nx
                            ng_y = (gy + dy_cell) % Ny
                            ng_z = (gz + dz_cell) % Nz
                            neigh = grid.get((ng_x, ng_y, ng_z))
                        else:
                            neigh = grid.get((gx + dx_cell, gy + dy_cell, gz + dz_cell))
                        if not neigh:
                            continue
                        for j in neigh:
                            if j <= i:
                                continue
                            dx = xp[j] - xi
                            dy = yp[j] - yi
                            dz = zp[j] - zi
                            if Lx > 0.0 and Ly > 0.0 and Lz > 0.0:
                                dx, dy, dz = _minimal_image(dx, dy, dz, Lx, Ly, Lz)
                            d = math.sqrt(dx * dx + dy * dy + dz * dz)
                            if d <= thresh:
                                pairs.append((i, j, dx, dy, dz, d))
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
# 自适应网格：空间哈希 (Numba 加速路径，三维)
# ============================================================

if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _numba_spatial_hash(gx: int, gy: int, gz: int, table_size: int) -> int:
        """三维空间哈希"""
        h = np.int64(gx) * np.int64(73856093) + np.int64(gy) * np.int64(19349663) + np.int64(gz) * np.int64(83492791)
        h = h % np.int64(table_size)
        if h < 0:
            h += np.int64(table_size)
        return h

    @njit(cache=True, parallel=True)
    def _numba_build_hash_grid(xp: np.ndarray, yp: np.ndarray, zp: np.ndarray,
                                Nx: int, Ny: int, Nz: int,
                                Lx: float, Ly: float, Lz: float,
                                table_size: int):
        n = xp.shape[0]
        gx_arr = np.empty(n, dtype=np.int64)
        gy_arr = np.empty(n, dtype=np.int64)
        gz_arr = np.empty(n, dtype=np.int64)
        hashes = np.empty(n, dtype=np.int64)
        cell_x = Lx / Nx
        cell_y = Ly / Ny
        cell_z = Lz / Nz
        # 第1遍：cell assignment（并行，每个 i 写独立位置，无竞争）
        for i in prange(n):
            gx = int(xp[i] / cell_x) % Nx
            gy = int(yp[i] / cell_y) % Ny
            gz = int(zp[i] / cell_z) % Nz
            gx_arr[i] = gx
            gy_arr[i] = gy
            gz_arr[i] = gz
            hashes[i] = _numba_spatial_hash(gx, gy, gz, table_size)
        # 第2遍~第4遍：counting sort（串行，有数据依赖）
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
        return hash_start, hash_count, sorted_ids, gx_arr, gy_arr, gz_arr

    @njit(cache=True)
    def _numba_find_contacts_hash(
        xp, yp, zp, hash_start, hash_count, sorted_ids,
        gx_arr, gy_arr, gz_arr,
        Nx, Ny, Nz, Lx, Ly, Lz,
        table_size, thresh, max_pairs
    ):
        n = xp.shape[0]
        pair_i = np.empty(max_pairs, dtype=np.int64)
        pair_j = np.empty(max_pairs, dtype=np.int64)
        pair_dx = np.empty(max_pairs, dtype=np.float64)
        pair_dy = np.empty(max_pairs, dtype=np.float64)
        pair_dz = np.empty(max_pairs, dtype=np.float64)
        pair_d = np.empty(max_pairs, dtype=np.float64)
        count = 0
        thresh_sq = thresh * thresh
        for i in range(n):
            xi = xp[i]; yi = yp[i]; zi = zp[i]
            gxi = gx_arr[i]; gyi = gy_arr[i]; gzi = gz_arr[i]
            for dx_cell in range(-1, 2):
                for dy_cell in range(-1, 2):
                    for dz_cell in range(-1, 2):
                        ng_x = (gxi + dx_cell) % Nx
                        ng_y = (gyi + dy_cell) % Ny
                        ng_z = (gzi + dz_cell) % Nz
                        h = _numba_spatial_hash(ng_x, ng_y, ng_z, table_size)
                        start = hash_start[h]
                        cnt = hash_count[h]
                        for k in range(start, start + cnt):
                            j = sorted_ids[k]
                            if j <= i:
                                continue
                            if gx_arr[j] != ng_x or gy_arr[j] != ng_y or gz_arr[j] != ng_z:
                                continue
                            dxij = xp[j] - xi
                            dyij = yp[j] - yi
                            dzij = zp[j] - zi
                            dxij, dyij, dzij = _numba_minimal_image(dxij, dyij, dzij, Lx, Ly, Lz)
                            d_sq = dxij*dxij + dyij*dyij + dzij*dzij
                            if d_sq <= thresh_sq:
                                if count < max_pairs:
                                    pair_i[count] = i
                                    pair_j[count] = j
                                    pair_dx[count] = dxij
                                    pair_dy[count] = dyij
                                    pair_dz[count] = dzij
                                    pair_d[count] = math.sqrt(d_sq)
                                    count += 1
        return pair_i[:count], pair_j[:count], pair_dx[:count], pair_dy[:count], pair_dz[:count], pair_d[:count]

    @njit(cache=True, parallel=True)
    def _numba_count_neighbors_hash_parallel(
        xp, yp, zp, hash_start, hash_count, sorted_ids,
        gx_arr, gy_arr, gz_arr,
        Nx, Ny, Nz, Lx, Ly, Lz,
        table_size, thresh
    ):
        n = xp.shape[0]
        neighbor_counts = np.zeros(n, dtype=np.int64)
        thresh_sq = thresh * thresh
        for i in prange(n):
            xi = xp[i]; yi = yp[i]; zi = zp[i]
            gxi = gx_arr[i]; gyi = gy_arr[i]; gzi = gz_arr[i]
            local_count = 0
            for dx_cell in range(-1, 2):
                for dy_cell in range(-1, 2):
                    for dz_cell in range(-1, 2):
                        ng_x = (gxi + dx_cell) % Nx
                        ng_y = (gyi + dy_cell) % Ny
                        ng_z = (gzi + dz_cell) % Nz
                        h = _numba_spatial_hash(ng_x, ng_y, ng_z, table_size)
                        start = hash_start[h]
                        cnt = hash_count[h]
                        for k in range(start, start + cnt):
                            j = sorted_ids[k]
                            if j <= i:
                                continue
                            if gx_arr[j] != ng_x or gy_arr[j] != ng_y or gz_arr[j] != ng_z:
                                continue
                            dxij = xp[j] - xi
                            dyij = yp[j] - yi
                            dzij = zp[j] - zi
                            dxij, dyij, dzij = _numba_minimal_image(dxij, dyij, dzij, Lx, Ly, Lz)
                            d_sq = dxij*dxij + dyij*dyij + dzij*dzij
                            if d_sq <= thresh_sq:
                                local_count += 1
            neighbor_counts[i] = local_count
        return neighbor_counts

    @njit(cache=True, parallel=True)
    def _numba_find_contacts_hash_parallel(
        xp, yp, zp, hash_start, hash_count, sorted_ids,
        gx_arr, gy_arr, gz_arr,
        Nx, Ny, Nz, Lx, Ly, Lz,
        table_size, thresh,
        pair_offsets, total_pairs
    ):
        n = xp.shape[0]
        pair_i = np.empty(total_pairs, dtype=np.int64)
        pair_j = np.empty(total_pairs, dtype=np.int64)
        pair_dx = np.empty(total_pairs, dtype=np.float64)
        pair_dy = np.empty(total_pairs, dtype=np.float64)
        pair_dz = np.empty(total_pairs, dtype=np.float64)
        pair_d = np.empty(total_pairs, dtype=np.float64)
        thresh_sq = thresh * thresh
        for i in prange(n):
            xi = xp[i]; yi = yp[i]; zi = zp[i]
            gxi = gx_arr[i]; gyi = gy_arr[i]; gzi = gz_arr[i]
            write_pos = pair_offsets[i]
            for dx_cell in range(-1, 2):
                for dy_cell in range(-1, 2):
                    for dz_cell in range(-1, 2):
                        ng_x = (gxi + dx_cell) % Nx
                        ng_y = (gyi + dy_cell) % Ny
                        ng_z = (gzi + dz_cell) % Nz
                        h = _numba_spatial_hash(ng_x, ng_y, ng_z, table_size)
                        start = hash_start[h]
                        cnt = hash_count[h]
                        for k in range(start, start + cnt):
                            j = sorted_ids[k]
                            if j <= i:
                                continue
                            if gx_arr[j] != ng_x or gy_arr[j] != ng_y or gz_arr[j] != ng_z:
                                continue
                            dxij = xp[j] - xi
                            dyij = yp[j] - yi
                            dzij = zp[j] - zi
                            dxij, dyij, dzij = _numba_minimal_image(dxij, dyij, dzij, Lx, Ly, Lz)
                            d_sq = dxij*dxij + dyij*dyij + dzij*dzij
                            if d_sq <= thresh_sq:
                                pair_i[write_pos] = i
                                pair_j[write_pos] = j
                                pair_dx[write_pos] = dxij
                                pair_dy[write_pos] = dyij
                                pair_dz[write_pos] = dzij
                                pair_d[write_pos] = math.sqrt(d_sq)
                                write_pos += 1
        return pair_i, pair_j, pair_dx, pair_dy, pair_dz, pair_d


def find_contacts_adaptive(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                           prm: NeighborParams) -> List[Tuple[int, int, float, float, float, float]]:
    """三维自适应邻域搜索（回退到基本网格搜索）"""
    return find_contacts(x, y, z, prm)


def find_contacts_numba_adaptive_arrays(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    prm: NeighborParams,
    use_parallel: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Numba 空间哈希邻域搜索（数组输出，避免构造 Python tuple/list）。"""
    if not _NUMBA_AVAILABLE:
        pairs = find_contacts(x, y, z, prm)
        m = len(pairs)
        if m == 0:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
            )
        pi = np.empty(m, dtype=np.int64)
        pj = np.empty(m, dtype=np.int64)
        pdx = np.empty(m, dtype=np.float64)
        pdy = np.empty(m, dtype=np.float64)
        pdz = np.empty(m, dtype=np.float64)
        pd = np.empty(m, dtype=np.float64)
        for k, (i, j, dx, dy, dz, d) in enumerate(pairs):
            pi[k] = int(i)
            pj[k] = int(j)
            pdx[k] = float(dx)
            pdy[k] = float(dy)
            pdz[k] = float(dz)
            pd[k] = float(d)
        return pi, pj, pdx, pdy, pdz, pd
    r, skin = float(prm.r), float(prm.skin)
    thresh = 2.0 * r + max(0.0, skin)
    cell = float(prm.cell_size or thresh)
    cell = max(cell, thresh)
    Lx = float(prm.Lx or 0.0)
    Ly = float(prm.Ly or 0.0)
    Lz = float(prm.Lz or 0.0)
    if Lx <= 0.0 or Ly <= 0.0 or Lz <= 0.0:
        pairs = find_contacts(x, y, z, prm)
        m = len(pairs)
        if m == 0:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
            )
        pi = np.empty(m, dtype=np.int64)
        pj = np.empty(m, dtype=np.int64)
        pdx = np.empty(m, dtype=np.float64)
        pdy = np.empty(m, dtype=np.float64)
        pdz = np.empty(m, dtype=np.float64)
        pd = np.empty(m, dtype=np.float64)
        for k, (i, j, dx, dy, dz, d) in enumerate(pairs):
            pi[k] = int(i)
            pj[k] = int(j)
            pdx[k] = float(dx)
            pdy[k] = float(dy)
            pdz[k] = float(dz)
            pd[k] = float(d)
        return pi, pj, pdx, pdy, pdz, pd
    n = x.shape[0]
    if n == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    Nx = max(1, int(Lx / cell))
    Ny = max(1, int(Ly / cell))
    Nz = max(1, int(Lz / cell))
    table_size = max(16, 2 * n + 1)
    xp, yp, zp = _numba_wrap_periodic(x, y, z, Lx, Ly, Lz)
    hash_start, hash_count, sorted_ids, gx_arr, gy_arr, gz_arr = _numba_build_hash_grid(
        xp, yp, zp, Nx, Ny, Nz, Lx, Ly, Lz, table_size
    )
    if use_parallel:
        neighbor_counts = _numba_count_neighbors_hash_parallel(
            xp, yp, zp, hash_start, hash_count, sorted_ids, gx_arr, gy_arr, gz_arr,
            Nx, Ny, Nz, Lx, Ly, Lz, table_size, thresh
        )
        pair_offsets = np.empty(n, dtype=np.int64)
        pair_offsets[0] = 0
        if n > 1:
            np.cumsum(neighbor_counts[:-1], out=pair_offsets[1:])
        total_pairs = int(pair_offsets[-1] + neighbor_counts[-1]) if n > 0 else 0
        if total_pairs == 0:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
            )
        pi, pj, pdx, pdy, pdz, pd = _numba_find_contacts_hash_parallel(
            xp, yp, zp, hash_start, hash_count, sorted_ids, gx_arr, gy_arr, gz_arr,
            Nx, Ny, Nz, Lx, Ly, Lz, table_size, thresh,
            pair_offsets, int(total_pairs)
        )
    else:
        max_pairs = min(n * n // 2, n * 50)
        pi, pj, pdx, pdy, pdz, pd = _numba_find_contacts_hash(
            xp, yp, zp, hash_start, hash_count, sorted_ids, gx_arr, gy_arr, gz_arr,
            Nx, Ny, Nz, Lx, Ly, Lz, table_size, thresh, max_pairs
        )
    return pi, pj, pdx, pdy, pdz, pd


def find_contacts_numba_adaptive(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                                  prm: NeighborParams,
                                  use_parallel: bool = False) -> List[Tuple[int, int, float, float, float, float]]:
    """Numba 三维空间哈希自适应邻域搜索（周期边界）"""
    pi, pj, pdx, pdy, pdz, pd = find_contacts_numba_adaptive_arrays(
        x, y, z, prm, use_parallel=use_parallel
    )
    pairs = []
    for k in range(len(pi)):
        pairs.append((int(pi[k]), int(pj[k]), float(pdx[k]), float(pdy[k]), float(pdz[k]), float(pd[k])))
    return pairs
