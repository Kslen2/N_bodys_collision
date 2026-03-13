from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np

# FCC 晶格 12 个最近邻偏移 (di, dj, dk)，其中 i+j+k 保持奇偶性
_FCC_DIRS = [
    (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
    (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
    (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
]


@dataclass
class ClusterInitParams:
    N1: int
    N2: int
    seed: int | None
    gap_factor: float
    z_target: float = 2.0
    vx_B: float = 0.0
    vy_B: float = 0.0
    vz_B: float = 0.0


def _fcc_to_xyz(i: int, j: int, k: int, r: float) -> Tuple[float, float, float]:
    """FCC 晶格整数坐标 → 笛卡尔坐标，相邻格点中心距 = 2r。"""
    a = r * math.sqrt(2.0)
    return (i * a, j * a, k * a)


def _build_fcc_cluster(
    N: int,
    r: float,
    rng: np.random.Generator,
    z_target: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在 FCC 晶格上构造随机连通团簇，相邻颗粒中心距固定为 2r。"""

    N = int(max(1, N))
    if N == 1:
        return np.array([0.0]), np.array([0.0]), np.array([0.0])

    occupied: dict[tuple[int, int, int], int] = {(0, 0, 0): 0}
    coord_list: list[tuple[int, int, int]] = [(0, 0, 0)]
    edge_count = 0

    deg_max = max(3, int(round(float(z_target))) + 1)

    frontier: set[tuple[int, int, int]] = set()
    for di, dj, dk in _FCC_DIRS:
        frontier.add((di, dj, dk))

    temperature = 0.5

    def _get_occupied_nbrs(ci: int, cj: int, ck: int) -> list[tuple[int, int, int]]:
        return [(ci + di, cj + dj, ck + dk) for di, dj, dk in _FCC_DIRS
                if (ci + di, cj + dj, ck + dk) in occupied]

    while len(coord_list) < N:
        valid: list[tuple[tuple[int, int, int], list[tuple[int, int, int]]]] = []
        scores: list[float] = []

        for site in frontier:
            nbrs = _get_occupied_nbrs(*site)
            z_new = len(nbrs)
            if z_new < 1:
                continue
            if z_new > deg_max:
                continue
            if any(occupied[n] + 1 > deg_max for n in nbrs):
                continue

            valid.append((site, nbrs))

            new_z_avg = 2.0 * (edge_count + z_new) / (len(coord_list) + 1)
            dev = abs(new_z_avg - z_target)

            if z_target <= 3.0:
                if z_new == 1:
                    parent_deg = occupied[nbrs[0]]
                    branch_bonus = -0.3 if parent_deg >= 1 else 0.0
                    scores.append(dev + branch_bonus)
                else:
                    scores.append(dev + 2.0)
            else:
                scores.append(dev)

        if valid:
            sc = np.array(scores)
            weights = np.exp(-sc / temperature)
            weights /= weights.sum()
            idx = int(rng.choice(len(valid), p=weights))
            chosen_site, chosen_nbrs = valid[idx]
        else:
            chosen_site = None
            chosen_nbrs: list[tuple[int, int, int]] = []
            best_cnt = 999
            for site in frontier:
                nbrs = _get_occupied_nbrs(*site)
                if nbrs and len(nbrs) < best_cnt:
                    best_cnt = len(nbrs)
                    chosen_site = site
                    chosen_nbrs = nbrs
            if chosen_site is None:
                i0, j0, k0 = coord_list[0]
                chosen_site = (i0 + 1, j0 + 1, k0)
                chosen_nbrs = [coord_list[0]]

        z_placed = len(chosen_nbrs)
        coord_list.append(chosen_site)
        occupied[chosen_site] = z_placed
        for n in chosen_nbrs:
            occupied[n] += 1
        edge_count += z_placed

        frontier.discard(chosen_site)
        ci, cj, ck = chosen_site
        for di, dj, dk in _FCC_DIRS:
            nb = (ci + di, cj + dj, ck + dk)
            if nb not in occupied:
                frontier.add(nb)

    xs = np.zeros(N, dtype=float)
    ys = np.zeros(N, dtype=float)
    zs = np.zeros(N, dtype=float)
    for idx, (ci, cj, ck) in enumerate(coord_list):
        xs[idx], ys[idx], zs[idx] = _fcc_to_xyz(ci, cj, ck, r)
    xs -= xs.mean()
    ys -= ys.mean()
    zs -= zs.mean()

    actual_z = 2.0 * edge_count / N
    deg_vals = list(occupied.values())
    print(f"  [Cluster] N={N}, z_target={z_target:.2f}, actual <z>={actual_z:.2f}, "
          f"deg range=[{min(deg_vals)}, {max(deg_vals)}]")

    return xs, ys, zs


def initialize_two_clusters(params: ClusterInitParams, r: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                           np.ndarray, np.ndarray, np.ndarray,
                                                                           np.ndarray, np.ndarray, np.ndarray]:
    """生成由两团簇 A/B 组成的三维系统初始状态。

    返回:
        x, y, z, vx, vy, vz, omega_x, omega_y, omega_z
        （粒子顺序：前 N1 个为 A，后 N2 个为 B）
    """

    if params.seed is not None:
        rng = np.random.default_rng(int(params.seed))
    else:
        rng = np.random.default_rng()

    N1 = int(max(1, params.N1))
    N2 = int(max(1, params.N2))
    z_target = float(getattr(params, "z_target", 2.0))

    xA, yA, zA = _build_fcc_cluster(N1, r, rng, z_target)
    xB, yB, zB = _build_fcc_cluster(N2, r, rng, z_target)

    xA -= float(np.mean(xA))
    yA -= float(np.mean(yA))
    zA -= float(np.mean(zA))

    xB -= float(np.mean(xB))
    yB -= float(np.mean(yB))
    zB -= float(np.mean(zB))

    # 根据“右侧 B 的最左颗粒与 A 的最右颗粒相距 gap_factor * r”的条件，确定 B 的平移量
    xA_max = float(np.max(xA))
    xB_min = float(np.min(xB))
    dx = params.gap_factor * float(r) + xA_max - xB_min
    xB_shifted = xB + dx
    yB_shifted = yB
    zB_shifted = zB

    x = np.concatenate([xA, xB_shifted])
    y = np.concatenate([yA, yB_shifted])
    z = np.concatenate([zA, zB_shifted])

    vxA = np.zeros(N1, dtype=float)
    vyA = np.zeros(N1, dtype=float)
    vzA = np.zeros(N1, dtype=float)
    vxB = np.full(N2, float(params.vx_B), dtype=float)
    vyB = np.full(N2, float(params.vy_B), dtype=float)
    vzB = np.full(N2, float(params.vz_B), dtype=float)
    vx = np.concatenate([vxA, vxB])
    vy = np.concatenate([vyA, vyB])
    vz = np.concatenate([vzA, vzB])

    omega_x = np.zeros_like(x)
    omega_y = np.zeros_like(x)
    omega_z = np.zeros_like(x)

    return x, y, z, vx, vy, vz, omega_x, omega_y, omega_z
