from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

# 六角轴坐标的 6 个邻居偏移 (dq, ds)
_HEX_DIRS = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]


@dataclass
class ClusterInitParams:
    N1: int
    N2: int
    seed: int | None
    gap_factor: float
    z_target: float = 2.0
    vx_B: float = 0.0
    vy_B: float = 0.0


def _axial_to_xy(q: int, s: int, r: float) -> Tuple[float, float]:
    """六角轴坐标 → 笛卡尔坐标, 相邻格点中心距 = 2r* (pointy-top)."""
    # pointy-top 六角格: 邻居中心距 = a * sqrt(3), 令其 = 2r → a = 2r/sqrt(3)
    d = 2.0 * r
    a = d / math.sqrt(3.0)
    x = a * math.sqrt(3.0) * (q + 0.5 * s)
    y = a * 1.5 * s
    return x, y


def _build_hex_cluster(
    N: int,
    r: float,
    rng: np.random.Generator,
    z_target: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """在六角晶格上构造随机连通团簇, 相邻颗粒中心距固定为 2r.

    算法:
      在六角晶格上以前沿扩展的方式逐个添加格点。每次从所有合法
      前沿候选中, 按 softmax 概率加权随机选择一个(而非贪心选最优),
      权重取决于放入该格点后全局平均配位数与 z_target 的偏差。

    效果:
      - z_target ≈ 2  → 偏树状/链状, 有随机分支, 较疏松
      - z_target ≈ 3~4 → 中等致密, 有较多分叉
      - z_target ≈ 5~6 → 偏块状/六角堆积, 较致密
      - 不同 seed 在同一 z_target 下会生成明显不同的随机结构
    """

    N = int(max(1, N))
    if N == 1:
        return np.array([0.0]), np.array([0.0])

    # ---- 状态初始化 ----
    occupied: dict[tuple[int, int], int] = {(0, 0): 0}   # coord → degree
    coord_list: list[tuple[int, int]] = [(0, 0)]
    edge_count = 0

    # 度数上限: 至少 3, 以允许分支点(树状结构需要 degree≥3 的节点)
    # z_target 较高时进一步放大上限使结构更致密
    deg_max = max(3, int(round(float(z_target))) + 1)

    # 前沿集合: 与已占用格点相邻且未被占用的格点
    frontier: set[tuple[int, int]] = set()
    for dq, ds in _HEX_DIRS:
        frontier.add((dq, ds))

    # softmax 温度
    temperature = 0.5

    def _get_occupied_nbrs(q: int, s: int) -> list[tuple[int, int]]:
        return [(q + dq, s + ds) for dq, ds in _HEX_DIRS
                if (q + dq, s + ds) in occupied]

    # ---- 逐粒子添加 ----
    while len(coord_list) < N:
        valid: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
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

            # ---- 树状优先打分 ----
            # 基础: z_avg 偏差
            new_z_avg = 2.0 * (edge_count + z_new) / (len(coord_list) + 1)
            dev = abs(new_z_avg - z_target)

            if z_target <= 3.0:
                # 低 z_target → 鼓励树状:
                #   (a) 强烈偏好 z_new==1 (纯叶添加, 不产生环)
                #   (b) 在 z_new==1 中, 偏好 parent 的度已经 ≥2 (创建分支点)
                if z_new == 1:
                    parent_deg = occupied[nbrs[0]]
                    # parent 度越高说明已经是分支点, 给一点额外惩罚避免过度集中
                    # parent 度=1 或 2 时给奖励(把叶/链节点变成分支点)
                    branch_bonus = -0.3 if parent_deg >= 1 else 0.0
                    scores.append(dev + branch_bonus)
                else:
                    # z_new>1 意味着创建环/多连接, 大幅惩罚
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
            # 退化: 所有候选都超度数限制, 选邻居数最少的前沿点
            chosen_site = None
            chosen_nbrs: list[tuple[int, int]] = []
            best_cnt = 999
            for site in frontier:
                nbrs = _get_occupied_nbrs(*site)
                if nbrs and len(nbrs) < best_cnt:
                    best_cnt = len(nbrs)
                    chosen_site = site
                    chosen_nbrs = nbrs
            if chosen_site is None:
                q0, s0 = coord_list[0]
                chosen_site = (q0 + 1, s0)
                chosen_nbrs = [coord_list[0]]

        # 放置新粒子
        z_placed = len(chosen_nbrs)
        coord_list.append(chosen_site)
        occupied[chosen_site] = z_placed
        for n in chosen_nbrs:
            occupied[n] += 1
        edge_count += z_placed

        # 更新前沿
        frontier.discard(chosen_site)
        cq, cs = chosen_site
        for dq, ds in _HEX_DIRS:
            nb = (cq + dq, cs + ds)
            if nb not in occupied:
                frontier.add(nb)

    # ---- 坐标转换 & 质心归零 ----
    xs = np.zeros(N, dtype=float)
    ys = np.zeros(N, dtype=float)
    for i, (q, s) in enumerate(coord_list):
        xs[i], ys[i] = _axial_to_xy(q, s, r)
    xs -= xs.mean()
    ys -= ys.mean()

    # 打印统计信息方便调试
    actual_z = 2.0 * edge_count / N
    deg_vals = list(occupied.values())
    print(f"  [Cluster] N={N}, z_target={z_target:.2f}, actual <z>={actual_z:.2f}, "
          f"deg range=[{min(deg_vals)}, {max(deg_vals)}]")

    return xs, ys


def initialize_two_clusters(params: ClusterInitParams, r: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """生成由两团簇 A/B 组成的系统初始状态。

    返回:
        x, y, vx, vy, omega  （粒子顺序：前 N1 个为 A，后 N2 个为 B）
    """

    if params.seed is not None:
        rng = np.random.default_rng(int(params.seed))
    else:
        rng = np.random.default_rng()

    N1 = int(max(1, params.N1))
    N2 = int(max(1, params.N2))
    z_target = float(getattr(params, "z_target", 2.0))

    # 分别在原点附近构造 A、B 团簇形状（随机连通团簇，受 z_target 指导）
    xA, yA = _build_hex_cluster(N1, r, rng, z_target)
    xB, yB = _build_hex_cluster(N2, r, rng, z_target)

    # A 团簇质心严格放在原点（微调数值误差）
    xA -= float(np.mean(xA))
    yA -= float(np.mean(yA))

    # B 团簇质心初始也放在 x 轴上（y≈0）
    xB -= float(np.mean(xB))
    yB -= float(np.mean(yB))

    # 根据“右侧 B 的最左颗粒与 A 的最右颗粒相距 gap_factor * r”的条件，确定 B 的平移量
    xA_max = float(np.max(xA))
    xB_min = float(np.min(xB))
    dx = params.gap_factor * float(r) + xA_max - xB_min
    xB_shifted = xB + dx
    yB_shifted = yB  # 维持在 x 轴附近

    # 拼接两个团簇
    x = np.concatenate([xA, xB_shifted])
    y = np.concatenate([yA, yB_shifted])

    # 速度：A 团簇静止，B 团簇整体以 (vx_B, vy_B) 运动，内部无相对速度
    vxA = np.zeros(N1, dtype=float)
    vyA = np.zeros(N1, dtype=float)
    vxB = np.full(N2, float(params.vx_B), dtype=float)
    vyB = np.full(N2, float(params.vy_B), dtype=float)
    vx = np.concatenate([vxA, vxB])
    vy = np.concatenate([vyA, vyB])

    # 初始自旋全部设为 0
    omega = np.zeros_like(x)

    return x, y, vx, vy, omega
