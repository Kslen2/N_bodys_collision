from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class InitParams:
    N: int
    h: float
    vmin: float
    vmax: float
    seed: int | None = None


def initialize_grid(params: InitParams, r: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    生成 N×N 网格的初始粒子：每个网格中心放一粒，速度幅值均匀随机于 [vmin, vmax]，方向均匀随机。

    Returns:
        x, y, vx, vy, omega 数组（均为 float64）。omega 初始为 0。
    """
    if params.seed is not None:
        random.seed(params.seed)
        np.random.seed(params.seed)

    N = int(max(1, params.N))
    h = float(params.h)
    L = N * h
    half = 0.5 * L
    x0 = -half + 0.5 * h
    y0 = -half + 0.5 * h

    xs: List[float] = []
    ys: List[float] = []
    vxs: List[float] = []
    vys: List[float] = []

    for iy in range(N):
        cy = y0 + iy * h
        for ix in range(N):
            cx = x0 + ix * h
            xs.append(cx)
            ys.append(cy)
            theta = random.random() * 2.0 * math.pi
            vmag = random.uniform(params.vmin, params.vmax)
            vxs.append(vmag * math.cos(theta))
            vys.append(vmag * math.sin(theta))

    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    vx = np.array(vxs, dtype=float)
    vy = np.array(vys, dtype=float)
    omega = np.zeros_like(x)
    return x, y, vx, vy, omega
