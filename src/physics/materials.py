from __future__ import annotations

from dataclasses import dataclass

@dataclass
class Materials:
    r: float
    rho: float
    E: float
    nu: float
    gamma: float
    MU_S: float
    MU_K: float
    e_n: float  # 法向恢复系数（0<e_n<1），用于 Tsuji 阻尼公式
    mu_r: float
    c_rot: float

    @property
    def mass(self) -> float:
        # 切换为 3D 球质量以与 two_cluster 对齐：m = (4/3) π r^3 ρ
        import math
        return (4.0 / 3.0) * math.pi * (self.r ** 3) * self.rho

    @property
    def E_star(self) -> float:
        # 等效模量（相同材料、相同半径粒子）
        return self.E / (2.0 * (1.0 - self.nu * self.nu))

    @property
    def G_star(self) -> float:
        """等效剪切模量 G*（相同材料、相同半径粒子）。

        对于两相同各向同性弹性球体，
        1/G* = 2(2-ν)/G,  G = E/[2(1+ν)]
        → G* = E / (4 (2-ν) (1+ν))
        """
        return self.E / (4.0 * (2.0 - self.nu) * (1.0 + self.nu))
