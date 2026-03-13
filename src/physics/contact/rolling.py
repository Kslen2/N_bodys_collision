from __future__ import annotations

import math
from dataclasses import dataclass

@dataclass
class RollingParams:
    mu_r: float  # 相对法向力的滚动阻力系数
    r: float

class RollingResistance:
    def __init__(self, prm: RollingParams) -> None:
        self.p = prm

    def torque_3d(self, Fn_abs: float,
                  omega_x: float, omega_y: float, omega_z: float
                  ) -> tuple[float, float, float]:
        """
        三维滚动阻力矩：M_r = - mu_r * Fn * r * normalize(omega)
        分别返回 (Mx, My, Mz)。
        """
        if Fn_abs <= 0.0:
            return 0.0, 0.0, 0.0
        omega_mag = math.sqrt(omega_x * omega_x + omega_y * omega_y + omega_z * omega_z)
        if omega_mag < 1e-18:
            return 0.0, 0.0, 0.0
        T = self.p.mu_r * Fn_abs * self.p.r
        inv_om = 1.0 / omega_mag
        return (-T * omega_x * inv_om,
                -T * omega_y * inv_om,
                -T * omega_z * inv_om)
