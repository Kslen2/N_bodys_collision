from __future__ import annotations

from dataclasses import dataclass

@dataclass
class RollingParams:
    mu_r: float  # 相对法向力的滚动阻力系数
    r: float

class RollingResistance:
    def __init__(self, prm: RollingParams) -> None:
        self.p = prm

    def torque(self, Fn_abs: float, omega_rel: float) -> float:
        """
        简化滚动阻力矩：M_r = - mu_r * Fn * r * sign(omega_rel)
        """
        if Fn_abs <= 0.0:
            return 0.0
        sgn = 1.0 if omega_rel > 0.0 else -1.0
        return - self.p.mu_r * Fn_abs * self.p.r * sgn
