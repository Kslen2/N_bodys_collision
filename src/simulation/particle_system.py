from __future__ import annotations

from dataclasses import dataclass
import numpy as np

@dataclass
class SystemState:
    x: np.ndarray
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    omega: np.ndarray
    mass: float
    r: float

    def size(self) -> int:
        return int(self.x.size)
