from __future__ import annotations

import os
import sys
import argparse
import yaml
import numpy as np

# 确保项目根目录加入 sys.path，便于脚本直接运行
_HERE = os.path.abspath(os.path.dirname(__file__))
_PKG_ROOT = os.path.abspath(os.path.join(_HERE, ".."))           # dust_collision_DEM/
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from src.physics.materials import Materials
from src.initialization.grid import InitParams, initialize_grid
from src.simulation.particle_system import SystemState
from src.simulation.simulator import DEMSimulator, SimParams


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run multibody full-DEM simulation (JKR)")
    p.add_argument("--config", type=str, default=os.path.join(_PKG_ROOT, "configs", "default.yaml"))
    p.add_argument("--numba", action="store_true", help="Enable Numba acceleration if available")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = os.path.abspath(args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    mtl = cfg.get("materials", {})
    ini = cfg.get("initialization", {})
    sim = cfg.get("simulation", {})

    materials = Materials(
        r=float(mtl["r"]), rho=float(mtl["rho"]), E=float(mtl["E"]), nu=float(mtl["nu"]), gamma=float(mtl["gamma"]),
        MU_S=float(mtl["MU_S"]), MU_K=float(mtl["MU_K"]), e_n=float(mtl["e_n"]), mu_r=float(mtl["mu_r"]), c_rot=float(mtl["c_rot"]),
    )

    init_params = InitParams(
        N=int(ini["N"]), h=float(ini["h"]), vmin=float(ini["vmin"]), vmax=float(ini["vmax"]), seed=int(ini.get("seed", 0))
    )

    x, y, vx, vy, omega = initialize_grid(init_params, materials.r)
    mass = materials.mass
    state = SystemState(x=x, y=y, vx=vx, vy=vy, omega=omega, mass=mass, r=materials.r)

    # 盒子严格设为 N*h（可用 domain_scale 额外缩放，默认 1.0）
    L = float(init_params.N) * float(init_params.h) * float(sim.get("domain_scale", 1.0))
    sim_params = SimParams(
        dt=float(sim["dt"]), t_end=float(sim["t_end"]), skin_factor=float(sim.get("skin_factor", 0.2)),
        snapshot_interval=float(sim.get("snapshot_interval", 0.0)), save_stride=int(sim.get("save_stride", 1)),
        snapshot_dir=str(sim.get("snapshot_dir", "data/snapshot_saver")), domain_scale=float(sim.get("domain_scale", 1.0)),
        Lx=L, Ly=L,
        use_numba=bool(sim.get("use_numba", args.numba)),
        use_numba_parallel=bool(sim.get("use_numba_parallel", False)),
        snapshot_show_velocity=bool(sim.get("snapshot_show_velocity", True)),
        snapshot_show_dt_factor=bool(sim.get("snapshot_show_dt_factor", False)),
        use_cpp=bool(sim.get("use_cpp", False)),
        # 自适应时间步长（按全局最大速度分档）
        adaptive_timestep=bool(sim.get("adaptive_timestep", False)),
        adaptive_dt_factors=tuple(float(v) for v in sim.get("adaptive_dt_factors", [1.0, 5.0, 10.0, 20.0, 50.0])),
        adaptive_vmax_ratio_thresholds=tuple(float(v) for v in sim.get("adaptive_vmax_ratio_thresholds", [0.2, 0.1, 0.05, 0.02])),
        use_neighbor_cache=bool(sim.get("use_neighbor_cache", False)),
        use_jkr_interp=bool(sim.get("use_jkr_interp", False)),
        jkr_interp_points=int(sim.get("jkr_interp_points", 512)),
        jkr_interp_alpha_factor=float(sim.get("jkr_interp_alpha_factor", 5.0)),
    )

    runner = DEMSimulator(materials, sim_params, state)
    runner.run()


if __name__ == "__main__":
    main()
