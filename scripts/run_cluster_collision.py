from __future__ import annotations

import os
import sys
import argparse
import time
import math
import concurrent.futures
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 确保项目根目录加入 sys.path，便于脚本直接运行
_HERE = os.path.abspath(os.path.dirname(__file__))
_PKG_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from src.physics.materials import Materials
from src.initialization.cluster import ClusterInitParams, initialize_two_clusters
from src.simulation.particle_system import SystemState
from src.simulation.simulator import DEMSimulator, SimParams


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run A/B cluster collision simulation")
    p.add_argument("--config", type=str, default=os.path.join(_PKG_ROOT, "configs", "cluster_collision.yaml"))
    # 团簇参数
    # 若不提供，则从 YAML 中的 cluster 段读取
    p.add_argument("--N1", type=int, default=None, help="Number of particles in cluster A (override YAML if set)")
    p.add_argument("--N2", type=int, default=None, help="Number of particles in cluster B (override YAML if set)")
    p.add_argument("--seed", type=int, default=None, help="Seed controlling cluster internal structure (override YAML)")
    p.add_argument("--gap_factor", type=float, default=None, help="Initial gap between clusters in units of particle radius (override YAML)")
    # B 团簇质心速度
    p.add_argument("--vxB", type=float, default=None, help="Initial x-velocity of cluster B (override YAML, m/s)")
    p.add_argument("--vyB", type=float, default=None, help="Initial y-velocity of cluster B (override YAML, m/s)")
    # 运行与输出控制
    p.add_argument("--t_end", type=float, default=None, help="Override total simulation time (s)")
    p.add_argument("--snapshot_interval", type=float, default=None, help="Override snapshot interval (s)")
    p.add_argument("--snapshot_dir", type=str, default=None, help="Override snapshot output directory")
    p.add_argument("--numba", action="store_true", help="Enable Numba acceleration if available")
    # 两颗粒碰撞判据（仅在 N1=N2=1 时使用），可被 YAML 中 cluster 段提供的值覆盖
    p.add_argument("--adhesion_vn_thresh", type=float, default=None,
                   help="Normal relative velocity threshold for adhesion (override YAML, m/s)")
    p.add_argument("--adhesion_steps", type=int, default=None,
                   help="Number of consecutive steps below threshold to declare adhesion (override YAML)")
    return p.parse_args()


def _compute_jkr_alpha_f(materials: Materials) -> float:
    """按 EDEM 标准 JKR 公式计算临界重叠量 alpha_f。

    simulator 中 Numba 核心使用的同一公式：
        F_c = (3/2) π Γ R*
        alpha_f = (3 F_c^2 / (16 R* E*^2))^(1/3)
    """

    r = float(materials.r)
    gamma = float(materials.gamma)
    E_star = float(materials.E_star)
    R_eff = 0.5 * r
    F_c = 1.5 * math.pi * gamma * R_eff
    alpha_f = ((3.0 * F_c * F_c) / (16.0 * R_eff * E_star * E_star)) ** (1.0 / 3.0)
    return alpha_f


def _run_two_body_with_detection(
    runner: DEMSimulator,
    materials: Materials,
    sim_params: SimParams,
    adhesion_vn_thresh: float,
    adhesion_steps: int,
) -> None:
    """针对 N1=N2=1（两颗粒）情况，显式追踪碰撞、分离和黏附。

    结果写入 snapshot_dir/two_body_result.txt。
    """

    r = float(materials.r)
    alpha_f = _compute_jkr_alpha_f(materials)

    # 只考虑粒子 0 (A) 与 1 (B)
    iA, iB = 0, 1

    def _relative_geom_and_vn() -> tuple[float, float]:
        """返回 (距离 d, 相对法向速度 v_rel_n)，考虑周期最小镜像。"""

        dx = float(runner.state.x[iB] - runner.state.x[iA])
        dy = float(runner.state.y[iB] - runner.state.y[iA])
        dx, dy = runner._minimal_image(dx, dy)  # type: ignore[attr-defined]
        d = math.sqrt(dx * dx + dy * dy)
        if d <= 1e-18:
            # 极端重合情况下，法向方向不确定，返回 0
            return d, 0.0
        nx = dx / d
        ny = dy / d
        dvx = float(runner.state.vx[iB] - runner.state.vx[iA])
        dvy = float(runner.state.vy[iB] - runner.state.vy[iA])
        v_rel_n = dvx * nx + dvy * ny
        return d, v_rel_n

    snap_dt = float(sim_params.snapshot_interval or 0.0)
    next_snap = snap_dt if snap_dt > 0.0 else float("inf")
    snap_idx = 0

    # 报告与初始快照
    runner.report.open()
    if snap_dt > 0.0:
        runner.saver.save(
            runner.state,
            0.0,
            runner.Lx,
            runner.Ly,
            os.path.join(sim_params.snapshot_dir, "snap_000000_t0.000e+00.png"),
        )
        KE_t0, KE_r0, KE_sum0 = _energies(runner)
        vmax0 = float(np.max(np.sqrt(runner.state.vx ** 2 + runner.state.vy ** 2)))
        runner.report.write_step(0.0, KE_t0, KE_r0, KE_sum0, vmax0)

    t = 0.0
    step = 0
    start_time = time.perf_counter()

    has_collided = False
    stable_counter = 0
    status = "running"
    event_time = None
    event_vn = None

    while t < sim_params.t_end:
        # 基于当前状态判断几何关系和相对法向速度
        d, v_rel_n = _relative_geom_and_vn()
        delta_n = 2.0 * r - d

        if (not has_collided) and (delta_n >= 0.0):
            has_collided = True

        if has_collided:
            # 分离判据：δ_n ≤ -alpha_f
            if delta_n <= -alpha_f:
                status = "separated"
                event_time = t
                event_vn = v_rel_n
                break
            # 黏附判据：在接触/近接阶段，连续很多步 |v_n| 小于阈值
            if abs(v_rel_n) < adhesion_vn_thresh:
                stable_counter += 1
            else:
                stable_counter = 0
            if stable_counter >= adhesion_steps:
                status = "adhesive"
                event_time = t
                event_vn = 0.0
                break

        # 计算受力并推进一步
        fx, fy, tau, _max_abs_vn, _min_approach = runner._compute_forces()

        # 快照与能量报告（与 DEMSimulator.run 的顺序保持一致：在积分前写出当前时刻）
        if t >= next_snap:
            snap_idx += 1
            fname = f"snap_{snap_idx:06d}_t{t:.3e}.png"
            runner.saver.save(
                runner.state,
                t,
                runner.Lx,
                runner.Ly,
                os.path.join(sim_params.snapshot_dir, fname),
            )
            KE_t, KE_r, KE_sum = _energies(runner)
            vmax = float(np.max(np.sqrt(runner.state.vx ** 2 + runner.state.vy ** 2)))
            runner.report.write_step(t, KE_t, KE_r, KE_sum, vmax)
            next_snap += snap_dt

        runner._integrate(fx, fy, tau)
        runner._apply_periodic_wrap()

        t += sim_params.dt
        step += 1

    # 结束后的处理与结果记录
    if not has_collided:
        status = "no_collision"
        event_time = None
        event_vn = None
    elif status == "running":
        # 已发生碰撞但在 t_end 内既未明显分离也未满足黏附判据
        status = "inconclusive"
        event_time = t
        _d_last, v_rel_n_last = _relative_geom_and_vn()
        event_vn = v_rel_n_last

    # 最终快照与报告收尾
    runner.saver.save(
        runner.state,
        t,
        runner.Lx,
        runner.Ly,
        os.path.join(sim_params.snapshot_dir, "final_snapshot.png"),
    )
    KE_tf, KE_rf, KE_sumf = _energies(runner)
    vmaxf = float(np.max(np.sqrt(runner.state.vx ** 2 + runner.state.vy ** 2)))
    runner.report.write_step(t, KE_tf, KE_rf, KE_sumf, vmaxf)
    elapsed = time.perf_counter() - start_time
    runner.report.write_runtime(elapsed)
    runner.report.close()

    # 写出两体碰撞结果
    os.makedirs(sim_params.snapshot_dir, exist_ok=True)
    result_path = os.path.join(sim_params.snapshot_dir, "two_body_result.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        # status 字段用于程序后处理：
        #   no_collision : 在给定模拟时间内未发生碰撞
        #   separated    : 发生碰撞后再次分离（记录分离瞬间的相对法向速度）
        #   adhesive     : 发生碰撞且满足黏附判据（相对法向速度视为 0）
        #   inconclusive : 已碰撞，但在 t_end 之前既未明显分离也未满足黏附判据
        f.write(f"status\t{status}\n")
        # 额外给出一行中文说明，方便直接阅读结果
        if status == "no_collision":
            f.write("status_desc\t未发生碰撞\n")
        elif status == "separated":
            f.write("status_desc\t碰撞后分离\n")
        elif status == "adhesive":
            f.write("status_desc\t碰撞后黏附\n")
        else:
            f.write("status_desc\t已碰撞但未明确分离/黏附\n")
        if event_time is not None:
            f.write(f"t_event(s)\t{event_time:.6e}\n")
        else:
            f.write("t_event(s)\tNA\n")
        if event_vn is not None:
            f.write(f"v_rel_n(m/s)\t{event_vn:.6e}\n")
        else:
            f.write("v_rel_n(m/s)\tNA\n")
        f.write("# v_rel_n > 0 表示 B 沿着连线方向远离 A 的法向速度。\n")


def _simulate_two_body_case(
    materials: Materials,
    sim_cfg: dict,
    base_cluster: ClusterInitParams,
    vxB: float,
    vyB: float,
    dt: float,
    t_end: float,
    adhesion_vn_thresh: float,
    adhesion_steps: int,
) -> tuple[str, float | None]:
    """单次两颗粒碰撞模拟（无任何快照/报告输出），返回 (status, event_vn)。"""

    # 若初始速度全为 0，显然无法发生接触，直接返回以节省计算
    if abs(vxB) <= 0.0 and abs(vyB) <= 0.0:
        return "no_collision", None

    # 构造当前速度下的团簇参数
    cluster_params = ClusterInitParams(
        N1=base_cluster.N1,
        N2=base_cluster.N2,
        seed=base_cluster.seed,
        gap_factor=base_cluster.gap_factor,
        z_target=base_cluster.z_target,
        vx_B=float(vxB),
        vy_B=float(vyB),
    )

    x, y, vx, vy, omega = initialize_two_clusters(cluster_params, materials.r)
    mass = materials.mass
    state = SystemState(x=x, y=y, vx=vx, vy=vy, omega=omega, mass=mass, r=materials.r)

    # 盒子大小：按当前粒子包围盒估算并留出裕量
    x_min = float(np.min(x)); x_max = float(np.max(x))
    y_min = float(np.min(y)); y_max = float(np.max(y))
    span_x = x_max - x_min; span_y = y_max - y_min
    margin = 10.0 * float(materials.r)
    Lx = max(span_x + margin, 20.0 * float(materials.r))
    Ly = max(span_y + margin, 20.0 * float(materials.r))

    snapshot_dir = os.path.abspath(sim_cfg.get("snapshot_dir", "data/cluster_collision"))

    sim_params = SimParams(
        dt=float(dt),
        t_end=float(t_end),
        skin_factor=float(sim_cfg.get("skin_factor", 0.2)),
        snapshot_interval=0.0,      # 扫描模式下不输出过程快照
        save_stride=1,
        snapshot_dir=snapshot_dir,
        domain_scale=1.0,
        Lx=Lx,
        Ly=Ly,
        use_numba=bool(sim_cfg.get("use_numba", False)),
        use_numba_parallel=bool(sim_cfg.get("use_numba_parallel", False)),
        snapshot_show_velocity=False,
        use_cpp=bool(sim_cfg.get("use_cpp", False)),
        adaptive_timestep=False,
        dt_large_factor=float(sim_cfg.get("dt_large_factor", 20.0)),
        vn_threshold=float(sim_cfg.get("vn_threshold", 1e-4)),
        stable_steps_required=int(sim_cfg.get("stable_steps_required", 50)),
        approach_time_factor=float(sim_cfg.get("approach_time_factor", 2.0)),
        use_neighbor_cache=bool(sim_cfg.get("use_neighbor_cache", False)),
        use_jkr_interp=bool(sim_cfg.get("use_jkr_interp", False)),
        jkr_interp_points=int(sim_cfg.get("jkr_interp_points", 512)),
        jkr_interp_alpha_factor=float(sim_cfg.get("jkr_interp_alpha_factor", 5.0)),
    )

    runner = DEMSimulator(materials, sim_params, state)

    r = float(materials.r)
    alpha_f = _compute_jkr_alpha_f(materials)
    iA, iB = 0, 1

    def _relative_geom_and_vn() -> tuple[float, float]:
        # 扫描模式下不使用周期最小镜像，直接在物理空间计算相对几何量
        dx = float(runner.state.x[iB] - runner.state.x[iA])
        dy = float(runner.state.y[iB] - runner.state.y[iA])
        d = math.sqrt(dx * dx + dy * dy)
        if d <= 1e-18:
            return d, 0.0
        nx = dx / d; ny = dy / d
        dvx = float(runner.state.vx[iB] - runner.state.vx[iA])
        dvy = float(runner.state.vy[iB] - runner.state.vy[iA])
        v_rel_n = dvx * nx + dvy * ny
        return d, v_rel_n

    t = 0.0
    has_collided = False
    stable_counter = 0
    status = "running"
    event_vn: float | None = None

    # 扫描模式下：无周期边界。以 [-Lx/2, Lx/2]×[-Ly/2, Ly/2] 为物理边界，
    # 若粒子在尚未发生碰撞前离开边界，则视为“未碰撞”并提前结束。
    half_x = 0.5 * runner.Lx
    half_y = 0.5 * runner.Ly

    while t < sim_params.t_end:
        d, v_rel_n = _relative_geom_and_vn()
        delta_n = 2.0 * r - d

        if (not has_collided) and (delta_n >= 0.0):
            has_collided = True

        if has_collided:
            if delta_n <= -alpha_f:
                status = "separated"
                event_vn = v_rel_n
                break
            if abs(v_rel_n) < adhesion_vn_thresh:
                stable_counter += 1
            else:
                stable_counter = 0
            if stable_counter >= adhesion_steps:
                status = "adhesive"
                event_vn = 0.0
                break

        fx, fy, tau, _max_abs_vn, _min_approach = runner._compute_forces()
        runner._integrate(fx, fy, tau)

        # 边界判定：不开启周期边界，若尚未发生碰撞且粒子越界，则视为未碰撞
        if not has_collided:
            out_x_low = np.any(runner.state.x < -half_x)
            out_x_high = np.any(runner.state.x > half_x)
            out_y_low = np.any(runner.state.y < -half_y)
            out_y_high = np.any(runner.state.y > half_y)
            if out_x_low or out_x_high or out_y_low or out_y_high:
                status = "no_collision"
                event_vn = None
                break

        t += sim_params.dt

    if not has_collided:
        status = "no_collision"
        event_vn = None
    elif status == "running":
        status = "inconclusive"
        _d_last, v_rel_n_last = _relative_geom_and_vn()
        event_vn = v_rel_n_last

    return status, event_vn


def _scan_worker(
    args: tuple[
        Materials,
        dict,
        ClusterInitParams,
        float,
        float,
        float,
        float,
        float,
        int,
        int,
    ]
) -> tuple[int, int, str, float | None]:
    """并行扫描单个速度点的辅助函数。

    返回 (iy, ix, status, event_vn)，由主进程汇总到网格。
    """

    (
        materials,
        sim_cfg,
        base_cluster,
        vxB,
        vyB,
        dt,
        t_end,
        adhesion_vn_thresh,
        adhesion_steps,
        ix,
        iy,
    ) = args

    status, event_vn = _simulate_two_body_case(
        materials,
        sim_cfg,
        base_cluster,
        vxB,
        vyB,
        dt,
        t_end,
        adhesion_vn_thresh,
        adhesion_steps,
    )
    return iy, ix, status, event_vn


def _scan_velocity_heatmap(
    materials: Materials,
    sim_cfg: dict,
    cluster_cfg: dict,
    base_cluster: ClusterInitParams,
    dt: float,
    t_end: float,
) -> None:
    """在 (|Vx|, Vy) 网格上扫描两颗粒碰撞结果，并输出一张热力图。"""

    vx_min = float(cluster_cfg.get("vx_min", 0.0))
    vx_max = float(cluster_cfg.get("vx_max", 1.0))
    nx = int(cluster_cfg.get("vx_steps", 10))
    vy_min = float(cluster_cfg.get("vy_min", 0.0))
    vy_max = float(cluster_cfg.get("vy_max", 1.0))
    ny = int(cluster_cfg.get("vy_steps", 10))
    heatmap_filename = str(cluster_cfg.get("heatmap_filename", "vn_heatmap.png"))

    adhesion_vn_thresh = float(cluster_cfg.get("adhesion_vn_thresh", 1.0e-4))
    adhesion_steps = int(cluster_cfg.get("adhesion_steps", 50))

    vx_vals = np.linspace(vx_min, vx_max, nx)
    vy_vals = np.linspace(vy_min, vy_max, ny)

    Z = np.full((ny, nx), np.nan, dtype=float)

    # 构造所有任务列表：每个网格点一个速度组合
    tasks: list[tuple[Materials, dict, ClusterInitParams, float, float, float, float, float, int, int]] = []
    for iy, vy in enumerate(vy_vals):
        for ix, vx_abs in enumerate(vx_vals):
            vxB = -float(vx_abs)  # B 从右向左运动：取 vxB 为负，横轴用 |Vx|
            vyB = float(vy)
            tasks.append(
                (
                    materials,
                    sim_cfg,
                    base_cluster,
                    vxB,
                    vyB,
                    dt,
                    t_end,
                    adhesion_vn_thresh,
                    adhesion_steps,
                    ix,
                    iy,
                )
            )

    total = len(tasks)
    if total == 0:
        return

    print(f"[SCAN] 共 {total} 个速度组合，开始并行扫描……")

    # 并行执行，主进程汇总结果并打印进度条
    done = 0
    next_pct = 0.0
    bar_width = 30

    def _print_progress(done_count: int) -> None:
        ratio = done_count / total
        filled = int(bar_width * ratio)
        bar = "#" * filled + "-" * (bar_width - filled)
        sys.stdout.write(f"\r[SCAN] 进度: {done_count}/{total} ({ratio*100:5.1f}%) [{bar}]")
        sys.stdout.flush()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(_scan_worker, task) for task in tasks]
        for fut in concurrent.futures.as_completed(futures):
            iy, ix, status, event_vn = fut.result()
            if status != "no_collision" and event_vn is not None:
                Z[iy, ix] = abs(float(event_vn))

            done += 1
            pct = done * 100.0 / total
            if pct >= next_pct:
                _print_progress(done)
                next_pct += 5.0

    # 最终保证进度条到 100%
    _print_progress(total)
    print()  # 换行

    base_dir = os.path.abspath(sim_cfg.get("snapshot_dir", "data/cluster_collision"))
    os.makedirs(base_dir, exist_ok=True)
    out_path = os.path.join(base_dir, heatmap_filename)

    VX, VY = np.meshgrid(vx_vals, vy_vals)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(6, 5))
    pcm = ax.pcolormesh(VX, VY, Z, cmap=cmap, shading="auto")
    ax.set_xlabel("|Vx| (m/s)")
    ax.set_ylabel("Vy (m/s)")
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("|v_n| at outcome (m/s)")
    ax.set_title("Two-particle collision outcome")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _energies(runner: DEMSimulator) -> tuple[float, float, float]:
    m = runner.state.mass
    KE_trans = 0.5 * m * float(np.sum(runner.state.vx ** 2 + runner.state.vy ** 2))
    I = 0.4 * m * (runner.state.r ** 2)
    KE_rot = 0.5 * I * float(np.sum(runner.state.omega ** 2))
    return KE_trans, KE_rot, KE_trans + KE_rot


def _vn_gamma_scan_worker(
    args: tuple[
        Materials,
        dict,
        ClusterInitParams,
        float,
        float,
        float,
        float,
        float,
        int,
        int,
    ]
) -> tuple[int, int, str, float | None]:
    """(Vn, gamma) 网格并行扫描单个点的辅助函数。

    返回 (ig, ivn, status, event_vn)，由主进程汇总到网格。
    """

    (
        base_materials,
        sim_cfg,
        base_cluster,
        vn,
        gamma,
        dt,
        t_end,
        adhesion_vn_thresh,
        adhesion_steps,
        ivn,
        ig,
    ) = args

    # 基于给定 gamma 构造局部材料参数
    materials = Materials(
        r=float(base_materials.r),
        rho=float(base_materials.rho),
        E=float(base_materials.E),
        nu=float(base_materials.nu),
        gamma=float(gamma),
        MU_S=float(base_materials.MU_S),
        MU_K=float(base_materials.MU_K),
        e_n=float(base_materials.e_n),
        mu_r=float(base_materials.mu_r),
        c_rot=float(base_materials.c_rot),
    )

    # 仅法向入射：B 从右向左，Vy=0
    vxB = -float(vn)
    vyB = 0.0

    status, event_vn = _simulate_two_body_case(
        materials,
        sim_cfg,
        base_cluster,
        vxB,
        vyB,
        dt,
        t_end,
        adhesion_vn_thresh,
        adhesion_steps,
    )
    return ig, ivn, status, event_vn


def _scan_vn_gamma_heatmap(
    materials: Materials,
    sim_cfg: dict,
    cluster_cfg: dict,
    base_cluster: ClusterInitParams,
    dt: float,
    t_end: float,
) -> None:
    """在 (Vn, gamma) 网格上扫描两颗粒碰撞结果，并输出一张热力图。

    横坐标为法向入射速度 Vn，纵坐标为表面能 gamma，
    相图值为碰撞后法向相对速度大小 |v_n|。
    """

    vn_min = float(cluster_cfg.get("vn_min", 0.0))
    vn_max = float(cluster_cfg.get("vn_max", 1.0))
    nv = int(cluster_cfg.get("vn_steps", 10))
    gamma_min = float(cluster_cfg.get("gamma_min", materials.gamma))
    gamma_max = float(cluster_cfg.get("gamma_max", materials.gamma))
    ng = int(cluster_cfg.get("gamma_steps", 5))
    heatmap_filename = str(cluster_cfg.get("vn_gamma_heatmap_filename", "vn_gamma_heatmap.png"))

    adhesion_vn_thresh = float(cluster_cfg.get("adhesion_vn_thresh", 1.0e-4))
    adhesion_steps = int(cluster_cfg.get("adhesion_steps", 50))

    vn_vals = np.linspace(vn_min, vn_max, nv)
    gamma_vals = np.linspace(gamma_min, gamma_max, ng)

    Z = np.full((ng, nv), np.nan, dtype=float)

    # 构造所有任务列表：每个网格点一个 (Vn, gamma)
    tasks: list[tuple[Materials, dict, ClusterInitParams, float, float, float, float, float, int, int]] = []
    for ig, gamma in enumerate(gamma_vals):
        for ivn, vn in enumerate(vn_vals):
            tasks.append(
                (
                    materials,
                    sim_cfg,
                    base_cluster,
                    float(vn),
                    float(gamma),
                    dt,
                    t_end,
                    adhesion_vn_thresh,
                    adhesion_steps,
                    ivn,
                    ig,
                )
            )

    total = len(tasks)
    if total == 0:
        return

    print(f"[SCAN Vn-gamma] 共 {total} 个 (Vn, gamma) 组合，开始并行扫描……")

    # 并行执行，主进程汇总结果并打印进度条
    done = 0
    bar_width = 30

    def _print_progress(done_count: int) -> None:
        ratio = done_count / total
        filled = int(bar_width * ratio)
        bar = "#" * filled + "-" * (bar_width - filled)
        sys.stdout.write(f"\r[SCAN Vn-gamma] 进度: {done_count}/{total} ({ratio*100:5.1f}%) [{bar}]")
        sys.stdout.flush()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(_vn_gamma_scan_worker, task) for task in tasks]
        for fut in concurrent.futures.as_completed(futures):
            ig, ivn, status, event_vn = fut.result()
            if status != "no_collision" and event_vn is not None:
                Z[ig, ivn] = abs(float(event_vn))

            done += 1
            _print_progress(done)

    # 最终保证进度条到 100%
    _print_progress(total)
    print()  # 换行

    base_dir = os.path.abspath(sim_cfg.get("snapshot_dir", "data/cluster_collision"))
    os.makedirs(base_dir, exist_ok=True)
    out_path = os.path.join(base_dir, heatmap_filename)

    VN, G = np.meshgrid(vn_vals, gamma_vals)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(6, 5))
    pcm = ax.pcolormesh(VN, G, Z, cmap=cmap, shading="auto")
    ax.set_xlabel("Vn (m/s)")
    ax.set_ylabel("gamma (J/m^2)")
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("|v_n| at outcome (m/s)")
    ax.set_title("Two-particle collision outcome in (Vn, gamma)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    cfg_path = os.path.abspath(args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    mtl_cfg = cfg.get("materials", {})
    sim_cfg = cfg.get("simulation", {})
    cluster_cfg = cfg.get("cluster", {})

    materials = Materials(
        r=float(mtl_cfg["r"]),
        rho=float(mtl_cfg["rho"]),
        E=float(mtl_cfg["E"]),
        nu=float(mtl_cfg["nu"]),
        gamma=float(mtl_cfg["gamma"]),
        MU_S=float(mtl_cfg["MU_S"]),
        MU_K=float(mtl_cfg["MU_K"]),
        e_n=float(mtl_cfg["e_n"]),
        mu_r=float(mtl_cfg["mu_r"]),
        c_rot=float(mtl_cfg["c_rot"]),
    )

    # ----- 初始化两团簇：参数优先来自命令行，其次 YAML 的 cluster 段 -----
    def _get_int(name: str, arg_val: int | None, default_cfg: dict, default_key: str, fallback: int) -> int:
        if arg_val is not None:
            return int(arg_val)
        if default_key in default_cfg and default_cfg[default_key] is not None:
            return int(default_cfg[default_key])
        return int(fallback)

    def _get_float(name: str, arg_val: float | None, default_cfg: dict, default_key: str, fallback: float) -> float:
        if arg_val is not None:
            return float(arg_val)
        if default_key in default_cfg and default_cfg[default_key] is not None:
            return float(default_cfg[default_key])
        return float(fallback)

    N1_val = _get_int("N1", args.N1, cluster_cfg, "N1", 1)
    N2_val = _get_int("N2", args.N2, cluster_cfg, "N2", 1)
    seed_val = _get_int("seed", args.seed, cluster_cfg, "seed", 0)
    gap_val = _get_float("gap_factor", args.gap_factor, cluster_cfg, "gap_factor", 1.0)
    z_target_val = float(cluster_cfg.get("z_target", 2.0))
    vxB_val = _get_float("vxB", args.vxB, cluster_cfg, "vxB", -1.0)
    vyB_val = _get_float("vyB", args.vyB, cluster_cfg, "vyB", 0.0)

    cluster_params = ClusterInitParams(
        N1=N1_val,
        N2=N2_val,
        seed=seed_val,
        gap_factor=gap_val,
        z_target=z_target_val,
        vx_B=vxB_val,
        vy_B=vyB_val,
    )

    # ----- 仿真时间参数（供单次 & 扫描共用）-----
    dt = float(sim_cfg["dt"])
    t_end = float(args.t_end) if args.t_end is not None else float(sim_cfg["t_end"])

    # 若启用扫描模式且为两颗粒，则直接进行相应扫描并输出热力图
    scan_enabled = bool(cluster_cfg.get("scan_enabled", False))
    vn_gamma_scan_enabled = bool(cluster_cfg.get("vn_gamma_scan_enabled", False))
    if (N1_val == 1) and (N2_val == 1):
        if scan_enabled:
            _scan_velocity_heatmap(materials, sim_cfg, cluster_cfg, cluster_params, dt, t_end)
            return
        if vn_gamma_scan_enabled:
            _scan_vn_gamma_heatmap(materials, sim_cfg, cluster_cfg, cluster_params, dt, t_end)
            return

    # ----- 常规模拟：初始化两团簇并建立系统状态 -----
    x, y, vx, vy, omega = initialize_two_clusters(cluster_params, materials.r)
    mass = materials.mass
    state = SystemState(x=x, y=y, vx=vx, vy=vy, omega=omega, mass=mass, r=materials.r)

    # ----- 盒子大小：按当前粒子包围盒估算并留出裕量 -----
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    span_x = x_max - x_min
    span_y = y_max - y_min
    margin = 10.0 * float(materials.r)
    Lx = max(span_x + margin, 20.0 * float(materials.r))
    Ly = max(span_y + margin, 20.0 * float(materials.r))

    # ----- 仿真参数 -----
    snapshot_interval = (
        float(args.snapshot_interval)
        if args.snapshot_interval is not None
        else float(sim_cfg.get("snapshot_interval", 0.0))
    )
    save_stride = int(sim_cfg.get("save_stride", 1))
    base_snap_dir = str(sim_cfg.get("snapshot_dir", "data/cluster_collision"))
    if args.snapshot_dir is not None:
        snapshot_dir = os.path.abspath(args.snapshot_dir)
    else:
        # 在基础目录下创建一个按 N1/N2/seed 区分的子目录，避免与其他模拟冲突
        sub = f"N1_{N1_val}_N2_{N2_val}_seed{seed_val}_gap{gap_val:.1f}_vxB{vxB_val:.1f}_vyB{vyB_val:.1f}"
        snapshot_dir = os.path.abspath(os.path.join(base_snap_dir, sub))

    sim_params = SimParams(
        dt=dt,
        t_end=t_end,
        skin_factor=float(sim_cfg.get("skin_factor", 0.2)),
        snapshot_interval=snapshot_interval,
        save_stride=save_stride,
        snapshot_dir=snapshot_dir,
        domain_scale=1.0,
        Lx=Lx,
        Ly=Ly,
        use_numba=bool(sim_cfg.get("use_numba", args.numba)),
        use_numba_parallel=bool(sim_cfg.get("use_numba_parallel", False)),
        snapshot_show_velocity=bool(sim_cfg.get("snapshot_show_velocity", True)),
        use_cpp=bool(sim_cfg.get("use_cpp", False)),
        adaptive_timestep=False,  # 对于本脚本，默认关闭自适应步长与团簇冻结
        dt_large_factor=float(sim_cfg.get("dt_large_factor", 20.0)),
        vn_threshold=float(sim_cfg.get("vn_threshold", 1e-4)),
        stable_steps_required=int(sim_cfg.get("stable_steps_required", 50)),
        approach_time_factor=float(sim_cfg.get("approach_time_factor", 2.0)),
        use_neighbor_cache=bool(sim_cfg.get("use_neighbor_cache", False)),
        use_jkr_interp=bool(sim_cfg.get("use_jkr_interp", False)),
        jkr_interp_points=int(sim_cfg.get("jkr_interp_points", 512)),
        jkr_interp_alpha_factor=float(sim_cfg.get("jkr_interp_alpha_factor", 5.0)),
    )

    runner = DEMSimulator(materials, sim_params, state)

    # 两颗粒碰撞判据参数：命令行优先，其次 YAML cluster 段，最后使用函数内缺省
    adhesion_vn_thresh = _get_float(
        "adhesion_vn_thresh", args.adhesion_vn_thresh, cluster_cfg, "adhesion_vn_thresh", 1.0e-4
    )
    adhesion_steps = _get_int(
        "adhesion_steps", args.adhesion_steps, cluster_cfg, "adhesion_steps", 50
    )

    # 如果是两颗粒（N1=N2=1），使用专门的碰撞判据与结果输出
    if int(N1_val) == 1 and int(N2_val) == 1 and state.size() == 2:
        _run_two_body_with_detection(
            runner,
            materials,
            sim_params,
            adhesion_vn_thresh=float(adhesion_vn_thresh),
            adhesion_steps=int(adhesion_steps),
        )
    else:
        # 多粒子或多团簇情形：直接使用通用 DEM 模拟器
        runner.run()


if __name__ == "__main__":
    main()
