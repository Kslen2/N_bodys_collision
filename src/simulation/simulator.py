from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
import os
import sys
import numpy as np
import time
import math as _math

# 可选：Numba 加速
try:
    import numba as _nb
    from numba import njit as _njit, prange as _prange
    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False

# 可选：C++ pybind11 扩展
try:
    import dem_kernels as _cppkern
    _CPP_AVAILABLE = True
except Exception:
    _CPP_AVAILABLE = False

# 允许直接运行本文件时仍能导入依赖：优先相对导入，失败则注入项目根后用绝对导入
try:
    from ..physics.materials import Materials
    from ..physics.contact.jkr_normal import JKRParams, JKRNormal
    from ..physics.contact.tangential import TangentialParams, TangentialFriction
    from ..physics.contact.rolling import RollingParams, RollingResistance
    from .particle_system import SystemState
    from .contact_detector import NeighborParams, find_contacts, find_contacts_numba, connected_components_from_contacts
    from ..io.snapshot_saver import SnapshotSaver
    from ..visualization.report_generator import ReportWriter
except ImportError:
    _HERE = os.path.abspath(os.path.dirname(__file__))
    _PKG_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))  # multibody_collision_DEM/src → multibody_collision_DEM
    _PROJECT_ROOT = os.path.abspath(os.path.join(_PKG_ROOT, ".."))  # dust_collision_model
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    from multibody_collision_DEM.src.physics.materials import Materials
    from multibody_collision_DEM.src.physics.contact.jkr_normal import JKRParams, JKRNormal
    from multibody_collision_DEM.src.physics.contact.tangential import TangentialParams, TangentialFriction
    from multibody_collision_DEM.src.physics.contact.rolling import RollingParams, RollingResistance
    from multibody_collision_DEM.src.simulation.particle_system import SystemState
    from multibody_collision_DEM.src.simulation.contact_detector import NeighborParams, find_contacts, find_contacts_numba, connected_components_from_contacts
    from multibody_collision_DEM.src.io.snapshot_saver import SnapshotSaver
    from multibody_collision_DEM.src.visualization.report_generator import ReportWriter


@dataclass
class SimParams:
    dt: float
    t_end: float
    skin_factor: float
    snapshot_interval: float
    save_stride: int
    snapshot_dir: str
    domain_scale: float = 1.0
    Lx: float | None = None
    Ly: float | None = None
    # 是否启用 Numba 加速（需安装 numba）
    use_numba: bool = False
    # 是否启用 Numba 多线程并行（prange）
    use_numba_parallel: bool = False
    # 快照是否绘制速度信息（速度箭头 + 最快颗粒标注）
    snapshot_show_velocity: bool = True
    # 快照左下角是否显示当前时间步长倍率
    snapshot_show_dt_factor: bool = False
    # 是否启用 C++ pybind11 内核（需先构建并安装 dem_kernels）
    use_cpp: bool = False
    # ---------- 自适应时间步长参数（按全局最大速度平滑缩放） ----------
    adaptive_timestep: bool = False  # 是否启用按全局最大速度平滑缩放的自适应时间步长
    adaptive_dt_min_factor: float = 1.0
    adaptive_dt_max_factor: float = 50.0
    adaptive_vmax_ema_alpha: float = 0.2  # v_approx = alpha*v_now + (1-alpha)*v_approx
    # 邻域缓存（Verlet 邻居）开关
    use_neighbor_cache: bool = False
    # JKR 法向力是否使用插值近似（替代牛顿迭代）
    use_jkr_interp: bool = False
    # JKR 插值表参数：点数与上界因子（alpha_max = factor * a0）
    jkr_interp_points: int = 512
    jkr_interp_alpha_factor: float = 5.0


class DEMSimulator:
    def __init__(self, materials: Materials, sim: SimParams, state: SystemState) -> None:
        self.mat = materials
        self.sim = sim
        self.state = state
        # domain: 优先使用外部传入的 Lx/Ly，否则按位置估算并乘以 domain_scale
        if (self.sim.Lx is not None) and (self.sim.Ly is not None):
            self.Lx = float(self.sim.Lx)
            self.Ly = float(self.sim.Ly)
        else:
            L_est = self._estimate_box_length()
            self.Lx = self.sim.domain_scale * L_est
            self.Ly = self.sim.domain_scale * L_est
        # neighbor/search（需已知 Lx/Ly）
        self.skin = float(sim.skin_factor) * self.mat.r
        self.neigh = NeighborParams(r=self.mat.r, skin=self.skin, cell_size=None, Lx=self.Lx, Ly=self.Ly)
        # contact models
        self.jkr = JKRNormal(JKRParams(E_star=self.mat.E_star, gamma=self.mat.gamma, r=self.mat.r, e_n=self.mat.e_n))
        # 根据配置选择是否在 JKR 模型中使用插值近似
        try:
            self.jkr.use_interp = bool(getattr(self.sim, "use_jkr_interp", False))
        except Exception:
            self.jkr.use_interp = False

        # 为 Numba 力内核准备 JKR 插值表（若开启 use_jkr_interp 且启用 Numba）
        self._jkr_interp_use = bool(getattr(self.sim, "use_jkr_interp", False)) and bool(getattr(self.sim, "use_numba", False)) and _NUMBA_AVAILABLE
        if self._jkr_interp_use:
            # 懒构建 JKR 内部插值表，然后转为 numpy 数组供 Numba 使用
            if not getattr(self.jkr, "_interp_ready", False):
                try:
                    pts = int(getattr(self.sim, "jkr_interp_points", 512))
                    af = float(getattr(self.sim, "jkr_interp_alpha_factor", 5.0))
                    self.jkr._build_interp_table(n_points=pts, alpha_factor=af)
                except Exception:
                    # 构建失败则在 Numba 中回退牛顿迭代
                    self._jkr_interp_use = False
            if self._jkr_interp_use and getattr(self.jkr, "_interp_ready", False):
                self._jkr_alpha_min = float(self.jkr._interp_alpha_min or 0.0)
                self._jkr_alpha_max = float(self.jkr._interp_alpha_max or 0.0)
                self._jkr_dalpha = float(self.jkr._interp_dalpha or 1.0)
                self._jkr_a_tab = np.array(self.jkr._interp_a or [0.0, 0.0], dtype=np.float64)
                self._jkr_F_tab = np.array(self.jkr._interp_F or [0.0, 0.0], dtype=np.float64)
            else:
                self._jkr_alpha_min = 0.0
                self._jkr_alpha_max = 0.0
                self._jkr_dalpha = 1.0
                self._jkr_a_tab = np.zeros(2, dtype=np.float64)
                self._jkr_F_tab = np.zeros(2, dtype=np.float64)
        else:
            # 插值未启用时也准备占位参数，方便统一传入 Numba
            self._jkr_alpha_min = 0.0
            self._jkr_alpha_max = 0.0
            self._jkr_dalpha = 1.0
            self._jkr_a_tab = np.zeros(2, dtype=np.float64)
            self._jkr_F_tab = np.zeros(2, dtype=np.float64)

        self.tan = TangentialFriction(TangentialParams(MU_S=self.mat.MU_S, MU_K=self.mat.MU_K, G_star=self.mat.G_star, e_n=self.mat.e_n))
        self.roll = RollingResistance(RollingParams(mu_r=self.mat.mu_r, r=self.mat.r))
        # histories
        # 迁移：为减少 Python 热路径开销，优先使用与邻域缓存对齐的数组映射
        # 当未启用邻域缓存时，仍保留字典以保持兼容
        self.tangential_hist: Dict[Tuple[int,int], float] = {}
        self.contact_started: Dict[Tuple[int,int], bool] = {}
        self._pair_hist: np.ndarray | None = None         # 与 _nbr_pairs 对齐的切向历史
        self._pair_started_arr: np.ndarray | None = None  # 与 _nbr_pairs 对齐的接触标记（0/1 浮点）
        self._pair_index: Dict[Tuple[int,int], int] = {}  # (min(i,j), max(i,j)) -> 索引
        # outputs
        os.makedirs(self.sim.snapshot_dir, exist_ok=True)
        self.saver = SnapshotSaver(self.sim.snapshot_dir, snapshot_show_velocity=bool(getattr(self.sim, "snapshot_show_velocity", True)))
        self.report = ReportWriter(self.sim.snapshot_dir)
        self._next_snap = 0.0
        # 邻域缓存（Verlet 邻居）：按粒子最大位移判断何时重建
        # 仅当 sim.use_neighbor_cache=True 时启用
        self._nbr_pairs: List[Tuple[int, int]] | None = None
        self._nbr_ref_x: np.ndarray | None = None
        self._nbr_ref_y: np.ndarray | None = None
        self._nbr_rebuild_count: int = 0
        self._nbr_query_steps: int = 0
        # 选择后端（仅输出一次）
        if getattr(self.sim, "use_cpp", False) and _CPP_AVAILABLE:
            self.backend = "cpp"
        elif getattr(self.sim, "use_numba", False) and _NUMBA_AVAILABLE:
            self.backend = "numba"
        else:
            self.backend = "python"
        print(f"[DEM] Backend: {self.backend}")
        # ---------- 自适应时间步长相关（按全局最大速度平滑缩放） ----------
        self._adaptive_enabled = bool(getattr(self.sim, "adaptive_timestep", False))
        self._dt_normal = float(self.sim.dt)
        self._adaptive_dt_min_factor = max(1.0, float(getattr(self.sim, "adaptive_dt_min_factor", 1.0)))
        self._adaptive_dt_max_factor = max(self._adaptive_dt_min_factor, float(getattr(self.sim, "adaptive_dt_max_factor", 50.0)))
        self._adaptive_vmax_ema_alpha = float(getattr(self.sim, "adaptive_vmax_ema_alpha", 0.2))
        self._adaptive_vmax_ema_alpha = min(max(self._adaptive_vmax_ema_alpha, 0.0), 1.0)
        v_abs0 = np.sqrt(self.state.vx * self.state.vx + self.state.vy * self.state.vy)
        self._vmax_init = max(float(np.max(v_abs0)), 1e-30)
        self._vmax_approx = self._vmax_init
        self._adaptive_factor_sum = 0.0
        self._adaptive_factor_min = self._adaptive_dt_max_factor
        self._adaptive_factor_max = self._adaptive_dt_min_factor
        self._contacts_2r: List[Tuple[int,int]] = []
        if self._adaptive_enabled:
            print(
                f"[DEM] Adaptive timestep: ON "
                f"(vmax0={self._vmax_init:.3e}, "
                f"min_factor={self._adaptive_dt_min_factor:.2f}, "
                f"max_factor={self._adaptive_dt_max_factor:.2f}, "
                f"ema_alpha={self._adaptive_vmax_ema_alpha:.2f})"
            )
        # Numba 缓冲：避免每步重复分配
        self._nbuf_capacity = 0
        self._nb_i_idx = None
        self._nb_j_idx = None
        self._nb_dx = None
        self._nb_dy = None
        self._nb_d = None
        self._nb_hist_in = None
        self._nb_started_in = None

    def _ensure_nb_buf(self, m: int) -> None:
        if not _NUMBA_AVAILABLE:
            return
        if m <= self._nbuf_capacity:
            return
        # 以 1.5x 增长，减少频繁扩容
        new_cap = int(max(8, _math.ceil(m * 1.5)))
        self._nb_i_idx = np.empty(new_cap, dtype=np.int64)
        self._nb_j_idx = np.empty(new_cap, dtype=np.int64)
        self._nb_dx = np.empty(new_cap, dtype=np.float64)
        self._nb_dy = np.empty(new_cap, dtype=np.float64)
        self._nb_d = np.empty(new_cap, dtype=np.float64)
        self._nb_hist_in = np.empty(new_cap, dtype=np.float64)
        self._nb_started_in = np.empty(new_cap, dtype=np.float64)
        self._nbuf_capacity = new_cap

    # ------------------------------
    # 邻域缓存（Verlet 邻居）
    # ------------------------------

    def _minimal_image(self, dx: float, dy: float) -> Tuple[float, float]:
        """2D 周期最小镜像位移，假定坐标已包裹到 [-L/2, L/2)。"""
        half_x = 0.5 * self.Lx
        half_y = 0.5 * self.Ly
        if dx > half_x:
            dx -= self.Lx
        elif dx < -half_x:
            dx += self.Lx
        if dy > half_y:
            dy -= self.Ly
        elif dy < -half_y:
            dy += self.Ly
        return dx, dy

    def _rebuild_neighbor_pairs(self) -> None:
        """重建邻接对并更新参考坐标。

        使用现有的 find_contacts / find_contacts_numba 构建一次宽邻居表，
        只缓存 (i, j) 索引，几何信息在后续每步基于当前坐标重新计算。
        """
        n = self.state.size()
        if n == 0:
            self._nbr_pairs = []
            self._nbr_ref_x = None
            self._nbr_ref_y = None
            return

        # 统计重建次数
        self._nbr_rebuild_count += 1

        if _NUMBA_AVAILABLE and bool(getattr(self.sim, "use_numba", False)):
            use_parallel = bool(getattr(self.sim, "use_numba_parallel", False))
            raw_pairs = find_contacts_numba(self.state.x, self.state.y, self.neigh, use_parallel=use_parallel)
        else:
            raw_pairs = find_contacts(self.state.x, self.state.y, self.neigh)

        # 只缓存索引对，避免保存过期的几何信息
        new_pairs = [(int(i), int(j)) for (i, j, _dx, _dy, _d) in raw_pairs]
        old_pairs = self._nbr_pairs or []
        old_hist = self._pair_hist
        old_started = self._pair_started_arr
        # 构建旧映射以便迁移历史
        old_map: Dict[Tuple[int,int], int] = {}
        if old_pairs:
            for idx, (pi, pj) in enumerate(old_pairs):
                key = (pi, pj) if pi < pj else (pj, pi)
                old_map[key] = idx
        # 更新缓存与参考坐标
        self._nbr_pairs = new_pairs
        # 记录参考坐标（上次构建邻居表时的位置）
        self._nbr_ref_x = self.state.x.copy()
        self._nbr_ref_y = self.state.y.copy()
        # 重建数组映射并迁移历史
        m = len(new_pairs)
        self._pair_hist = np.zeros(m, dtype=np.float64)
        self._pair_started_arr = np.zeros(m, dtype=np.float64)
        self._pair_index.clear()
        for k, (i, j) in enumerate(new_pairs):
            key = (i, j) if i < j else (j, i)
            self._pair_index[key] = k
            if old_hist is not None and old_started is not None:
                old_k = old_map.get(key, -1)
                if old_k >= 0:
                    self._pair_hist[k] = float(old_hist[old_k])
                    self._pair_started_arr[k] = float(old_started[old_k])

    def _get_neighbor_pairs(self) -> List[Tuple[int, int]]:
        """返回当前步可用的邻接对 (i, j)，按最大位移判断是否需要重建。

        使用 skin 作为 Verlet 皮肤：当粒子相对上次重建的最大位移
        超过 skin/2 时，重建邻居表。
        """
        n = self.state.size()
        if n == 0:
            return []

        # 若未开启邻域缓存，退化为每步完整重建
        if not bool(getattr(self.sim, "use_neighbor_cache", False)):
            if _NUMBA_AVAILABLE and bool(getattr(self.sim, "use_numba", False)):
                use_parallel = bool(getattr(self.sim, "use_numba_parallel", False))
                raw_pairs = find_contacts_numba(self.state.x, self.state.y, self.neigh, use_parallel=use_parallel)
            else:
                raw_pairs = find_contacts(self.state.x, self.state.y, self.neigh)
            return [(int(i), int(j)) for (i, j, _dx, _dy, _d) in raw_pairs]

        need_rebuild = False
        if self._nbr_pairs is None or self._nbr_ref_x is None or self._nbr_ref_y is None:
            need_rebuild = True
        elif self._nbr_ref_x.shape[0] != n:
            # 粒子数发生变化，必须重建
            need_rebuild = True
        else:
            # 检查最大位移
            dx = self.state.x - self._nbr_ref_x
            dy = self.state.y - self._nbr_ref_y
            max_disp2 = float(np.max(dx * dx + dy * dy))
            # 使用 skin/2 作为安全阈值（Verlet 标准）
            thresh = 0.5 * float(self.skin)
            if thresh <= 0.0:
                # 若未配置 skin，则每步重建，退化为原行为
                need_rebuild = True
            else:
                need_rebuild = (max_disp2 >= (thresh * thresh))

        if need_rebuild:
            self._rebuild_neighbor_pairs()

        # 统计查询步数
        self._nbr_query_steps += 1
        return self._nbr_pairs or []

    # ------------------------------
    # Numba 并行力计算内核（可选）
    # ------------------------------
    if _NUMBA_AVAILABLE:
        @_njit(fastmath=True)
        def _numba_build_geometry_from_idx(
            x: np.ndarray,
            y: np.ndarray,
            Lx: float,
            Ly: float,
            i_idx: np.ndarray,
            j_idx: np.ndarray,
            dx_arr: np.ndarray,
            dy_arr: np.ndarray,
            d_arr: np.ndarray,
        ) -> None:
            """根据 (i,j) 索引，在 Numba 内部计算 dx, dy, d（含周期最小镜像）。"""
            m = i_idx.size
            half_x = 0.5 * Lx
            half_y = 0.5 * Ly
            for k in range(m):
                i = int(i_idx[k]); j = int(j_idx[k])
                dx = x[j] - x[i]
                dy = y[j] - y[i]
                if dx > half_x:
                    dx -= Lx
                elif dx < -half_x:
                    dx += Lx
                if dy > half_y:
                    dy -= Ly
                elif dy < -half_y:
                    dy += Ly
                dx_arr[k] = dx
                dy_arr[k] = dy
                d_arr[k] = _math.sqrt(dx * dx + dy * dy)

        @_njit(parallel=True, fastmath=True)
        def _numba_force_kernel(
            n: int,
            i_idx: np.ndarray,
            j_idx: np.ndarray,
            dx_arr: np.ndarray,
            dy_arr: np.ndarray,
            d_arr: np.ndarray,
            vx: np.ndarray,
            vy: np.ndarray,
            omega: np.ndarray,
            r: float,
            mass: float,
            E_star: float,
            gamma: float,
            e_n: float,
            mu_r: float,
            nu: float,
            MU_S: float,
            MU_K: float,
            dt: float,
            eta_c: float,
            hist_in: np.ndarray,
            started_in: np.ndarray,
            use_interp: int,
            alpha_min: float,
            alpha_max: float,
            dalpha: float,
            a_tab: np.ndarray,
            F_tab: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                    float, float]:
            fx = np.zeros(n)
            fy = np.zeros(n)
            tau = np.zeros(n)
            hist_out = np.empty(i_idx.size)
            contacts_mask = np.zeros(i_idx.size)
            # 自适应步长追踪量（在力循环中顺手计算，零额外开销）
            max_abs_vn = 0.0          # 所有接触对中最大 |v_n|
            min_approach_time = 1e30  # 所有未接触近邻对的最小碰撞接近时间
            mu_eff = 0.5 * mass
            R_eff = 0.5 * r

            # 预先计算 Tsuji 系数与 EDEM 标准拉脱力
            e_n_clamped = e_n
            if e_n_clamped < 1.0e-4:
                e_n_clamped = 1.0e-4
            if e_n_clamped > 0.9999:
                e_n_clamped = 0.9999
            loge = _math.log(e_n_clamped)
            if loge != 0.0:
                beta_tsuji = loge / _math.sqrt(loge * loge + _math.pi * _math.pi)
            else:
                beta_tsuji = 0.0

            # EDEM 标准：F_c = (3/2)πΓR*
            F_c = 1.5 * _math.pi * gamma * R_eff

            # 平衡接触半径 a0 = (9πΓR*²/E*)^(1/3)
            a0 = ((9.0 * _math.pi * gamma * R_eff * R_eff) / E_star) ** (1.0 / 3.0)

            # 临界重叠量 α_f = (3F_c²/(16R*E*²))^(1/3)
            alpha_f = ((3.0 * F_c * F_c) / (16.0 * R_eff * E_star * E_star)) ** (1.0 / 3.0)

            # 系数 2πΓ/E* 用于隐式公式
            coeff_alpha = 2.0 * _math.pi * gamma / E_star

            # 等效剪切模量 G*，与 Materials.G_star 一致：G* = E / (4 (2-ν) (1+ν))
            # 由 E_star 反解 E（相同材料、相同半径粒子）: E = 2 E* (1-ν^2)
            E_equiv = 2.0 * E_star * (1.0 - nu * nu)
            G_star = E_equiv / (4.0 * (2.0 - nu) * (1.0 + nu))
            for k in _prange(i_idx.size):
                i = int(i_idx[k]); j = int(j_idx[k])
                dxk = dx_arr[k]; dyk = dy_arr[k]; dk = d_arr[k]
                nx = dxk / (dk + 1e-18)
                ny = dyk / (dk + 1e-18)
                tx = -ny; ty = nx
                delta_n = 2.0 * r - dk
                v_rel_n = (vx[j] - vx[i]) * nx + (vy[j] - vy[i]) * ny
                v_rel_t = (vx[j] - vx[i]) * tx + (vy[j] - vy[i]) * ty
                # 接触点滑动速度：v_slip = v_rel - r*(ω_i + ω_j)*t
                v_rel_t_eff = v_rel_t - r * (omega[i] + omega[j])
                started = started_in[k] > 0.5

                # --- 自适应步长追踪 ---
                if delta_n > 0.0 or started:
                    # 接触对：追踪最大 |v_n|——用 max() 保证 prange 归约被识别
                    max_abs_vn = max(max_abs_vn, abs(v_rel_n))
                else:
                    # 未接触对：计算接近时间预警
                    v_approach = -v_rel_n
                    if v_approach > 1e-18:
                        gap = dk - 2.0 * r  # > 0
                        t_contact = gap / v_approach
                        min_approach_time = min(min_approach_time, t_contact)

                # ========== EDEM 标准 JKR 法向力 ==========
                # 可选：使用插值表近似 F_elastic 与 a_contact，否则回退牛顿迭代。
                Fn = 0.0
                a_contact = 0.0  # 接触半径，用于切向力计算

                if delta_n > -alpha_f:
                    # 未接触且无黏附历史，不计算力
                    if delta_n < 0.0 and (not started):
                        Fn = 0.0
                        a_contact = 0.0
                    else:
                        F_elastic = 0.0
                        used_interp = False
                        if use_interp == 1:
                            # 插值区间内：线性插值 a(α), F_elastic(α)
                            if (delta_n > alpha_min) and (delta_n < alpha_max):
                                t_alpha = (delta_n - alpha_min) / dalpha
                                idx = int(t_alpha)
                                n_tab = a_tab.size
                                if idx < 0:
                                    idx = 0
                                    frac = 0.0
                                elif idx >= n_tab - 1:
                                    idx = n_tab - 2
                                    frac = 1.0
                                else:
                                    frac = t_alpha - idx
                                a0_tab = a_tab[idx]
                                a1_tab = a_tab[idx + 1]
                                F0_tab = F_tab[idx]
                                F1_tab = F_tab[idx + 1]
                                a_interp = a0_tab + (a1_tab - a0_tab) * frac
                                F_elastic = F0_tab + (F1_tab - F0_tab) * frac
                                if a_interp > 0.0:
                                    a_contact = a_interp
                                    used_interp = True

                        if not used_interp:
                            # 牛顿迭代求解 a：隐式公式 α = a²/R* - √(2πΓa/E*)
                            if delta_n >= 0.0:
                                a_init = _math.sqrt(R_eff * max(delta_n, 1e-18))
                                a = max(a_init, 0.5 * a0)
                            else:
                                a = a0

                            for _iter in range(20):
                                # α(a) = a²/R* - √(2πΓa/E*)
                                alpha_model = (a * a) / R_eff - _math.sqrt(coeff_alpha * a)
                                f_newt = alpha_model - delta_n
                                if abs(f_newt) < 1e-12 * max(1.0, abs(delta_n)):
                                    break
                                # dα/da = 2a/R* - (1/2)√(2πΓ/(E*a))
                                d_alpha_da = 2.0 * a / R_eff - 0.5 * _math.sqrt(coeff_alpha / max(a, 1e-30))
                                if abs(d_alpha_da) < 1e-30:
                                    break
                                a_new = a - f_newt / d_alpha_da
                                if a_new <= 0.0:
                                    a_new = 0.5 * a
                                a = min(a_new, 10.0 * a0)

                            if a > 0.0:
                                a_contact = a
                                a3 = a * a * a
                                # EDEM 标准法向力：F_n = 4E*a³/(3R*) - √(8πΓE*a³)
                                F_hertz = (4.0 * E_star * a3) / (3.0 * R_eff)
                                F_adhesion = _math.sqrt(8.0 * _math.pi * gamma * E_star * a3)
                                F_elastic = F_hertz - F_adhesion

                        if a_contact > 0.0:
                            # Tsuji 阻尼：F_dn = -2*sqrt(5/6)*β*sqrt(k_n*m*)*v^n
                            # k_n = 2*E*a
                            k_n = 2.0 * E_star * a_contact
                            if beta_tsuji != 0.0 and k_n > 0.0:
                                eta_n = -2.0 * _math.sqrt(5.0 / 6.0) * beta_tsuji * _math.sqrt(max(mu_eff * k_n, 1e-18))
                                F_damping = -eta_n * v_rel_n
                            else:
                                F_damping = 0.0

                            Fn = F_elastic + F_damping
                        else:
                            Fn = 0.0
                else:
                    # 接触断裂：δ_n ≤ -α_f
                    Fn = 0.0
                    a_contact = 0.0

                # 切向力
                Ft = 0.0
                if a_contact > 0.0 and Fn != 0.0:
                    # 使用牛顿迭代求得的接触半径
                    k_t = 8.0 * G_star * a_contact
                    if k_t > 0.0:
                        delta_t = hist_in[k] + v_rel_t_eff * dt
                        F_ct = -k_t * delta_t

                        # Tsuji 切向阻尼: F_dt = -η_t v_t,
                        # 其中 η_t = -2 sqrt(5/6) β sqrt(k_t m*)
                        if beta_tsuji != 0.0:
                            eta_t = -2.0 * _math.sqrt(5.0 / 6.0) * beta_tsuji * _math.sqrt(max(k_t * mu_eff, 1.0e-18))
                            F_dt = -eta_t * v_rel_t_eff
                        else:
                            F_dt = 0.0

                        F_trial = F_ct + F_dt

                        # 黏附摩擦上限：|F_n^t + 2 F_c|
                        F_normal_eff = Fn + 2.0 * F_c
                        F_limit_static = MU_S * abs(F_normal_eff)

                        if abs(F_trial) <= F_limit_static:
                            Ft = F_trial
                            hist_out[k] = delta_t
                        else:
                            F_limit_slide = MU_K * abs(F_normal_eff)
                            if F_limit_slide > 0.0:
                                # 滑动摩擦力方向与滑动速度相反
                                if v_rel_t_eff > 1e-18:
                                    v_sign = 1.0
                                elif v_rel_t_eff < -1e-18:
                                    v_sign = -1.0
                                else:
                                    v_sign = 0.0
                                Ft = -F_limit_slide * v_sign
                                hist_out[k] = -(F_limit_static / k_t) * v_sign
                            else:
                                Ft = 0.0
                                hist_out[k] = 0.0
                        contacts_mask[k] = 1.0
                    else:
                        hist_out[k] = hist_in[k]
                else:
                    hist_out[k] = hist_in[k]

                # 力与力矩叠加
                Fx = Fn * nx + Ft * tx
                Fy = Fn * ny + Ft * ty
                fx[i] -= Fx; fy[i] -= Fy
                fx[j] += Fx; fy[j] += Fy
                tau[i] += -Ft * r
                tau[j] += -Ft * r
                # 滚动阻力矩
                if delta_n > 0.0:
                    T_roll = mu_r * abs(Fn) * r
                    sgn_i = 1.0 if omega[i] > 1e-18 else (-1.0 if omega[i] < -1e-18 else 0.0)
                    sgn_j = 1.0 if omega[j] > 1e-18 else (-1.0 if omega[j] < -1e-18 else 0.0)
                    tau[i] += -T_roll * sgn_i
                    tau[j] += -T_roll * sgn_j

            return fx, fy, tau, hist_out, contacts_mask, max_abs_vn, min_approach_time

    def _estimate_box_length(self) -> float:
        if not self.state.x.size:
            return 10.0 * self.mat.r
        x_sorted = np.sort(self.state.x)
        y_sorted = np.sort(self.state.y)
        xspan = float(x_sorted[-1] - x_sorted[0])
        yspan = float(y_sorted[-1] - y_sorted[0])
        # 估计网格步长 h: 取非零最小相邻差值
        def _step_est(arr: np.ndarray) -> float:
            if arr.size < 2:
                return 2.0 * self.mat.r
            diffs = np.diff(np.sort(arr))
            # 过滤浮点微小噪声，取正的最小差
            pos = diffs[diffs > 1e-15]
            return float(np.min(pos)) if pos.size else 2.0 * self.mat.r
        hx = _step_est(self.state.x)
        hy = _step_est(self.state.y)
        h_est = max(hx, hy)
        # 目标边长约等于 N*h ≈ span + h
        return max(xspan + h_est, yspan + h_est)

    def _compute_forces(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """返回 (fx, fy, tau, max_abs_vn, min_approach_time)。

        max_abs_vn       : 所有接触对中最大 |v_n|（团簇内部稳定指标）
        min_approach_time : 未接触近邻对中最小碰撞接近时间（碰撞预警）
        """
        n = self.state.size()
        fx = np.zeros(n); fy = np.zeros(n); tau = np.zeros(n)
        # 使用邻域缓存（Verlet 邻居）：仅在位移超过 skin/2 时重建索引对
        idx_pairs = self._get_neighbor_pairs()

        # 快路径：未启用 C++、未启用团簇冻结，且使用 Numba 与邻域缓存时，
        # 直接在 Numba 内部根据 (i,j) 计算几何量，避免 Python 端的大循环。
        use_fast_numba_idx = (
            _NUMBA_AVAILABLE
            and bool(getattr(self.sim, "use_numba", False))
            and not bool(getattr(self.sim, "use_cpp", False))
            and bool(getattr(self.sim, "use_neighbor_cache", False))
        )

        if use_fast_numba_idx and idx_pairs:
            m = len(idx_pairs)
            self._ensure_nb_buf(m)
            i_idx = self._nb_i_idx[:m]
            j_idx = self._nb_j_idx[:m]
            dx_arr = self._nb_dx[:m]
            dy_arr = self._nb_dy[:m]
            d_arr = self._nb_d[:m]
            # 直接使用与邻域缓存对齐的数组映射
            hist_in = self._nb_hist_in[:m]
            started_in = self._nb_started_in[:m]
            for k, (i, j) in enumerate(idx_pairs):
                i_idx[k] = i; j_idx[k] = j
                # 与缓存数组对齐拷贝输入历史
                if self._pair_hist is not None and self._pair_started_arr is not None:
                    hist_in[k] = float(self._pair_hist[k])
                    started_in[k] = float(self._pair_started_arr[k])
                else:
                    # 回退：若未启用缓存数组（极少见），保持旧逻辑
                    key = (i, j) if i < j else (j, i)
                    hist_in[k] = float(self.tangential_hist.get(key, 0.0))
                    started_in[k] = 1.0 if bool(self.contact_started.get(key, False)) else 0.0

            # 在 Numba 内部根据当前坐标构建 dx, dy, d（含周期最小镜像）
            DEMSimulator._numba_build_geometry_from_idx(
                self.state.x, self.state.y,
                float(self.Lx), float(self.Ly),
                i_idx, j_idx, dx_arr, dy_arr, d_arr,
            )

            eta_c = float(getattr(self.jkr, "eta_c", 0.0))
            fx, fy, tau, hist_out, contacts_mask, nb_max_vn, nb_min_at = DEMSimulator._numba_force_kernel(
                n,
                i_idx, j_idx, dx_arr, dy_arr, d_arr,
                self.state.vx, self.state.vy, self.state.omega,
                self.state.r, self.state.mass,
                self.mat.E_star, self.mat.gamma, self.mat.e_n, self.mat.mu_r,
                self.mat.nu, self.mat.MU_S, self.mat.MU_K,
                self.sim.dt, eta_c,
                hist_in, started_in,
                1 if self._jkr_interp_use else 0,
                float(self._jkr_alpha_min),
                float(self._jkr_alpha_max),
                float(self._jkr_dalpha),
                self._jkr_a_tab,
                self._jkr_F_tab,
            )
            # 旋转黏滞统一叠加
            tau += - self.mat.c_rot * self.state.omega
            # 合并切向历史和接触标记
            contacts_2r: List[Tuple[int,int]] = []
            # 输出写回数组映射，并构造 2r 邻接列表
            if self._pair_hist is not None and self._pair_started_arr is not None:
                self._pair_hist[:m] = hist_out
                for k, (i, j) in enumerate(idx_pairs):
                    if contacts_mask[k] > 0.5:
                        self._pair_started_arr[k] = 1.0
                        key = (i, j) if i < j else (j, i)
                        contacts_2r.append(key)
            else:
                # 回退：保持旧的字典方式
                for k, (i, j) in enumerate(idx_pairs):
                    key = (i, j) if i < j else (j, i)
                    self.tangential_hist[key] = float(hist_out[k])
                    if contacts_mask[k] > 0.5:
                        self.contact_started[key] = True
                        contacts_2r.append(key)
            self._contacts_2r = contacts_2r
            # 在未启用自适应时间步长时，返回的 max_vn/min_approach 仅用于保持接口一致
            return fx, fy, tau, float(nb_max_vn), float(nb_min_at)

        # 通用路径：需要 C++、团簇冻结或未启用邻域缓存时，
        # 仍在 Python 端重建 (dx, dy, d) 列表并沿用原有逻辑。
        pairs: List[Tuple[int, int, float, float, float]] = []
        r = self.state.r
        # 邻接表是宽邻域（2r+skin），此处只保留物理上可能相关的近邻
        max_dist = 2.0 * r + max(0.0, float(self.skin))
        max_dist2 = max_dist * max_dist
        for (i, j) in idx_pairs:
            dx = float(self.state.x[j] - self.state.x[i])
            dy = float(self.state.y[j] - self.state.y[i])
            dx, dy = self._minimal_image(dx, dy)
            d2 = dx * dx + dy * dy
            if d2 <= max_dist2:
                d = math.sqrt(d2)
                pairs.append((i, j, dx, dy, d))
        # 若启用并可用 C++ 内核，优先使用 C++
        if (_CPP_AVAILABLE) and bool(getattr(self.sim, "use_cpp", False)) and (len(pairs) > 0):
            m = len(pairs)
            i_idx = np.empty(m, dtype=np.int64)
            j_idx = np.empty(m, dtype=np.int64)
            dx_arr = np.empty(m, dtype=np.float64)
            dy_arr = np.empty(m, dtype=np.float64)
            d_arr = np.empty(m, dtype=np.float64)
            hist_in = np.empty(m, dtype=np.float64)
            started_in = np.empty(m, dtype=np.float64)
            keys: List[Tuple[int,int]] = []
            for k, (i, j, dx, dy, d) in enumerate(pairs):
                i_idx[k] = i; j_idx[k] = j
                dx_arr[k] = dx; dy_arr[k] = dy; d_arr[k] = d
                key = (i, j) if i < j else (j, i)
                keys.append(key)
                hist_in[k] = float(self.tangential_hist.get(key, 0.0))
                started_in[k] = 1.0 if bool(self.contact_started.get(key, False)) else 0.0
            eta_c = float(getattr(self.jkr, "eta_c", 0.0))
            fx, fy, tau, hist_out, contacts_mask = _cppkern.forces_kernel(
                n,
                i_idx, j_idx, dx_arr, dy_arr, d_arr,
                self.state.vx, self.state.vy, self.state.omega,
                self.state.r, self.state.mass,
                self.mat.E_star, self.mat.gamma, self.mat.e_n, self.mat.mu_r,
                self.mat.nu, self.mat.MU_S, self.mat.MU_K,
                self.sim.dt, eta_c,
                hist_in, started_in,
            )
            tau += - self.mat.c_rot * self.state.omega
            contacts_2r: List[Tuple[int,int]] = []
            for k, key in enumerate(keys):
                self.tangential_hist[key] = float(hist_out[k])
                if contacts_mask[k] > 0.5:
                    self.contact_started[key] = True
                    contacts_2r.append(key)
            self._contacts_2r = contacts_2r
            # C++ 内核暂不返回追踪量，使用保守默认值
            return fx, fy, tau, float('inf'), 1e30

        # 如果启用并可用 Numba，加速对遍历
        if (_NUMBA_AVAILABLE) and bool(getattr(self.sim, "use_numba", False)) and (len(pairs) > 0):
            m = len(pairs)
            self._ensure_nb_buf(m)
            i_idx = self._nb_i_idx[:m]
            j_idx = self._nb_j_idx[:m]
            dx_arr = self._nb_dx[:m]
            dy_arr = self._nb_dy[:m]
            d_arr = self._nb_d[:m]
            hist_in = self._nb_hist_in[:m]
            started_in = self._nb_started_in[:m]
            keys: List[Tuple[int,int]] = []
            for k, (i, j, dx, dy, d) in enumerate(pairs):
                i_idx[k] = i; j_idx[k] = j
                dx_arr[k] = dx; dy_arr[k] = dy; d_arr[k] = d
                key = (i, j) if i < j else (j, i)
                keys.append(key)
                hist_in[k] = float(self.tangential_hist.get(key, 0.0))
                started_in[k] = 1.0 if bool(self.contact_started.get(key, False)) else 0.0
            # 预先计算 JKR 拉伸判别长度 eta_c（沿用现有 JKR 实现的值）
            eta_c = float(getattr(self.jkr, "eta_c", 0.0))
            fx, fy, tau, hist_out, contacts_mask, nb_max_vn, nb_min_at = DEMSimulator._numba_force_kernel(
                n,
                i_idx, j_idx, dx_arr, dy_arr, d_arr,
                self.state.vx, self.state.vy, self.state.omega,
                self.state.r, self.state.mass,
                self.mat.E_star, self.mat.gamma, self.mat.e_n, self.mat.mu_r,
                self.mat.nu, self.mat.MU_S, self.mat.MU_K,
                self.sim.dt, eta_c,
                hist_in, started_in,
                1 if self._jkr_interp_use else 0,
                float(self._jkr_alpha_min),
                float(self._jkr_alpha_max),
                float(self._jkr_dalpha),
                self._jkr_a_tab,
                self._jkr_F_tab,
            )
            # 旋转黏滞统一叠加
            tau += - self.mat.c_rot * self.state.omega
            # 合并切向历史和接触标记
            contacts_2r: List[Tuple[int,int]] = []
            for k, key in enumerate(keys):
                self.tangential_hist[key] = float(hist_out[k])
                if contacts_mask[k] > 0.5:
                    self.contact_started[key] = True
                    contacts_2r.append(key)
            self._contacts_2r = contacts_2r
            return fx, fy, tau, float(nb_max_vn), float(nb_min_at)

        # 否则走原有单线程路径
        # 真接触列表（2r 内）用于连通体与切向历史
        contacts_2r: List[Tuple[int,int]] = []
        r = self.state.r
        mu = 0.5 * self.state.mass
        py_max_abs_vn = 0.0
        py_min_approach = 1e30

        for (i, j, dx, dy, d) in pairs:
            nx = dx / (d + 1e-18)
            ny = dy / (d + 1e-18)
            tx = -ny; ty = nx
            delta_n = 2.0 * r - d
            v_rel_n = (self.state.vx[j] - self.state.vx[i]) * nx + (self.state.vy[j] - self.state.vy[i]) * ny
            v_rel_t = (self.state.vx[j] - self.state.vx[i]) * tx + (self.state.vy[j] - self.state.vy[i]) * ty
            # 接触点滑动速度：v_slip = v_rel - r*(ω_i + ω_j)*t
            v_rel_t_eff = v_rel_t - r * (self.state.omega[i] + self.state.omega[j])

            key = (i, j) if i < j else (j, i)
            # 使用数组映射优先（启用邻域缓存时）
            if bool(getattr(self.sim, "use_neighbor_cache", False)) and self._pair_index:
                idx = self._pair_index.get(key, -1)
                started = (self._pair_started_arr[idx] > 0.5) if (idx >= 0 and self._pair_started_arr is not None) else False
            else:
                started = bool(self.contact_started.get(key, False))

            # --- 自适应步长追踪 ---
            if delta_n > 0.0 or started:
                abs_vn = abs(v_rel_n)
                if abs_vn > py_max_abs_vn:
                    py_max_abs_vn = abs_vn
            else:
                v_approach = -v_rel_n
                if v_approach > 1e-18:
                    gap = d - 2.0 * r
                    t_contact = gap / v_approach
                    if t_contact < py_min_approach:
                        py_min_approach = t_contact

            Fn = self.jkr.force(delta_n, v_rel_n, mu, contact_started=started)
            Ft = 0.0
            # 仅当法向有真实接触（Fn!=0 且 jkr 求得 a>0）时计算切向力
            a_contact = float(self.jkr.last_newton_a or 0.0)
            if Fn != 0.0 and a_contact > 0.0:
                # 切向历史优先使用数组映射（启用邻域缓存时）
                if bool(getattr(self.sim, "use_neighbor_cache", False)) and self._pair_index:
                    idx = self._pair_index.get(key, -1)
                    dth = float(self._pair_hist[idx]) if (idx >= 0 and self._pair_hist is not None) else 0.0
                    Ft, dth_new = self.tan.force(a_contact, dth, v_rel_t_eff, self.sim.dt, Fn, self.jkr.F_c, mu)
                    if idx >= 0 and self._pair_hist is not None and self._pair_started_arr is not None:
                        self._pair_hist[idx] = dth_new
                        self._pair_started_arr[idx] = 1.0
                    contacts_2r.append(key)
                else:
                    dth = self.tangential_hist.get(key, 0.0)
                    Ft, dth_new = self.tan.force(a_contact, dth, v_rel_t_eff, self.sim.dt, Fn, self.jkr.F_c, mu)
                    self.tangential_hist[key] = dth_new
                    contacts_2r.append(key)
                    self.contact_started[key] = True

            Fx = Fn * nx + Ft * tx
            Fy = Fn * ny + Ft * ty
            fx[i] -= Fx; fy[i] -= Fy
            fx[j] += Fx; fy[j] += Fy

            tau[i] += -Ft * r
            tau[j] += -Ft * r
            if delta_n > 0.0:
                T_roll = self.mat.mu_r * abs(Fn) * r
                sgn_i = 1.0 if self.state.omega[i] > 1e-18 else (-1.0 if self.state.omega[i] < -1e-18 else 0.0)
                sgn_j = 1.0 if self.state.omega[j] > 1e-18 else (-1.0 if self.state.omega[j] < -1e-18 else 0.0)
                tau[i] += -T_roll * sgn_i
                tau[j] += -T_roll * sgn_j

        tau += - self.mat.c_rot * self.state.omega
        self._contacts_2r = contacts_2r
        return fx, fy, tau, py_max_abs_vn, py_min_approach

    def _apply_periodic_wrap(self) -> None:
        # 周期坐标包裹，保持中心域 [-L/2, L/2)
        self.state.x = (self.state.x + 0.5 * self.Lx) % self.Lx - 0.5 * self.Lx
        self.state.y = (self.state.y + 0.5 * self.Ly) % self.Ly - 0.5 * self.Ly

    def _integrate(self, fx: np.ndarray, fy: np.ndarray, tau: np.ndarray) -> None:
        dt = self.sim.dt
        m = self.state.mass
        I_p = 0.4 * m * (self.state.r ** 2)  # 单粒子惯量
        self.state.vx += (fx / m) * dt
        self.state.vy += (fy / m) * dt
        self.state.omega += (tau / I_p) * dt
        self.state.x += self.state.vx * dt
        self.state.y += self.state.vy * dt

    def _energies(self) -> Tuple[float, float, float]:
        m = self.state.mass
        KE_trans = 0.5 * m * float(np.sum(self.state.vx**2 + self.state.vy**2))
        I = 0.4 * m * (self.state.r ** 2)
        KE_rot = 0.5 * I * float(np.sum(self.state.omega**2))
        return KE_trans, KE_rot, KE_trans + KE_rot

    def _cluster_metrics(self) -> Tuple[int, float, float]:
        """返回当前结构指标：(连通体数量, 全局平均配位数, 连通体平均回转半径)。"""
        n = int(self.state.size())
        if n <= 0:
            return 0, 0.0, 0.0

        contacts = self._contacts_2r if self._contacts_2r else []
        comps = connected_components_from_contacts(n, contacts)
        cluster_count = int(len(comps))

        # 全局平均配位数：接触图中每个粒子的度数平均
        degree = np.zeros(n, dtype=np.int64)
        for i, j in contacts:
            degree[int(i)] += 1
            degree[int(j)] += 1
        coordination_mean = float(np.mean(degree))

        # 连通体平均回转半径（包含单粒子连通体，其 Rg 视为 0）
        rg_sum = 0.0
        x = self.state.x
        y = self.state.y
        use_periodic = (self.Lx > 0.0) and (self.Ly > 0.0)
        for comp in comps:
            if not comp:
                continue
            if len(comp) == 1:
                # 单粒子连通体 Rg=0
                continue
            mb = np.asarray(comp, dtype=np.int64)
            ref = int(mb[0])
            dx = x[mb] - x[ref]
            dy = y[mb] - y[ref]
            if use_periodic:
                dx -= np.round(dx / self.Lx) * self.Lx
                dy -= np.round(dy / self.Ly) * self.Ly
            cx = float(np.mean(dx))
            cy = float(np.mean(dy))
            rg = float(np.sqrt(np.mean((dx - cx) ** 2 + (dy - cy) ** 2)))
            rg_sum += rg

        cluster_rg_mean = rg_sum / max(cluster_count, 1)
        return cluster_count, coordination_mean, cluster_rg_mean

    def run(self, t_picture: float | None = None) -> None:
        t = 0.0; step = 0
        start_time = time.perf_counter()
        snap_dt = float(self.sim.snapshot_interval or 0.0)
        next_snap = snap_dt if snap_dt > 0 else float('inf')
        # 初始快照与报告
        self.report.open()
        # 记录本次模拟的关键参数（注释行，不影响绘图解析）
        try:
            param_dict = {
                "materials.r": self.mat.r,
                "materials.rho": self.mat.rho,
                "materials.E": self.mat.E,
                "materials.nu": self.mat.nu,
                "materials.gamma": self.mat.gamma,
                "materials.MU_S": self.mat.MU_S,
                "materials.MU_K": self.mat.MU_K,
                "materials.e_n": self.mat.e_n,
                "materials.mu_r": self.mat.mu_r,
                "materials.c_rot": self.mat.c_rot,
                "sim.dt": self.sim.dt,
                "sim.t_end": self.sim.t_end,
                "sim.skin_factor": self.sim.skin_factor,
                "sim.snapshot_interval": self.sim.snapshot_interval,
                "sim.save_stride": self.sim.save_stride,
                "sim.domain_scale": self.sim.domain_scale,
                "sim.Lx": self.Lx,
                "sim.Ly": self.Ly,
                "sim.use_numba": getattr(self.sim, "use_numba", False),
                "sim.use_numba_parallel": getattr(self.sim, "use_numba_parallel", False),
                "sim.use_cpp": getattr(self.sim, "use_cpp", False),
                "sim.adaptive_timestep": getattr(self.sim, "adaptive_timestep", False),
                "sim.adaptive_dt_min_factor": self._adaptive_dt_min_factor,
                "sim.adaptive_dt_max_factor": self._adaptive_dt_max_factor,
                "sim.adaptive_vmax_ema_alpha": self._adaptive_vmax_ema_alpha,
                "sim.adaptive_vmax_init": self._vmax_init,
                "state.N": int(self.state.size()),
            }
            # 若 ReportWriter 支持写入参数，则调用；旧版本保持兼容
            write_params = getattr(self.report, "write_params", None)
            if callable(write_params):
                write_params(param_dict)
        except Exception:
            # 参数写入失败不应影响主仿真
            pass
        if snap_dt > 0:
            init_dt_info = None
            if bool(getattr(self.sim, "snapshot_show_dt_factor", False)):
                init_dt_info = f"dt={self._dt_normal:.1e}x1"
            self.saver.save(
                self.state,
                0.0,
                self.Lx,
                self.Ly,
                os.path.join(self.sim.snapshot_dir, "snap_000000_t0.000e+00.png"),
                dt_info=init_dt_info,
            )
            KE_t0, KE_r0, KE_sum0 = self._energies()
            vmax0 = float(np.max(np.sqrt(self.state.vx**2 + self.state.vy**2)))
            cluster_count0, coord_mean0, rg_mean0 = self._cluster_metrics()
            self.report.write_step(0.0, KE_t0, KE_r0, KE_sum0, vmax0, cluster_count0, coord_mean0, rg_mean0)
        snap_idx = 0
        # 统计
        n_nonbase_steps = 0
        n_total_steps = 0

        while t < self.sim.t_end:
            # ---- 自适应时间步长决策（基于当前全局最大速度） ----
            current_dt = self._dt_normal
            dt_factor = 1.0
            vmax_now = float(np.max(np.sqrt(self.state.vx * self.state.vx + self.state.vy * self.state.vy)))
            if self._adaptive_enabled:
                a = self._adaptive_vmax_ema_alpha
                self._vmax_approx = a * vmax_now + (1.0 - a) * self._vmax_approx
                vmax_eff = max(self._vmax_approx, 1e-30)
                raw_factor = self._vmax_init / vmax_eff
                dt_factor = min(max(raw_factor, self._adaptive_dt_min_factor), self._adaptive_dt_max_factor)
                current_dt = self._dt_normal * dt_factor
            self.sim.dt = current_dt
            is_large_dt = (current_dt > self._dt_normal * 1.01)
            if is_large_dt:
                n_nonbase_steps += 1
            n_total_steps += 1
            if self._adaptive_enabled:
                self._adaptive_factor_sum += dt_factor
                if dt_factor < self._adaptive_factor_min:
                    self._adaptive_factor_min = dt_factor
                if dt_factor > self._adaptive_factor_max:
                    self._adaptive_factor_max = dt_factor

            fx, fy, tau, max_abs_vn, min_approach_time = self._compute_forces()

            # 快照与报告
            if t >= next_snap:
                snap_idx += 1
                dt_tag = "_Ldt" if is_large_dt else ""
                fname = f"snap_{snap_idx:06d}_t{t:.3e}{dt_tag}.png"
                dt_info = None
                if bool(getattr(self.sim, "snapshot_show_dt_factor", False)):
                    dt_info = f"dt={self._dt_normal:.1e}x{dt_factor:.2f}, vmax={vmax_now:.2e}"
                self.saver.save(self.state, t, self.Lx, self.Ly,
                                os.path.join(self.sim.snapshot_dir, fname),
                                dt_info=dt_info,
                                max_abs_vn=max_abs_vn)
                KE_t, KE_r, KE_sum = self._energies()
                vmax = float(np.max(np.sqrt(self.state.vx**2 + self.state.vy**2)))
                cluster_count, coord_mean, rg_mean = self._cluster_metrics()
                self.report.write_step(t, KE_t, KE_r, KE_sum, vmax, cluster_count, coord_mean, rg_mean)
                next_snap += snap_dt

            self._integrate(fx, fy, tau)
            self._apply_periodic_wrap()
            t += current_dt
            step += 1

        # 恢复基准步长
        self.sim.dt = self._dt_normal
        # 结束快照
        self.saver.save(self.state, t, self.Lx, self.Ly, os.path.join(self.sim.snapshot_dir, "final_snapshot.png"))
        KE_tf, KE_rf, KE_sumf = self._energies()
        vmaxf = float(np.max(np.sqrt(self.state.vx**2 + self.state.vy**2)))
        cluster_countf, coord_meanf, rg_meanf = self._cluster_metrics()
        self.report.write_step(t, KE_tf, KE_rf, KE_sumf, vmaxf, cluster_countf, coord_meanf, rg_meanf)
        elapsed = time.perf_counter() - start_time
        self.report.write_runtime(elapsed)
        self.report.close()
        # 自适应步长统计
        if self._adaptive_enabled:
            ratio = n_nonbase_steps / max(n_total_steps, 1) * 100
            mean_factor = self._adaptive_factor_sum / max(n_total_steps, 1)
            print(f"[DEM] Adaptive stats: {n_nonbase_steps}/{n_total_steps} non-base-dt steps ({ratio:.1f}%)")
            print(
                "[DEM] Adaptive factor stats: "
                f"mean={mean_factor:.2f}, "
                f"min={self._adaptive_factor_min:.2f}, "
                f"max={self._adaptive_factor_max:.2f}"
            )

        # 邻域缓存统计
        if bool(getattr(self.sim, "use_neighbor_cache", False)):
            print(f"[DEM] Neighbor cache: rebuilds={self._nbr_rebuild_count}, steps={self._nbr_query_steps}")


if __name__ == "__main__":
    print("请运行入口脚本：multibody_collision_DEM/scripts/run_simulation.py，或使用 -m 方式运行包模块。")
