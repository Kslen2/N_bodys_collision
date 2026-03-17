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
    from ..physics.quaternion import update_orientations
    from .particle_system import SystemState
    from .contact_detector import (
        NeighborParams,
        find_contacts,
        find_contacts_numba,
        find_contacts_numba_arrays,
        connected_components_from_contacts,
        find_contacts_adaptive,
        find_contacts_numba_adaptive,
        find_contacts_numba_adaptive_arrays,
    )
    from ..io.snapshot_saver import SnapshotSaver
    from ..io.trajectory_recorder import TrajectoryRecorder, TrajectoryRecorderConfig
    from ..visualization.report_generator import ReportWriter
except ImportError:
    _HERE = os.path.abspath(os.path.dirname(__file__))
    _PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    from src.physics.materials import Materials
    from src.physics.contact.jkr_normal import JKRParams, JKRNormal
    from src.physics.contact.tangential import TangentialParams, TangentialFriction
    from src.physics.contact.rolling import RollingParams, RollingResistance
    from src.simulation.particle_system import SystemState
    from src.simulation.contact_detector import (
        NeighborParams,
        find_contacts,
        find_contacts_numba,
        find_contacts_numba_arrays,
        connected_components_from_contacts,
        find_contacts_adaptive,
        find_contacts_numba_adaptive,
        find_contacts_numba_adaptive_arrays,
    )
    from src.io.snapshot_saver import SnapshotSaver
    from src.io.trajectory_recorder import TrajectoryRecorder, TrajectoryRecorderConfig
    from src.visualization.report_generator import ReportWriter
    from src.physics.quaternion import update_orientations


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
    Lz: float | None = None
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
    # 自适应动态网格（四叉树 / 空间哈希）
    use_adaptive_grid: bool = False
    # 轨迹记录（HDF5）
    trajectory_record: bool = True
    trajectory_stride: int = 100
    trajectory_filename: str = "trajectory.h5"


# ============================================================
# Numba Kernels (Module Level)
# ============================================================
@_njit(cache=True)
def _numba_minimal_image(dx: float, dy: float, dz: float, 
                      Lx: float, Ly: float, Lz: float):
    # 手动 inline 以确保一致性
    if dx > 0.5 * Lx: dx -= Lx
    elif dx < -0.5 * Lx: dx += Lx
    if dy > 0.5 * Ly: dy -= Ly
    elif dy < -0.5 * Ly: dy += Ly
    if dz > 0.5 * Lz: dz -= Lz
    elif dz < -0.5 * Lz: dz += Lz
    return dx, dy, dz

if _NUMBA_AVAILABLE:
    @_njit(cache=True, parallel=True)
    def _numba_build_geometry_from_idx(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        Lx: float,
        Ly: float,
        Lz: float,
        i_idx: np.ndarray,
        j_idx: np.ndarray,
        dx_arr: np.ndarray,
        dy_arr: np.ndarray,
        dz_arr: np.ndarray,
        d_arr: np.ndarray,
    ) -> None:
        """根据 (i,j) 索引，在 Numba 内部计算 dx, dy, dz, d（含周期最小镜像）。"""
        m = i_idx.size
        for k in _prange(m):
            i = int(i_idx[k]); j = int(j_idx[k])
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            dz = z[j] - z[i]

            dx, dy, dz = _numba_minimal_image(dx, dy, dz, Lx, Ly, Lz)

            dx_arr[k] = dx
            dy_arr[k] = dy
            dz_arr[k] = dz
            d_arr[k] = _math.sqrt(dx * dx + dy * dy + dz * dz)

    @_njit(cache=True)
    def _numba_calc_one_pair_force(
        i: int, j: int, k: int,
        dxk: float, dyk: float, dzk: float, dk: float,
        vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
        omega_x: np.ndarray, omega_y: np.ndarray, omega_z: np.ndarray,
        r: float, mass: float,
        E_star: float, gamma: float, e_n: float, mu_r: float,
        nu: float, MU_S: float, MU_K: float,
        dt: float, eta_c: float,
        hist_in: np.ndarray, started_in: np.ndarray,
        use_interp: int, alpha_min: float, alpha_max: float, dalpha: float,
        a_tab: np.ndarray, F_tab: np.ndarray,
        hist_out: np.ndarray, contacts_mask: np.ndarray,
    ) -> tuple[float, float, float, float, float, float, float, float, float, float, float]:
        """
        计算单个接触对的力。
        返回: (Fx_total, Fy_total, Fz_total, 
               tau_xi, tau_yi, tau_zi, 
               tau_xj, tau_yj, tau_zj, 
               max_vn, min_approach)
        Fx_total... 是作用在 j 上的总力 (对 i 为负)。
        """
        Fx_total = 0.0; Fy_total = 0.0; Fz_total = 0.0
        tau_xi = 0.0; tau_yi = 0.0; tau_zi = 0.0
        tau_xj = 0.0; tau_yj = 0.0; tau_zj = 0.0
        max_abs_vn = 0.0
        min_approach_time = 1e30

        inv_d = 1.0 / (dk + 1e-18)
        nx = dxk * inv_d
        ny = dyk * inv_d
        nz = dzk * inv_d
        delta_n = 2.0 * r - dk

        # 相对速度
        dvx = vx[j] - vx[i]
        dvy = vy[j] - vy[i]
        dvz = vz[j] - vz[i]
        v_rel_n = dvx * nx + dvy * ny + dvz * nz

        # 接触点相对速度 = v_rel - r*(ω_i + ω_j) × n
        wx = omega_x[i] + omega_x[j]
        wy = omega_y[i] + omega_y[j]
        wz = omega_z[i] + omega_z[j]
        cross_x = wy * nz - wz * ny
        cross_y = wz * nx - wx * nz
        cross_z = wx * ny - wy * nx

        vt_x = dvx - v_rel_n * nx - r * cross_x
        vt_y = dvy - v_rel_n * ny - r * cross_y
        vt_z = dvz - v_rel_n * nz - r * cross_z

        started = started_in[k] > 0.5

        if delta_n > 0.0 or started:
            max_abs_vn = abs(v_rel_n)
        else:
            v_approach = -v_rel_n
            if v_approach > 1e-18:
                gap = dk - 2.0 * r
                t_contact = gap / v_approach
                min_approach_time = t_contact

        mu_eff = 0.5 * mass
        R_eff = 0.5 * r

        # 计算 beta_tsuji (可以预计算，这里为了内联方便重算)
        e_n_clamped = e_n
        if e_n_clamped < 1.0e-4: e_n_clamped = 1.0e-4
        if e_n_clamped > 0.9999: e_n_clamped = 0.9999
        loge = _math.log(e_n_clamped)
        if loge != 0.0:
            beta_tsuji = loge / _math.sqrt(loge * loge + _math.pi * _math.pi)
        else:
            beta_tsuji = 0.0

        F_c = 1.5 * _math.pi * gamma * R_eff
        # 这些常数本可以在外层计算，但内联后 Numba 可能会优化
        # 这里为了保持接口简单，我们在内部计算，或者依赖 Numba 的常量折叠
        # 为性能考虑，最好这些是常量。但它们来自 self.mat。
        # 暂时保持原样。
        a0 = ((9.0 * _math.pi * gamma * R_eff * R_eff) / E_star) ** (1.0 / 3.0)
        alpha_f = ((3.0 * F_c * F_c) / (16.0 * R_eff * E_star * E_star)) ** (1.0 / 3.0)
        coeff_alpha = 2.0 * _math.pi * gamma / E_star
        E_equiv = 2.0 * E_star * (1.0 - nu * nu)
        G_star = E_equiv / (4.0 * (2.0 - nu) * (1.0 + nu))

        Fn = 0.0
        a_contact = 0.0

        if delta_n > -alpha_f:
            if delta_n < 0.0 and (not started):
                Fn = 0.0; a_contact = 0.0
            else:
                F_elastic = 0.0
                used_interp = False
                if use_interp == 1:
                    if (delta_n > alpha_min) and (delta_n < alpha_max):
                        t_alpha = (delta_n - alpha_min) / dalpha
                        idx = int(t_alpha)
                        n_tab = a_tab.size
                        if idx < 0: idx = 0; frac = 0.0
                        elif idx >= n_tab - 1: idx = n_tab - 2; frac = 1.0
                        else: frac = t_alpha - idx
                        a_interp = a_tab[idx] + (a_tab[idx + 1] - a_tab[idx]) * frac
                        F_elastic = F_tab[idx] + (F_tab[idx + 1] - F_tab[idx]) * frac
                        if a_interp > 0.0 and F_elastic != 0.0:
                            a_contact = a_interp; used_interp = True

                if not used_interp:
                    if delta_n >= 0.0:
                        a_init = _math.sqrt(R_eff * max(delta_n, 1e-18))
                        a = max(a_init, 0.5 * a0)
                    else:
                        a = a0
                    
                    for _iter in range(20):
                        alpha_model = (a * a) / R_eff - _math.sqrt(coeff_alpha * a)
                        f_newt = alpha_model - delta_n
                        if abs(f_newt) < 1e-12 * max(1.0, abs(delta_n)): break
                        d_alpha_da = 2.0 * a / R_eff - 0.5 * _math.sqrt(coeff_alpha / max(a, 1e-30))
                        if abs(d_alpha_da) < 1e-30: break
                        a_new = a - f_newt / d_alpha_da
                        if a_new <= 0.0: a_new = 0.5 * a
                        a = min(a_new, 10.0 * a0)
                    
                    if a > 0.0:
                        a_contact = a
                        a3 = a * a * a
                        F_hertz = (4.0 * E_star * a3) / (3.0 * R_eff)
                        F_adhesion = _math.sqrt(8.0 * _math.pi * gamma * E_star * a3)
                        F_elastic = F_hertz - F_adhesion

                if a_contact > 0.0:
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
            Fn = 0.0; a_contact = 0.0

        # 切向力
        Ft_x = 0.0; Ft_y = 0.0; Ft_z = 0.0
        if a_contact > 0.0 and Fn != 0.0:
            k_t = 8.0 * G_star * a_contact
            if k_t > 0.0:
                dt_prev_x = hist_in[k*3]; dt_prev_y = hist_in[k*3+1]; dt_prev_z = hist_in[k*3+2]
                dot_prev_n = dt_prev_x * nx + dt_prev_y * ny + dt_prev_z * nz
                dt_prev_x -= dot_prev_n * nx; dt_prev_y -= dot_prev_n * ny; dt_prev_z -= dot_prev_n * nz

                dt_hist_x = dt_prev_x + vt_x * dt
                dt_hist_y = dt_prev_y + vt_y * dt
                dt_hist_z = dt_prev_z + vt_z * dt
                dot_dn = dt_hist_x * nx + dt_hist_y * ny + dt_hist_z * nz
                dt_hist_x -= dot_dn * nx; dt_hist_y -= dot_dn * ny; dt_hist_z -= dot_dn * nz

                F_ct_x = -k_t * dt_hist_x; F_ct_y = -k_t * dt_hist_y; F_ct_z = -k_t * dt_hist_z

                if beta_tsuji != 0.0:
                    eta_t = -2.0 * _math.sqrt(5.0 / 6.0) * beta_tsuji * _math.sqrt(max(k_t * mu_eff, 1.0e-18))
                    F_dt_x = -eta_t * vt_x; F_dt_y = -eta_t * vt_y; F_dt_z = -eta_t * vt_z
                else:
                    F_dt_x = 0.0; F_dt_y = 0.0; F_dt_z = 0.0

                Ft_trial_x = F_ct_x + F_dt_x
                Ft_trial_y = F_ct_y + F_dt_y
                Ft_trial_z = F_ct_z + F_dt_z
                Ft_trial_mag = _math.sqrt(Ft_trial_x**2 + Ft_trial_y**2 + Ft_trial_z**2)

                F_normal_eff = Fn + 2.0 * F_c
                F_limit_static = MU_S * abs(F_normal_eff)

                if Ft_trial_mag <= F_limit_static:
                    Ft_x = Ft_trial_x; Ft_y = Ft_trial_y; Ft_z = Ft_trial_z
                    hist_out[k*3] = dt_hist_x; hist_out[k*3+1] = dt_hist_y; hist_out[k*3+2] = dt_hist_z
                else:
                    F_limit_slide = MU_K * abs(F_normal_eff)
                    vt_mag = _math.sqrt(vt_x * vt_x + vt_y * vt_y + vt_z * vt_z)
                    if F_limit_slide > 0.0 and vt_mag > 1e-18:
                        inv_vt = 1.0 / vt_mag
                        Ft_x = -F_limit_slide * vt_x * inv_vt
                        Ft_y = -F_limit_slide * vt_y * inv_vt
                        Ft_z = -F_limit_slide * vt_z * inv_vt
                        hist_out[k*3]   = F_limit_static / k_t * vt_x * inv_vt
                        hist_out[k*3+1] = F_limit_static / k_t * vt_y * inv_vt
                        hist_out[k*3+2] = F_limit_static / k_t * vt_z * inv_vt
                    else:
                        Ft_x = 0.0; Ft_y = 0.0; Ft_z = 0.0
                        hist_out[k*3] = 0.0; hist_out[k*3+1] = 0.0; hist_out[k*3+2] = 0.0
                contacts_mask[k] = 1.0
            else:
                hist_out[k*3] = hist_in[k*3]; hist_out[k*3+1] = hist_in[k*3+1]; hist_out[k*3+2] = hist_in[k*3+2]
        else:
            hist_out[k*3] = hist_in[k*3]; hist_out[k*3+1] = hist_in[k*3+1]; hist_out[k*3+2] = hist_in[k*3+2]

        Fx_total = Fn * nx + Ft_x
        Fy_total = Fn * ny + Ft_y
        Fz_total = Fn * nz + Ft_z
        
        # Torque
        torque_x = -r * (ny * Ft_z - nz * Ft_y)
        torque_y = -r * (nz * Ft_x - nx * Ft_z)
        torque_z = -r * (nx * Ft_y - ny * Ft_x)
        tau_xi += torque_x; tau_yi += torque_y; tau_zi += torque_z
        tau_xj += torque_x; tau_yj += torque_y; tau_zj += torque_z

        if delta_n > 0.0:
            T_roll = mu_r * abs(Fn) * r
            oi_mag = _math.sqrt(omega_x[i]**2 + omega_y[i]**2 + omega_z[i]**2)
            if oi_mag > 1e-18:
                inv_oi = 1.0 / oi_mag
                tau_xi -= T_roll * omega_x[i] * inv_oi
                tau_yi -= T_roll * omega_y[i] * inv_oi
                tau_zi -= T_roll * omega_z[i] * inv_oi
            oj_mag = _math.sqrt(omega_x[j]**2 + omega_y[j]**2 + omega_z[j]**2)
            if oj_mag > 1e-18:
                inv_oj = 1.0 / oj_mag
                tau_xj -= T_roll * omega_x[j] * inv_oj
                tau_yj -= T_roll * omega_y[j] * inv_oj
                tau_zj -= T_roll * omega_z[j] * inv_oj
        
        return (Fx_total, Fy_total, Fz_total, 
                tau_xi, tau_yi, tau_zi, 
                tau_xj, tau_yj, tau_zj, 
                max_abs_vn, min_approach_time)

    @_njit(cache=True)
    def _numba_force_kernel(
        n: int,
        i_idx: np.ndarray, j_idx: np.ndarray,
        dx_arr: np.ndarray, dy_arr: np.ndarray, dz_arr: np.ndarray, d_arr: np.ndarray,
        vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
        omega_x: np.ndarray, omega_y: np.ndarray, omega_z: np.ndarray,
        r: float, mass: float,
        E_star: float, gamma: float, e_n: float, mu_r: float,
        nu: float, MU_S: float, MU_K: float,
        dt: float, eta_c: float,
        hist_in: np.ndarray, started_in: np.ndarray,
        use_interp: int, alpha_min: float, alpha_max: float, dalpha: float,
        a_tab: np.ndarray, F_tab: np.ndarray,
        fx: np.ndarray, fy: np.ndarray, fz: np.ndarray,
        tau_x: np.ndarray, tau_y: np.ndarray, tau_z: np.ndarray,
        hist_out: np.ndarray, contacts_mask: np.ndarray,
    ) -> tuple[float, float]:
        for i in range(n):
            fx[i] = 0.0; fy[i] = 0.0; fz[i] = 0.0
            tau_x[i] = 0.0; tau_y[i] = 0.0; tau_z[i] = 0.0
        for k in range(i_idx.size):
            contacts_mask[k] = 0.0

        max_abs_vn = 0.0
        min_approach_time = 1e30

        for k in range(i_idx.size):
            i = int(i_idx[k]); j = int(j_idx[k])
            dxk = dx_arr[k]; dyk = dy_arr[k]; dzk = dz_arr[k]; dk = d_arr[k]
            
            (Fx, Fy, Fz, txi, tyi, tzi, txj, tyj, tzj, mvn, mat) = _numba_calc_one_pair_force(
                i, j, k, dxk, dyk, dzk, dk,
                vx, vy, vz, omega_x, omega_y, omega_z,
                r, mass, E_star, gamma, e_n, mu_r,
                nu, MU_S, MU_K, dt, eta_c,
                hist_in, started_in,
                use_interp, alpha_min, alpha_max, dalpha,
                a_tab, F_tab, hist_out, contacts_mask
            )
            
            fx[i] -= Fx; fy[i] -= Fy; fz[i] -= Fz
            fx[j] += Fx; fy[j] += Fy; fz[j] += Fz
            tau_x[i] += txi; tau_y[i] += tyi; tau_z[i] += tzi
            tau_x[j] += txj; tau_y[j] += tyj; tau_z[j] += tzj
            
            max_abs_vn = max(max_abs_vn, mvn)
            min_approach_time = min(min_approach_time, mat)

        return max_abs_vn, min_approach_time

    @_njit(cache=True, parallel=True)
    def _numba_force_kernel_parallel(
        num_threads: int,
        n: int,
        i_idx: np.ndarray, j_idx: np.ndarray,
        dx_arr: np.ndarray, dy_arr: np.ndarray, dz_arr: np.ndarray, d_arr: np.ndarray,
        vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
        omega_x: np.ndarray, omega_y: np.ndarray, omega_z: np.ndarray,
        r: float, mass: float,
        E_star: float, gamma: float, e_n: float, mu_r: float,
        nu: float, MU_S: float, MU_K: float,
        dt: float, eta_c: float,
        hist_in: np.ndarray, started_in: np.ndarray,
        use_interp: int, alpha_min: float, alpha_max: float, dalpha: float,
        a_tab: np.ndarray, F_tab: np.ndarray,
        fx: np.ndarray, fy: np.ndarray, fz: np.ndarray,
        tau_x: np.ndarray, tau_y: np.ndarray, tau_z: np.ndarray,
        hist_out: np.ndarray, contacts_mask: np.ndarray,
        fx_temp: np.ndarray, fy_temp: np.ndarray, fz_temp: np.ndarray,
        tx_temp: np.ndarray, ty_temp: np.ndarray, tz_temp: np.ndarray,
        mvn_temp: np.ndarray, mat_temp: np.ndarray,
    ) -> tuple[float, float]:
        # 并行计算：避免竞争
        # 使用外部预分配的线程私有力累加数组 + 手动块划分

        # 归零线程私有缓冲区（并行安全：每个 (t,i) 唯一）
        for t in _prange(num_threads):
            for i in range(n):
                fx_temp[t, i] = 0.0
                fy_temp[t, i] = 0.0
                fz_temp[t, i] = 0.0
                tx_temp[t, i] = 0.0
                ty_temp[t, i] = 0.0
                tz_temp[t, i] = 0.0
            mvn_temp[t] = 0.0
            mat_temp[t] = 1e30

        # 归零 contacts_mask (parallel safe because each k unique)
        for k in _prange(i_idx.size):
            contacts_mask[k] = 0.0

        m = i_idx.size
        if m == 0:
            for i in _prange(n):
                fx[i]=0; fy[i]=0; fz[i]=0; tau_x[i]=0; tau_y[i]=0; tau_z[i]=0
            return 0.0, 1e30

        chunk_size = (m + num_threads - 1) // num_threads

        for t in _prange(num_threads):
            start = t * chunk_size
            end = min(start + chunk_size, m)

            local_mvn = 0.0
            local_mat = 1e30

            for k in range(start, end):
                i = int(i_idx[k]); j = int(j_idx[k])
                dxk = dx_arr[k]; dyk = dy_arr[k]; dzk = dz_arr[k]; dk = d_arr[k]

                (Fx, Fy, Fz, txi, tyi, tzi, txj, tyj, tzj, mvn, mat) = _numba_calc_one_pair_force(
                    i, j, k, dxk, dyk, dzk, dk,
                    vx, vy, vz, omega_x, omega_y, omega_z,
                    r, mass, E_star, gamma, e_n, mu_r,
                    nu, MU_S, MU_K, dt, eta_c,
                    hist_in, started_in,
                    use_interp, alpha_min, alpha_max, dalpha,
                    a_tab, F_tab, hist_out, contacts_mask
                )

                fx_temp[t, i] -= Fx; fy_temp[t, i] -= Fy; fz_temp[t, i] -= Fz
                fx_temp[t, j] += Fx; fy_temp[t, j] += Fy; fz_temp[t, j] += Fz
                tx_temp[t, i] += txi; ty_temp[t, i] += tyi; tz_temp[t, i] += tzi
                tx_temp[t, j] += txj; ty_temp[t, j] += tyj; tz_temp[t, j] += tzj

                local_mvn = max(local_mvn, mvn)
                local_mat = min(local_mat, mat)

            mvn_temp[t] = local_mvn
            mat_temp[t] = local_mat

        # Reduction (parallel over particles)
        for i in _prange(n):
            sum_fx = 0.0; sum_fy = 0.0; sum_fz = 0.0
            sum_tx = 0.0; sum_ty = 0.0; sum_tz = 0.0
            for t in range(num_threads):
                sum_fx += fx_temp[t, i]
                sum_fy += fy_temp[t, i]
                sum_fz += fz_temp[t, i]
                sum_tx += tx_temp[t, i]
                sum_ty += ty_temp[t, i]
                sum_tz += tz_temp[t, i]
            fx[i] = sum_fx; fy[i] = sum_fy; fz[i] = sum_fz
            tau_x[i] = sum_tx; tau_y[i] = sum_ty; tau_z[i] = sum_tz

        return np.max(mvn_temp), np.min(mat_temp)


class DEMSimulator:
    def __init__(self, materials: Materials, sim: SimParams, state: SystemState) -> None:
        self.mat = materials
        self.sim = sim
        self.state = state
        # 安全设置 Numba 并行线程数，确保和外部环境一致，避免资源竞争
        self._num_threads = 4
        if _NUMBA_AVAILABLE:
            try:
                import os
                # 尝试从环境变量获取
                env_threads = os.environ.get("OMP_NUM_THREADS") or os.environ.get("NUMBA_NUM_THREADS")
                if env_threads:
                    self._num_threads = int(env_threads)
                else:
                    self._num_threads = int(_nb.config.NUMBA_DEFAULT_NUM_THREADS)
                if self._num_threads < 1:
                   self._num_threads = 1
            except Exception:
                import os as _os_fallback
                self._num_threads = max(1, _os_fallback.cpu_count() or 4)

        # domain: 优先使用外部传入的 Lx/Ly/Lz，否则按位置估算并乘以 domain_scale
        if (self.sim.Lx is not None) and (self.sim.Ly is not None) and (self.sim.Lz is not None):
            self.Lx = float(self.sim.Lx)
            self.Ly = float(self.sim.Ly)
            self.Lz = float(self.sim.Lz)
        else:
            L_est = self._estimate_box_length()
            self.Lx = self.sim.domain_scale * L_est
            self.Ly = self.sim.domain_scale * L_est
            self.Lz = self.sim.domain_scale * L_est
        # neighbor/search（需已知 Lx/Ly/Lz）
        self.skin = float(sim.skin_factor) * self.mat.r
        self.neigh = NeighborParams(r=self.mat.r, skin=self.skin, cell_size=None, Lx=self.Lx, Ly=self.Ly, Lz=self.Lz)
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
        self.tangential_hist: Dict[Tuple[int,int], np.ndarray] = {}  # key -> [dt_x, dt_y, dt_z]
        self.contact_started: Dict[Tuple[int,int], bool] = {}
        self._pair_hist: np.ndarray | None = None         # 与 _nbr_pairs 对齐的切向历史 (m*3,)
        self._pair_started_arr: np.ndarray | None = None  # 与 _nbr_pairs 对齐的接触标记（0/1 浮点）
        self._pair_index: Dict[Tuple[int,int], int] = {}  # (min(i,j), max(i,j)) -> 索引
        # outputs
        os.makedirs(self.sim.snapshot_dir, exist_ok=True)
        self.saver = SnapshotSaver(self.sim.snapshot_dir, snapshot_show_velocity=bool(getattr(self.sim, "snapshot_show_velocity", True)))
        self.traj = TrajectoryRecorder(
            self.sim.snapshot_dir,
            TrajectoryRecorderConfig(
                enabled=bool(getattr(self.sim, "trajectory_record", True)),
                stride=int(getattr(self.sim, "trajectory_stride", 100)),
                filename=str(getattr(self.sim, "trajectory_filename", "trajectory.h5")),
            ),
        )
        self.report = ReportWriter(self.sim.snapshot_dir)
        self._next_snap = 0.0
        # 邻域缓存（Verlet 邻居）：按粒子最大位移判断何时重建
        # 仅当 sim.use_neighbor_cache=True 时启用
        self._nbr_pairs: List[Tuple[int, int]] | None = None
        self._nbr_ref_x: np.ndarray | None = None
        self._nbr_ref_y: np.ndarray | None = None
        self._nbr_ref_z: np.ndarray | None = None
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
        v_abs0 = np.sqrt(self.state.vx * self.state.vx + self.state.vy * self.state.vy + self.state.vz * self.state.vz)
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
        self._nb_dz = None
        self._nb_d = None
        self._nb_hist_in = None
        self._nb_started_in = None
        self._nb_hist_out = None
        self._nb_contacts_mask = None
        # 并行力核心线程私有缓冲区
        self._par_buf_n = 0
        self._par_buf_nt = 0
        self._nb_fx_temp = None
        self._nb_fy_temp = None
        self._nb_fz_temp = None
        self._nb_tx_temp = None
        self._nb_ty_temp = None
        self._nb_tz_temp = None
        self._nb_mvn_temp = None
        self._nb_mat_temp = None

    def _ensure_nb_buf(self, m: int) -> None:
        if not _NUMBA_AVAILABLE:
            return
        if m <= self._nbuf_capacity:
            return
        new_cap = int(max(8, _math.ceil(m * 1.5)))
        self._nb_i_idx = np.empty(new_cap, dtype=np.int64)
        self._nb_j_idx = np.empty(new_cap, dtype=np.int64)
        self._nb_dx = np.empty(new_cap, dtype=np.float64)
        self._nb_dy = np.empty(new_cap, dtype=np.float64)
        self._nb_dz = np.empty(new_cap, dtype=np.float64)
        self._nb_d = np.empty(new_cap, dtype=np.float64)
        self._nb_hist_in = np.empty(new_cap * 3, dtype=np.float64)
        self._nb_started_in = np.empty(new_cap, dtype=np.float64)
        self._nb_hist_out = np.empty(new_cap * 3, dtype=np.float64)
        self._nb_contacts_mask = np.empty(new_cap, dtype=np.float64)
        self._nbuf_capacity = new_cap

    def _ensure_parallel_buf(self, n: int) -> None:
        """预分配力核心并行缓冲区（线程私有力/力矩及归约临时数组）。"""
        if not _NUMBA_AVAILABLE:
            return
        nt = self._num_threads
        if self._par_buf_n >= n and self._par_buf_nt >= nt:
            return
        self._par_buf_n = n
        self._par_buf_nt = nt
        self._nb_fx_temp = np.zeros((nt, n), dtype=np.float64)
        self._nb_fy_temp = np.zeros((nt, n), dtype=np.float64)
        self._nb_fz_temp = np.zeros((nt, n), dtype=np.float64)
        self._nb_tx_temp = np.zeros((nt, n), dtype=np.float64)
        self._nb_ty_temp = np.zeros((nt, n), dtype=np.float64)
        self._nb_tz_temp = np.zeros((nt, n), dtype=np.float64)
        self._nb_mvn_temp = np.zeros(nt, dtype=np.float64)
        self._nb_mat_temp = np.full(nt, 1e30, dtype=np.float64)

    # ------------------------------
    # 邻域缓存（Verlet 邻居）
    # ------------------------------

    def _minimal_image(self, dx: float, dy: float, dz: float) -> Tuple[float, float, float]:
        """三维周期最小镜像位移。"""
        half_x = 0.5 * self.Lx
        half_y = 0.5 * self.Ly
        half_z = 0.5 * self.Lz
        if dx > half_x:
            dx -= self.Lx
        elif dx < -half_x:
            dx += self.Lx
        if dy > half_y:
            dy -= self.Ly
        elif dy < -half_y:
            dy += self.Ly
        if dz > half_z:
            dz -= self.Lz
        elif dz < -half_z:
            dz += self.Lz
        return dx, dy, dz

    def _rebuild_neighbor_pairs(self) -> None:
        """重建邻接对并更新参考坐标（三维）。"""
        n = self.state.size()
        if n == 0:
            self._nbr_pairs = []
            self._nbr_ref_x = None
            self._nbr_ref_y = None
            self._nbr_ref_z = None
            return

        self._nbr_rebuild_count += 1

        _use_adaptive = bool(getattr(self.sim, "use_adaptive_grid", False))
        if _NUMBA_AVAILABLE and bool(getattr(self.sim, "use_numba", False)):
            use_parallel = bool(getattr(self.sim, "use_numba_parallel", False))
            if _use_adaptive:
                raw_pairs = find_contacts_numba_adaptive(self.state.x, self.state.y, self.state.z, self.neigh, use_parallel=use_parallel)
            else:
                raw_pairs = find_contacts_numba(self.state.x, self.state.y, self.state.z, self.neigh, use_parallel=use_parallel)
        else:
            if _use_adaptive:
                raw_pairs = find_contacts_adaptive(self.state.x, self.state.y, self.state.z, self.neigh)
            else:
                raw_pairs = find_contacts(self.state.x, self.state.y, self.state.z, self.neigh)

        new_pairs = sorted(
            [(int(i), int(j)) for (i, j, _dx, _dy, _dz, _d) in raw_pairs],
            key=lambda p: (p[0], p[1]),
        )
        old_pairs = self._nbr_pairs or []
        old_hist = self._pair_hist
        old_started = self._pair_started_arr
        old_map: Dict[Tuple[int,int], int] = {}
        if old_pairs:
            for idx, (pi, pj) in enumerate(old_pairs):
                key = (pi, pj) if pi < pj else (pj, pi)
                old_map[key] = idx
        self._nbr_pairs = new_pairs
        self._nbr_ref_x = self.state.x.copy()
        self._nbr_ref_y = self.state.y.copy()
        self._nbr_ref_z = self.state.z.copy()
        m = len(new_pairs)
        self._pair_hist = np.zeros(m * 3, dtype=np.float64)  # 3 components per pair
        self._pair_started_arr = np.zeros(m, dtype=np.float64)
        self._pair_index.clear()
        for k, (i, j) in enumerate(new_pairs):
            key = (i, j) if i < j else (j, i)
            self._pair_index[key] = k
            if old_hist is not None and old_started is not None:
                old_k = old_map.get(key, -1)
                if old_k >= 0:
                    self._pair_hist[k*3]   = float(old_hist[old_k*3])
                    self._pair_hist[k*3+1] = float(old_hist[old_k*3+1])
                    self._pair_hist[k*3+2] = float(old_hist[old_k*3+2])
                    self._pair_started_arr[k] = float(old_started[old_k])

    def _get_neighbor_pairs(self) -> List[Tuple[int, int]]:
        """返回当前步可用的邻接对 (i, j)，按最大位移判断是否需要重建（三维）。"""
        n = self.state.size()
        if n == 0:
            return []

        if not bool(getattr(self.sim, "use_neighbor_cache", False)):
            _use_adaptive = bool(getattr(self.sim, "use_adaptive_grid", False))
            if _NUMBA_AVAILABLE and bool(getattr(self.sim, "use_numba", False)):
                use_parallel = bool(getattr(self.sim, "use_numba_parallel", False))
                if _use_adaptive:
                    raw_pairs = find_contacts_numba_adaptive(self.state.x, self.state.y, self.state.z, self.neigh, use_parallel=use_parallel)
                else:
                    raw_pairs = find_contacts_numba(self.state.x, self.state.y, self.state.z, self.neigh, use_parallel=use_parallel)
            else:
                if _use_adaptive:
                    raw_pairs = find_contacts_adaptive(self.state.x, self.state.y, self.state.z, self.neigh)
                else:
                    raw_pairs = find_contacts(self.state.x, self.state.y, self.state.z, self.neigh)
            return sorted(
                [(int(i), int(j)) for (i, j, _dx, _dy, _dz, _d) in raw_pairs],
                key=lambda p: (p[0], p[1]),
            )

        need_rebuild = False
        if self._nbr_pairs is None or self._nbr_ref_x is None or self._nbr_ref_y is None or self._nbr_ref_z is None:
            need_rebuild = True
        elif self._nbr_ref_x.shape[0] != n:
            need_rebuild = True
        else:
            dx = self.state.x - self._nbr_ref_x
            dy = self.state.y - self._nbr_ref_y
            dz = self.state.z - self._nbr_ref_z
            max_disp2 = float(np.max(dx * dx + dy * dy + dz * dz))
            thresh = 0.5 * float(self.skin)
            if thresh <= 0.0:
                need_rebuild = True
            else:
                need_rebuild = (max_disp2 >= (thresh * thresh))

        if need_rebuild:
            self._rebuild_neighbor_pairs()

        self._nbr_query_steps += 1
        return self._nbr_pairs or []



    def _estimate_box_length(self) -> float:
        if not self.state.x.size:
            return 10.0 * self.mat.r
        xspan = float(np.max(self.state.x) - np.min(self.state.x))
        yspan = float(np.max(self.state.y) - np.min(self.state.y))
        zspan = float(np.max(self.state.z) - np.min(self.state.z))
        def _step_est(arr: np.ndarray) -> float:
            if arr.size < 2:
                return 2.0 * self.mat.r
            diffs = np.diff(np.sort(arr))
            pos = diffs[diffs > 1e-15]
            return float(np.min(pos)) if pos.size else 2.0 * self.mat.r
        hx = _step_est(self.state.x)
        hy = _step_est(self.state.y)
        hz = _step_est(self.state.z)
        h_est = max(hx, hy, hz)
        return max(xspan + h_est, yspan + h_est, zspan + h_est)

    def _compute_forces(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                       np.ndarray, np.ndarray, np.ndarray,
                                       float, float]:
        """返回 (fx, fy, fz, tau_x, tau_y, tau_z, max_abs_vn, min_approach_time)。"""
        n = self.state.size()
        fx = np.zeros(n); fy = np.zeros(n); fz = np.zeros(n)
        tau_x = np.zeros(n); tau_y = np.zeros(n); tau_z = np.zeros(n)

        # 准备并行参数
        use_parallel = bool(getattr(self.sim, "use_numba_parallel", False))
        num_threads = self._num_threads
        
        # 调试输出
        if not hasattr(self, "_parallel_debug_printed"):
            print(f"[DEBUG] Use Numba: {_NUMBA_AVAILABLE}")
            print(f"[DEBUG] Use Parallel: {use_parallel}")
            print(f"[DEBUG] Num Threads: {num_threads}")
            self._parallel_debug_printed = True


        use_fast_numba_idx = (
            _NUMBA_AVAILABLE
            and bool(getattr(self.sim, "use_numba", False))
            and not bool(getattr(self.sim, "use_cpp", False))
            and bool(getattr(self.sim, "use_neighbor_cache", False))
        )

        use_numba_direct_arrays = (
            _NUMBA_AVAILABLE
            and bool(getattr(self.sim, "use_numba", False))
            and not bool(getattr(self.sim, "use_cpp", False))
            and not bool(getattr(self.sim, "use_neighbor_cache", False))
        )

        if use_numba_direct_arrays:
            use_parallel = bool(getattr(self.sim, "use_numba_parallel", False))
            use_adaptive = bool(getattr(self.sim, "use_adaptive_grid", False))
            if use_adaptive:
                i_idx_raw, j_idx_raw, dx_arr_raw, dy_arr_raw, dz_arr_raw, d_arr_raw = find_contacts_numba_adaptive_arrays(
                    self.state.x, self.state.y, self.state.z, self.neigh, use_parallel=use_parallel
                )
            else:
                i_idx_raw, j_idx_raw, dx_arr_raw, dy_arr_raw, dz_arr_raw, d_arr_raw = find_contacts_numba_arrays(
                    self.state.x, self.state.y, self.state.z, self.neigh, use_parallel=use_parallel
                )

            m = int(i_idx_raw.size)
            if m > 0:
                self._ensure_nb_buf(m)
                i_idx = self._nb_i_idx[:m]
                j_idx = self._nb_j_idx[:m]
                dx_arr = self._nb_dx[:m]
                dy_arr = self._nb_dy[:m]
                dz_arr = self._nb_dz[:m]
                d_arr = self._nb_d[:m]
                hist_in = self._nb_hist_in[:m*3]
                started_in = self._nb_started_in[:m]
                hist_out = self._nb_hist_out[:m*3]
                contacts_mask = self._nb_contacts_mask[:m]

                i_idx[:] = i_idx_raw
                j_idx[:] = j_idx_raw
                dx_arr[:] = dx_arr_raw
                dy_arr[:] = dy_arr_raw
                dz_arr[:] = dz_arr_raw
                d_arr[:] = d_arr_raw

                # 使用 _pair_index + _pair_hist 数组进行快速历史查找
                # 替代逐对 Python dict.get 循环
                old_pair_index = self._pair_index
                old_pair_hist = self._pair_hist
                old_pair_started = self._pair_started_arr
                keys: List[Tuple[int, int]] = [None] * m  # type: ignore
                hist_in[:] = 0.0
                started_in[:] = 0.0
                if old_pair_hist is not None and old_pair_started is not None and old_pair_index:
                    for k in range(m):
                        i = int(i_idx[k]); j = int(j_idx[k])
                        key = (i, j) if i < j else (j, i)
                        keys[k] = key
                        old_k = old_pair_index.get(key, -1)
                        if old_k >= 0:
                            hist_in[k*3]   = old_pair_hist[old_k*3]
                            hist_in[k*3+1] = old_pair_hist[old_k*3+1]
                            hist_in[k*3+2] = old_pair_hist[old_k*3+2]
                            started_in[k]  = old_pair_started[old_k]
                else:
                    for k in range(m):
                        i = int(i_idx[k]); j = int(j_idx[k])
                        key = (i, j) if i < j else (j, i)
                        keys[k] = key
                        h_arr = self.tangential_hist.get(key)
                        if h_arr is not None:
                            hist_in[k*3] = h_arr[0]; hist_in[k*3+1] = h_arr[1]; hist_in[k*3+2] = h_arr[2]
                        started_in[k] = 1.0 if self.contact_started.get(key, False) else 0.0

                eta_c = float(getattr(self.jkr, "eta_c", 0.0))

                kern_args = (
                    n,
                    i_idx, j_idx, dx_arr, dy_arr, dz_arr, d_arr,
                    self.state.vx, self.state.vy, self.state.vz,
                    self.state.omega_x, self.state.omega_y, self.state.omega_z,
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
                    fx, fy, fz,
                    tau_x, tau_y, tau_z,
                    hist_out, contacts_mask,
                )
                if use_parallel:
                    self._ensure_parallel_buf(n)
                    nb_max_vn, nb_min_at = _numba_force_kernel_parallel(
                        num_threads, *kern_args,
                        self._nb_fx_temp, self._nb_fy_temp, self._nb_fz_temp,
                        self._nb_tx_temp, self._nb_ty_temp, self._nb_tz_temp,
                        self._nb_mvn_temp, self._nb_mat_temp,
                    )
                else:
                    nb_max_vn, nb_min_at = _numba_force_kernel(*kern_args)

                contacts_2r: List[Tuple[int, int]] = []
                # 用 _pair_hist / _pair_started_arr / _pair_index 数组化存储历史
                # 避免逐对创建 np.array 的开销
                new_pair_hist = np.zeros(m * 3, dtype=np.float64)
                new_pair_started = np.zeros(m, dtype=np.float64)
                new_pair_index: Dict[Tuple[int,int], int] = {}
                new_pair_hist[:m*3] = hist_out[:m*3]
                for k in range(m):
                    key = keys[k]
                    new_pair_index[key] = k
                    if contacts_mask[k] > 0.5:
                        new_pair_started[k] = 1.0
                        contacts_2r.append(key)
                self._pair_hist = new_pair_hist
                self._pair_started_arr = new_pair_started
                self._pair_index = new_pair_index
                self._contacts_2r = contacts_2r
                return fx, fy, fz, tau_x, tau_y, tau_z, float(nb_max_vn), float(nb_min_at)

            self._contacts_2r = []
            return fx, fy, fz, tau_x, tau_y, tau_z, 0.0, 1e30

        idx_pairs = self._get_neighbor_pairs()

        if use_fast_numba_idx and idx_pairs:
            m = len(idx_pairs)
            self._ensure_nb_buf(m)
            i_idx = self._nb_i_idx[:m]
            j_idx = self._nb_j_idx[:m]
            dx_arr = self._nb_dx[:m]
            dy_arr = self._nb_dy[:m]
            dz_arr = self._nb_dz[:m]
            d_arr = self._nb_d[:m]
            hist_in = self._nb_hist_in[:m*3]
            started_in = self._nb_started_in[:m]
            hist_out = self._nb_hist_out[:m*3]
            contacts_mask = self._nb_contacts_mask[:m]
            for k, (i, j) in enumerate(idx_pairs):
                i_idx[k] = i; j_idx[k] = j
                if self._pair_hist is not None and self._pair_started_arr is not None:
                    hist_in[k*3]   = float(self._pair_hist[k*3])
                    hist_in[k*3+1] = float(self._pair_hist[k*3+1])
                    hist_in[k*3+2] = float(self._pair_hist[k*3+2])
                    started_in[k] = float(self._pair_started_arr[k])
                else:
                    key = (i, j) if i < j else (j, i)
                    h_arr = self.tangential_hist.get(key)
                    if h_arr is None:
                        hist_in[k*3] = 0.0; hist_in[k*3+1] = 0.0; hist_in[k*3+2] = 0.0
                    else:
                        hist_in[k*3] = float(h_arr[0]); hist_in[k*3+1] = float(h_arr[1]); hist_in[k*3+2] = float(h_arr[2])
                    started_in[k] = 1.0 if bool(self.contact_started.get(key, False)) else 0.0

            _numba_build_geometry_from_idx(
                self.state.x, self.state.y, self.state.z,
                float(self.Lx), float(self.Ly), float(self.Lz),
                i_idx, j_idx, dx_arr, dy_arr, dz_arr, d_arr,
            )

            eta_c = float(getattr(self.jkr, "eta_c", 0.0))
            
            kern_args = (
                n,
                i_idx, j_idx, dx_arr, dy_arr, dz_arr, d_arr,
                self.state.vx, self.state.vy, self.state.vz,
                self.state.omega_x, self.state.omega_y, self.state.omega_z,
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
                fx, fy, fz,
                tau_x, tau_y, tau_z,
                hist_out, contacts_mask,
            )
            if use_parallel:
                self._ensure_parallel_buf(n)
                nb_max_vn, nb_min_at = _numba_force_kernel_parallel(
                    num_threads, *kern_args,
                    self._nb_fx_temp, self._nb_fy_temp, self._nb_fz_temp,
                    self._nb_tx_temp, self._nb_ty_temp, self._nb_tz_temp,
                    self._nb_mvn_temp, self._nb_mat_temp,
                )
            else:
                nb_max_vn, nb_min_at = _numba_force_kernel(*kern_args)
            contacts_2r: List[Tuple[int,int]] = []
            if self._pair_hist is not None and self._pair_started_arr is not None:
                self._pair_hist[:m*3] = hist_out
                for k, (i, j) in enumerate(idx_pairs):
                    if contacts_mask[k] > 0.5:
                        self._pair_started_arr[k] = 1.0
                        key = (i, j) if i < j else (j, i)
                        contacts_2r.append(key)
            else:
                for k, (i, j) in enumerate(idx_pairs):
                    key = (i, j) if i < j else (j, i)
                    self.tangential_hist[key] = np.array([float(hist_out[k*3]), float(hist_out[k*3+1]), float(hist_out[k*3+2])])
                    if contacts_mask[k] > 0.5:
                        self.contact_started[key] = True
                        contacts_2r.append(key)
            self._contacts_2r = contacts_2r
            return fx, fy, fz, tau_x, tau_y, tau_z, float(nb_max_vn), float(nb_min_at)

        # 通用路径：需要 C++、团簇冻结或未启用邻域缓存时
        pairs: List[Tuple[int, int, float, float, float, float]] = []
        r = self.state.r
        max_dist = 2.0 * r + max(0.0, float(self.skin))
        max_dist2 = max_dist * max_dist
        for (i, j) in idx_pairs:
            dx = float(self.state.x[j] - self.state.x[i])
            dy = float(self.state.y[j] - self.state.y[i])
            dz = float(self.state.z[j] - self.state.z[i])
            dx, dy, dz = self._minimal_image(dx, dy, dz)
            d2 = dx * dx + dy * dy + dz * dz
            if d2 <= max_dist2:
                d = math.sqrt(d2)
                pairs.append((i, j, dx, dy, dz, d))

        # Numba 通用路径
        if (_NUMBA_AVAILABLE) and bool(getattr(self.sim, "use_numba", False)) and (len(pairs) > 0):
            m = len(pairs)
            self._ensure_nb_buf(m)
            i_idx = self._nb_i_idx[:m]
            j_idx = self._nb_j_idx[:m]
            dx_arr = self._nb_dx[:m]
            dy_arr = self._nb_dy[:m]
            dz_arr = self._nb_dz[:m]
            d_arr = self._nb_d[:m]
            hist_in = self._nb_hist_in[:m*3]
            started_in = self._nb_started_in[:m]
            hist_out = self._nb_hist_out[:m*3]
            contacts_mask = self._nb_contacts_mask[:m]
            keys: List[Tuple[int,int]] = []
            for k, (i, j, dx, dy, dz, d) in enumerate(pairs):
                i_idx[k] = i; j_idx[k] = j
                dx_arr[k] = dx; dy_arr[k] = dy; dz_arr[k] = dz; d_arr[k] = d
                key = (i, j) if i < j else (j, i)
                keys.append(key)
                h_arr = self.tangential_hist.get(key)
                if h_arr is None:
                    hist_in[k*3] = 0.0; hist_in[k*3+1] = 0.0; hist_in[k*3+2] = 0.0
                else:
                    hist_in[k*3] = float(h_arr[0]); hist_in[k*3+1] = float(h_arr[1]); hist_in[k*3+2] = float(h_arr[2])
                started_in[k] = 1.0 if bool(self.contact_started.get(key, False)) else 0.0
            eta_c = float(getattr(self.jkr, "eta_c", 0.0))
            
            kern_args = (
                n,
                i_idx, j_idx, dx_arr, dy_arr, dz_arr, d_arr,
                self.state.vx, self.state.vy, self.state.vz,
                self.state.omega_x, self.state.omega_y, self.state.omega_z,
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
                fx, fy, fz,
                tau_x, tau_y, tau_z,
                hist_out, contacts_mask,
            )
            if use_parallel:
                self._ensure_parallel_buf(n)
                nb_max_vn, nb_min_at = _numba_force_kernel_parallel(
                    num_threads, *kern_args,
                    self._nb_fx_temp, self._nb_fy_temp, self._nb_fz_temp,
                    self._nb_tx_temp, self._nb_ty_temp, self._nb_tz_temp,
                    self._nb_mvn_temp, self._nb_mat_temp,
                )
            else:
                nb_max_vn, nb_min_at = _numba_force_kernel(*kern_args)
            contacts_2r: List[Tuple[int,int]] = []
            for k, key in enumerate(keys):
                self.tangential_hist[key] = np.array([float(hist_out[k*3]), float(hist_out[k*3+1]), float(hist_out[k*3+2])])
                if contacts_mask[k] > 0.5:
                    self.contact_started[key] = True
                    contacts_2r.append(key)
            self._contacts_2r = contacts_2r
            return fx, fy, fz, tau_x, tau_y, tau_z, float(nb_max_vn), float(nb_min_at)

        # Python 单线程回退路径
        contacts_2r: List[Tuple[int,int]] = []
        r = self.state.r
        mu = 0.5 * self.state.mass
        py_max_abs_vn = 0.0
        py_min_approach = 1e30

        for (i, j, dx, dy, dz, d) in pairs:
            inv_d = 1.0 / (d + 1e-18)
            nx = dx * inv_d
            ny = dy * inv_d
            nz = dz * inv_d
            delta_n = 2.0 * r - d

            dvx = self.state.vx[j] - self.state.vx[i]
            dvy = self.state.vy[j] - self.state.vy[i]
            dvz = self.state.vz[j] - self.state.vz[i]
            v_rel_n = dvx * nx + dvy * ny + dvz * nz

            # 接触点相对速度
            wx = self.state.omega_x[i] + self.state.omega_x[j]
            wy = self.state.omega_y[i] + self.state.omega_y[j]
            wz = self.state.omega_z[i] + self.state.omega_z[j]
            cross_x = wy * nz - wz * ny
            cross_y = wz * nx - wx * nz
            cross_z = wx * ny - wy * nx
            vt_x = dvx - v_rel_n * nx - r * cross_x
            vt_y = dvy - v_rel_n * ny - r * cross_y
            vt_z = dvz - v_rel_n * nz - r * cross_z

            key = (i, j) if i < j else (j, i)
            if bool(getattr(self.sim, "use_neighbor_cache", False)) and self._pair_index:
                idx = self._pair_index.get(key, -1)
                started = (self._pair_started_arr[idx] > 0.5) if (idx >= 0 and self._pair_started_arr is not None) else False
            else:
                started = bool(self.contact_started.get(key, False))

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
            Ft_x = 0.0; Ft_y = 0.0; Ft_z = 0.0
            a_contact = float(self.jkr.last_newton_a or 0.0)
            if Fn != 0.0 and a_contact > 0.0:
                if bool(getattr(self.sim, "use_neighbor_cache", False)) and self._pair_index:
                    idx = self._pair_index.get(key, -1)
                    if idx >= 0 and self._pair_hist is not None and self._pair_started_arr is not None:
                        dth_x = float(self._pair_hist[idx*3])
                        dth_y = float(self._pair_hist[idx*3+1])
                        dth_z = float(self._pair_hist[idx*3+2])
                    else:
                        dth_x = 0.0; dth_y = 0.0; dth_z = 0.0
                    Ft_x, Ft_y, Ft_z, new_x, new_y, new_z = self.tan.force_3d(
                        a_contact, dth_x, dth_y, dth_z, vt_x, vt_y, vt_z,
                        nx, ny, nz, self.sim.dt, Fn, self.jkr.F_c, mu)
                    if idx >= 0 and self._pair_hist is not None and self._pair_started_arr is not None:
                        self._pair_hist[idx*3]   = new_x
                        self._pair_hist[idx*3+1] = new_y
                        self._pair_hist[idx*3+2] = new_z
                        self._pair_started_arr[idx] = 1.0
                    contacts_2r.append(key)
                else:
                    h_arr = self.tangential_hist.get(key, np.zeros(3))
                    Ft_x, Ft_y, Ft_z, new_x, new_y, new_z = self.tan.force_3d(
                        a_contact, float(h_arr[0]), float(h_arr[1]), float(h_arr[2]),
                        vt_x, vt_y, vt_z, nx, ny, nz,
                        self.sim.dt, Fn, self.jkr.F_c, mu)
                    self.tangential_hist[key] = np.array([new_x, new_y, new_z])
                    contacts_2r.append(key)
                    self.contact_started[key] = True

            Fx_total = Fn * nx + Ft_x
            Fy_total = Fn * ny + Ft_y
            Fz_total = Fn * nz + Ft_z
            fx[i] -= Fx_total; fy[i] -= Fy_total; fz[i] -= Fz_total
            fx[j] += Fx_total; fy[j] += Fy_total; fz[j] += Fz_total

            # 按标准公式 τ = r × F_t 计算：
            # n 定义为 i->j，当前 Ft 定义为作用在 j 上的切向力。
            # 对 i：r_i = +r n, F_t,i = -Ft；对 j：r_j = -r n, F_t,j = +Ft。
            # 因而两者代数形式一致，均为 -r (n × Ft)。
            torque_x = -r * (ny * Ft_z - nz * Ft_y)
            torque_y = -r * (nz * Ft_x - nx * Ft_z)
            torque_z = -r * (nx * Ft_y - ny * Ft_x)
            tau_x[i] += torque_x; tau_y[i] += torque_y; tau_z[i] += torque_z
            tau_x[j] += torque_x; tau_y[j] += torque_y; tau_z[j] += torque_z

            if delta_n > 0.0:
                T_roll = self.mat.mu_r * abs(Fn) * r
                oi_mag = math.sqrt(self.state.omega_x[i]**2 + self.state.omega_y[i]**2 + self.state.omega_z[i]**2)
                if oi_mag > 1e-18:
                    inv_oi = 1.0 / oi_mag
                    tau_x[i] += -T_roll * self.state.omega_x[i] * inv_oi
                    tau_y[i] += -T_roll * self.state.omega_y[i] * inv_oi
                    tau_z[i] += -T_roll * self.state.omega_z[i] * inv_oi
                oj_mag = math.sqrt(self.state.omega_x[j]**2 + self.state.omega_y[j]**2 + self.state.omega_z[j]**2)
                if oj_mag > 1e-18:
                    inv_oj = 1.0 / oj_mag
                    tau_x[j] += -T_roll * self.state.omega_x[j] * inv_oj
                    tau_y[j] += -T_roll * self.state.omega_y[j] * inv_oj
                    tau_z[j] += -T_roll * self.state.omega_z[j] * inv_oj

        self._contacts_2r = contacts_2r
        return fx, fy, fz, tau_x, tau_y, tau_z, py_max_abs_vn, py_min_approach

    def _apply_periodic_wrap(self) -> None:
        self.state.x = (self.state.x + 0.5 * self.Lx) % self.Lx - 0.5 * self.Lx
        self.state.y = (self.state.y + 0.5 * self.Ly) % self.Ly - 0.5 * self.Ly
        self.state.z = (self.state.z + 0.5 * self.Lz) % self.Lz - 0.5 * self.Lz

    def _integrate(self, fx: np.ndarray, fy: np.ndarray, fz: np.ndarray,
                   tau_x: np.ndarray, tau_y: np.ndarray, tau_z: np.ndarray,
                   half_step: bool = False) -> None:
        """Velocity-Verlet integrator."""
        dt = self.sim.dt
        m = self.state.mass
        I_p = 0.4 * m * (self.state.r ** 2)

        ax = fx / m
        ay = fy / m
        az = fz / m
        alpha_x = tau_x / I_p
        alpha_y = tau_y / I_p
        alpha_z = tau_z / I_p

        if not half_step: # Full step update for velocity and angular velocity
            self.state.vx += ax * dt
            self.state.vy += ay * dt
            self.state.vz += az * dt
            self.state.omega_x += alpha_x * dt
            self.state.omega_y += alpha_y * dt
            self.state.omega_z += alpha_z * dt
        else: # Half step for velocity, full step for position
            # v(t + dt/2) = v(t) + a(t) * dt/2
            self.state.vx += ax * dt * 0.5
            self.state.vy += ay * dt * 0.5
            self.state.vz += az * dt * 0.5
            
            # ω(t + dt/2) = ω(t) + α(t) * dt/2
            self.state.omega_x += alpha_x * dt * 0.5
            self.state.omega_y += alpha_y * dt * 0.5
            self.state.omega_z += alpha_z * dt * 0.5

            # x(t + dt) = x(t) + v(t + dt/2) * dt
            self.state.x += self.state.vx * dt
            self.state.y += self.state.vy * dt
            self.state.z += self.state.vz * dt

            # Update orientation q(t + dt) using ω(t + dt/2)
            omega = np.stack([self.state.omega_x, self.state.omega_y, self.state.omega_z], axis=-1)
            self.state.q = update_orientations(self.state.q, omega, dt)

    def _update_velocities_full_step(self, fx: np.ndarray, fy: np.ndarray, fz: np.ndarray,
                                     tau_x: np.ndarray, tau_y: np.ndarray, tau_z: np.ndarray) -> None:
        """Second half of Velocity-Verlet for velocities."""
        dt = self.sim.dt
        m = self.state.mass
        I_p = 0.4 * m * (self.state.r ** 2)
        
        ax = fx / m
        ay = fy / m
        az = fz / m
        alpha_x = tau_x / I_p
        alpha_y = tau_y / I_p
        alpha_z = tau_z / I_p

        # v(t + dt) = v(t + dt/2) + a(t + dt) * dt/2
        self.state.vx += ax * dt * 0.5
        self.state.vy += ay * dt * 0.5
        self.state.vz += az * dt * 0.5
        
        # ω(t + dt) = ω(t + dt/2) + α(t + dt) * dt/2
        self.state.omega_x += alpha_x * dt * 0.5
        self.state.omega_y += alpha_y * dt * 0.5
        self.state.omega_z += alpha_z * dt * 0.5

    def _energies(self) -> Tuple[float, float, float]:
        m = self.state.mass
        KE_trans = 0.5 * m * float(np.sum(self.state.vx**2 + self.state.vy**2 + self.state.vz**2))
        I = 0.4 * m * (self.state.r ** 2)
        KE_rot = 0.5 * I * float(np.sum(self.state.omega_x**2 + self.state.omega_y**2 + self.state.omega_z**2))
        return KE_trans, KE_rot, KE_trans + KE_rot

    def _cluster_metrics(self) -> Tuple[int, float, float]:
        n = int(self.state.size())
        if n <= 0:
            return 0, 0.0, 0.0

        contacts = self._contacts_2r if self._contacts_2r else []
        comps = connected_components_from_contacts(n, contacts)
        cluster_count = int(len(comps))

        degree = np.zeros(n, dtype=np.int64)
        for i, j in contacts:
            degree[int(i)] += 1
            degree[int(j)] += 1
        coordination_mean = float(np.mean(degree))

        rg_sum = 0.0
        x = self.state.x
        y = self.state.y
        z = self.state.z
        use_periodic = (self.Lx > 0.0) and (self.Ly > 0.0) and (self.Lz > 0.0)
        for comp in comps:
            if not comp:
                continue
            if len(comp) == 1:
                continue
            mb = np.asarray(comp, dtype=np.int64)
            ref = int(mb[0])
            dx = x[mb] - x[ref]
            dy = y[mb] - y[ref]
            dz = z[mb] - z[ref]
            if use_periodic:
                dx -= np.round(dx / self.Lx) * self.Lx
                dy -= np.round(dy / self.Ly) * self.Ly
                dz -= np.round(dz / self.Lz) * self.Lz
            cx = float(np.mean(dx))
            cy = float(np.mean(dy))
            cz = float(np.mean(dz))
            rg = float(np.sqrt(np.mean((dx - cx) ** 2 + (dy - cy) ** 2 + (dz - cz) ** 2)))
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
                "sim.Lz": self.Lz,
                "sim.use_numba": getattr(self.sim, "use_numba", False),
                "sim.use_numba_parallel": getattr(self.sim, "use_numba_parallel", False),
                "sim.use_cpp": getattr(self.sim, "use_cpp", False),
                "sim.trajectory_record": getattr(self.sim, "trajectory_record", True),
                "sim.trajectory_stride": getattr(self.sim, "trajectory_stride", 100),
                "sim.trajectory_filename": getattr(self.sim, "trajectory_filename", "trajectory.h5"),
                "sim.adaptive_timestep": getattr(self.sim, "adaptive_timestep", False),
                "sim.adaptive_dt_min_factor": self._adaptive_dt_min_factor,
                "sim.adaptive_dt_max_factor": self._adaptive_dt_max_factor,
                "sim.adaptive_vmax_ema_alpha": self._adaptive_vmax_ema_alpha,
                "sim.adaptive_vmax_init": self._vmax_init,
                "state.N": int(self.state.size()),
            }

            if bool(getattr(self.sim, "trajectory_record", True)):
                self.traj.start(
                    int(self.state.size()),
                    Lx=float(self.Lx),
                    Ly=float(self.Ly),
                    Lz=float(self.Lz),
                    metadata=param_dict,
                )

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
                Lz=self.Lz,
                dt_info=init_dt_info,
            )
            KE_t0, KE_r0, KE_sum0 = self._energies()
            vmax0 = float(np.max(np.sqrt(self.state.vx**2 + self.state.vy**2 + self.state.vz**2)))
            cluster_count0, coord_mean0, rg_mean0 = self._cluster_metrics()
            self.report.write_step(0.0, KE_t0, KE_r0, KE_sum0, vmax0, cluster_count0, coord_mean0, rg_mean0)

        if bool(getattr(self.sim, "trajectory_record", True)):
            self.traj.append(
                step=0,
                t=0.0,
                x=self.state.x,
                y=self.state.y,
                z=self.state.z,
                vx=self.state.vx,
                vy=self.state.vy,
                vz=self.state.vz,
            )
        snap_idx = 0
        # 统计
        n_nonbase_steps = 0
        n_total_steps = 0

        while t < self.sim.t_end:
            # ---- 自适应时间步长决策（基于当前全局最大速度） ----
            current_dt = self._dt_normal
            dt_factor = 1.0
            vmax_now = float(np.max(np.sqrt(self.state.vx * self.state.vx + self.state.vy * self.state.vy + self.state.vz * self.state.vz)))
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

            fx, fy, fz, tau_x, tau_y, tau_z, max_abs_vn, min_approach_time = self._compute_forces()

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
                                Lz=self.Lz,
                                dt_info=dt_info,
                                max_abs_vn=max_abs_vn)
                KE_t, KE_r, KE_sum = self._energies()
                vmax = float(np.max(np.sqrt(self.state.vx**2 + self.state.vy**2 + self.state.vz**2)))
                cluster_count, coord_mean, rg_mean = self._cluster_metrics()
                self.report.write_step(t, KE_t, KE_r, KE_sum, vmax, cluster_count, coord_mean, rg_mean)
                next_snap += snap_dt

            # Velocity-Verlet:
            # 1) v,w 半步推进 + x,q 全步推进
            self._integrate(fx, fy, fz, tau_x, tau_y, tau_z, half_step=True)
            self._apply_periodic_wrap()
            # 2) 在 t+dt 重新计算力与力矩
            fx_new, fy_new, fz_new, tau_x_new, tau_y_new, tau_z_new, _max_abs_vn2, _min_approach_time2 = self._compute_forces()
            # 3) 完成 v,w 的第二半步
            self._update_velocities_full_step(fx_new, fy_new, fz_new, tau_x_new, tau_y_new, tau_z_new)
            t += current_dt
            step += 1

            if bool(getattr(self.sim, "trajectory_record", True)) and self.traj.should_record(step):
                self.traj.append(
                    step=step,
                    t=t,
                    x=self.state.x,
                    y=self.state.y,
                    z=self.state.z,
                    vx=self.state.vx,
                    vy=self.state.vy,
                    vz=self.state.vz,
                )

        # 恢复基准步长
        self.sim.dt = self._dt_normal
        # 结束快照
        self.saver.save(self.state, t, self.Lx, self.Ly, os.path.join(self.sim.snapshot_dir, "final_snapshot.png"), Lz=self.Lz)
        KE_tf, KE_rf, KE_sumf = self._energies()
        vmaxf = float(np.max(np.sqrt(self.state.vx**2 + self.state.vy**2 + self.state.vz**2)))
        cluster_countf, coord_meanf, rg_meanf = self._cluster_metrics()
        self.report.write_step(t, KE_tf, KE_rf, KE_sumf, vmaxf, cluster_countf, coord_meanf, rg_meanf)
        elapsed = time.perf_counter() - start_time

        if bool(getattr(self.sim, "trajectory_record", True)) and self.traj.last_step != step:
            self.traj.append(
                step=step,
                t=t,
                x=self.state.x,
                y=self.state.y,
                z=self.state.z,
                vx=self.state.vx,
                vy=self.state.vy,
                vz=self.state.vz,
            )

        self.report.write_runtime(elapsed)
        self.report.close()
        if bool(getattr(self.sim, "trajectory_record", True)):
            self.traj.close(runtime_seconds=elapsed)
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
