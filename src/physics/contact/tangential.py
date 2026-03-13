from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class TangentialParams:
    MU_S: float  # 最大静摩擦系数
    MU_K: float  # 滑动摩擦系数（Savkoor/Thornton 模型中的 μ_s）
    G_star: float  # 等效剪切模量 G*
    e_n: float  # 恢复系数，用于 Tsuji 阻尼


class TangentialFriction:
    """三维切向接触模型：弹簧 + Tsuji 阻尼 + 黏附摩擦上限。

    采用文献中的公式：
    - 切向弹性力: F_ij^{ct} = -k_t δ_t,  k_t = 8 G* a
    - 切向阻尼力: F_ij^{dt} = -2 sqrt(5/6) β sqrt(k_t m*) v^t
    - 滑动摩擦上限: F_ij^t = -μ_s |F_n^t + 2 F_c|
    """

    def __init__(self, prm: TangentialParams) -> None:
        self.p = prm
        # Tsuji 系数 β = ln e / sqrt(ln^2 e + π^2)
        e_n_clamped = min(max(self.p.e_n, 1.0e-4), 0.9999)
        loge = math.log(e_n_clamped)
        if loge == 0.0:
            self.beta_tsuji = 0.0
        else:
            self.beta_tsuji = loge / math.sqrt(loge * loge + math.pi * math.pi)

    def force_3d(
        self,
        a: float,
        dt_hist_x: float, dt_hist_y: float, dt_hist_z: float,
        vt_x: float, vt_y: float, vt_z: float,
        nx: float, ny: float, nz: float,
        dt: float,
        Fn: float,
        F_c: float,
        mu_contact: float,
    ) -> tuple[float, float, float, float, float, float]:
        """计算三维切向总力与更新后的切向位移历史。

        参数:
        - a: 接触半径 a
        - dt_hist_x/y/z: 历史切向位移向量 δ_t
        - vt_x/y/z: 当前切向相对速度向量（含转动修正）
        - nx/ny/nz: 接触法向单位向量
        - dt: 时间步长
        - Fn: 当前法向力 F_n^t（含黏附与阻尼）
        - F_c: 拉脱力幅值 F_c (>0)
        - mu_contact: 等效质量 m*

        返回: (Ft_x, Ft_y, Ft_z, new_dt_x, new_dt_y, new_dt_z)
        """

        if a <= 0.0:
            return 0.0, 0.0, 0.0, dt_hist_x, dt_hist_y, dt_hist_z

        # 切向刚度 k_t = 8 G* a
        k_t = 8.0 * self.p.G_star * a
        if k_t <= 0.0:
            return 0.0, 0.0, 0.0, dt_hist_x, dt_hist_y, dt_hist_z

        # 更新切向弹簧位移
        delta_x = dt_hist_x + vt_x * dt
        delta_y = dt_hist_y + vt_y * dt
        delta_z = dt_hist_z + vt_z * dt

        # 重新投影到切平面（消除法向漂移）
        dot_dn = delta_x * nx + delta_y * ny + delta_z * nz
        delta_x -= dot_dn * nx
        delta_y -= dot_dn * ny
        delta_z -= dot_dn * nz

        # 弹性力
        Fct_x = -k_t * delta_x
        Fct_y = -k_t * delta_y
        Fct_z = -k_t * delta_z

        # Tsuji 切向阻尼
        if self.beta_tsuji != 0.0 and mu_contact > 0.0:
            eta_t = -2.0 * math.sqrt(5.0 / 6.0) * self.beta_tsuji * math.sqrt(k_t * mu_contact)
            Fdt_x = -eta_t * vt_x
            Fdt_y = -eta_t * vt_y
            Fdt_z = -eta_t * vt_z
        else:
            Fdt_x = Fdt_y = Fdt_z = 0.0

        F_trial_x = Fct_x + Fdt_x
        F_trial_y = Fct_y + Fdt_y
        F_trial_z = Fct_z + Fdt_z
        F_trial_mag = math.sqrt(F_trial_x * F_trial_x + F_trial_y * F_trial_y + F_trial_z * F_trial_z)

        # 黏附摩擦上限：|F_n^t + 2 F_c|
        F_normal_eff = Fn + 2.0 * F_c
        F_limit_static = self.p.MU_S * abs(F_normal_eff)

        if F_trial_mag <= F_limit_static:
            # 静摩擦区：切向弹簧 + 阻尼
            return F_trial_x, F_trial_y, F_trial_z, delta_x, delta_y, delta_z

        # 滑动摩擦
        F_limit_slide = self.p.MU_K * abs(F_normal_eff)
        if F_limit_slide <= 0.0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        v_t_mag = math.sqrt(vt_x * vt_x + vt_y * vt_y + vt_z * vt_z)
        if v_t_mag > 1e-18:
            inv_vt = 1.0 / v_t_mag
            Ft_x = -F_limit_slide * vt_x * inv_vt
            Ft_y = -F_limit_slide * vt_y * inv_vt
            Ft_z = -F_limit_slide * vt_z * inv_vt
            # 将弹簧位移截断在最大静摩擦对应的位置
            slip_scale = F_limit_static / k_t
            d_slip_x = -slip_scale * vt_x * inv_vt
            d_slip_y = -slip_scale * vt_y * inv_vt
            d_slip_z = -slip_scale * vt_z * inv_vt
        else:
            Ft_x = Ft_y = Ft_z = 0.0
            d_slip_x = d_slip_y = d_slip_z = 0.0

        return Ft_x, Ft_y, Ft_z, d_slip_x, d_slip_y, d_slip_z
