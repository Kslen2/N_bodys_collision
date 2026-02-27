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
    """切向接触模型：弹簧 + Tsuji 阻尼 + 黏附摩擦上限。

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

    def force(
        self,
        a: float,
        delta_t_hist: float,
        v_rel_t: float,
        dt: float,
        Fn: float,
        F_c: float,
        mu_contact: float,
    ) -> tuple[float, float]:
        """计算切向总力与更新后的切向位移历史。

        参数:
        - a: 接触半径 a
        - delta_t_hist: 历史切向位移 δ_t
        - v_rel_t: 当前切向相对速度 v^t（含转动修正）
        - dt: 时间步长
        - Fn: 当前法向力 F_n^t（含黏附与阻尼）
        - F_c: 拉脱力幅值 F_c (>0)
        - mu_contact: 等效质量 m*

        返回: (F_t, new_delta_t_hist)
        """

        if a <= 0.0:
            return 0.0, delta_t_hist

        # 切向刚度 k_t = 8 G* a
        k_t = 8.0 * self.p.G_star * a
        if k_t <= 0.0:
            return 0.0, delta_t_hist

        # 更新切向弹簧位移
        delta_t = delta_t_hist + v_rel_t * dt
        F_ct = -k_t * delta_t

        # Tsuji 切向阻尼: F_dt = -η_t v^t,
        # 其中 η_t = -2 sqrt(5/6) β sqrt(k_t m*)
        if self.beta_tsuji != 0.0 and mu_contact > 0.0:
            eta_t = -2.0 * math.sqrt(5.0 / 6.0) * self.beta_tsuji * math.sqrt(k_t * mu_contact)
            F_dt = -eta_t * v_rel_t
        else:
            F_dt = 0.0

        F_trial = F_ct + F_dt

        # 黏附摩擦上限：|F_n^t + 2 F_c|
        F_normal_eff = Fn + 2.0 * F_c
        F_limit_static = self.p.MU_S * abs(F_normal_eff)

        if abs(F_trial) <= F_limit_static:
            # 静摩擦区：切向弹簧 + 阻尼
            return F_trial, delta_t

        # 滑动摩擦：F_t = -μ_k |F_n^t + 2F_c| * sign(v_rel)
        # 摩擦力方向与滑动速度相反
        F_limit_slide = self.p.MU_K * abs(F_normal_eff)
        if F_limit_slide <= 0.0:
            return 0.0, 0.0

        if v_rel_t > 1e-18:
            v_sign = 1.0
        elif v_rel_t < -1e-18:
            v_sign = -1.0
        else:
            v_sign = 0.0
        Ft = -F_limit_slide * v_sign

        # 将弹簧位移截断在最大静摩擦对应的位置，避免无界增长
        delta_t_slip = -(F_limit_static / k_t) * v_sign
        return Ft, delta_t_slip
