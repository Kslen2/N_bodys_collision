from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class JKRParams:
    """参数集

    E_star : 等效弹性模量 E* (Pa)
    gamma  : 有效界面黏附能 Γ (J/m^2)
    r      : 单个粒子半径 r (=R1=R2) (m)
    e_n    : 法向恢复系数（0<e_n<1）

    注意：理论上 Γ=γ1+γ2-2√(γ1γ2)，当前代码仍将输入的 materials.gamma
    视为等效 Γ，以保持与既有参数含义兼容。
    """

    E_star: float
    gamma: float
    r: float
    e_n: float  # 法向恢复系数，用于 Tsuji 阻尼公式


class JKRNormal:
    """EDEM 标准 JKR 黏附接触模型（法向）。

    按 EDEM 官方公式实现：
    - 法向力:     F_n = 4E*a³/(3R*) - √(8πΓE*a³)
    - 重叠量:     α = a²/R* - √(2πΓa/E*)
    - 拉脱力:     F_c = (3/2)πR*Γ
    - 临界重叠:   α_f = (3F_c²/(16R*E*²))^(1/3)

    接触状态逻辑: 未接触 / 黏附接触 / 接触断裂
    """

    def __init__(self, prm: JKRParams) -> None:
        self.p = prm

        # === 一、基础衍生参数 ===
        # 当前实现为两等半径、相同材料：R1=R2=r, E1=E2, ν1=ν2
        # 等效半径: 1/R* = 1/R1 + 1/R2  → R* = (R1 R2)/(R1+R2) = r/2
        self.R_eff = 0.5 * self.p.r  # R*

        # 等效弹性模量: 1/E* = (1-ν1^2)/E1 + (1-ν2^2)/E2
        # 已在 materials.E_star 中预先计算, 这里直接使用
        self.E_star = self.p.E_star

        # 界面黏附能 Γ: 理论上 Γ = γ1 + γ2 - 2√(γ1γ2)
        # 目前将输入的 gamma 直接视为 Γ
        self.Gamma = self.p.gamma

        # === 二、模型特征参数 (EDEM 标准) ===
        # 拉脱力 F_c = (3/2)πΓR*  (注意：EDEM 定义与 Chokshi 不同)
        self.F_c = 1.5 * math.pi * self.Gamma * self.R_eff

        # 平衡接触面半径 a0 = (9πΓR*^2 / E*)^(1/3)
        self.a0 = ((9.0 * math.pi * self.Gamma * (self.R_eff**2)) / self.E_star) ** (1.0 / 3.0)

        # 临界重叠量（断裂时）α_f = (3F_c² / (16 R* E*²))^(1/3)
        # 也可以表示为 α_f = -(3F_c²/(16R*E*²))^(1/3)（负值表示拉伸）
        self.alpha_f = ((3.0 * self.F_c**2) / (16.0 * self.R_eff * self.E_star**2)) ** (1.0 / 3.0)

        # 为兼容性保留 delta_c（临界拉伸重叠量取负）
        self.delta_c = self.alpha_f  # |α_f|，用于判断断裂

        # 法向恢复系数与 Tsuji 阻尼系数相关参数
        # β = ln e / sqrt(ln^2 e + π^2)
        # 为数值稳定对 e_n 做轻微裁剪
        e_n_clamped = min(max(self.p.e_n, 1.0e-4), 0.9999)
        loge = math.log(e_n_clamped)
        if loge == 0.0:
            self.beta_tsuji = 0.0
        else:
            self.beta_tsuji = loge / math.sqrt(loge * loge + math.pi * math.pi)

        # 为 numba 旧接口保留一个拉伸长度标记
        self.eta_c = self.delta_c

        # 预计算常量 2πΓ/E* 用于隐式公式
        self._coeff_alpha = 2.0 * math.pi * self.Gamma / self.E_star

        # 牛顿迭代设置
        self._newton_max_iter = 20
        self._newton_tol = 1.0e-12

        # 调试用：记录最近一次牛顿迭代信息
        self.last_newton_iter: int | None = None
        self.last_newton_a0: float | None = None
        self.last_newton_a: float | None = None

        # 插值近似相关（可选）：默认关闭，由上层根据配置决定是否启用
        self.use_interp: bool = False
        self._interp_ready: bool = False
        self._interp_alpha_min: float | None = None
        self._interp_alpha_max: float | None = None
        self._interp_dalpha: float | None = None
        self._interp_a: list[float] | None = None
        self._interp_F: list[float] | None = None

    # === 三、隐式关系 α(a) 及其导数 (EDEM 标准) ===
    def _alpha_from_a(self, a: float) -> float:
        """由接触半径 a 计算法向重叠量 α（EDEM 公式3）。

        α = a²/R* - √(2πΓa/E*)
        """
        if a <= 0.0:
            return 0.0
        # α = a²/R* - √(2πΓa/E*)
        return (a * a) / self.R_eff - math.sqrt(self._coeff_alpha * a)

    def _d_alpha_da(self, a: float) -> float:
        """α(a) 对 a 的导数 dα/da。

        dα/da = 2a/R* - (1/2)√(2πΓ/(E*a))
        """
        if a <= 0.0:
            return 0.0
        return 2.0 * a / self.R_eff - 0.5 * math.sqrt(self._coeff_alpha / a)

    # === 四补充、预计算插值表 ===
    def _build_interp_table(self, n_points: int = 512, alpha_factor: float = 5.0) -> None:
        """构建 JKR 法向力与接触半径关于重叠量 α 的插值表。

        仅在 use_interp=True 且首次使用时调用一次，后续查询走线性插值。
        插值区间选择 [-alpha_f, alpha_max]，其中 alpha_max 取一个
        相对保守的上界（这里选 5*a0，对一般碰撞足够覆盖）。
        区间外仍可回退到牛顿迭代，保证物理正确性。
        """
        alpha_min = -self.delta_c
        # 可调上界：alpha_max = factor * a0（默认 5.0）
        if not math.isfinite(alpha_factor) or alpha_factor <= 0.0:
            alpha_factor = 5.0
        alpha_max = float(alpha_factor) * self.a0
        if n_points < 2:
            n_points = 2
        dalpha = (alpha_max - alpha_min) / (n_points - 1)

        a_list: list[float] = [0.0] * n_points
        F_list: list[float] = [0.0] * n_points

        for i in range(n_points):
            alpha = alpha_min + i * dalpha
            a = self._solve_contact_radius(alpha)
            a_list[i] = a
            if a > 0.0:
                a3 = a * a * a
                F_hertz = (4.0 * self.E_star * a3) / (3.0 * self.R_eff)
                F_adhesion = math.sqrt(8.0 * math.pi * self.Gamma * self.E_star * a3)
                F_elastic = F_hertz - F_adhesion
            else:
                F_elastic = 0.0
            F_list[i] = F_elastic

        self._interp_alpha_min = alpha_min
        self._interp_alpha_max = alpha_max
        self._interp_dalpha = dalpha
        self._interp_a = a_list
        self._interp_F = F_list
        self._interp_ready = True

    def _interp_lookup(self, alpha: float) -> tuple[float, float]:
        """在插值表中查询给定重叠量 alpha 对应的 (F_elastic, a_contact)。"""
        if (not self._interp_ready) or self._interp_a is None or self._interp_F is None:
            # 理论上不应发生，调用前会保证已构建
            return 0.0, 0.0

        alpha_min = float(self._interp_alpha_min)
        alpha_max = float(self._interp_alpha_max)
        dalpha = float(self._interp_dalpha)
        a_tab = self._interp_a
        F_tab = self._interp_F
        n = len(a_tab)

        # 区间外统一返回 0，由上层决定是否回退牛顿
        if alpha <= alpha_min:
            return 0.0, 0.0
        if alpha >= alpha_max:
            return 0.0, 0.0

        t = (alpha - alpha_min) / dalpha
        idx = int(t)
        if idx < 0:
            idx = 0
            frac = 0.0
        elif idx >= n - 1:
            idx = n - 2
            frac = 1.0
        else:
            frac = t - idx

        a0 = a_tab[idx]
        a1 = a_tab[idx + 1]
        F0 = F_tab[idx]
        F1 = F_tab[idx + 1]
        a = a0 + (a1 - a0) * frac
        F_elastic = F0 + (F1 - F0) * frac
        return F_elastic, a

    # === 四、牛顿迭代：给定重叠量 α 求解 a ===
    def _solve_contact_radius(self, alpha: float) -> float:
        """给定法向重叠量 α，通过牛顿迭代求解接触半径 a。

        若 α <= -α_f，视为接触断裂，返回 a=0。
        """
        # 接触断裂判据
        if alpha <= -self.delta_c:
            self.last_newton_iter = 0
            self.last_newton_a0 = 0.0
            self.last_newton_a = 0.0
            return 0.0

        # 初始迭代猜测：
        # 压缩区 (α>=0): 近似用 Hertz a ≈ sqrt(R* α) 修正到不小于 0.5*a0
        # 黏附拉伸区 (-α_f < α < 0): 从平衡半径 a0 起步
        if alpha >= 0.0:
            a_init = math.sqrt(self.R_eff * max(alpha, 1.0e-18))
            a = max(a_init, 0.5 * self.a0)
        else:
            a = self.a0

        self.last_newton_a0 = a
        it = 0

        for it in range(1, self._newton_max_iter + 1):
            alpha_model = self._alpha_from_a(a)
            f = alpha_model - alpha
            if abs(f) < self._newton_tol * max(1.0, abs(alpha)):
                break
            d_alpha_da = self._d_alpha_da(a)
            if abs(d_alpha_da) < 1.0e-30:
                break
            a_new = a - f / d_alpha_da
            # 保证正值并避免数值震荡
            if a_new <= 0.0 or not math.isfinite(a_new):
                a_new = 0.5 * a
            # 限制物理上合理的上界
            a = min(a_new, 10.0 * self.a0)

        self.last_newton_iter = it
        self.last_newton_a = a
        return max(a, 0.0)

    # === 五、法向力计算 ===
    def force(self, delta_n: float, v_rel_n: float, mu_contact: float, contact_started: bool) -> float:
        """返回法向力 F_n（i->j 为正）。

        参数:
        - delta_n: 宏观法向重叠量 α = 2r - d (>0 压缩, <0 拉伸)
        - v_rel_n: 法向相对速度
        - mu_contact: 接触对等效质量 (两等质量 m: μ = m/2)
        - contact_started: 该对颗粒历史上是否发生过真实接触

        接触状态逻辑:
        1. 未接触阶段: α < 0 且从未接触 → F_n = 0
        2. 黏附/接触阶段: α ≥ 0 或 (-α_f < α < 0) → 通过隐式关系求 a 再算 F_n
        3. 接触断裂阶段: α ≤ -α_f → F_n = 0, a = 0
        """

        # 1) 未接触阶段（首次物理接触前，α<0 且无黏附历史）
        if (delta_n < 0.0) and (not contact_started):
            return 0.0

        # 2) 接触断裂阶段：α ≤ -α_f → 视为完全分离
        if delta_n <= -self.delta_c:
            return 0.0

        # 3) 黏附/接触阶段：
        #    - 默认：牛顿迭代求 a
        #    - 若 use_interp=True：优先用插值表近似 a 与 F_elastic，
        #      超出插值区间时回退牛顿迭代，保证物理正确性。
        if self.use_interp:
            if not self._interp_ready:
                self._build_interp_table()

            F_elastic, a = self._interp_lookup(delta_n)
            if (a <= 0.0) or (F_elastic == 0.0):
                # 插值区间外或无接触，回退牛顿迭代以保证精度
                a = self._solve_contact_radius(delta_n)
                if a <= 0.0:
                    return 0.0
                a3 = a * a * a
                F_hertz = (4.0 * self.E_star * a3) / (3.0 * self.R_eff)
                F_adhesion = math.sqrt(8.0 * math.pi * self.Gamma * self.E_star * a3)
                F_elastic = F_hertz - F_adhesion
        else:
            a = self._solve_contact_radius(delta_n)
            if a <= 0.0:
                return 0.0
            a3 = a * a * a
            F_hertz = (4.0 * self.E_star * a3) / (3.0 * self.R_eff)
            F_adhesion = math.sqrt(8.0 * math.pi * self.Gamma * self.E_star * a3)
            F_elastic = F_hertz - F_adhesion

        # 4) 阻尼项：基于当前切线刚度估算 k_n = dF_n/dδ_n
        # 这里采用文献 Tsuji 等提出的法向阻尼力公式：
        #   F_dn = -2 * sqrt(5/6) * β * sqrt(k_n * m*) * v^n
        # 其中 β = ln e / sqrt(ln^2 e + π^2)，k_n = 2 E* a，m* 为等效质量
        if mu_contact > 0.0 and self.beta_tsuji != 0.0:
            k_n_tsuji = 2.0 * self.E_star * a
            if k_n_tsuji > 0.0:
                eta_n = -2.0 * math.sqrt(5.0 / 6.0) * self.beta_tsuji * math.sqrt(k_n_tsuji * mu_contact)
                F_damping = -eta_n * v_rel_n
            else:
                F_damping = 0.0
        else:
            F_damping = 0.0

        return F_elastic + F_damping
