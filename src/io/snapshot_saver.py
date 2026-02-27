from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from ..simulation.particle_system import SystemState


class SnapshotSaver:
    def __init__(self, out_dir: str, snapshot_show_velocity: bool = True) -> None:
        self.out_dir = out_dir
        self.snapshot_show_velocity = bool(snapshot_show_velocity)
        os.makedirs(out_dir, exist_ok=True)

    def save(self, state: SystemState, t: float, Lx: float, Ly: float, path: str,
             *, dt_info: str | None = None, max_abs_vn: float | None = None) -> None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect('equal')
        # 绘图轴采用真实边界，保证颗粒完整显示
        ax.set_xlim(-0.5*Lx, 0.5*Lx)
        ax.set_ylim(-0.5*Ly, 0.5*Ly)
        title = f"t = {t:.3e} s"
        # if max_abs_vn is not None:
        #     title += f"  |v_n|_max = {max_abs_vn:.2e}"
        ax.set_title(title)
        # 用真实半径绘制圆形颗粒（主图层）
        for xi, yi in zip(state.x, state.y):
            circ = Circle((xi, yi), radius=state.r, facecolor='tab:blue', edgecolor='black', linewidth=0.3, alpha=0.8)
            ax.add_patch(circ)
        # 周期边界的显示平铺：在四周绘制复制影像，保证视觉连续性
        offsets = [(-Lx, 0.0), (Lx, 0.0), (0.0, -Ly), (0.0, Ly), (-Lx, -Ly), (-Lx, Ly), (Lx, -Ly), (Lx, Ly)]
        x_min, x_max = -0.5*Lx, 0.5*Lx
        y_min, y_max = -0.5*Ly, 0.5*Ly
        for ox, oy in offsets:
            for xi, yi in zip(state.x, state.y):
                xcp = xi + ox; ycp = yi + oy
                if (x_min - state.r) <= xcp <= (x_max + state.r) and (y_min - state.r) <= ycp <= (y_max + state.r):
                    circ = Circle((xcp, ycp), radius=state.r, facecolor='tab:blue', edgecolor='black', linewidth=0.3, alpha=0.3)
                    ax.add_patch(circ)
        # 速度箭头：按 v/vmax 归一化，最大箭头长度约为 2r（增强可见性）
        if self.snapshot_show_velocity and state.vx is not None and state.vy is not None:
            vmag = np.hypot(state.vx, state.vy)
            if vmag.size > 0:
                vmax = float(np.max(vmag))
                if vmax > 0.0:
                    scale_len = 2.0 * state.r
                    U = scale_len * (state.vx / vmax)
                    V = scale_len * (state.vy / vmax)
                    ax.quiver(state.x, state.y, U, V, angles='xy', scale_units='xy', scale=1.0,
                              color='orange', width=0.004, alpha=0.9)
        # 标注当前快照中速度最快的颗粒（仅在主域内的原始坐标处标注）
        if self.snapshot_show_velocity and state.vx is not None and state.vy is not None:
            vmag = np.hypot(state.vx, state.vy)
            if vmag.size > 0:
                imax = int(np.argmax(vmag))
                x_fast = float(state.x[imax])
                y_fast = float(state.y[imax])
                v_fast = float(vmag[imax])
                # 高亮圈：稍大半径的空心红圈以突出显示
                highlight = Circle((x_fast, y_fast), radius=state.r*1.15, facecolor='none', edgecolor='red', linewidth=1.8, alpha=0.9)
                ax.add_patch(highlight)
                # 在标注点旁打印速度大小
                ax.annotate(f"v_max = {v_fast:.3e} m/s", xy=(x_fast, y_fast), xytext=(8, 8), textcoords='offset points',
                            color='red', fontsize=8, bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='red', alpha=0.6))
        # 绘制两层边界：真实边界（淡色）与视觉边界（内缩 r，深色）
        inset_x_min = -0.5*Lx + state.r
        inset_x_max =  0.5*Lx - state.r
        inset_y_min = -0.5*Ly + state.r
        inset_y_max =  0.5*Ly - state.r
        ax.add_patch(plt.Rectangle((-0.5*Lx, -0.5*Ly), Lx, Ly, fill=False, color='gray', lw=0.8, alpha=0.6))
        # ax.add_patch(plt.Rectangle((inset_x_min, inset_y_min), inset_x_max - inset_x_min, inset_y_max - inset_y_min, fill=False, color='k', lw=1.2))
        ax.grid(True, ls='--', alpha=0.2)
        # 左下角显示步长信息
        if dt_info:
            ax.text(0.02, 0.02, dt_info, transform=ax.transAxes,
                    fontsize=8, color='purple', va='bottom', ha='left',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='purple', alpha=0.7))
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
