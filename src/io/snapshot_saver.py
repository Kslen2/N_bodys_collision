from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: F401

from ..simulation.particle_system import SystemState


def _draw_box_wireframe(ax, Lx: float, Ly: float, Lz: float, **kwargs) -> None:
    """绘制周期边界盒子的线框。"""
    hx, hy, hz = 0.5 * Lx, 0.5 * Ly, 0.5 * Lz
    corners = [
        (-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
        (-hx, -hy, hz),  (hx, -hy, hz),  (hx, hy, hz),  (-hx, hy, hz),
    ]
    edges = [
        (0,1),(1,2),(2,3),(3,0),  # bottom
        (4,5),(5,6),(6,7),(7,4),  # top
        (0,4),(1,5),(2,6),(3,7),  # verticals
    ]
    for i0, i1 in edges:
        xs = [corners[i0][0], corners[i1][0]]
        ys = [corners[i0][1], corners[i1][1]]
        zs = [corners[i0][2], corners[i1][2]]
        ax.plot(xs, ys, zs, **kwargs)


class SnapshotSaver:
    def __init__(self, out_dir: str, snapshot_show_velocity: bool = True) -> None:
        self.out_dir = out_dir
        self.snapshot_show_velocity = bool(snapshot_show_velocity)
        self.vmax_colorbar: float | None = None  # 固定色标上限，首帧自动设定
        os.makedirs(out_dir, exist_ok=True)

    def save(self, state: SystemState, t: float, Lx: float, Ly: float, path: str,
             *, Lz: float | None = None,
             dt_info: str | None = None, max_abs_vn: float | None = None) -> None:
        if Lz is None:
            Lz = max(Lx, Ly)

        fig_inches = 7.0
        dpi = 150
        fig = plt.figure(figsize=(fig_inches, fig_inches))
        ax = fig.add_subplot(111, projection='3d')

        title = f"t = {t:.3e} s  (N={state.size()}, r={state.r:.2e} m)"
        ax.set_title(title, fontsize=9)

        hx, hy, hz = 0.5 * Lx, 0.5 * Ly, 0.5 * Lz
        ax.set_xlim(-hx, hx)
        ax.set_ylim(-hy, hy)
        ax.set_zlim(-hz, hz)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # ---------- 计算与物理半径匹配的散点标记大小 ----------
        # matplotlib scatter 的 s 参数单位是 point²（1 point = 1/72 inch）
        # 3D 投影下的近似：取主轴范围对应图形有效宽度（约 fig 的 60%）
        L_max = max(Lx, Ly, Lz)
        effective_plot_inches = fig_inches * 0.55  # 3D 投影有效绘图区约占图幅 55%
        points_per_data = (effective_plot_inches * 72.0) / L_max
        # 散点标记面积 = π * (r_points)² ，其中 r_points = r * points_per_data
        r_points = state.r * points_per_data
        marker_area = np.pi * r_points ** 2
        # 为极端尺寸设置上下限保证可见性
        marker_area = max(4.0, min(800.0, marker_area))

        # ---------- 绘制粒子 ----------
        vmag = np.sqrt(state.vx**2 + state.vy**2 + state.vz**2)
        # 首帧自动锁定色标上限
        if self.vmax_colorbar is None:
            self.vmax_colorbar = float(np.max(vmag)) if vmag.size > 0 else 1.0
            if self.vmax_colorbar <= 0.0:
                self.vmax_colorbar = 1.0
        sc = ax.scatter(
            state.x, state.y, state.z,
            s=marker_area,
            c=vmag,
            cmap='coolwarm',
            vmin=0.0,
            vmax=self.vmax_colorbar,
            edgecolors='k',
            linewidths=0.3,
            alpha=0.85,
            depthshade=True,
        )
        cbar = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.08)
        cbar.set_label("|v| (m/s)", fontsize=8)

        # ---------- 速度箭头 ----------
        if self.snapshot_show_velocity and vmag.size > 0:
            vmax_val = float(np.max(vmag))
            if vmax_val > 0.0:
                scale_len = 2.0 * state.r
                U = scale_len * (state.vx / vmax_val)
                V = scale_len * (state.vy / vmax_val)
                W = scale_len * (state.vz / vmax_val)
                ax.quiver(
                    state.x, state.y, state.z,
                    U, V, W,
                    length=1.0, normalize=False,
                    color='orange', alpha=0.7, linewidth=0.6, arrow_length_ratio=0.3,
                )

        # ---------- 标注最快粒子 ----------
        if self.snapshot_show_velocity and vmag.size > 0:
            imax = int(np.argmax(vmag))
            v_fast = float(vmag[imax])
            ax.scatter(
                [state.x[imax]], [state.y[imax]], [state.z[imax]],
                s=marker_area * 2.5,
                facecolors='none', edgecolors='red', linewidths=1.8,
                depthshade=False,
            )
            ax.text(
                float(state.x[imax]), float(state.y[imax]), float(state.z[imax]),
                f"  v_max={v_fast:.2e}",
                fontsize=7, color='red',
            )

        # ---------- 盒子线框 ----------
        _draw_box_wireframe(ax, Lx, Ly, Lz, color='gray', lw=0.6, alpha=0.5)

        # ---------- 步长信息与粒子半径标注 ----------
        info_parts = []
        if dt_info:
            info_parts.append(dt_info)
        # info_parts.append(f"r={state.r:.2e} m, 2r={2*state.r:.2e} m")
        ax.text2D(0.02, 0.02, "\n".join(info_parts), transform=ax.transAxes,
                  fontsize=7, color='purple', va='bottom', ha='left',
                  bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='purple', alpha=0.7))

        ax.set_box_aspect([Lx, Ly, Lz])

        fig.tight_layout()
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
