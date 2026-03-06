from __future__ import annotations

import os
import matplotlib.pyplot as plt


dir = 'data/40_small_t_20e-5_seed6'  # 可改为具体结果目录，如 data/cluster_collision/N1_..._vyB...


def _resolve_target(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', path))


def main() -> None:
    target = _resolve_target(dir)
    report = os.path.join(target, 'report.txt')
    if not os.path.exists(report):
        print(f'report.txt not found under: {target}')
        return

    ts = []
    cluster_counts = []
    coord_means = []
    rg_means = []

    with open(report, 'r', encoding='utf-8') as f:
        for ln in f:
            stripped = ln.strip()
            if not stripped or stripped.startswith('#'):
                continue
            parts = stripped.split()
            # 新格式：至少 8 列
            # t, KE_trans, KE_rot, KE_total, Vmax, cluster_count, coordination_mean, cluster_rg_mean
            if len(parts) >= 8:
                try:
                    t = float(parts[0])
                    cc = float(parts[5])
                    z_mean = float(parts[6])
                    rg_mean = float(parts[7])
                except ValueError:
                    continue
                ts.append(t)
                cluster_counts.append(cc)
                coord_means.append(z_mean)
                rg_means.append(rg_mean)

    if not ts:
        print('No valid cluster metric data found. Ensure report.txt contains >= 8 columns per data row.')
        return

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    axes[0].plot(ts, cluster_counts, linewidth=1.2, color='tab:blue')
    axes[0].set_ylabel('Cluster count')
    axes[0].grid(True, ls='--', alpha=0.3)

    axes[1].plot(ts, coord_means, linewidth=1.2, color='tab:orange')
    axes[1].set_ylabel('Mean coordination')
    axes[1].grid(True, ls='--', alpha=0.3)

    axes[2].plot(ts, rg_means, linewidth=1.2, color='tab:green')
    axes[2].set_xlabel('t (s)')
    axes[2].set_ylabel('Mean cluster $R_g$ (m)')
    axes[2].grid(True, ls='--', alpha=0.3)

    fig.suptitle(f'Cluster evolution: {os.path.basename(target)}')
    fig.tight_layout()

    out = os.path.join(target, 'cluster_evolution.png')
    fig.savefig(out, dpi=150)
    print(f'Saved: {out}')
    plt.show()


if __name__ == '__main__':
    main()
