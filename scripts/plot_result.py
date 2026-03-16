from __future__ import annotations

import os
import matplotlib.pyplot as plt

dir = 'data/N10_2.5e-10dt_50e-5_verlet1_h54'  # 以项目根目录为基准的相对路径，或绝对路径


def _resolve_target(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", path))


def _parse_report(report: str) -> dict[str, list[float]]:
    data: dict[str, list[float]] = {
        "t": [],
        "ke_trans": [],
        "ke_rot": [],
        "ke_total": [],
        "vmax": [],
        "cluster_count": [],
        "coord_mean": [],
        "rg_mean": [],
    }
    with open(report, "r", encoding="utf-8") as f:
        for ln in f:
            stripped = ln.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()

            # 基础列：t, KE_trans, KE_rot, KE_total
            if len(parts) < 4:
                continue

            try:
                t = float(parts[0])
                ke_t = float(parts[1])
                ke_r = float(parts[2])
                ke_s = float(parts[3])
            except ValueError:
                continue

            data["t"].append(t)
            data["ke_trans"].append(ke_t)
            data["ke_rot"].append(ke_r)
            data["ke_total"].append(ke_s)

            # 第 5 列：Vmax（若存在）
            if len(parts) >= 5:
                try:
                    data["vmax"].append(float(parts[4]))
                except ValueError:
                    data["vmax"].append(float("nan"))

            # 第 6~8 列：cluster 指标（若存在）
            if len(parts) >= 8:
                try:
                    data["cluster_count"].append(float(parts[5]))
                    data["coord_mean"].append(float(parts[6]))
                    data["rg_mean"].append(float(parts[7]))
                except ValueError:
                    # 遇到坏行则跳过该行的团簇指标，保持长度一致
                    pass

    return data


def _plot_energies(target: str, dir_name: str, data: dict[str, list[float]]) -> None:
    ts = data["t"]
    ket = data["ke_trans"]
    ker = data["ke_rot"]
    kes = data["ke_total"]
    vmax_list = data["vmax"]
    has_vmax = len(vmax_list) == len(ts) and len(vmax_list) > 0

    if has_vmax:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))

    ax1.plot(ts, ket, label="KE_trans", linewidth=1.2)
    ax1.plot(ts, ker, label="KE_rot", linewidth=1.2)
    ax1.plot(ts, kes, label="KE_total", linewidth=1.2, linestyle="--")
    ax1.set_ylabel("Energy (J)")
    ax1.set_title(f"Simulation: {dir_name}")
    ax1.legend(fontsize=9)
    ax1.grid(True, ls="--", alpha=0.3)

    if has_vmax:
        ax2.plot(ts, vmax_list, color="tab:red", linewidth=1.2, label="Vmax")
        ax2.set_xlabel("t (s)")
        ax2.set_ylabel("Vmax (m/s)")
        ax2.legend(fontsize=9)
        ax2.grid(True, ls="--", alpha=0.3)
    else:
        ax1.set_xlabel("t (s)")

    out = os.path.join(target, "energies.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")


def _plot_cluster(target: str, dir_name: str, data: dict[str, list[float]]) -> None:
    ts = data["t"]
    cluster_counts = data["cluster_count"]
    coord_means = data["coord_mean"]
    rg_means = data["rg_mean"]

    has_cluster = (
        len(cluster_counts) == len(ts)
        and len(coord_means) == len(ts)
        and len(rg_means) == len(ts)
        and len(ts) > 0
    )
    if not has_cluster:
        print("Cluster metrics unavailable in report.txt (need >= 8 columns). Skip cluster_evolution.png")
        return

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    axes[0].plot(ts, cluster_counts, linewidth=1.2, color="tab:blue")
    axes[0].set_ylabel("Cluster count")
    axes[0].grid(True, ls="--", alpha=0.3)

    axes[1].plot(ts, coord_means, linewidth=1.2, color="tab:orange")
    axes[1].set_ylabel("Mean coordination")
    axes[1].grid(True, ls="--", alpha=0.3)

    axes[2].plot(ts, rg_means, linewidth=1.2, color="tab:green")
    axes[2].set_xlabel("t (s)")
    axes[2].set_ylabel("Mean cluster $R_g$ (m)")
    axes[2].grid(True, ls="--", alpha=0.3)

    fig.suptitle(f"Cluster evolution: {dir_name}")
    fig.tight_layout()

    out = os.path.join(target, "cluster_evolution.png")
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")


def main() -> None:
    target = _resolve_target(dir)

    report = os.path.join(target, "report.txt")
    if not os.path.exists(report):
        print(f"report.txt not found under: {target}")
        return

    data = _parse_report(report)
    ts = data["t"]

    if not ts:
        print(f"No valid data found in: {report}")
        return

    dir_name = os.path.basename(target)

    _plot_energies(target, dir_name, data)
    _plot_cluster(target, dir_name, data)
    plt.show()


if __name__ == "__main__":
    main()
