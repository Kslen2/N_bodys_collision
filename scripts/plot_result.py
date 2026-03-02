from __future__ import annotations

import os
import matplotlib.pyplot as plt

dir = 'data/30_small_t_30e-5_seed6'  # 以项目根目录为基准的相对路径，或绝对路径


def main() -> None:
    target = dir
    # 支持相对路径：以项目根为基准
    if not os.path.isabs(target):
        target = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", target))

    report = os.path.join(target, "report.txt")
    if not os.path.exists(report):
        print(f"report.txt not found under: {target}")
        return

    # 解析 report.txt: # t(s)  KE_trans  KE_rot  KE_total  Vmax
    ts, ket, ker, kes, vmax_list = [], [], [], [], []
    has_vmax = False
    with open(report, "r", encoding="utf-8") as f:
        for ln in f:
            stripped = ln.strip()
            # 跳过所有注释行（包含参数、runtime 等），只保留纯数据行
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) >= 5:
                t, a, b, c, v = map(float, parts[:5])
                ts.append(t); ket.append(a); ker.append(b); kes.append(c); vmax_list.append(v)
                has_vmax = True
            elif len(parts) == 4:
                t, a, b, c = map(float, parts)
                ts.append(t); ket.append(a); ker.append(b); kes.append(c)
            # 其他长度的行（如可能的将来扩展）一律忽略，保持兼容

    if not ts:
        print(f"No valid data found in: {report}")
        return

    dir_name = os.path.basename(target)

    if has_vmax:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))

    # --- 能量子图 ---
    ax1.plot(ts, ket, label="KE_trans", linewidth=1.2)
    ax1.plot(ts, ker, label="KE_rot", linewidth=1.2)
    ax1.plot(ts, kes, label="KE_total", linewidth=1.2, linestyle="--")
    ax1.set_ylabel("Energy (J)")
    ax1.set_title(f"Simulation: {dir_name}")
    ax1.legend(fontsize=9)
    ax1.grid(True, ls='--', alpha=0.3)

    # --- Vmax 子图 ---
    if has_vmax:
        ax2.plot(ts, vmax_list, color="tab:red", linewidth=1.2, label="Vmax")
        ax2.set_xlabel("t (s)")
        ax2.set_ylabel("Vmax (m/s)")
        ax2.legend(fontsize=9)
        ax2.grid(True, ls='--', alpha=0.3)
    else:
        ax1.set_xlabel("t (s)")

    out = os.path.join(target, "energies.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
