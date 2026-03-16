from __future__ import annotations

import plot_result as _plot_result


dir = 'data/N10_1e-9dt_10e-5_verlet_h54'  # 可改为具体结果目录，如 data/cluster_collision/N1_..._vyB...


def main() -> None:
    _plot_result.dir = dir
    _plot_result.main()


if __name__ == '__main__':
    main()
