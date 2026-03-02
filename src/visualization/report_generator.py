from __future__ import annotations

import os
from typing import Mapping, Any


class ReportWriter:
    def __init__(self, out_dir: str) -> None:
        self.out_dir = out_dir
        self.f = None

    def open(self) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        self.f = open(os.path.join(self.out_dir, "report.txt"), "w", encoding="utf-8")
        self.f.write(
            "# t(s)\tKE_trans\tKE_rot\tKE_total\tVmax"
            "\tcluster_count\tcoordination_mean\tcluster_rg_mean\n"
        )

    def write_step(
        self,
        t: float,
        KE_t: float,
        KE_r: float,
        KE_sum: float,
        vmax: float,
        cluster_count: int | None = None,
        coordination_mean: float | None = None,
        cluster_rg_mean: float | None = None,
    ) -> None:
        if self.f is not None:
            cc = int(cluster_count) if cluster_count is not None else -1
            z_mean = float(coordination_mean) if coordination_mean is not None else float("nan")
            rg_mean = float(cluster_rg_mean) if cluster_rg_mean is not None else float("nan")
            self.f.write(
                f"{t:.6e}\t{KE_t:.6e}\t{KE_r:.6e}\t{KE_sum:.6e}\t{vmax:.6e}"
                f"\t{cc:d}\t{z_mean:.6e}\t{rg_mean:.6e}\n"
            )

    def close(self) -> None:
        if self.f is not None:
            self.f.close()
            self.f = None

    def write_runtime(self, seconds: float) -> None:
        # 以注释形式追加总运行用时，避免影响绘图解析
        if self.f is not None:
            self.f.write(f"# runtime(s)\t{seconds:.6f}\n")

    def write_params(self, params: Mapping[str, Any]) -> None:
        """以注释形式记录本次模拟的关键参数。

        采用 "# param.<key>\t<value>" 的统一前缀，
        不会影响 plot_result 对数据行的解析，且
        旧版 report.txt（无该段）仍然兼容。
        """
        if self.f is None:
            return
        for key, value in params.items():
            # 将值转成字符串，避免复杂结构破坏一行格式
            self.f.write(f"# param.{key}\t{value}\n")

